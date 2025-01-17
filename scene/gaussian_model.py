#
# Copyright (C) 2023 - 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from gaussian_hierarchy._C import load_hierarchy, write_hierarchy, load_dynamic_hierarchy, write_dynamic_hierarchy, get_morton_indices
from scene.OurAdam import Adam
from utils.reloc_utils import compute_relocation_cuda
import random
import math
hierarchy_node_depth = 0
hierarchy_node_parent = 1
hierarchy_node_child_count = 2
hierarchy_node_first_child = 3
hierarchy_node_next_sibling = 4
hierarchy_node_max_side_length = 5


class GaussianModel:
    # give every leaf node a random number and set their parents to the sum of the two numbers. 
    # if it exceeds a threshold, the node will be part of the cut.
    def get_random_cut_2(self, p):
        current_set = torch.where(torch.logical_and(self.nodes[:, hierarchy_node_child_count] == 0, self.nodes[:, hierarchy_node_next_sibling] > 0))[0]
        parents = self.nodes[current_set, hierarchy_node_parent]
        
    
    # Start with a leaf cut and select a random subset of the leaf nodes. Then, it goes from the bottom level
    # upward, pushing the cut of all nodes in the random subset to the next level upward 
    # (but only if its sibling is also in the subset). 
    def get_random_cut(self, p):
        number_gaussians = len(self.nodes)-self.skybox_points
        cut = torch.zeros(len(self.nodes)).cuda()
        current_set = torch.where(self.nodes[:, hierarchy_node_child_count] == 0)[0]
        cut[current_set] = 1
        subset_size = int(math.floor(len(current_set)*p))
        subset = current_set[(torch.randperm(len(current_set))[:subset_size])]
        current_depth = torch.max(self.nodes[subset, hierarchy_node_depth])
        while(len(subset) > 0):
            depth_subset = subset[self.nodes[subset, hierarchy_node_depth] == current_depth]
            first_children = depth_subset[self.nodes[depth_subset, hierarchy_node_next_sibling] > 0]
            siblings = self.nodes[first_children, hierarchy_node_next_sibling]
            
            siblings_are_cut = cut[siblings] == 1
            first_children = first_children[siblings_are_cut]
            siblings = siblings[siblings_are_cut]
            
            parents = self.nodes[first_children, hierarchy_node_parent]
            cut[parents] = 1
            cut[first_children] = 0
            cut[siblings] = 0
            subset = torch.cat((parents, subset[self.nodes[subset, hierarchy_node_depth] < current_depth]))
            current_depth -= 1
        return torch.where(cut == 1)[0]
            
            
            
        
        
    def recompute_weights(self):
        ellipse_surface = self.get_scaling[self.skybox_points:, 0] * self.get_scaling[self.skybox_points:, 1] + self.get_scaling[self.skybox_points:, 0] * self.get_scaling[self.skybox_points:, 2] + self.get_scaling[self.skybox_points:, 1] * self.get_scaling[self.skybox_points:, 2]
        self.weights = ellipse_surface * self.get_opacity.squeeze()[self.skybox_points:]
        first_children = self.nodes[self.skybox_points:, hierarchy_node_first_child]
        first_children = first_children[first_children != 0]
        siblings = self.nodes[first_children, hierarchy_node_next_sibling]
        # TODO: Update for non-binary trees
        normalization = self.weights[siblings-self.skybox_points] + self.weights[first_children-self.skybox_points]
        self.weights[first_children-self.skybox_points] /= normalization
        self.weights[siblings-self.skybox_points] /= normalization
        self.weights[0] = 1
        self.weights = torch.cat((torch.full([self.skybox_points], -99).cuda(), self.weights))
        
    def sort_morton(self):
        #pass
        xyz = self._xyz[self.skybox_points:]
        codes = torch.zeros_like(xyz[:, 0], dtype=torch.int64)
        get_morton_indices(xyz, torch.min(self._xyz[self.skybox_points:], dim=0)[0], torch.max(self._xyz[self.skybox_points:], dim=0)[0], codes)
        indices = torch.argsort(codes)
        indices = indices.to(torch.int)
        # Make sure the root node stays in place
        root_index = torch.where(indices==0)[0][0]
        indices[root_index] = indices[0]
        indices[0] = 0
        
        indices += self.skybox_points
        with torch.no_grad():
            self._opacity[indices] = self._opacity[self.skybox_points:].clone().detach().requires_grad_(True)
            self._xyz[indices] = self._xyz[self.skybox_points:].clone().detach().requires_grad_(True)
            self._scaling[indices] = self._scaling[self.skybox_points:].clone().detach().requires_grad_(True)
            self._rotation[indices] = self._rotation[self.skybox_points:].clone().detach().requires_grad_(True)
            self._features_dc[indices] = self._features_dc[self.skybox_points:].clone().detach().requires_grad_(True)
            self._features_rest[indices] = self._features_rest[self.skybox_points:].clone().detach().requires_grad_(True)
            self.nodes[self.skybox_points:, hierarchy_node_parent] = indices[self.nodes[self.skybox_points:, hierarchy_node_parent] - self.skybox_points]
            sibling_mask = self.nodes[:, hierarchy_node_next_sibling] > 0
            #sibling_mask_with_skybox = torch.cat((torch.zeros(self.skybox_points, dtype=torch.bool), sibling_mask))
            self.nodes[sibling_mask, hierarchy_node_next_sibling] = indices[self.nodes[sibling_mask, hierarchy_node_next_sibling] - self.skybox_points]
            child_mask = self.nodes[:, hierarchy_node_first_child] > 0
            #child_mask_with_skybox = torch.cat((torch.zeros(self.skybox_points, dtype=torch.bool), child_mask))
            self.nodes[child_mask, hierarchy_node_first_child] = indices[self.nodes[child_mask, hierarchy_node_first_child]- self.skybox_points]
            self.nodes[self.skybox_points, hierarchy_node_parent] = -1
            self.nodes[self.skybox_points, hierarchy_node_next_sibling] = 0
            self.nodes[indices] = self.nodes.clone()[self.skybox_points:]
            pass
            
        
    def merge_gaussians(self, indices):
        ellipse_surface  = lambda scale: scale[0] * scale[1] + scale[0] * scale[2] + scale[1] * scale[2]
        weights = []
        for index in indices:
            weights.append(self.get_opacity[index] * ellipse_surface(self.get_scaling[index]))
        weights /= np.sum(weights)
        new_position = self._xyz[indices]*weights
        new_features_dc = self._features_dc[indices]*weights
        new_features_rest = self._features_rest[indices]*weights
        pos_differences = self._xyz[indices] - new_position
        

    def compute_bounding_sphere_divergence(self, samples=1000):
        diverged = 0
        for i in range(samples):
            random_node = random.randrange(1, len(self.nodes))
            bounding_sphere_radius = torch.max(self.get_scaling[random_node]).item()
            bounding_sphere_position = self._xyz[random_node].detach().cpu()
            parent = self.nodes[random_node, hierarchy_node_parent]
            parent_sphere_radius = torch.max(self.get_scaling[parent])
            parent_sphere_position = self._xyz[parent].detach().cpu()
            for j in range(100):
                while True:
                    # Generate a random point in a cube of side 2*radius centered at the origin
                    point = (torch.rand(3)-0.5)*2
                    if torch.linalg.vector_norm(point) <= 1:
                        if torch.linalg.vector_norm((point*bounding_sphere_radius+bounding_sphere_position)-parent_sphere_position) > parent_sphere_radius:
                            diverged += 1
                        break
        print(diverged)
        return diverged/(100*samples)
        
    
    def sanity_check_hierarchy(self):
        print("Commencing Sanity Check of Hierarchy")
        self.sanity_counter = 0
        def sanity_check_rec(node):
            self.sanity_counter += 1
            if self.sanity_counter > len(self._xyz):
                print("Infinite Recursion")
            if self.nodes[node][hierarchy_node_child_count] == 0:
                return
            child_iterator = self.nodes[node][hierarchy_node_first_child]
            children = [child_iterator]
            while(self.nodes[child_iterator][hierarchy_node_next_sibling] != 0):
                child_iterator = self.nodes[child_iterator][hierarchy_node_next_sibling]
                children.append(child_iterator)
            if len(children) == 1:
                print(f"Error: Node {node} has single child")
            if len(children) > self.nodes[node][hierarchy_node_child_count]:
                print(f"Error: Parent has more children ({len(children)}) than expected ({self.nodes[node][hierarchy_node_child_count]})")
            if len(children) < self.nodes[node][hierarchy_node_child_count]:
                print(f"Error: Parent has less children ({len(children)}) than expected ({self.nodes[node][hierarchy_node_child_count]})")
            
            for child in children:
                if self.nodes[child][hierarchy_node_parent] != node:
                    print(f"Error: Siblings have different parents ({self.nodes[child][hierarchy_node_parent]} instead of {node})")
                if self.nodes[child][hierarchy_node_depth] != self.nodes[node][hierarchy_node_depth]+1:
                    print(f"Error: Child has depth {self.nodes[child][hierarchy_node_depth]}, but parent has depth {self.nodes[node][hierarchy_node_depth]}")
                #if torch.prod(self.get_scaling[child]).item() > (torch.prod(self.get_scaling[node]).item() * 1.1):
                #    print(f"Warning: Child has max scale {torch.prod(self.get_scaling[child]).item()}, but parent has max scale {torch.prod(self.get_scaling[node]).item()}")
                sanity_check_rec(child)
                
        sanity_check_rec(self.skybox_points)
        if self.sanity_counter != len(self._xyz):
            print(f"Error: Reached {self.sanity_counter} out of {len(self._xyz)} nodes by recursion")
        print(f"Finished Sanity Check of Hierarchy ( {self.sanity_counter} / {len(self._xyz)} nodes)")


    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize




    def __init__(self, sh_degree : int):
        self.is_hierarchy = False
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.nodes = None
        self.boxes = None

        self.pretrained_exposures = None

        self.skybox_points = 0
        self.skybox_locked = True

        self.setup_functions()

    def get_skybox_indices(self):
        return torch.arange(0, self.skybox_points)
    
    def get_number_of_leaf_nodes(self):
        return torch.sum(self.nodes[:, hierarchy_node_child_count] == 0).item()
    
    def get_number_of_levels(self):
        return torch.max(self.nodes[:, hierarchy_node_depth]).item()
    
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        return self._exposure[self.exposure_mapping[image_name]]
        # return self._exposure

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1


    def create_from_pcd(
            self, 
            pcd : BasicPointCloud, 
            cam_infos : int,
            spatial_lr_scale : float,
            skybox_points: int,
            scaffold_file: str,
            bounds_file: str,
            skybox_locked: bool):
        
        self.spatial_lr_scale = spatial_lr_scale

        xyz = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        
        minimum,_ = torch.min(xyz, axis=0)
        maximum,_ = torch.max(xyz, axis=0)
        mean = 0.5 * (minimum + maximum)

        self.skybox_locked = skybox_locked
        if scaffold_file != "" and skybox_points > 0:
            print(f"Overriding skybox_points: loading skybox from scaffold_file: {scaffold_file}")
            skybox_points = 0
        if skybox_points > 0:
            self.skybox_points = skybox_points
            radius = torch.linalg.norm(maximum - mean)

            theta = (2.0 * torch.pi * torch.rand(skybox_points, device="cuda")).float()
            phi = (torch.arccos(1.0 - 1.4 * torch.rand(skybox_points, device="cuda"))).float()
            skybox_xyz = torch.zeros((skybox_points, 3))
            skybox_xyz[:, 0] = radius * 10 * torch.cos(theta)*torch.sin(phi)
            skybox_xyz[:, 1] = radius * 10 * torch.sin(theta)*torch.sin(phi)
            skybox_xyz[:, 2] = radius * 10 * torch.cos(phi)
            skybox_xyz += mean.cpu()
            xyz = torch.concat((skybox_xyz.cuda(), xyz))
            fused_color = torch.concat((torch.ones((skybox_points, 3)).cuda(), fused_color))
            fused_color[:skybox_points,0] *= 0.7
            fused_color[:skybox_points,1] *= 0.8
            fused_color[:skybox_points,2] *= 0.95

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = RGB2SH(fused_color)
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(distCUDA2(xyz), 0.0000001)
        if scaffold_file == "" and skybox_points > 0:
            dist2[:skybox_points] *= 10
            dist2[skybox_points:] = torch.clamp_max(dist2[skybox_points:], 10) 
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((xyz.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        if scaffold_file == "" and skybox_points > 0:
            opacities = self.inverse_opacity_activation(0.02 * torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda"))
            opacities[:skybox_points] = 0.7
        else: 
            opacities = self.inverse_opacity_activation(0.01 * torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda"))

        features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
        features_rest = features[:,:,1:].transpose(1, 2).contiguous()

        self.scaffold_points = None
        if scaffold_file != "": 
            scaffold_xyz, features_dc_scaffold, features_extra_scaffold, opacities_scaffold, scales_scaffold, rots_scaffold = self.load_ply_file(scaffold_file + "/point_cloud.ply", 1)
            scaffold_xyz = torch.from_numpy(scaffold_xyz).float()
            features_dc_scaffold = torch.from_numpy(features_dc_scaffold).permute(0, 2, 1).float()
            features_extra_scaffold = torch.from_numpy(features_extra_scaffold).permute(0, 2, 1).float()
            opacities_scaffold = torch.from_numpy(opacities_scaffold).float()
            scales_scaffold = torch.from_numpy(scales_scaffold).float()
            rots_scaffold = torch.from_numpy(rots_scaffold).float()

            with open(scaffold_file + "/pc_info.txt") as f:
                skybox_points = int(f.readline())

            self.skybox_points = skybox_points
            with open(os.path.join(bounds_file, "center.txt")) as centerfile:
                with open(os.path.join(bounds_file, "extent.txt")) as extentfile:
                    centerline = centerfile.readline()
                    extentline = extentfile.readline()

                    c = centerline.split(' ')
                    e = extentline.split(' ')
                    center = torch.Tensor([float(c[0]), float(c[1]), float(c[2])]).cuda()
                    extent = torch.Tensor([float(e[0]), float(e[1]), float(e[2])]).cuda()

            distances1 = torch.abs(scaffold_xyz.cuda() - center)
            selec = torch.logical_and(
                torch.max(distances1[:,0], distances1[:,1]) > 0.5 * extent[0],
                torch.max(distances1[:,0], distances1[:,1]) < 1.5 * extent[0])
            selec[:skybox_points] = True

            self.scaffold_points = selec.nonzero().size(0)

            xyz = torch.concat((scaffold_xyz.cuda()[selec], xyz))
            features_dc = torch.concat((features_dc_scaffold.cuda()[selec,0:1,:], features_dc))

            filler = torch.zeros((features_extra_scaffold.cuda()[selec,:,:].size(0), 15, 3))
            filler[:,0:3,:] = features_extra_scaffold.cuda()[selec,:,:]
            features_rest = torch.concat((filler.cuda(), features_rest))
            scales = torch.concat((scales_scaffold.cuda()[selec], scales))
            rots = torch.concat((rots_scaffold.cuda()[selec], rots))
            opacities = torch.concat((opacities_scaffold.cuda()[selec], opacities))

        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.requires_grad_(True))
        self._features_rest = nn.Parameter(features_rest.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}

        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))
        print("Number of points at initialisation : ", self._xyz.shape[0])

    def training_setup(self, training_args, our_adam=True):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]

        if our_adam:
            self.optimizer = Adam(l, lr=0.0, eps=1e-15)
        else:
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            
        if self.pretrained_exposures is None:
            self.exposure_optimizer = torch.optim.Adam([self._exposure])
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final, lr_delay_steps=training_args.exposure_lr_delay_steps, lr_delay_mult=training_args.exposure_lr_delay_mult, max_steps=training_args.iterations)

       
    def load_ply_file(self, path, degree):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        return xyz, features_dc, features_extra, opacities, scales, rots


        
        
    
    # scaffold file is only required for number of skybox points
    def create_from_hier(self, path, spatial_lr_scale : float, scaffold_file : str):
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.is_hierarchy = True
        self.spatial_lr_scale = spatial_lr_scale
        print(path)
        #xyz, shs_all, alpha, scales, rots, nodes, boxes = load_hierarchy(path)
        xyz, shs_all, alpha, scales, rots, nodes = load_dynamic_hierarchy(path)
        # set first child to 0 for all nodes that do not have children (because this is fucked up in some hierarchy files)
        #alpha = torch.sigmoid((alpha))
        #alpha = torch.sigmoid(alpha)
        #scales = torch.log(scales)

        base = os.path.dirname(path)

        try:
            with open(os.path.join(base, "anchors.bin"), mode='rb') as f:
                bytes = f.read()
                int_val = int.from_bytes(bytes[:4], "little", signed="False")
                dt = np.dtype(np.int32)
                vals = np.frombuffer(bytes[4:], dtype=dt) 
                self.anchors = torch.from_numpy(vals).long().cuda()
        except:
            print("WARNING: NO ANCHORS FOUND")
            self.anchors = torch.Tensor([]).long()

        #retrieve exposure
        exposure_file = os.path.join(base, "../../exposure.json")
        if os.path.exists(exposure_file):
            with open(exposure_file, "r") as f:
                exposures = json.load(f)

            self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
        else:
            print(f"No exposure to be loaded at {exposure_file}")
            self.pretrained_exposures = None

        
        if scaffold_file:
            #retrieve skybox
            skybox_points = 100_000
            self.skybox_points = 100_000
            if scaffold_file != "":
                scaffold_xyz, features_dc_scaffold, features_extra_scaffold, opacities_scaffold, scales_scaffold, rots_scaffold = self.load_ply_file(scaffold_file + "/point_cloud.ply", 1)
                scaffold_xyz = torch.from_numpy(scaffold_xyz).float()
                features_dc_scaffold = torch.from_numpy(features_dc_scaffold).permute(0, 2, 1).float()
                features_extra_scaffold = torch.from_numpy(features_extra_scaffold).permute(0, 2, 1).float()
                opacities_scaffold = torch.from_numpy(opacities_scaffold).float()
                scales_scaffold = torch.from_numpy(scales_scaffold).float()
                rots_scaffold = torch.from_numpy(rots_scaffold).float()
                skybox_points = 100_000
                with open(scaffold_file + "/pc_info.txt") as f:
                    skybox_points = int(f.readline())
                    
                self.skybox_points = skybox_points
    
            # add skybox points to the front of array
            if self.skybox_points > 0:
                if scaffold_file != "":
                    skybox_xyz, features_dc_sky, features_rest_sky, opacities_sky, scales_sky, rots_sky = scaffold_xyz[:skybox_points], features_dc_scaffold[:skybox_points], features_extra_scaffold[:skybox_points], opacities_scaffold[:skybox_points], scales_scaffold[:skybox_points], rots_scaffold[:skybox_points]
    
                opacities_sky = torch.sigmoid(opacities_sky)
                xyz = torch.cat((skybox_xyz, xyz))
                alpha = torch.cat((opacities_sky, alpha))
                scales = torch.cat((scales_sky, scales))
                rots = torch.cat((rots_sky, rots))
                filler = torch.zeros(features_dc_sky.size(0), 16, 3)
                filler[:, :1, :] = features_dc_sky
                filler[:, 1:4, :] = features_rest_sky
                shs_all = torch.cat((filler, shs_all))
            nodes[:, hierarchy_node_first_child] += self.skybox_points
            nodes[:, hierarchy_node_parent] += self.skybox_points
            
            nodes[nodes[:,hierarchy_node_next_sibling]>0, hierarchy_node_next_sibling] += self.skybox_points
            nodes[0, hierarchy_node_parent] = -1
            nodes = torch.cat((torch.full((self.skybox_points, 6), -99, dtype=torch.int32), nodes)) 
            nodes[skybox_points:, 3] = torch.where(nodes[skybox_points:, 2]==2, nodes[skybox_points:, 3], torch.zeros_like(nodes[skybox_points:,3]))

        
        self._xyz = nn.Parameter(xyz.cuda().requires_grad_(True))
        self._features_dc = nn.Parameter(shs_all.cuda()[:,:1,:].requires_grad_(True))
        self._features_rest = nn.Parameter(shs_all.cuda()[:,1:16,:].requires_grad_(True))
        self._opacity = nn.Parameter(alpha.cuda().requires_grad_(True))
        self._scaling = nn.Parameter(scales.cuda().requires_grad_(True))
        self._rotation = nn.Parameter(rots.cuda().requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        #self.opacity_activation = torch.abs
        #self.inverse_opacity_activation = torch.abs
        
        self.hierarchy_path = path
        self.nodes = nodes.cuda()
        #self.boxes = boxes.cuda()

    def create_from_pt(self, path, spatial_lr_scale : float ):
        self.spatial_lr_scale = spatial_lr_scale

        xyz = torch.load(path + "/done_xyz.pt")
        shs_dc = torch.load(path + "/done_dc.pt")
        shs_rest = torch.load(path + "/done_rest.pt")
        alpha = torch.load(path + "/done_opacity.pt")
        scales = torch.load(path + "/done_scaling.pt")
        rots = torch.load(path + "/done_rotation.pt")

        self._xyz = nn.Parameter(xyz.cuda().requires_grad_(True))
        self._features_dc = nn.Parameter(shs_dc.cuda().requires_grad_(True))
        self._features_rest = nn.Parameter(shs_rest.cuda().requires_grad_(True))
        self._opacity = nn.Parameter(alpha.cuda().requires_grad_(True))
        self._scaling = nn.Parameter(scales.cuda().requires_grad_(True))
        self._rotation = nn.Parameter(rots.cuda().requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def save_hier(self):
        print(f"Result hierarchy written to {self.hierarchy_path}_opt")
        write_dynamic_hierarchy(self.hierarchy_path + "_opt",
                        self._xyz,
                        torch.cat((self._features_dc, self._features_rest), 1),
                        self.opacity_activation(self._opacity),
                        self._scaling,
                        self._rotation,
                        self.nodes)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''

        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_pt(self, path):
        mkdir_p(path)

        torch.save(self._xyz.detach().cpu(), os.path.join(path, "done_xyz.pt"))
        torch.save(self._features_dc.cpu(), os.path.join(path, "done_dc.pt"))
        torch.save(self._features_rest.cpu(), os.path.join(path, "done_rest.pt"))
        torch.save(self._opacity.cpu(), os.path.join(path, "done_opacity.pt"))
        torch.save(self._scaling, os.path.join(path, "done_scaling.pt"))
        torch.save(self._rotation, os.path.join(path, "done_rotation.pt"))

        import struct
        def load_pt(path):
            xyz = torch.load(os.path.join(path, "done_xyz.pt")).detach().cpu()
            features_dc = torch.load(os.path.join(path, "done_dc.pt")).detach().cpu()
            features_rest = torch.load( os.path.join(path, "done_rest.pt")).detach().cpu()
            opacity = torch.load(os.path.join(path, "done_opacity.pt")).detach().cpu()
            scaling = torch.load(os.path.join(path, "done_scaling.pt")).detach().cpu()
            rotation = torch.load(os.path.join(path, "done_rotation.pt")).detach().cpu()

            return xyz, features_dc, features_rest, opacity, scaling, rotation


        xyz, features_dc, features_rest, opacity, scaling, rotation = load_pt(path)

        my_int = xyz.size(0)
        with open(os.path.join(path, "point_cloud.bin"), 'wb') as f:
            f.write(struct.pack('i', my_int))
            f.write(xyz.numpy().tobytes())
            print(features_dc[0])
            print(features_rest[0])
            f.write(torch.cat((features_dc, features_rest), dim=1).numpy().tobytes())
            f.write(opacity.numpy().tobytes())
            f.write(scaling.numpy().tobytes())
            f.write(rotation.numpy().tobytes())


    def save_ply(self, path, only_leaves=False):
        mkdir_p(os.path.dirname(path))
        if only_leaves:
            mask = self.nodes[:, hierarchy_node_child_count] == 0
        else:
            # all true mask
            mask = torch.ones(len(self._opacity), dtype=torch.bool)
        xyz = self._xyz[mask].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc[mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest[mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        #opacities = self.inverse_opacity_activation(self._opacity[mask]).detach().cpu().numpy()
        opacities = (self._opacity[mask]).detach().cpu().numpy()
        scale = self._scaling[mask].detach().cpu().numpy()
        rotation = self._rotation[mask].detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = torch.cat((self._opacity[:self.skybox_points], inverse_sigmoid(torch.min(self.get_opacity[self.skybox_points:], torch.ones_like(self.get_opacity[self.skybox_points:])*0.01))), 0)
        #opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        xyz, features_dc, features_extra, opacities, scales, rots = self.load_ply_file(path, self.max_sh_degree)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    # removes the pruned gaussians from the optimizer and return tensors with the pruned gaussian properties removed
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        
        
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
        
        indices = torch.where(mask)
        self._xyz[indices] = 0
        self._scaling[indices] = 0
        self._opacity[indices] = 0
        nodes[indices]
                    
            #self.nodes = 
        

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    # reset_params is from MCMC
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, reset_params=True):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if reset_params:
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        #self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad * self.max_radii2D * torch.pow(self.get_opacity.flatten(), 1/5.0) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.get_opacity.flatten() > 0.15)

        #selected_pts_mask = torch.logical_and(selected_pts_mask,
        #                                      torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # Only densify leaf nodes
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.nodes[:,hierarchy_node_child_count] == 0)        
        
        print(f"Split {len(torch.where(selected_pts_mask)[0])} points")
        # No densification of the scaffold
        if self.scaffold_points is not None:
            selected_pts_mask[:self.scaffold_points] = False


        stds = self.get_scaling[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat_interleave(repeats=N,dim=0)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat_interleave(repeats=N,dim=0) / (0.7*N))
        new_rotation = self._rotation[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        new_features_dc = self._features_dc[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        new_features_rest = self._features_rest[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        new_opacity = self._opacity[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        #new_boxes = self._opacity[selected_pts_mask].repeat(N,1)
        
        new_nodes = torch.zeros((len(new_xyz), 6), dtype=torch.int32)
        for index, node in enumerate(torch.where(selected_pts_mask)[0]):
            full_index = len(self._xyz) + index*2
            self.nodes[node][hierarchy_node_child_count] = 2
            self.nodes[node][hierarchy_node_first_child] = full_index
            
            
            new_nodes[index*2][hierarchy_node_depth] = self.nodes[node][hierarchy_node_depth] + 1
            new_nodes[index*2][hierarchy_node_parent] = node
            new_nodes[index*2][hierarchy_node_child_count] = 0
            new_nodes[index*2][hierarchy_node_first_child] = -1
            new_nodes[index*2][hierarchy_node_next_sibling] = full_index + 1
            
            new_nodes[index*2+1][hierarchy_node_depth] = self.nodes[node][hierarchy_node_depth] + 1
            new_nodes[index*2+1][hierarchy_node_parent] = node
            new_nodes[index*2+1][hierarchy_node_child_count] = 0
            new_nodes[index*2+1][hierarchy_node_first_child] = -1
            new_nodes[index*2+1][hierarchy_node_next_sibling] = 0
        self.nodes = torch.cat((self.nodes, new_nodes.to('cuda')))

        
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        #prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        #self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, N=2):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) * self.max_radii2D * torch.pow(self.get_opacity.flatten(), 1/5.0) >= grad_threshold, True, False)
        # Don't densify points that will be pruned
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.get_opacity.flatten() > 0.15)
        # Don't densify points that will be pruned
        #selected_pts_mask = torch.logical_and(selected_pts_mask,
        #                                      torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # No densification of the scaffold
        if self.scaffold_points is not None:
            selected_pts_mask[:self.scaffold_points] = False
        # Only densify leaf nodes
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.nodes[:,hierarchy_node_child_count] == 0)

        
        print(f"Clone {len(torch.where(selected_pts_mask)[0])} points")
        new_xyz = self._xyz[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat_interleave(repeats=N,dim=0))
        new_features_dc = self._features_dc[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        new_features_rest = self._features_rest[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        new_opacities = self._opacity[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        
        new_rotation = self._rotation[selected_pts_mask].repeat_interleave(repeats=N,dim=0)


        new_nodes = torch.zeros((len(new_xyz), 6), dtype=torch.int32)
        for index, node in enumerate(torch.where(selected_pts_mask)[0]):
            full_index = len(self._xyz) + index*2
            self.nodes[node][hierarchy_node_child_count] = 2
            self.nodes[node][hierarchy_node_first_child] = full_index
            
            
            new_nodes[index*2][hierarchy_node_depth] = self.nodes[node][hierarchy_node_depth] + 1
            new_nodes[index*2][hierarchy_node_parent] = node
            new_nodes[index*2][hierarchy_node_child_count] = 0
            new_nodes[index*2][hierarchy_node_first_child] = -1
            new_nodes[index*2][hierarchy_node_next_sibling] = full_index + 1
            
            new_nodes[index*2+1][hierarchy_node_depth] = self.nodes[node][hierarchy_node_depth] + 1
            new_nodes[index*2+1][hierarchy_node_parent] = node
            new_nodes[index*2+1][hierarchy_node_child_count] = 0
            new_nodes[index*2+1][hierarchy_node_first_child] = -1
            new_nodes[index*2+1][hierarchy_node_next_sibling] = 0
        self.nodes = torch.cat((self.nodes, new_nodes.to('cuda')))

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)


    def densify(self, grads, grad_threshold, scene_extent, N=2):
        
        # Replace 2D gradients with 3D gradients
        #grads = self._xyz.grad
        
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) * self.max_radii2D * torch.pow(self.get_opacity.flatten(), 1/5.0) >= grad_threshold, True, False)
        # Don't densify points that will be pruned
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.get_opacity.flatten() > 0.15)
        # Don't densify points that will be pruned
        #selected_pts_mask = torch.logical_and(selected_pts_mask,
        #                                      torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # No densification of the scaffold
        if self.scaffold_points is not None:
            selected_pts_mask[:self.scaffold_points] = False
        # Only densify leaf nodes
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.nodes[:,hierarchy_node_child_count] == 0)
        
        #selected_pts_mask = torch.logical_and(selected_pts_mask, torch.min(self._scaling, dim=1)[0] > -10)
        
        print(f"Densify {len(torch.where(selected_pts_mask)[0])} points")
        new_xyz = self._xyz[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat_interleave(repeats=N,dim=0) / (0.8*N))
        new_features_dc = self._features_dc[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        new_features_rest = self._features_rest[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        #new_opacities = self._opacity[selected_pts_mask].repeat_interleave(repeats=N,dim=0)
        new_opacities = self.inverse_opacity_activation(self.get_opacity[selected_pts_mask].repeat_interleave(repeats=N,dim=0) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat_interleave(repeats=N,dim=0)


        new_nodes = torch.zeros((len(new_xyz), 6), dtype=torch.int32)
        for index, node in enumerate(torch.where(selected_pts_mask)[0]):
            full_index = len(self._xyz) + index*2
            self.nodes[node][hierarchy_node_child_count] = 2
            self.nodes[node][hierarchy_node_first_child] = full_index
            
            
            new_nodes[index*2][hierarchy_node_depth] = self.nodes[node][hierarchy_node_depth] + 1
            new_nodes[index*2][hierarchy_node_parent] = node
            new_nodes[index*2][hierarchy_node_child_count] = 0
            new_nodes[index*2][hierarchy_node_first_child] = -1
            new_nodes[index*2][hierarchy_node_next_sibling] = full_index + 1
            
            new_nodes[index*2+1][hierarchy_node_depth] = self.nodes[node][hierarchy_node_depth] + 1
            new_nodes[index*2+1][hierarchy_node_parent] = node
            new_nodes[index*2+1][hierarchy_node_child_count] = 0
            new_nodes[index*2+1][hierarchy_node_first_child] = -1
            new_nodes[index*2+1][hierarchy_node_next_sibling] = 0
        self.nodes = torch.cat((self.nodes, new_nodes.to('cuda')))

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        print("")


    def densify_and_prune(self, max_grad, min_opacity, extent):
        grads = self.xyz_gradient_accum 
        grads[grads.isnan()] = 0.0

        self.densify(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if self.scaffold_points is not None:
            prune_mask[:self.scaffold_points] = False

        #self.prune_points(prune_mask)

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, indices, skybox_points = 100000):
        update_filter_indices = torch.where(update_filter)[0]
        update_indices = indices[update_filter_indices]
        #self.xyz_gradient_accum[update_indices] = torch.max(torch.norm(viewspace_point_tensor.grad[indices][update_filter,:2], dim=-1, keepdim=True), self.xyz_gradient_accum[update_indices])
        # I use xyz gradients here instead of screen space gradients
        self.xyz_gradient_accum[indices] = torch.max(torch.norm(viewspace_point_tensor.grad[skybox_points:skybox_points+len(indices)][:,:2], dim=-1, keepdim=True), self.xyz_gradient_accum[indices])
        #print(torch.sum(torch.abs(torch.max(torch.norm(viewspace_point_tensor.grad[indices][update_filter,:2], dim=-1, keepdim=True), self.xyz_gradient_accum[update_indices]))))
        #print(torch.max((self.xyz_gradient_accum[update_indices])))
        self.denom[indices] += 1
        


#region MCMC
    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {"xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "scaling" : self._scaling,
            "rotation" : self._rotation}
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            
            if inds is not None:
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
            else:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
            del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group['params'][0]] = stored_state
            optimizable_tensors[group["name"]] = group["params"][0]
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"] 
        # Thomas Bug
        torch.cuda.empty_cache()
        return optimizable_tensors
    
    def _update_params(self, idxs, ratio):
        new_opacity, new_scaling = compute_relocation_cuda(
            opacity_old=self.get_opacity[idxs, 0],
            scale_old=self.get_scaling[idxs],
            N=ratio[idxs, 0] + 1
        )
        new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max=1.0 - torch.finfo(torch.float32).eps, min=0.005)
        new_opacity = self.inverse_opacity_activation(new_opacity)
        new_scaling = self.scaling_inverse_activation(new_scaling.reshape(-1, 3))
        return self._xyz[idxs], self._features_dc[idxs], self._features_rest[idxs], new_opacity, new_scaling, self._rotation[idxs]
    
    def _sample_alives(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
        return sampled_idxs, ratio
    
    def relocate_gs(self, dead_mask=None):
        if dead_mask.sum() == 0:
            return
        alive_mask = ~dead_mask 
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_mask = torch.logical_and(alive_mask, self.nodes[:, hierarchy_node_child_count] == 0)
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]
        
        if alive_indices.shape[0] <= 0:
            return
        # sample from alive ones based on opacity
        probs = (self.get_opacity[alive_indices, 0]) 
        
        remove_siblings = self.nodes[dead_indices, hierarchy_node_next_sibling]
        dead_indices = dead_indices[~torch.isin(dead_indices, remove_siblings)]
        # get_sibling cannot be vectorized :(
        next_sibling_mask = self.nodes[dead_indices, hierarchy_node_next_sibling] > 0
        sibling_indices = torch.zeros_like(dead_indices, dtype=torch.int32)
        sibling_indices[next_sibling_mask] = self.nodes[dead_indices[next_sibling_mask], hierarchy_node_next_sibling]
        sibling_indices[~next_sibling_mask] = self.nodes[self.nodes[dead_indices[~next_sibling_mask], hierarchy_node_parent], hierarchy_node_first_child]

        
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])
        (
            self._xyz[dead_indices], 
            self._features_dc[dead_indices],
            self._features_rest[dead_indices],
            self._opacity[dead_indices],
            self._scaling[dead_indices],
            self._rotation[dead_indices] 
        ) = self._update_params(reinit_idx, ratio=ratio)
        
        
        
        
        
        
        def get_sibling(node):
            if self.nodes[node, hierarchy_node_next_sibling] > 0:
                return self.nodes[node, hierarchy_node_next_sibling]
            else: 
                return self.nodes[self.nodes[node, hierarchy_node_parent], hierarchy_node_first_child]
        
        
        # propogate the not-dead sibling to parent
        parent_indices = self.nodes[dead_indices, hierarchy_node_parent]
        
        self._xyz[parent_indices] = self._xyz[sibling_indices]
        self._opacity[parent_indices] = self._opacity[sibling_indices]
        self._features_dc[parent_indices] = self._features_dc[sibling_indices]
        self._features_rest[parent_indices] = self._features_rest[sibling_indices]
        self._scaling[parent_indices] = self._scaling[sibling_indices]
        self._rotation[parent_indices] = self._rotation[sibling_indices]
        
        self.nodes[parent_indices, hierarchy_node_child_count] = 0
        self.nodes[parent_indices, hierarchy_node_first_child] = -1

        
        # respawn nodes now have 2 children
        self.nodes[reinit_idx, hierarchy_node_child_count] = 2
        self.nodes[reinit_idx, hierarchy_node_first_child] = dead_indices.to(torch.int32)
        
        # dead nodes get new parents from leaf nodes
        self.nodes[dead_indices, hierarchy_node_depth] = self.nodes[reinit_idx, 0] + 1
        self.nodes[dead_indices, hierarchy_node_parent] = reinit_idx.to(torch.int32)
        self.nodes[dead_indices, hierarchy_node_child_count] = 0
        self.nodes[dead_indices, hierarchy_node_first_child] = -1
        self.nodes[dead_indices, hierarchy_node_next_sibling] = sibling_indices
        
        self.nodes[sibling_indices, hierarchy_node_depth] = self.nodes[reinit_idx, 0] + 1
        self.nodes[sibling_indices, hierarchy_node_parent] = reinit_idx.to(torch.int32)
        self.nodes[sibling_indices, hierarchy_node_child_count] = 0
        self.nodes[sibling_indices, hierarchy_node_first_child] = -1
        self.nodes[sibling_indices, hierarchy_node_next_sibling] = 0
        
        self._xyz[sibling_indices] = self._xyz[dead_indices]
        self._opacity[sibling_indices] = self._opacity[dead_indices]
        self._features_dc[sibling_indices] = self._features_dc[dead_indices]
        self._opacity[sibling_indices] = self._opacity[dead_indices]
        self._scaling[sibling_indices] = self._scaling[dead_indices]
        self.replace_tensors_to_optimizer(inds=sibling_indices)
         
        
    def add_new_gs(self, cap_max):
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)
        if num_gs <= 0:
            return 0
        print(f"Spawn {num_gs} new Gaussians")
        alive_indices=torch.where(self.nodes[:, hierarchy_node_child_count] == 0)[0]
        probs = self.get_opacity.squeeze(-1)[alive_indices]
        
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs, alive_indices=alive_indices)
        add_idx = torch.where(ratio == 1)[0]
        ratio = torch.zeros_like(ratio)
        ratio[add_idx] = 1
        (
            new_xyz, 
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation 
        ) = self._update_params(add_idx, ratio=ratio)
        
        
        new_xyz = new_xyz.repeat_interleave(repeats=2, dim=0)
        new_features_dc = new_features_dc.repeat_interleave(repeats=2, dim=0)
        new_features_rest = new_features_rest.repeat_interleave(repeats=2, dim=0)
        new_opacity = new_opacity.repeat_interleave(repeats=2, dim=0)
        new_scaling = new_scaling.repeat_interleave(repeats=2, dim=0)
        new_rotation = new_rotation.repeat_interleave(repeats=2, dim=0)
        
        
        new_nodes = torch.zeros((len(new_xyz), 6), dtype=torch.int32)        
        for index, node in enumerate(add_idx):
            full_index = len(self._xyz) + index*2
            self.nodes[node][hierarchy_node_child_count] = 2
            self.nodes[node][hierarchy_node_first_child] = full_index


            new_nodes[index*2][hierarchy_node_depth] = self.nodes[node][hierarchy_node_depth] + 1
            new_nodes[index*2][hierarchy_node_parent] = node
            new_nodes[index*2][hierarchy_node_child_count] = 0
            new_nodes[index*2][hierarchy_node_first_child] = -1
            new_nodes[index*2][hierarchy_node_next_sibling] = full_index + 1

            new_nodes[index*2+1][hierarchy_node_depth] = self.nodes[node][hierarchy_node_depth] + 1
            new_nodes[index*2+1][hierarchy_node_parent] = node
            new_nodes[index*2+1][hierarchy_node_child_count] = 0
            new_nodes[index*2+1][hierarchy_node_first_child] = -1
            new_nodes[index*2+1][hierarchy_node_next_sibling] = 0
        self.nodes = torch.cat((self.nodes, new_nodes.to('cuda')))        
        

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, reset_params=False)
        return num_gs
#endregion