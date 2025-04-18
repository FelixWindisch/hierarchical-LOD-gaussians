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
from utils.general_utils import get_expon_lr_func
import os
import torch
from torch import nn
import debug_utils
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render_post, render, render_coarse, render_on_disk, render_vanilla, render_stp, network_gui
import sys
from scene import Scene, GaussianModel, OurAdam
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import strip_symmetric, build_scaling_rotation
import math
import torchvision
from fused_ssim import fused_ssim
import random
from torch.utils.tensorboard import SummaryWriter
from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights, expand_to_size_dynamic, get_interpolation_weights_dynamic
import time
from torchviz import make_dot
import consistency_graph
from scipy.spatial import KDTree
import numpy as np
from gaussian_hierarchy._C import  get_spt_cut_cuda
#from stp_gaussian_rasterization import ExtendedSettings
from gaussian_renderer import occlusion_cull
import psutil
import gc
# to check CPU RAM usage
pid = os.getpid()

clock_start = True
clock_time = time.time()
def clock():
    global clock_start
    global clock_time
    if clock_start:
        clock_start = False
        clock_time = time.time()
    else:
        clock_start = True
        return time.time()-clock_time


def direct_collate(x):
    return x


WriteTensorBoard = True
#Standard
densify_interval = 5000
lr_multiplier = 1
#Unused
Random_Hierarchy_Cut = True
Only_Noise_Visible = True
#MCMC
Max_Cap = 50_000_000
MCMC_Densification = True
MCMC_Noise_LR = 0  #5e5
lambda_scaling = 0
lambda_opacity = 0.01
#Hierarchical
Gaussian_Interpolation = False
# Upward Propagation D
Gradient_Propagation = False
Propagation_Strength = 1.0
#Culling
Use_Bounding_Spheres = False
Use_Occlusion_Culling = False
Use_Frustum_Culling = True
Use_MIP_respawn = False
# SPTs
Storage_Device = 'cpu'
lambda_hierarchy = 0.00
SPT_Root_Volume = 100 # 0.02
SPT_Target_Granularity = 0.00228
Min_SPT_Size = 256
Cache_SPTs = True
Reuse_SPT_Tolerarance = 0.9
Max_Gaussian_Budget = 100_000_000
Distance_Multiplier_Until_Budget = 1.5



#View Selection
Use_Consistency_Graph = False
# Rasterizer
Rasterizer = "Vanilla"
Anti_Aliasing = True
# Optimizer
Global_ADAM = False

non_blocking=False

Max_SH_Degree = 1


def training(dataset, opt:OptimizationParams, pipe, saving_iterations, checkpoint_iterations, checkpoint, debug_from, cons_graph):
    post_backward_peak = 0
    opt.densification_interval = densify_interval
    prev_peak_memory =0
    prev_number_rendered =0
    #torch.cuda.memory._record_memory_history()
    #torch.autograd.set_detect_anomaly(True)
    network_gui.init("127.0.0.1", 6009)
    #splat_settings = ExtendedSettings.from_json('/home/felix-windisch/hierarchical-LOD-gaussians/configs/vanilla.json')
    if WriteTensorBoard:
        writer = SummaryWriter()
        hyper_params = {
            "Max_Cap": Max_Cap,
            "Random_Hierarchy_Cut": Random_Hierarchy_Cut,
            "Only_Noise_Visible": Only_Noise_Visible,
            "MCMC_Densification": MCMC_Densification,
            "Gaussian_Interpolation": Gaussian_Interpolation,
            "Gradient_Propagation": Gradient_Propagation,
            "Storage_Device": Storage_Device,
            "Propagation_Strength": Propagation_Strength,
            "lambda_hierarchy": lambda_hierarchy,
            "SPT_Root_Volume": SPT_Root_Volume,
            "SPT_Target_Granularity": SPT_Target_Granularity,
            "Cache_SPTs": Cache_SPTs,
            "Reuse_SPT_Tolerarance": Reuse_SPT_Tolerarance,
            "MCMC_Noise_LR": MCMC_Noise_LR,
            "Use_Bounding_Spheres": Use_Bounding_Spheres,
            "Use_Consistency_Graph" : Use_Consistency_Graph,
            "Use_Frustum_Culling" : Use_Frustum_Culling,
            "Use_Occlusion_Culling" : Use_Occlusion_Culling,
            "lambda_scaling" : lambda_scaling,
            "lambda_opacity" : lambda_opacity,
            "Resterizer" : Rasterizer
        }
        metrics = {"empty" : 0}
        writer.add_hparams(hyper_params, metrics)
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(Max_SH_Degree)
    gaussians.active_sh_degree = Max_SH_Degree
    gaussians.scaffold_points = None
    #with torch.no_grad():
    #    gaussians._features_dc = gaussians._features_dc.abs() 
    scene = Scene(dataset, gaussians, resolution_scales = [1], create_from_hier=True)
    gaussians._xyz.requires_grad_(False)
    gaussians._opacity.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._features_dc.requires_grad_(False)
    gaussians._features_rest.requires_grad_(False)
    #with torch.no_grad():
    gaussians._opacity.clamp_(0, 0.99999)
    gaussians._opacity = gaussians.inverse_opacity_activation(gaussians._opacity)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        
    #parent_indices = torch.zeros(gaussians._xyz.size(0)).int().cuda()
    #nodes_for_render_indices = torch.zeros(gaussians._xyz.size(0)).int().cuda()
    #num_siblings = torch.zeros(gaussians._xyz.size(0)).int().cuda()
    
    
    optimizer_state = gaussians.move_storage_to(Storage_Device, int(round(Max_Cap * 1.1)), False)
    print(f"Moved to {Storage_Device}")
    gaussians.build_hierarchical_SPT(SPT_Root_Volume, SPT_Target_Granularity, Min_SPT_Size, use_bounding_spheres=Use_Bounding_Spheres)
    print("Built SPTs")
    #scene.dump_gaussians("Dump", False, file_name="SPTNodes.ply", indices=spt_nodes.to(Storage_Device))
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    #opt.iterations = 15000
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1


    iteration = first_iter
    # Dataloader loads data from disk
    # DONT SHUFFLE IF USING CONSISTENCY GRAPH
    training_generator = DataLoader(scene.getTrainCameras(), num_workers = 8, prefetch_factor = 1, persistent_workers = True, collate_fn=direct_collate, shuffle=not Use_Consistency_Graph)
    train_camera_data_set = scene.getTrainCameras()
    view_positions = []    
    if Use_MIP_respawn:
        for camera in train_camera_data_set:
            view_positions.append(camera.camera_center)
        view_positions = np.array(view_positions)
        camera_KD_tree = KDTree(view_positions)
    if Use_Consistency_Graph:
        current_camera_index = list(cons_graph.nodes())[0]
    else:
        current_camera_index = 0
    if Gradient_Propagation:
        gaussians.recompute_weights()
    
    #means3D, opacity, scales, rotations, features_dc, features_rest, render_indices = torch.empty((0, 3), device='cuda', dtype=torch.float32), torch.empty((0, 1), device='cuda', dtype=torch.float32), torch.empty((0, 3), device='cuda', dtype=torch.float32), torch.empty((0,4), device='cuda', dtype=torch.float32), torch.empty((0, 1, 3), device='cuda', dtype=torch.float32), torch.empty((0,15, 3), device='cuda', dtype=torch.float32), torch.empty(0, device='cuda', dtype=torch.int32)
    if Cache_SPTs:
        render_indices = torch.arange(0, gaussians.skybox_points, device='cuda', dtype = torch.int32)
        means3D = gaussians._xyz[:gaussians.skybox_points].cuda().contiguous()
        opacity = gaussians._opacity[:gaussians.skybox_points].cuda().contiguous()
        scales = gaussians._scaling[:gaussians.skybox_points].cuda().contiguous()
        rotations = gaussians._rotation[:gaussians.skybox_points].cuda().contiguous()
        features_dc = gaussians._features_dc[:gaussians.skybox_points].cuda().contiguous()
        features_rest = gaussians._features_rest[:gaussians.skybox_points].cuda().contiguous()
    else:
        means3D, opacity, scales, rotations, features_dc, features_rest, render_indices = torch.empty((0, 3), device='cuda', dtype=torch.float32), torch.empty((0, 1), device='cuda', dtype=torch.float32), torch.empty((0, 3), device='cuda', dtype=torch.float32), torch.empty((0,4), device='cuda', dtype=torch.float32), torch.empty((0, 1, 3), device='cuda', dtype=torch.float32), torch.empty((0,15, 3), device='cuda', dtype=torch.float32), torch.empty(0, device='cuda', dtype=torch.int32)
    
    
    
    
    parameters = []
    for values, name, lr in zip([means3D, features_dc, opacity, scales, rotations, features_rest], 
                                                ["xyz", "f_dc", "opacity", "scaling", "rotation", "f_rest"],
                                                [opt.position_lr_init * gaussians.spatial_lr_scale, opt.feature_lr, opt.opacity_lr, opt.scaling_lr, opt.rotation_lr, opt.feature_lr]):
        parameters.append({'params': [values], 'lr': lr * lr_multiplier, "name": name, 
                             "exp_avgs" : torch.zeros_like(values, device='cuda'), "exp_avgs_sqs" : torch.zeros_like(values, device='cuda')})
    prev_SPT_distances = torch.empty(0, dtype = torch.float32, device='cuda')
    prev_SPT_indices = torch.empty(0, dtype = torch.int32, device='cuda')
    prev_SPT_counts = torch.empty(0, dtype = torch.int32, device='cuda')

    train_image_counts = torch.ones(len(train_camera_data_set), dtype=torch.int32)
    
    gaussians.xyz_scheduler_args = get_expon_lr_func(lr_init=opt.position_lr_init*gaussians.spatial_lr_scale,
                                                    lr_final=opt.position_lr_final*gaussians.spatial_lr_scale,
                                                    lr_delay_mult=opt.position_lr_delay_mult,
                                                    max_steps=opt.position_lr_max_steps)
    print("Gaussians Initialized")
    #gaussians.sort_morton()
    while iteration < opt.iterations + 1:
        for viewpoint_batch in training_generator:
            for viewpoint_cam in viewpoint_batch:
                if Use_Consistency_Graph:
                    current_camera_index = int(consistency_graph.metropolis_hastings_walk(cons_graph, (str(current_camera_index))))
                    print(current_camera_index)
                    #current_camera_index = consistency_graph.random_walk_node(cons_graph, current_camera_index, train_image_counts)
                    #train_image_counts[current_camera_index] += 1
                    viewpoint_cam = train_camera_data_set[int(current_camera_index)]
                #camera_direction = torch.tensor(viewpoint_cam.R[:, 2], dtype=torch.float32)
                #viewpoint_cam = train_camera_data_set[0]
                #recompute Gaussian weights
                if Gradient_Propagation and iteration % 10 == 0:
                    gaussians.recompute_weights()
                
                viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
                viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()

                #Then with blending training
                iter_start.record()
                
                xyz_lr = gaussians.xyz_scheduler_args(iteration)
                # Every 1000 its we increase the levels of SH up to a maximum degree
                if iteration % 1000 == 0:
                    gaussians.oneupSHdegree()

                clock()
                
                #if not Cache_SPTs:
                #    ############# Individual Gaussian Cache
                #    new_render_indices, SPT_indices, SPT_distances = gaussians.get_SPT_cut(viewpoint_cam.camera_center.cuda(),  viewpoint_cam.full_proj_transform.cuda(), 0, viewpoint_cam, pipe, Use_Bounding_Spheres, Use_Frustum_Culling, Use_Occlusion_Culling)
                #    keep_mask = torch.isin(render_indices, new_render_indices).to(Storage_Device)
                #    write_back_mask = ~keep_mask
                #    write_back_indices = render_indices[write_back_mask].to(Storage_Device)
                #    load_mask = ~torch.isin(new_render_indices, render_indices)
                #    load_from_disk_indices = new_render_indices[load_mask].to(Storage_Device)
                #    render_indices = torch.cat((render_indices[keep_mask], new_render_indices[load_mask]))
                #    print(f"Load Percentage: {len(load_from_disk_indices) * 100.0 / len(render_indices):.2f}" )
                #    
                #    hierarchy_cut_time = clock()
                #    clock()
                #    
                #    # Write back the training results
                #    with torch.no_grad():
                #        gaussians._xyz[write_back_indices, :] = means3D[write_back_mask, :].detach().to(Storage_Device, non_blocking=False)
                #        gaussians._opacity[write_back_indices, :] = opacity[write_back_mask, :].detach().to(Storage_Device, non_blocking=False)
                #        gaussians._features_dc[write_back_indices, :] = features_dc[write_back_mask, :,  :].detach().to(Storage_Device, non_blocking=False)
                #        gaussians._features_rest[write_back_indices, :] = features_rest[write_back_mask, :].detach().to(Storage_Device, non_blocking=False)
                #        gaussians._scaling[write_back_indices, :] = scales[write_back_mask, :].detach().to(Storage_Device, non_blocking=False)
                #        gaussians._rotation[write_back_indices, :] = rotations[write_back_mask, :].detach().to(Storage_Device, non_blocking=False)
                #    
                #    means3D = nn.Parameter(torch.cat((means3D[keep_mask].detach(), gaussians._xyz[load_from_disk_indices].cuda(non_blocking=non_blocking))).contiguous())
                #    opacity = nn.Parameter(torch.cat((opacity[keep_mask].detach(), gaussians._opacity[load_from_disk_indices].cuda(non_blocking=non_blocking))).contiguous())
                #    scales = nn.Parameter(torch.cat((scales[keep_mask].detach(), gaussians._scaling[load_from_disk_indices].cuda(non_blocking=non_blocking))).contiguous())
                #    rotations = nn.Parameter(torch.cat((rotations[keep_mask].detach(), gaussians._rotation[load_from_disk_indices].cuda(non_blocking=non_blocking))).contiguous())
                #    # TODO: ABS?
                #    features_dc = nn.Parameter(torch.cat((features_dc[keep_mask].detach(), gaussians._features_dc[load_from_disk_indices].cuda(non_blocking=non_blocking))).contiguous())
                #    features_rest = nn.Parameter(torch.cat((features_rest[keep_mask].detach(), gaussians._features_rest[load_from_disk_indices].cuda(non_blocking=non_blocking))).contiguous())
                #    torch.cuda.synchronize()
                #    shs = torch.cat((features_dc, features_rest), dim=1).contiguous()
                #    
                #    #scene.dump_gaussians("Dump", False, file_name="View.ply", indices=render_indices.to(Storage_Device))
                #    
                #    parameters_new = []
                #    for index, (values, name, lr) in enumerate(zip([means3D, features_dc, opacity, scales, rotations, features_rest], 
                #                            ["xyz", "f_dc", "opacity", "scaling", "rotation", "f_rest"],
                #                            [xyz_lr, opt.feature_lr, opt.opacity_lr, opt.scaling_lr, opt.rotation_lr, opt.feature_lr])):
                #        optimizer_state[name]["exp_avgs"][write_back_indices] = parameters[index]["exp_avgs"][write_back_mask].to(Storage_Device)
                #        optimizer_state[name]["exp_avgs_sqs"][write_back_indices] = parameters[index]["exp_avgs_sqs"][write_back_mask].to(Storage_Device)
                #        exp_avgs = torch.cat((parameters[index]["exp_avgs"][keep_mask], optimizer_state[name]["exp_avgs"][load_from_disk_indices].cuda())).contiguous()
                #        exp_avgs_sqs = torch.cat((parameters[index]["exp_avgs_sqs"][keep_mask], optimizer_state[name]["exp_avgs_sqs"][load_from_disk_indices].cuda())).contiguous()
                #        parameters_new.append({'params': [values], 'lr': lr*lr_multiplier, "name": name, 
                #         "exp_avgs" : exp_avgs, "exp_avgs_sqs" : exp_avgs_sqs})
                #
                #
                #    #render_indices_cpu = indices
                #    parameters = parameters_new
                #    load_write_time = clock()
                #    clock()
                #    ############# Individual Gaussian Cache
                if Cache_SPTs:
                    number_to_render = 1000000000
                    distance_multiplier = 1
                    while number_to_render > Max_Gaussian_Budget:
                        if distance_multiplier > 1:
                            print("Lowered Granularity")
                        ############# SPT Cache
                        if Use_Bounding_Spheres:
                            bounds = gaussians.bounding_sphere_radii
                        else: 
                            bounds = (gaussians.scaling_activation(torch.max(gaussians.upper_tree_scaling, dim=-1)[0]) * 3.0)
                        planes = gaussians.extract_frustum_planes(viewpoint_cam.full_proj_transform.cuda())
                        if Use_Frustum_Culling:
                            cull = lambda indices : gaussians.frustum_cull_spheres(gaussians.upper_tree_xyz[indices], bounds[indices], planes)
                        else:
                            cull = lambda indices : torch.ones(len(indices), dtype = torch.bool)
                        camera_position = viewpoint_cam.camera_center.cuda()
                        LOD_detail_cut = lambda indices : gaussians.min_distance_squared[indices] > (camera_position - gaussians.upper_tree_xyz[indices]).square().sum(dim=-1) * distance_multiplier
                        #LOD_detail_cut = lambda indices : torch.ones_like(indices, dtype=torch.bool)
                        coarse_cut = gaussians.cut_hierarchy_on_condition(gaussians.upper_tree_nodes, LOD_detail_cut, return_upper_tree=False, root_node=0, leave_out_of_cut_condition=cull)

                        if Use_Occlusion_Culling:
                            bg_color = [0, 0, 0]
                            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                            temp = len(coarse_cut)
                            occlusion_indices = gaussians.upper_tree_nodes[coarse_cut, 5]
                            occlusion_mask = occlusion_cull(occlusion_indices.to(Storage_Device), gaussians, viewpoint_cam, pipe, background).cuda()
                            coarse_cut = coarse_cut[occlusion_mask]
                            print(f"Occlusion Cull {temp - len(coarse_cut)} out of {temp} upper tree gaussians")
                        leaf_mask = gaussians.upper_tree_nodes[coarse_cut, 2] == 0
                        leaf_nodes = coarse_cut[leaf_mask]
                        # separate the cut into leafs that contain an SPT and those that don't
                        SPT_indices = gaussians.upper_tree_nodes[leaf_nodes][gaussians.upper_tree_nodes[leaf_nodes, 3] >= 0, 3]
                        upper_tree_nodes_to_render = gaussians.upper_tree_nodes[leaf_nodes][gaussians.upper_tree_nodes[leaf_nodes, 3] <= 0, 5]

                        SPT_node_indices = leaf_nodes[gaussians.upper_tree_nodes[leaf_nodes, 3] >= 0]
                        SPT_distances = (gaussians.upper_tree_xyz[SPT_node_indices] - camera_position).pow(2).sum(1).sqrt() * distance_multiplier


                        SPTs_prev_to_new = torch.searchsorted(SPT_indices, prev_SPT_indices)
                        # What if a prev_SPT_index is bigger than all SPT_indices?
                        valid = (SPTs_prev_to_new < SPT_indices.numel())
                        valid[valid.clone()] &= (SPT_indices[SPTs_prev_to_new[valid]] == prev_SPT_indices[valid])
                        prev_distances_compare = prev_SPT_distances[valid]
                        distances_compare = SPT_distances[SPTs_prev_to_new[valid]]
                        close_enough = torch.isclose(distances_compare, prev_distances_compare, rtol=Reuse_SPT_Tolerarance, atol=0.05)
                        keep_SPT_indices = SPT_indices[SPTs_prev_to_new[valid][close_enough]]

                        valid_non_zero = torch.nonzero(valid, as_tuple=True)[0]
                        close_enough_non_zero = torch.nonzero(close_enough, as_tuple=True)[0]
                        SPT_keep_counts_indices = valid_non_zero[close_enough_non_zero]

                        #load_SPT_distances = SPT_distances[keep_SPT_indices[valid]][close_enough]

                        keep_gaussians_mask = torch.zeros(len(render_indices), dtype=torch.bool, device='cuda')
                        for i in SPT_keep_counts_indices:
                            if i == len(prev_SPT_counts)-1:
                                to = len(render_indices) - len(load_from_disk_indices)
                            else:
                                to = prev_SPT_counts[i+1]
                            keep_gaussians_mask[prev_SPT_counts[i]:to] = True

                        # Keep Skybox
                        keep_gaussians_mask[:gaussians.skybox_points] = True


                        mask = torch.isin(SPT_indices, keep_SPT_indices)
                        load_SPT_indices = SPT_indices[~mask]


                        load_SPT_distances = SPT_distances[~mask]
                        #load_SPT_distances = (gaussians.upper_tree_xyz[load_SPT_indices] - camera_position).pow(2).sum(1).sqrt() * distance_multiplier

                        if len(load_SPT_indices) > 0:
                            #LOAD SPT CUT
                            cut_SPTs, SPT_counts = get_spt_cut_cuda(len(load_SPT_indices), gaussians.SPT_gaussian_indices, gaussians.SPT_starts, gaussians.SPT_max, gaussians.SPT_min, load_SPT_indices, load_SPT_distances)
                        else:
                            print("No SPTs loaded")
                            cut_SPTs, SPT_counts = torch.empty(0, dtype=torch.int32, device='cuda'), torch.empty(0, dtype=torch.int32, device='cuda')
                        SPT_counts += gaussians.skybox_points
                        SPT_indices = torch.cat((keep_SPT_indices, load_SPT_indices))

                        SPT_counts_new = torch.zeros(len(keep_SPT_indices) + len(load_SPT_indices),dtype=torch.int32, device='cuda')
                        # compact the prefix sum of SPT_counts
                        prefix = 0
                        for index, i in enumerate(SPT_keep_counts_indices):
                            SPT_counts_new[index] = prefix
                            if i == len(prev_SPT_counts)-1:
                                to = len(render_indices) - len(load_from_disk_indices)
                            else:
                                to = prev_SPT_counts[i+1]
                            prefix += (to - prev_SPT_counts[i]).item()

                        SPT_counts_new[len(keep_SPT_indices):] = SPT_counts + prefix
                        # TODO: Remove SPT_distances that are loaded
                        SPT_distances = torch.cat((prev_SPT_distances[SPT_keep_counts_indices], load_SPT_distances))

                        load_from_disk_indices = torch.cat((cut_SPTs,  upper_tree_nodes_to_render))
                        
                        write_back_mask = ~keep_gaussians_mask
                        write_back_indices = render_indices[write_back_mask].detach().to(Storage_Device)
                        #write_back_indices.is_guilty=True

                        render_indices = torch.cat((render_indices[keep_gaussians_mask], load_from_disk_indices))
                        print(f"Load Percent: {len(load_from_disk_indices)/ len(render_indices)}")
                        load_from_disk_indices = load_from_disk_indices.to(Storage_Device)
                        number_to_render = len(render_indices)
                        distance_multiplier = distance_multiplier * Distance_Multiplier_Until_Budget
                    hierarchy_cut_time = clock()
                    clock()
                    
                    # not sure why this is needed
                    #with torch.no_grad():
                    # Write back the training results
                    #stream = torch.cuda.Stream()

                    #with torch.cuda.stream(stream):
                    gaussians._xyz[write_back_indices, :] = means3D[write_back_mask, :].detach().to(Storage_Device, non_blocking=non_blocking)
                    gaussians._opacity[write_back_indices, :] = opacity[write_back_mask, :].detach().to(Storage_Device, non_blocking=non_blocking)
                    gaussians._features_dc[write_back_indices, :] = features_dc[write_back_mask, :,  :].detach().to(Storage_Device, non_blocking=non_blocking)
                    gaussians._features_rest[write_back_indices, :] = features_rest[write_back_mask, :].detach().to(Storage_Device, non_blocking=non_blocking)
                    gaussians._scaling[write_back_indices, :] = scales[write_back_mask, :].detach().to(Storage_Device, non_blocking=non_blocking)
                    gaussians._rotation[write_back_indices, :] = rotations[write_back_mask, :].detach().to(Storage_Device, non_blocking=non_blocking)
                    #torch.cuda.synchronize()
                    
                    means3D = means3D[keep_gaussians_mask]
                    opacity = opacity[keep_gaussians_mask]
                    scales = scales[keep_gaussians_mask]
                    rotations = rotations[keep_gaussians_mask]
                    features_dc = features_dc[keep_gaussians_mask]
                    features_rest = features_rest[keep_gaussians_mask]
                    torch.cuda.empty_cache()
                    #stream = torch.cuda.Stream()

                    #with torch.cuda.stream(stream):
                    means3D = nn.Parameter(torch.cat((means3D.detach(), gaussians._xyz[load_from_disk_indices].cuda(non_blocking=non_blocking))).contiguous())
                    opacity = nn.Parameter(torch.cat((opacity.detach(), gaussians._opacity[load_from_disk_indices].cuda(non_blocking=non_blocking))).contiguous())
                    scales = nn.Parameter(torch.cat((scales.detach(), gaussians._scaling[load_from_disk_indices].cuda(non_blocking=non_blocking))).contiguous())
                    rotations = nn.Parameter(torch.cat((rotations.detach(), gaussians._rotation[load_from_disk_indices].cuda(non_blocking=non_blocking))).contiguous())
                    # TODO: ABS?
                    features_dc = nn.Parameter(torch.cat((features_dc.detach(), gaussians._features_dc[load_from_disk_indices].cuda(non_blocking=non_blocking))).contiguous())
                    features_rest = nn.Parameter(torch.cat((features_rest.detach(), gaussians._features_rest[load_from_disk_indices].cuda(non_blocking=non_blocking))).contiguous())
                    #shs = torch.cat((features_dc, features_rest), dim=1).contiguous()
                    
                    #render_indices_cpu = render_indices.to(Storage_Device)
                    
                    parameters_new = []
                    for index, (values, name, lr) in enumerate(zip([means3D, features_dc, opacity, scales, rotations, features_rest], 
                                            ["xyz", "f_dc", "opacity", "scaling", "rotation", "f_rest"],
                                            [xyz_lr, opt.feature_lr, opt.opacity_lr, opt.scaling_lr, opt.rotation_lr, opt.feature_lr])):
                        optimizer_state[name]["exp_avgs"][write_back_indices] = parameters[index]["exp_avgs"][write_back_mask].to(Storage_Device)
                        optimizer_state[name]["exp_avgs_sqs"][write_back_indices] = parameters[index]["exp_avgs_sqs"][write_back_mask].to(Storage_Device)
                        exp_avgs = torch.cat((parameters[index]["exp_avgs"][keep_gaussians_mask], optimizer_state[name]["exp_avgs"][load_from_disk_indices].cuda())).contiguous()
                        exp_avgs_sqs = torch.cat((parameters[index]["exp_avgs_sqs"][keep_gaussians_mask], optimizer_state[name]["exp_avgs_sqs"][load_from_disk_indices].cuda())).contiguous()
                        parameters_new.append({'params': [values], 'lr': lr*lr_multiplier, "name": name, 
                         "exp_avgs" : exp_avgs, "exp_avgs_sqs" : exp_avgs_sqs})
                    parameters = parameters_new
                    
                    prev_SPT_indices = SPT_indices
                    prev_SPT_distances = SPT_distances
                    prev_SPT_counts = SPT_counts_new
                    
                    load_write_time = clock()
                    
                    del write_back_mask
                    del keep_gaussians_mask
                    del write_back_indices
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                # parent indices contains as many elements as indices
                #interpolation_weights = torch.zeros(gaussians._xyz.size(0), dtype=torch.float32, device = gaussians._xyz.device)
                #interpolation_weights[:len(render_indices)] = 1.0
                pre_render_peak = torch.cuda.max_memory_allocated(device='cuda')
                torch.cuda.reset_peak_memory_stats()
                # Render
                if Rasterizer == "Hierarchical":
                    render_pkg = render_on_disk(
                    viewpoint_cam, 
                    means3D,
                    gaussians.opacity_activation(opacity),
                    gaussians.scaling_activation(scales), 
                    gaussians.rotation_activation(rotations),
                    shs,
                    pipe, 
                    background,
                    sh_degree = gaussians.active_sh_degree)
                elif Rasterizer == "Vanilla":
                    render_pkg = render_vanilla(
                        viewpoint_cam, 
                        means3D,
                        gaussians.opacity_activation(opacity),
                        gaussians.scaling_activation(scales), 
                        gaussians.rotation_activation(rotations),
                        features_dc,
                        features_rest,
                        pipe, 
                        background,
                        #splat_args=splat_settings,
                        sh_degree = gaussians.active_sh_degree,
                        anti_aliasing=Anti_Aliasing
                        )
                else:
                    render_pkg = render_stp(
                        viewpoint_cam, 
                        means3D,
                        gaussians.opacity_activation(opacity),
                        gaussians.scaling_activation(scales), 
                        gaussians.rotation_activation(rotations),
                        shs,
                        pipe, 
                        background,
                        splat_args=splat_settings,
                        sh_degree = gaussians.active_sh_degree,
                        )

                post_render_peak = torch.cuda.max_memory_allocated(device='cuda')
                torch.cuda.reset_peak_memory_stats()
                image = render_pkg["render"]#, render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                #torchvision.utils.save_image(image, os.path.join(scene.model_path, str(iteration) + "Sky.png"))
                #exit()
                #if iteration == 500:
                #    torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
                #    exit()
                
                # Loss
                gt_image = viewpoint_cam.original_image.cuda()
                #if iteration == 1500:
                #    gaussians.save_hier()
                if iteration % 50 == 0 or iteration == 1:
                    torchvision.utils.save_image(image, os.path.join(scene.model_path, str(iteration) + ".png"))
                    #torchvision.utils.save_image(gt_image, os.path.join(scene.model_path, str(iteration) + "_gt.png"))
                #if viewpoint_cam.alpha_mask is not None:
                #    Ll1 = l1_loss(image * viewpoint_cam.alpha_mask.cuda(), gt_image)
                #    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - fused_ssim((image * viewpoint_cam.alpha_mask.cuda()).unsqueeze(0), gt_image.unsqueeze(0)))
                #else:
                Ll1 = l1_loss(image, gt_image) 
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
                    
                #parents = gaussians.nodes[indices, 1]
                #hierarchy_loss = 0 #torch.sum(torch.clamp_min(torch.max(torch.abs(gaussians.get_scaling[indices]), dim=-1)[0] - torch.max(torch.abs(gaussians.get_scaling[parents]), dim=-1)[0], 0)) / len(indices)
                #loss = loss + lambda_hierarchy * hierarchy_loss
                #MCMC
                if MCMC_Densification:
                    opacity_loss = torch.sum(torch.abs(gaussians.opacity_activation(opacity))) / len(render_indices)
                    scaling_loss = torch.sum(torch.abs(gaussians.scaling_activation(scales))) / len(render_indices)
                        # loss for active gaussians that have high opacity/scale
                        #all_indices = torch.cat((indices, parents)).unique()
                        #opacity_loss = torch.sum(torch.abs(gaussians.get_opacity.squeeze()[all_indices] * interpolation_weights[all_indices])) / len(all_indices)
                        
                        #scaling_loss = torch.sum(torch.sum(torch.abs(gaussians.get_scaling[all_indices]), dim=-1) * interpolation_weights[all_indices]) / (3*len(all_indices))
                if lambda_opacity > 0:
                    loss = loss + lambda_opacity * opacity_loss    
                if lambda_scaling > 0:
                    loss = loss + lambda_scaling * scaling_loss
                #MCMC
                if math.isnan(loss):
                            torchvision.utils.save_image(image, os.path.join(scene.model_path, "Error" + ".png"))
                            print("gradients collapsed :(")
                #make_dot(loss).render("graph", format="png"                
                loss.backward()
                post_backward_peak = torch.cuda.max_memory_allocated(device='cuda')
                torch.cuda.reset_peak_memory_stats()
                ############## DEBUG LABELS
                if False:
                    render_indices.label = 'render_indices'
                    means3D.label = "means3D"
                    opacity.label = "opacity"
                    scales.label = "scales" 
                    rotations.label = "rotations" 
                    features_dc.label = "features_dc"
                    features_rest.label = "features_rest"
                    #shs.label = 'shs'
                    means3D.grad.label = "means3Dgrad"
                    opacity.grad.label = "opacitygrad"
                    scales.grad.label = "scalesgrad" 
                    rotations.grad.label = "rotationsgrad" 
                    features_dc.grad.label = "features_dcgrad"
                    features_rest.grad.label = "features_restgrad"
                    for param in parameters: 
                        param['exp_avgs'].label = param['name'] + '_exp_avgs'
                        param['exp_avgs_sqs'].label = param['name'] + '_exp_avgs_sqs'
                    SPT_counts_new.label = "SPT_counts_new"
                    SPT_counts.label = "SPT_counts"
                    #write_back_mask.label = "write_back_mask"
                    SPT_indices.label = "SPT_indices"
                    keep_SPT_indices.label = "keep_SPT_indices"
                    SPTs_prev_to_new.label = "SPTs_prev_to_new"
                    SPT_distances.label = "SPT_distances"
                    SPT_node_indices.label = "SPT_node_indices"
                    LOD_detail_cut.label = "LOD_detail_cut"
                    coarse_cut.label = "coarse_cut"
                    leaf_mask.label = "leaf_mask"
                    leaf_nodes.label = "leaf_nodes"
                    valid.label = "valid"
                    prev_distances_compare.label = "prev_distances_compare"
                    distances_compare.label = "distances_compare"
                    close_enough.label = "close_enough"
                    valid_non_zero.label = "valid_non_zero"
                    close_enough_non_zero.label = "close_enough_non_zero"
                    SPT_keep_counts_indices.label = "SPT_keep_counts_indices"
                    #keep_gaussians_mask.label = "keep_gaussians_mask"
                    mask.label = "mask"
                    upper_tree_nodes_to_render.label = "upper_tree_nodes_to_render"
                    cut_SPTs.label = "cut_SPTs"
                    prev_SPT_indices.label = "prev_SPT_indices"
                    prev_SPT_distances.label = "prev_SPT_distances"
                    prev_SPT_counts.label = "prev_SPT_counts"
                    gaussians.SPT_starts.label = "SPT_starts"
                    gaussians.SPT_min.label = "SPT_min"
                    gaussians.SPT_max.label = "SPT_max"
                    gaussians.SPT_gaussian_indices.label = "SPT_gaussian_indices" 
                    gaussians.upper_tree_nodes.label = "upper_tree_nodes"
                    gaussians.upper_tree_xyz.label = "upper_tree_xyz"
                    gaussians.upper_tree_scaling.label = "upper_tree_scaling"
                    gaussians.min_distance_squared.label = "min_distance_squared"
                    gt_image.label = "gt_image"
                    bounds.label = "bounds"
                    planes.label = "planes"
                    render_pkg["render"].label = "image"
                    render_pkg["viewspace_points"].label = "means2D"
                    render_pkg["viewspace_points"].grad.label = "means2D_grad"
                ############## DEBUG LABELS
                
                if means3D.grad.isnan().sum() > 0:
                            torchvision.utils.save_image(image, os.path.join(scene.model_path, "Error" + ".png"))
                            print("gradients collapsed :(")
                # Write values for every iteration
                if WriteTensorBoard and iteration % 10 == 0:
                    writer.add_scalar('VRAM usage', torch.cuda.memory_allocated(0), iteration)
                    process = psutil.Process(pid)
                    mem_info = process.memory_info()
                    writer.add_scalar('CPU RAM usage', mem_info.rss / 1024 ** 2, iteration)
                    writer.add_scalar('Peak VRAM usage', torch.cuda.max_memory_allocated(device='cuda'), iteration)
                    writer.add_scalar('Total Loss', loss, iteration)
                    writer.add_scalar('Opacity Loss', opacity_loss, iteration)
                    writer.add_scalar('Scaling Loss', scaling_loss, iteration)
                    writer.add_scalar('Mean Opacity', torch.mean(gaussians.opacity_activation(opacity)), iteration)
                    writer.add_scalar('Mean Scaling', torch.mean((gaussians.scaling_activation(scales))), iteration)
                    writer.add_scalar('Number of Gaussians loaded', len(load_from_disk_indices), iteration)
                    writer.add_scalar('Percentage of Gaussians loaded', len(load_from_disk_indices)*100 /len(render_indices), iteration)
                    writer.add_scalar('Number of Gaussians rendered', len(render_indices), iteration)
                    writer.add_scalar('Hierarchy Cut Time', hierarchy_cut_time, iteration)
                    writer.add_scalar('Memory Load / Write Time', load_write_time, iteration)
                    writer.add_scalar('Rendered SPTs', len(SPT_indices), iteration)
                    writer.add_scalar('Rendered SPTs Percentage', len(SPT_indices)/len(gaussians.SPT_starts), iteration)
                    total_SPT_nodes = gaussians.SPT_starts[SPT_indices+1] - gaussians.SPT_starts[SPT_indices]
                    writer.add_scalar('Rendered SPTs Detail', (len(render_indices)-gaussians.skybox_points)/total_SPT_nodes.sum(), iteration)
                    
                    writer.add_scalar('Peak before Render', pre_render_peak, iteration)
                    writer.add_scalar('Peak during Render', post_render_peak, iteration)
                    writer.add_scalar('Peak during backwards', post_backward_peak, iteration)
                    

                iter_end.record()

                with torch.no_grad():
                    # Progress bar
                    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                    if torch.cuda.max_memory_allocated(device='cuda') > prev_peak_memory:
                        prev_peak_memory = torch.cuda.max_memory_allocated(device='cuda')
                        print(f"New peak memory reached {len(render_indices):_}, before: {prev_number_rendered:_}")
                    prev_number_rendered = len(render_indices)
                    if iteration % 10 == 0:
                        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Size": f"{gaussians.size:_}/{gaussians._xyz.size(0):_}", "Peak memory": f"{torch.cuda.max_memory_allocated(device='cuda'):_}"})
                        progress_bar.update(10)

                    # Log and save
                    if (iteration in saving_iterations):
                        print("\n[ITER {}] Saving Gaussians".format(iteration))
                        scene.save(iteration)
                        print("peak memory: ", torch.cuda.max_memory_allocated(device='cuda'))

                    if iteration == opt.iterations:
                        gaussians.save_hier()
                        progress_bar.close()
                        #scene.dump_gaussians("Dump", only_leaves=True, file_name="ResultCloud")
                        print(f"Hierarchy bounding sphere divergence: {scene.gaussians.compute_bounding_sphere_divergence()}")

                        debug_utils.render_depth_slices(scene, pipe, dataset.scaffold_file)
                        debug_utils.render_level_slices(scene, pipe, dataset.scaffold_file)
                        return


                    
                    if MCMC_Densification and opt.densify_from_iter < iteration < opt.densify_until_iter and iteration % opt.densification_interval == 0:
                        print("-----------------DENSIFY!--------------------")
                            
                        gaussians._xyz[render_indices, :] = means3D.detach().to(Storage_Device)
                        gaussians._features_dc[render_indices, :] = features_dc.detach().to(Storage_Device)
                        gaussians._features_rest[render_indices, :] = features_rest.detach().to(Storage_Device)
                        gaussians._opacity[render_indices, :] = opacity.detach().to(Storage_Device)
                        gaussians._scaling[render_indices, :] = scales.detach().to(Storage_Device)
                        gaussians._rotation[render_indices, :] = rotations.detach().to(Storage_Device)    
                        for index, name in enumerate(["xyz", "f_dc", "opacity", "scaling", "rotation", "f_rest"]):
                            optimizer_state[name]["exp_avgs"][render_indices] = parameters[index]["exp_avgs"].to(Storage_Device)
                            optimizer_state[name]["exp_avgs_sqs"][render_indices] = parameters[index]["exp_avgs_sqs"].to(Storage_Device)
                        means3D, opacity, scales, rotations, features_dc, features_rest, render_indices = torch.empty((0, 3), device='cuda', dtype=torch.float32), torch.empty((0, 1), device='cuda', dtype=torch.float32), torch.empty((0, 3), device='cuda', dtype=torch.float32), torch.empty((0,4), device='cuda', dtype=torch.float32), torch.empty((0, 1, 3), device='cuda', dtype=torch.float32), torch.empty((0,15, 3), device='cuda', dtype=torch.float32), torch.empty(0, device='cuda', dtype=torch.int32)
                        prev_SPT_indices, prev_SPT_distances, prev_SPT_counts = torch.empty(0, device='cuda', dtype=torch.int32), torch.empty(0, device='cuda', dtype=torch.float32), torch.empty(0, device='cuda', dtype=torch.int32)
                        if Cache_SPTs:
                            render_indices = torch.arange(0, gaussians.skybox_points, device='cuda')
                            means3D = gaussians._xyz[:gaussians.skybox_points].cuda().detach().contiguous()
                            opacity = gaussians._opacity[:gaussians.skybox_points].cuda().detach().contiguous()
                            scales = gaussians._scaling[:gaussians.skybox_points].cuda().detach().contiguous()
                            rotations = gaussians._rotation[:gaussians.skybox_points].cuda().detach().contiguous()
                            features_dc = gaussians._features_dc[:gaussians.skybox_points].cuda().detach().contiguous()
                            features_rest = gaussians._features_rest[:gaussians.skybox_points].cuda().detach().contiguous()
                        
                        parameters = []
                        for values, name, lr in zip([means3D, features_dc, opacity, scales, rotations, features_rest], 
                                                ["xyz", "f_dc", "opacity", "scaling", "rotation", "f_rest"],
                                                [xyz_lr, opt.feature_lr, opt.opacity_lr, opt.scaling_lr, opt.rotation_lr, opt.feature_lr]):
                            parameters.append({'params': [values], 'lr': lr*lr_multiplier, "name": name, 
                             "exp_avgs" : torch.zeros_like(values), "exp_avgs_sqs" : torch.zeros_like(values)})
                        
                        
                        #dead_mask = gaussians.nodes[:size, 2] == 0
                        
                        dead_indices = torch.where((gaussians._opacity[:gaussians.size] <= gaussians.inverse_opacity_activation(torch.tensor(0.005)).item()).squeeze(-1))[0]

                        # Find SPT root nodes
                        gaussians.add_new_gs(cap_max=Max_Cap, size=gaussians.size)
                        SPT_root_hierarchy_indices = gaussians.build_hierarchical_SPT(SPT_Root_Volume, SPT_Target_Granularity, use_bounding_spheres=Use_Bounding_Spheres)
                        #torch.save(gaussians.SPT_min, "SPT_min.pt")
                        #torch.save(gaussians.SPT_max, "SPT_max.pt")
                        #torch.save(gaussians.SPT_starts, "SPT_starts.pt")
                        #torch.save(gaussians.upper_tree_xyz[SPT_node_indices], "SPT_positions.pt")
                        # Make sure that we don't mark newly spawned Gaussians as dead
                        dead_mask =torch.zeros(gaussians.size, dtype=torch.bool, device=Storage_Device)
                        dead_mask[dead_indices] = True
                        if Use_MIP_respawn:
                            # get the 10 closest cameras for each SPT root
                            # Points have to be on cpu to convert to numpy
                            min_distance, camera_indices = camera_KD_tree.query(gaussians._xyz[SPT_root_hierarchy_indices].cpu(), 1)
                            mask =torch.zeros(len(gaussians.SPT_gaussian_indices), dtype=torch.bool, device='cuda')
                            for i in range(len(gaussians.SPT_starts)-1):
                                mask[gaussians.SPT_starts[i] : gaussians.SPT_starts[i+1]] = gaussians.SPT_max[gaussians.SPT_starts[i] : gaussians.SPT_starts[i+1]] < min_distance[i]
                            dead_indices = gaussians.SPT_gaussian_indices[mask]
                            dead_mask[dead_indices.to(Storage_Device)] = True 
                            print(f"Respawn {len(dead_indices)} Gaussians because they were too small to see")
                        # only redistribute leaf nodes
                        dead_mask = torch.logical_and(dead_mask, gaussians.nodes[:gaussians.size, 2] == 0)
                        print(f"Respawn {torch.sum(dead_mask)} Gaussians")
                        gaussians.relocate_gs(dead_mask, gaussians.size, optimizer_state, storage_device=Storage_Device)
                           
                        gaussians.build_hierarchical_SPT(SPT_Root_Volume, SPT_Target_Granularity, use_bounding_spheres=Use_Bounding_Spheres)
                        
                        print(f"Max Train Image: {torch.max(train_image_counts)}, Min Train Image: {torch.min(train_image_counts)}")
                        torch.cuda.empty_cache()
                        if Gradient_Propagation:
                            gaussians.recompute_weights()

                        # Per-Densification Statistics
                        if WriteTensorBoard:
                            writer.add_scalar('Number of Hierarchy Levels', gaussians.get_number_of_levels(), iteration)
                            writer.add_scalar('Lowest leaf node level', torch.min(gaussians.nodes[gaussians.nodes[:, 3] <= 0, 0][gaussians.skybox_points:]).item(), iteration)
                            writer.add_scalar('Number of Gaussians', gaussians.size, iteration)
                            writer.add_scalar('Number of SPTs', len(gaussians.SPT_starts), iteration)
                            writer.add_scalar('Mean Number of Gaussians per SPT', torch.mean((gaussians.SPT_starts[1:] - gaussians.SPT_starts[:-1]).float()), iteration)
                            writer.add_scalar('Number of Dead Gaussians in SPTs', (gaussians.SPT_min==gaussians.SPT_max).sum(), iteration)
                            writer.add_scalar('Number of Respawns due to MIP Filter', len(dead_indices), iteration)
                            writer.add_scalar('Number of Respawns', torch.sum(dead_mask), iteration)
                            writer.add_scalar('Proportions of Gaussians in SPT', len(gaussians.SPT_min) / gaussians.size, iteration)
                            if Use_Bounding_Spheres:
                                mean_bounding_sphere_radius = gaussians.bounding_sphere_radii.mean()
                                mean_covariance_max_scale = torch.max(gaussians.scaling_activation(gaussians._scaling[gaussians.upper_tree_nodes[:, 5].to(Storage_Device)]), dim=-1)[0].mean()
                                writer.add_scalar('Mean Difference between Bounding Radius and 3Sigma', mean_bounding_sphere_radius - mean_covariance_max_scale, iteration)
                            
                    elif iteration < opt.iterations:
                        means3D.grad[0:gaussians.skybox_points, :] = 0
                        rotations.grad[0:gaussians.skybox_points, :] = 0
                        features_dc.grad[0:gaussians.skybox_points, :, :] = 0
                        features_rest.grad[0:gaussians.skybox_points, :, :] = 0
                        opacity.grad[0:gaussians.skybox_points, :] = 0
                        scales.grad[0:gaussians.skybox_points, :] = 0
                        if torch.sum(torch.isnan(opacity.grad)) > 0 or torch.sum(torch.isnan(means3D.grad)) > 0 or torch.sum(torch.isnan(scales.grad)) > 0:
                            print("Gradients Collapsed :(")
                            pass
                        #relevant = (opacity.grad.flatten() != 0).nonzero()
                        relevant = torch.ones(len(render_indices), dtype=torch.bool)
                        for param in parameters:
                            optimizer_function = OurAdam._global_single_tensor_adam2 if Global_ADAM else OurAdam._single_tensor_adam2
                            optimizer_function([param["params"][0]], 
                                                        [param["params"][0].grad], 
                                                        [param["exp_avgs"]], 
                                                        [param["exp_avgs_sqs"]],
                                                        None, 
                                                        [torch.tensor(iteration)], 
                                                        amsgrad=False, 
                                                        beta1 = 0.9, 
                                                        beta2 = 0.999, 
                                                        lr = param["lr"], 
                                                        #relevant=relevant, 
                                                        weight_decay=0, 
                                                        eps=1e-8, 
                                                        maximize=False, 
                                                        capturable=False)

                        if torch.sum(torch.isnan(opacity)) > 0 or torch.sum(torch.isnan(means3D)) > 0 or torch.sum(torch.isnan(scales)) > 0:
                            pass
                        
                        #if not On_Disk and gaussians._opacity.grad != None:
                        #    relevant = (gaussians._opacity.grad.flatten() != 0).nonzero()
                        #    relevant = relevant.flatten().long()
                        #    if Gradient_Propagation:
                        #        prev_xyz = gaussians._xyz[relevant].detach().clone()
                        #        prev_scaling = gaussians._scaling[relevant].detach().clone()
                        #        prev_features_dc = gaussians._features_dc[relevant].detach().clone()
                        #        prev_opacity = gaussians._opacity[relevant].detach().clone()
                        #    if(relevant.size(0) > 0):
                        #        gaussians.optimizer.step(relevant)
                        #    if Gradient_Propagation:
                        #        changed = relevant.detach().clone()
                        #        d_xyz = (gaussians._xyz[changed] - prev_xyz) * Propagation_Strength
                        #        d_scale = (gaussians._scaling[changed] - prev_scaling) * Propagation_Strength
                        #        d_features_dc = (gaussians._features_dc[changed] - prev_features_dc) * Propagation_Strength
                        #        d_opacity = (gaussians._opacity[changed] - prev_opacity) * Propagation_Strength 
                        #        # Upward Propagation:
                        #        with torch.no_grad():
                        #            #for i in range(1):
                        #            while(torch.sum(d_xyz) > 0.0001):
                        #                #print(len(relevant))
                        #                parents = gaussians.nodes[changed, 1]
                        #                unique_elements, counts = torch.unique(parents, return_counts=True)
                        #                unique_only = unique_elements[counts == 1]
#
                        #                mask = torch.isin(parents, unique_only)
                        #                mask = torch.logical_and(mask, parents > gaussians.skybox_points)
                        #                
                        #                #d_xyz = torch.zeros_like(d_xyz).scatter_add_()
                        #                
                        #                parents = parents[mask]
                        #                d_xyz *= gaussians.weights[changed].view(-1, 1)
                        #                d_scale *= gaussians.weights[changed].view(-1, 1)
                        #                d_features_dc[:, 0, :] *= gaussians.weights[changed].view(-1, 1)
                        #                d_xyz = d_xyz[mask]
                        #                d_scale = d_scale[mask]
                        #                d_features_dc = d_features_dc[mask]
                        #                d_opacity = d_opacity[mask]
                        #                gaussians._xyz[parents] +=  d_xyz
                        #                #gaussians._scaling[parents] +=  d_scale
                        #                gaussians._features_dc[parents] +=  d_features_dc
                        #                gaussians._opacity[parents] += d_opacity
                        #                changed = parents
                        #    gaussians.optimizer.zero_grad(set_to_none = True)
                            
                        if MCMC_Densification and MCMC_Noise_LR > 0:
                            
                            def op_sigmoid(x, k=100, x0=0.995):
                                return 1 / (1 + torch.exp(-k * (x - x0)))
                            # 5e5 = opt.noise_lr
                            if Only_Noise_Visible:
                                L = build_scaling_rotation(gaussians.scaling_activation(scales), gaussians.rotation_activation(rotations))
                                actual_covariance = L @ L.transpose(1, 2)
                                noise = torch.randn_like(means3D) * (op_sigmoid(1- gaussians.opacity_activation(opacity)))*MCMC_Noise_LR*xyz_lr
                                noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                                means3D += noise

                            
                            
                    if (iteration in checkpoint_iterations):
                        print("\n[ITER {}] Saving Checkpoint".format(iteration))
                        torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

                    iteration += 1
    

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    print(args)
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")
