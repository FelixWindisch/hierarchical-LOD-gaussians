
import os
import torch
from utils.loss_utils import l1_loss, ssim
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import math
import torchvision
from gaussian_renderer import render_coarse, render, render_post
import matplotlib
from matplotlib import colormaps
from scene import cameras
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('TkAgg')
from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights, expand_to_size_dynamic, get_interpolation_weights_dynamic

def direct_collate(x):
    return x


def get_gaussians_per_limit_normalized(scene, min_limit, max_limit, steps, no_images):
    gaussians = scene.gaussians
    result = torch.zeros((steps, no_images))
    render_indices = torch.zeros(gaussians._xyz.size(0)).int().cuda()
    parent_indices = torch.zeros(gaussians._xyz.size(0)).int().cuda()
    nodes_for_render_indices = torch.zeros(gaussians._xyz.size(0)).int().cuda()
    number_of_leaf_nodes = gaussians.get_number_of_leaf_nodes()
    n = 0
    limits = torch.linspace(min_limit, max_limit, steps)
    training_generator = DataLoader(scene.getTrainCameras(), num_workers = 8, prefetch_factor = 1, persistent_workers = True, collate_fn=direct_collate)
    for viewpoint_batch in training_generator:
        for viewpoint_cam in viewpoint_batch: 
            for index, limit in enumerate(limits):
                viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
                viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()

                camera_direction = torch.tensor([0.0, 0.0, 1.0])
                camera_direction = torch.matmul(torch.from_numpy(viewpoint_cam.R).to(torch.float32), camera_direction)


                to_render = expand_to_size_dynamic(
                        gaussians.nodes,
                        gaussians._xyz,
                        gaussians.get_scaling,
                        limit,
                        viewpoint_cam.camera_center,
                        camera_direction,
                        render_indices,
                        parent_indices,
                        nodes_for_render_indices)
                result[index, n] = to_render
            #result[:, n] /= result[0, n]
            n += 1
            if n >= no_images:
                return result
            

def plot_path_to_root(nodes, node, xyz):
    points = []
    n = node
    while(nodes[n, 1] != -1):
        points.append(xyz[n].cpu().detach().numpy())
        n = nodes[n, 1]
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, color='red', label='Points')

    ax.plot(x, y, z, color='blue', label='Line')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    

def generate_some_flat_scene_images(scene : Scene, pipe : PipelineParams, output_dir,  no_images = 10, indices = None):
    with torch.no_grad():
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        gaussians = scene.gaussians
        
        

        n = 0
        training_generator = DataLoader(scene.getTrainCameras(), num_workers = 8, prefetch_factor = 1, persistent_workers = True, collate_fn=direct_collate)
        for viewpoint_batch in training_generator:
            for viewpoint_cam in viewpoint_batch:
                
                viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
                viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()
                
                render_pkg = render_coarse(viewpoint_cam, gaussians, pipe, background, indices = indices)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                print(os.path.join(output_dir, "flat_" + str(n) + ".png"))
                torchvision.utils.save_image(image, os.path.join(output_dir, "flat_" + str(n) + ".png"))
                n += 1
                if n > no_images:
                    return None

def generate_hierarchy_scene_image(viewpoint_cam, scene : Scene, pipe : PipelineParams, limit = 0.05, show_depth=False):
    with torch.no_grad():
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        gaussians : GaussianModel = scene.gaussians
        #if show_depth:
        #    gaussians._opacity += 1
        render_indices = torch.zeros(gaussians._xyz.size(0)).int().cuda()
        parent_indices = torch.zeros(gaussians._xyz.size(0)).int().cuda()
        nodes_for_render_indices = torch.zeros(gaussians._xyz.size(0)).int().cuda()
        interpolation_weights = torch.zeros(gaussians._xyz.size(0)).float().cuda()
        num_siblings = torch.zeros(gaussians._xyz.size(0)).int().cuda()
        to_render = 0
        
        n = 0
        if show_depth:
            # the depth value is stored at index 0
            #depth_colors = [[gaussians.nodes[i][0].item(), gaussians.nodes[i][0].item(), gaussians.nodes[i][0].item()] for i in range(len(gaussians.nodes))]
            depth_colors= torch.transpose(torch.stack((gaussians.nodes[:, 0],gaussians.nodes[:, 0],gaussians.nodes[:, 0]), dim=0), 0, 1)
            override_colors = (torch.tensor(depth_colors, dtype=torch.float32, device="cuda"))/torch.max(gaussians.nodes[:, 0])
        else: 
            override_colors = None
                    
        viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
        viewpoint_cam.projection_matrix = cameras.getProjectionMatrix(   
                                                              znear=viewpoint_cam.znear, 
                                                              zfar=viewpoint_cam.zfar, 
                                                              fovX=viewpoint_cam.FoVx, 
                                                              fovY=viewpoint_cam.FoVy, 
                                                              primx = 0.5, primy=0.5).transpose(0,1).cuda()
        
        viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
        viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()

        R  = viewpoint_cam.world_view_transform[:3, :3].cuda()
        #viewpoint_cam.T  = viewpoint_cam.world_view_transform[3, :]
        camera_direction = torch.tensor([0.0, 0.0, 1.0])
        camera_direction = torch.matmul(R.to(torch.float32).cpu(), camera_direction)
        # output : render_indices, parent_indices, nodes_for_render_indices
        to_render = expand_to_size_dynamic(
                    gaussians.nodes,
                    gaussians._xyz,
                    gaussians.get_scaling,
                    limit,
                    viewpoint_cam.camera_center,
                    camera_direction,
                    render_indices,
                    parent_indices,
                    nodes_for_render_indices)
                #to_render = min(to_render, 569200)
               
                # indices == nodes_for_render_indices !
        indices = render_indices[:to_render].int()
        node_indices = nodes_for_render_indices[:to_render]


        # parent indices contains as many elements as indices

        get_interpolation_weights_dynamic(
            node_indices,
            limit,
            gaussians.nodes,
            gaussians._xyz,
            gaussians._scaling,
            viewpoint_cam.camera_center.cpu(),
            camera_direction,
            interpolation_weights,
            num_siblings
        )

        # num siblings always includes the node itself
        render_pkg = render_post(
            viewpoint_cam, 
            gaussians, 
            pipe, 
            background, 
            render_indices=indices,
            parent_indices = parent_indices,
            interpolation_weights = interpolation_weights,
            num_node_siblings = num_siblings,
            use_trained_exp=True, iteration=0,
            override_color=override_colors
            )
        image = render_pkg["render"]
        return image

def generate_some_hierarchy_scene_images(scene : Scene, pipe : PipelineParams, output_dir, limit = 0.05, no_images = 10, show_depth=False):
    with torch.no_grad():
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        gaussians : GaussianModel = scene.gaussians
        
        #if show_depth:
        #    gaussians._opacity += 1
        render_indices = torch.zeros(gaussians._xyz.size(0)).int().cuda()
        parent_indices = torch.zeros(gaussians._xyz.size(0)).int().cuda()
        nodes_for_render_indices = torch.zeros(gaussians._xyz.size(0)).int().cuda()
        interpolation_weights = torch.zeros(gaussians._xyz.size(0)).float().cuda()
        num_siblings = torch.zeros(gaussians._xyz.size(0)).int().cuda()
        to_render = 0
        
        n = 0
        training_generator = DataLoader(scene.getTrainCameras(), num_workers = 8, prefetch_factor = 1, persistent_workers = True, collate_fn=direct_collate)
        for viewpoint_batch in training_generator:
                for viewpoint_cam in viewpoint_batch:
                    
                    viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                    viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
                    viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                    viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()
                    # output : render_indices, parent_indices, nodes_for_render_indices
                    to_render = expand_to_size(
                        gaussians.nodes,
                        gaussians.boxes,
                        limit,
                        viewpoint_cam.camera_center,
                        torch.zeros((3)),
                        render_indices,
                        parent_indices,
                        nodes_for_render_indices
                    )
                    print("torender: " + str(to_render))
                    indices = render_indices[:to_render].int()
                    node_indices = nodes_for_render_indices[:to_render]

                    
                    # outputs : interpolation_weights, num_siblings
                    get_interpolation_weights(
                        node_indices,
                        limit,
                        gaussians.nodes,
                        gaussians.boxes,
                        viewpoint_cam.camera_center.cpu(),
                        torch.zeros((3)),
                        interpolation_weights,
                        num_siblings
                    )

                    if show_depth:
                        # the depth value is stored at index 0
                        #depth_colors = [[gaussians.nodes[i][0].item(), gaussians.nodes[i][0].item(), gaussians.nodes[i][0].item()] for i in range(len(gaussians.nodes))]
                        depth_colors= torch.transpose(torch.stack((gaussians.nodes[:, 0],gaussians.nodes[:, 0],gaussians.nodes[:, 0]), dim=0), 0, 1)
                        override_colors = 2*(torch.sigmoid(torch.tensor(depth_colors, dtype=torch.float32, device="cuda"))-0.5)
                    else: 
                        override_colors = None
                        
                    #render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                    render_pkg = render_post(
                        viewpoint_cam, 
                        gaussians, 
                        pipe, 
                        background, 
                        override_color=override_colors,
                        render_indices=indices,
                        parent_indices = parent_indices,
                        interpolation_weights = interpolation_weights,
                        num_node_siblings = num_siblings,
                        use_trained_exp=True,
                        )
                    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    out_path = os.path.join(output_dir, "hier_"  +str(n)  + ("_depth" if show_depth else "")+ ".png")
                    print(out_path)
                    torchvision.utils.save_image(image, out_path)
                    n += 1
                    if n > no_images:
                        #if show_depth:
                        #    gaussians._opacity -= 1
                        return None
              
def render_level_slices(scene : Scene, pipe : PipelineParams, output_dir):
    with torch.no_grad():
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        gaussians = scene.gaussians
        

        n = 0
        training_generator = DataLoader(scene.getTrainCameras(), num_workers = 8, prefetch_factor = 1, persistent_workers = True, collate_fn=direct_collate)
        for viewpoint_batch in training_generator:
            for viewpoint_cam in viewpoint_batch:
                
                viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
                viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()
                
                indices = torch.where(gaussians.nodes[:, 2] == 0)[0]
                number_of_levels = torch.min(gaussians.nodes[indices, 0]).item()-1
                # for each level in the hierarchy:
                for level in range(number_of_levels):
                    render_pkg = render_coarse(viewpoint_cam, gaussians, pipe, background, indices = indices)
                    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    print(os.path.join(output_dir, "render_level_" + str(number_of_levels-level) + ".png"))
                    torchvision.utils.save_image(image, os.path.join(output_dir, "render_level_" + str(number_of_levels-level) + ".png"))
                    indices = gaussians.nodes[indices, 1].unique()
                return None
              
                    
def render_depth_slices(scene : Scene, pipe : PipelineParams, output_dir):
    with torch.no_grad():
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        gaussians = scene.gaussians
        

        n = 0
        training_generator = DataLoader(scene.getTrainCameras(), num_workers = 8, prefetch_factor = 1, persistent_workers = True, collate_fn=direct_collate)
        for viewpoint_batch in training_generator:
            for viewpoint_cam in viewpoint_batch:
                
                viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
                viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()
                
                # for each level in the hierarchy:
                for depth in range(torch.max(gaussians.nodes[:, 0])):
                    indices = torch.where(gaussians.nodes[:, 0] == depth)[0]
                    render_pkg = render_coarse(viewpoint_cam, gaussians, pipe, background, indices = indices)
                    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    print(os.path.join(output_dir, "render_depth_" + str(depth) + ".png"))
                    torchvision.utils.save_image(image, os.path.join(output_dir, "render_depth_" + str(depth) + ".png"))
                return None
                    
def generate_some_hierarchy_scene_images_dynamic(scene : Scene, pipe : PipelineParams, output_dir, limit = 0.05, no_images = 10, show_depth=False):
    with torch.no_grad():
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        gaussians : GaussianModel = scene.gaussians
        
        #if show_depth:
        #    gaussians._opacity += 1
        render_indices = torch.zeros(gaussians._xyz.size(0)).int().cuda()
        parent_indices = torch.zeros(gaussians._xyz.size(0)).int().cuda()
        nodes_for_render_indices = torch.zeros(gaussians._xyz.size(0)).int().cuda()
        interpolation_weights = torch.zeros(gaussians._xyz.size(0)).float().cuda()
        num_siblings = torch.zeros(gaussians._xyz.size(0)).int().cuda()
        to_render = 0
        
        n = 0
        training_generator = DataLoader(scene.getTrainCameras(), num_workers = 8, prefetch_factor = 1, persistent_workers = True, collate_fn=direct_collate, shuffle=False)
        #debug_subset = Subset(training_generator, list(range(no_images)))
        for viewpoint_batch in training_generator:
                for viewpoint_cam in viewpoint_batch:
                    
                    viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                    viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
                    viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                    viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()
                    # output : render_indices, parent_indices, nodes_for_render_indices
                    to_render = expand_to_size_dynamic(
                        gaussians.nodes,
                        gaussians.get_xyz,
                        gaussians.get_scaling,
                        limit,
                        viewpoint_cam.camera_center,
                        torch.zeros((3)),
                        render_indices,
                        parent_indices,
                        nodes_for_render_indices
                    )
                    print("torender: " + str(to_render))
                    indices = render_indices[:to_render].int()
                    node_indices = nodes_for_render_indices[:to_render]

                    
                    # outputs : interpolation_weights, num_siblings
                    get_interpolation_weights_dynamic(
                        node_indices,
                        limit,
                        gaussians.nodes,
                        gaussians.get_xyz,
                        gaussians.get_scaling,
                        viewpoint_cam.camera_center.cpu(),
                        torch.zeros((3)),
                        interpolation_weights,
                        num_siblings
                    )

                    if show_depth:
                        # the depth value is stored at index 0
                        #depth_colors = [[gaussians.nodes[i][0].item(), gaussians.nodes[i][0].item(), gaussians.nodes[i][0].item()] for i in range(len(gaussians.nodes))]
                        depth_colors= torch.transpose(torch.stack((gaussians.nodes[:, 0],gaussians.nodes[:, 0],gaussians.nodes[:, 0]), dim=0), 0, 1)
                        override_colors = 2*(torch.sigmoid(torch.tensor(depth_colors, dtype=torch.float32, device="cuda"))-0.5)
                    else: 
                        override_colors = None
                    #indices[-1]=(len(gaussians._xyz))
                    #indices[0]=0
                    render_pkg = render(viewpoint_cam, gaussians, pipe, background, indices=indices)
                    pipe.debug = True
                    #render_pkg = render_post(
                    #    viewpoint_cam, 
                    #    gaussians, 
                    #    pipe, 
                    #    background, 
                    #    override_color=override_colors,
                    #    render_indices=indices,
                    #    parent_indices=parent_indices,
                    #    interpolation_weights=interpolation_weights,
                    #    num_node_siblings=num_siblings,
                    #    use_trained_exp=True
                    #    )

                    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    out_path = os.path.join(output_dir, "hier_"  +str(n)  + ("_depth" if show_depth else "")+ ".png")
                    print(out_path)
                    torchvision.utils.save_image(image, out_path)
                    n += 1
                    if n > no_images:
                        #if show_depth:
                        #    gaussians._opacity -= 1
                        return None