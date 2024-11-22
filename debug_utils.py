
import os
import torch
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render_post
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import math
import torchvision
from gaussian_renderer import render_coarse
from matplotlib import colormaps
from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights

def direct_collate(x):
    return x


def generate_some_flat_scene_images(scene : Scene, pipe : PipelineParams, output_dir,  no_images = 10):
    with torch.no_grad():
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        gaussians = scene.gaussians
        
        indices = None

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
                torchvision.utils.save_image(image, os.path.join(output_dir, str(n) + ".png"))
                n += 1
                if n > no_images:
                    return None

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