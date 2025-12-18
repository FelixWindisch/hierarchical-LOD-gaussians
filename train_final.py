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
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
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
import view_graph_utils
from scipy.spatial import KDTree
import numpy as np
from gaussian_hierarchy._C import  get_spt_cut_cuda
#from stp_gaussian_rasterization import ExtendedSettings
import psutil
import gc
from datetime import datetime
import random
from globals import *
from preprocess.read_write_model import qvec2rotmat, rotmat2qvec, qvec2rotmat_torch, rotmat2qvec_torch, rotation_matrix_to_quaternion
import lod_slang_gaussian_rasterization.api.inria_3dgs as gaussian_renderer
import slang_gaussian_rasterization.api.inria_3dgs as occlusion_renderer
import slang_beta_rasterization.api.inria_3dgs as beta_renderer
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from fused_ssim import fused_ssim
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F
# to check CPU RAM usage
pid = os.getpid()

def plot_gaussian_columns(means, variances, colors,  min_vals, max_vals, weights=None, height=500, y_range=(-5, 5)):
    """
    Generates an image where each column is a Gaussian distribution.
    
    Args:
        means (Tensor): Shape (N,). Means of the distributions.
        variances (Tensor): Shape (N,). Variances of the distributions.
        colors (Tensor): Shape (N, 3). RGB colors in range [0, 1].
        height (int): Height of the resulting image in pixels.
        y_range (tuple): The (min, max) value represented by the image height.
        
    Returns:
        Tensor: Image tensor of shape (Height, N, 3).
    """
    device = means.device
    min_vals = min_vals.to(device)
    max_vals = max_vals.to(device)
    num_gaussians = means.shape[0]
    if weights is None:
        weights = torch.ones(num_gaussians, device=device)
    w = weights.view(1, -1)
    # 1. Setup the coordinate system
    # We create a column vector of Y coordinates (Height, 1)
    y_coords = torch.linspace(y_range[0], y_range[1], height, device=device).view(-1, 1)
    
    # Reshape inputs for broadcasting:
    # Means/Vars: (1, N)
    mu = means.view(1, -1)
    sigma2 = variances.view(1, -1)
    
    # 2. Calculate Gaussian Intensity (Unnormalized Kernel)
    # Result shape: (Height, N)
    # We use the kernel exp(-dist^2 / 2var) so the peak is always 1.0 (max opacity)
    squared_diff = (y_coords - mu)**2
    intensity = torch.exp(-squared_diff / (2 * (sigma2**2)))
    
    intensity *= w  # Apply weights to intensity
    # 3. Prepare for Color Mixing
    # Expand Intensity to (Height, N, 1) to broadcast against colors
    alpha = intensity.unsqueeze(-1)
    
    # Expand Colors to (1, N, 3) to broadcast against rows
    rgb = colors.view(1, num_gaussians, 3)
    
    # Define White (1, 1, 1)
    white = torch.tensor([1.0, 1.0, 1.0], device=device)
    
    # 4. Mix Colors
    # Formula: alpha * Color + (1 - alpha) * White
    image_tensor = alpha * rgb + (1 - alpha) * white
    def val_to_row_indices(values):
        """Converts data values to row indices [0, height-1]"""
        # 1. Normalize values to 0.0 - 1.0 based on y_range
        norm = (values - y_range[0]) / (y_range[1] - y_range[0])
        # 2. Scale to image height
        indices = (norm * (height - 1)).round().long()
        return indices

    # Convert min/max tensors to pixel indices
    min_idx = val_to_row_indices(min_vals)
    max_idx = val_to_row_indices(max_vals)
    
    # Column indices [0, 1, ..., N-1]
    col_idx = torch.arange(num_gaussians, device=device)
    
    # Define Red Color
    red = torch.tensor([1.0, 0.0, 0.0], device=device)

    # --- Plot Min Markers ---
    # We create a mask to ensure we don't try to draw outside the image (if min < y_range)
    valid_min = (min_idx >= 0) & (min_idx < height)
    # Apply Red using Advanced Indexing: image[rows, cols] = red
    image_tensor[min_idx[valid_min], col_idx[valid_min], :] = red
    
    # --- Plot Max Markers ---
    valid_max = (max_idx >= 0) & (max_idx < height)
    image_tensor[max_idx[valid_max], col_idx[valid_max], :] = red
    # Clip just in case, though math guarantees [0,1] bounds
    return image_tensor.clamp(0, 1)


def occlusion_cull_slang(gaussians, camera, pipe, background, opacity_multiplier = 1, scale_multiplier = 1):
    #shs = torch.cat((features_dc, features_rest), dim=1).contiguous()
    render_pkg = occlusion_renderer.render(
        camera, 
        gaussians.SPT_means3D, 
        gaussians.opacity_activation(gaussians.SPT_opacity), 
        gaussians.scaling_activation(gaussians.SPT_scales), 
        gaussians.rotation_activation(gaussians.SPT_rotations), 
        #gaussians.SPT_opacity,
        #gaussians.SPT_scales,
        #gaussians.SPT_rotations,
        gaussians.SPT_features_dc, 
        gaussians.SPT_features_rest, 
        gaussians.beta_activation(gaussians.SPT_beta*4),
        pipe, background)
    #torchvision.utils.save_image(render_pkg["render"], "occlusion.png")
    return render_pkg["contribution"], render_pkg["render"]




clock_start = True
clock_time = time.time()
def clock(print_time=False):
    global clock_start
    global clock_time
    if print_time:
        print(time.time()-clock_time, "Seconds")
        return  
    if clock_start:
        clock_start = False
        clock_time = time.time()
    else:
        clock_start = True
        return time.time()-clock_time



sub_clock_start = True
sub_clock_time = time.time()
def sub_clock(print_time=False):
    global sub_clock_start
    global sub_clock_time
    if print_time:
        print(time.time()-sub_clock_time, "Seconds")
        return  
    if sub_clock_start:
        sub_clock_start = False
        sub_clock_time = time.time()
    else:
        sub_clock_start = True
        return time.time()-sub_clock_time

def direct_collate(x):
    return x


Write_Tensor_Board = True
#Standard
#Culling


Use_Occlusion_Culling = False
# SPTs
Reuse_SPT_Tolerance_Closer = 2
Reuse_SPT_Tolerance_Farther = 1.3
Revive_Gaussians = False
#View Selection
# Rasterizer
Rasterizer = "Vanilla"
# Optimizer
Global_ADAM = False

non_blocking=False



# Start and end indices for the properties tensor from CPU Memory




def training(dataset, opt:OptimizationParams, pipe, saving_iterations, checkpoint_iterations, checkpoint, debug_from, view_graph, evaluation = False, viewer = False):
    global SH_properties, features_rest2, SH_properties, SH_properties_single, Write_Tensor_Board
    #opt.iterations = 20

    PSNR_total = 0
    SSIM_total = 0
    LPIPS_total = 0
    __post_backward_peak = 0
    __prev_peak_memory = 0
    __prev_number_rendered = 0
    
    depth_l1_weight = get_expon_lr_func(1.0, 0.01, max_steps=opt.position_lr_max_steps)

    #torch.cuda.memory._record_memory_history()
    #torch.autograd.set_detect_anomaly(True)
    
    if Write_Tensor_Board and not evaluation:
        writer = SummaryWriter("CampusChunkFinal_" + str(uuid.uuid4())[:8])
    gaussians = GaussianModel(opt.SH_degree)
    
    scene = Scene(dataset, gaussians, resolution_scales=[1], create_from_hier=True, llff_hold=opt.llff_hold)
    gaussians.max_sh_degree = opt.SH_degree
    gaussians.active_sh_degree = min(1, gaussians.max_sh_degree)
    
    features_rest2 = 14 + number_SH_properties[gaussians.max_sh_degree] * 3
    d_mu1 = features_rest2 
    d_mu2 = features_rest2 +1
    d_sigma1 = features_rest2 +1
    d_sigma2 = features_rest2 +2
    number_properties = d_sigma2
    range2[-1] = features_rest2
    range1.append(d_mu1)
    range1.append(d_sigma1)
    range2.append(d_mu2)
    range2.append(d_sigma2)
    
    SH_properties_single = number_SH_properties[gaussians.max_sh_degree] 
    SH_properties = number_SH_properties[gaussians.max_sh_degree] * 3
    
    # This is the focal length with which the SPT distances are computed
    base_focal_length = scene.getTrainCameras()[0].focal_length
    SPT_Target_Granularity = opt.target_granularity_pixels / base_focal_length
    
    for gaussian_parameter in [gaussians._xyz, gaussians._opacity, gaussians._rotation, gaussians._scaling, gaussians._features_dc, gaussians._features_rest]:
        gaussian_parameter.requires_grad_(False)
        
    gaussians.compact_gaussians(opt.storage_device, opt.cap_max, densification=opt.densification, prune_unused_gaussians=opt.prune_unused)
    # Add distance_mu and distance_sigma to properties
    temp = torch.zeros((len(gaussians.properties), d_sigma2*3), device=opt.storage_device)  
    temp[:, :features_rest2] = gaussians.properties[:, :features_rest2]
    gaussians.properties = temp
    torch.cuda.empty_cache()
    print(f"Gaussians Moved to {opt.storage_device}")
    
    if opt.optimize_exposure:
        gaussians.init_exposure_optimization(opt, scene.cam_infos)
    
    gaussians.build_hierarchical_SPT(opt.SPT_root_volume, SPT_Target_Granularity, opt.min_SPT_size, use_bounding_spheres=opt.use_bounding_spheres, revive_gaussians=Revive_Gaussians)
    
    
    #gaussians.sort_SPT_order()

    # set leaf nodes min to 0
    gaussians.SPT_min[gaussians.SPT_min < -1] = 0
    #TODO: What to do with this hardcoded value?
    gaussians.SPT_max[gaussians.SPT_max > 1e11] = gaussians.SPT_min[gaussians.SPT_max > 1e11] + 100
    
    gaussians.properties[gaussians.SPT_gaussian_indices, d_mu1] = ((gaussians.SPT_max + gaussians.SPT_min) / 2.0).cpu()
    difference = gaussians.SPT_max - gaussians.SPT_min
    
    # torch.sqrt(2*math.pi) ~= 2.5
    gaussians.properties[gaussians.SPT_gaussian_indices, d_sigma1] = ((difference) / math.sqrt(2*math.pi)).cpu()
    gaussians.SPT_max = (gaussians.properties[gaussians.SPT_gaussian_indices.cpu(), d_mu1] + 3 * gaussians.properties[gaussians.SPT_gaussian_indices.cpu(), d_sigma1]).cuda()
    gaussians.SPT_min = (gaussians.properties[gaussians.SPT_gaussian_indices.cpu(), d_mu1] - 3 * gaussians.properties[gaussians.SPT_gaussian_indices.cpu(), d_sigma1]).cuda()
    print(f"Built {len(gaussians.SPT_starts)} SPTs, which contain {len(gaussians.SPT_gaussian_indices)*100/(len(gaussians.SPT_gaussian_indices) + len(gaussians.upper_tree_nodes))} % of Gaussians")

    # Setup the SPTs for Frustum culling
    SPT_root_indices = gaussians.upper_tree_nodes[torch.logical_and(gaussians.upper_tree_nodes[:, hierarchy_node_child_count] == 0, gaussians.upper_tree_nodes[:, hierarchy_node_first_child] >= 0), 5].cpu()
    all_SPT_indices = gaussians.upper_tree_nodes[torch.logical_and(gaussians.upper_tree_nodes[:, hierarchy_node_child_count] == 0, gaussians.upper_tree_nodes[:, hierarchy_node_first_child] >= 0), hierarchy_node_first_child]
    upper_SPT_indices = torch.where(torch.logical_and(gaussians.upper_tree_nodes[:, hierarchy_node_child_count] == 0, gaussians.upper_tree_nodes[:, hierarchy_node_first_child] >= 0))[0]
    sorted, sort_indices = gaussians.upper_tree_nodes[upper_SPT_indices, hierarchy_node_first_child].sort()
    
    
    gaussians.SPT_means3D = nn.Parameter(gaussians.properties[SPT_root_indices, xyz1:xyz2].cuda().contiguous())
    gaussians.SPT_scales = nn.Parameter((gaussians.properties[SPT_root_indices, scales1:scales2].cuda().contiguous()))
    gaussians.SPT_rotations = nn.Parameter((gaussians.properties[SPT_root_indices, rotation1:rotation2].cuda().contiguous()))
    gaussians.SPT_features_dc = nn.Parameter(gaussians.properties[SPT_root_indices, features1:features2].cuda().unsqueeze(1).contiguous())
    gaussians.SPT_opacity = nn.Parameter((gaussians.properties[SPT_root_indices, opacity1].cuda().unsqueeze(1).contiguous()))
    gaussians.SPT_features_rest = nn.Parameter(gaussians.properties[SPT_root_indices, features_rest1: features_rest2].cuda().reshape(len(SPT_root_indices), SH_properties_single, 3).contiguous())
    gaussians.SPT_beta = nn.Parameter((torch.zeros(len(SPT_root_indices))).cuda().contiguous())

    upper_SPT_indices = upper_SPT_indices[sort_indices]

    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # rolling average loss for logging
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(0, opt.iterations), desc="Training progress")

    iteration = 0
    
    # DONT SHUFFLE WHEN USING CONSISTENCY GRAPH
    training_generator = DataLoader(scene.getTrainCameras(), num_workers = 8, prefetch_factor = 1, persistent_workers = True, collate_fn=direct_collate, shuffle=not opt.graph_view_select)
    if opt.graph_view_select:
        train_camera_data_set = scene.getTrainCameras()
        current_camera_index = list(view_graph.nodes())[0]
    else:
        current_camera_index = 0        
    
    
    gaussian_indices = torch.arange(0, gaussians.skybox_points, device='cuda', dtype = torch.int32)
    
    #TODO: Compact this
    means3D = gaussians.properties[:gaussians.skybox_points, xyz1:xyz2].cuda().contiguous()
    scales = gaussians.properties[:gaussians.skybox_points, scales1:scales2].cuda().contiguous()
    rotations = gaussians.properties[:gaussians.skybox_points, rotation1:rotation2].cuda().contiguous()
    features_dc = gaussians.properties[:gaussians.skybox_points, features1:features2].cuda().unsqueeze(1).contiguous()
    opacity = gaussians.properties[:gaussians.skybox_points, opacity1].cuda().unsqueeze(1).contiguous()
    features_rest = gaussians.properties[:gaussians.skybox_points, features_rest1: features_rest2].cuda().reshape(gaussians.skybox_points, SH_properties_single, 3).contiguous()
    distance_mu = gaussians.properties[:gaussians.skybox_points, d_mu1: d_mu2].cuda().contiguous()
    distance_sigma = gaussians.properties[:gaussians.skybox_points, d_sigma1: d_sigma2].cuda().contiguous()

    
    
    if opt.densification == "classic":
        densification_criterium = torch.zeros(gaussians.skybox_points, device='cuda', dtype=torch.float32)
        densification_criterium_cache = torch.empty((0), device='cuda', dtype=torch.float32)
    if opt.prune_unused and not evaluation:
        contributed = torch.zeros(gaussians.skybox_points, device='cuda', dtype=torch.bool)
        contributed_cache = torch.empty((0), device='cuda', dtype=torch.bool)
        
    #TODO: Compact this
    means3D_cache = torch.empty((0, 3), device='cuda', dtype=torch.float32)
    opacity_cache = torch.empty((0, 1), device='cuda', dtype=torch.float32)
    scales_cache = torch.empty((0, 3), device='cuda', dtype=torch.float32)
    rotations_cache = torch.empty((0, 4), device='cuda', dtype=torch.float32)
    features_dc_cache = torch.empty((0, 1, 3), device='cuda', dtype=torch.float32)
    features_rest_cache = torch.empty((0, SH_properties_single, 3), device='cuda', dtype=torch.float32)
    distance_mu_cache = torch.empty((0, 1), device='cuda', dtype=torch.float32)
    distance_sigma_cache = torch.empty((0, 1), device='cuda', dtype=torch.float32)
    
    parameters = []
    for values, name, lr in zip([means3D, scales, rotations, features_dc, opacity, features_rest, distance_mu, distance_sigma], 
                                                ["xyz", "scaling", "rotation", "f_dc", "opacity",  "f_rest",  "distance_mu",  "distance_sigma"],
                                                [opt.position_lr_init * gaussians.spatial_lr_scale, opt.scaling_lr, opt.rotation_lr, opt.feature_lr, opt.opacity_lr, opt.feature_lr, opt.opacity_lr, opt.opacity_lr]):
        parameters.append({'params': [values], 'lr': lr * opt.lr_multiplier, "name": name, 
                             "exp_avgs" : torch.zeros_like(values, device='cuda'), "exp_avgs_sqs" : torch.zeros_like(values, device='cuda')})
        
    SPT_parameters = []
    for values, name, lr in zip([gaussians.SPT_means3D, gaussians.SPT_scales, gaussians.SPT_rotations, gaussians.SPT_features_dc, gaussians.SPT_opacity, gaussians.SPT_features_rest, gaussians.SPT_beta], 
                                                ["xyz", "scaling", "rotation", "f_dc", "opacity",  "f_rest"],
                                                [
                                                 opt.position_lr_init * gaussians.spatial_lr_scale,
                                                 opt.scaling_lr, 
                                                 opt.rotation_lr,
                                                 opt.feature_lr,
                                                 opt.opacity_lr,
                                                 opt.feature_lr,
                                                 opt.beta_lr
                                                 ]):
        SPT_parameters.append({'params': [values], 'lr': lr * opt.lr_multiplier, "name": name, 
                             "exp_avgs" : torch.zeros_like(values, device='cuda'), "exp_avgs_sqs" : torch.zeros_like(values, device='cuda')})  
        
        
    prev_SPT_distances = torch.empty(0, dtype = torch.float32, device='cuda')
    prev_SPT_indices = torch.empty(0, dtype = torch.int32, device='cuda')
    prev_SPT_starts = torch.empty(0, dtype = torch.int32, device='cuda')
        
    gaussians.xyz_scheduler_args = get_expon_lr_func(lr_init=opt.position_lr_init*gaussians.spatial_lr_scale,
                                                    lr_final=opt.position_lr_final*gaussians.spatial_lr_scale,
                                                    lr_delay_mult=opt.position_lr_delay_mult,
                                                    max_steps=opt.position_lr_max_steps)
    
    
    SPT_contribution = None
    
    
    if evaluation:
        gaussians.load_smooth_hier("Something")
        #training_generator = DataLoader(scene.getTestCameras(), num_workers = 8, prefetch_factor = 1, persistent_workers = True, collate_fn=direct_collate, shuffle=False)
        opt.iterations = len(training_generator)
        opt.use_GPU_caching = False
        opt.vary_distance_multiplier = False
        opt.optimize_exposure = False
        opt.prune_unused = False
        Write_Tensor_Board = False
        print(f"EVALUATE {opt.iterations} images")
        progress_bar = tqdm(range(0, opt.iterations), desc="Training progress")
    if viewer:
        gaussians.load_smooth_hier("Something")
        opt.iterations = 10000000000000000000
        opt.vary_distance_multiplier = False
        opt.optimize_exposure = False
        network_gui.init("127.0.0.1", 6009)
        while network_gui.conn is None:
            network_gui.try_connect()
            print("Try connect")
        viewer_options = {
        "distance_multiplier" : 1.0, 
        "render_SPTs" : False, 
        "freeze_view" : False, 
        "color_distance" : False, 
        "color_size" : False,
        "reuse_SPT_tolerance" : 0.0,
        "separate_SPTs" : False,
        "highlight_leaves" : False,
        "show_occlusion" : False,
        "use_occlusion_culling" : False,
        "record_traj" : False
        }
        distance_multiplier = viewer_options["distance_multiplier"]
    #lines = 400000
    #colors =  F.normalize(gaussians.properties[gaussians.SPT_gaussian_indices[:lines].cpu(),features1:features2].abs(), p=2, dim=1)
    #image = plot_gaussian_columns(gaussians.properties[gaussians.SPT_gaussian_indices[:lines].cpu(),d_mu1],gaussians.properties[gaussians.SPT_gaussian_indices[:lines].cpu(),d_sigma1], colors, gaussians.SPT_max[:lines],gaussians.SPT_min[:lines],  height=300, y_range=(-100,900)) #gaussians.opacity_activation(gaussians.properties[gaussians.SPT_gaussian_indices[:lines].cpu(),opacity1]),
    #torchvision.utils.save_image(image.permute(2,0,1), "SPT_distance_distribution.png")
    #return
    print("Gaussians Initialized")
    prev_cam_center = torch.zeros(3, device='cuda', dtype=torch.float32)
    print("Current Time:", datetime.now().strftime("%H:%M:%S"))
    

    
    
    while iteration < opt.iterations + 1 and (not viewer or network_gui.conn):
        for viewpoint_batch in training_generator:
            for viewpoint_cam in viewpoint_batch:
                sub_clock()
                if opt.graph_view_select:
                    current_camera_index = int(view_graph_utils.random_walk_node(view_graph, (current_camera_index)))
                    viewpoint_cam = train_camera_data_set[int(current_camera_index)]
                    
                    # reset to new random view every 100 iterations
                    if iteration % 100 == 0:
                        current_camera_index = random.randint(0, len(train_camera_data_set) - 1)
                viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
                viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()
                if iteration >= opt.iterations:
                    if evaluation:
                        print(f"Evaluated {iteration} / {opt.iterations} images")
                        print(f"Average PSNR: {PSNR_total/opt.iterations}, SSIM: {SSIM_total/opt.iterations}, LPIPS: {LPIPS_total/opt.iterations}")
                    else:
                        gaussians.save_smooth_hier()
                    exit()
                
                xyz_lr = gaussians.xyz_scheduler_args(iteration)
                if opt.optimize_exposure:
                    for param_group in gaussians.exposure_optimizer.param_groups:
                        param_group['lr'] = gaussians.exposure_scheduler_args(iteration)

                if viewer:
                    viewpoint_cam, do_training, keep_alive_, scaling_modifer, slider = network_gui.receive()
                    for key, value in slider.items():
                            if key in viewer_options:
                                if type(viewer_options[key]) is bool:
                                    viewer_options[key] = value > 0
                                if type(viewer_options[key]) is float:
                                    viewer_options[key] = value
                    viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                    viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                    viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()
                    viewpoint_cam.projection_matrix = getProjectionMatrix(znear=viewpoint_cam.znear, zfar=viewpoint_cam.zfar, fovX=viewpoint_cam.FoVx, fovY=viewpoint_cam.FoVy, primx = 0, primy=0).transpose(0,1).cuda()
                clock()
                
                
                if not viewer:
                    distance_multiplier = base_focal_length / viewpoint_cam.focal_length
                else:
                    distance_multiplier = viewer_options["distance_multiplier"]
                if iteration % 10 != 0 and opt.vary_distance_multiplier:
                    distance_multiplier = distance_multiplier * (1 + torch.pow(torch.rand(1),4) * 5).cuda()
                    
                ############# SPT Cache
                camera_position = viewpoint_cam.camera_center.cuda()
                
                if Use_Occlusion_Culling:
                    occlusions, occlusion_image = occlusion_cull_slang(gaussians, viewpoint_cam, pipe, background)
                    occlusion_mask = (occlusions > 0).cuda()

                                            

                SPT_indices, indices = torch.sort(all_SPT_indices[occlusion_mask])
                
                
                upper_tree_nodes_to_render = torch.empty(0, dtype=torch.int32, device='cuda')

                
                SPT_upper_tree_indices = upper_SPT_indices[SPT_indices]

                SPT_distances = (gaussians.upper_tree_xyz[SPT_upper_tree_indices] - camera_position).pow(2).sum(1).sqrt() * distance_multiplier
                
                #TODO: Clean up
                ### Band Aid Fix
                if len(SPT_indices) == 0:
                    # Just load whatever is already in memory
                    SPT_indices = torch.zeros(1, dtype=torch.int32, device='cuda')
                    if prev_SPT_indices.size(0) > 0:
                        SPT_indices[0] = prev_SPT_indices[-1]
                    else:
                        # Or just load the first one, whatever
                        SPT_indices[0] = torch.zeros(1, dtype=torch.int32, device='cuda')
                    SPT_distances = torch.zeros(1, dtype=torch.int32, device='cuda')
                    SPT_distances[0] = 100000
                    print("No SPT in image")
                    clock()
                    continue
                ### Band Aid Fix
                
                
                

                SPT_distances = (gaussians.upper_tree_xyz[SPT_upper_tree_indices] - camera_position).pow(2).sum(1).sqrt() * distance_multiplier

                
                prev_to_new_SPT_order = torch.searchsorted(SPT_indices, prev_SPT_indices)
                
                equal_SPT_cache_mask = (prev_to_new_SPT_order < len(SPT_indices)) & (SPT_indices[prev_to_new_SPT_order.clamp_max(len(SPT_indices)-1)] == prev_SPT_indices)
                prev_equal_SPT_cache_indices = torch.nonzero(equal_SPT_cache_mask, as_tuple=True)[0]
                equal_SPT_cache_indices = prev_to_new_SPT_order[equal_SPT_cache_mask]

                prev_distances_compare = prev_SPT_distances[prev_equal_SPT_cache_indices]
                distances_compare = SPT_distances[equal_SPT_cache_indices]
                #close_enough = torch.isclose(distances_compare, prev_distances_compare, rtol=Reuse_SPT_Tolerarance, atol=0.05)
                if opt.use_GPU_caching:
                    close_enough = (prev_distances_compare/distances_compare) > (1.0/  Reuse_SPT_Tolerance_Closer) #0.3
                    close_enough &= (prev_distances_compare/distances_compare) < Reuse_SPT_Tolerance_Farther #0.5
                else:
                    close_enough = torch.zeros(len(prev_distances_compare), dtype=torch.bool, device='cuda')
                print(close_enough.sum().item(), "SPTs reused this frame")
                reuse_SPT_indices = SPT_indices[equal_SPT_cache_indices[close_enough]]


                prev_keep_SPT_cache_indices = prev_equal_SPT_cache_indices[close_enough]

                # Keep all the gaussians that are contained in an STP that is reused and close enough
                # Cumulative Sum Trick
                reuse_gaussians_mask = torch.zeros(len(gaussian_indices)+1, dtype=torch.int32, device='cuda')
                debug_x = prev_SPT_starts[prev_keep_SPT_cache_indices]
                if len(debug_x) > 0:
                    if torch.max(debug_x, 0)[0].item() > len(reuse_gaussians_mask):
                        print("Debug X is bigger than reuse_gaussians_mask")
                    if torch.min(debug_x, 0)[0].item() < 0:
                        print("Debug X is less than 0")
                reuse_gaussians_mask[debug_x] += 1
                debug_y = prev_SPT_starts[prev_keep_SPT_cache_indices+1]
                reuse_gaussians_mask[debug_y] -= 1
                reuse_gaussians_mask = reuse_gaussians_mask.cumsum(0)[:-1].bool()
                
                load_SPT_mask = torch.zeros(len(SPT_indices), device='cuda', dtype=torch.bool)
                load_SPT_mask.scatter_(0, equal_SPT_cache_indices[close_enough].to(torch.int64), True)      
                load_SPT_mask = ~load_SPT_mask                  
                load_SPT_indices = SPT_indices[load_SPT_mask]
                load_SPT_distances = SPT_distances[load_SPT_mask]

                if len(load_SPT_indices) > 0:
                    #LOAD SPT CUT
                    # load_SPT_gaussian_indices contains the indices in gaussians.properties of all Gaussians to load
                    # load_SPT_starts contains the borders between the individual SPTs, starting at 0
                    load_SPT_gaussian_indices, load_SPT_starts = get_spt_cut_cuda(len(load_SPT_indices), gaussians.SPT_gaussian_indices, gaussians.SPT_starts, gaussians.SPT_max, gaussians.SPT_min, load_SPT_indices, load_SPT_distances)
                else:
                    print("No SPTs loaded")
                    load_SPT_gaussian_indices, load_SPT_starts = torch.empty(0, dtype=torch.int32, device='cuda'), torch.empty(0, dtype=torch.int32, device='cuda')
                
                
                
                #TODO: Clean up
                ### BAND AID FIX
                difference = load_SPT_starts[1:] - load_SPT_starts[:-1]
                empty_SPTs = torch.where(difference == 0)[0]
                if len(empty_SPTs) > 0:
                    #print(f"Empty SPTs {empty_SPTs} encountered")
                    mask = torch.ones(len(load_SPT_starts), dtype=torch.bool, device='cuda')
                    mask.scatter_(0, empty_SPTs, False)
                    load_SPT_starts = load_SPT_starts[mask]
                    load_SPT_distances = load_SPT_distances[mask]
                    load_SPT_indices = load_SPT_indices[mask]
                if len(load_SPT_starts) > 0:    
                    if len(load_SPT_gaussian_indices) == load_SPT_starts[-1]:
                        #print("Last SPT empty")
                        load_SPT_starts = load_SPT_starts[:-1]
                        load_SPT_distances = load_SPT_distances[:-1]
                        load_SPT_indices = load_SPT_indices[:-1]
                ### BAND AID FIX
                
                assert(len(load_SPT_starts.unique()) == len(load_SPT_starts))
                cache_SPT_cache_indices = torch.where(~equal_SPT_cache_mask)[0]    
                cache_SPT_indices = prev_SPT_indices[cache_SPT_cache_indices]
                SPT_indices = torch.cat((load_SPT_indices, reuse_SPT_indices, cache_SPT_indices))
                

                SPT_starts_new = torch.zeros(len(load_SPT_indices) + len(reuse_SPT_indices) + len(cache_SPT_indices) + 1,dtype=torch.int32, device='cuda')
                
                # compact the prefix sum of SPT_counts
                SPT_starts_new[:len(load_SPT_starts)] = load_SPT_starts + gaussians.skybox_points
                SPT_starts_new[len(load_SPT_starts)] = len(load_SPT_gaussian_indices) + gaussians.skybox_points
                
                sizes = prev_SPT_starts[prev_keep_SPT_cache_indices + 1] - prev_SPT_starts[prev_keep_SPT_cache_indices]
                SPT_starts_new[len(load_SPT_starts) + 1:len(load_SPT_starts) + 1 + len(sizes)] = torch.cumsum(sizes, dim=0) +  len(load_SPT_gaussian_indices) + gaussians.skybox_points

                number_of_gaussians_to_render = SPT_starts_new[len(load_SPT_starts) + len(sizes)]
                
                if len(cache_SPT_indices) > 0:
                    sizes = prev_SPT_starts[cache_SPT_cache_indices + 1] - prev_SPT_starts[cache_SPT_cache_indices]
                    SPT_starts_new[-len(sizes):] = torch.cumsum(sizes, dim=0) + number_of_gaussians_to_render
                else:
                    print("No cached SPTs")
                
                assert(len(SPT_starts_new.unique()) == len(SPT_starts_new))
                SPT_distances = torch.cat((load_SPT_distances, prev_SPT_distances[prev_keep_SPT_cache_indices], prev_SPT_distances[cache_SPT_cache_indices]))

                load_from_disk_indices = torch.cat((upper_tree_nodes_to_render, load_SPT_gaussian_indices))    
                    
                # Cumulative Sum Trick
                cache_gaussians_mask = torch.zeros(len(gaussian_indices)+1, dtype=torch.int32, device='cuda')
                debug_x = prev_SPT_starts[cache_SPT_cache_indices]
                if len(debug_x) > 0:
                    if torch.max(debug_x, 0)[0].item() > len(reuse_gaussians_mask):
                        print("Debug X is bigger than reuse_gaussians_mask")
                    if torch.min(debug_x, 0)[0].item() < 0:
                        print("Debug X is less than 0")
                cache_gaussians_mask[debug_x] += 1
                debug_y = prev_SPT_starts[cache_SPT_cache_indices+1]
                cache_gaussians_mask[debug_y] -= 1
                cache_gaussians_mask = cache_gaussians_mask.cumsum(0)[:-1].bool()
                
                ##############################
                write_back_SPT_indices = prev_SPT_indices[prev_equal_SPT_cache_indices[~close_enough]]
                write_back_SPT_cache_indices = prev_equal_SPT_cache_indices[~close_enough]
                ##############################
                # Write back from Cache to CPU Memory
                if len(gaussian_indices) > opt.cache_size or iteration % opt.clear_cache_interval == 0:
                    #  The cache is regularly cleared
                    if iteration % opt.clear_cache_interval == 0:
                        print("Clear Cache")
                        to_reduce = len(gaussian_indices)
                    else:
                        print("Reduce Cache")
                        to_reduce = len(gaussian_indices) - opt.cache_size_after_reduction
                    
                    # write back the last few SPTs that have the highest age
                    reduced = 0
                    number_SPTs_to_write_back = 0
                    SPTs_to_write_back = []
                    while(reduced < to_reduce and number_SPTs_to_write_back < len(cache_SPT_indices)):
                        number_SPTs_to_write_back += 1
                        SPTs_to_write_back.append(cache_SPT_indices[-number_SPTs_to_write_back])
                        reduced += SPT_starts_new[-number_SPTs_to_write_back] - SPT_starts_new[-(number_SPTs_to_write_back+1)]
                    if number_SPTs_to_write_back >= len(SPT_indices)-1:
                        print("Not enough SPTs to write back")
                        number_SPTs_to_write_back -= 1
                    SPTs_to_write_back = torch.tensor(SPTs_to_write_back, device='cuda')
                    for index in SPTs_to_write_back:
                        prev_index = (prev_SPT_indices == index).nonzero(as_tuple=True)[0].item()
                        cache_gaussians_mask[prev_SPT_starts[prev_index] : prev_SPT_starts[prev_index+1]] = False
                    if number_SPTs_to_write_back > 0:
                        write_back_SPT_indices = torch.cat((write_back_SPT_indices, torch.tensor(SPTs_to_write_back, device='cuda')))
                        SPT_starts_new = SPT_starts_new[:-number_SPTs_to_write_back]
                        SPT_distances = SPT_distances[:-number_SPTs_to_write_back]
                        SPT_indices = SPT_indices[:-number_SPTs_to_write_back]
 
                write_back_mask = ~torch.logical_or(reuse_gaussians_mask, cache_gaussians_mask)
                #dont write back the skybox
                write_back_mask[:gaussians.skybox_points] = False
                write_back_indices = gaussian_indices[write_back_mask].detach().to(opt.storage_device)


                prev_gaussian_indices =  gaussian_indices.clone()
                
                
                gaussian_indices = torch.cat((gaussian_indices[:gaussians.skybox_points], load_from_disk_indices, gaussian_indices[reuse_gaussians_mask], gaussian_indices[cache_gaussians_mask]))
                #print(f"Load Percent: {len(load_from_disk_indices) * 100/ number_of_gaussians_to_render}")
                load_from_disk_indices = load_from_disk_indices.to(opt.storage_device)
                SPT_starts_new += len(upper_tree_nodes_to_render)
                
                assert(SPT_starts_new[-1] == len(gaussian_indices))                    
                    
                __hierarchy_cut_time = clock()
                clock()


                # TODO: Compact this
                means3D_full = torch.cat((means3D, means3D_cache)).detach()
                opacity_full = torch.cat((opacity, opacity_cache)).detach()
                scales_full = torch.cat((scales, scales_cache)).detach()
                rotations_full = torch.cat((rotations, rotations_cache)).detach()
                features_dc_full = torch.cat((features_dc, features_dc_cache)).detach()
                features_rest_full = torch.cat((features_rest, features_rest_cache)).detach()
                distance_mu_full = torch.cat((distance_mu, distance_mu_cache)).detach()
                distance_sigma_full = torch.cat((distance_sigma, distance_sigma_cache)).detach()
                
                write_back_tensors = [means3D_full[write_back_mask],  scales_full[write_back_mask], rotations_full[write_back_mask], features_dc_full[write_back_mask].squeeze(1), opacity_full[write_back_mask], features_rest_full[write_back_mask].reshape(len(write_back_indices), SH_properties), distance_mu_full[write_back_mask], distance_sigma_full[write_back_mask]]
                for index in range(8):
                    if index == 5:
                        write_back_tensors.append(parameters[index]["exp_avgs"][write_back_mask].reshape(len(write_back_indices), SH_properties))
                    elif index == 3:
                        write_back_tensors.append(parameters[index]["exp_avgs"][write_back_mask].squeeze(1))
                    else:
                        write_back_tensors.append(parameters[index]["exp_avgs"][write_back_mask])
                for index in range(8):
                    if index == 5:
                        write_back_tensors.append(parameters[index]["exp_avgs_sqs"][write_back_mask].reshape(len(write_back_indices), SH_properties))
                    elif index ==3:
                        write_back_tensors.append(parameters[index]["exp_avgs_sqs"][write_back_mask].squeeze(1))
                    else:
                        write_back_tensors.append(parameters[index]["exp_avgs_sqs"][write_back_mask])
                
                gaussians.properties[write_back_indices, :] = torch.cat((write_back_tensors), dim=1).cpu() 
                
                #Write back to SPT_max, SPT_min
                #TODO: accelerate
                if len(write_back_indices) > 0 and not evaluation:
                    current_idx = 0
                    for write_back_SPT, write_back_cache_SPT in zip(write_back_SPT_indices, write_back_SPT_cache_indices):
                        
                        SPT_start = gaussians.SPT_starts[write_back_SPT].item()
                        SPT_end = gaussians.SPT_starts[write_back_SPT +1].item()
                        SPT_len = prev_SPT_starts[write_back_cache_SPT+1] - prev_SPT_starts[write_back_cache_SPT]
                        mask = torch.isin(gaussians.SPT_gaussian_indices[SPT_start : SPT_end].cuda(), prev_gaussian_indices[prev_SPT_starts[write_back_cache_SPT]:prev_SPT_starts[write_back_cache_SPT+1]].cuda())
                        indices = torch.nonzero(mask).squeeze()
                        
                        gaussians.SPT_max[SPT_start + indices] = (gaussians.properties[write_back_indices[current_idx : current_idx + SPT_len], d_mu1] + 3 * gaussians.properties[write_back_indices[current_idx : current_idx + SPT_len], d_sigma1]).cuda()
                        gaussians.SPT_min[SPT_start + indices] = (gaussians.properties[write_back_indices[current_idx : current_idx + SPT_len], d_mu1] - 3 * gaussians.properties[write_back_indices[current_idx : current_idx + SPT_len], d_sigma1]).cuda()
                        current_idx += SPT_len

                full_mask = torch.cat((torch.where(reuse_gaussians_mask)[0], torch.where(cache_gaussians_mask)[0]))
                
                if opt.densification == "classic":
                    densification_criterium_full = torch.cat((densification_criterium, densification_criterium_cache)).detach()
                    gaussians._densification_criterium[write_back_indices] = densification_criterium_full[write_back_mask].to(opt.storage_device, non_blocking=non_blocking)
                    densification_criterium = torch.cat((densification_criterium[:gaussians.skybox_points], gaussians._densification_criterium[load_from_disk_indices].cuda(non_blocking=non_blocking), densification_criterium_full[reuse_gaussians_mask])).detach()
                    densification_criterium_cache = densification_criterium_full[cache_gaussians_mask]
                if opt.prune_unused and not evaluation:
                    contributed_full = torch.cat((contributed, contributed_cache)).detach()
                    gaussians._contributed[write_back_indices] = contributed_full[write_back_mask].to(opt.storage_device, non_blocking=non_blocking)
                    contributed = torch.cat((contributed[:gaussians.skybox_points], gaussians._contributed[load_from_disk_indices].cuda(non_blocking=non_blocking), contributed_full[reuse_gaussians_mask])).detach()
                    contributed_cache = contributed_full[cache_gaussians_mask]
                torch.cuda.empty_cache()

                load_tensor = gaussians.properties[load_from_disk_indices, :].cuda(non_blocking=non_blocking)
                
                
                means3D = nn.Parameter(torch.cat((means3D[:gaussians.skybox_points], load_tensor[:, xyz1:xyz2].cuda(non_blocking=non_blocking), means3D_full[reuse_gaussians_mask])).contiguous())
                scales = nn.Parameter(torch.cat((scales[:gaussians.skybox_points], load_tensor[:, scales1:scales2].cuda(non_blocking=non_blocking), scales_full[reuse_gaussians_mask])).contiguous())
                rotations = nn.Parameter(torch.cat((rotations[:gaussians.skybox_points], load_tensor[:, rotation1:rotation2].cuda(non_blocking=non_blocking), rotations_full[reuse_gaussians_mask])).contiguous())
                features_dc = nn.Parameter(torch.cat((features_dc[:gaussians.skybox_points], load_tensor[:, features1:features2].cuda(non_blocking=non_blocking).unsqueeze(1), features_dc_full[reuse_gaussians_mask])).contiguous())
                opacity = nn.Parameter(torch.cat((opacity[:gaussians.skybox_points], load_tensor[:, opacity1].cuda(non_blocking=non_blocking).unsqueeze(1), opacity_full[reuse_gaussians_mask])).contiguous())

                features_rest = nn.Parameter(torch.cat((features_rest[:gaussians.skybox_points], load_tensor[:, features_rest1:features_rest2].cuda(non_blocking=non_blocking).reshape(len(load_tensor), SH_properties_single, 3 ), features_rest_full[reuse_gaussians_mask])).contiguous())
                distance_mu = nn.Parameter(torch.cat((distance_mu[:gaussians.skybox_points], load_tensor[:, d_mu1].cuda(non_blocking=non_blocking).unsqueeze(1), distance_mu_full[reuse_gaussians_mask])).contiguous())
                distance_sigma = nn.Parameter(torch.cat((distance_sigma[:gaussians.skybox_points], load_tensor[:, d_sigma1].cuda(non_blocking=non_blocking).unsqueeze(1), distance_sigma_full[reuse_gaussians_mask])).contiguous())


                
                
                means3D_cache = means3D_full[cache_gaussians_mask]
                scales_cache = scales_full[cache_gaussians_mask]
                rotations_cache = rotations_full[cache_gaussians_mask]
                features_dc_cache = features_dc_full[cache_gaussians_mask]
                opacity_cache = opacity_full[cache_gaussians_mask]
                features_rest_cache = features_rest_full[cache_gaussians_mask]
                distance_mu_cache = distance_mu_full[cache_gaussians_mask]
                distance_sigma_cache = distance_sigma_full[cache_gaussians_mask]
                

                parameters_new = []
                for index, (values, name, lr) in enumerate(zip([means3D,  scales, rotations, features_dc, opacity, features_rest, distance_mu, distance_sigma], 
                                        ["xyz", "scaling", "rotation", "f_dc", "opacity", "f_rest", "distance_mu", "distance_sigma"],
                                        [xyz_lr, opt.scaling_lr, opt.rotation_lr, opt.feature_lr, opt.opacity_lr, opt.feature_lr, opt.opacity_lr, opt.opacity_lr])):
                    if index == 5:
                        exp_avgs = torch.cat((parameters[index]["exp_avgs"][:gaussians.skybox_points], load_tensor[:, range1[index] + number_properties:range2[index] + number_properties].cuda().reshape(len(load_tensor), SH_properties_single, 3), parameters[index]["exp_avgs"][full_mask])).contiguous()
                        exp_avgs_sqs = torch.cat((parameters[index]["exp_avgs_sqs"][:gaussians.skybox_points], load_tensor[:, range1[index] + 2*number_properties:range2[index] + 2*number_properties].cuda().reshape(len(load_tensor), SH_properties_single, 3), parameters[index]["exp_avgs_sqs"][full_mask])).contiguous()
                    elif index == 3:                            
                        exp_avgs = torch.cat((parameters[index]["exp_avgs"][:gaussians.skybox_points], load_tensor[:, range1[index] + number_properties:range2[index] + number_properties].cuda().unsqueeze(1), parameters[index]["exp_avgs"][full_mask])).contiguous()
                        exp_avgs_sqs = torch.cat((parameters[index]["exp_avgs_sqs"][:gaussians.skybox_points], load_tensor[:, range1[index] + 2*number_properties:range2[index] + 2*number_properties].cuda().unsqueeze(1), parameters[index]["exp_avgs_sqs"][full_mask])).contiguous()
                    else:
                        exp_avgs = torch.cat((parameters[index]["exp_avgs"][:gaussians.skybox_points], load_tensor[:, range1[index] + number_properties:range2[index] + number_properties].cuda(), parameters[index]["exp_avgs"][full_mask])).contiguous()
                        exp_avgs_sqs = torch.cat((parameters[index]["exp_avgs_sqs"][:gaussians.skybox_points], load_tensor[:, range1[index] + 2*number_properties:range2[index] + 2*number_properties].cuda(), parameters[index]["exp_avgs_sqs"][full_mask])).contiguous()
                    
                    
                    parameters_new.append({'params': [values], 'lr': lr*opt.lr_multiplier, "name": name, 
                     "exp_avgs" : exp_avgs, "exp_avgs_sqs" : exp_avgs_sqs})
                parameters = parameters_new
                
                prev_SPT_indices = SPT_indices
                prev_SPT_distances = SPT_distances
                prev_SPT_starts = SPT_starts_new
                
                load_write_time = clock()
                del load_tensor
                del write_back_mask
                del reuse_gaussians_mask
                del write_back_indices
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                pre_render_peak = torch.cuda.max_memory_allocated(device='cuda')
                torch.cuda.reset_peak_memory_stats()
                # Render
                
                if iteration % int(math.floor(opt.iterations * opt.SH_increase_after_train_percent)) == 0 and iteration > 0:
                    gaussians.oneupSHdegree()
                
                gaussian_SPT_indices = torch.zeros(len(means3D), dtype=torch.int32, device='cuda')
                gaussian_SPT_indices[SPT_starts_new[:-len(cache_SPT_indices)-1]] = 1 
                gaussian_SPT_indices = torch.cumsum(gaussian_SPT_indices, dim=0) - 1
                distances = SPT_distances[gaussian_SPT_indices]
                distances[:SPT_starts_new[0]] = 0.0

                render_pkg = gaussian_renderer.render(
                            viewpoint_cam, 
                            means3D,
                            gaussians.opacity_activation(opacity),
                            gaussians.scaling_activation(scales), 
                            gaussians.rotation_activation(rotations),
                            features_dc,
                            features_rest,
                            distance_mu,
                            distance_sigma,
                            distances,
                            pipe, 
                            background,
                            sh_degree = gaussians.active_sh_degree,
                            #anti_aliasing=dataset.anti_aliasing,
                            #use_trained_exp = opt.optimize_exposure,
                            #gaussians=gaussians
                            )
                if viewer:
                    net_image = render_pkg["render"].cpu()
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().to('cpu').numpy())
                    train_params = {"Num_Rendered" : len(gaussian_indices), "Number_of_SPTs" : len(SPT_indices), "Percentage_Rendered" : len(gaussian_indices)/gaussians.size, "Percentage_SPTs" : len(SPT_indices)/len(gaussians.SPT_starts)}
                    network_gui.send(net_image_bytes, json.dumps({"iteration" : 99, "num_gaussians" : gaussians.size, "loss" : 0, "sh_degree":1, "error" : 0, "paused" : False, "train_params" : train_params})) #dataset.source_path)
                    continue
                contribution = render_pkg["contribution"]
                __post_render_peak = torch.cuda.max_memory_allocated(device='cuda')
                torch.cuda.reset_peak_memory_stats()
                
                
                image = render_pkg["render"]#, render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                
                gt_image = viewpoint_cam.original_image.cuda()
                if evaluation:
                    #with torch.no_grad():
                    #    if viewpoint_cam.alpha_mask.cuda() is not None:
                    #        image *= viewpoint_cam.alpha_mask.cuda() 
                    #        gt_image *= viewpoint_cam.alpha_mask.cuda()
                    psnr_current = psnr(image.detach() * viewpoint_cam.alpha_mask.cuda() , gt_image).mean().double().item()
                    ssim_current = ssim(image.detach() * viewpoint_cam.alpha_mask.cuda() , gt_image).mean().double().item()
                    lpips_current = lpips(image.detach() * viewpoint_cam.alpha_mask.cuda() , gt_image, net_type='vgg').mean().double().item()
                    print(f"{psnr_current}, {ssim_current}, {lpips_current}")
                    PSNR_total += psnr_current  
                    SSIM_total += ssim_current
                    LPIPS_total += lpips_current
                
                
                # Loss
                
                if viewpoint_cam.alpha_mask is not None:
                    #print(f"Alpha mask: {viewpoint_cam.alpha_mask.sum()} / {viewpoint_cam.alpha_mask.nelement()}")
                    Ll1 = l1_loss(image * viewpoint_cam.alpha_mask.cuda(), gt_image)
                    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - fused_ssim((image * viewpoint_cam.alpha_mask.cuda()).unsqueeze(0), gt_image.unsqueeze(0)))
                else:
                    Ll1 = l1_loss(image, gt_image) 
                
                
                
                
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
                image_loss = loss.clone().detach()
                if viewpoint_cam.invdepthmap is not None and False:
                    invDepth = render_pkg["depth"]
                    mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                    Ll1depth_pure = torch.abs((invDepth  - mono_invdepth)).mean()
                    Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
                    #print(f"depth loss: {Ll1depth}")
                    loss += Ll1depth
                    Ll1depth = Ll1depth.item()
                else:
                    Ll1depth = 0
                if iteration % 50 == 0 or iteration == 1:
                    torchvision.utils.save_image(image, os.path.join(scene.model_path, str(iteration) + ".png"))
                    
                if evaluation:
                    torchvision.utils.save_image(image, os.path.join("output", str(iteration) + ".png"))
                    torchvision.utils.save_image(gt_image, os.path.join("output", str(iteration) + "_gt.png"))
                
                iteration_time = sub_clock()
                #hierarchy_loss = 0 #torch.sum(torch.clamp_min(torch.max(torch.abs(gaussians.get_scaling[indices]), dim=-1)[0] - torch.max(torch.abs(gaussians.get_scaling[parents]), dim=-1)[0], 0)) / len(indices)
                if opt.densification == "MCMC":
                    contributing_gaussians = torch.where(contribution > 0.0001)[0]
                    contributing_gaussians = contributing_gaussians[gaussians.nodes[gaussian_indices[contributing_gaussians].cpu(), hierarchy_node_child_count] == 0].cuda()
                
                    number_of_contributing_gaussians = contributing_gaussians.sum().item()
                    opacity_loss = torch.sum((gaussians.opacity_activation(opacity[contributing_gaussians]))) / number_of_contributing_gaussians
                    scaling_loss = torch.sum((gaussians.scaling_activation(scales[contributing_gaussians])))  / number_of_contributing_gaussians
                    
                if opt.prune_unused and not evaluation:
                    contributed = torch.logical_or(contributed, contribution > 0.0001)
                    
                if opt.lambda_opacity > 0 and opt.densification == "MCMC":
                    loss = loss + opt.lambda_opacity * opacity_loss
                if opt.lambda_scaling > 0 and opt.densification == "MCMC":
                    loss = loss + opt.lambda_scaling * scaling_loss
                    
                if math.isnan(loss):
                            torchvision.utils.save_image(image, os.path.join(scene.model_path, "Error" + ".png"))
                            print("gradients collapsed :(")
                            continue
                     
                loss.backward()
                if math.isnan(loss):
                        torchvision.utils.save_image(image, os.path.join(scene.model_path, "Error" + ".png"))
                        print("gradients collapsed :(")
                        continue
                __post_backward_peak = torch.cuda.max_memory_allocated(device='cuda')
                
                # occlusion loss
                interval_sums = SPT_starts_new.clone()
                interval_sums = torch.tensor([render_pkg["contribution"][start:end].sum() for start, end in zip(SPT_starts_new[:-1], SPT_starts_new[1:])]).cuda()
                SPT_contribution = torch.zeros(len(gaussians.SPT_starts)-1, dtype=interval_sums.dtype).cuda()
                SPT_contribution.index_add_(0, SPT_indices, interval_sums)
                SPT_contribution = SPT_contribution[all_SPT_indices]
                
                lambda_occlusion = 0.000000001
                occlusion_loss = lambda_occlusion * (occlusions - SPT_contribution.detach()).abs().sum()
                occlusion_loss.backward()
                
                #This needs to happen after backward
                if opt.densification == "classic":
                    densification_criterium = torch.max(torch.norm(render_pkg["viewspace_points"].grad, dim=-1), densification_criterium)
                
                # Write values for every iteration
                #region Tensorboard
                if Write_Tensor_Board:
                    writer.add_scalar('Total Loss', image_loss, iteration)
                    writer.add_scalar('Distance_To_Last_view', torch.linalg.norm(viewpoint_cam.camera_center - prev_cam_center), iteration)
                prev_cam_center = viewpoint_cam.camera_center
                if Write_Tensor_Board and iteration % 10 == 0:
                    writer.add_scalar('VRAM usage', torch.cuda.memory_allocated(0), iteration)
                    process = psutil.Process(pid)
                    mem_info = process.memory_info()
                    writer.add_scalar('CPU RAM usage', mem_info.rss / 1024 ** 2, iteration)
                    writer.add_scalar('Peak VRAM usage', torch.cuda.max_memory_allocated(device='cuda'), iteration)
                    
                    #writer.add_scalar('Opacity Loss', opacity_loss, iteration)
                    #writer.add_scalar('Scaling Loss', scaling_loss, iteration)
                    writer.add_scalar('Mean Opacity', torch.mean(gaussians.opacity_activation(opacity)), iteration)
                    writer.add_scalar('Mean Scaling', torch.mean((gaussians.scaling_activation(scales))), iteration)
                    writer.add_scalar('Number of Gaussians loaded', len(load_from_disk_indices), iteration)
                    writer.add_scalar('Percentage of Gaussians loaded', len(load_from_disk_indices)*100 /len(gaussian_indices), iteration)
                    writer.add_scalar('Number of Gaussians rendered', len(gaussian_indices), iteration)
                    writer.add_scalar('Hierarchy Cut Time', __hierarchy_cut_time, iteration)
                    writer.add_scalar('Memory Load / Write Time', load_write_time, iteration)
                    writer.add_scalar('Rendered SPTs', len(SPT_indices), iteration)
                    writer.add_scalar('Rendered SPTs Percentage', len(SPT_indices)/len(gaussians.SPT_starts), iteration)
                    total_SPT_nodes = gaussians.SPT_starts[SPT_indices+1] - gaussians.SPT_starts[SPT_indices]
                    writer.add_scalar('Rendered SPTs Detail', (len(gaussian_indices)-gaussians.skybox_points)/total_SPT_nodes.sum(), iteration)
                    
                    writer.add_scalar('Peak before Render', pre_render_peak, iteration)
                    writer.add_scalar('Peak during Render', __post_render_peak, iteration)
                    writer.add_scalar('Peak during backwards', __post_backward_peak, iteration)
                #endregion

                with torch.no_grad():
                    # Progress bar
                    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                    if torch.cuda.max_memory_allocated(device='cuda') > __prev_peak_memory:
                        __prev_peak_memory = torch.cuda.max_memory_allocated(device='cuda')
                        print(f"New peak memory reached {len(gaussian_indices):_}, before: {__prev_number_rendered:_}")
                    __prev_number_rendered = len(gaussian_indices)
                    if iteration % 10 == 0:
                        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Size": f"{gaussians.size:_}/{gaussians.properties.size(0):_}", "Peak memory": f"{__prev_peak_memory:_}"})
                        progress_bar.update(10)
                    torch.cuda.reset_peak_memory_stats()
                    # Log and save
                    if (iteration in saving_iterations):
                        gaussians.save_hier(file_name=f"iteration_{iteration}")
                        print("\n[ITER {}] Saving Gaussians".format(iteration))
                        print("peak memory: ", torch.cuda.max_memory_allocated(device='cuda'))

                   


                    #region Densification

                        
                    if opt.densify_from_iter < iteration < opt.densify_until_iter and iteration % opt.densification_interval == 0:
                        print("-----------------DENSIFY!--------------------")
                        indices = torch.where(torch.isnan(opacity))
                        with torch.no_grad():
                            opacity[indices] = 0.1
                        
                        write_back_tensors = [means3D,  scales, rotations, features_dc.squeeze(1), opacity, features_rest.reshape(len(features_rest), SH_properties), distance_mu, distance_sigma]
                        caches = [means3D_cache, scales_cache, rotations_cache, features_dc_cache.squeeze(1), opacity_cache, features_rest_cache.reshape(len(features_rest_cache), SH_properties), distance_mu_cache, distance_sigma_cache]
                        
                        write_back_tensors_densify = [torch.cat((tensor, cache)) for tensor, cache in zip(write_back_tensors, caches)]
                        #TODO: Refactor this
                        #for ADAM_Parameter in ["exp_avgs", "exp_avgs_sqs"]:
                        for index in range(len(write_back_tensors)):
                            if index == 5:
                                write_back_tensors_densify.append(parameters[index]["exp_avgs"].reshape(len(gaussian_indices), SH_properties))
                            elif index ==3:
                                write_back_tensors_densify.append(parameters[index]["exp_avgs"].squeeze(1))
                            else:
                                write_back_tensors_densify.append(parameters[index]["exp_avgs"])
                        for index in range(len(write_back_tensors)):
                            if index == 5:
                                write_back_tensors_densify.append(parameters[index]["exp_avgs_sqs"].reshape(len(gaussian_indices), SH_properties))
                            elif index ==3:
                                write_back_tensors_densify.append(parameters[index]["exp_avgs_sqs"].squeeze(1))
                            else:
                                write_back_tensors_densify.append(parameters[index]["exp_avgs_sqs"])
                        
                        gaussians.properties[gaussian_indices, :] = torch.cat((write_back_tensors_densify), dim=1).cpu() 
                        del write_back_tensors
                        del write_back_tensors_densify
                        
                        if opt.densification == "classic":
                                # Don't write back the densification criterium, it is reset anyway
                                #gaussians._densification_criterium[gaussian_indices] = torch.cat((densification_criterium, densification_criterium_cache)).to(opt.storage_device)  
                                densification_criterium_cache = torch.empty((0), device='cuda', dtype=torch.float32)
                                densification_criterium = torch.zeros(gaussians.skybox_points, device='cuda', dtype=torch.float32)
                        if opt.prune_unused and not evaluation:
                            # Don't write back the contributed, it is reset anyway
                            contributed_cache = torch.empty((0), device='cuda', dtype=torch.bool)
                            contributed = torch.zeros(gaussians.skybox_points, device='cuda', dtype=torch.bool)
                        prev_SPT_indices, prev_SPT_distances, prev_SPT_starts = torch.empty(0, device='cuda', dtype=torch.int32), torch.empty(0, device='cuda', dtype=torch.float32), torch.empty(0, device='cuda', dtype=torch.int32)
                        if opt.use_GPU_caching:
                            temp = gaussians.properties[:gaussians.skybox_points, :number_properties].cuda()
                            
                            gaussian_indices = torch.arange(0, gaussians.skybox_points, device='cuda')
                            means3D = temp[:, xyz1:xyz2].cuda().detach().contiguous()
                            opacity = temp[:, opacity1].cuda().detach().unsqueeze(1).contiguous()
                            scales = temp[:, scales1:scales2].cuda().detach().contiguous()
                            rotations = temp[:, rotation1:rotation2].cuda().detach().contiguous()
                            features_dc = temp[:, features1:features2].cuda().detach().unsqueeze(1).contiguous()
                            features_rest = temp[:, features_rest1:features_rest2].cuda().detach().reshape(len(temp), SH_properties_single, 3).contiguous()
                            distance_mu = temp[:, d_mu1:d_mu2].cuda().detach().contiguous()
                            distance_sigma = temp[:, d_sigma1:d_sigma2].cuda().detach().contiguous()
                            
                            means3D_cache = torch.empty((0, 3), device='cuda', dtype=torch.float32)
                            opacity_cache = torch.empty((0, 1), device='cuda', dtype=torch.float32)
                            scales_cache = torch.empty((0, 3), device='cuda', dtype=torch.float32)
                            rotations_cache = torch.empty((0, 4), device='cuda', dtype=torch.float32)
                            features_dc_cache = torch.empty((0, 1, 3), device='cuda', dtype=torch.float32)
                            features_rest_cache = torch.empty((0, SH_properties_single, 3), device='cuda', dtype=torch.float32)
                            distance_mu_cache = torch.empty((0, 1), device='cuda', dtype=torch.float32)
                            distance_sigma_cache = torch.empty((0, 1), device='cuda', dtype=torch.float32)
                            del temp
                            
                        parameters = []
                        for values, name, lr in zip([means3D,  scales, rotations, features_dc, opacity, features_rest, distance_mu, distance_sigma], 
                                                ["xyz", "scaling", "rotation", "f_dc", "opacity",  "f_rest",  "d_mu",  "d_sigma"],
                                                [xyz_lr, opt.scaling_lr, opt.rotation_lr, opt.feature_lr, opt.opacity_lr,  opt.feature_lr, opt.opacity_lr, opt.opacity_lr]):
                            parameters.append({'params': [values], 'lr': lr*opt.lr_multiplier, "name": name, 
                             "exp_avgs" : torch.zeros_like(values), "exp_avgs_sqs" : torch.zeros_like(values)})
                        
                        
                        #On the first densification iteration after 2* #images iterations, prune all leaf gaussians that were never seen
                        if opt.prune_unused and iteration % (2*len(training_generator)) < opt.densification_interval:
                            prune_mask = torch.logical_and(~gaussians._contributed[:gaussians.size], gaussians.nodes[:gaussians.size, hierarchy_node_child_count] == 0)
                            print(f"Pruning {prune_mask.sum()} unused Gaussians")
                            gaussians.properties[torch.where(prune_mask)[0], opacity1] = -99
                            gaussians._contributed[:] = False
                        
                        
                        dead_indices = torch.where((gaussians.properties[:gaussians.size, opacity1] <= gaussians.inverse_opacity_activation(torch.tensor(0.005)).item()).squeeze(-1))[0]

                        # Find SPT root nodes
                        gaussians.add_new_gs_smooth(cap_max=opt.cap_max, size=gaussians.size, densification=opt.densification, densify_percent=opt.densify_percent, densify_threshold=opt.densify_grad_threshold)
                        # Make sure that we don't mark newly spawned Gaussians as dead
                        dead_mask =torch.zeros(gaussians.size, dtype=torch.bool, device=opt.storage_device)
                        dead_mask[dead_indices] = True
                        
                        # only redistribute leaf nodes
                        print(f"Respawn {torch.sum(dead_mask)} Gaussians")
                        gaussians.relocate_gs_smooth(dead_mask, gaussians.size, storage_device=opt.storage_device, densification=opt.densification)
                           
                        
                        if opt.densification == "classic":
                            gaussians._densification_criterium[:] = 0
                        
                        
                        
                        #print(f"Max Train Image: {torch.max(train_image_counts)}, Min Train Image: {torch.min(train_image_counts)}")
                        torch.cuda.empty_cache()


                        # Per-Densification Statistics
                        #region Tensorboard
                        if Write_Tensor_Board:
                            writer.add_scalar('Number of Hierarchy Levels', gaussians.get_number_of_levels(), iteration)
                            writer.add_scalar('Lowest leaf node level', torch.min(gaussians.nodes[gaussians.nodes[:, 3] <= 0, 0][gaussians.skybox_points:]).item(), iteration)
                            writer.add_scalar('Number of Gaussians', gaussians.size, iteration)
                            writer.add_scalar('Number of SPTs', len(gaussians.SPT_starts), iteration)
                            writer.add_scalar('Mean Number of Gaussians per SPT', torch.mean((gaussians.SPT_starts[1:] - gaussians.SPT_starts[:-1]).float()), iteration)
                            writer.add_scalar('Number of Dead Gaussians in SPTs', (gaussians.SPT_min==gaussians.SPT_max).sum(), iteration)
                            writer.add_scalar('Number of Respawns due to MIP Filter', len(dead_indices), iteration)
                            writer.add_scalar('Number of Respawns', torch.sum(dead_mask), iteration)
                            writer.add_scalar('Proportions of Gaussians in SPT', len(gaussians.SPT_min) / gaussians.size, iteration)
                            if opt.use_bounding_spheres:
                                mean_bounding_sphere_radius = gaussians.bounding_sphere_radii.mean()
                                mean_covariance_max_scale = torch.max(gaussians.scaling_activation(gaussians._scaling[gaussians.upper_tree_nodes[:, 5].to(Storage_Device)]), dim=-1)[0].mean()
                                writer.add_scalar('Mean Difference between Bounding Radius and 3Sigma', mean_bounding_sphere_radius - mean_covariance_max_scale, iteration)
                        #endregion
                        
                    #region Optimization
                    elif iteration < opt.iterations and not evaluation:
                        if opt.optimize_exposure:
                            #print("optimize exposure")
                            gaussians.exposure_optimizer.step()
                            gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                        
                        #zero gradients of Skybox
                        means3D.grad[0:gaussians.skybox_points, :] = 0
                        rotations.grad[0:gaussians.skybox_points, :] = 0
                        features_dc.grad[0:gaussians.skybox_points, :, :] = 0
                        features_rest.grad[0:gaussians.skybox_points, :, :] = 0
                        opacity.grad[0:gaussians.skybox_points, :] = 0
                        scales.grad[0:gaussians.skybox_points, :] = 0

    
                        
                        #if torch.isnan(means3D.grad).any() or (torch.isnan(opacity.grad)).any() or torch.isnan(scales.grad).any():
                        #    print("Gradients Collapsed :(")
                        #    indices = torch.where(torch.isnan(means3D.grad) | torch.isnan(opacity.grad) | torch.isnan(scales.grad))[0].unique()
                        #    means3D.grad[indices] = 0
                        #    opacity.grad[indices] = 0
                        #    scales.grad[indices] = 0
                        #    rotations.grad[indices] = 0
                        #    pass
                        #relevant = (opacity.grad.flatten() != 0).nonzero()
                        
                        if opt.dampen_scale_grad:
                            
                            view_dir = torch.tensor(viewpoint_cam.R, device='cuda', dtype=torch.float32) @ torch.tensor([0, 0, 1], device='cuda', dtype=torch.float32)
                            R = qvec2rotmat_torch(rotations)
                            S = torch.diag_embed(gaussians.scaling_activation(scales))
                            covariances = R @ S @ S.transpose(1, 2) @ R.transpose(1, 2)
                            scale_along_view_dir = torch.sqrt(view_dir.T @ covariances @ view_dir)
                            
                        
                        
                        for param in SPT_parameters:                            
                            OurAdam._single_tensor_adam2([param["params"][0]], 
                                                        [param["params"][0].grad], 
                                                        [param["exp_avgs"]], 
                                                        [param["exp_avgs_sqs"]],
                                                        None, 
                                                        [torch.tensor(iteration)], 
                                                        amsgrad=False, 
                                                        beta1 = 0.9, 
                                                        beta2 = 0.999, 
                                                        lr =  0.1*param["lr"], 
                                                        #relevant=relevant, 
                                                        weight_decay=0, 
                                                        eps=1e-8, 
                                                        maximize=False, 
                                                        capturable=False)
                            param["params"][0].grad.detach_()
                            param["params"][0].grad.zero_()
                        
                        
                        for param in parameters:
                            optimizer_function = OurAdam._global_single_tensor_adam2 if Global_ADAM else OurAdam._single_tensor_adam2
                            
                            optimizer_function([param["params"][0]], 
                                                        [param["params"][0].grad], 
                                                        [param["exp_avgs"][:len(param["params"][0])]], 
                                                        [param["exp_avgs_sqs"][:len(param["params"][0])]],
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
                        if opt.dampen_scale_grad:
                            R = qvec2rotmat_torch(rotations)
                            S = torch.diag_embed(gaussians.scaling_activation(scales))
                            covariances = R @ S @ S.transpose(1, 2) @ R.transpose(1, 2)
                            scaling_factors = scale_along_view_dir / torch.sqrt(view_dir.T @ covariances @ view_dir)
                            
                            values, vectors = torch.linalg.eigh(covariances)
                            v_eig = vectors @ view_dir 
                            Lambda = torch.diag_embed(values)
                            Lambda_new = Lambda + (scaling_factors ** 2 - 1)[:, None, None] * (Lambda * (v_eig[:, :, None] * v_eig[:, None, :]))
                            
                            #dampened_covs = covariances + (scaling_factors ** 2 - 1)[:, None, None] * (torch.outer(view_dir, view_dir) @ covariances @ torch.outer(view_dir, view_dir))
                            eigenvalues, eigenvectors = torch.diagonal(Lambda_new, dim1 =-2, dim2=-1), vectors
                            eigenvalues.clamp_min_(0)
                            with torch.no_grad():
                                rotations = rotation_matrix_to_quaternion(eigenvectors)
                                scales = gaussians.scaling_inverse_activation(torch.sqrt(eigenvalues))
                            
                        if torch.sum(torch.isnan(opacity)) > 0 or torch.sum(torch.isnan(means3D)) > 0 or torch.sum(torch.isnan(scales)) > 0:
                            pass
                        
                        if opt.noise_lr > 0 and opt.densification == "MCMC":
                            def op_sigmoid(x, k=100, x0=0.995):
                                return 1 / (1 + torch.exp(-k * (x - x0)))
                            # 5e5 = opt.noise_lr
            
                            
                            L = build_scaling_rotation(gaussians.scaling_activation(scales[contributing_gaussians]), gaussians.rotation_activation(rotations[contributing_gaussians]))
                            actual_covariance = L @ L.transpose(1, 2)
                            noise = torch.randn_like(means3D[contributing_gaussians]) * (op_sigmoid(1- gaussians.opacity_activation(opacity[contributing_gaussians])))*opt.noise_lr*xyz_lr
                            noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                            means3D[contributing_gaussians] += noise
                    if (iteration in checkpoint_iterations):
                        print("\n[ITER {}] Saving Checkpoint".format(iteration))
                        torch.save(gaussians.properties, scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                        #torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

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
    parser.add_argument("--config", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    print(args)
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, view_graph=None)

    print("\nTraining complete.")
