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
from stp_gaussian_rasterization import ExtendedSettings
from gaussian_renderer import occlusion_cull
import json
import pickle
from utils.image_utils import psnr
from lpipsPyTorch import lpips

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


WriteTensorBoard = False
#Standard
densify_interval = 200
lr_multiplier = 1
#Unused
Random_Hierarchy_Cut = True
Only_Noise_Visible = True
#MCMC
Max_Cap = 25_000_000
MCMC_Densification = True
MCMC_Noise_LR = 0  #5e5
lambda_scaling = 0
lambda_opacity = 0
#Hierarchical
Gaussian_Interpolation = False
# Upward Propagation 
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
SPT_Root_Volume = 100#100 #0.025
Target_Granularity_Pixels = 2
Cache_SPTs = True
Reuse_SPT_Tolerarance = 0.1
#View Selection
Use_Consistency_Graph = False
# Rasterizer
Rasterizer = "Vanilla"


SH_properties = [0, 3, 8, 15]
SH_properties_single = SH_properties[1] 
SH_properties = SH_properties[1] * 3
xyz1 = 0
xyz2 = 3
scales1 = 3
scales2 = 6
rotation1 = 6
rotation2 = 10
features1 = 10
features2 = 13
opacity1 = 13
opacity2 = 14
features_rest1 = 14
features_rest2 = 14 + SH_properties
number_properties = features_rest2

range1 = [xyz1, scales1, rotation1, features1, opacity1, features_rest1]
range2 = [xyz2, scales2, rotation2, features2, opacity2, features_rest2]

non_blocking=False
def render(dataset, opt:OptimizationParams, pipe, saving_iterations, checkpoint_iterations, checkpoint, debug_from,  hierarchy_path, replay=False, cam_path_id=0):
    camera_path_id = random.randint(0, 100000) if not replay else cam_path_id
    global Reuse_SPT_Tolerarance
    global Use_Occlusion_Culling
    opt.densification_interval = densify_interval
    #torch.cuda.memory._record_memory_history()
    #torch.autograd.set_detect_anomaly(True)
    
    splat_settings = ExtendedSettings.from_json('/home/felix-windisch/hierarchical-LOD-gaussians/configs/vanilla.json')
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.active_sh_degree = 1
    gaussians.max_sh_degree = 1
    gaussians.scaffold_points = None
    with torch.no_grad():
        gaussians._features_dc = gaussians._features_dc.abs() 
    dataset.eval = False
    dataset.hierarchy = hierarchy_path
    scene = Scene(dataset, gaussians, resolution_scales = [1], create_from_hier=True)
    gaussians.skybox_points = 100000
    base_focal_length = scene.getTrainCameras()[0].focal_length
    SPT_Target_Granularity = (1.0/base_focal_length) * Target_Granularity_Pixels
    with torch.no_grad():
        gaussians._opacity.clamp_(0, 0.99999)
        gaussians._opacity = gaussians.inverse_opacity_activation(gaussians._opacity)
        
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        

    
    #scene.dump_gaussians("Dump", only_leaves=True)

    gaussians.move_storage_to(Storage_Device, None, False, False, False)

    gaussians.build_hierarchical_SPT(SPT_Root_Volume, SPT_Target_Granularity, use_bounding_spheres=Use_Bounding_Spheres)
    print(f"Built {len(gaussians.SPT_starts)} SPTs")
    #scene.dump_gaussians("Dump", False, file_name="SPTNodes.ply", indices=spt_nodes.to(Storage_Device))
    #exit()
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    #progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    indices = None

    iteration = first_iter
    # Dataloader loads data from disk
    # DONT SHUFFLE IF USING CORRESPONDENCE GRAPH

    
    if Gradient_Propagation:
        gaussians.recompute_weights()
    
    #means3D, opacity, scales, rotations, features_dc, features_rest, gaussian_indices = torch.empty((0, 3), device='cuda', dtype=torch.float32), torch.empty((0, 1), device='cuda', dtype=torch.float32), torch.empty((0, 3), device='cuda', dtype=torch.float32), torch.empty((0,4), device='cuda', dtype=torch.float32), torch.empty((0, 1, 3), device='cuda', dtype=torch.float32), torch.empty((0,15, 3), device='cuda', dtype=torch.float32), torch.empty(0, device='cuda', dtype=torch.int32)
    if Cache_SPTs:
        means3D = gaussians.properties[:gaussians.skybox_points, xyz1:xyz2].cuda().contiguous()
        scales = gaussians.properties[:gaussians.skybox_points, scales1:scales2].cuda().contiguous()
        rotations = gaussians.properties[:gaussians.skybox_points, rotation1:rotation2].cuda().contiguous()
        features_dc = gaussians.properties[:gaussians.skybox_points, features1:features2].cuda().unsqueeze(1).contiguous()
        opacity = gaussians.properties[:gaussians.skybox_points, opacity1].cuda().unsqueeze(1).contiguous()
        features_rest = gaussians.properties[:gaussians.skybox_points, features_rest1: features_rest2].cuda().reshape(gaussians.skybox_points, 3, SH_properties_single).contiguous()
        
    else:
        means3D, opacity, scales, rotations, features_dc, features_rest, gaussian_indices = torch.empty((0, 3), device='cuda', dtype=torch.float32), torch.empty((0, 1), device='cuda', dtype=torch.float32), torch.empty((0, 3), device='cuda', dtype=torch.float32), torch.empty((0,4), device='cuda', dtype=torch.float32), torch.empty((0, 1, 3), device='cuda', dtype=torch.float32), torch.empty((0,15, 3), device='cuda', dtype=torch.float32), torch.empty(0, device='cuda', dtype=torch.int32)
    
    
    
    
    gaussians.xyz_scheduler_args = get_expon_lr_func(lr_init=opt.position_lr_init*gaussians.spatial_lr_scale,
                                                    lr_final=opt.position_lr_final*gaussians.spatial_lr_scale,
                                                    lr_delay_mult=opt.position_lr_delay_mult,
                                                    max_steps=opt.position_lr_max_steps)
    
    training_generator = DataLoader(scene.getTrainCameras(), num_workers = 8, prefetch_factor = 1, persistent_workers = True, collate_fn=direct_collate, shuffle=False)
    psnrs = 0.0
    ssims = 0.0
    lpipss = 0.0
    
    
    psnr_aerial = 0.0
    ssims_aerial = 0.0
    lpipss_aerial = 0.0
    
    
    psnr_street = 0.0
    ssims_street = 0.0
    lpipss_street = 0.0
    
    street_images = 0
    aerial_images = 0
    distance_multiplier = 1
    for x in [0]:
        for viewpoint_batch in training_generator:
            for viewpoint_cam in viewpoint_batch:
                viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                #viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
                viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()
                ############# SPT Cache
                if Use_Bounding_Spheres:
                    bounds = gaussians.bounding_sphere_radii
                else: 
                    bounds = (gaussians.scaling_activation(torch.max(gaussians.upper_tree_scaling, dim=-1)[0]) * 3.0)
                planes = gaussians.extract_frustum_planes(viewpoint_cam.full_proj_transform.cuda())
                if Use_Frustum_Culling:
                    frustum_cull = lambda indices : gaussians.frustum_cull_spheres(gaussians.upper_tree_xyz[indices], bounds[indices], planes)
                else:
                    frustum_cull = lambda indices : torch.ones(len(indices), dtype = torch.bool)
                camera_position = viewpoint_cam.camera_center.cuda()
                LOD_detail_cut = lambda indices : gaussians.min_distance_squared[indices] > (camera_position - gaussians.upper_tree_xyz[indices]).square().sum(dim=-1) * distance_multiplier
                # The coarse cut contains intermediate nodes from the upper tree and leaf nodes, with some leaf nodes containing SPTs
                coarse_cut = gaussians.cut_hierarchy_on_condition(gaussians.upper_tree_nodes, LOD_detail_cut, return_upper_tree=False, root_node=0, leave_out_of_cut_condition=frustum_cull)
                if Use_Occlusion_Culling:
                    bg_color = [0, 0, 0]
                    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                    temp = len(coarse_cut)
                    occlusion_indices = gaussians.upper_tree_nodes[coarse_cut, 5]
                    occlusion_mask, occlusion_image = occlusion_cull(occlusion_indices.to(Storage_Device), gaussians, viewpoint_cam, pipe, background)
                    occlusion_mask = occlusion_mask.cuda()
                    coarse_cut = coarse_cut[occlusion_mask]
                    print(f"Occlusion Cull {temp - len(coarse_cut)} out of {temp} upper tree gaussians")
                # leaf nodes have 0 children
                cut_leaf_nodes = coarse_cut[gaussians.upper_tree_nodes[coarse_cut, 2] == 0]
                # separate the cut into leafs that contain an SPT and those that don't
                # The SPT indices are the child indices of those nodes that have a 0 child count
                SPT_indices = gaussians.upper_tree_nodes[cut_leaf_nodes][gaussians.upper_tree_nodes[cut_leaf_nodes, 3] >= 0, 3]
                upper_tree_nodes_to_render = gaussians.upper_tree_nodes[coarse_cut][gaussians.upper_tree_nodes[coarse_cut, 3] <= 0, 5]
                SPT_upper_tree_indices = cut_leaf_nodes[gaussians.upper_tree_nodes[cut_leaf_nodes, 3] >= 0]
                SPT_distances = (gaussians.upper_tree_xyz[SPT_upper_tree_indices] - camera_position).pow(2).sum(1).sqrt() * distance_multiplier

                
                
                # Keep all the gaussians that are contained in an STP that is reused and close enough
                # Cumulative Sum Trick
                
                # Keep Skybox
                #keep_gaussians_mask[:gaussians.skybox_points] = True
                #mask = torch.isin(SPT_indices, keep_SPT_indices)
                  
                load_SPT_indices = SPT_indices
                load_SPT_distances = SPT_distances
                if len(load_SPT_indices) > 0:
                    #LOAD SPT CUT
                    #load_SPT_distances = torch.full((len(load_SPT_indices),), 10000.0).cuda()
                    load_SPT_gaussian_indices, load_SPT_starts = get_spt_cut_cuda(len(load_SPT_indices), gaussians.SPT_gaussian_indices, gaussians.SPT_starts, gaussians.SPT_max, gaussians.SPT_min, load_SPT_indices, load_SPT_distances)
                else:
                    print("No SPTs loaded")
                    load_SPT_gaussian_indices, load_SPT_starts = torch.empty(0, dtype=torch.int32, device='cuda'), torch.empty(0, dtype=torch.int32, device='cuda')
                print(clock())
                #SPT_counts += gaussians.skybox_points
                ### BAND AID FIX
                #difference = load_SPT_starts[1:] - load_SPT_starts[:-1]
                #empty_SPTs = torch.where(difference == 0)[0]
                #if len(empty_SPTs) > 0:
                #    print(f"Empty SPTs {empty_SPTs} encountered")
                #    #mask = torch.ones(len(load_SPT_starts), dtype=torch.bool, device='cuda')
                #    #mask.scatter_(0, empty_SPTs, False)
                #    #load_SPT_starts = load_SPT_starts[mask]
                #    #load_SPT_distances = load_SPT_distances[mask]
                #    #load_SPT_indices = load_SPT_indices[mask]
                if len(load_SPT_starts) > 0:    
                    if len(load_SPT_gaussian_indices) == load_SPT_starts[-1]:
                        print("Last SPT empty")
                        load_SPT_starts = load_SPT_starts[:-1]
                        load_SPT_distances = load_SPT_distances[:-1]
                        load_SPT_indices = load_SPT_indices[:-1]
                #    
                ### BAND AID FIX
                assert(len(load_SPT_starts.unique()) == len(load_SPT_starts))
                #cache_SPT_cache_indices = torch.where(~equal_SPT_cache_mask)[0]    
                #cache_SPT_indices = prev_SPT_indices[cache_SPT_cache_indices]
                #SPT_starts_new = torch.zeros(len(load_SPT_indices) + len(reuse_SPT_indices) + 1,dtype=torch.int32, device='cuda')
                # compact the prefix sum of SPT_counts
                #SPT_starts_new[:len(load_SPT_starts)] = load_SPT_starts + gaussians.skybox_points
                #SPT_starts_new[len(load_SPT_starts)] = len(load_SPT_gaussian_indices) + gaussians.skybox_points
                #prefix = len(cut_SPTs) + gaussians.skybox_points
                #for index, i in enumerate(SPT_keep_counts_indices):
                #    SPT_starts_new[index + len(SPT_counts)] = prefix
                #    prefix += (prev_SPT_counts[i+1] - prev_SPT_counts[i]).item()
                #sizes = prev_SPT_starts[prev_keep_SPT_cache_indices + 1] - prev_SPT_starts[prev_keep_SPT_cache_indices]
                #SPT_starts_new[len(load_SPT_starts) + 1:len(load_SPT_starts) + 1 + len(sizes)] = torch.cumsum(sizes, dim=0) +  len(load_SPT_gaussian_indices) + gaussians.skybox_points
                #number_of_gaussians_to_render = SPT_starts_new[len(load_SPT_starts) + len(sizes)]
                #SPT_distances = torch.cat((load_SPT_distances, prev_SPT_distances[prev_keep_SPT_cache_indices]))
                load_from_disk_indices = torch.cat((upper_tree_nodes_to_render, load_SPT_gaussian_indices))

                load_from_disk_indices = load_from_disk_indices.to(Storage_Device)

                
                distance_multiplier = distance_multiplier
                
                
                load_tensor = gaussians.properties[load_from_disk_indices, :].cuda(non_blocking=non_blocking)
                means3D = nn.Parameter(torch.cat((means3D[:gaussians.skybox_points], load_tensor[:, xyz1:xyz2].cuda(non_blocking=non_blocking))).contiguous())
                opacity = nn.Parameter(torch.cat((opacity[:gaussians.skybox_points], load_tensor[:, opacity1:opacity2].cuda(non_blocking=non_blocking))).contiguous())
                scales = nn.Parameter(torch.cat((scales[:gaussians.skybox_points], load_tensor[:, scales1:scales2].cuda(non_blocking=non_blocking))).contiguous())
                rotations = nn.Parameter(torch.cat((rotations[:gaussians.skybox_points], load_tensor[:, rotation1:rotation2].cuda(non_blocking=non_blocking))).contiguous())
                # TODO: ABS?
                features_dc = nn.Parameter(torch.cat((features_dc[:gaussians.skybox_points], load_tensor[:, features1:features2].cuda(non_blocking=non_blocking).unsqueeze(1))).contiguous())
                features_rest = nn.Parameter(torch.cat((features_rest[:gaussians.skybox_points], load_tensor[:, features_rest1:features_rest2].cuda(non_blocking=non_blocking).reshape(len(load_tensor), 3, SH_properties_single))).contiguous())

                torch.cuda.empty_cache()
                    
                    
                
                
                

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
                    override_color = None, 
                    sh_degree = gaussians.active_sh_degree,
                    )
                torch.cuda.empty_cache()
                image = render_pkg["render"]
                
                
                gt_image = viewpoint_cam.original_image.cuda()
                
                
                #torchvision.utils.save_image(image,  "output/eval" + str(iteration) + ".png")
                #torchvision.utils.save_image(gt_image,  "output/eval" + str(iteration) + "_gt.png")
                iteration += 1
                psnr_current = psnr(image.detach(), gt_image).mean().double()
                ssim_current = ssim(image.detach(), gt_image).mean().double()
                lpips_current = 0 #lpips(image, gt_image, net_type='vgg').mean().double()
                
                psnrs += psnr_current
                print(psnr(image, gt_image).mean().double())
                ssims += ssim_current
                lpipss += lpips_current
                print(f"PSNR: {psnr_current:.5f} SSIM: {ssim_current:.5f} LPIPS: {lpips_current:.5f}")
                print(image.shape[2])
                if image.shape[2]  < 1100:
                    street_images += 1
                    psnr_street += psnr_current
                    ssims_street += ssim_current
                    lpipss_street += lpips_current
                else:
                    aerial_images += 1
                    psnr_aerial += psnr_current
                    ssims_aerial += ssim_current
                    lpipss_aerial += lpips_current
                    
                print()
                ####### RENDER
    psnrs /= len(scene.getTrainCameras())
    ssims /= len(scene.getTrainCameras())
    lpipss /= len(scene.getTrainCameras())
    psnr_aerial /= aerial_images
    ssims_aerial /= aerial_images
    lpipss_aerial /= aerial_images
    psnr_street /= street_images
    ssims_street /= street_images
    lpipss_street /= street_images
    print(f"FINAL PSNR: {psnrs:.5f} SSIM: {ssims:.5f} LPIPS: {lpipss:.5f}")
    print(f"AERIAL PSNR: {psnr_aerial:.5f} SSIM: {ssims_aerial:.5f} LPIPS: {lpipss_aerial:.5f}")
    print(f"STREET PSNR: {psnr_street:.5f} SSIM: {ssims_street:.5f} LPIPS: {lpipss_street:.5f}")
                

                
                

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
    parser.add_argument("--hierarchy_path", type=str, default = None)
    parser.add_argument('--replay', type=bool, default=False)
    parser.add_argument('--ID', type=int, default=0)
    args = parser.parse_args(sys.argv[1:])
    print(args)
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    render(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.hierarchy_path, args.replay, args.ID)

    print("\nTraining complete.")
