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
Storage_Device = 'cuda'
lambda_hierarchy = 0.00
SPT_Root_Volume = 0.025
SPT_Target_Granularity = 0.00005
Cache_SPTs = True
Reuse_SPT_Tolerarance = 0.25
#View Selection
Use_Consistency_Graph = False
# Rasterizer
Rasterizer = "Vanilla"


non_blocking=False
def training(dataset, opt:OptimizationParams, pipe, saving_iterations, checkpoint_iterations, checkpoint, debug_from,  hierarchy_path):
    global Reuse_SPT_Tolerarance
    opt.densification_interval = densify_interval
    #torch.cuda.memory._record_memory_history()
    #torch.autograd.set_detect_anomaly(True)
    
    splat_settings = ExtendedSettings.from_json('/home/felix-windisch/hierarchical-LOD-gaussians/configs/vanilla.json')
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
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.active_sh_degree = dataset.sh_degree
    gaussians.scaffold_points = None
    with torch.no_grad():
        gaussians._features_dc = gaussians._features_dc.abs() 
    dataset.eval = True
    dataset.hierarchy = hierarchy_path
    scene = Scene(dataset, gaussians, resolution_scales = [1], create_from_hier=True)
    with torch.no_grad():
        gaussians._opacity.clamp_(0, 0.99999)
        gaussians._opacity = gaussians.inverse_opacity_activation(gaussians._opacity)
        
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        
    render_indices = torch.zeros(gaussians._xyz.size(0)).int().cuda()
    parent_indices = torch.zeros(gaussians._xyz.size(0)).int().cuda()
    nodes_for_render_indices = torch.zeros(gaussians._xyz.size(0)).int().cuda()
    num_siblings = torch.zeros(gaussians._xyz.size(0)).int().cuda()
    
    #scene.dump_gaussians("Dump", only_leaves=True)

    optimizer_state = gaussians.move_storage_to_render(Storage_Device, None)

    gaussians.build_hierarchical_SPT(SPT_Root_Volume, SPT_Target_Granularity, use_bounding_spheres=Use_Bounding_Spheres)
    print("Built SPTs")
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
    
    #means3D, opacity, scales, rotations, features_dc, features_rest, render_indices = torch.empty((0, 3), device='cuda', dtype=torch.float32), torch.empty((0, 1), device='cuda', dtype=torch.float32), torch.empty((0, 3), device='cuda', dtype=torch.float32), torch.empty((0,4), device='cuda', dtype=torch.float32), torch.empty((0, 1, 3), device='cuda', dtype=torch.float32), torch.empty((0,15, 3), device='cuda', dtype=torch.float32), torch.empty(0, device='cuda', dtype=torch.int32)
    if Cache_SPTs:
        render_indices = torch.arange(0, gaussians.skybox_points, device='cuda')
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
    gaussians.xyz_scheduler_args = get_expon_lr_func(lr_init=opt.position_lr_init*gaussians.spatial_lr_scale,
                                                    lr_final=opt.position_lr_final*gaussians.spatial_lr_scale,
                                                    lr_delay_mult=opt.position_lr_delay_mult,
                                                    max_steps=opt.position_lr_max_steps)



    ######## VIEWER
    network_gui.init("127.0.0.1", 6009)
    while True:
        if network_gui.conn == None:
            print("Try Connect")
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, keep_alive_, scaling_modifer, slider = network_gui.receive()
                if "distance_multiplier" in slider:
                    distance_multiplier = slider["distance_multiplier"]
                else:
                    distance_multiplier = 1.0
                if "render_SPTs" in slider:
                    render_SPTs = slider["render_SPTs"] > 0
                    if render_SPTs:
                        Reuse_SPT_Tolerarance = 0.0
                else:
                    render_SPTs = False
                if "freeze_view" in slider:    
                    freeze_view = slider["freeze_view"] > 0
                else:
                    freeze_view = False
                    
                if "distance_color" in slider and render_SPTs:    
                    color_distance = slider["distance_color"] > 0
                else:
                    color_distance = False
                if "size_color" in slider and render_SPTs and not color_distance:    
                    color_size = slider["size_color"] > 0
                else:
                    color_size = False
                if "reuse_SPT_tolerance" in slider and not render_SPTs:    
                    Reuse_SPT_Tolerarance = slider["reuse_SPT_tolerance"]
                if "separate_SPTs" in slider and not render_SPTs:    
                    separate_SPTs = slider["separate_SPTs"] > 0
                else:
                    separate_SPTs = False
                    
                    
                if custom_cam != None:
                    ####### RENDER
                    viewpoint_cam = custom_cam
                    viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                    #viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
                    viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                    viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()
                    
                    ############# SPT Cache
                    if Use_Bounding_Spheres:
                        bounds = gaussians.bounding_sphere_radii
                    else: 
                        bounds = (gaussians.scaling_activation(torch.max(gaussians.upper_tree_scaling, dim=-1)[0]) * 2.5)
                    planes = gaussians.extract_frustum_planes(viewpoint_cam.full_proj_transform.cuda())
                    if Use_Frustum_Culling:
                        cull = lambda indices : gaussians.frustum_cull_spheres(gaussians.upper_tree_xyz[indices], bounds[indices], planes)
                    else:
                        cull = lambda indices : torch.ones(len(indices), dtype = torch.bool)
                    camera_position = viewpoint_cam.camera_center.cuda()
                    if not freeze_view:
                        LOD_detail_cut = lambda indices : gaussians.min_distance_squared[indices] > ((camera_position - gaussians.upper_tree_xyz[indices]).square().sum(dim=-1) * distance_multiplier)
                        #LOD_detail_cut = lambda indices : torch.ones_like(indices, dtype=torch.bool)
                        coarse_cut = gaussians.cut_hierarchy_on_condition(gaussians.upper_tree_nodes, LOD_detail_cut, return_upper_tree=False, root_node=0, leave_out_of_cut_condition=cull)

                        if Use_Occlusion_Culling:
                            bg_color = [0, 0, 0]
                            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                            temp = len(coarse_cut)
                            occlusion_indices = gaussians.upper_tree_nodes[coarse_cut, 5]
                            occlusion_mask = occlusion_cull(occlusion_indices.to(Storage_Device), gaussians, custom_cam, pipe, background).cuda()
                            coarse_cut = coarse_cut[occlusion_mask]
                            print(f"Occlusion Cull {temp - len(coarse_cut)} out of {temp} upper tree gaussians")
                        leaf_mask = gaussians.upper_tree_nodes[coarse_cut, 2] == 0
                        leaf_nodes = coarse_cut[leaf_mask]
                        # separate the cut into leafs that contain an SPT and those that don't
                        SPT_indices = gaussians.upper_tree_nodes[leaf_nodes][gaussians.upper_tree_nodes[leaf_nodes, 3] >= 0, 3]
                        upper_tree_nodes_to_render = gaussians.upper_tree_nodes[leaf_nodes][gaussians.upper_tree_nodes[leaf_nodes, 3] <= 0, 5]
                        if render_SPTs:
                            upper_tree_nodes_to_render = torch.empty(0, dtype=torch.int32, device='cuda')

                        SPT_node_indices = leaf_nodes[gaussians.upper_tree_nodes[leaf_nodes, 3] >= 0]
                        SPT_distances = (gaussians.upper_tree_xyz[SPT_node_indices] - camera_position).pow(2).sum(1).sqrt() * distance_multiplier
                        if render_SPTs:
                            SPT_distances = torch.full((len(SPT_node_indices),), 10000, device='cuda', dtype=torch.float)

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
                                to = len(render_indices)
                            else:
                                to = prev_SPT_counts[i+1]
                            keep_gaussians_mask[prev_SPT_counts[i]:to] = True

                        # Keep Skybox
                        keep_gaussians_mask[:gaussians.skybox_points] = True


                        mask = torch.isin(SPT_indices, keep_SPT_indices)
                        load_SPT_indices = SPT_indices[~mask]

                        load_SPT_distances = (gaussians.upper_tree_xyz[load_SPT_indices] - camera_position).pow(2).sum(1).sqrt() * distance_multiplier
                        if render_SPTs:
                            load_SPT_distances = torch.full((len(load_SPT_distances),), 100000000, device='cuda', dtype=torch.float)
                            #cut_SPTs = gaussians.SPT_
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
                                to = len(render_indices)
                            else:
                                to = prev_SPT_counts[i+1]
                            prefix += (to - prev_SPT_counts[i]).item()

                        SPT_counts_new[len(keep_SPT_indices):] = SPT_counts + prefix
                        # TODO: Remove SPT_distances that are loaded
                        SPT_distances = torch.cat((prev_SPT_distances[SPT_keep_counts_indices], load_SPT_distances))

                        load_from_disk_indices = torch.cat((cut_SPTs,  upper_tree_nodes_to_render))

                        write_back_mask = ~keep_gaussians_mask
                        write_back_indices = render_indices[write_back_mask].to(Storage_Device)

                        render_indices = torch.cat((render_indices[keep_gaussians_mask], load_from_disk_indices))

                        load_from_disk_indices = load_from_disk_indices.to(Storage_Device)
                        hierarchy_cut_time = clock()
                        clock()


                        means3D = nn.Parameter(torch.cat((means3D[keep_gaussians_mask].detach(), gaussians._xyz[load_from_disk_indices].cuda(non_blocking=non_blocking))).contiguous())
                        opacity = nn.Parameter(torch.cat((opacity[keep_gaussians_mask].detach(), gaussians._opacity[load_from_disk_indices].cuda(non_blocking=non_blocking))).contiguous())
                        scales = nn.Parameter(torch.cat((scales[keep_gaussians_mask].detach(), gaussians._scaling[load_from_disk_indices].cuda(non_blocking=non_blocking))).contiguous())
                        rotations = nn.Parameter(torch.cat((rotations[keep_gaussians_mask].detach(), gaussians._rotation[load_from_disk_indices].cuda(non_blocking=non_blocking))).contiguous())
                        # TODO: ABS?
                        features_dc = nn.Parameter(torch.cat((features_dc[keep_gaussians_mask].detach(), gaussians._features_dc[load_from_disk_indices].cuda(non_blocking=non_blocking))).contiguous())
                        features_rest = nn.Parameter(torch.cat((features_rest[keep_gaussians_mask].detach(), gaussians._features_rest[load_from_disk_indices].cuda(non_blocking=non_blocking))).contiguous())
                        shs = torch.cat((features_dc, features_rest), dim=1).contiguous()
                        torch.cuda.synchronize()

                        #parameters_new = []
                        #for index, (values, name, lr) in enumerate(zip([means3D, features_dc, opacity, scales, rotations, features_rest], 
                        #                        ["xyz", "f_dc", "opacity", "scaling", "rotation", "f_rest"],
                        #                        [xyz_lr, opt.feature_lr, opt.opacity_lr, opt.scaling_lr, opt.rotation_lr, opt.feature_lr])):
                        #    optimizer_state[name]["exp_avgs"][write_back_indices] = parameters[index]["exp_avgs"][write_back_mask].to(Storage_Device)
                        #    optimizer_state[name]["exp_avgs_sqs"][write_back_indices] = parameters[index]["exp_avgs_sqs"][write_back_mask].to(Storage_Device)
                        #    exp_avgs = torch.cat((parameters[index]["exp_avgs"][keep_gaussians_mask], optimizer_state[name]["exp_avgs"][load_from_disk_indices].cuda())).contiguous()
                        #    exp_avgs_sqs = torch.cat((parameters[index]["exp_avgs_sqs"][keep_gaussians_mask], optimizer_state[name]["exp_avgs_sqs"][load_from_disk_indices].cuda())).contiguous()
                        #    parameters_new.append({'params': [values], 'lr': lr*lr_multiplier, "name": name, 
                        #     "exp_avgs" : exp_avgs, "exp_avgs_sqs" : exp_avgs_sqs})
                        #parameters = parameters_new

                        prev_SPT_indices = SPT_indices
                        prev_SPT_distances = SPT_distances
                        prev_SPT_counts = SPT_counts_new

                        load_write_time = clock()
                        torch.cuda.empty_cache()
                    
                    
                    
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
                        if color_distance:
                            colors_precomp = torch.zeros_like(features_dc)
                            SPT_distances_remapped = (SPT_distances - SPT_distances.min()) / (SPT_distances.max() - SPT_distances.min())
                            colors_precomp[:, 0] = SPT_distances_remapped
                            colors_precomp[:, 1] = SPT_distances_remapped
                            colors_precomp[:, 2] = SPT_distances_remapped
                        elif color_size:
                            colors_precomp = torch.zeros_like(features_dc)
                            SPT_sizes = SPT_counts_new
                            SPT_sizes_remapped = (SPT_sizes - SPT_sizes.min()) / (SPT_sizes.max() - SPT_sizes.min())
                            colors_precomp[:, 0] = SPT_sizes_remapped
                            colors_precomp[:, 1] = SPT_sizes_remapped
                            colors_precomp[:, 2] = SPT_sizes_remapped
                        elif separate_SPTs:
                            colors_precomp = torch.zeros_like(features_dc)
                            color = torch.zeros_like(render_indices)
                            color[-len(upper_tree_nodes_to_render):] = 1
                            colors_precomp[:, 0] = color
                            colors_precomp[:, 1] = 0.2
                            colors_precomp[:, 2] = 0.2
                        else:
                            colors_precomp = None
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
                            override_color = colors_precomp, 
                            sh_degree = gaussians.active_sh_degree,
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


                    image = render_pkg["render"]
                    ####### RENDER
                    net_image = image.cpu()
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().to('cpu').numpy())
                train_params = {"Num_Rendered" : len(render_indices), "Number_of_SPTs" : len(SPT_indices), "Percentage_Rendered" : len(render_indices)/len(gaussians._xyz), "Percentage_SPTs" : len(SPT_indices)/len(gaussians.SPT_starts)}
                network_gui.send(net_image_bytes, json.dumps({"iteration" : 99, "num_gaussians" : len(gaussians._xyz), "loss" : 0, "sh_degree":1, "error" : 0, "paused" : False, "train_params" : train_params})) #dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive_):
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None
######## VIEWER

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
    args = parser.parse_args(sys.argv[1:])
    print(args)
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.hierarchy_path)

    print("\nTraining complete.")
