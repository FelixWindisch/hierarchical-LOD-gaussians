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

import os
import torch
import debug_utils
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render_post, render, render_coarse, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import strip_symmetric, build_scaling_rotation
import math
import torchvision
from torch.utils.tensorboard import SummaryWriter
from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights, expand_to_size_dynamic, get_interpolation_weights_dynamic
  



def direct_collate(x):
    return x

Only_Noise_Visible = True
MCMC_Densification = True
Gaussian_Interpolation = False
Gradient_Propagation = True
Propagation_Strength = 1.0
lambda_hierarchy = 0.01

def training(dataset, opt:OptimizationParams, pipe, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    network_gui.init("127.0.0.1", 6009)
    #torch.autograd.set_detect_anomaly(True)
    writer = SummaryWriter()
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.active_sh_degree = dataset.sh_degree
    gaussians.scaffold_points = None
    
    
    dataset.eval = True
    scene = Scene(dataset, gaussians, resolution_scales = [1], create_from_hier=True)
    #gaussians.sort_morton()
    #with torch.no_grad():
    #    gaussians._opacity.sigmoid_()
    gaussians.training_setup(opt, our_adam=True)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    indices = None

    iteration = first_iter
    training_generator = DataLoader(scene.getTrainCameras(), num_workers = 8, prefetch_factor = 1, persistent_workers = True, collate_fn=direct_collate, shuffle=True)

    limit = 0.001

    render_indices = torch.zeros(gaussians._xyz.size(0)).int().cuda()
    parent_indices = torch.zeros(gaussians._xyz.size(0)).int().cuda()
    nodes_for_render_indices = torch.zeros(gaussians._xyz.size(0)).int().cuda()
    
    num_siblings = torch.zeros(gaussians._xyz.size(0)).int().cuda()
    to_render = 0
    
    limmax = 0.02
    limmin = 0.0001
    if Gradient_Propagation:
        gaussians.recompute_weights()
    
        
    while iteration < opt.iterations + 1:
        for viewpoint_batch in training_generator:
            for viewpoint_cam in viewpoint_batch:
                if network_gui.conn == None:
                    network_gui.try_connect()
                while network_gui.conn != None:
                    try:
                        net_image_bytes = None
                        custom_cam, do_training, keep_alive_, scaling_modifer, slider = network_gui.receive()
                        
                        if custom_cam != None:
                            limit = slider["x"]/100.0
                            net_image = debug_utils.generate_hierarchy_scene_image(custom_cam, scene, pipe, limit=limit)

                            net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                        
                        network_gui.send(net_image_bytes, "") #dataset.source_path)
                        if do_training and ((iteration < int(opt.iterations)) or not keep_alive_):
                            break
                    except Exception as e:
                        print(e)
                        network_gui.conn = None
                
                #recompute Gaussian weights
                if Gradient_Propagation and iteration % 10 == 0:
                    gaussians.recompute_weights()
                
                sample = torch.rand(1).item()
                # target granularity
                limit = math.pow(2, sample * (math.log2(limmax) - math.log2(limmin)) + math.log2(limmin))
                if iteration % 250 == 0:
                    limit = 0
                scale = 1
                writer.add_scalar('Limit', limit, iteration)
                viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
                viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()

                #Then with blending training
                iter_start.record()

                xyz_lr = gaussians.update_learning_rate(iteration)

                # Every 1000 its we increase the levels of SH up to a maximum degree
                if iteration % 1000 == 0:
                    gaussians.oneupSHdegree()

                camera_direction = torch.tensor([0.0, 0.0, 1.0])
                camera_direction = torch.matmul(torch.from_numpy(viewpoint_cam.R).to(torch.float32), camera_direction)

                to_render = expand_to_size_dynamic(
                    gaussians.nodes,
                    gaussians._xyz,
                    gaussians.get_scaling,
                    limit * scale,
                    viewpoint_cam.camera_center,
                    camera_direction,
                    render_indices,
                    parent_indices,
                    nodes_for_render_indices)

                #print(f"render {to_render} Gaussians out of {gaussians.get_number_of_leaf_nodes()}")

                # indices == nodes_for_render_indices !
                indices = render_indices[:to_render].int()
                node_indices = nodes_for_render_indices[:to_render]
                
                # parent indices contains as many elements as indices
                interpolation_weights = torch.zeros(gaussians._xyz.size(0)).float().cuda()
                if Gaussian_Interpolation:
                    get_interpolation_weights_dynamic(
                        node_indices,
                        limit * scale,
                        gaussians.nodes,
                        gaussians._xyz,
                        gaussians._scaling,
                        viewpoint_cam.camera_center.cpu(),
                        camera_direction,
                        interpolation_weights,
                        num_siblings
                    )
                else:
                    interpolation_weights[:len(indices)] = 1
                
                # Render
                if (iteration - 1) == debug_from:
                    pipe.debug = True

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
                    use_trained_exp=True, iteration=iteration
                    )
                #render_pkg = render_coarse(
                #    viewpoint_cam, 
                #    gaussians, 
                #    pipe, 
                #    background, 
                #    indices=indices
                #    )
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                if iteration % 250 == 0:
                    torchvision.utils.save_image(image, os.path.join(scene.model_path, str(iteration) + ".png"))
                    writer.add_image('images', image, iteration)
                # Loss
                gt_image = viewpoint_cam.original_image.cuda()
                if viewpoint_cam.alpha_mask is not None:
                    Ll1 = l1_loss(image * viewpoint_cam.alpha_mask.cuda(), gt_image)
                    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image * viewpoint_cam.alpha_mask.cuda(), gt_image))
                else:
                    Ll1 = l1_loss(image, gt_image) 
                    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))


                parents = gaussians.nodes[indices, 1]
                hierarchy_loss = torch.sum(torch.clamp_min(torch.max(torch.abs(gaussians.get_scaling[indices]), dim=-1)[0] - torch.max(torch.abs(gaussians.get_scaling[parents]), dim=-1)[0], 0)) / len(indices)
                loss = loss + lambda_hierarchy * hierarchy_loss
                #MCMC
                if MCMC_Densification:
                    # 0.01 = args.opacity_reg/scale_reg
                    
                    
                    # loss for active gaussians that have high opacity/scale
                    all_indices = torch.cat((indices, parents)).unique()
                    opacity_loss = torch.sum(torch.abs(gaussians.get_opacity.squeeze()[all_indices] * interpolation_weights[all_indices])) / len(all_indices)
                    loss = loss + 0.1 * opacity_loss
                    scaling_loss = torch.sum(torch.sum(torch.abs(gaussians.get_scaling[all_indices]), dim=-1) * interpolation_weights[all_indices]) / (3*len(all_indices))
                    loss = loss + 0.01 * scaling_loss
                    writer.add_scalar('Hierarchy Loss', hierarchy_loss, iteration)
                    writer.add_scalar('Opacity Loss', opacity_loss, iteration)
                    writer.add_scalar('Scaling Loss', scaling_loss, iteration)
                #MCMC
                if math.isnan(loss):
                            torchvision.utils.save_image(image, os.path.join(scene.model_path, "Error" + ".png"))
                            print("gradients collapsed :(")
                
                loss.backward()
                writer.add_scalar('Loss', loss, iteration)
                writer.add_scalar('Number of hierarchy levels', gaussians.get_number_of_levels(), iteration)
                writer.add_scalar('Number of Gaussians', len(gaussians._xyz), iteration)
                writer.add_scalars('Opacity', {"min":torch.min(gaussians.get_opacity), "mean": torch.mean(gaussians.get_opacity), "max": torch.max(gaussians.get_opacity)}, iteration)
                writer.add_scalars('Scaling', {"min":torch.min(gaussians.get_scaling), "mean": torch.mean(gaussians.get_scaling), "max": torch.max(gaussians.get_scaling)}, iteration)

                iter_end.record()

                with torch.no_grad():
                    # Progress bar
                    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                    if iteration % 10 == 0:
                        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Size": f"{gaussians._xyz.size(0)}", "Peak memory": f"{torch.cuda.max_memory_allocated(device='cuda')}"})
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


                    # Densification
                    if iteration < opt.densify_until_iter and not MCMC_Densification:
                        
                        # Keep track of max radii in image-space for pruning
                        gaussians.max_radii2D[indices[visibility_filter]] = torch.max(gaussians.max_radii2D[indices[visibility_filter]], radii)
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, indices)
#
                        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                            print(os.path.join(scene.model_path, str(iteration) + ".png"))

                            print("-----------------DENSIFY!--------------------")
                            gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent)
                            if Gradient_Propagation:
                                gaussians.recompute_weights()
                            print(f"Current LOD depth: {torch.max(gaussians.nodes[:, 0])}")
                            #gaussians.sanity_check_hierarchy()
                        # Resetting Opacity fucks our hierarchy                       
                        #if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        #    print("-----------------RESET OPACITY!-------------")
                        #    gaussians.reset_opacity()
                    # Optimizer step
                    
                    if MCMC_Densification and iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        print("-----------------DENSIFY!--------------------")
                        
                        dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
                        # only redistribute leaf nodes
                        dead_mask = torch.logical_and(dead_mask, gaussians.nodes[:, 2] == 0)
                        print(f"Respawn {torch.sum(dead_mask)} Gaussians")
                        gaussians.relocate_gs(dead_mask=dead_mask)
                        gaussians.add_new_gs(cap_max=3_000_000)   
                        if Gradient_Propagation:
                            gaussians.recompute_weights()
                    elif iteration < opt.iterations:
                        #print(f"Opacity Grad Sum: {torch.sum(gaussians._opacity.grad)}")
                        if gaussians._xyz.grad != None:
                            if gaussians.skybox_points != 0 and gaussians.skybox_locked: #No post-opt for skybox
                                gaussians._xyz.grad[0:gaussians.skybox_points, :] = 0
                                gaussians._rotation.grad[0:gaussians.skybox_points, :] = 0
                                gaussians._features_dc.grad[0:gaussians.skybox_points, :, :] = 0
                                gaussians._features_rest.grad[0:gaussians.skybox_points, :, :] = 0
                                gaussians._opacity.grad[0:gaussians.skybox_points, :] = 0
                                gaussians._scaling.grad[0:gaussians.skybox_points, :] = 0
                            
                        
                        #gaussians.optimizer.step()
                        #gaussians.optimizer.zero_grad(set_to_none = True)
                        ## OurAdam version
                        
                        if gaussians._opacity.grad != None:
                            relevant = (gaussians._opacity.grad.flatten() != 0).nonzero()
                            relevant = relevant.flatten().long()
                            if Gradient_Propagation:
                                prev_xyz = gaussians._xyz[relevant].detach().clone()
                                prev_scaling = gaussians._scaling[relevant].detach().clone()
                                prev_features_dc = gaussians._features_dc[relevant].detach().clone()
                                prev_opacity = gaussians._opacity[relevant].detach().clone()
                            if(relevant.size(0) > 0):
                                gaussians.optimizer.step(relevant)
                            if Gradient_Propagation:
                                changed = relevant.detach().clone()
                                d_xyz = (gaussians._xyz[changed] - prev_xyz) * Propagation_Strength
                                d_scale = (gaussians._scaling[changed] - prev_scaling) * Propagation_Strength
                                d_features_dc = (gaussians._features_dc[changed] - prev_features_dc) * Propagation_Strength
                                d_opacity = (gaussians._opacity[changed] - prev_opacity) * Propagation_Strength 
                                # Upward Propagation:
                                with torch.no_grad():
                                    #for i in range(1):
                                    while(torch.sum(d_xyz) > 0.0001):
                                        #print(len(relevant))
                                        parents = gaussians.nodes[changed, 1]
                                        unique_elements, counts = torch.unique(parents, return_counts=True)
                                        unique_only = unique_elements[counts == 1]

                                        mask = torch.isin(parents, unique_only)
                                        mask = torch.logical_and(mask, parents > gaussians.skybox_points)
                                        
                                        #d_xyz = torch.zeros_like(d_xyz).scatter_add_()
                                        
                                        parents = parents[mask]
                                        d_xyz *= gaussians.weights[changed].view(-1, 1)
                                        d_scale *= gaussians.weights[changed].view(-1, 1)
                                        d_features_dc[:, 0, :] *= gaussians.weights[changed].view(-1, 1)
                                        d_xyz = d_xyz[mask]
                                        d_scale = d_scale[mask]
                                        d_features_dc = d_features_dc[mask]
                                        d_opacity = d_opacity[mask]
                                        gaussians._xyz[parents] +=  d_xyz
                                        #gaussians._scaling[parents] +=  d_scale
                                        gaussians._features_dc[parents] +=  d_features_dc
                                        gaussians._opacity[parents] += d_opacity
                                        changed = parents
                            gaussians.optimizer.zero_grad(set_to_none = True)
                            
                        if MCMC_Densification:
                            
                            def op_sigmoid(x, k=100, x0=0.995):
                                return 1 / (1 + torch.exp(-k * (x - x0)))
                            # 5e5 = opt.noise_lr
                            if Only_Noise_Visible:
                                L = build_scaling_rotation(gaussians.get_scaling[indices], gaussians.get_rotation[indices])
                                actual_covariance = L @ L.transpose(1, 2)
                                noise = torch.randn_like(gaussians._xyz[indices]) * (op_sigmoid(1- gaussians.get_opacity[indices]))*5e5*xyz_lr
                                noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                                gaussians._xyz[indices]+= noise
                            else:
                                L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                                actual_covariance = L @ L.transpose(1, 2)
                                noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_opacity))*5e5*xyz_lr
                                noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                                gaussians._xyz.add_(noise)

                            
                        
                        

                            
                            
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
    exit()
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")
