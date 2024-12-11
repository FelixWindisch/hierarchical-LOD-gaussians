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
from gaussian_renderer import render_post, render, render_coarse
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
from torch.utils.tensorboard import SummaryWriter
from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights, expand_to_size_dynamic, get_interpolation_weights_dynamic

def direct_collate(x):
    return x



def training(dataset, opt : OptimizationParams, pipe, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    #torch.autograd.set_detect_anomaly(True)
    writer = SummaryWriter()
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.active_sh_degree = dataset.sh_degree
    gaussians.scaffold_points = None
    scene = Scene(dataset, gaussians, resolution_scales = [1], create_from_hier=True)
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
    interpolation_weights = torch.zeros(gaussians._xyz.size(0)).float().cuda()
    num_siblings = torch.zeros(gaussians._xyz.size(0)).int().cuda()
    to_render = 0

    limmax = 0.02
    limmin = 0.0001

    while iteration < opt.iterations + 1:
        for viewpoint_batch in training_generator:
            for viewpoint_cam in viewpoint_batch:

                sample = torch.rand(1).item()
                # target granularity
                limit = math.pow(2, sample * (math.log2(limmax) - math.log2(limmin)) + math.log2(limmin))
                limit = 0
                scale = 1

                viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
                viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()

                #Then with blending training
                iter_start.record()

                gaussians.update_learning_rate(iteration)

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
                to_render = min(to_render, 569200)
                if to_render >= 569200:
                    print("Number of Gaussians overflowed")
                print(f"render {to_render} Gaussians out of {gaussians.get_number_of_leaf_nodes()}")

                # indices == nodes_for_render_indices !
                indices = render_indices[:to_render].int()
                node_indices = nodes_for_render_indices[:to_render]
                
                
                # parent indices contains as many elements as indices
                
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
                    use_trained_exp=True,
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
                            
                        progress_bar.close()
                        scene.dump_gaussians("Dump", only_leaves=True, file_name="ResultCloud")
                        print(f"Hierarchy bounding sphere divergence: {scene.gaussians.compute_bounding_sphere_divergence()}")

                        debug_utils.render_depth_slices(scene, pipe, dataset.scaffold_file)
                        debug_utils.render_level_slices(scene, pipe, dataset.scaffold_file)
                        return


                    # Densification
                    if iteration < opt.densify_until_iter:
                        #visibility_filter = visibility_filter[indices]
                        # Keep track of max radii in image-space for pruning
                        gaussians.max_radii2D[indices[visibility_filter]] = torch.max(gaussians.max_radii2D[indices[visibility_filter]], radii)
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, indices)
#
                        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                            print(os.path.join(scene.model_path, str(iteration) + ".png"))

                            print("-----------------DENSIFY!--------------------")
                            gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent)
                            print(f"Current LOD depth: {torch.max(gaussians.nodes[:, 0])}")
                            #gaussians.sanity_check_hierarchy()
                        # Resetting Opacity fucks our hierarchy                       
                        #if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        #    print("-----------------RESET OPACITY!-------------")
                        #    gaussians.reset_opacity()
                    # Optimizer step
                    if iteration < opt.iterations:
                        #print(f"Opacity Grad Sum: {torch.sum(gaussians._opacity.grad)}")
                        if gaussians._xyz.grad != None:
                            if gaussians.skybox_points != 0 and gaussians.skybox_locked: #No post-opt for skybox
                                gaussians._xyz.grad[0:gaussians.skybox_points, :] = 0
                                gaussians._rotation.grad[0:gaussians.skybox_points, :] = 0
                                gaussians._features_dc.grad[0:gaussians.skybox_points, :, :] = 0
                                gaussians._features_rest.grad[0:gaussians.skybox_points, :, :] = 0
                                gaussians._opacity.grad[0:gaussians.skybox_points, :] = 0
                                gaussians._scaling.grad[0:gaussians.skybox_points, :] = 0
                            
                            if torch.sum(torch.isnan(gaussians._scaling.grad)) > 0:
                                torchvision.utils.save_image(image, os.path.join(scene.model_path, "Error" + ".png"))
                                print("gradients collapsed :(")
                            else:
                                #gaussians.optimizer.step()
                                #gaussians.optimizer.zero_grad(set_to_none = True)
                                ## OurAdam version
                                if gaussians._opacity.grad != None:
                                    relevant = (gaussians._opacity.grad.flatten() != 0).nonzero()
                                    relevant = relevant.flatten().long()
                                    if(relevant.size(0) > 0):
                                        gaussians.optimizer.step(relevant)
                                    gaussians.optimizer.zero_grad(set_to_none = True)
                            #torchvision.utils.save_image(image, os.path.join(scene.model_path, "Last" + ".png"))
                            # This would prevent further training of the leaves
                            #gaussians._xyz.grad[gaussians.anchors, :] = 0
                            #gaussians._rotation.grad[gaussians.anchors, :] = 0
                            #gaussians._features_dc.grad[gaussians.anchors, :, :] = 0
                            #gaussians._features_rest.grad[gaussians.anchors, :, :] = 0
                            #gaussians._opacity.grad[gaussians.anchors, :] = 0
                            #gaussians._scaling.grad[gaussians.anchors, :] = 0
                            
                            
                            # Clamp values to avoid NaNs?
                            #with torch.no_grad():
                            #    gaussians._scaling.clamp_min_(-15)
                            #    gaussians._scaling.clamp_max_(10)
                            #    gaussians._opacity.clamp_min_(-3)
                            #    gaussians._opacity.clamp_max_(3)
                                #gaussians._opacity.clamp_max_(-10)
                        

                            
                            
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
