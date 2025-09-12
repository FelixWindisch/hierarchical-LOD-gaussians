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
from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights, expand_to_size_dynamic, get_interpolation_weights_dynamic
import matplotlib 
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
def direct_collate(x):
    return x

def eval_dynamic(dataset, pipe):
    psnr_test = 0.0
    ssims = 0.0
    lpipss = 0.0
    dataset.eval = True
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.active_sh_degree = dataset.sh_degree
    gaussians.scaffold_points = None
    dataset.source_path = "/home/felix-windisch/Datasets/example_dataset_LOD/example_dataset/camera_calibration/aligned"
    dataset.images = "../rectified/images"
    dataset.hierarchy="/home/felix-windisch/Datasets/example_dataset_LOD/example_dataset/output/scaffold/point_cloud/iteration_30000/hierarchy.dhier_opt"
    output_dir = "/home/felix-windisch/Datasets/example_dataset_LOD/example_dataset/output/evaluation"
    scene = Scene(dataset, gaussians, resolution_scales = [1], create_from_hier=True)
    gaussians._opacity = gaussians.inverse_opacity_activation(gaussians._opacity)

    gaussians_per_limit =  debug_utils.get_gaussians_per_limit_normalized(scene, 0, 0.1, 100, 5)
    #plt.axes().set_yscale('log')
    for i in range(5):
        plt.plot(np.linspace(0, 0.1, 100), gaussians_per_limit[:, i])
    #plt.show()
    limits = [0, 0.01, 0.1]
    training_generator = DataLoader(scene.getTestCameras(), num_workers = 8, prefetch_factor = 1, persistent_workers = True, collate_fn=direct_collate, shuffle=False)
    index = 0
    for limit in limits:
        for viewpoint_batch in tqdm(training_generator):
                for viewpoint_cam in viewpoint_batch:

                    #print(index:=index+1)

                    image = debug_utils.generate_hierarchy_scene_image(viewpoint_cam, scene, pipe, limit=limit, show_depth=False)

                    #torchvision.utils.save_image(image, os.path.join(output_dir, "eval" + str(index) + ".png"))

                    gt_image = viewpoint_cam.original_image.cuda()
                    #torchvision.utils.save_image(gt_image, os.path.join(output_dir, "gt" + str(index) + ".png"))

                    psnr_test += psnr(image, gt_image).mean().double()
                    
                    ssims += ssim(image, gt_image).mean().double()
                    #lpipss += lpips(image, gt_image, net_type='vgg').mean().double()
        psnr_test /= len(scene.getTestCameras())
        ssims /= len(scene.getTestCameras())
        lpipss /= len(scene.getTestCameras())
        print(f"limit: {limit}, PSNR: {psnr_test:.5f} SSIM: {ssims:.5f} LPIPS: {lpipss:.5f}")
    
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


    args.save_iterations.append(args.iterations)
    # Initialize system state (RNG)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    eval_dynamic(lp.extract(args), pp.extract(args))