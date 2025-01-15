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

def render_dynamic(dataset, pipe):
    network_gui.init("127.0.0.1", 6009)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.active_sh_degree = dataset.sh_degree
    gaussians.scaffold_points = None
    dataset.source_path = "/home/felix-windisch/Datasets/example_dataset_LOD/example_dataset/camera_calibration/aligned"
    dataset.hierarchy="/home/felix-windisch/Datasets/example_dataset_LOD/example_dataset/output/scaffold/point_cloud/iteration_30000/Alpha.dhier"
    dataset.scaffold_file=""
    scene = Scene(dataset, gaussians, resolution_scales = [1], create_from_hier=True)
    gaussians._opacity = gaussians.inverse_opacity_activation(gaussians._opacity)
    #debug_utils.render_level_slices(scene, pipe, "/home/felix-windisch/Datasets/example_dataset_LOD/example_dataset/output/")
    while True:
        if network_gui.conn == None:
            network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive_, scaling_modifer, slider = network_gui.receive()
                    if custom_cam != None:
                        limit = slider["x"]/100.0
                        show_depth = False
                        if "y" in slider:
                            show_depth = slider["y"] > 1
                        print(limit)
                        net_image = debug_utils.generate_hierarchy_scene_image(custom_cam, scene, pipe, limit=limit, show_depth=show_depth)

                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, "") #dataset.source_path)
                    if not keep_alive_:
                        break
                except Exception as e:
                    print(e)
                    network_gui.conn = None
                
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
    render_dynamic(lp.extract(args), pp.extract(args))