import os, sys
import subprocess
import argparse
import time
import platform
import torch
from pathlib import Path
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
import debug_utils
import train_post
import consistency_graph
import networkx as nx
import train_coarse
def submit_job(slurm_args):
    """Submit a job using sbatch and return the job ID."""    
    try:
        result = subprocess.run(slurm_args, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error when submitting a job: {e}")
        sys.exit(1)
    # Extract job ID from sbatch output
    job_id = result.stdout.strip().split()[-1]
    print(f"submitted job {job_id}")

    return job_id

def is_job_finished(job_id):
    """Check if the job has finished using sacct."""
    result = subprocess.run(['sacct', '-j', job_id, '--format=State', '--noheader', '--parsable2'], capture_output=True, text=True)
    # Get job state
    job_state = result.stdout.split('\n')[0]
    return job_state if job_state in {'COMPLETED', 'FAILED', 'CANCELLED'} else ""

def setup_dirs(images, depths, masks, colmap, chunks, output, project):
    images_dir = "../rectified/images" if images == "" else images
    depths_dir = "../rectified/depths" if depths == "" else depths
    if masks == "":
        if os.path.exists(os.path.join(project, "camera_calibration/rectified/masks")):
            masks_dir = "../rectified/masks"
        else:
            masks_dir = ""
    else:
        masks_dir = masks
    #colmap_dir = os.path.join(project) if colmap == "" else colmap
    colmap_dir = os.path.join(project, "camera_calibration", "aligned") if colmap == "" else colmap
    chunks_dir = os.path.join(project, "camera_calibration", "chunks") if chunks == "" else chunks
    output_dir = os.path.join(project, "output") if output == "" else output

    return images_dir, depths_dir, masks_dir, colmap_dir, chunks_dir, output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.add_argument('--project_dir', required=True, help="Only the project dir has to be specified, other directories will be set according to the ones created using generate_colmap and generate_chunks scripts. They still can be explicitly specified.")
    parser.add_argument('--env_name', default="hierarchical_3d_gaussians")
    parser.add_argument('--extra_training_args', default="", help="Additional arguments that can be passed to training scripts. Not passed to slurm yet")
    parser.add_argument('--colmap_dir', default="")
    parser.add_argument('--images_dir', default="")
    parser.add_argument('--masks_dir', default="")
    parser.add_argument('--depths_dir', default="")
    parser.add_argument('--chunks_dir', default="")
    
    parser.add_argument('--output_dir', default="")
    parser.add_argument('--use_slurm', action="store_true", default=False)
    parser.add_argument('--skip_if_exists', action="store_true", default=False, help="Skip training a chunk if it already has a hierarchy")
    parser.add_argument('--keep_running', action="store_true", default=False, help="Keep running even if a chunk processing fails")
    args = parser.parse_args()
    
    
    model_params = model_params.extract(args)
    pipeline_params = pipeline_params.extract(args)
    
    
    
    print(args.extra_training_args)

    os_name = platform.system()
    f_path = Path(__file__)
    images_dir, depths_dir, masks_dir, colmap_dir, chunks_dir, output_dir = setup_dirs(
        args.images_dir, args.depths_dir,
        args.masks_dir, args.colmap_dir,
        args.chunks_dir, args.output_dir,
        args.project_dir
    )

    start_time = time.time()
    
    if not os.path.exists(output_dir):
        print(f"creating output dir: {output_dir}")
        os.makedirs(os.path.join(output_dir, "scaffold"))
        os.makedirs(os.path.join(output_dir, "trained_chunks"))

    slurm_args = ["sbatch"]

    ## First step is coarse optimization to generate a scaffold that will be used later.
    if args.skip_if_exists and os.path.exists(os.path.join(output_dir, "scaffold/point_cloud/iteration_30000/point_cloud.ply")):
        print("Skipping coarse")
    else:
        train_coarse_args =  " ".join([
            "python", "train_coarse.py",
            "-s", colmap_dir,
            "--save_iterations", "-1",
            "-i", images_dir,
            "--skybox_num", "100000",
            "--model_path", os.path.join(output_dir, "scaffold")
        ])
        if masks_dir != "":
            train_coarse_args += " --alpha_masks " + masks_dir
        if args.extra_training_args != "": 
            train_coarse_args += " " + args.extra_training_args

        try:
            optimization_params = OptimizationParams(parser)
            model_params.source_path = colmap_dir
            model_params.images = images_dir
            model_params.model_path = os.path.join(output_dir, "scaffold")
            model_params.skybox_num = 100000
            optimization_params.iterations = 30000
            train_coarse.training(model_params, optimization_params, pipeline_params, [30000], [], False, -1)
            #subprocess.run(train_coarse_args, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing train_coarse: {e}")
            sys.exit(1)
            
            
    model_params.source_path = colmap_dir #os.path.join(colmap_dir, "../rectified/")
    model_params.images = images_dir
    
    graph_path = os.path.join(colmap_dir, "consistency_graph.edge_list")
    if not os.path.isfile(graph_path) or True:
        if os.path.isfile(os.path.join(colmap_dir, "../unrectified/database.db")):   
            # Build consistency graph
            cg = consistency_graph.load_consistency_graph(os.path.join(colmap_dir, "../unrectified/"))
            # remove training images
            cg.remove_nodes_from([(i*10)+1 for i in range(0, len(cg.nodes())//10 + 2)])
        else:
            print("No COLMAP Database found for consistency graph!")
            cg = None
            
    model_params.source_path = colmap_dir #os.path.join(colmap_dir, "../rectified/")
    model_params.images = images_dir
    
        
    # Randomize Initialization
    #gaussians = GaussianModel(1)
    #gaussians.load_ply(os.path.join(output_dir, "scaffold/point_cloud/iteration_30000/point_cloud.ply"))
    #with torch.no_grad():
    #    gaussians._features_dc[:100000] = torch.rand_like(gaussians._features_dc[:100000]) * (torch.max(gaussians._features_dc[:100000]) - torch.min(gaussians._features_dc[:100000])) + torch.min(gaussians._features_dc[:100000])
    #    gaussians._scaling[:100000] = torch.rand_like(gaussians._scaling[:100000]) * (torch.max(gaussians._scaling[:100000]) - torch.min(gaussians._scaling[:100000])) + torch.min(gaussians._scaling[:100000])
    #    gaussians._opacity[:100000] = torch.rand_like(gaussians._opacity[:100000]) * (torch.max(gaussians._opacity[:100000]) - torch.min(gaussians._opacity[:100000])) + torch.min(gaussians._opacity[:100000])
#
    #    gaussians._features_dc[100000:] = torch.rand_like(gaussians._features_dc[100000:]) * (torch.max(gaussians._features_dc[100000:]) - torch.min(gaussians._features_dc[100000:])) + torch.min(gaussians._features_dc[100000:])
    #    gaussians._scaling[100000:] = torch.rand_like(gaussians._scaling[100000:]) * (torch.max(gaussians._scaling[100000:]) - torch.min(gaussians._scaling[100000:])) + torch.min(gaussians._scaling[100000:]) 
    #    gaussians._scaling[100000:] += 3
    #    gaussians._opacity[100000:] = torch.rand_like(gaussians._opacity[100000:]) * (torch.max(gaussians._opacity[100000:]) - torch.min(gaussians._opacity[100000:])) + torch.min(gaussians._opacity[100000:])
    #    gaussians._xyz[100000:] = torch.rand_like(gaussians._xyz[100000:]) * (torch.max(gaussians._xyz[100000:]) - torch.min(gaussians._xyz[100000:])) + torch.min(gaussians._xyz[100000:])
    #gaussians.inverse_opacity_activation = lambda x:x
    #gaussians.save_ply(os.path.join(output_dir, "scaffold/point_cloud/iteration_30000/point_cloud_random.ply"))
    # Randomize Initialization


        
    # ==================================== Scaffold finished ==============================
    hierarchy_creator_args = "submodules/gaussianhierarchy/build/Release/GaussianHierarchyCreator.exe " if os_name == "Windows" else "submodules/gaussianhierarchy/build/GaussianHierarchyCreator "
    hierarchy_creator_args = os.path.join(f_path.parent, hierarchy_creator_args)
    try:
        subprocess.run(
        hierarchy_creator_args + " ".join([
                os.path.join(output_dir, "scaffold/point_cloud/iteration_30000/point_cloud.ply"),
                os.path.join(output_dir, "/../camera_calibration/aligned"),
                os.path.join(output_dir, "scaffold/point_cloud/iteration_30000/")
                ,os.path.join(output_dir, "scaffold/point_cloud/iteration_30000/")
            ]),
            shell=True, check=True, text=True
        )
    except subprocess.CalledProcessError as e:
                print(f"Error executing hierarchy_creator: {e}")
                # TODO: WTF is happening here?
                if not args.keep_running and False:
                    sys.exit(1)
                    
    # ==================================== Hierarchy finished ==============================
    
    
    #gaussians = GaussianModel(1)
    model_params.hierarchy = os.path.join(output_dir, "scaffold/point_cloud/iteration_30000/", "hierarchy.dhier")
    model_params.model_path = output_dir
    model_params.scaffold_file = os.path.join(output_dir, "scaffold/point_cloud/iteration_30000/")
    #hierarchy_scene = Scene(model_params, gaussians, resolution_scales = [1], create_from_hier=True, shuffle=True)
    #print(f"Hierarchy bounding sphere divergence: {hierarchy_scene.gaussians.compute_bounding_sphere_divergence()}")
    
    #debug_utils.generate_some_flat_scene_images(hierarchy_scene, pipeline_params, output_dir, 4, indices=torch.cat((hierarchy_scene.gaussians.get_skybox_indices(), torch.where(hierarchy_scene.gaussians.nodes[:, 2] == 0)[0].cpu())))
    
    #hierarchy_scene.dump_gaussians("Dump", only_leaves=True)
    #debug_utils.render_depth_slices(hierarchy_scene, pipeline_params, output_dir)
    #debug_utils.render_level_slices(hierarchy_scene, pipeline_params, output_dir)
    #print(f"Number of hierarchy leaf nodes: {hierarchy_scene.gaussians.get_number_of_leaf_nodes()}")
    #print(f"Number of hierarchy nodes: {len(hierarchy_scene.gaussians._xyz)}")
    #debug_utils.generate_some_flat_scene_images(hierarchy_scene, pipeline_params, output_dir)

    #debug_utils.generate_some_hierarchy_scene_images_dynamic(hierarchy_scene, pipeline_params, output_dir, limit=0.0000001, no_images=3)
    
    optimization_params = OptimizationParams(parser)
    optimization_params = optimization_params.extract(args)
    optimization_params.position_lr_init=2e-05
    optimization_params.position_lr_final=2e-07
    optimization_params.position_lr_delay_mult=0.01
    optimization_params.position_lr_max_steps=30000
    optimization_params.feature_lr=0.0005
    optimization_params.opacity_lr=0.01
    optimization_params.scaling_lr=0.001
    optimization_params.rotation_lr=0.001
    optimization_params.exposure_lr_init=0.001
    optimization_params.exposure_lr_final=0.0001
    optimization_params.exposure_lr_delay_steps=5000
    optimization_params.exposure_lr_delay_mult=0.001
    optimization_params.percent_dense=0.001
    optimization_params.iterations=30000 
    optimization_params.lambda_dssim=0.2
    optimization_params.densification_interval=500
    optimization_params.opacity_reset_interval=3000
    optimization_params.densify_from_iter=1
    optimization_params.densify_until_iter=25000
    #optimization_params.densify_grad_threshold=0.015
    optimization_params.densify_grad_threshold=0.015
    optimization_params.depth_l1_weight_init=1.0
    optimization_params.depth_l1_weight_final=0.01

    
    #Standard 3DGS training parameters
    optimization_params.iterations = 50_000
    optimization_params.position_lr_init =  0.0000056 #0.0000016 #0.00016
    #     #optimization_params.position_lr_init = 0.016
    optimization_params.position_lr_final = 0.00000001 #0.000000016 #0.0000016
    optimization_params.position_lr_delay_mult = 0.01
    optimization_params.position_lr_max_steps = 50_000
    optimization_params.feature_lr = 0.0025
    optimization_params.opacity_lr = 0.025
    optimization_params.scaling_lr = 0.005
    optimization_params.rotation_lr = 0.001
    optimization_params.exposure_lr_init = 0.01
    optimization_params.exposure_lr_final = 0.001
    optimization_params.exposure_lr_delay_steps = 0
    optimization_params.exposure_lr_delay_mult = 0.0
    optimization_params.percent_dense = 0.01
    optimization_params.lambda_dssim = 0.2
    optimization_params.densification_interval = 100
    optimization_params.opacity_reset_interval = 3000
    optimization_params.densify_from_iter = 100
    optimization_params.densify_until_iter = 50_000
    optimization_params.densify_grad_threshold = 0.15
    optimization_params.depth_l1_weight_init = 1.0
    optimization_params.depth_l1_weight_final = 0.01


    train_post.training(model_params, optimization_params, pipeline_params, [], [], [], [], cg)
    
    exit()
    post_opt_chunk_args =  " ".join([
        "python", "-u train_post.py",
        "--iterations 15000", "--feature_lr 0.0005",
        "--opacity_lr 0.01", "--scaling_lr 0.001", "--save_iterations -1",
        f"-i {images_dir}",  f"--scaffold_file {output_dir}/scaffold/point_cloud/iteration_30000",
    ])
    # Post optimization of Scaffold
    print(f"post optimizing scaffold")
    try:
        subprocess.run(
            post_opt_chunk_args + " -s " + colmap_dir + 
            " --model_path " + os.path.join(output_dir, "scaffold/point_cloud/iteration_30000/") +
            " --hierarchy " + os.path.join(output_dir, "scaffold/point_cloud/iteration_30000/", "hierarchy.dhier") + " --save_iterations " + str(14000),
            shell=True, check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error executing train_post: {e}")
        if not args.keep_running:
            sys.exit(1) # TODO: log where it fails and don't add it to the consolidation and add a warning at the end

    # ================================== Hierarchy trained ====================================
    
    
    
    
    exit(666)






    subprocess.run(
                    post_opt_chunk_args + " -s " + source_chunk + 
                    " --model_path " + trained_chunk +
                    " --hierarchy " + os.path.join(trained_chunk, "hierarchy.hier"),
                    shell=True, check=True
                )





    ## Now we can train each chunks using the scaffold previously created
    train_chunk_args =  " ".join([
        "python", "-u train_single.py",
        "--save_iterations -1",
        f"-i {images_dir}", f"-d {depths_dir}",
        f"--scaffold_file {output_dir}/scaffold/point_cloud/iteration_30000",
        "--skybox_locked" 
    ])
    if masks_dir != "":
        train_chunk_args += " --alpha_masks " + masks_dir
    if args.extra_training_args != "": 
        train_chunk_args += " " + args.extra_training_args

    hierarchy_creator_args = "submodules/gaussianhierarchy/build/Release/GaussianHierarchyCreator.exe " if os_name == "Windows" else "submodules/gaussianhierarchy/build/GaussianHierarchyCreator "
    hierarchy_creator_args = os.path.join(f_path.parent.parent, hierarchy_creator_args)

    post_opt_chunk_args =  " ".join([
        "python", "-u train_post.py",
        "--iterations 15000", "--feature_lr 0.0005",
        "--opacity_lr 0.01", "--scaling_lr 0.001", "--save_iterations -1",
        f"-i {images_dir}",  f"--scaffold_file {output_dir}/scaffold/point_cloud/iteration_30000",
    ])
    if masks_dir != "":
        post_opt_chunk_args += " --alpha_masks " + masks_dir
    if args.extra_training_args != "": 
        post_opt_chunk_args += " " + args.extra_training_args

    
    chunk_names = os.listdir(chunks_dir)
    for chunk_name in chunk_names:
        source_chunk = os.path.join(chunks_dir, chunk_name)
        trained_chunk = os.path.join(output_dir, "trained_chunks", chunk_name)

        if args.skip_if_exists and os.path.exists(os.path.join(trained_chunk, "hierarchy.hier_opt")):
            print(f"Skipping {chunk_name}")
        else:
            ## Training can be done in parallel using slurm.
            
            print(f"Training chunk {chunk_name}")
            try:
                subprocess.run(
                    train_chunk_args + " -s "+ source_chunk + 
                    " --model_path " + trained_chunk +
                    " --bounds_file "+ source_chunk,
                    shell=True, check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Error executing train_single: {e}")
                if not args.keep_running:
                    sys.exit(1)

            # Generate a hierarchy within each chunks
            print(f"Generating hierarchy for chunk {chunk_name}")
            try:
                subprocess.run(
                hierarchy_creator_args + " ".join([
                        os.path.join(trained_chunk, "point_cloud/iteration_30000/point_cloud.ply"),
                        source_chunk,
                        trained_chunk,
                        os.path.join(output_dir, "scaffold/point_cloud/iteration_30000")
                    ]),
                    shell=True, check=True, text=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Error executing hierarchy_creator: {e}")
                if not args.keep_running:
                    sys.exit(1)

            # Post optimization on each chunks
            print(f"post optimizing chunk {chunk_name}")
            try:
                subprocess.run(
                    post_opt_chunk_args + " -s "+ source_chunk + 
                    " --model_path " + trained_chunk +
                    " --hierarchy " + os.path.join(trained_chunk, "hierarchy.hier"),
                    shell=True, check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Error executing train_post: {e}")
                if not args.keep_running:
                    sys.exit(1) # TODO: log where it fails and don't add it to the consolidation and add a warning at the end

    if args.use_slurm:
        # Check every 10 sec all the jobs status
        all_finished = False
        all_status = []
        last_count = 0
        print(f"Waiting for chunks to be trained in parallel ...")

        while not all_finished:
            # print("Checking status of all jobs...")
            all_status = [is_job_finished(id) for id in submitted_jobs_ids if is_job_finished(id) != ""]
            if last_count != all_status.count("COMPLETED"):
                last_count = all_status.count("COMPLETED")
                print(f"processed [{last_count} / {len(chunk_names)} chunks].")

            all_finished = len(all_status) == len(submitted_jobs_ids)
    
            if not all_finished:
                time.sleep(10)  # Wait before checking again
        
        if not all(status == "COMPLETED" for status in all_status):
            print("At least one job failed or was cancelled, check at error logs.")

    end_time = time.time()
    print(f"Successfully trained in {(end_time - start_time)/60.0} minutes.")

    ## Consolidation to create final hierarchy
    hierarchy_merger_path = "submodules/gaussianhierarchy/build/Release/GaussianHierarchyMerger.exe" if os_name == "Windows" else "submodules/gaussianhierarchy/build/GaussianHierarchyMerger"
    hierarchy_merger_path = os.path.join(f_path.parent.parent, hierarchy_merger_path)

    consolidation_args = [
        hierarchy_merger_path, f"{output_dir}/trained_chunks",
        "0", chunks_dir, f"{output_dir}/merged.hier" 
    ]
  
    consolidation_args = consolidation_args + chunk_names
    print(f"Consolidation...")
    if args.use_slurm:
        consolidation = submit_job(slurm_args + [
                f"--error={output_dir}/consolidation_log.err", f"--output={output_dir}/consolidation_log.out",
                "consolidate.slurm"] + consolidation_args)        

        while is_job_finished(consolidation) == "":
            time.sleep(10)
    else:
        try:
            subprocess.run(consolidation_args, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing consolidation: {e}")
            sys.exit(1)

    end_time = time.time()
    print(f"Total time elapsed for training and consolidation {(end_time - start_time)/60.0} minutes.")
