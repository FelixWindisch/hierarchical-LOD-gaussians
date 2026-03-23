# A LoD of Gaussians

This repository contains the official authors' implementation associated with the paper "A LoD of Gaussians: Unified Training and Rendering for Ultra-Large-Scale Reconstruction with External Memory". 
## Setup

Make sure to clone the repo using `--recursive`:
```
git clone -b Refactor https://github.com/FelixWindisch/hierarchical-LOD-gaussians.git --recursive
cd hierarchical-LOD-gaussians
```
### Prerequisite

Setting up the conda environment:
```
conda create -n LOD
conda activate LOD
conda install python=3.10
conda install -c nvidia cuda-toolkit=12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```
Install the last 5 dependencies with --no-build-isolation if you get errors.

### Compiling hierarchy generator and merger
These files were adapted from Hierarchical 3DGS and can be built as follows:
```
cd submodules/gaussianhierarchy
cmake . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --config Release
cd ../..
```
## Running the method

#### Dataset 
To get started, prepare a dataset. We follow the structure from Hierarchical 3DGS, which means all images should be in root/camera_calibration/rectified/images folder and colmap camera poses (cameras, images, points3d) should be in root/camera_calibration/aligned/sparse/0.
If depth images or masks are used, place them in root/camera_calibration/rectified/depths and root/camera_calibration/rectified/masks respectively.
You can then start training by 
```
python train.py --project-dir root --config default.json --skip_if_exists
```
The training will output a .dhier file, which can be rendered and evaluated:
```
python eval_hierarchy.py --hierarchy_path /path/to/result.dhier_opt -s root/camera_calibration/aligned -i root/camera_calibration/rectified/images --config default.json
python hierarchy_viewer.py --hierarchy_path /path/to/result.dhier_opt -s root/camera_calibration/aligned  --config default.json
```
```eval_hierarchy``` will render all images in the test set (use the llffhold in your config parameter to designate every nth image for testing) and output quality metrics.
```hierarchy_viewer``` allows interactive viewing of the results. This can be done using the networked inria viewer, but we recommend installing SplatViz (https://github.com/Florian-Barthel/splatviz) and running it with ```python run_main.py --mode=attach``` while ```hierarchy_viewer``` is running.

