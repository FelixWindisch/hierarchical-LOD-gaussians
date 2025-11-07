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
python train.py --project-dir root --config default.json
```
The training will output a .dhier file, which can be rendered and evaluated:
```
python eval_hierarchy.py --project-dir root --config default.json
python render_hierarchy.py --project-dir root --config default.json
```
