
Write_Tensor_Board = True
#Standard
densify_interval = 1000
densify_radii = True
lr_multiplier = 1
#Unused
Random_Hierarchy_Cut = True
Only_Noise_Visible = True
#MCMC
Max_Cap = 3_000_000
MCMC_Densification = True
MCMC_Noise_LR = 100  #5e5
lambda_scaling = 0
lambda_opacity = 0.01
#Hierarchical
Gaussian_Interpolation = False
# Upward Propagation D
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
SPT_Root_Volume = 100 # 0.02
Target_Granularity_Pixels = 4

Min_SPT_Size = 256
Cache_SPTs = True

Reuse_SPT_Tolerance_Closer = 2
Reuse_SPT_Tolerance_Farther = 2 

Max_Gaussian_Budget = 70_000_000
Distance_Multiplier_Until_Budget = 1.5

Cache_Size = 15_000_000
Cache_Size_After_Reduction = 12_000_000

#View Selection
Use_Consistency_Graph = False
# Rasterizer
Rasterizer = "Vanilla"
Anti_Aliasing = True
# Optimizer
Global_ADAM = False

non_blocking=False

Max_SH_Degree = 1




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
            "Target_Granularity_Pixels": Target_Granularity_Pixels,
            "Cache_SPTs": Cache_SPTs,
            "MCMC_Noise_LR": MCMC_Noise_LR,
            "Use_Bounding_Spheres": Use_Bounding_Spheres,
            "Use_Consistency_Graph" : Use_Consistency_Graph,
            "Use_Frustum_Culling" : Use_Frustum_Culling,
            "Use_Occlusion_Culling" : Use_Occlusion_Culling,
            "lambda_scaling" : lambda_scaling,
            "lambda_opacity" : lambda_opacity,
            "Resterizer" : Rasterizer
        }