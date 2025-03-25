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

import torch
from torch import nn
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import alt_gaussian_rasterization
#import stp_gaussian_rasterization
#from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from diff_gaussian_rasterization import _C
import numpy as np


def occlusion_cull(indices, gaussians, camera, pipe, background):
    means3D = gaussians._xyz[indices].cuda().contiguous()
    opacity = gaussians.opacity_activation(gaussians._opacity[indices].cuda().contiguous())
    scales = gaussians.scaling_activation(gaussians._scaling[indices].cuda().contiguous())
    rotations = gaussians.rotation_activation(gaussians._rotation[indices].cuda().contiguous())
    features_dc = gaussians._features_dc[indices].cuda().contiguous()
    features_rest = gaussians._features_rest[indices].cuda().contiguous()
    shs = torch.cat((features_dc, features_rest), dim=1).contiguous()
    
    return render_on_disk(camera, means3D, opacity, scales, rotations, shs, pipe, background)["seen"].to(torch.bool).cpu()

def render(
        viewpoint_camera, pc, 
        pipe, 
        bg_color : torch.Tensor, 
        scaling_modifier = 1.0, 
        override_color = None, 
        indices = None, 
        use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    render_indices = torch.empty(0).int().cuda()
    parent_indices = torch.empty(0).int().cuda()
    interpolation_weights = torch.empty(0).float().cuda()
    num_siblings = torch.empty(0).int().cuda()

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        # This is false for render_coarse
        do_depth=True,
        render_indices=render_indices,
        parent_indices=parent_indices,
        interpolation_weights=interpolation_weights,
        num_node_kids=num_siblings
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    
    if indices is not None:
        means3D = means3D[indices].contiguous()
        means2D = means2D[indices].contiguous()
        shs = shs[indices].contiguous()
        opacity = opacity[indices].contiguous()
        scales = scales[indices].contiguous()
        rotations = rotations[indices].contiguous() 

    rendered_image, radii, depth_image = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    
    # This is missing in render_coarse
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]
    rendered_image = rendered_image.clamp(0, 1)
    # This is missing in render_coarse


    subfilter = radii > 0
    if indices is not None:
        vis_filter = torch.zeros(pc._xyz.size(0), dtype=bool, device="cuda")
        w = vis_filter[indices]
        w[subfilter] = True
        vis_filter[indices] = w
    else:
        vis_filter = subfilter

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth" : depth_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : vis_filter.nonzero().flatten().long(),
            "radii": radii[subfilter]}

def render_on_disk(
    viewpoint_camera, 
        means3D,
        opacity,
        scales, 
        rotations,
        shs,
        pipe, 
        bg_color : torch.Tensor, 
        scaling_modifier = 1.0, 
        override_color = None,
        sh_degree = 3
        ):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    screenspace_points = torch.zeros_like(means3D, dtype=torch.float32, requires_grad=False, device="cuda") + 0
    means2D = nn.Parameter(screenspace_points)

    render_indices = torch.empty(0).int().cuda()
    parent_indices = torch.empty(0).int().cuda()
    interpolation_weights = torch.empty(0).float().cuda()
    num_node_siblings = torch.empty(0).int().cuda()
    colors_precomp = None
    cov3D_precomp = None
    
    pipe.debug = True
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier= scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=int(sh_degree),
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        render_indices=render_indices,
        parent_indices=parent_indices,
        interpolation_weights=interpolation_weights,
        num_node_kids=num_node_siblings,
        do_depth=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    

    
    rendered_image, seen, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    rendered_image = rendered_image.clamp(0, 1)
    #radii = radii[100000:]
    #vis_filter = radii > 0

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            #"visibility_filter" : vis_filter,
            #"radii": radii[vis_filter]
            "seen" : seen}




# render with hierarchy, interpolate Gaussians with their parent nodes beforehand
def render_post(
        viewpoint_camera, 
        gaussians, 
        pipe, 
        bg_color : torch.Tensor, 
        scaling_modifier = 1.0, 
        override_color = None, 
        render_indices = torch.Tensor([]).int(),
        parent_indices = torch.Tensor([]).int(),
        interpolation_weights = torch.Tensor([]).float(),
        # number of siblings
        num_node_siblings = torch.Tensor([]).int(),
        interp_python = True,
        on_disk = True,
        use_trained_exp = False, iteration=0):
    """
    Render the scene from a hierarchy.  
    
    Background tensor (bg_color) must be on GPU!
    """
    
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        #print("screenspace_points do not retain grad???")
        pass
    
    means3D = gaussians.get_xyz
    means2D = screenspace_points
    opacity = gaussians.get_opacity
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = gaussians.get_covariance(scaling_modifier)
    else:
        scales = gaussians.get_scaling
        rotations = gaussians.get_rotation
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
            dir_pp = (gaussians.get_xyz - viewpoint_camera.camera_center.repeat(gaussians.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = gaussians.get_features
    else:
        # Technically, we don't need the SHs if we precompute color, maybe remove?
        #shs = pc.get_features
        colors_precomp = override_color
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if render_indices.size(0) != 0:
        render_inds = render_indices.long()
        if interp_python:
            num_entries = render_indices.size(0)
            interps = interpolation_weights[:num_entries].unsqueeze(1)
            interps_inv = (1 - interpolation_weights[:num_entries]).unsqueeze(1)
            parent_inds = parent_indices[:num_entries].long()
            means3D_base = (interps * means3D[render_inds] + interps_inv * means3D[parent_inds]).contiguous()
            scales_base = (interps * scales[render_inds] + interps_inv * scales[parent_inds]).contiguous()
            if override_color is None:
                shs_base = (interps.unsqueeze(2) * shs[render_inds] + interps_inv.unsqueeze(2) * shs[parent_inds]).contiguous()
            parents = rotations[parent_inds]
            rots = rotations[render_inds]
            dots = torch.bmm(rots.unsqueeze(1), parents.unsqueeze(2)).flatten()
            parents[dots < 0] *= -1
            rotations_base = ((interps * rots) + interps_inv * parents).contiguous()
            opacity_base = (interps * opacity[render_inds] + interps_inv * opacity[parent_inds]).contiguous()
            if gaussians.skybox_points == 0:
                skybox_inds = torch.Tensor([]).long()
            else:
                # The end index is fucking inclusive
                skybox_inds = torch.range(0, gaussians.skybox_points-1, device="cuda").long()
                # skybox_inds = torch.range(pc._xyz.size(0) - pc.skybox_points, pc._xyz.size(0)-1, device="cuda").long()
            means3D = torch.cat((means3D[skybox_inds], means3D_base)).contiguous() 
            if override_color is None: 
                shs = torch.cat((shs[skybox_inds], shs_base)).contiguous() 
            opacity = torch.cat((opacity[skybox_inds], opacity_base )).contiguous()  
            rotations = torch.cat((rotations[skybox_inds], rotations_base )).contiguous()    
            means2D = means2D[:(gaussians.skybox_points + num_entries)].contiguous()     
            scales = torch.cat((scales[skybox_inds], scales_base)).contiguous()  
            interpolation_weights = interpolation_weights.clone().detach()
            # Skybox Points are not interpolated
            interpolation_weights = torch.cat((torch.ones(gaussians.skybox_points, device='cuda', dtype=torch.int32), interpolation_weights[:-gaussians.skybox_points]))
            num_node_siblings = torch.cat((torch.ones(gaussians.skybox_points, device='cuda', dtype=torch.int32), num_node_siblings[:-gaussians.skybox_points]))
            #interpolation_weights[0:pc.skybox_points] = 1.0 
            #num_node_siblings[0:pc.skybox_points] = 1 
        else:
            means3D = means3D[render_inds].contiguous()
            means2D = means2D[render_inds].contiguous()
            if override_color is None: 
                shs = shs[render_inds].contiguous()
            opacity = opacity[render_inds].contiguous()
            scales = scales[render_inds].contiguous()
            rotations = rotations[render_inds].contiguous() 
        render_indices = torch.Tensor([]).int()
        parent_indices = torch.Tensor([]).int()
        interpolation_weights = torch.Tensor([]).float()
    pipe.debug = True
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        render_indices=render_indices,
        parent_indices=parent_indices,
        interpolation_weights=interpolation_weights,
        num_node_kids=num_node_siblings,
        do_depth=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    
    
    rendered_image, radii, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    #if use_trained_exp and pc.pretrained_exposures:
    #    try:
    #        exposure = pc.pretrained_exposures[viewpoint_camera.image_name]
    #        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]
    #    except Exception as e:
    #        print(f"Exposures should be optimized in single. Missing exposure for image {viewpoint_camera.image_name}")
    rendered_image = rendered_image.clamp(0, 1)
    radii = radii[gaussians.skybox_points:]
    vis_filter = radii > 0

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : vis_filter,
            "radii": radii[vis_filter]}


# Exactly like render, but without the option to use exposure and without returning depth data (depth regularization is only used in chunk training)
def render_coarse(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, zfar=0.0, override_color = None, indices = None):
    """
    Render the scene for the coarse optimization. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    render_indices = torch.empty(0).int().cuda()
    parent_indices = torch.empty(0).int().cuda()
    interpolation_weights = torch.empty(0).float().cuda()
    num_siblings = torch.empty(0).int().cuda()

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=True,
        do_depth=False,
        render_indices=render_indices,
        parent_indices=parent_indices,
        interpolation_weights=interpolation_weights,
        num_node_kids=num_siblings
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if indices is not None:
        means3D = means3D[indices].contiguous()
        means2D = means2D[indices].contiguous()
        shs = shs[indices].contiguous()
        opacity = (opacity[indices].contiguous())
        scales = scales[indices].contiguous()
        rotations = rotations[indices].contiguous() 

    rendered_image, radii, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    

    
    subfilter = radii > 0
    if indices is not None:
        vis_filter = torch.zeros(pc._xyz.size(0), dtype=bool, device="cuda")
        w = vis_filter[indices]
        w[subfilter] = True
        vis_filter[indices] = w
    else:
        vis_filter = subfilter

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : vis_filter,
            "radii": radii[subfilter]}


def render_stp(viewpoint_camera,  
        means3D,
        opacity,
        scales, 
        rotations,
        shs, pipe, bg_color : torch.Tensor, splat_args = None, sh_degree=3, scaling_modifier = 1.0, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    #screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        inv_viewprojmatrix=viewpoint_camera.full_proj_transform_inverse,
        sh_degree=sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=True,
        render_depth=False,
        settings=splat_args)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    size = len(means3D)
    #means3D = torch.ones((size,3), device='cuda')
    #opacity = torch.ones(size, device='cuda')
    #scales = torch.full((size,3), 0.001, device='cuda')
    #rotations = torch.zeros((size,4), device='cuda')
    #rotations[:, 3] = 1
    #shs = shs.abs()
    #shs = torch.zeros_like(shs)

    
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    means2D = (screenspace_points)
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    
    colors_precomp = None
    #override_color = torch.ones((len(means3D), 3), device='cuda', dtype=torch.float32)
    #if not override_color is None:
    #    shs = None
    #    colors_precomp = override_color
    #if True or pipe.compute_cov3D_python:
    #    cov3D_precomp = gaussians.covariance_activation(scales, 1.0, rotations)
        
    #if True or pipe.convert_SHs_python:
    #    shs_view = shs.transpose(1, 2).view(-1, 3, (3+1)**2)
    #    dir_pp = (means3D - viewpoint_camera.camera_center.repeat(shs.shape[0], 1))
    #    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    #    sh2rgb = eval_sh(sh_degree, shs_view, dir_pp_normalized)
    #    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    number_to_render = 10000
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D[:number_to_render],
        means2D = means2D[:number_to_render],
        shs = shs[:number_to_render],
        colors_precomp = None, # colors_precomp[:number_to_render],
        opacities = (opacity)[:number_to_render],
        scales = scales[:number_to_render],
        rotations = rotations[:number_to_render],
        cov3D_precomp = None)
    print("DONE RENDERING!")
    # Apply exposure to rendered image (training only)
    #if use_trained_exp:
    #    exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
    #    rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points
        }
    
    return out


def render_vanilla(viewpoint_camera, 
        means3D,
        opacity,
        scales, 
        rotations,
        shs, pipe, bg_color : torch.Tensor, sh_degree=3, scaling_modifier = 1.0, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    #screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = alt_gaussian_rasterization.GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False    )

    rasterizer = alt_gaussian_rasterization.GaussianRasterizer(raster_settings=raster_settings)
    size = len(means3D)
    #means3D = torch.ones((size,3), device='cuda')
    #opacity = torch.ones(size, device='cuda')
    #scales = torch.ones((size,3), device='cuda')
    #rotations = torch.zeros((size,4), device='cuda')
    #rotations[:, 0] = 1
    #shs = torch.ones((size, 16, 3))
    
    
    screenspace_points = torch.zeros_like(means3D, dtype=torch.float32, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    means2D = (screenspace_points)
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    
    #if True or pipe.convert_SHs_python:
    #    shs_view = shs.transpose(1, 2).view(-1, 3, (3+1)**2)
    #    dir_pp = (means3D - viewpoint_camera.camera_center.repeat(shs.shape[0], 1))
    #    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    #    sh2rgb = eval_sh(sh_degree, shs_view, dir_pp_normalized)
    #    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs, # shs,
        colors_precomp = None, #colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None)
    #print("DONE RENDERING!")
    # Apply exposure to rendered image (training only)
    #if use_trained_exp:
    #    exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
    #    rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points
        }
    
    return out