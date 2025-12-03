# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from slang_gaussian_rasterization.internal.render_grid import RenderGrid
import slang_gaussian_rasterization.internal.slang.slang_modules as slang_modules
from slang_gaussian_rasterization.internal.tile_shader_slang import vertex_and_tile_shader

def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook

def render_alpha_blend_tiles_slang_raw(xyz_ws, rotations, scales, opacity, 
                                       sh_coeffs, beta, active_sh,
                                       world_view_transform, proj_mat, cam_pos,
                                       fovy, fovx, height, width, tile_size=16):
    
    render_grid = RenderGrid(height,
                             width,
                             tile_height=tile_size,
                             tile_width=tile_size)
    sorted_gauss_idx, tile_ranges, radii, xyz_vs, inv_cov_vs, rgb = vertex_and_tile_shader(xyz_ws,
                                                                                           rotations,
                                                                                           scales,
                                                                                           sh_coeffs,
                                                                                           active_sh,
                                                                                           world_view_transform,
                                                                                           proj_mat,
                                                                                           cam_pos,
                                                                                           fovy,
                                                                                           fovx,
                                                                                           render_grid)
   
    # retain_grad fails if called with torch.no_grad() under evaluation
    try:
        xyz_vs.retain_grad()
    except:
        pass

    image_rgb, contribution = AlphaBlendTiledRender.apply(
        sorted_gauss_idx,
        tile_ranges,
        xyz_vs,
        inv_cov_vs,
        opacity,
        rgb,
        beta,
        render_grid)
    
    render_pkg = {
        'render': image_rgb.permute(2,0,1)[:3, ...],
        'viewspace_points': xyz_vs,
        'visibility_filter': radii > 0,
        'radii': radii,
        'contribution': contribution
    }

    return render_pkg


class AlphaBlendTiledRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                sorted_gauss_idx, tile_ranges,
                xyz_vs, inv_cov_vs, opacity, rgb, beta, render_grid, device="cuda"):
        output_img = torch.zeros((render_grid.image_height, 
                                  render_grid.image_width, 4), 
                                 device=device)
        
        contribution = torch.zeros((xyz_vs.shape[0]), device=device)
        d_contribution = torch.zeros((xyz_vs.shape[0]), device=device)
        n_contributors = torch.zeros((render_grid.image_height, 
                                      render_grid.image_width, 1),
                                     dtype=torch.int32, device=device)

        assert (render_grid.tile_height, render_grid.tile_width) in slang_modules.alpha_blend_shaders, (
            'Alpha Blend Shader was not compiled for this tile'
            f' {render_grid.tile_height}x{render_grid.tile_width} configuration, available configurations:'
            f' {slang_modules.alpha_blend_shaders.keys()}'
        )

        alpha_blend_tile_shader = slang_modules.alpha_blend_shaders[(render_grid.tile_height, render_grid.tile_width)]
        splat_kernel_with_args = alpha_blend_tile_shader.splat_tiled(
            sorted_gauss_idx=sorted_gauss_idx,
            tile_ranges=tile_ranges,
            xyz_vs=xyz_vs, inv_cov_vs=inv_cov_vs, 
            opacity=opacity, rgb=rgb, 
            beta=beta,
            output_img=output_img,
            contribution=contribution,
            d_contribution=d_contribution,
            n_contributors=n_contributors,
            grid_height=render_grid.grid_height,
            grid_width=render_grid.grid_width,
            tile_height=render_grid.tile_height,
            tile_width=render_grid.tile_width
        )
        splat_kernel_with_args.launchRaw(
            blockSize=(render_grid.tile_width, 
                       render_grid.tile_height, 1),
            gridSize=(render_grid.grid_width, 
                      render_grid.grid_height, 1)
        )

        ctx.save_for_backward(sorted_gauss_idx, tile_ranges,
                              xyz_vs, inv_cov_vs, opacity, rgb, 
                              output_img, n_contributors, contribution)
        ctx.render_grid = render_grid

        
        return output_img, contribution

    @staticmethod
    def backward(ctx, grad_output_img, grad_contribution):
        (sorted_gauss_idx, tile_ranges, 
         xyz_vs, inv_cov_vs, opacity, rgb, 
         output_img, n_contributors, contribution) = ctx.saved_tensors
        render_grid = ctx.render_grid

        xyz_vs_grad = torch.zeros_like(xyz_vs)
        inv_cov_vs_grad = torch.zeros_like(inv_cov_vs)
        opacity_grad = torch.zeros_like(opacity)
        rgb_grad = torch.zeros_like(rgb)
        beta = torch.zeros((xyz_vs.shape[0]), device=xyz_vs.device)
        beta_grad = torch.zeros_like(beta)
        #breakpoint()

        assert (render_grid.tile_height, render_grid.tile_width) in slang_modules.alpha_blend_shaders, (
            'Alpha Blend Shader was not compiled for this tile'
            f' {render_grid.tile_height}x{render_grid.tile_width} configuration, available configurations:'
            f' {slang_modules.alpha_blend_shaders.keys()}'
        )

        alpha_blend_tile_shader = slang_modules.alpha_blend_shaders[(render_grid.tile_height, render_grid.tile_width)]

        kernel_with_args = alpha_blend_tile_shader.splat_tiled.bwd(
            sorted_gauss_idx=sorted_gauss_idx,
            tile_ranges=tile_ranges,
            xyz_vs=(xyz_vs, xyz_vs_grad),
            inv_cov_vs=(inv_cov_vs, inv_cov_vs_grad),
            opacity=(opacity, opacity_grad),
            rgb=(rgb, rgb_grad),
            beta=(beta, beta_grad),
            output_img=(output_img, grad_output_img),
            contribution=contribution.contiguous(),# grad_contribution.contiguous()),
            d_contribution=grad_contribution.contiguous(),# grad_contribution.contiguous()),
            n_contributors=n_contributors,
            grid_height=render_grid.grid_height,
            grid_width=render_grid.grid_width,
            tile_height=render_grid.tile_height,
            tile_width=render_grid.tile_width)
        
        kernel_with_args.launchRaw(
            blockSize=(render_grid.tile_width, 
                       render_grid.tile_height, 1),
            gridSize=(render_grid.grid_width, 
                      render_grid.grid_height, 1)
        )
        
        return None, None, xyz_vs_grad, inv_cov_vs_grad, opacity_grad, rgb_grad, None, None
