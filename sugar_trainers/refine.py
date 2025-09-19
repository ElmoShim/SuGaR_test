import os
import numpy as np
import torch
import open3d as o3d
from pytorch3d.loss import mesh_normal_consistency
from sugar_scene.gs_model import GaussianSplattingWrapper
from sugar_scene.sugar_model import SuGaR, convert_refined_sugar_into_gaussians
from sugar_scene.sugar_optimizer import OptimizationParams, SuGaROptimizer
from sugar_utils.loss_utils import ssim, l1_loss

from rich.console import Console
import time


def refined_training(args):
    CONSOLE = Console(width=120)

    # ====================Parameters====================
    num_device = args.gpu        

    # -----Model parameters-----    
    n_skip_images_for_eval_split = 8

    sh_levels = 4  

    # Learning rates and scheduling

    
    position_lr_init=0.00016
    position_lr_final=0.0000016
    position_lr_delay_mult=0.01
    position_lr_max_steps=30_000
    feature_lr=0.0025
    opacity_lr=0.05
    scaling_lr=0.005
    rotation_lr=0.001
        
    # Data processing and batching
    train_num_images_per_batch = 1  # 1 for full images

    # Loss functions
    loss_function = 'l1+dssim'  # 'l1' or 'l2' or 'l1+dssim'
    dssim_factor = 0.2

    # n_gaussians_per_surface_triangle=6  # 1, 3, 4 or 6    
    
    sh_warmup_every = 1000
    current_sh_levels = 1        

    # -----Log and save-----
    print_loss_every_n_iterations = 200
    save_model_every_n_iterations = 1_000_000 # 500, 1_000_000  # TODO
    # save_milestones = [2000, 7_000, 15_000]
    save_milestones = []

    # ====================End of parameters====================

    if args.output_dir is None:
        if len(args.scene_path.split("/")[-1]) > 0:
            args.output_dir = os.path.join("./output/refined", args.scene_path.split("/")[-1])
        else:
            args.output_dir = os.path.join("./output/refined", args.scene_path.split("/")[-2])
            
    source_path = args.scene_path
    gs_checkpoint_path = args.checkpoint_path
    surface_mesh_to_bind_path = args.mesh_path
    mesh_name = surface_mesh_to_bind_path.split("/")[-1].split(".")[0]
    iteration_to_load = 7000   
    
    surface_mesh_normal_consistency_factor = args.normal_consistency_factor    
    n_gaussians_per_surface_triangle = args.gaussians_per_triangle
    n_vertices_in_fg = args.n_vertices_in_fg
    num_iterations = args.refinement_iterations
    
    sugar_checkpoint_path = 'sugarfine_' + mesh_name.replace('sugarmesh_', '') + '_normalconsistencyXX_gaussperfaceYY/'
    sugar_checkpoint_path = os.path.join(args.output_dir, sugar_checkpoint_path)
    sugar_checkpoint_path = sugar_checkpoint_path.replace(
    'XX', str(surface_mesh_normal_consistency_factor).replace('.', '')
        ).replace(
        'YY', str(n_gaussians_per_surface_triangle).replace('.', '')
    )
    
    
    use_white_background = args.white_background
    
    export_ply_at_the_end = args.export_ply    
    
    CONSOLE.print("-----Parsed parameters-----")
    CONSOLE.print("Source path:", source_path)
    CONSOLE.print("   > Content:", len(os.listdir(source_path)))
    CONSOLE.print("Gaussian Splatting checkpoint path:", gs_checkpoint_path)
    CONSOLE.print("   > Content:", len(os.listdir(gs_checkpoint_path)))
    CONSOLE.print("SUGAR checkpoint path:", sugar_checkpoint_path)
    CONSOLE.print("Surface mesh to bind to:", surface_mesh_to_bind_path)
    CONSOLE.print("Iteration to load:", iteration_to_load)
    CONSOLE.print("Normal consistency factor:", surface_mesh_normal_consistency_factor)
    CONSOLE.print("Number of gaussians per surface triangle:", n_gaussians_per_surface_triangle)
    CONSOLE.print("Number of vertices in the foreground:", n_vertices_in_fg)    
    CONSOLE.print("Use white background:", use_white_background)
    CONSOLE.print("Export ply at the end:", export_ply_at_the_end)
    CONSOLE.print("----------------------------")
    
    # Setup device
    torch.cuda.set_device(num_device)
    CONSOLE.print("Using device:", num_device)
    device = torch.device(f'cuda:{num_device}')
    CONSOLE.print(torch.cuda.memory_summary())
    
    torch.autograd.set_detect_anomaly(False)
    
    # Creates save directory if it does not exist
    os.makedirs(sugar_checkpoint_path, exist_ok=True)
        

    # Load Gaussian Splatting checkpoint 
    CONSOLE.print(f"\nLoading config {gs_checkpoint_path}...")
    
    nerfmodel = GaussianSplattingWrapper(
        source_path=source_path,
        output_path=gs_checkpoint_path,
        iteration_to_load=iteration_to_load,
        load_gt_images=True,
        eval_split=False,
        eval_split_interval=n_skip_images_for_eval_split,
        white_background=use_white_background,
        )

    CONSOLE.print(f'{len(nerfmodel.training_cameras)} training images detected.')
    CONSOLE.print(f'The model has been trained for {iteration_to_load} steps.')    
    
    CONSOLE.print(f'\nCamera resolution scaled to '
          f'{nerfmodel.training_cameras.gs_cameras[0].image_height} x '
          f'{nerfmodel.training_cameras.gs_cameras[0].image_width}'
          )

    # Point cloud            
    points = torch.randn(1000, 3, device=nerfmodel.device)
    colors = torch.rand(1000, 3, device=nerfmodel.device)        
            
        
    # surface_mesh_to_bind_full_path = os.path.join('./results/meshes/', surface_mesh_to_bind_path)
    surface_mesh_to_bind_full_path = surface_mesh_to_bind_path
    CONSOLE.print(f'\nLoading mesh to bind to: {surface_mesh_to_bind_full_path}...')
    o3d_mesh = o3d.io.read_triangle_mesh(surface_mesh_to_bind_full_path)
    CONSOLE.print("Mesh to bind to loaded.")

        
    # Background tensor if needed
    if use_white_background:
        bg_tensor = torch.ones(3, dtype=torch.float, device=nerfmodel.device)
    else:
        bg_tensor = torch.zeros(3, dtype=torch.float, device=nerfmodel.device)
    
    # ====================Initialize SuGaR model====================
    # Construct SuGaR model
    sugar = SuGaR(
        nerfmodel=nerfmodel,
        points=points, #nerfmodel.gaussians.get_xyz.data,
        colors=colors, #0.5 + _C0 * nerfmodel.gaussians.get_features.data[:, 0, :],
        initialize=True,
        sh_levels=sh_levels,
        learnable_positions=True,
        triangle_scale=1.0,
        keep_track_of_knn=False,
        knn_to_track=0,
        beta_mode=None,
        freeze_gaussians=False,
        surface_mesh_to_bind=o3d_mesh,
        surface_mesh_thickness=None,
        learn_surface_mesh_positions=True,
        learn_surface_mesh_opacity=True,
        learn_surface_mesh_scales=True,
        n_gaussians_per_surface_triangle=n_gaussians_per_surface_triangle,
    )
        
        
    CONSOLE.print(f'\nSuGaR model has been initialized.')
    CONSOLE.print(sugar)
    CONSOLE.print(f'Number of parameters: {sum(p.numel() for p in sugar.parameters() if p.requires_grad)}')
    CONSOLE.print(f'Checkpoints will be saved in {sugar_checkpoint_path}')
    
    CONSOLE.print("\nModel parameters:")
 
    torch.cuda.empty_cache()
    
    # Compute scene extent
    cameras_spatial_extent = sugar.get_cameras_spatial_extent()
    
    
    # ====================Initialize optimizer====================    
    bbox_radius = cameras_spatial_extent        
    spatial_lr_scale = 10. * bbox_radius / torch.tensor(n_vertices_in_fg).pow(1/2).item()
    print("Using as spatial_lr_scale:", spatial_lr_scale, "with bbox_radius:", bbox_radius, "and n_vertices_in_fg:", n_vertices_in_fg)
    
    opt_params = OptimizationParams(
        iterations=num_iterations,
        position_lr_init=position_lr_init,
        position_lr_final=position_lr_final,
        position_lr_delay_mult=position_lr_delay_mult,
        position_lr_max_steps=position_lr_max_steps,
        feature_lr=feature_lr,
        opacity_lr=opacity_lr,
        scaling_lr=scaling_lr,
        rotation_lr=rotation_lr,
    )
    optimizer = SuGaROptimizer(sugar, opt_params, spatial_lr_scale=spatial_lr_scale)
    CONSOLE.print("Optimizer initialized.")
    CONSOLE.print("Optimization parameters:")
    CONSOLE.print(opt_params)
    
    CONSOLE.print("Optimizable parameters:")
    for param_group in optimizer.optimizer.param_groups:
        CONSOLE.print(param_group['name'], param_group['lr'])
                        
    
    # ====================Loss function====================
    def loss_fn(pred_rgb, gt_rgb):
        return (1.0 - dssim_factor) * l1_loss(pred_rgb, gt_rgb) + dssim_factor * (1.0 - ssim(pred_rgb, gt_rgb))
    CONSOLE.print(f'Using loss function: {loss_function}')
    
    
    # ====================Start training====================
    sugar.train()
    epoch = 0
    iteration = 0
    train_losses = []
    t0 = time.time()
        
    
    for batch in range(9_999_999):
        if iteration >= num_iterations:
            break
        
        # Shuffle images
        shuffled_idx = torch.randperm(len(nerfmodel.training_cameras))
        train_num_images = len(shuffled_idx)
        
        # We iterate on images
        for i in range(0, train_num_images, train_num_images_per_batch):
            iteration += 1
            
            # Update learning rates
            optimizer.update_learning_rate(iteration)
            
            
            start_idx = i
            end_idx = min(i+train_num_images_per_batch, train_num_images)
            
            camera_indices = shuffled_idx[start_idx:end_idx]
            
            # Computing rgb predictions            
            outputs = sugar.render_image_gaussian_rasterizer( 
                camera_indices=camera_indices.item(),
                verbose=False,
                bg_color = bg_tensor,
                sh_deg=current_sh_levels-1,
                sh_rotations=None,
                compute_color_in_rasterizer=True,
                compute_covariance_in_rasterizer=True,
                return_2d_radii=False,
                quaternions=None,
                use_same_scale_in_all_directions=False,
                return_opacities=False)
                            
            pred_rgb = outputs.view(-1, sugar.image_height, sugar.image_width, 3)                
            pred_rgb = pred_rgb.transpose(-1, -2).transpose(-2, -3)  # TODO: Change for torch.permute
            
            # Gather rgb ground truth
            gt_image = nerfmodel.get_gt_image(camera_indices=camera_indices)           
            gt_rgb = gt_image.view(-1, sugar.image_height, sugar.image_width, 3)
            gt_rgb = gt_rgb.transpose(-1, -2).transpose(-2, -3)
                
            # Compute loss 
            loss = loss_fn(pred_rgb, gt_rgb)                                        
                
            # Surface mesh optimization                                      
            loss = loss
            
            # Update parameters
            loss.backward()
            
            # Optimization step
            optimizer.step()
            optimizer.zero_grad(set_to_none = True)
            
            # Print loss
            if iteration==1 or iteration % print_loss_every_n_iterations == 0:
                CONSOLE.print(f'\n-------------------\nIteration: {iteration}')
                train_losses.append(loss.detach().item())
                CONSOLE.print(f"loss: {loss:>7f}  [{iteration:>5d}/{num_iterations:>5d}]",
                    "computed in", (time.time() - t0) / 60., "minutes.")
                with torch.no_grad():
                    scales = sugar.scaling.detach()
             
                    CONSOLE.print("------Stats-----")
                    CONSOLE.print("---Min, Max, Mean, Std")
                    CONSOLE.print("Points:", sugar.points.min().item(), sugar.points.max().item(), sugar.points.mean().item(), sugar.points.std().item(), sep='   ')
                    CONSOLE.print("Scaling factors:", sugar.scaling.min().item(), sugar.scaling.max().item(), sugar.scaling.mean().item(), sugar.scaling.std().item(), sep='   ')
                    CONSOLE.print("Quaternions:", sugar.quaternions.min().item(), sugar.quaternions.max().item(), sugar.quaternions.mean().item(), sugar.quaternions.std().item(), sep='   ')
                    CONSOLE.print("Sh coordinates dc:", sugar._sh_coordinates_dc.min().item(), sugar._sh_coordinates_dc.max().item(), sugar._sh_coordinates_dc.mean().item(), sugar._sh_coordinates_dc.std().item(), sep='   ')
                    CONSOLE.print("Sh coordinates rest:", sugar._sh_coordinates_rest.min().item(), sugar._sh_coordinates_rest.max().item(), sugar._sh_coordinates_rest.mean().item(), sugar._sh_coordinates_rest.std().item(), sep='   ')
                    CONSOLE.print("Opacities:", sugar.strengths.min().item(), sugar.strengths.max().item(), sugar.strengths.mean().item(), sugar.strengths.std().item(), sep='   ')                    
                t0 = time.time()
                
            # Save model
            if (iteration % save_model_every_n_iterations == 0) or (iteration in save_milestones):
                CONSOLE.print("Saving model...")
                model_path = os.path.join(sugar_checkpoint_path, f'{iteration}.pt')
                sugar.save_model(path=model_path,
                                train_losses=train_losses,
                                epoch=epoch,
                                iteration=iteration,
                                optimizer_state_dict=optimizer.state_dict(),
                                )
                # if optimize_triangles and iteration >= optimize_triangles_from:
                #     rm.save_model(os.path.join(rc_checkpoint_path, f'rm_{iteration}.pt'))
                CONSOLE.print("Model saved.")
            
            if iteration >= num_iterations:
                break
            
            if (iteration > 0) and (current_sh_levels < sh_levels) and (iteration % sh_warmup_every == 0):
                current_sh_levels += 1
                CONSOLE.print("Increasing number of spherical harmonics levels to", current_sh_levels)

        
        epoch += 1

    CONSOLE.print(f"Training finished after {num_iterations} iterations with loss={loss.detach().item()}.")
    CONSOLE.print("Saving final model...")
    model_path = os.path.join(sugar_checkpoint_path, f'{iteration}.pt')
    sugar.save_model(path=model_path,
                    train_losses=train_losses,
                    epoch=epoch,
                    iteration=iteration,
                    optimizer_state_dict=optimizer.state_dict(),
                    )

    CONSOLE.print("Final model saved.")
    
    if True:
        # Build path
        CONSOLE.print("\nExporting ply file with refined Gaussians...")
        
        
        # os.makedirs(refined_ply_save_dir, exist_ok=True)
        
        # Export and save ply
        refined_gaussians = convert_refined_sugar_into_gaussians(sugar)


        output_ply_path = os.path.abspath( os.path.join(args.output_dir, "final.ply") )
        refined_gaussians.save_ply(output_ply_path)
        CONSOLE.print("Ply file exported. This file is needed for using the dedicated viewer.")
    
    return model_path