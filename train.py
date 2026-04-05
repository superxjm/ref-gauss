#
# Copyright (C) 2023, Inria
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
import open3d as o3d
from random import randint
from utils.loss_utils import calculate_loss, l1_loss
from gaussian_renderer import render_surfel, render_initial, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import numpy as np
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from datetime import datetime
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from utils.image_utils import visualize_depth
from utils.graphics_utils import linear_to_srgb
from utils.mesh_utils import GaussianExtractor, post_process_mesh
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import OpenEXR
import Imath
from utils.refl_utils import sample_camera_rays_unnormalize

def export_all_views_at_iteration(scene, render_fn, pipe, background, opt, iteration, model_path):
    print(f"\n[EXPORT] Starting full export for iteration {iteration}...")
    
    # 1. 定义输出根目录，例如 output/xxx/out20000
    export_root = os.path.join(model_path, f"out{iteration}")
    os.makedirs(export_root, exist_ok=True)

    # 2. 定义16个子文件夹的名称 (对应 render_pkg 的内容)
    # 注意：这里的顺序必须和下面 visualization_list 组装的顺序严格一致
    folder_names = [
        "01_gt", "02_render", "03_diffuse_map_2dgs", "03_diffuse_map_ngp", "04_specular_map",
        "05_albedo_map", "06_roughness_map", "07_refl_strength", 
        "08_alpha", "09_depth", "10_rend_normal", "11_surf_normal", 
        "12_error"
    ]
    # 如果开启了 indirect，会多出3张图，凑齐16张
    if opt.indirect:
        folder_names.extend(["14_visibility", "15_direct_light", "16_indirect_light"])

    # 3. 创建所有子文件夹
    for name in folder_names:
        os.makedirs(os.path.join(export_root, name), exist_ok=True)

    # 4. 遍历所有训练相机
    with torch.no_grad():
        for cam in tqdm(scene.getTrainCameras(), desc=f"Exporting views {iteration}"):
            # 渲染当前视角
            render_pkg = render_fn(cam, scene.gaussians, pipe, background, srgb=opt.srgb, opt=opt)
            
            # 准备数据 (逻辑参考 save_training_vis 的非 initial_stage 部分)
            gt_image = cam.original_image.cuda()
            error_map = torch.abs(gt_image - render_pkg["render"])
            

            # [新增] 创建空白的占位 Tensor，以防 render_initial 阶段缺少 PBR 相关的键
            dummy_3ch = torch.zeros_like(render_pkg["render"])
            dummy_1ch = torch.zeros_like(render_pkg["render"][0:1, ...]) # 单通道

            rend_normal = render_pkg["rend_normal"]
            rend_normal = torch.nn.functional.normalize(rend_normal, dim=0)
            surf_normal = render_pkg["surf_normal"]
            surf_normal = torch.nn.functional.normalize(surf_normal, dim=0) 

            # 组装列表，顺序必须和上面的 folder_names 一一对应
            render_pkg["diffuse_map_2dgs"] = linear_to_srgb(render_pkg["diffuse_map_2dgs"])
            render_pkg["diffuse_map_ngp"] = linear_to_srgb(render_pkg["diffuse_map_ngp"])
            render_pkg["specular_map"] = linear_to_srgb(render_pkg["specular_map"])
            visualization_list = [
                gt_image,                                    
                render_pkg["render"],                        
                render_pkg.get("diffuse_map_2dgs", dummy_3ch),   
                render_pkg.get("diffuse_map_ngp", dummy_3ch),  
                render_pkg.get("specular_map", dummy_3ch),  
                render_pkg.get("albedo_map", dummy_3ch),       
                render_pkg.get("roughness_map", dummy_1ch).repeat(3, 1, 1),  
                render_pkg.get("refl_strength_map", dummy_1ch).repeat(3, 1, 1),
                render_pkg.get("rend_alpha", dummy_1ch).repeat(3, 1, 1),        
                visualize_depth(render_pkg.get("surf_depth", dummy_1ch)),       
                rend_normal * 0.5 + 0.5,                       
                surf_normal * 0.5 + 0.5,                      
                error_map,                                     
            ]

            if opt.indirect:
                visualization_list += [
                    render_pkg.get("visibility", dummy_1ch).repeat(3, 1, 1),  # 14
                    render_pkg.get("direct_light", dummy_3ch),                # 15
                    render_pkg.get("indirect_light", dummy_3ch),              # 16
                ]
            
            # 5. 保存每一张图到对应的文件夹
            # 确保列表长度和文件夹数量一致，防止越界
            num_items = min(len(visualization_list), len(folder_names))
            
            for i in range(num_items):
                img_tensor = visualization_list[i]
                folder_name = folder_names[i]
                
                # 文件名使用相机名
                save_path = os.path.join(export_root, folder_name, f"{cam.image_name}.png")
                save_image(img_tensor, save_path)
    
    print(f"[EXPORT] Done. Saved to {export_root}")

def read_exr_depth(path):
    exr = OpenEXR.InputFile(path)
    header = exr.header()

    # image size
    dw = header['dataWindow']
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1

    # always force FLOAT32 output
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    channels = header['channels'].keys()

    # Depth usually stored in Z channel
    if "Z" in channels:
        depth = exr.channel("Z", FLOAT)
    else:
        # fallback: sometimes stored in R
        depth = exr.channel("R", FLOAT)

    depth = np.frombuffer(depth, dtype=np.float32).reshape(h, w)
    return depth

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, model_path, debug_from=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger()

    # Set up parameters 
    TOT_ITER = opt.iterations + 1
    TEST_INTERVAL = 1000
    MESH_EXTRACT_INTERVAL = 2000

    # For real scenes
    USE_ENV_SCOPE = opt.use_env_scope  # False
    if USE_ENV_SCOPE:
        center = [float(c) for c in opt.env_scope_center]
        ENV_CENTER = torch.tensor(center, device='cuda')
        ENV_RADIUS = opt.env_scope_radius
        REFL_MSK_LOSS_W = 0.4


    gaussians = GaussianModel(dataset.sh_degree)
    set_gaussian_para(gaussians, opt, vol=(opt.volume_render_until_iter > opt.init_until_iter)) # #
    scene = Scene(dataset, gaussians)  # init all parameters(pos, scale, rot...) from pcds
    print('ensure_material_mlp')
    gaussians.ensure_material_mlp(training_args=opt, raise_on_fail=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    gaussExtractor = GaussianExtractor(gaussians, render_initial, pipe, bg_color=bg_color) 

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_normal_smooth_for_log = 0.0
    ema_depth_smooth_for_log = 0.0
    ema_psnr_for_log = 0.0
    psnr_test = 0

    progress_bar = tqdm(range(first_iter, TOT_ITER), desc="Training progress")
    first_iter += 1
    iteration = first_iter

    print(f'Propagation until: {opt.normal_prop_until_iter }')
    print(f'Densify until: {opt.densify_until_iter}')
    print(f'Total iterations: {TOT_ITER}')


    initial_stage = opt.initial
    if not initial_stage:
        opt.init_until_iter = 0


    # Training loop
    while iteration < TOT_ITER:
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Increase SH levels every 2000 iterations
        if iteration > opt.feature_rest_from_iter and iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        # Control the init stage
        if iteration > opt.init_until_iter:
            initial_stage = False
        
        # Control the indirect stage
        if iteration == opt.indirect_from_iter + 1:
            opt.indirect = 1

        # if iteration == (opt.volume_render_until_iter + 1) and opt.volume_render_until_iter > opt.init_until_iter:
        #     reset_gaussian_para(gaussians, opt)

        # Initialize envmap
        if not initial_stage:
            envmap2 = gaussians.get_envmap_2 
            envmap2.build_mips()
            envmap = gaussians.get_envmap 
            envmap.build_mips()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        # print(f'training camera num: {len(scene.getTrainCameras())}')

        # Set render
        render = select_render_method(iteration, opt, initial_stage)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, srgb=opt.srgb, opt=opt)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = viewpoint_cam.original_image.cuda()
        
        ######################
        # 尝试从相机获取 normal
        gt_normal = getattr(viewpoint_cam, "normal", None)
        gt_depth = getattr(viewpoint_cam, "gt_depth", None)
        gt_albedo = getattr(viewpoint_cam, "albedo", None)
        dilated_edges = getattr(viewpoint_cam, "dilated_edges", None)

        # 如果相机里没有 normal，且这是第一次遇到这个相机，我们尝试去硬盘加载
        if gt_normal is None:
            normal_path = os.path.join(dataset.source_path, "normal", f"{viewpoint_cam.image_name}.png")
            # depth_path = os.path.join(dataset.source_path, "gt_depth", f"{viewpoint_cam.image_name}.exr")
            albedo_path = os.path.join(dataset.source_path, "albedo", f"{viewpoint_cam.image_name}.png")
            
            # 检查文件是否存在
            if os.path.exists(normal_path):
                try:
                    from PIL import Image
                    import torchvision.transforms.functional as tf   

                    # depth_image = read_exr_depth(depth_path)
                    # if depth_image.shape[1] != viewpoint_cam.image_width or depth_image.shape[0] != viewpoint_cam.image_height:
                    #     depth_image = cv2.resize(depth_image, (viewpoint_cam.image_width, viewpoint_cam.image_height), interpolation=cv2.INTER_NEAREST)
                    # gt_depth = torch.tensor(depth_image, dtype=torch.float32).cuda()

                    with Image.open(albedo_path) as pil_img:
                        # 确保和 GT Image 尺寸一致
                        if pil_img.size != (viewpoint_cam.image_width, viewpoint_cam.image_height):
                            pil_img = pil_img.resize((viewpoint_cam.image_width, viewpoint_cam.image_height), Image.NEAREST)
                        # 转为 Tensor 并移到 CUDA
                        # 此时范围是 [0, 1]
                        gt_albedo = tf.to_tensor(pil_img).cuda()
                        # 只取前3个通道 (以防是 RGBA)
                        if gt_albedo.shape[0] > 3:
                            gt_albedo = gt_albedo[:3, ...]
                        # 缓存到相机对象中
                        # print(gt_albedo.shape)
                        # input()
                        setattr(viewpoint_cam, "albedo", gt_albedo)

                    with Image.open(normal_path) as pil_img:
                        # 确保和 GT Image 尺寸一致
                        if pil_img.size != (viewpoint_cam.image_width, viewpoint_cam.image_height):
                            pil_img = pil_img.resize((viewpoint_cam.image_width, viewpoint_cam.image_height), Image.NEAREST)
                        # 转为 Tensor 并移到 CUDA
                        # 此时范围是 [0, 1]
                        gt_normal = tf.to_tensor(pil_img).cuda()
                        # 只取前3个通道 (以防是 RGBA)
                        if gt_normal.shape[0] > 3:
                            gt_normal = gt_normal[:3, ...]
                        gt_normal[1] = 1 - gt_normal[1]
                        gt_normal[2] = 1 - gt_normal[2]
                        # 转为 NumPy 数组并转换为灰度图像
                        normal_np = gt_normal.permute(1, 2, 0).cpu().numpy()  # Convert from CHW to HWC
                        gray_image = cv2.cvtColor(normal_np, cv2.COLOR_RGB2GRAY)  # 转换为灰度图 
                        # Canny 边缘检测
                        edges = cv2.Canny((gray_image * 255).astype(np.uint8), threshold1=100, threshold2=200)
                        kernel = np.ones((11, 11), np.uint8)  # 使用一个 5x5 的内核
                        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
                        dilated_edges = torch.from_numpy(dilated_edges).unsqueeze(0).cuda()
                        # cv2.imwrite('gray_image.jpg', (gray_image * 255).astype(np.uint8))
                        # cv2.imwrite('normal_image.jpg', (normal_np * 255).astype(np.uint8))
                        # cv2.imwrite('dilated_edges.jpg', dilated_edges)
                        # input()
                    
                    # 缓存到相机对象中
                    # 这样下次训练到这个视角时，就不用再读硬盘了，速度不会变慢
                    # setattr(viewpoint_cam, "gt_depth", gt_depth)
                    setattr(viewpoint_cam, "normal", gt_normal)
                    setattr(viewpoint_cam, "dilated_edges", dilated_edges)
                    
                    # 打印一次成功信息 (仅限前几次)
                    if iteration < first_iter + 50: 
                        print(f"[LOADER] Loaded normal for {viewpoint_cam.image_name}")

                except Exception as e:
                    print(f"[ERROR] Failed to load normal {normal_path}: {e}")
            else:
                setattr(viewpoint_cam, "normal", False) 
        ######################

        total_loss, tb_dict = calculate_loss(viewpoint_cam, gaussians, render_pkg, opt, iteration)
        dist_loss, normal_loss, loss, Ll1, normal_smooth_loss, depth_smooth_loss = tb_dict["loss_dist"], tb_dict["loss_normal_render_depth"], tb_dict["loss0"], tb_dict["loss_l1"], tb_dict["loss_normal_smooth"], tb_dict["loss_depth_smooth"] 

        ######################
        if gt_normal is not None:
            gt_normal = gt_normal.cuda()
            dilated_edges = dilated_edges.cuda()
            rend_normal_cam = render_pkg['rend_normal_cam'] 
            rend_normal = render_pkg['rend_normal'] 
            surf_depth = render_pkg['surf_depth'] 
            rend_alpha = render_pkg['rend_alpha'] 

            gt_normal = gt_normal * 2.0 - 1.0 
            gt_normal = torch.nn.functional.normalize(gt_normal, dim=0) 

            envmap = gaussians.get_envmap
            envmap_loss = envmap.compute_loss()
            total_loss += envmap_loss

            # depth_gt_loss = 1.0 * ((surf_depth - gt_depth) ** 2).mean()
            # total_loss += depth_gt_loss
    
            normal_gt_error = 1 - (rend_normal_cam * gt_normal).sum(dim=0)[None]
            # normal_gt_error = 1 - (rend_normal * gt_normal).sum(dim=0)[None]
            normal_gt_loss = 0.5 * (normal_gt_error[dilated_edges < 255]).mean()
            normal_gt_loss += 0.1 * (normal_gt_error[dilated_edges == 255]).mean()
            total_loss += normal_gt_loss

            alpha_error = torch.abs(1 - rend_alpha)
            lambda_alpha = getattr(opt, "lambda_alpha", 0.1)
            alpha_loss = lambda_alpha * (alpha_error).mean()
            total_loss += alpha_loss

            # TODO 加一个约束albedo的loss，约束albedo的梯度和mvinverse的albedo的梯度相等

            if 'roughness_map' in render_pkg:
                rend_roughness = render_pkg['roughness_map']
                roughness_error = torch.abs(1 - rend_roughness)
                lambda_roughness = getattr(opt, "lambda_roughness", 0.005)
                roughness_loss = lambda_roughness * (roughness_error).mean()
                total_loss += roughness_loss

            if 'shadow_map' in render_pkg:
                rend_shadow = render_pkg['shadow_map']
                shadow_error = torch.abs(1 - rend_shadow)
                lambda_shadow = getattr(opt, "lambda_shadow", 0.005)
                shadow_loss = lambda_shadow * (shadow_error).mean()
                total_loss += shadow_loss

            # Diffuse-Albedo Loss 和 Albedo Gradient Loss
            if gt_albedo is not None and gt_albedo is not False and (not initial_stage):
                diffuse_map = render_pkg['diffuse_map_2dgs']  # [3, H, W]
                shadow_map = render_pkg['shadow_map']    # [1, H, W]
                albedo_map = render_pkg['albedo_map']  # [3, H, W]
                rend_normal = render_pkg['rend_normal']  # [3, H, W]

                diffuse_map_ngp = render_pkg['diffuse_map_ngp']  # [3, H, W]
                lambda_diffuse_ngp = getattr(opt, "lambda_diffuse_ngp", 0.1)
                diffuse_ngp_loss = F.l1_loss(diffuse_map, diffuse_map_ngp)
                total_loss += lambda_diffuse_ngp * diffuse_ngp_loss

                # 获取 envmap
                envmap_2 = gaussians.get_envmap_2

                # 计算3D位置
                surf_depth = render_pkg['surf_depth']  # [1, H, W]
                rays_d, rays_o = sample_camera_rays_unnormalize(
                    viewpoint_cam.HWK,
                    viewpoint_cam.R,
                    viewpoint_cam.T
                )  # rays_d: [H, W, 3], rays_o: [3]
                intersections = rays_o + surf_depth.permute(1, 2, 0) * rays_d  # [H, W, 3]

                # 用法线方向查询 diffuse envmap
                # rend_normal_for_diffuse: [3, H, W] -> [H, W, 3]
                normal_for_env_diffuse = rend_normal.permute(1, 2, 0).detach()  # [H, W, 3]

                # 查询 diffuse envmap (mode="diffuse")
                # 传入 xyz=intersections 以选择最近的探针
                if hasattr(envmap_2, 'light'):
                    # MultiEnvLight
                    env_diffuse = envmap_2(normal_for_env_diffuse.reshape(-1, 3), mode="diffuse", roughness=None, xyz=intersections.reshape(-1, 3))
                    env_diffuse = env_diffuse.reshape(normal_for_env_diffuse.shape)  # [H, W, 3]
                else:
                    # 单个 EnvLight
                    env_diffuse = envmap_2(normal_for_env_diffuse.reshape(-1, 3), mode="diffuse", roughness=None)
                # print(env_diffuse.shape)
                # print(normal_for_env_diffuse.shape)
                # input()
                env_diffuse = env_diffuse.reshape(normal_for_env_diffuse.shape)  # [H, W, 3]

                # 转换为 [3, H, W]
                env_diffuse = env_diffuse.permute(2, 0, 1)  # [3, H, W]

                import torchvision
                # 打印一下env_diffuse 图 和 normal_for_envmap 图
                if iteration % 1000 == 0:
                    env_diffuse_vis = env_diffuse.detach().cpu()
                    # env_diffuse_vis = env_diffuse_vis.clamp(0.0, 1.0)
                    # normal_for_envmap_vis = normal_for_envmap.detach().cpu()
                    # normal_for_envmap_vis = normal_for_envmap_vis * 0.5 + 0.5
                    # normal_for_envmap_vis = normal_for_envmap_vis.clamp(0.0, 1.0)
                    save_image(env_diffuse_vis, os.path.join(model_path, f"debug_env_diffuse_iter{iteration}.png"))
                    env_diff = gaussians.render_env_map_diffuse()
                    env_spec = gaussians.render_env_map_spec()

                    grid = []
                    for idx, env2 in enumerate(env_diff["env2"]):
                        grid.append(env2.permute(2, 0, 1) / 4.0)
                    if len(grid) >= 3:
                        grid = make_grid(grid, nrow=3, padding=10)
                    else:
                        grid = make_grid(grid, nrow=1, padding=10)

                    grid2 = []
                    for idx, env2 in enumerate(env_spec["env2"]):
                        grid2.append(env2.permute(2, 0, 1) / 4.0)
                    if len(grid2) >= 3:
                        grid2 = make_grid(grid2, nrow=3, padding=10)
                    else:
                        grid2 = make_grid(grid2, nrow=1, padding=10)
                    save_image(grid, os.path.join(model_path, f"diff_env_map{iteration}_env.png"))
                    save_image(grid2, os.path.join(model_path, f"spec_env_map{iteration}_env.png"))
                    
                    # save_image(normal_for_envmap_vis, os.path.join(model_path, f"debug_normal_for_envmap_iter{iteration}.png"))

                pseudo_basecolor = albedo_map * env_diffuse * shadow_map
                # pseudo_basecolor = albedo_map * env_diffuse
                if iteration % 1000 == 0 and (not initial_stage):
                    pseudo_basecolor_vis = pseudo_basecolor.detach().cpu()
                    pseudo_basecolor_vis = linear_to_srgb(pseudo_basecolor_vis)
                    save_image(pseudo_basecolor_vis, os.path.join(model_path, f"debug_pseudo_basecolor_iter{iteration}.png"))

                # Loss 1: Diffuse-Albedo L1 Loss
                lambda_diffuse_albedo = getattr(opt, "lambda_diffuse_albedo", 0.05)
                diffuse_albedo_loss = F.l1_loss(diffuse_map, pseudo_basecolor)
                total_loss += lambda_diffuse_albedo * diffuse_albedo_loss

                # Loss 2: Albedo Gradient L1 Loss
                def compute_spatial_gradient(img):
                    """计算图像的空间梯度"""
                    # img: [C, H, W]
                    # 返回 x 和 y 方向的梯度
                    # x 方向梯度: img[:, :, 1:] - img[:, :, :-1]  -> [C, H, W-1]
                    # y 方向梯度: img[:, 1:, :] - img[:, :-1, :]  -> [C, H-1, W]
                    grad_x = img[:, :, 1:] - img[:, :, :-1]  # [C, H, W-1]
                    grad_y = img[:, 1:, :] - img[:, :-1, :]  # [C, H-1, W]
                    return grad_x, grad_y

                # 计算两个 albedo 的梯度
                delta_render_x, delta_render_y = compute_spatial_gradient(albedo_map)
                
                # 打印两个delta
                # print(f"[AlbedoGradLoss] iter={iteration}, delta_render_x={delta_render_x.mean().item():.6f}, delta_render_y={delta_render_y.mean().item():.6f}")
                
                delta_gt_x, delta_gt_y = compute_spatial_gradient(gt_albedo)

                # Gradient L1 Loss (分别对 x 和 y 方向的梯度计算 loss)
                lambda_albedo_grad = getattr(opt, "lambda_albedo_grad", 0.5)
                albedo_grad_loss = F.l1_loss(delta_render_x, delta_gt_x) + F.l1_loss(delta_render_y, delta_gt_y)
                total_loss += lambda_albedo_grad * albedo_grad_loss

                if iteration % 1000 == 0:
                    print(f"[DiffuseAlbedoLoss] iter={iteration}, diffuse_loss={diffuse_albedo_loss.item():.6f}, grad_loss={albedo_grad_loss.item():.6f}")
           
        ######################

        def get_outside_msk():
            return None if not USE_ENV_SCOPE else torch.sum((gaussians.get_xyz - ENV_CENTER[None])**2, dim=-1) > ENV_RADIUS**2
        
        if USE_ENV_SCOPE and 'refl_strength_map' in render_pkg:
            refls = gaussians.get_refl
            refl_msk_loss = refls[get_outside_msk()].mean()
            total_loss += REFL_MSK_LOSS_W * refl_msk_loss
        
        total_loss.backward()

        iter_end.record()

        # if iteration > 20000 and iteration % 1000 == 0:
        #     exclusive_msk = gaussExtractor.filter_invisible_gs(scene.getTrainCameras())
        #     gaussians.reset_opacity0(exclusive_msk)

        with torch.no_grad():
            
            if iteration % TEST_INTERVAL == 0 or iteration == first_iter + 1 or iteration == opt.volume_render_until_iter + 1:
                save_training_vis(viewpoint_cam, gaussians, background, render, pipe, opt, iteration, initial_stage)
                if iteration in [20000, 30000]:
                    print(f"\n[TRIGGER] Exporting separate folders for iteration {iteration}...")
                    current_render = select_render_method(iteration, opt, initial_stage)
                    export_all_views_at_iteration(scene, current_render, pipe, background, opt, iteration, model_path)

            ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss + 0.6 * ema_normal_for_log
            ema_normal_smooth_for_log = 0.4 * normal_smooth_loss + 0.6 * ema_normal_smooth_for_log
            ema_depth_smooth_for_log = 0.4 * depth_smooth_loss + 0.6 * ema_depth_smooth_for_log
            ema_psnr_for_log = 0.4 * psnr(image, gt_image).mean().double().item() + 0.6 * ema_psnr_for_log
            if iteration % TEST_INTERVAL == 0:
                psnr_test = evaluate_psnr(scene, render, {"pipe": pipe, "bg_color": background, "opt": opt})
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Distort": f"{ema_dist_for_log:.{5}f}",
                    "Normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                    "PSNR-train": f"{ema_psnr_for_log:.{4}f}",
                    "PSNR-test": f"{psnr_test:.{4}f}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == TOT_ITER:
                progress_bar.close()

            if tb_writer:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, {"pipe": pipe, "bg_color": background, "opt":opt})

            # if iteration in saving_iterations:
            #     print(f"\n[ITER {iteration}] Saving Gaussians")
            #     scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter and iteration != opt.volume_render_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration <= opt.init_until_iter:
                    opacity_reset_intval = 3000
                    densification_interval = 100
                elif iteration <= opt.normal_prop_until_iter :
                    opacity_reset_intval = 3000
                    densification_interval = opt.densification_interval_when_prop
                else:
                    opacity_reset_intval = 3000
                    densification_interval = 100

                if iteration > opt.densify_from_iter and iteration % densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_opacity_threshold, scene.cameras_extent,
                                                size_threshold)

                HAS_RESET0 = False
                if iteration % opacity_reset_intval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    HAS_RESET0 = True
                    outside_msk = get_outside_msk()
                    # gaussians.reset_opacity0()
                    # gaussians.reset_refl(exclusive_msk=outside_msk)
                if opt.opac_lr0_interval > 0 and (
                        opt.init_until_iter < iteration <= opt.normal_prop_until_iter ) and iteration % opt.opac_lr0_interval == 0:
                    gaussians.set_opacity_lr(opt.opacity_lr)
                if (opt.init_until_iter < iteration <= opt.normal_prop_until_iter ) and iteration % opt.normal_prop_interval == 0:
                    if not HAS_RESET0:
                        outside_msk = get_outside_msk()
                        # gaussians.reset_opacity1(exclusive_msk=outside_msk)
                        if iteration > opt.volume_render_until_iter and opt.volume_render_until_iter > opt.init_until_iter:
                            gaussians.dist_color(exclusive_msk=outside_msk)
                            # gaussians.dist_albedo(exclusive_msk=outside_msk)

                        # gaussians.reset_scale(exclusive_msk=outside_msk)
                        if opt.opac_lr0_interval > 0 and iteration != opt.normal_prop_until_iter :
                            gaussians.set_opacity_lr(0.0)
                
            if (iteration >= opt.indirect_from_iter and iteration % MESH_EXTRACT_INTERVAL == 0) or iteration == (opt.indirect_from_iter):
                if not HAS_RESET0:
                    gaussExtractor.reconstruction(scene.getTrainCameras())
                    mesh = gaussExtractor.extract_mesh_unbounded(resolution=opt.mesh_res)
                    # if 'ref_real' in dataset.source_path:
                    #     mesh = gaussExtractor.extract_mesh_unbounded(resolution=opt.mesh_res)
                    # else:
                    #     depth_trunc = (gaussExtractor.radius * 2.0) if opt.depth_trunc < 0  else opt.depth_trunc
                    #     voxel_size = (depth_trunc / opt.mesh_res) if opt.voxel_size < 0 else opt.voxel_size
                    #     sdf_trunc = 5.0 * voxel_size if opt.sdf_trunc < 0 else opt.sdf_trunc
                    #     mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
                    mesh = post_process_mesh(mesh, cluster_to_keep=opt.num_cluster)
                    ply_path = os.path.join(model_path,f'test_{iteration:06d}.ply')
                    o3d.io.write_triangle_mesh(ply_path, mesh)
                    gaussians.update_mesh(mesh)

            if iteration < TOT_ITER:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + f"/chkpnt{iteration}.pth")

        iteration += 1








# ============================================================
# Utils for training


def select_render_method(iteration, opt, initial_stage):

    if initial_stage:
        render = render_initial
    else:   
        render = render_surfel

    return render


def set_gaussian_para(gaussians, opt, vol=False):
    gaussians.enlarge_scale = opt.enlarge_scale
    gaussians.rough_msk_thr = opt.rough_msk_thr 
    gaussians.init_roughness_value = opt.init_roughness_value
    gaussians.init_refl_value = opt.init_refl_value
    gaussians.refl_msk_thr = opt.refl_msk_thr

def reset_gaussian_para(gaussians, opt):
    gaussians.reset_ori_color()
    gaussians.reset_refl_strength(opt.init_refl_value)
    gaussians.reset_roughness(opt.init_roughness_value)
    gaussians.refl_msk_thr = opt.refl_msk_thr
    gaussians.rough_msk_thr = opt.rough_msk_thr




def save_training_vis(viewpoint_cam, gaussians, background, render_fn, pipe, opt, iteration, initial_stage):
    with torch.no_grad():
        render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background, srgb=opt.srgb, opt=opt)

        error_map = torch.abs(viewpoint_cam.original_image.cuda() - render_pkg["render"])

        if initial_stage:
            visualization_list = [
                viewpoint_cam.original_image.cuda(),
                render_pkg["render"], 
                render_pkg["rend_alpha"].repeat(3, 1, 1),
                visualize_depth(render_pkg["surf_depth"]),  
                render_pkg["rend_normal"] * 0.5 + 0.5, 
                render_pkg["surf_normal"] * 0.5 + 0.5, 
                error_map 
            ]
        else:
            render_pkg["diffuse_map_2dgs"] = linear_to_srgb(render_pkg["diffuse_map_2dgs"])
            render_pkg["diffuse_map_ngp"] = linear_to_srgb(render_pkg["diffuse_map_ngp"])
            render_pkg["specular_map"] = linear_to_srgb(render_pkg["specular_map"])
            render_pkg["albedo_map"] = linear_to_srgb(render_pkg["albedo_map"])
            visualization_list = [
                viewpoint_cam.original_image.cuda(),  
                render_pkg["render"],  
                render_pkg["diffuse_map_2dgs"],
                render_pkg["diffuse_map_ngp"],
                render_pkg["specular_map"],
                render_pkg["albedo_map"],  
                render_pkg["roughness_map"].repeat(3, 1, 1),
                render_pkg["refl_strength_map"].repeat(3, 1, 1),  
                render_pkg["rend_alpha"].repeat(3, 1, 1),  
                visualize_depth(render_pkg["surf_depth"]),  
                render_pkg["rend_normal"] * 0.5 + 0.5,  
                render_pkg["surf_normal"] * 0.5 + 0.5, 
                render_pkg["shadow_map"].repeat(3, 1, 1), 
                error_map, 
            ]
            if opt.indirect:
                visualization_list += [
                    render_pkg["visibility"].repeat(3, 1, 1),
                    render_pkg["direct_light"],
                    render_pkg["indirect_light"],
                ]
  

        grid = torch.stack(visualization_list, dim=0)
        grid = make_grid(grid, nrow=4)
        scale = grid.shape[-2] / 800
        grid = F.interpolate(grid[None], (int(grid.shape[-2] / scale), int(grid.shape[-1] / scale)))[0]
        save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}.png"))

        if not initial_stage:
            if opt.volume_render_until_iter > opt.init_until_iter and iteration <= opt.volume_render_until_iter:
                env_dict = gaussians.render_env_map_2() 
            else:
                env_dict = gaussians.render_env_map()

            # print(torch.max(env_dict["env1"]))
            # print(torch.max(env_dict["env2"]))
            # input()
            grid = []
            for idx, env2 in enumerate(env_dict["env2"]):
                grid.append(env2.permute(2, 0, 1) / 10.0)
            if len(grid) >= 3:
                grid = make_grid(grid, nrow=3, padding=10)
            else:
                grid = make_grid(grid, nrow=1, padding=10)
            save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}_env.png"))
            # for idx, (env1, env2) in enumerate(zip(env_dict["env1"], env_dict["env2"])):
            #     # print(torch.max(env1))
            #     # print(torch.max(env2))
            #     grid = [
            #         env1.permute(2, 0, 1) / 10.0,
            #         env2.permute(2, 0, 1) / 10.0,
            #     ]
            #     grid = make_grid(grid, nrow=1, padding=10)
            #     # Add the index (idx) to the filename
            #     save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}_env_{idx:03d}.png"))
      
NORM_CONDITION_OUTSIDE = False
def prepare_output_and_logger():    
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    args.visualize_path = os.path.join(args.model_path, "visualize")
    
    os.makedirs(args.visualize_path, exist_ok=True)
    print("Visualization folder: {}".format(args.visualize_path))
    
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderkwargs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1, iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(tqdm(config['cameras'])):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, **renderkwargs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

@torch.no_grad()
def evaluate_psnr(scene, renderFunc, renderkwargs):
    psnr_test = 0.0
    torch.cuda.empty_cache()
    if len(scene.getTestCameras()):
        for viewpoint in scene.getTestCameras():
            render_pkg = renderFunc(viewpoint, scene.gaussians, **renderkwargs)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            psnr_test += psnr(image, gt_image).mean().double()

        psnr_test /= len(scene.getTestCameras())
        
    torch.cuda.empty_cache()
    return psnr_test





# ============================================================================
# Main function


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000,20000,30000,40000,50000,60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations = args.test_iterations + [i for i in range(10000, args.iterations+1, 5000)]
    args.test_iterations.append(args.volume_render_until_iter)

    
    if not args.model_path:
        # 获取当前时间并格式化为精确到分钟
        current_time = datetime.now().strftime('%m%d_%H%M')
        # 获取args.source_path的最后一个子目录名
        last_subdir = os.path.basename(os.path.normpath(args.source_path))

        
        # 生成带有时间戳和opt属性的简洁输出目录
        args.model_path = os.path.join(
            "../output/", f"{last_subdir}/",
            f"{last_subdir}-{current_time}"
        )

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.model_path)

    # All done
    print("\nTraining complete.")