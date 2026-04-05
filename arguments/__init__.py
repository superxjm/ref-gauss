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


from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                elif t == list: # #
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, nargs="+")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                elif t == list: # #
                    group.add_argument("--" + key, default=value, nargs="+")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        # Rendering Settings
        self.sh_degree = 3
        self._resolution = -1
        self._white_background = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        
        # Paths
        self._source_path = ""
        self._model_path = ""
        self._images = "images"

        # Device Settings
        self.data_device = "cuda"
        self.eval = False

        # EnvLight Settings
        self.envmap_max_res = 128
        self.envmap_max_roughness = 0.5
        self.envmap_min_roughness = 0.08
        self.relight = False

        # MaterialMLP Settings
        self.mlp_material_feature_dim = 0
        self.mlp_material_encoding = "hash"
        self.mlp_material_hidden_dim = 64
        self.mlp_material_num_hidden_layers = 2
        self.mlp_material_pixel_stride = 1
        self.mlp_material_hash_n_levels = 32
        self.mlp_material_hash_n_features_per_level = 2
        self.mlp_material_hash_log2_hashmap_size = 19
        self.mlp_material_hash_base_resolution = 16
        self.mlp_material_hash_finest_resolution = None
        self.mlp_material_hash_per_level_scale = 1.3
        self.mlp_material_hash_bbox_pad = 0.1

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        group = super().extract(args)
        group.source_path = os.path.abspath(group.source_path)
        return group


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        # Processing Settings
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.use_asg = False

        # Debugging
        self.depth_ratio = 0.0
        self.debug = False

        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # Learning Rate Settings
        self.iterations = 50_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.features_lr = 0.0075 
        self.indirect_lr = 0.0075 
        self.asg_lr = 0.0075 
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001


        self.ori_color_lr = 0.0075 
        self.refl_strength_lr =  0.005 
        self.roughness_lr =  0.005 
        self.metalness_lr = 0.01
        self.normal_lr = 0.006

        self.envmap_cubemap_lr = 0.01
        
        # Densification Settings
        self.percent_dense = 0.01

        # Regularization Parameters
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0
        self.lambda_normal_render_depth = 0.05
        self.lambda_normal_smooth = 0.0
        self.lambda_depth_smooth = 0.0


        # initial values
        self.init_roughness_value = 0.1
        self.init_refl_value = 0.01
        self.init_refl_value_vol = 0.01
        self.rough_msk_thr = 0.01
        self.refl_msk_thr = 0.02
        self.refl_msk_thr_vol = 0.02

        self.enlarge_scale = 1.5


        # Opacity and Densify Settings
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 25000 

        # # Extra settings
        self.densify_grad_threshold = 0.0002
        self.prune_opacity_threshold = 0.05



        # Stage Settings
        self.initial = 0
        self.init_until_iter = 0 
        self.volume_render_until_iter = 18000 
        self.normal_smooth_from_iter = 0
        self.normal_smooth_until_iter = 18000
                
        self.indirect = 0
        self.indirect_from_iter =  20000 

        self.feature_rest_from_iter = 13_000
        self.normal_prop_until_iter = 25_000 

        self.normal_prop_interval = 1000
        self.opac_lr0_interval = 200
        self.densification_interval_when_prop = 500



        self.normal_loss_start = 0
        self.dist_loss_start = 3000

        # Environmental Scoping
        self.use_env_scope = False
        self.env_scope_center = [0., 0., 0.]
        self.env_scope_radius = 0.0
        
        # SRGB Transformation
        self.srgb = True

        # mesh
        self.voxel_size = -1.0
        self.depth_trunc = -1.0
        self.sdf_trunc = -1.0
        self.mesh_res = 512
        self.num_cluster = 1

        # Flash settings
        self.flash = False
        self.flash_from_iter = 20000
        self.flash_only = True       # True: after flash_from_iter, only use flash views
        self.flash_mlp_lr = 0.001
        self.lambda_flash = 1.0
        self.use_flash_indirect = True  # 是否在 render_surfel_flash 中加入预计算间接光
        self.flash_indirect_from_iter = 30000  # 闪光灯间接光启用迭代次数 (预计算+加载缓存)
        self.flash_indirect_trace_eps = 0.05  # 仅用于间接光(dif/spec)二次光线起点偏移
        self.flash_reproj_vis_eps_abs = 5e-3
        self.flash_reproj_vis_eps_rel = 5e-3
        self.flash_clean_legacy_precomp_names = True
        self.flash_debug_dump_dir = ""  # 若非空: 导出 world_pos 点云与 reflect_dir 箭头
        self.flash_debug_arrow_stride = 32
        self.flash_debug_arrow_scale = 0.02
        self.debug_flash_use_cached_env_render = False
        self.debug_flash_cache_on_cpu = True
        self.flash_training_mode = "auto"  # auto|single_flash|dual_flash_nonflash|single_mixed
        self.flash_batch_mode = "dual"  # legacy: 与 flash_only 一起在 flash_training_mode=auto 时映射到旧逻辑
        self.use_flash_nearest_nonflash = False  # 仅在 dual_flash_nonflash 模式生效
        self.diffuse_mode = "ngp"  # ngp|2dgs
        self.lambda_ngp_diffuse = 1

        self.mlp_material_lr_init = 0.001
        self.mlp_material_lr_final = 0.0001
        self.mlp_material_lr_delay_mult = 0.1
        self.mlp_material_lr_max_steps = 40_000
        self.mlp_material_voxel_size = 0.001

        # Ref-alignment / ablation switches (use 0/1 so they can be toggled from CLI)
        self.use_ref_shadow = 1
        self.use_ref_albedo_grad_loss = 1
        self.use_ref_diffuse_albedo_loss = 1
        self.use_ref_albedo_losses_on_flash = 1
        self.use_ref_env_trilinear = 1
        self.use_ref_fixed_probe_grid = 1
        self.ref_fixed_probe_grid_res = [3, 2, 1]
        self.lambda_diffuse_albedo = 0.05
        self.lambda_albedo_grad = 0.5
        self.lambda_shadow = 0.005


        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
