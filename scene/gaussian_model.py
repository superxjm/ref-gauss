import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from cubemapencoder import CubemapEncoder
from scene.light import EnvLight, MultiEnvLight
from scene.material_mlp import MaterialMLP
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, init_predefined_omega
from utils.general_utils import strip_symmetric, build_scaling_rotation, safe_normalize, flip_align_view
from utils.refl_utils import sample_camera_rays, get_env_rayd1, get_env_rayd2
import raytracing


def get_env_direction1(H, W):
    gy, gx = torch.meshgrid(torch.linspace(0.0 + 1.0 / H, 1.0 - 1.0 / H, H, device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device='cuda'),
                            indexing='ij')
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    env_directions = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    return env_directions


def get_env_direction2(H, W):
    gx, gy = torch.meshgrid(
        torch.linspace(-torch.pi, torch.pi, W, device='cuda'),
        torch.linspace(0, torch.pi, H, device='cuda'),
        indexing='xy'
    )
    env_directions = torch.stack((
        torch.sin(gy)*torch.cos(gx), 
        torch.sin(gy)*torch.sin(gx), 
        torch.cos(gy)
    ), dim=-1)
    return env_directions


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.refl_activation = torch.sigmoid
        self.inverse_refl_activation = inverse_sigmoid

        self.metalness_ativation = torch.sigmoid
        self.inverse_metalness_activation = inverse_sigmoid

        self.roughness_activation = torch.sigmoid
        self.inverse_roughness_activation = inverse_sigmoid

        self.color_activation = torch.sigmoid
        self.inverse_color_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        self.asg_param = init_predefined_omega(4, 8)


    def __init__(
        self,
        sh_degree: int,
        mlp_material_feature_dim: int = 0,
        mlp_material_encoding: str = "hash",
        mlp_material_hidden_dim: int = 64,
        mlp_material_num_hidden_layers: int = 2,
        mlp_material_hash_n_levels: int = 32,
        mlp_material_hash_n_features_per_level: int = 2,
        mlp_material_hash_log2_hashmap_size: int = 19,
        mlp_material_hash_base_resolution: int = 16,
        mlp_material_hash_finest_resolution: int | None = None,
        mlp_material_hash_per_level_scale: float | None = 1.3,
        mlp_material_hash_bbox_pad: float = 0.1,
        mlp_material_voxel_size: float = 0.0,
    ):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._refl_strength = torch.empty(0) 
        self._ori_color = torch.empty(0) 
        self._diffuse_color = torch.empty(0) 
        self._metalness = torch.empty(0) 
        self._roughness = torch.empty(0) 
        self._shadow = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._indirect_dc = torch.empty(0)
        self._indirect_rest = torch.empty(0)
        self._indirect_asg = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)

        self._normal1 = torch.empty(0)
        self._normal2 = torch.empty(0)

        self.optimizer = None
        self.free_radius = 0    
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.init_refl_value = 0.01
        self.init_roughness_value = 0.1 #[0,1]
        self.init_metalness_value = 0.5 #[0,1]
        self.init_ori_color = 0  
        self.enlarge_scale = 1.5
        self.refl_msk_thr = 0.02
        self.rough_msk_thr = 0.1

        self.env_map = None
        self.env_map_2 = None
        self.env_H, self.env_W = 256, 512
        self.env_directions1 = get_env_direction1(self.env_H, self.env_W)
        self.env_directions2 = get_env_direction2(self.env_H, self.env_W)
        self.ray_tracer = None
        self.setup_functions()

        # Flash-light MLPs
        self.mlp_light_intensity = None
        self.mlp_distance_attenuation = None
        self.mlp_material = None
        self.mlp_material_init_error = None
        self.mlp_material_pixel_stride = 1
        self.mlp_material_feature_dim = int(mlp_material_feature_dim)
        self.mlp_material_encoding = str(mlp_material_encoding)
        self.mlp_material_hidden_dim = int(mlp_material_hidden_dim)
        self.mlp_material_num_hidden_layers = int(mlp_material_num_hidden_layers)
        self.mlp_material_hash_n_levels = int(mlp_material_hash_n_levels)
        self.mlp_material_hash_n_features_per_level = int(mlp_material_hash_n_features_per_level)
        self.mlp_material_hash_log2_hashmap_size = int(mlp_material_hash_log2_hashmap_size)
        self.mlp_material_hash_base_resolution = int(mlp_material_hash_base_resolution)
        self.mlp_material_hash_finest_resolution = None if mlp_material_hash_finest_resolution is None else int(mlp_material_hash_finest_resolution)
        self.mlp_material_hash_per_level_scale = None if mlp_material_hash_per_level_scale is None else float(mlp_material_hash_per_level_scale)
        self.mlp_material_hash_bbox_pad = float(mlp_material_hash_bbox_pad)
        self.mlp_material_voxel_size = float(mlp_material_voxel_size)

    @torch.no_grad()
    def _update_material_hash_bbox_from_xyz(self):
        if self.mlp_material is None or self._xyz is None or self._xyz.numel() == 0:
            return

        xyz = self._xyz.detach()
        finite_mask = torch.isfinite(xyz).all(dim=1)
        if finite_mask.any():
            xyz = xyz[finite_mask]
        if xyz.numel() == 0:
            return

        box_min = xyz.min(dim=0).values
        box_max = xyz.max(dim=0).values
        extent = (box_max - box_min).clamp_min(1e-6)
        pad = self.mlp_material_hash_bbox_pad * extent
        self.mlp_material.set_bounding_box(box_min - pad, box_max + pad)

    def _create_material_mlp(self, device: str = "cuda"):
        self.mlp_material = MaterialMLP(
            feature_dim=self.mlp_material_feature_dim,
            encoding=self.mlp_material_encoding,
            hidden_dim=self.mlp_material_hidden_dim,
            num_hidden_layers=self.mlp_material_num_hidden_layers,
            voxel_size=self.mlp_material_voxel_size,
            hash_n_levels=self.mlp_material_hash_n_levels,
            hash_n_features_per_level=self.mlp_material_hash_n_features_per_level,
            hash_log2_hashmap_size=self.mlp_material_hash_log2_hashmap_size,
            hash_base_resolution=self.mlp_material_hash_base_resolution,
            hash_finest_resolution=self.mlp_material_hash_finest_resolution,
            hash_per_level_scale=self.mlp_material_hash_per_level_scale,
        ).to(device)
        self._update_material_hash_bbox_from_xyz()
        self.mlp_material_init_error = None

    def ensure_material_mlp(self, training_args=None, device: str = "cuda", raise_on_fail: bool = False):
        if training_args is not None:
            try:
                self.mlp_material_voxel_size = float(
                    getattr(training_args, "mlp_material_voxel_size", self.mlp_material_voxel_size) or 0.0
                )
            except Exception:
                pass
        if self.mlp_material is None:
            try:
                self._create_material_mlp(device=device)
            except Exception as exc:
                self.mlp_material = None
                self.mlp_material_init_error = str(exc)
                print(f"[MaterialMLP] Failed to initialize: {exc}")
                input()
                if raise_on_fail:
                    raise
                return False

        try:
            self.mlp_material.voxel_size = float(self.mlp_material_voxel_size)
        except Exception:
            pass

        self._update_material_hash_bbox_from_xyz()

        if training_args is not None and self.optimizer is not None:
            has_group = any(group.get("name") == "mlp_material" for group in self.optimizer.param_groups)
            if not has_group:
                material_lr = float(
                    getattr(
                        training_args,
                        "mlp_material_lr_init",
                        getattr(
                            training_args,
                            "material_mlp_lr",
                            getattr(training_args, "flash_mlp_lr", 0.001),
                        ),
                    )
                )
                self.optimizer.add_param_group(
                    {
                        "params": self.mlp_material.parameters(),
                        "lr": material_lr,
                        "name": "mlp_material",
                    }
                )
        return True

    def _build_default_shadow_raw(self, count: int, device: torch.device):
        default_shadow = inverse_sigmoid(torch.tensor(0.8, device=device)).item()
        return torch.full((count, 1), default_shadow, dtype=torch.float32, device=device)

    def _resolve_probe_grid_res(self, extent: torch.Tensor, args):
        use_fixed_probe_grid = bool(int(getattr(args, "use_ref_fixed_probe_grid", 1)))
        if use_fixed_probe_grid:
            raw_grid = getattr(args, "ref_fixed_probe_grid_res", [3, 2, 1])
            if torch.is_tensor(raw_grid):
                grid_res = raw_grid.to(device=extent.device, dtype=torch.long)
            else:
                grid_res = torch.tensor([int(v) for v in raw_grid], device=extent.device, dtype=torch.long)
        else:
            min_extent = extent.clamp_min(1e-6).min()
            grid_res = torch.round(extent / min_extent).to(device=extent.device, dtype=torch.long)
        return torch.clamp(grid_res, min=1)

    def _build_probe_grid(self, min_xyz: torch.Tensor, max_xyz: torch.Tensor, args):
        extent = max_xyz - min_xyz
        grid_res = self._resolve_probe_grid_res(extent, args)
        print(f"[MultiEnvLight] Grid Split: {grid_res} based on extent {extent.cpu().numpy()}")

        coords = []
        for axis in range(3):
            steps = int(grid_res[axis].item())
            step = extent[axis] / grid_res[axis].clamp_min(1)
            axis_coords = torch.linspace(
                min_xyz[axis] + step / 2.0,
                max_xyz[axis] - step / 2.0,
                steps,
                device=min_xyz.device,
            )
            coords.append(axis_coords)

        grid_x, grid_y, grid_z = torch.meshgrid(coords[0], coords[1], coords[2], indexing='ij')
        probe_centers = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
        grid_min = probe_centers.amin(dim=0)
        grid_max = probe_centers.amax(dim=0)
        return probe_centers, grid_res, grid_min, grid_max

    def _create_multi_env_light(self, probe_centers, grid_res, grid_min, grid_max, args):
        return MultiEnvLight(
            centers=probe_centers,
            k=4,
            grid_res=grid_res,
            grid_min=grid_min,
            grid_max=grid_max,
            path=None,
            device='cuda',
            max_res=args.envmap_max_res,
            min_roughness=args.envmap_min_roughness,
            max_roughness=args.envmap_max_roughness,
            trainable=True
        ).cuda()

    def capture(self):
        checkpoint_aux_state = {
            "material_state": self.mlp_material.state_dict() if self.mlp_material is not None else None,
            "mlp_light_intensity_state": self.mlp_light_intensity.state_dict() if self.mlp_light_intensity is not None else None,
            "mlp_distance_attenuation_state": self.mlp_distance_attenuation.state_dict() if self.mlp_distance_attenuation is not None else None,
        }
        return (
            self.active_sh_degree,
            self._xyz,
            self._refl_strength, 
            self._metalness, 
            self._roughness, 
            self._shadow,
            self._ori_color, 
            self._diffuse_color, 
            self._features_dc,
            self._features_rest,
            self._indirect_dc,
            self._indirect_rest,
            self._indirect_asg,
            self._scaling,
            self._rotation,
            self._opacity,
            self._normal1,  
            self._normal2,  
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            checkpoint_aux_state,
        )
    
    def restore(self, model_args, training_args):
        material_state = None
        mlp_light_intensity_state = None
        mlp_distance_attenuation_state = None
        checkpoint_aux_state = None
        if len(model_args) == 22:
            (self.active_sh_degree,
            self._xyz,
            self._refl_strength,
            self._metalness,
            self._roughness,
            self._ori_color,
            self._diffuse_color,
            self._features_dc,
            self._features_rest,
            self._indirect_dc,
            self._indirect_rest,
            self._indirect_asg,
            self._scaling,
            self._rotation,
            self._opacity,
            self._normal1,
            self._normal2,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale) = model_args
            self._shadow = nn.Parameter(
                self._build_default_shadow_raw(self._xyz.shape[0], self._xyz.device).requires_grad_(True)
            )
        elif len(model_args) == 23:
            if isinstance(model_args[-1], dict) and (
                "material_state" in model_args[-1]
                or "mlp_light_intensity_state" in model_args[-1]
                or "mlp_distance_attenuation_state" in model_args[-1]
            ):
                (self.active_sh_degree,
                self._xyz,
                self._refl_strength,
                self._metalness,
                self._roughness,
                self._ori_color,
                self._diffuse_color,
                self._features_dc,
                self._features_rest,
                self._indirect_dc,
                self._indirect_rest,
                self._indirect_asg,
                self._scaling,
                self._rotation,
                self._opacity,
                self._normal1,
                self._normal2,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                checkpoint_aux_state) = model_args
                self._shadow = nn.Parameter(
                    self._build_default_shadow_raw(self._xyz.shape[0], self._xyz.device).requires_grad_(True)
                )
            else:
                (self.active_sh_degree,
                self._xyz,
                self._refl_strength,
                self._metalness,
                self._roughness,
                self._shadow,
                self._ori_color,
                self._diffuse_color,
                self._features_dc,
                self._features_rest,
                self._indirect_dc,
                self._indirect_rest,
                self._indirect_asg,
                self._scaling,
                self._rotation,
                self._opacity,
                self._normal1,
                self._normal2,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale) = model_args
        elif len(model_args) == 24:
            (self.active_sh_degree,
            self._xyz,
            self._refl_strength,
            self._metalness,
            self._roughness,
            self._shadow,
            self._ori_color,
            self._diffuse_color,
            self._features_dc,
            self._features_rest,
            self._indirect_dc,
            self._indirect_rest,
            self._indirect_asg,
            self._scaling,
            self._rotation,
            self._opacity,
            self._normal1,
            self._normal2,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            checkpoint_aux_state) = model_args
        else:
            raise ValueError(f"Unexpected checkpoint format with {len(model_args)} entries")

        if isinstance(checkpoint_aux_state, dict) and (
            "material_state" in checkpoint_aux_state
            or "mlp_light_intensity_state" in checkpoint_aux_state
            or "mlp_distance_attenuation_state" in checkpoint_aux_state
        ):
            material_state = checkpoint_aux_state.get("material_state")
            mlp_light_intensity_state = checkpoint_aux_state.get("mlp_light_intensity_state")
            mlp_distance_attenuation_state = checkpoint_aux_state.get("mlp_distance_attenuation_state")
        else:
            material_state = checkpoint_aux_state

        if material_state is not None:
            self.ensure_material_mlp(training_args=None, raise_on_fail=False)
        self._indirect_asg = nn.Parameter(torch.zeros(self._rotation.shape[0], 32, 5, device='cuda').requires_grad_(True))
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        if material_state is not None and self.mlp_material is not None:
            self.mlp_material.load_state_dict(material_state, strict=True)
        if mlp_light_intensity_state is not None or mlp_distance_attenuation_state is not None:
            self.init_flash_mlps()
            if mlp_light_intensity_state is not None:
                self.mlp_light_intensity.load_state_dict(mlp_light_intensity_state, strict=True)
            if mlp_distance_attenuation_state is not None:
                self.mlp_distance_attenuation.load_state_dict(mlp_distance_attenuation_state, strict=True)
        # self.optimizer.load_state_dict(opt_dict)

    def set_opacity_lr(self, lr):   
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "opacity":
                param_group['lr'] = lr

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) 
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_refl(self): 
        return self.refl_activation(self._refl_strength)

    @property
    def get_rough(self): 
        return self.roughness_activation(self._roughness)

    @property
    def get_shadow(self):
        return torch.sigmoid(self._shadow)

    @property
    def get_ori_color(self): 
        return self.color_activation(self._ori_color)
    
    @property
    def get_diffuse_color(self): 
        return self.color_activation(self._diffuse_color)
    

    def get_normal(self, scaling_modifier, dir_pp_normalized, return_delta=False): 
        splat2world = self.get_covariance(scaling_modifier)
        normals_raw = splat2world[:,2,:3] 
        normals_raw, positive = flip_align_view(normals_raw, dir_pp_normalized)

        if return_delta:
            delta_normal1 = self._normal1 
            delta_normal2 = self._normal2 
            delta_normal = torch.stack([delta_normal1, delta_normal2], dim=-1) 
            idx = torch.where(positive, 0, 1).long()[:,None,:].repeat(1, 3, 1) 
            delta_normal = torch.gather(delta_normal, index=idx, dim=-1).squeeze(-1) 
            normals = delta_normal + normals_raw
            normals = safe_normalize(normals) 
            return normals, delta_normal
        else:
            normals = safe_normalize(normals_raw)
            return normals

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_rest(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return features_rest
    
    @property
    def get_features_dc(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return features_dc

    @property
    def get_features_and_set_rest_to_zero(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        features_rest = torch.zeros_like(features_rest)
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_view_dep_features(self):
        features_dc = torch.zeros_like(self._features_dc)
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_indirect(self):
        indirect_dc = self._indirect_dc
        indirect_rest = self._indirect_rest
        return torch.cat((indirect_dc, indirect_rest), dim=1)
    
    @property
    def get_asg(self):
        return self._indirect_asg
    
    def render_env_map(self, H=512):
        if H == self.env_H:
            directions1 = self.env_directions1
            directions2 = self.env_directions2
        else:
            W = H * 2
            directions1 = get_env_direction1(H, W)
            directions2 = get_env_direction2(H, W)
        if isinstance(self.env_map, MultiEnvLight):
            env1_all = self.env_map.light(directions1, mode="pure_env")
            env2_all = self.env_map.light(directions2, mode="pure_env")
            env1_list = [env1_all[i] for i in range(env1_all.shape[0])]
            env2_list = [env2_all[i] for i in range(env2_all.shape[0])]
            
            return {
                'env1': env1_list, 
                'env2': env2_list
            }
        return {'env1': self.env_map(directions1, mode="pure_env"), 'env2': self.env_map(directions2, mode="pure_env")}

    def render_env_map_2(self, H=512):
        if H == self.env_H:
            directions1 = self.env_directions1
            directions2 = self.env_directions2
        else:
            W = H * 2
            directions1 = get_env_direction1(H, W)
            directions2 = get_env_direction2(H, W)
        return {'env1': self.env_map_2(directions1, mode="pure_env"), 'env2': self.env_map_2(directions2, mode="pure_env")}

    ###
    def render_env_map_diffuse(self, H=512):
        # if H == self.env_H:
        #     directions1 = self.env_directions1
        #     directions2 = self.env_directions2
        # else:
        #     W = H * 2
        #     directions1 = get_env_direction1(H, W)
        #     directions2 = get_env_direction2(H, W)
        # return {'env1': self.env_map(directions1, mode="pure_env"), 'env2': self.env_map(directions2, mode="pure_env")}
        if H == self.env_H:
            directions1 = self.env_directions1
            directions2 = self.env_directions2
        else:
            W = H * 2
            directions1 = get_env_direction1(H, W)
            directions2 = get_env_direction2(H, W)
        if isinstance(self.env_map, MultiEnvLight):
            env1_all = self.env_map.light(directions1, mode="diffuse")
            env2_all = self.env_map.light(directions2, mode="diffuse")
            env1_list = [env1_all[i] for i in range(env1_all.shape[0])]
            env2_list = [env2_all[i] for i in range(env2_all.shape[0])]
            
            return {
                'env1': env1_list, 
                'env2': env2_list
            }
        return {'env1': self.env_map(directions1, mode="diffuse"), 'env2': self.env_map(directions2, mode="diffuse")}
    
    def render_env_map_spec(self, H=512):
        if H == self.env_H:
            directions1 = self.env_directions1
            directions2 = self.env_directions2
        else:
            W = H * 2
            directions1 = get_env_direction1(H, W)
            directions2 = get_env_direction2(H, W)
        roughness1 = torch.full(directions1.shape[:-1], 1, device=directions1.device, dtype=directions1.dtype)
        roughness2 = torch.full(directions2.shape[:-1], 1, device=directions2.device, dtype=directions2.dtype)
        if isinstance(self.env_map, MultiEnvLight):
            env1_all = self.env_map.light(directions1, roughness=roughness1)
            env2_all = self.env_map.light(directions2, roughness=roughness2)
            env1_list = [env1_all[i] for i in range(env1_all.shape[0])]
            env2_list = [env2_all[i] for i in range(env2_all.shape[0])]
            
            return {
                'env1': env1_list, 
                'env2': env2_list
            }
        return {'env1': self.env_map(directions1, roughness=roughness1), 'env2': self.env_map(directions2, roughness=roughness2)}
    ###

    @property   
    def get_envmap(self): 
        return self.env_map
    
    @property   
    def get_envmap_2(self): 
        return self.env_map_2
    
    @property   
    def get_refl_strength_to_total(self):
        refl = self.get_refl
        return (refl>0.1).sum() / refl.shape[0]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, args):
        print('create_from_pcd')
 
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        sh_features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        sh_features[:, :3, 0 ] = fused_color
        sh_features[:, 3:, 1:] = 0.0
        sh_indirect = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        asg_indirect = torch.zeros((fused_color.shape[0], 5, 32)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        refl = self.inverse_refl_activation(torch.ones_like(opacities).cuda() * self.init_refl_value)
        refl_strength = refl.cuda()

        metalness = self.inverse_metalness_activation(torch.ones_like(opacities).cuda() * self.init_metalness_value)
        metalness = metalness.cuda()

        roughness = self.inverse_roughness_activation(torch.ones_like(opacities).cuda() * self.init_roughness_value)
        roughness = roughness.cuda()
        shadow = self._build_default_shadow_raw(opacities.shape[0], opacities.device)

        def initialize_ori_color(point_cloud, init_color= 0.5, noise_level=0.05):
            diffuse_color = torch.full((point_cloud.shape[0], 3), init_color, dtype=torch.float, device="cuda")
            noise = (torch.rand(point_cloud.shape[0], 3, dtype=torch.float, device="cuda") - 0.5) * noise_level
            ori_color = diffuse_color + noise
            ori_color = torch.clamp(ori_color, 0.0, 1.0)
            return ori_color
        
        ori_color = self.inverse_color_activation(initialize_ori_color(fused_point_cloud))
        diffuse_color = self.inverse_color_activation(initialize_ori_color(fused_point_cloud))  # Initialize diffuse_color similarly

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))

        self._refl_strength = nn.Parameter(refl_strength.requires_grad_(True))  
        self._ori_color = nn.Parameter(ori_color.requires_grad_(True)) 
        self._diffuse_color = nn.Parameter(diffuse_color.requires_grad_(True))  # Initialize _diffuse_color
        self._roughness = nn.Parameter(roughness.requires_grad_(True)) 
        self._metalness = nn.Parameter(metalness.requires_grad_(True)) 
        self._shadow = nn.Parameter(shadow.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._features_dc = nn.Parameter(sh_features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(sh_features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._indirect_dc = nn.Parameter(sh_indirect[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._indirect_rest = nn.Parameter(sh_indirect[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._indirect_asg = nn.Parameter(asg_indirect.transpose(1, 2).contiguous().requires_grad_(True))
        
        normals1 = np.zeros_like(np.asarray(pcd.points, dtype=np.float32))
        normals2 = np.copy(normals1)
        self._normal1 = nn.Parameter(torch.from_numpy(normals1).to(self._xyz.device).requires_grad_(True))
        self._normal2 = nn.Parameter(torch.from_numpy(normals2).to(self._xyz.device).requires_grad_(True))

        # self.env_map = EnvLight(path=None, device='cuda', max_res=args.envmap_max_res, min_roughness=args.envmap_min_roughness, max_roughness=args.envmap_max_roughness, trainable=True).cuda()
        # self.env_map_2 = EnvLight(path=None, device='cuda', max_res=args.envmap_max_res, min_roughness=args.envmap_min_roughness, max_roughness=args.envmap_max_roughness, trainable=True).cuda()

        #########################################
        # 计算包围盒并初始化 MultiEnvLight
        points_np = pcd.points
        min_xyz = torch.tensor(points_np.min(axis=0)).cuda()
        max_xyz = torch.tensor(points_np.max(axis=0)).cuda()
        probe_centers, grid_res, grid_min, grid_max = self._build_probe_grid(min_xyz, max_xyz, args)
        print(f"[MultiEnvLight] Initialized {len(probe_centers)} probes at:\n{probe_centers}")

        # 初始化 MultiEnvLight
        self.env_map = self._create_multi_env_light(probe_centers, grid_res, grid_min, grid_max, args)
        self.env_map_2 = self._create_multi_env_light(probe_centers, grid_res, grid_min, grid_max, args)
        #########################################

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def init_flash_mlps(self):
        """Initialize flash light MLPs (call before training_setup if flash is enabled)."""
        self.mlp_light_intensity = LightMLP1D().cuda()
        self.mlp_distance_attenuation = LightDistanceMLP().cuda()

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        diffuse_mode = str(getattr(training_args, "diffuse_mode", "ngp")).strip().lower()
        use_ngp_diffuse = diffuse_mode in ("ngp", "instantngp", "instant-ngp", "material_mlp", "mlp")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.features_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.features_lr / 20.0, "name": "f_rest"},
            
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            # {'params': self.env_map.parameters(), 'lr': training_args.envmap_cubemap_lr, "name": "env"},     
            # {'params': self.env_map_2.parameters(), 'lr': training_args.envmap_cubemap_lr, "name": "env2"}
            {'params': self.env_map.parameters(), 'lr': training_args.envmap_cubemap_lr, "name": "env"}, 
            {'params': self.env_map_2.parameters(), 'lr': training_args.envmap_cubemap_lr, "name": "env2"},    
        ]

        self._normal1.requires_grad_(requires_grad=False)
        self._normal2.requires_grad_(requires_grad=False)
        l.extend([
            {'params': [self._refl_strength], 'lr': training_args.refl_strength_lr, "name": "refl_strength"},  
            {'params': [self._ori_color], 'lr': training_args.ori_color_lr, "name": "ori_color"},  
            {'params': [self._diffuse_color], 'lr': training_args.ori_color_lr, "name": "diffuse_color"},  
            {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},  
            {'params': [self._metalness], 'lr': training_args.metalness_lr, "name": "metalness"},  
            {'params': [self._shadow], 'lr': training_args.roughness_lr, "name": "shadow"},
            {'params': [self._normal1], 'lr': training_args.normal_lr, "name": "normal1"},
            {'params': [self._normal2], 'lr': training_args.normal_lr, "name": "normal2"},
            {'params': [self._indirect_dc], 'lr': training_args.indirect_lr, "name": "ind_dc"},
            {'params': [self._indirect_rest], 'lr': training_args.indirect_lr / 20.0, "name": "ind_rest"},
            {'params': [self._indirect_asg], 'lr': training_args.asg_lr, "name": "ind_asg"},
        ])

        # Flash MLP parameters
        if self.mlp_light_intensity is not None:
            flash_lr = getattr(training_args, 'flash_mlp_lr', 0.001)
            l.extend([
                {'params': self.mlp_light_intensity.parameters(), 'lr': flash_lr, "name": "mlp_light_intensity"},
                {'params': self.mlp_distance_attenuation.parameters(), 'lr': flash_lr, "name": "mlp_distance_attenuation"},
            ])

        if self.mlp_material is not None and use_ngp_diffuse:
            material_lr = float(
                getattr(
                    training_args,
                    "mlp_material_lr_init",
                    getattr(training_args, "material_mlp_lr", getattr(training_args, "flash_mlp_lr", 0.001)),
                )
            )
            l.append({'params': self.mlp_material.parameters(), 'lr': material_lr, "name": "mlp_material"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        if self.mlp_material is not None and use_ngp_diffuse:
            self.mlp_material_scheduler_args = get_expon_lr_func(
                lr_init=material_lr,
                lr_final=getattr(training_args, "mlp_material_lr_final", material_lr),
                lr_delay_mult=getattr(training_args, "mlp_material_lr_delay_mult", 1.0),
                max_steps=getattr(training_args, "mlp_material_lr_max_steps", training_args.position_lr_max_steps),
            )
        else:
            self.mlp_material_scheduler_args = None

    def update_learning_rate(self, iteration):
        xyz_lr = None
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                xyz_lr = lr
            elif param_group["name"] == "mlp_material" and self.mlp_material_scheduler_args is not None:
                param_group['lr'] = self.mlp_material_scheduler_args(iteration)
        return xyz_lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz','nx2', 'ny2', 'nz2']
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        for i in range(self._indirect_dc.shape[1]*self._indirect_dc.shape[2]):
            l.append('ind_dc_{}'.format(i))
        for i in range(self._indirect_rest.shape[1]*self._indirect_rest.shape[2]):
            l.append('ind_rest_{}'.format(i))
        for i in range(self._indirect_asg.shape[1]*self._indirect_asg.shape[2]):
            l.append('ind_asg_{}'.format(i))
        l.append('opacity')
        l.append('refl_strength') 
        l.append('metalness') 
        l.append('roughness') 
        l.append('shadow')
        for i in range(self._ori_color.shape[1]):
            l.append('ori_color_{}'.format(i))
        for i in range(self._diffuse_color.shape[1]):  # Add diffuse_color attributes
            l.append('diffuse_color_{}'.format(i))


        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        ind_dc = self._indirect_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        ind_rest = self._indirect_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        ind_asg = self._indirect_asg.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        refl_strength = self._refl_strength.detach().cpu().numpy()    
        metalness = self._metalness.detach().cpu().numpy()    
        roughness = self._roughness.detach().cpu().numpy()    
        shadow = self._shadow.detach().cpu().numpy()
        ori_color = self._ori_color.detach().cpu().numpy()    
        diffuse_color = self._diffuse_color.detach().cpu().numpy()  
        
        normals1 = self._normal1.detach().cpu().numpy()
        normals2 = self._normal2.detach().cpu().numpy() 

        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        attributes = np.concatenate((xyz, normals1, normals2, f_dc, f_rest, ind_dc, ind_rest, ind_asg, opacities, refl_strength, metalness, roughness, shadow, ori_color, diffuse_color, scale, rotation), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
        if self.env_map is not None:
            save_path = path.replace('.ply', '1.map')
            torch.save(self.env_map.state_dict(), save_path)

        if self.env_map_2 is not None:
            save_path = path.replace('.ply', '2.map')
            torch.save(self.env_map_2.state_dict(), save_path)

        # Save flash MLPs
        if self.mlp_light_intensity is not None:
            save_dir = os.path.dirname(path)
            torch.save(self.mlp_light_intensity.state_dict(), os.path.join(save_dir, 'light_mlp.pth'))
            torch.save(self.mlp_distance_attenuation.state_dict(), os.path.join(save_dir, 'distance_mlp.pth'))

        if self.mlp_material is not None:
            save_dir = os.path.dirname(path)
            torch.save({"state_dict": self.mlp_material.state_dict()}, os.path.join(save_dir, 'material_mlp.pth'))
                

    def reset_opacity0(self, exclusive_msk = None):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        opacities_new[exclusive_msk] = self._opacity[exclusive_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        if "opacity" not in optimizable_tensors: return
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity1(self, exclusive_msk = None):
        RESET_V = 0.9
        opacity_old = self.get_opacity
        o_msk = (opacity_old > RESET_V).flatten()
        if exclusive_msk is not None:
            o_msk = torch.logical_or(o_msk, exclusive_msk)
        opacities_new = torch.ones_like(opacity_old)*inverse_sigmoid(torch.tensor([RESET_V]).cuda())
        opacities_new[o_msk] = self._opacity[o_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        if "opacity" not in optimizable_tensors: return
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity1_strategy2(self):
        RESET_B = 1.5
        opacity_old = self.get_opacity
        opacities_new = inverse_sigmoid((opacity_old*RESET_B).clamp(0,0.99))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        if "opacity" not in optimizable_tensors: return
        self._opacity = optimizable_tensors["opacity"]


    def reset_refl(self, exclusive_msk = None):
        refl_new = inverse_sigmoid(torch.max(self.get_refl, torch.ones_like(self.get_refl)*self.init_refl_value))
        if exclusive_msk is not None:
            refl_new[exclusive_msk] = self._refl_strength[exclusive_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(refl_new, "refl_strength")
        if "refl_strength" not in optimizable_tensors: return
        self._refl_strength = optimizable_tensors["refl_strength"]


    def dist_rot(self): 
        REFL_MSK_THR = self.refl_msk_thr
        refl_msk = self.get_refl.flatten() > REFL_MSK_THR
        rot = self.get_rotation.clone()
        dist_rot = self.rotation_activation(rot + torch.randn_like(rot)*0.08)
        dist_rot[refl_msk] = rot[refl_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(dist_rot, "rotation")
        if "rotation" not in optimizable_tensors: return
        self._rotation = optimizable_tensors["rotation"]

    def dist_albedo(self, exclusive_msk = None):
        REFL_MSK_THR = self.refl_msk_thr
        DIST_RANGE = 0.4
        refl_msk = self.get_refl.flatten() > REFL_MSK_THR
        if exclusive_msk is not None:
            refl_msk = torch.logical_or(refl_msk, exclusive_msk)
        dcc = self._ori_color.clone()
        dist_dcc = dcc + (torch.rand_like(dcc)*DIST_RANGE*2-DIST_RANGE) 
        dist_dcc[refl_msk] = dcc[refl_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(dist_dcc, "ori_color")
        if "ori_color" not in optimizable_tensors: return
        self._ori_color = optimizable_tensors["ori_color"]

    def dist_color(self, exclusive_msk = None):
        REFL_MSK_THR = self.refl_msk_thr
        DIST_RANGE = 0.4
        refl_msk = self.get_refl.flatten() > REFL_MSK_THR
        if exclusive_msk is not None:
            refl_msk = torch.logical_or(refl_msk, exclusive_msk)
        dcc = self._features_dc.clone()
        dist_dcc = dcc + (torch.rand_like(dcc)*DIST_RANGE*2-DIST_RANGE) 
        dist_dcc[refl_msk] = dcc[refl_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(dist_dcc, "f_dc")
        if "f_dc" not in optimizable_tensors: return
        self._features_dc = optimizable_tensors["f_dc"]

    def enlarge_refl_scales(self, ret_raw=True, ENLARGE_SCALE=1.5, REFL_MSK_THR=0.02, ROUGH_MSK_THR=0.1, exclusive_msk=None):
        ENLARGE_SCALE = self.enlarge_scale
        REFL_MSK_THR = self.refl_msk_thr
        ROUGH_MSK_THR = self.rough_msk_thr

        refl_msk = self.get_refl.flatten() < REFL_MSK_THR
        rough_msk = self.get_rough.flatten() > ROUGH_MSK_THR
        combined_msk = torch.logical_or(refl_msk, rough_msk)
        if exclusive_msk is not None:
            combined_msk = torch.logical_or(combined_msk, exclusive_msk) 
        scales = self.get_scaling
        rmin_axis = (torch.ones_like(scales) * ENLARGE_SCALE)
        if ret_raw:
            scale_new = self.scaling_inverse_activation(scales * rmin_axis)
            scale_new[combined_msk] = self._scaling[combined_msk]
        else:
            scale_new = scales * rmin_axis
            scale_new[combined_msk] = scales[combined_msk]   
        return scale_new

    def reset_scale(self, exclusive_msk = None):
        scale_new = self.enlarge_refl_scales(ret_raw=True, exclusive_msk=exclusive_msk)
        optimizable_tensors = self.replace_tensor_to_optimizer(scale_new, "scaling")
        if "scaling" not in optimizable_tensors: return
        self._scaling = optimizable_tensors["scaling"]


    def reset_features(self, reset_value_dc=0.0, reset_value_rest=0.0):
        # 重置 features_dc
        features_dc_new = torch.full_like(self._features_dc, reset_value_dc, dtype=torch.float, device="cuda")
        # 重置 features_rest
        features_rest_new = torch.full_like(self._features_rest, reset_value_rest, dtype=torch.float, device="cuda")

        # 将新的features_dc和features_rest替换到优化器中
        optimizable_tensors = self.replace_tensor_to_optimizer(features_dc_new, "f_dc")
        optimizable_tensors.update(self.replace_tensor_to_optimizer(features_rest_new, "f_rest"))
        # 更新active_sh_degree
        self.active_sh_degree = 0

        # 更新类中的属性
        if "f_dc" in optimizable_tensors:
            self._features_dc = optimizable_tensors["f_dc"]
        if "f_rest" in optimizable_tensors:
            self._features_rest = optimizable_tensors["f_rest"]


    def reset_ori_color(self, reset_value=0.5, noise_level=0.05):
        diffuse_color = torch.full_like(self._ori_color, reset_value, dtype=torch.float, device="cuda")
        noise = (torch.rand_like(diffuse_color, dtype=torch.float, device="cuda") - 0.5) * noise_level
        ori_color_new = diffuse_color + noise
        ori_color_new = torch.clamp(ori_color_new, 0.0, 1.0)
        
        # 将重置后的 ori_color 更新到优化器中
        optimizable_tensors = self.replace_tensor_to_optimizer(self.inverse_color_activation(ori_color_new), "ori_color")
        if "ori_color" in optimizable_tensors:
            self._ori_color = optimizable_tensors["ori_color"]

    def reset_refl_strength(self, reset_value=0.01):
        refl_strength_new = torch.full_like(self._refl_strength, reset_value, dtype=torch.float, device="cuda")
        optimizable_tensors = self.replace_tensor_to_optimizer(self.inverse_refl_activation(refl_strength_new), "refl_strength")
        if "refl_strength" in optimizable_tensors:
            self._refl_strength = optimizable_tensors["refl_strength"]
    
    def reset_roughness(self, reset_value=0.1):
        roughness_new = torch.full_like(self._roughness, reset_value, dtype=torch.float, device="cuda")
        optimizable_tensors = self.replace_tensor_to_optimizer(self.inverse_refl_activation(roughness_new), "roughness")
        if "roughness" in optimizable_tensors:
            self._roughness = optimizable_tensors["roughness"]


    def load_ply(self, path, relight=False, args=None):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        # # 
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        refl_strength = np.asarray(plydata.elements[0]["refl_strength"])[..., np.newaxis] # #

        ori_color = np.stack((np.asarray(plydata.elements[0]['ori_color_0']),
                              np.asarray(plydata.elements[0]['ori_color_1']),
                              np.asarray(plydata.elements[0]['ori_color_2'])),  axis=1)
        diffuse_color = np.stack((np.asarray(plydata.elements[0]['diffuse_color_0']),
                                np.asarray(plydata.elements[0]['diffuse_color_1']),
                                np.asarray(plydata.elements[0]['diffuse_color_2'])),  axis=1)
        
        roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis] # #
        metalness = np.asarray(plydata.elements[0]["metalness"])[..., np.newaxis] # #
        try:
            shadow = np.asarray(plydata.elements[0]["shadow"])[..., np.newaxis]
        except Exception:
            shadow = np.full(
                (xyz.shape[0], 1),
                inverse_sigmoid(torch.tensor(0.8)).item(),
                dtype=np.float32,
            )

        normal1 = np.stack((np.asarray(plydata.elements[0]["nx"]),
                        np.asarray(plydata.elements[0]["ny"]),
                        np.asarray(plydata.elements[0]["nz"])),  axis=1)
        normal2 = np.stack((np.asarray(plydata.elements[0]["nx2"]),
                        np.asarray(plydata.elements[0]["ny2"]),
                        np.asarray(plydata.elements[0]["nz2"])),  axis=1)


        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        self.active_sh_degree = self.max_sh_degree
        
        indirect_dc = np.zeros((xyz.shape[0], 3, 1))
        indirect_dc[:, 0, 0] = np.asarray(plydata.elements[0]["ind_dc_0"])
        indirect_dc[:, 1, 0] = np.asarray(plydata.elements[0]["ind_dc_1"])
        indirect_dc[:, 2, 0] = np.asarray(plydata.elements[0]["ind_dc_2"])

        extra_ind_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("ind_rest_")]
        extra_ind_names = sorted(extra_ind_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_ind_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        indirect_extra = np.zeros((xyz.shape[0], len(extra_ind_names)))
        for idx, attr_name in enumerate(extra_ind_names):
            indirect_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        indirect_extra = indirect_extra.reshape((indirect_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        extra_asg_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("ind_asg_")]
        extra_asg_names = sorted(extra_asg_names, key = lambda x: int(x.split('_')[-1]))
        indirect_asg = np.zeros((xyz.shape[0], len(extra_asg_names)))
        for idx, attr_name in enumerate(extra_asg_names):
            indirect_asg[:, idx] = np.asarray(plydata.elements[0][attr_name])
        indirect_asg = indirect_asg.reshape((indirect_asg.shape[0], 5, -1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # #
        if not relight:
            map_path1 = path.replace('.ply', '1.map')
            map_path2 = path.replace('.ply', '2.map')
            if os.path.exists(map_path1)  and os.path.exists(map_path2):
                    state_dict1 = torch.load(map_path1)
                    state_dict2 = torch.load(map_path2)

                    # Determine probe centers: either from saved state_dict (new) or recompute (old checkpoint)
                    if 'centers' in state_dict1:
                        # New format: centers are saved in state_dict
                        probe_centers = state_dict1['centers'].cuda()
                        print(f"[Load] Loading probe centers from checkpoint: {probe_centers.shape[0]} probes")
                    else:
                        # Old format: reconstruct from xyz
                        print(f"[Load] Checkpoint is old format (no centers), recomputing from xyz...")
                        min_xyz = torch.tensor(xyz.min(axis=0)).cuda()
                        max_xyz = torch.tensor(xyz.max(axis=0)).cuda()
                        probe_centers, _, _, _ = self._build_probe_grid(min_xyz, max_xyz, args)
                        print(f"[Load] Recomputed {probe_centers.shape[0]} probes")

                    grid_res = state_dict1.get("grid_res", None)
                    grid_min = state_dict1.get("grid_min", None)
                    grid_max = state_dict1.get("grid_max", None)
                    # Create MultiEnvLight with correct probe centers
                    self.env_map = MultiEnvLight(
                        centers=probe_centers,
                        k=4,
                        grid_res=grid_res,
                        grid_min=grid_min,
                        grid_max=grid_max,
                        path=None,
                        device='cuda',
                        max_res=args.envmap_max_res,
                        min_roughness=args.envmap_min_roughness,
                        max_roughness=args.envmap_max_roughness,
                        trainable=True,
                        trilinear_interp=bool(int(getattr(args, "use_ref_env_trilinear", 1))),
                    ).cuda()
                    # Use strict=False to handle both old and new formats
                    self.env_map.load_state_dict(state_dict1, strict=False)
                    self.env_map.build_mips()

                    # Load env_map_2
                    if 'centers' in state_dict2:
                        probe_centers_2 = state_dict2['centers'].cuda()
                    else:
                        probe_centers_2 = probe_centers  # Use same centers for env_map_2

                    grid_res_2 = state_dict2.get("grid_res", grid_res)
                    grid_min_2 = state_dict2.get("grid_min", grid_min)
                    grid_max_2 = state_dict2.get("grid_max", grid_max)
                    self.env_map_2 = MultiEnvLight(
                        centers=probe_centers_2,
                        k=4,
                        grid_res=grid_res_2,
                        grid_min=grid_min_2,
                        grid_max=grid_max_2,
                        path=None,
                        device='cuda',
                        max_res=args.envmap_max_res,
                        min_roughness=args.envmap_min_roughness,
                        max_roughness=args.envmap_max_roughness,
                        trainable=True,
                        trilinear_interp=bool(int(getattr(args, "use_ref_env_trilinear", 1))),
                    ).cuda()
                    self.env_map_2.load_state_dict(state_dict2, strict=False)
                    self.env_map_2.build_mips()
        else:
            map_path = path.replace('.ply', '.hdr')
            self.env_map = EnvLight(path=map_path, device='cuda', trainable=True).cuda()


        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))

        self._refl_strength = nn.Parameter(torch.tensor(refl_strength, dtype=torch.float, device="cuda").requires_grad_(True))   # #
        self._metalness = nn.Parameter(torch.tensor(metalness, dtype=torch.float, device="cuda").requires_grad_(True))   # #
        self._roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))   # #
        self._shadow = nn.Parameter(torch.tensor(shadow, dtype=torch.float, device="cuda").requires_grad_(True))
        self._ori_color = nn.Parameter(torch.tensor(ori_color, dtype=torch.float, device="cuda").requires_grad_(True))   # #
        self._diffuse_color = nn.Parameter(torch.tensor(diffuse_color, dtype=torch.float, device="cuda").requires_grad_(True))   # #

        self._normal1 = nn.Parameter(torch.tensor(normal1, dtype=torch.float, device="cuda").requires_grad_(True))       # #
        self._normal2 = nn.Parameter(torch.tensor(normal2, dtype=torch.float, device="cuda").requires_grad_(True))       # #

        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        self._indirect_dc = nn.Parameter(torch.tensor(indirect_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._indirect_rest = nn.Parameter(torch.tensor(indirect_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._indirect_asg = nn.Parameter(torch.tensor(indirect_asg, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        # Load flash MLPs if saved
        light_mlp_path = os.path.join(os.path.dirname(path), 'light_mlp.pth')
        distance_mlp_path = os.path.join(os.path.dirname(path), 'distance_mlp.pth')
        if os.path.exists(light_mlp_path) and os.path.exists(distance_mlp_path):
            self.init_flash_mlps()
            self.mlp_light_intensity.load_state_dict(torch.load(light_mlp_path))
            self.mlp_distance_attenuation.load_state_dict(torch.load(distance_mlp_path))

        material_mlp_path = os.path.join(os.path.dirname(path), 'material_mlp.pth')
        if os.path.exists(material_mlp_path):
            payload = torch.load(material_mlp_path, map_location="cuda")
            if self.ensure_material_mlp(training_args=None, raise_on_fail=False):
                self.mlp_material.load_state_dict(payload.get("state_dict", payload), strict=True)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is None: continue
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ("mlp", "env", "env2", "mlp_light_intensity", "mlp_distance_attenuation", "mlp_material"):
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)

            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]

        self._refl_strength = optimizable_tensors['refl_strength']    # #
        self._ori_color = optimizable_tensors['ori_color']    # #
        self._diffuse_color = optimizable_tensors['diffuse_color']    # #
        self._roughness = optimizable_tensors['roughness']    # #
        self._metalness = optimizable_tensors['metalness']    # #
        self._shadow = optimizable_tensors['shadow']    # #
        self._normal1 = optimizable_tensors["normal1"]        # #
        self._normal2 = optimizable_tensors["normal2"]        # #

        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._indirect_dc = optimizable_tensors["ind_dc"]
        self._indirect_rest = optimizable_tensors["ind_rest"]
        self._indirect_asg = optimizable_tensors["ind_asg"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ("mlp", "env", "env2", "mlp_light_intensity", "mlp_distance_attenuation", "mlp_material"):
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_refl_strength, new_metalness, new_roughness, new_shadow, new_ori_color, new_diffuse_color, new_features_dc, new_features_rest, new_indirect_dc, new_indirect_asg, new_indirect_rest, new_opacities, new_scaling, new_rotation, new_normal1, new_normal2):
        d = {"xyz": new_xyz,
             
        "refl_strength": new_refl_strength,    # #
        "metalness": new_metalness,    # #
        "roughness": new_roughness,    # #
        "shadow": new_shadow,    # #
        "ori_color": new_ori_color,    # #
        "diffuse_color": new_diffuse_color,    # #
        "normal1" : new_normal1,       # #
        "normal2" : new_normal2,       # #

        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        
        "ind_dc": new_indirect_dc,
        "ind_rest": new_indirect_rest,
        "ind_asg": new_indirect_asg,
        
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]

        self._refl_strength = optimizable_tensors['refl_strength']    # #
        self._metalness = optimizable_tensors['metalness']    # #
        self._roughness = optimizable_tensors['roughness']    # #
        self._shadow = optimizable_tensors['shadow']    # #
        self._ori_color = optimizable_tensors['ori_color']    # #
        self._diffuse_color = optimizable_tensors['diffuse_color']    # #
        self._normal1 = optimizable_tensors["normal1"]        # #
        self._normal2 = optimizable_tensors["normal2"]        # #

        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        
        self._indirect_dc = optimizable_tensors["ind_dc"]
        self._indirect_rest = optimizable_tensors["ind_rest"]
        self._indirect_asg = optimizable_tensors["ind_asg"]
        
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_refl_strength = self._refl_strength[selected_pts_mask].repeat(N,1)   # #
        new_ori_color = self._ori_color[selected_pts_mask].repeat(N,1)   # #
        new_diffuse_color = self._diffuse_color[selected_pts_mask].repeat(N,1)   # #
        new_roughness = self._roughness[selected_pts_mask].repeat(N,1)   # #
        new_metalness = self._metalness[selected_pts_mask].repeat(N,1)   # #
        new_shadow = self._shadow[selected_pts_mask].repeat(N,1)   # #
        new_normal1 = self._normal1[selected_pts_mask].repeat(N,1)        # #
        new_normal2 = self._normal2[selected_pts_mask].repeat(N,1)       # #

        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        
        new_indirect_dc = self._indirect_dc[selected_pts_mask].repeat(N,1,1)
        new_indirect_rest = self._indirect_rest[selected_pts_mask].repeat(N,1,1)
        new_indirect_asg = self._indirect_asg[selected_pts_mask].repeat(N,1,1)
        
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_refl_strength, new_metalness, new_roughness, new_shadow, new_ori_color, new_diffuse_color, new_features_dc, new_features_rest, new_indirect_dc, new_indirect_asg, new_indirect_rest, new_opacity, new_scaling, new_rotation, new_normal1, new_normal2)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]

        new_refl_strength = self._refl_strength[selected_pts_mask]   # #
        new_metalness = self._metalness[selected_pts_mask]   # #
        new_roughness = self._roughness[selected_pts_mask]   # #
        new_shadow = self._shadow[selected_pts_mask]   # #
        new_ori_color = self._ori_color[selected_pts_mask]   # #
        new_diffuse_color = self._diffuse_color[selected_pts_mask]   # #
        new_normal1 = self._normal1[selected_pts_mask]       # #
        new_normal2 = self._normal2[selected_pts_mask]       # #

        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        
        new_indirect_dc = self._indirect_dc[selected_pts_mask]
        new_indirect_rest = self._indirect_rest[selected_pts_mask]
        new_indirect_asg = self._indirect_asg[selected_pts_mask]
        
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_refl_strength, new_metalness, new_roughness, new_shadow, new_ori_color, new_diffuse_color, new_features_dc, new_features_rest, new_indirect_dc, new_indirect_asg, new_indirect_rest, new_opacities, new_scaling, new_rotation, new_normal1, new_normal2)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)  # #
        self.denom[update_filter] += 1

    # #
    def set_requires_grad(self, attrib_name, state: bool):
        getattr(self, f"_{attrib_name}").requires_grad = state
        
    def update_mesh(self, mesh):
        vertices = np.asarray(mesh.vertices).astype(np.float32)
        faces = np.asarray(mesh.triangles).astype(np.int32)
        self.ray_tracer = raytracing.RayTracer(vertices, faces)

    def load_mesh_from_ply(self, model_path, iteration):
        import open3d as o3d
        import os

        ply_path = os.path.join(model_path, f'test_{iteration:06d}.ply')
        mesh = o3d.io.read_triangle_mesh(ply_path)
        self.update_mesh(mesh)
        
    
