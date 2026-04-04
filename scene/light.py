import torch
import imageio
import numpy as np
from . import renderutils as ru
from .light_utils import *
import nvdiffrast.torch as dr
import imageio
import numpy as np


def linear_to_srgb(linear):
    """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""

    srgb0 = 323 / 25 * linear
    srgb1 = (211 * np.clip(linear,1e-4,255) ** (5 / 12) - 11) / 200
    return np.where(linear <= 0.0031308, srgb0, srgb1)

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

class EnvLightOldVersion(torch.nn.Module):

    def __init__(self, path=None, device=None, scale=1.0, min_res=16, max_res=128, min_roughness=0.08, max_roughness=0.5, trainable=False):
        super().__init__()
        self.device = device if device is not None else 'cuda' # only supports cuda
        self.scale = scale # scale of the hdr values
        self.min_res = min_res # minimum resolution for mip-map
        self.max_res = max_res # maximum resolution for mip-map
        self.min_roughness = min_roughness
        self.max_roughness = max_roughness
        self.trainable = trainable

        # init an empty cubemap
        self.base = torch.nn.Parameter(
            torch.zeros(6, self.max_res, self.max_res, 3, dtype=torch.float32, device=self.device),
            requires_grad=self.trainable,
        )
        
        # try to load from file (.hdr or .exr)
        if path is not None:
            self.load(path)
        
        self.build_mips()


    def load(self, path):
        """
        Load an .hdr or .exr environment light map file and convert it to cubemap.
        """
        # # load latlong env map from file
        # image = imageio.imread(path)  # Load .hdr file
        # if image.dtype != np.float32:
        #     image = image.astype(np.float32) / 255.0  # Scale to [0,1] if not already in float
        # 从文件中加载图像
        hdr_image = imageio.imread(path)
        
        if hdr_image.dtype != np.float32:
            raise ValueError("HDR image should be in float32 format.")

        ldr_image = linear_to_srgb(hdr_image)
        # 确保图像为浮点类型
        image = torch.from_numpy(ldr_image).to(self.device) *  self.scale
        image = torch.clamp(image, 0.001 , 1-0.001)
        image = inverse_sigmoid(image)

        # Convert from latlong to cubemap format
        cubemap = latlong_to_cubemap(image, [self.max_res, self.max_res], self.device)

        # Assign the cubemap to the base parameter
        self.base.data = cubemap 

    def build_mips(self, cutoff=0.99):
        """
        Build mip-maps for specular reflection based on cubemap.
        """
        self.specular = [self.base]
        while self.specular[-1].shape[1] > self.min_res:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = ru.diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.max_roughness - self.min_roughness) + self.min_roughness
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff) 

        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)

    def get_mip(self, roughness):
        """
        Map roughness to mip level.
        """
        return torch.where(
            roughness < self.max_roughness, 
            (torch.clamp(roughness, self.min_roughness, self.max_roughness) - self.min_roughness) / (self.max_roughness - self.min_roughness) * (len(self.specular) - 2), 
            (torch.clamp(roughness, self.max_roughness, 1.0) - self.max_roughness) / (1.0 - self.max_roughness) + len(self.specular) - 2
        )
        

    def __call__(self, l, mode=None, roughness=None):
        """
        Query the environment light based on direction and roughness.
        """
        prefix = l.shape[:-1]
        if len(prefix) != 3:  # Reshape to [B, H, W, -1] if necessary
            l = l.reshape(1, 1, -1, l.shape[-1])
            if roughness is not None:
                roughness = roughness.reshape(1, 1, -1, 1)

        if mode == "diffuse":
            # Diffuse lighting
            light = dr.texture(self.diffuse[None, ...], l, filter_mode='linear', boundary_mode='cube')
        elif mode == "pure_env":
            # Pure environment light (no mip-map)
            light = dr.texture(self.base[None, ...], l, filter_mode='linear', boundary_mode='cube')
        else:
            # Specular lighting with mip-mapping
            miplevel = self.get_mip(roughness)
            light = dr.texture(
                self.specular[0][None, ...], 
                l,
                mip=list(m[None, ...] for m in self.specular[1:]), 
                mip_level_bias=miplevel[..., 0], 
                filter_mode='linear-mipmap-linear', 
                boundary_mode='cube'
            )

        light = light.view(*prefix, -1)
        
        return torch.sigmoid(light) * 10.0

class EnvLight(torch.nn.Module):

    def __init__(self, n_probes=1, path=None, device=None, scale=1.0, min_res=16, max_res=128, min_roughness=0.08, max_roughness=0.5, trainable=False):
        super().__init__()
        self.n_probes = n_probes
        self.device = device if device is not None else 'cuda' # only supports cuda
        self.scale = scale # scale of the hdr values
        self.min_res = min_res # minimum resolution for mip-map
        self.max_res = max_res # maximum resolution for mip-map
        self.min_roughness = min_roughness
        self.max_roughness = max_roughness
        self.trainable = trainable
        """
        # init an empty cubemap
        self.base = torch.nn.Parameter(
            torch.zeros(6, self.max_res, self.max_res, 3, dtype=torch.float32, device=self.device),
            requires_grad=self.trainable,
        )
        """
        # 增加 n_probes 作为第 0 维度
        self.base = torch.nn.Parameter(
            torch.zeros(self.n_probes, 6, self.max_res, self.max_res, 3, dtype=torch.float32, device=self.device),
            requires_grad=self.trainable,
        )

        # try to load from file (.hdr or .exr)
        if path is not None:
            self.load(path)
        
        self.build_mips()


    def load(self, path):
        """
        Load an .hdr or .exr environment light map file and convert it to cubemap.
        """
        # # load latlong env map from file
        # image = imageio.imread(path)  # Load .hdr file
        # if image.dtype != np.float32:
        #     image = image.astype(np.float32) / 255.0  # Scale to [0,1] if not already in float
        # 从文件中加载图像
        hdr_image = imageio.imread(path)
        
        if hdr_image.dtype != np.float32:
            raise ValueError("HDR image should be in float32 format.")

        ldr_image = linear_to_srgb(hdr_image)
        # 确保图像为浮点类型
        image = torch.from_numpy(ldr_image).to(self.device) *  self.scale
        image = torch.clamp(image, 0.001 , 1-0.001)
        image = inverse_sigmoid(image)

        # Convert from latlong to cubemap format
        cubemap = latlong_to_cubemap(image, [self.max_res, self.max_res], self.device)

        # Assign the cubemap to the base parameter
        # self.base.data = cubemap 
        # 增加 batch 维度，并沿第 0 维复制 self.n_probes 次
        self.base.data = cubemap.unsqueeze(0).repeat(self.n_probes, 1, 1, 1, 1)

    def build_mips(self, cutoff=0.99):
        """
        Build mip-maps for specular reflection based on cubemap.
        """
        self.specular = [self.base]
        """
        while self.specular[-1].shape[1] > self.min_res:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = ru.diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.max_roughness - self.min_roughness) + self.min_roughness
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff) 

        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)
        """

        while self.specular[-1].shape[2] > self.min_res:
            # 将多探针拆开逐一传递给旧工具函数，再沿着第 0 维 stack 起来
            next_mip = torch.stack([cubemap_mip.apply(self.specular[-1][i]) for i in range(self.n_probes)], dim=0)
            self.specular += [next_mip]

        self.diffuse = torch.stack([ru.diffuse_cubemap(self.specular[-1][i]) for i in range(self.n_probes)], dim=0)

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.max_roughness - self.min_roughness) + self.min_roughness
            # 同理处理每层的 specular
            self.specular[idx] = torch.stack([ru.specular_cubemap(self.specular[idx][i], roughness, cutoff) for i in range(self.n_probes)], dim=0) 

        # 处理最后一层的 specular
        self.specular[-1] = torch.stack([ru.specular_cubemap(self.specular[-1][i], 1.0, cutoff) for i in range(self.n_probes)], dim=0)

    def get_mip(self, roughness):
        """
        Map roughness to mip level.
        """
        return torch.where(
            roughness < self.max_roughness, 
            (torch.clamp(roughness, self.min_roughness, self.max_roughness) - self.min_roughness) / (self.max_roughness - self.min_roughness) * (len(self.specular) - 2), 
            (torch.clamp(roughness, self.max_roughness, 1.0) - self.max_roughness) / (1.0 - self.max_roughness) + len(self.specular) - 2
        )
    
    def compute_loss(self):
        # 1. Zero-constraint loss (鼓励接近零)
        loss_to_zero = torch.mean(torch.abs(self.base))

        # 2. Channel similarity loss (鼓励三个通道相似)
        loss_channels_similarity = torch.mean(torch.var(self.base, dim=4))

        # 权重，可以根据需要调整
        lambda_zero = 0.0001  # 约束到零的损失的权重
        lambda_similarity = 0.0001  # 通道相似性的损失权重

        # 计算总损失
        total_loss = lambda_similarity * loss_channels_similarity # lambda_zero * loss_to_zero# + 

        return total_loss

    def __call__(self, l, mode=None, roughness=None):
        """
        Query the environment light based on direction and roughness.
        """
        
        original_shape = l.shape[:-1]
        
        # 将 l 展平并扩展为 [n_probes, 1, 像素数量, 3] 
        # 这样 nvdiffrast 就会一次性计算所有探针在这些射线上的光照
        l_input = l.reshape(-1, l.shape[-1]).view(1, 1, -1, l.shape[-1]).expand(self.n_probes, -1, -1, -1).contiguous()
        r_input = None
        if roughness is not None:
            r_input = roughness.reshape(-1, 1).view(1, 1, -1, 1).expand(self.n_probes, -1, -1, -1).contiguous()

        if mode == "diffuse":
            # Diffuse lighting
            light = dr.texture(self.diffuse, l_input, filter_mode='linear', boundary_mode='cube')
        elif mode == "pure_env":
            # Pure environment light (no mip-map)
            light = dr.texture(self.base, l_input, filter_mode='linear', boundary_mode='cube')
        else:
            # Specular lighting with mip-mapping
            miplevel = self.get_mip(r_input)
            light = dr.texture(
                self.specular[0], 
                l_input,
                mip=list(m for m in self.specular[1:]),
                mip_level_bias=miplevel[..., 0], 
                filter_mode='linear-mipmap-linear', 
                boundary_mode='cube'
            )

        # light = light.view(*prefix, -1)
        light = light.view(self.n_probes, *original_shape, -1)

        return torch.sigmoid(light) * 10.0

class MultiEnvLight(torch.nn.Module):
    def __init__(self, centers, k=4, grid_res=None, grid_min=None, grid_max=None, path=None, device=None, scale=1.0, min_res=16, max_res=128, min_roughness=0.08, max_roughness=0.5, trainable=False):
    # def __init__(self, centers, k=4, path=None, device=None, scale=1.0, min_res=16, max_res=128, min_roughness=0.08, max_roughness=0.5, trainable=False):
        super().__init__()
        self.k = k
        # 中心为 buffer
        self.register_buffer("centers", centers)

        if grid_res is None:
            unique_x = torch.unique(centers[:, 0])
            unique_y = torch.unique(centers[:, 1])
            unique_z = torch.unique(centers[:, 2])
            grid_res = torch.tensor([unique_x.numel(), unique_y.numel(), unique_z.numel()], device=centers.device, dtype=torch.long)
        else:
            grid_res = torch.as_tensor(grid_res, device=centers.device, dtype=torch.long)

        if grid_min is None:
            grid_min = centers.amin(dim=0)
        else:
            grid_min = torch.as_tensor(grid_min, device=centers.device, dtype=centers.dtype)

        if grid_max is None:
            grid_max = centers.amax(dim=0)
        else:
            grid_max = torch.as_tensor(grid_max, device=centers.device, dtype=centers.dtype)

        self.register_buffer("grid_res", grid_res)
        self.register_buffer("grid_min", grid_min)
        self.register_buffer("grid_max", grid_max)

        expected_probes = int(torch.prod(self.grid_res).item())
        if expected_probes != centers.shape[0]:
            raise ValueError(f"grid_res {self.grid_res.tolist()} expects {expected_probes} probes, but got {centers.shape[0]}")

        # 创建 N 个EnvLight 实例 
        """
        self.lights = torch.nn.ModuleList([
            EnvLight(path, device, scale, min_res, max_res, min_roughness, max_roughness, trainable)
            for _ in range(len(centers))
        ])
        """
        self.light = EnvLight(n_probes=len(centers), path=path, device=device, scale=scale, min_res=min_res, max_res=max_res, min_roughness=min_roughness, max_roughness=max_roughness, trainable=trainable)

    def training_setup(self, training_args):
        # 直接调用底层的 light 
        if hasattr(self.light, 'training_setup'):
            self.light.training_setup(training_args)
    
    def compute_loss(self):
        # 直接调用底层的 light
        return self.light.compute_loss()

    def build_mips(self):
        # 直接调用，不再需要循环
        self.light.build_mips()

    def _compute_trilinear_corners(self, xyz_flat):
        res = self.grid_res
        coord_max = (res - 1).to(dtype=xyz_flat.dtype)

        span = (self.grid_max - self.grid_min).clamp_min(1e-6)
        coord = (xyz_flat - self.grid_min) / span * coord_max
        coord = torch.clamp(coord, min=0.0)
        coord = torch.minimum(coord, coord_max)

        i0 = torch.floor(coord).to(torch.long)
        i1 = torch.minimum(i0 + 1, res - 1)
        t = coord - i0.to(dtype=coord.dtype)

        single_axis = (res == 1)
        if torch.any(single_axis):
            t[:, single_axis] = 0.0
            i0[:, single_axis] = 0
            i1[:, single_axis] = 0

        stride_x = res[1] * res[2]
        stride_y = res[2]

        corner_indices = []
        corner_weights = []
        for ox in (0, 1):
            ix = i0[:, 0] if ox == 0 else i1[:, 0]
            wx = (1.0 - t[:, 0]) if ox == 0 else t[:, 0]
            for oy in (0, 1):
                iy = i0[:, 1] if oy == 0 else i1[:, 1]
                wy = (1.0 - t[:, 1]) if oy == 0 else t[:, 1]
                for oz in (0, 1):
                    iz = i0[:, 2] if oz == 0 else i1[:, 2]
                    wz = (1.0 - t[:, 2]) if oz == 0 else t[:, 2]
                    corner_indices.append(ix * stride_x + iy * stride_y + iz)
                    corner_weights.append(wx * wy * wz)

        corner_indices = torch.stack(corner_indices, dim=1)
        corner_weights = torch.stack(corner_weights, dim=1)
        corner_weights = corner_weights / corner_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return corner_indices, corner_weights

    def _sample_selected_probes(self, l_flat, mode, r_flat, probe_ids):
        n_samples = l_flat.shape[0]
        n_selected = probe_ids.shape[0]

        l_input = l_flat.view(1, 1, -1, 3).expand(n_selected, -1, -1, -1).contiguous()

        if mode == "diffuse":
            light = dr.texture(self.light.diffuse[probe_ids], l_input, filter_mode='linear', boundary_mode='cube')
        elif mode == "pure_env":
            light = dr.texture(self.light.base[probe_ids], l_input, filter_mode='linear', boundary_mode='cube')
        else:
            if r_flat is None:
                raise ValueError("roughness is required when querying specular environment light")
            r_input = r_flat.view(1, 1, -1, 1).expand(n_selected, -1, -1, -1).contiguous()
            miplevel = self.light.get_mip(r_input)
            light = dr.texture(
                self.light.specular[0][probe_ids],
                l_input,
                mip=[m[probe_ids] for m in self.light.specular[1:]],
                mip_level_bias=miplevel[..., 0],
                filter_mode='linear-mipmap-linear',
                boundary_mode='cube'
            )

        light = light.view(n_selected, n_samples, 3)
        return torch.sigmoid(light) * 10.0
    
    def __call__(self, l, mode=None, roughness=None, xyz=None):
        # 增加 xyz 计算距离权重
        if xyz is None:
            # self.light 会返回所有探针的光照 [n_probes, ...], 我们取第 0 个即可
            return self.light(l, mode, roughness)[0]
        
        # 否则基于距离的加权插值
        input_shape = l.shape
        l_flat = l.view(-1, 3)
        xyz_flat = xyz.view(-1, 3)
        r_flat = None
        if roughness is not None:
            r_flat = roughness.view(-1, 1)

        # n_probes = self.centers.shape[0]
        # # 计算像素到每个中心的距离
        # dists = torch.cdist(xyz_flat, self.centers)  # [n_pixels, n_probes]
        # # 找到最近的 k 个中心
        # actyal_k = min(self.k, n_probes)
        # topk_dists, topk_indices = torch.topk(dists, k=actyal_k, dim=1, largest=False) 
        # # 计算权重（距离倒数的平方）
        # weights = 1.0 / (topk_dists。pow(2) + 1e-6)
        # weights = weights / weights.sum(dim=1, keepdim=True)  # [n_pixels, k] 
        # # 使用 torch.gather 提取 Top-k 哥探针对应的颜色
        # # gather_indices 需要与all_colors 维度一致：[]n_probes, k, 3]
        # gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, 3)  
        # topk_colors = torch.gather(all_colors, 1, gather_indices)  # [n_pixels, k, 3]
        # # 乘以权重并对 k 维度求和（混合）
        # weights = weights.unsqueeze(-1)  # [n_pixels, k, 1]
        # output = (topk_colors * weights).sum(dim=1)  # [n_pixels, 3]


        # 根据 xyz 落到规则网格后做三线性插值，最多使用 8 个角点 probe
        corner_indices, corner_weights = self._compute_trilinear_corners(xyz_flat)

        unique_probe_ids, inverse = torch.unique(corner_indices.reshape(-1), sorted=True, return_inverse=True)
        sampled_colors = self._sample_selected_probes(l_flat, mode, r_flat, unique_probe_ids)

        sampled_colors = sampled_colors.permute(1, 0, 2)  # [n_pixels, n_unique, 3]
        mapped_indices = inverse.view(corner_indices.shape)
        corner_colors = torch.gather(sampled_colors, 1, mapped_indices.unsqueeze(-1).expand(-1, -1, 3))

        output = (corner_colors * corner_weights.unsqueeze(-1)).sum(dim=1)

        return output.reshape(input_shape)  