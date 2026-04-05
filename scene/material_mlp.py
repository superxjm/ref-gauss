import torch
from torch import nn

try:
    import tinycudann as tcnn  # type: ignore
except Exception:  # pragma: no cover
    tcnn = None


class MaterialMLP(nn.Module):
    """Material MLP: (xyz + feature) -> (albedo3, roughness1, metallic1).

    This is intentionally view-invariant (no view_dir/cam_dir inputs).
    """

    def __init__(
        self,
        feature_dim: int,
        encoding: str = "hash",
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        pe_frequencies: int = 10,
        roughness_min: float = 0.1,
        voxel_size: float = 0.0,
        # Hash encoding params (used when encoding=='hash')
        hash_n_levels: int = 32,
        hash_n_features_per_level: int = 2,
        hash_log2_hashmap_size: int = 19,
        hash_base_resolution: int = 16,
        hash_finest_resolution: int | None = None,
        hash_per_level_scale: float | None = None,
        hash_bounding_box: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.encoding = str(encoding)
        self.hidden_dim = int(hidden_dim)
        self.num_hidden_layers = int(num_hidden_layers)
        # Kept for backwards-compatible signature; unused in hash-only implementation.
        self.pe_frequencies = int(pe_frequencies)
        self.roughness_min = float(roughness_min)
        self.voxel_size = float(voxel_size)

        # If True, albedo output is detached (no gradients) while roughness/metallic remain trainable.
        # This is useful when albedo should be treated as fixed but you still want to optimize R/M.
        self.freeze_albedo: bool = False

        if self.encoding != "hash":
            raise ValueError(
                f"MaterialMLP has been simplified to only support encoding='hash' (got {self.encoding!r})."
            )

        # Bounding box used to map xyz -> [0,1] -> [-1,1] for hash encodings.
        if hash_bounding_box is None:
            # Placeholder bbox; should be updated from scene/anchors via set_bounding_box().
            box_min = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32)
            box_max = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
            hash_bounding_box = (box_min, box_max)
        self.register_buffer("_hash_bbox_min", hash_bounding_box[0].detach().to(dtype=torch.float32))
        self.register_buffer("_hash_bbox_max", hash_bounding_box[1].detach().to(dtype=torch.float32))

        self.tcnn_mlp = None
        self.tcnn_post_mlp = None

        if tcnn is None:
            raise ImportError("tinycudann is required for encoding='hash' but could not be imported")

        hash_encoding = {
            "otype": "HashGrid",
            "n_levels": int(hash_n_levels),
            "n_features_per_level": int(hash_n_features_per_level),
            "log2_hashmap_size": int(hash_log2_hashmap_size),
            "base_resolution": int(hash_base_resolution),
            "per_level_scale": float(1.3 if hash_per_level_scale is None else hash_per_level_scale),
        }
        hash_network = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": int(self.hidden_dim),
            "n_hidden_layers": int(self.num_hidden_layers),
        }
        # 先用 tcnn 做 HashGrid 编码 + MLP，得到一个低维特征（这里固定 64 维）
        self.tcnn_mlp = tcnn.NetworkWithInputEncoding(3, 5, hash_encoding, hash_network)

        # self.tcnn_mlp = tcnn.NetworkWithInputEncoding(3, 64, hash_encoding, hash_network)
        # 再把 [hash_feat(64) + voxel内相对位置(3)] concat，用一个小 MLP 输出 5 维材质
        # head_hidden = int(min(64, self.hidden_dim))
        # self.tcnn_post_mlp = nn.Sequential(
        #     nn.Linear(64 + 3, head_hidden),
        #     nn.ReLU(True),
        #     nn.Linear(head_hidden, head_hidden),
        #     nn.ReLU(True),
        #     nn.Linear(head_hidden, 5),
        # )

        print(f"MaterialMLP: encoding={self.encoding}, feature_dim={self.feature_dim}")

    def forward(self, xyz: torch.Tensor, feature: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward.

        Args:
            xyz: [N,3]
            feature: [N,F] or None

        Returns:
            albedo: [N,3] in [0,1]
            roughness: [N,1] in [roughness_min,1]
            metallic: [N,1] in [0,1]
        """
        if xyz.dim() != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must be [N,3], got {xyz.shape}")

        # Normalize xyz to [0,1] based on bounding box, then to [-1,1] for hash encoding
        eps = 1e-6
        extent = (self._hash_bbox_max - self._hash_bbox_min).clamp_min(eps)
        pos01 = (xyz - self._hash_bbox_min) / extent
        y = self.tcnn_mlp(pos01 * 2.0 - 1.0)

        albedo = torch.sigmoid(y[:, 0:3])
        roughness = torch.sigmoid(y[:, 3:4])
        roughness = self.roughness_min + (1.0 - self.roughness_min) * roughness
        metallic = torch.sigmoid(y[:, 4:5])
        # if self.freeze_albedo:
        #     albedo = albedo.detach()
        return albedo, roughness, metallic

    @torch.no_grad()
    def set_bounding_box(self, box_min: torch.Tensor, box_max: torch.Tensor) -> None:
        """Update hash encoding bounding box in-place."""
        self._hash_bbox_min.copy_(box_min.detach().to(dtype=torch.float32, device=self._hash_bbox_min.device))
        self._hash_bbox_max.copy_(box_max.detach().to(dtype=torch.float32, device=self._hash_bbox_max.device))


# class MaterialMLP(nn.Module):
#     """Material MLP: (xyz + feature) -> (albedo3, roughness1, metallic1).

#     This is intentionally view-invariant (no view_dir/cam_dir inputs).
#     """

#     def __init__(
#         self,
#         feature_dim: int,
#         encoding: str = "hash",
#         hidden_dim: int = 256,
#         num_hidden_layers: int = 6,
#         pe_frequencies: int = 10,
#         roughness_min: float = 0.1,
#         voxel_size: float = 0.001,
#         # Hash encoding params (used when encoding=='hash')
#         hash_n_levels: int = 16,
#         hash_n_features_per_level: int = 2,
#         hash_log2_hashmap_size: int = 19,
#         hash_base_resolution: int = 16,
#         hash_finest_resolution: int | None = None,
#         hash_per_level_scale: float | None = None,
#         hash_bounding_box: tuple[torch.Tensor, torch.Tensor] | None = None,
#     ):
#         super().__init__()
#         self.feature_dim = int(feature_dim)
#         self.encoding = str(encoding)
#         self.hidden_dim = int(hidden_dim)
#         self.num_hidden_layers = int(num_hidden_layers)
#         # Kept for backwards-compatible signature; unused in hash-only implementation.
#         self.pe_frequencies = int(pe_frequencies)
#         self.roughness_min = float(roughness_min)
#         self.voxel_size = float(voxel_size)

#         # If True, albedo output is detached (no gradients) while roughness/metallic remain trainable.
#         # This is useful when albedo should be treated as fixed but you still want to optimize R/M.
#         self.freeze_albedo: bool = False

#         if self.encoding != "hash":
#             raise ValueError(
#                 f"MaterialMLP has been simplified to only support encoding='hash' (got {self.encoding!r})."
#             )

#         # Bounding box used to map xyz -> [0,1] -> [-1,1] for hash encodings.
#         if hash_bounding_box is None:
#             # Placeholder bbox; should be updated from scene/anchors via set_bounding_box().
#             box_min = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32)
#             box_max = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
#             hash_bounding_box = (box_min, box_max)
#         self.register_buffer("_hash_bbox_min", hash_bounding_box[0].detach().to(dtype=torch.float32))
#         self.register_buffer("_hash_bbox_max", hash_bounding_box[1].detach().to(dtype=torch.float32))

#         self.tcnn_mlp = None
#         self.tcnn_post_mlp = None

#         if tcnn is None:
#             raise ImportError("tinycudann is required for encoding='hash' but could not be imported")

#         hash_encoding = {
#             "otype": "HashGrid",
#             "n_levels": int(hash_n_levels),
#             "n_features_per_level": int(hash_n_features_per_level),
#             "log2_hashmap_size": int(hash_log2_hashmap_size),
#             "base_resolution": int(hash_base_resolution),
#             "per_level_scale": float(1.3 if hash_per_level_scale is None else hash_per_level_scale),
#         }
#         hash_network = {
#             "otype": "FullyFusedMLP",
#             "activation": "ReLU",
#             "output_activation": "None",
#             "n_neurons": int(self.hidden_dim),
#             "n_hidden_layers": int(self.num_hidden_layers),
#         }
#         # self.tcnn_mlp = tcnn.NetworkWithInputEncoding(3, 5, hash_encoding, hash_network)


#         # 先用 tcnn 做 HashGrid 编码 + MLP，得到一个低维特征（固定 64 维）
#         self.tcnn_mlp = tcnn.NetworkWithInputEncoding(3, 64, hash_encoding, hash_network)

#         # 再把 [hash_feat(64) + voxel内相对位置(3)] concat，用一个小 MLP 输出 5 维材质
#         head_hidden = int(min(64, self.hidden_dim))
#         self.tcnn_post_mlp = nn.Sequential(
#             nn.Linear(64 + 3, head_hidden),
#             nn.ReLU(True),
#             nn.Linear(head_hidden, head_hidden),
#             nn.ReLU(True),
#             nn.Linear(head_hidden, 5),
#         )

#         print(f"MaterialMLP: encoding={self.encoding}, feature_dim={self.feature_dim}")

#     def forward(self, xyz: torch.Tensor, feature: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         # vs = self.voxel_size
#         vs = 0.001
#         origin = self._hash_bbox_min.to(xyz.device, xyz.dtype)
#         vidx = torch.floor((xyz - origin) / vs)
#         xyz_center = (vidx + 0.5) * vs + origin

#         # voxel 内相对位置 → [-1,1]
#         relpos = (xyz - xyz_center) / (vs * 0.5)

#         # hash 输入 → [-1,1]
#         extent = (self._hash_bbox_max - self._hash_bbox_min).clamp_min(1e-6)
#         pos_norm = (xyz_center - self._hash_bbox_min) / extent
#         pos_hash = pos_norm * 2.0 - 1.0

#         feat64 = self.tcnn_mlp(pos_hash)

#         y = self.tcnn_post_mlp(torch.cat([feat64, relpos], dim=1))

#         # y = self.tcnn_mlp(pos_hash)

#         albedo = torch.sigmoid(y[:, 0:3])
#         roughness = torch.sigmoid(y[:, 3:4])
#         roughness = self.roughness_min + (1.0 - self.roughness_min) * roughness
#         metallic = torch.sigmoid(y[:, 4:5])
#         if self.freeze_albedo:
#             albedo = albedo.detach()
#         return albedo, roughness, metallic

#     @torch.no_grad()
#     def set_bounding_box(self, box_min: torch.Tensor, box_max: torch.Tensor) -> None:
#         """Update hash encoding bounding box in-place."""
#         self._hash_bbox_min.copy_(box_min.detach().to(dtype=torch.float32, device=self._hash_bbox_min.device))
#         self._hash_bbox_max.copy_(box_max.detach().to(dtype=torch.float32, device=self._hash_bbox_max.device))

# class MaterialMLP(nn.Module):
#     """Material MLP: (xyz + feature) -> (albedo3, roughness1, metallic1).

#     This is intentionally view-invariant (no view_dir/cam_dir inputs).
#     """

#     def __init__(
#         self,
#         feature_dim: int,
#         encoding: str = "hash",
#         hidden_dim: int = 64,
#         num_hidden_layers: int = 2,
#         pe_frequencies: int = 10,
#         roughness_min: float = 0.1,
#         voxel_size: float = 0.0,
#         # Hash encoding params (used when encoding=='hash')
#         hash_n_levels: int = 32,
#         hash_n_features_per_level: int = 2,
#         hash_log2_hashmap_size: int = 19,
#         hash_base_resolution: int = 16,
#         hash_finest_resolution: int | None = None,
#         hash_per_level_scale: float | None = None,
#         hash_bounding_box: tuple[torch.Tensor, torch.Tensor] | None = None,
#     ):
#         super().__init__()
#         self.feature_dim = int(feature_dim)
#         self.encoding = str(encoding)
#         self.hidden_dim = int(hidden_dim)
#         self.num_hidden_layers = int(num_hidden_layers)
#         # Kept for backwards-compatible signature; unused in hash-only implementation.
#         self.pe_frequencies = int(pe_frequencies)
#         self.roughness_min = float(roughness_min)
#         self.voxel_size = float(voxel_size)

#         # If True, albedo output is detached (no gradients) while roughness/metallic remain trainable.
#         # This is useful when albedo should be treated as fixed but you still want to optimize R/M.
#         self.freeze_albedo: bool = False

#         if self.encoding != "hash":
#             raise ValueError(
#                 f"MaterialMLP has been simplified to only support encoding='hash' (got {self.encoding!r})."
#             )

#         # Bounding box used to map xyz -> [0,1] -> [-1,1] for hash encodings.
#         if hash_bounding_box is None:
#             # Placeholder bbox; should be updated from scene/anchors via set_bounding_box().
#             box_min = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32)
#             box_max = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
#             hash_bounding_box = (box_min, box_max)
#         self.register_buffer("_hash_bbox_min", hash_bounding_box[0].detach().to(dtype=torch.float32))
#         self.register_buffer("_hash_bbox_max", hash_bounding_box[1].detach().to(dtype=torch.float32))

#         self.tcnn_mlp = None
#         self.tcnn_post_mlp = None

#         if tcnn is None:
#             raise ImportError("tinycudann is required for encoding='hash' but could not be imported")

#         hash_encoding = {
#             "otype": "HashGrid",
#             "n_levels": int(hash_n_levels),
#             "n_features_per_level": int(hash_n_features_per_level),
#             "log2_hashmap_size": int(hash_log2_hashmap_size),
#             "base_resolution": int(hash_base_resolution),
#             "per_level_scale": float(1.3 if hash_per_level_scale is None else hash_per_level_scale),
#         }
#         hash_network = {
#             "otype": "FullyFusedMLP",
#             "activation": "ReLU",
#             "output_activation": "None",
#             "n_neurons": int(self.hidden_dim),
#             "n_hidden_layers": int(self.num_hidden_layers),
#         }
#         # 先用 tcnn 做 HashGrid 编码 + MLP，得到一个低维特征（这里固定 64 维）
#         self.tcnn_mlp = tcnn.NetworkWithInputEncoding(3, 5, hash_encoding, hash_network)

#         # self.tcnn_mlp = tcnn.NetworkWithInputEncoding(3, 64, hash_encoding, hash_network)
#         # 再把 [hash_feat(64) + voxel内相对位置(3)] concat，用一个小 MLP 输出 5 维材质
#         # head_hidden = int(min(64, self.hidden_dim))
#         # self.tcnn_post_mlp = nn.Sequential(
#         #     nn.Linear(64 + 3, head_hidden),
#         #     nn.ReLU(True),
#         #     nn.Linear(head_hidden, head_hidden),
#         #     nn.ReLU(True),
#         #     nn.Linear(head_hidden, 5),
#         # )

#         print(f"MaterialMLP: encoding={self.encoding}, feature_dim={self.feature_dim}")

#     def forward(self, xyz: torch.Tensor, feature: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """Forward.

#         Args:
#             xyz: [N,3]
#             feature: [N,F] or None

#         Returns:
#             albedo: [N,3] in [0,1]
#             roughness: [N,1] in [roughness_min,1]
#             metallic: [N,1] in [0,1]
#         """
#         if xyz.dim() != 2 or xyz.shape[1] != 3:
#             raise ValueError(f"xyz must be [N,3], got {xyz.shape}")

#         # Normalize xyz to [0,1] based on bounding box, then to [-1,1] for hash encoding
#         eps = 1e-6
#         extent = (self._hash_bbox_max - self._hash_bbox_min).clamp_min(eps)
#         pos01 = (xyz - self._hash_bbox_min) / extent
#         y = self.tcnn_mlp(pos01 * 2.0 - 1.0)

#         albedo = torch.sigmoid(y[:, 0:3])
#         roughness = torch.sigmoid(y[:, 3:4])
#         roughness = self.roughness_min + (1.0 - self.roughness_min) * roughness
#         metallic = torch.sigmoid(y[:, 4:5])
#         if self.freeze_albedo:
#             albedo = albedo.detach()
#         return albedo, roughness, metallic

#     @torch.no_grad()
#     def set_bounding_box(self, box_min: torch.Tensor, box_max: torch.Tensor) -> None:
#         """Update hash encoding bounding box in-place."""
#         self._hash_bbox_min.copy_(box_min.detach().to(dtype=torch.float32, device=self._hash_bbox_min.device))
#         self._hash_bbox_max.copy_(box_max.detach().to(dtype=torch.float32, device=self._hash_bbox_max.device))
