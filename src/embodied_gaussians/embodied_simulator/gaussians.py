# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
import torch


@dataclass
class GaussianState:
    means: torch.Tensor  # (n_gaussians, 3)
    quats: torch.Tensor  # (n_gaussians, 4) (w, x, y, z)
    colors_logits: torch.Tensor  # (n_gaussians, 3)
    opacities_logits: torch.Tensor  # (n_gaussians,)
    scale_log: torch.Tensor  # (n_gaussians, 3)

    @property
    def colors(self):
        return self.colors_logits.sigmoid()

    @property
    def opacities(self):
        return self.opacities_logits.sigmoid()

    @property
    def scales(self):
        return self.scale_log.exp()

    @property
    def num_gaussians(self):
        return self.means.shape[0]

    def copy(self, src: "GaussianState"):
        with torch.no_grad():
            self.means.copy_(src.means)
            self.quats.copy_(src.quats)
            self.colors_logits.copy_(src.colors_logits)
            self.opacities_logits.copy_(src.opacities_logits)
            self.scale_log.copy_(src.scale_log)

    def clone(self):
        with torch.no_grad():
            return GaussianState(
                means=self.means.clone(),
                quats=self.quats.clone(),
                colors_logits=self.colors_logits.clone(),
                opacities_logits=self.opacities_logits.clone(),
                scale_log=self.scale_log.clone(),
            )

    def slice(self, slice_obj):
        with torch.no_grad():
            return GaussianState(
                means=self.means[slice_obj],
                quats=self.quats[slice_obj],
                colors_logits=self.colors_logits[slice_obj],
                opacities_logits=self.opacities_logits[slice_obj],
                scale_log=self.scale_log[slice_obj],
            )

    def reshape(self, shape):
        with torch.no_grad():
            return GaussianState(
                means=self.means.reshape(*shape, 3),
                quats=self.quats.reshape(*shape, 4),
                colors_logits=self.colors_logits.reshape(*shape, 3),
                opacities_logits=self.opacities_logits.reshape(*shape),
                scale_log=self.scale_log.reshape(*shape, 3),
            )
    
@dataclass
class GaussianModel:
    means: torch.Tensor  # (n_gaussians, 3)
    quats: torch.Tensor  # (n_gaussians, 4) (w, x, y, z)
    scales: torch.Tensor  # (n_gaussians, 3)
    opacities: torch.Tensor  # (n_gaussians,)
    colors: torch.Tensor  # (n_gaussians, 3)
    body_ids: torch.Tensor  # (n_gaussians,)
    # Sparse soft-body bindings.  These arrays contain one row per soft
    # Gaussian rather than one row per Gaussian in the whole scene.
    soft_gaussian_ids: torch.Tensor  # (n_soft_gaussians,)
    soft_gaussian_particle_indices: torch.Tensor  # (n_soft_gaussians, 4)
    soft_gaussian_tet_ids: torch.Tensor  # (n_soft_gaussians,)
    soft_gaussian_barycentric_weights: torch.Tensor  # (n_soft_gaussians, 4)
    soft_gaussian_rest_offsets: torch.Tensor  # (n_soft_gaussians, 3)
    # 0: tetrahedral polar-rotation binding; 1: physical surface-face binding;
    # 2: high-resolution visual-surface face-centroid binding.
    soft_gaussian_binding_modes: torch.Tensor  # (n_soft_gaussians,)
    soft_gaussian_face_particle_indices: torch.Tensor  # (n_soft_gaussians, 3)
    soft_gaussian_rest_face_frames: torch.Tensor  # (n_soft_gaussians, 3, 3)
    # Mode 2 reconstructs each of the visual face's three vertices from a
    # physical boundary face.  The three physical supports for each of the
    # three visual vertices are flattened to nine columns per Gaussian.
    soft_gaussian_visual_vertex_particle_indices: torch.Tensor  # (n_soft, 9)
    soft_gaussian_visual_vertex_weights: torch.Tensor  # (n_soft, 9)
    soft_gaussian_visual_vertex_rest_offsets: torch.Tensor  # (n_soft, 3, 3)
    soft_gaussian_visual_vertex_rest_physical_frames: torch.Tensor  # (n_soft, 3, 3, 3)
    soft_gaussian_rest_visual_face_frames: torch.Tensor  # (n_soft, 3, 3)
    # Inverse of [rest_edge_01, rest_edge_02, rest_unit_normal].
    soft_gaussian_rest_visual_face_poses: torch.Tensor  # (n_soft, 3, 3)
    soft_tet_rest_poses: torch.Tensor  # (n_tetrahedra, 3, 3), inverse rest Dm
    num_body_gaussians: int

    @property
    def num_gaussians(self):
        return self.means.shape[0]

    @property
    def device(self):
        return self.means.device

    @property
    def num_soft_gaussians(self):
        return self.soft_gaussian_ids.shape[0]

    def state(self):
        return GaussianState(
            means=self.means.clone(),
            quats=self.quats.clone(),
            colors_logits=self.colors.logit(),
            opacities_logits=self.opacities.logit(),
            scale_log=self.scales.log(),
        )

    def copy_from_state(self, state: GaussianState):
        with torch.no_grad():
            # self.means.copy_(state.means)
            # self.quats.copy_(state.quats)
            self.scales.copy_(state.scales)
            self.colors.copy_(state.colors)
            self.opacities.copy_(state.opacities)
