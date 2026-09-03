"""Differentiable visual residual mapping for tetrahedral tissue.

This module adapts the residual-mapping inner loop from Liang et al.
(arXiv:2309.11656) to a Gaussian-rendered observation.  The optimized variable
is a residual displacement on physical particles, not an independent Gaussian
pose.  Bound Gaussian means are reconstructed from that residual and rendered
by a caller-provided differentiable function.  Normalized distance, volume,
shape, spatial, temporal, and magnitude terms regularize the visual data term.

The solver deliberately returns a separate corrected state.  It never writes
to a Warp simulation state; a runtime caller must perform its own contact and
no-flip validation before accepting the correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VisualTissueResidualMappingSettings:
    iterations: int = 4
    learning_rate_m: float = 2.5e-5
    image_scale: float = 0.25
    robust_loss_beta: float = 0.05
    data_weight: float = 1.0
    distance_weight: float = 0.4
    volume_weight: float = 1.0
    shape_weight: float = 0.008
    spatial_weight: float = 0.05
    temporal_weight: float = 0.10
    magnitude_weight: float = 0.01
    maximum_residual_m: float = 0.00025
    minimum_volume_ratio: float = 0.02
    maximum_relative_volume_loss_below_floor: float = 0.02
    maximum_local_volume_projection_passes: int = 4
    maximum_backtracks: int = 10

    def validate(self) -> None:
        if self.iterations < 1:
            raise ValueError("Visual residual iterations must be positive")
        if self.learning_rate_m <= 0.0:
            raise ValueError("Visual residual learning rate must be positive")
        if not 0.0 < self.image_scale <= 1.0:
            raise ValueError("Visual residual image scale must be in (0, 1]")
        if self.robust_loss_beta < 0.0:
            raise ValueError("Visual residual robust beta must be non-negative")
        weights = (
            self.data_weight,
            self.distance_weight,
            self.volume_weight,
            self.shape_weight,
            self.spatial_weight,
            self.temporal_weight,
            self.magnitude_weight,
        )
        if min(weights) < 0.0:
            raise ValueError("Visual residual weights must be non-negative")
        if self.maximum_residual_m <= 0.0:
            raise ValueError("Maximum visual residual must be positive")
        if not 0.0 < self.minimum_volume_ratio <= 1.0:
            raise ValueError("Minimum volume ratio must be in (0, 1]")
        if not 0.0 <= self.maximum_relative_volume_loss_below_floor < 1.0:
            raise ValueError(
                "Relative volume loss below the floor must be in [0, 1)"
            )
        if self.maximum_local_volume_projection_passes < 0:
            raise ValueError(
                "Local volume projection passes must be non-negative"
            )
        if self.maximum_backtracks < 0:
            raise ValueError("Maximum backtracks must be non-negative")


@dataclass(frozen=True)
class VisualTissueResidualMappingResult:
    residual: torch.Tensor
    corrected_positions: torch.Tensor
    corrected_gaussian_means: torch.Tensor
    initial_visual_loss: float
    final_visual_loss: float
    initial_camera_visual_losses: tuple[float, ...]
    final_camera_visual_losses: tuple[float, ...]
    camera_weight_sums: tuple[float, ...]
    camera_active_pixel_counts: tuple[int, ...]
    camera_mask_coverage_fractions: tuple[float, ...]
    visual_gradient_norm: torch.Tensor
    final_total_loss: float
    final_distance_loss: float
    final_volume_loss: float
    final_shape_loss: float
    final_spatial_loss: float
    final_temporal_loss: float
    final_magnitude_loss: float
    maximum_residual_m: float
    rms_residual_m: float
    minimum_volume_ratio: float
    initial_minimum_volume_ratio: float
    inverted_tetrahedra: int
    initial_inverted_tetrahedra: int
    newly_inverted_tetrahedra: int
    dynamically_excluded_particles: int
    locally_frozen_particles: int
    local_volume_projection_passes: int
    backtrack_count: int


def _tetra_volumes(positions: torch.Tensor, tets: torch.Tensor) -> torch.Tensor:
    points = positions[tets]
    return torch.linalg.det(
        torch.stack(
            (
                points[:, 1] - points[:, 0],
                points[:, 2] - points[:, 0],
                points[:, 3] - points[:, 0],
            ),
            dim=-1,
        )
    ) / 6.0


def _unique_tetra_edges(tets: torch.Tensor) -> torch.Tensor:
    corner_pairs = torch.tensor(
        ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
        dtype=torch.long,
        device=tets.device,
    )
    edges = tets[:, corner_pairs].reshape(-1, 2)
    edges = torch.sort(edges, dim=1).values
    return torch.unique(edges, dim=0, sorted=True)


def strict_one_tetrahedron_ring_mask(
    seed_mask: torch.Tensor,
    tet_indices: torch.Tensor,
) -> torch.Tensor:
    """Return seeds plus exactly one incident-tetrahedron particle ring.

    The expansion is deliberately computed once from the original seeds. A
    newly included neighbor never becomes another source, which prevents a
    live grip exclusion from spreading through the whole tissue graph.
    """
    if seed_mask.ndim != 1:
        raise ValueError("One-ring seed mask must be one-dimensional")
    if seed_mask.dtype != torch.bool:
        raise ValueError("One-ring seed mask must be boolean")
    if tet_indices.ndim != 2 or tet_indices.shape[1] != 4:
        raise ValueError("One-ring tetrahedra must have shape [tetrahedron, 4]")
    if tet_indices.numel() and (
        int(tet_indices.min()) < 0
        or int(tet_indices.max()) >= len(seed_mask)
    ):
        raise ValueError("One-ring tetrahedron particle index is out of range")
    expanded = seed_mask.clone()
    if not tet_indices.numel() or not bool(seed_mask.any().item()):
        return expanded
    incident_tetrahedra = seed_mask[tet_indices].any(dim=1)
    if bool(incident_tetrahedra.any().item()):
        expanded[tet_indices[incident_tetrahedra].reshape(-1)] = True
    return expanded


def equal_camera_visual_loss(
    rendered_colors: torch.Tensor,
    target_colors: torch.Tensor,
    pixel_weights: torch.Tensor,
    *,
    robust_loss_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute independently normalized camera losses and their equal mean."""
    if rendered_colors.shape != target_colors.shape:
        raise ValueError("Rendered and target colors must have identical shape")
    if rendered_colors.ndim != 4 or rendered_colors.shape[-1] != 3:
        raise ValueError("Colors must have shape [camera, height, width, 3]")
    if pixel_weights.shape != rendered_colors.shape[:3]:
        raise ValueError("Pixel weights must have shape [camera, height, width]")
    if not bool(torch.isfinite(pixel_weights).all().item()):
        raise ValueError("Pixel weights contain non-finite values")
    if bool(torch.any(pixel_weights < 0.0).item()):
        raise ValueError("Pixel weights must be non-negative")
    if robust_loss_beta > 0.0:
        pixel_loss = F.smooth_l1_loss(
            rendered_colors,
            target_colors,
            beta=robust_loss_beta,
            reduction="none",
        ).mean(dim=-1)
    else:
        pixel_loss = F.mse_loss(
            rendered_colors, target_colors, reduction="none"
        ).mean(dim=-1)
    weight_sums = pixel_weights.sum(dim=(1, 2))
    active = weight_sums > 0.0
    if not bool(active.any().item()):
        raise ValueError("Visual residual mapping has no active camera pixels")
    camera_losses = (
        (pixel_loss * pixel_weights).sum(dim=(1, 2))
        / torch.clamp(weight_sums, min=1.0)
    )
    return camera_losses[active].mean(), camera_losses


class TetrahedralGaussianVisualResidualMapper:
    """Optimize a bounded physical-particle residual through Gaussian images."""

    def __init__(
        self,
        *,
        rest_positions: torch.Tensor,
        tet_indices: torch.Tensor,
        fixed_mask: torch.Tensor,
        soft_gaussian_ids: torch.Tensor,
        gaussian_particle_indices: torch.Tensor,
        gaussian_particle_weights: torch.Tensor,
        visual_vertex_rest_offsets: torch.Tensor | None = None,
        visual_vertex_rest_physical_frames: torch.Tensor | None = None,
        rest_visual_face_poses: torch.Tensor | None = None,
        rest_gaussian_quats: torch.Tensor | None = None,
        rest_gaussian_scales: torch.Tensor | None = None,
        settings: VisualTissueResidualMappingSettings | None = None,
    ) -> None:
        self.settings = settings or VisualTissueResidualMappingSettings()
        self.settings.validate()
        self.rest_positions = rest_positions.detach().to(dtype=torch.float32)
        self.tet_indices = tet_indices.detach().to(
            device=self.rest_positions.device, dtype=torch.long
        )
        self.fixed_mask = fixed_mask.detach().to(
            device=self.rest_positions.device, dtype=torch.bool
        )
        self.soft_gaussian_ids = soft_gaussian_ids.detach().to(
            device=self.rest_positions.device, dtype=torch.long
        )
        self.gaussian_particle_indices = gaussian_particle_indices.detach().to(
            device=self.rest_positions.device, dtype=torch.long
        )
        self.gaussian_particle_weights = gaussian_particle_weights.detach().to(
            device=self.rest_positions.device, dtype=torch.float32
        )
        covariance_inputs = (
            visual_vertex_rest_offsets,
            visual_vertex_rest_physical_frames,
            rest_visual_face_poses,
            rest_gaussian_quats,
            rest_gaussian_scales,
        )
        if any(value is not None for value in covariance_inputs) and not all(
            value is not None for value in covariance_inputs
        ):
            raise ValueError(
                "Exact visual-face covariance requires every rest binding input"
            )
        self.deformation_covariance_enabled = all(
            value is not None for value in covariance_inputs
        )
        if self.rest_positions.ndim != 2 or self.rest_positions.shape[1] != 3:
            raise ValueError("rest_positions must have shape [particle, 3]")
        if self.tet_indices.ndim != 2 or self.tet_indices.shape[1] != 4:
            raise ValueError("tet_indices must have shape [tetrahedron, 4]")
        if self.fixed_mask.shape != (len(self.rest_positions),):
            raise ValueError("fixed_mask count does not match particles")
        if self.gaussian_particle_indices.ndim != 2:
            raise ValueError("Gaussian particle supports must be two-dimensional")
        if self.gaussian_particle_weights.shape != self.gaussian_particle_indices.shape:
            raise ValueError("Gaussian support indices and weights disagree")
        if len(self.soft_gaussian_ids) != len(self.gaussian_particle_indices):
            raise ValueError("Soft Gaussian id and support counts disagree")
        if self.tet_indices.numel() and (
            int(self.tet_indices.min()) < 0
            or int(self.tet_indices.max()) >= len(self.rest_positions)
        ):
            raise ValueError("Tetrahedron particle index is out of range")
        if self.gaussian_particle_indices.numel() and (
            int(self.gaussian_particle_indices.min()) < 0
            or int(self.gaussian_particle_indices.max()) >= len(self.rest_positions)
        ):
            raise ValueError("Gaussian support particle index is out of range")
        weight_sums = self.gaussian_particle_weights.sum(dim=1)
        if not torch.allclose(
            weight_sums,
            torch.ones_like(weight_sums),
            atol=2.0e-5,
            rtol=0.0,
        ):
            raise ValueError("Each Gaussian physical support must sum to one")

        self.edges = _unique_tetra_edges(self.tet_indices)
        rest_edge_vectors = (
            self.rest_positions[self.edges[:, 1]]
            - self.rest_positions[self.edges[:, 0]]
        )
        self.rest_edge_lengths = torch.linalg.vector_norm(
            rest_edge_vectors, dim=1
        ).clamp_min(1.0e-9)
        self.rest_volumes = _tetra_volumes(
            self.rest_positions, self.tet_indices
        )
        if bool(torch.any(self.rest_volumes.abs() <= 1.0e-15).item()):
            raise ValueError("Residual mapper received a degenerate tetrahedron")
        rest_points = self.rest_positions[self.tet_indices]
        rest_shape = torch.stack(
            (
                rest_points[:, 1] - rest_points[:, 0],
                rest_points[:, 2] - rest_points[:, 0],
                rest_points[:, 3] - rest_points[:, 0],
            ),
            dim=-1,
        )
        self.inverse_rest_shape = torch.linalg.inv(rest_shape)
        self.visual_vertex_rest_offsets: torch.Tensor | None = None
        self.visual_vertex_rest_physical_frames: torch.Tensor | None = None
        self.rest_visual_face_poses: torch.Tensor | None = None
        self.rest_gaussian_covariances: torch.Tensor | None = None
        if self.deformation_covariance_enabled:
            if self.gaussian_particle_indices.shape[1] != 9:
                raise ValueError(
                    "Exact visual-face covariance requires nine supports"
                )
            gaussian_count = len(self.soft_gaussian_ids)
            assert visual_vertex_rest_offsets is not None
            assert visual_vertex_rest_physical_frames is not None
            assert rest_visual_face_poses is not None
            assert rest_gaussian_quats is not None
            assert rest_gaussian_scales is not None
            self.visual_vertex_rest_offsets = (
                visual_vertex_rest_offsets.detach().to(
                    device=self.rest_positions.device, dtype=torch.float32
                )
            )
            self.visual_vertex_rest_physical_frames = (
                visual_vertex_rest_physical_frames.detach().to(
                    device=self.rest_positions.device, dtype=torch.float32
                )
            )
            self.rest_visual_face_poses = rest_visual_face_poses.detach().to(
                device=self.rest_positions.device, dtype=torch.float32
            )
            rest_quats = rest_gaussian_quats.detach().to(
                device=self.rest_positions.device, dtype=torch.float32
            )
            rest_scales = rest_gaussian_scales.detach().to(
                device=self.rest_positions.device, dtype=torch.float32
            )
            expected_shapes = {
                "visual_vertex_rest_offsets": (gaussian_count, 3, 3),
                "visual_vertex_rest_physical_frames": (
                    gaussian_count,
                    3,
                    3,
                    3,
                ),
                "rest_visual_face_poses": (gaussian_count, 3, 3),
                "rest_gaussian_quats": (gaussian_count, 4),
                "rest_gaussian_scales": (gaussian_count, 3),
            }
            actual = {
                "visual_vertex_rest_offsets": self.visual_vertex_rest_offsets,
                "visual_vertex_rest_physical_frames": (
                    self.visual_vertex_rest_physical_frames
                ),
                "rest_visual_face_poses": self.rest_visual_face_poses,
                "rest_gaussian_quats": rest_quats,
                "rest_gaussian_scales": rest_scales,
            }
            for name, expected in expected_shapes.items():
                if actual[name].shape != expected:
                    raise ValueError(
                        f"{name} has shape {actual[name].shape}, expected {expected}"
                    )
            if bool(torch.any(rest_scales <= 0.0).item()):
                raise ValueError("Rest Gaussian scales must be positive")
            rest_rotations = self._quaternion_matrices_wxyz(rest_quats)
            self.rest_gaussian_covariances = (
                rest_rotations
                @ torch.diag_embed(rest_scales.square())
                @ rest_rotations.transpose(-1, -2)
            )

    @classmethod
    def from_visual_face_centroid_bindings(
        cls,
        *,
        rest_positions: torch.Tensor,
        tet_indices: torch.Tensor,
        fixed_mask: torch.Tensor,
        soft_gaussian_ids: torch.Tensor,
        visual_vertex_particle_indices: torch.Tensor,
        visual_vertex_weights: torch.Tensor,
        visual_vertex_rest_offsets: torch.Tensor | None = None,
        visual_vertex_rest_physical_frames: torch.Tensor | None = None,
        rest_visual_face_poses: torch.Tensor | None = None,
        rest_gaussian_quats: torch.Tensor | None = None,
        rest_gaussian_scales: torch.Tensor | None = None,
        settings: VisualTissueResidualMappingSettings | None = None,
    ) -> "TetrahedralGaussianVisualResidualMapper":
        """Build the linear centroid derivative from three embedded vertices.

        Runtime mode-2 bindings store three physical supports for each of a
        visual triangle's three vertices.  A Gaussian lies at their arithmetic
        centroid, so each stored vertex weight contributes one third.
        Offset-frame rotation is intentionally excluded from this first-order
        translational Jacobian and remains a documented approximation.
        """
        if visual_vertex_particle_indices.ndim != 2 or (
            visual_vertex_particle_indices.shape[1] != 9
        ):
            raise ValueError("Visual centroid bindings must have nine supports")
        if visual_vertex_weights.shape != visual_vertex_particle_indices.shape:
            raise ValueError("Visual centroid support weights disagree")
        return cls(
            rest_positions=rest_positions,
            tet_indices=tet_indices,
            fixed_mask=fixed_mask,
            soft_gaussian_ids=soft_gaussian_ids,
            gaussian_particle_indices=visual_vertex_particle_indices,
            gaussian_particle_weights=visual_vertex_weights / 3.0,
            visual_vertex_rest_offsets=visual_vertex_rest_offsets,
            visual_vertex_rest_physical_frames=(
                visual_vertex_rest_physical_frames
            ),
            rest_visual_face_poses=rest_visual_face_poses,
            rest_gaussian_quats=rest_gaussian_quats,
            rest_gaussian_scales=rest_gaussian_scales,
            settings=settings,
        )

    @staticmethod
    def _quaternion_matrices_wxyz(quaternions: torch.Tensor) -> torch.Tensor:
        quaternions = F.normalize(quaternions, dim=-1)
        w, x, y, z = quaternions.unbind(dim=-1)
        return torch.stack(
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
            dim=-1,
        ).reshape(*quaternions.shape[:-1], 3, 3)

    def deformed_gaussian_geometry(
        self, particle_positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct exact mode-2 means and deformation covariances."""
        if not self.deformation_covariance_enabled:
            raise RuntimeError("Exact visual-face covariance is not configured")
        assert self.visual_vertex_rest_offsets is not None
        assert self.visual_vertex_rest_physical_frames is not None
        assert self.rest_visual_face_poses is not None
        assert self.rest_gaussian_covariances is not None
        support_indices = self.gaussian_particle_indices.reshape(-1, 3, 3)
        support_weights = (
            self.gaussian_particle_weights.reshape(-1, 3, 3) * 3.0
        )
        support_positions = particle_positions[support_indices]
        embedded = (
            support_positions * support_weights[..., None]
        ).sum(dim=2)
        physical_edge_x = support_positions[:, :, 1] - support_positions[:, :, 0]
        physical_normal = F.normalize(
            torch.cross(
                physical_edge_x,
                support_positions[:, :, 2] - support_positions[:, :, 0],
                dim=-1,
            ),
            dim=-1,
            eps=1.0e-12,
        )
        physical_tangent_x = F.normalize(
            physical_edge_x, dim=-1, eps=1.0e-12
        )
        physical_tangent_y = F.normalize(
            torch.cross(physical_normal, physical_tangent_x, dim=-1),
            dim=-1,
            eps=1.0e-12,
        )
        current_physical_frames = torch.stack(
            (physical_tangent_x, physical_tangent_y, physical_normal),
            dim=-1,
        )
        physical_rotations = (
            current_physical_frames
            @ self.visual_vertex_rest_physical_frames.transpose(-1, -2)
        )
        visual_vertices = embedded + torch.matmul(
            physical_rotations,
            self.visual_vertex_rest_offsets[..., None],
        ).squeeze(-1)
        means = visual_vertices.mean(dim=1)
        visual_edge_x = visual_vertices[:, 1] - visual_vertices[:, 0]
        visual_edge_y = visual_vertices[:, 2] - visual_vertices[:, 0]
        visual_normal = F.normalize(
            torch.cross(visual_edge_x, visual_edge_y, dim=-1),
            dim=-1,
            eps=1.0e-12,
        )
        current_visual_shapes = torch.stack(
            (visual_edge_x, visual_edge_y, visual_normal), dim=-1
        )
        surface_deformation = (
            current_visual_shapes @ self.rest_visual_face_poses
        )
        covariances = (
            surface_deformation
            @ self.rest_gaussian_covariances
            @ surface_deformation.transpose(-1, -2)
        )
        return means, covariances

    @staticmethod
    def _clip_vectors(vectors: torch.Tensor, maximum_norm: float) -> torch.Tensor:
        norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
        scales = torch.clamp(maximum_norm / norms.clamp_min(1.0e-12), max=1.0)
        return vectors * scales

    def corrected_gaussian_means(
        self,
        base_gaussian_means: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        support_residual = residual[self.gaussian_particle_indices]
        soft_correction = (
            support_residual * self.gaussian_particle_weights[..., None]
        ).sum(dim=1)
        # Runtime can render only the packed tissue subset because the tool
        # and image background are zero-weighted by the observation mask.  A
        # standalone tissue model is already packed in this same order.
        if len(base_gaussian_means) == len(self.soft_gaussian_ids):
            return base_gaussian_means + soft_correction
        full_correction = torch.zeros_like(base_gaussian_means).index_copy(
            0, self.soft_gaussian_ids, soft_correction
        )
        return base_gaussian_means + full_correction

    def _constraint_losses(
        self, corrected_positions: torch.Tensor, residual: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        edge_vectors = (
            corrected_positions[self.edges[:, 1]]
            - corrected_positions[self.edges[:, 0]]
        )
        edge_lengths = torch.linalg.vector_norm(edge_vectors, dim=1)
        distance = torch.mean(
            ((edge_lengths - self.rest_edge_lengths) / self.rest_edge_lengths) ** 2
        )
        volumes = _tetra_volumes(corrected_positions, self.tet_indices)
        volume_ratio = volumes / self.rest_volumes
        volume = torch.mean((volume_ratio - 1.0) ** 2)

        points = corrected_positions[self.tet_indices]
        current_shape = torch.stack(
            (
                points[:, 1] - points[:, 0],
                points[:, 2] - points[:, 0],
                points[:, 3] - points[:, 0],
            ),
            dim=-1,
        )
        deformation_gradient = current_shape @ self.inverse_rest_shape
        right_cauchy_green = (
            deformation_gradient.transpose(1, 2) @ deformation_gradient
        )
        identity = torch.eye(
            3,
            dtype=corrected_positions.dtype,
            device=corrected_positions.device,
        )[None]
        shape = torch.mean((right_cauchy_green - identity) ** 2)

        residual_scale_sq = self.settings.maximum_residual_m**2
        spatial = torch.mean(
            (residual[self.edges[:, 1]] - residual[self.edges[:, 0]]) ** 2
        ) / residual_scale_sq
        magnitude = torch.mean(residual**2) / residual_scale_sq
        return {
            "distance": distance,
            "volume": volume,
            "shape": shape,
            "spatial": spatial,
            "magnitude": magnitude,
            "volume_ratio": volume_ratio,
        }

    def physical_quality_metrics(
        self, positions: torch.Tensor
    ) -> dict[str, float | int]:
        """Evaluate geometry/constraint diagnostics without a visual solve."""
        positions = positions.detach().to(
            device=self.rest_positions.device, dtype=torch.float32
        )
        if positions.shape != self.rest_positions.shape:
            raise ValueError("Physical metric positions have the wrong shape")
        residual = positions - positions
        with torch.no_grad():
            losses = self._constraint_losses(positions, residual)
            ratios = losses["volume_ratio"]
        return {
            "distance_loss": float(losses["distance"].item()),
            "volume_loss": float(losses["volume"].item()),
            "shape_loss": float(losses["shape"].item()),
            "minimum_volume_ratio": float(ratios.min().item()),
            "inverted_tetrahedra": int(
                torch.count_nonzero(ratios <= 0.0).item()
            ),
        }

    def _visual_loss(
        self,
        gaussian_means: torch.Tensor,
        render_colors: Callable[..., torch.Tensor],
        target_colors: torch.Tensor,
        pixel_weights: torch.Tensor,
        gaussian_covariances: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rendered = (
            render_colors(gaussian_means)
            if gaussian_covariances is None
            else render_colors(gaussian_means, gaussian_covariances)
        )
        return equal_camera_visual_loss(
            rendered,
            target_colors,
            pixel_weights,
            robust_loss_beta=self.settings.robust_loss_beta,
        )

    def _candidate_gaussian_geometry(
        self,
        *,
        physical_positions: torch.Tensor,
        residual: torch.Tensor,
        base_gaussian_means: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.deformation_covariance_enabled:
            return self.deformed_gaussian_geometry(
                physical_positions + residual
            )
        return (
            self.corrected_gaussian_means(base_gaussian_means, residual),
            None,
        )

    def solve(
        self,
        *,
        physical_positions: torch.Tensor,
        base_gaussian_means: torch.Tensor,
        target_colors: torch.Tensor,
        pixel_weights: torch.Tensor,
        render_colors: Callable[..., torch.Tensor],
        previous_residual: torch.Tensor | None = None,
        dynamic_exclusion_mask: torch.Tensor | None = None,
    ) -> VisualTissueResidualMappingResult:
        settings = self.settings
        device = self.rest_positions.device
        physical_positions = physical_positions.detach().to(
            device=device, dtype=torch.float32
        )
        base_gaussian_means = base_gaussian_means.detach().to(
            device=device, dtype=torch.float32
        )
        target_colors = target_colors.detach().to(device=device, dtype=torch.float32)
        pixel_weights = pixel_weights.detach().to(device=device, dtype=torch.float32)
        if physical_positions.shape != self.rest_positions.shape:
            raise ValueError("Physical particle state has the wrong shape")
        if dynamic_exclusion_mask is None:
            dynamic_exclusion = torch.zeros_like(self.fixed_mask)
        else:
            dynamic_exclusion = dynamic_exclusion_mask.detach().to(
                device=device, dtype=torch.bool
            )
            if dynamic_exclusion.shape != self.fixed_mask.shape:
                raise ValueError(
                    "Dynamic residual exclusion mask has the wrong shape"
                )
        # Fixed particles and live kinematic controls are both immutable to
        # this solve.  The latter changes with capture/release, so it cannot be
        # baked into ``self.fixed_mask`` at mapper construction time.
        solve_exclusion = self.fixed_mask | dynamic_exclusion
        packed_soft_gaussians = (
            len(base_gaussian_means) == len(self.soft_gaussian_ids)
        )
        if not packed_soft_gaussians and self.soft_gaussian_ids.numel() and (
            int(self.soft_gaussian_ids.min()) < 0
            or int(self.soft_gaussian_ids.max()) >= len(base_gaussian_means)
        ):
            raise ValueError("Soft Gaussian id is outside the rendered model")
        if previous_residual is None:
            temporal_reference = torch.zeros_like(physical_positions)
        else:
            temporal_reference = previous_residual.detach().to(
                device=device, dtype=torch.float32
            )
            if temporal_reference.shape != physical_positions.shape:
                raise ValueError("Previous residual has the wrong shape")
            temporal_reference = self._clip_vectors(
                temporal_reference, settings.maximum_residual_m
            )
        temporal_reference = temporal_reference.masked_fill(
            solve_exclusion[:, None], 0.0
        )
        # Each online solve estimates an *increment* from the current physical
        # state.  Starting from the previous already-applied correction would
        # add it a second time and cause a static observation to drift.  The
        # previous result is therefore only a temporal regularization target.
        residual = torch.zeros_like(physical_positions).requires_grad_(True)
        optimizer = torch.optim.Adam([residual], lr=settings.learning_rate_m)

        # A visual-only gradient supplies a conservative node-level
        # supervision gate for material adaptation.  It already includes both
        # camera masks, tool occlusion and Gaussian visibility through the
        # renderer; physical regularizers cannot make an invisible node appear
        # supervised here.
        supervision_probe = torch.zeros_like(physical_positions).requires_grad_(
            True
        )
        initial_means, initial_covariances = self._candidate_gaussian_geometry(
            physical_positions=physical_positions,
            residual=supervision_probe,
            base_gaussian_means=base_gaussian_means,
        )
        initial_visual, initial_camera_visual = self._visual_loss(
            initial_means,
            render_colors,
            target_colors,
            pixel_weights,
            initial_covariances,
        )
        (visual_gradient,) = torch.autograd.grad(
            initial_visual, supervision_probe, retain_graph=False
        )
        visual_gradient_norm = torch.linalg.vector_norm(
            visual_gradient.detach(), dim=1
        )
        visual_gradient_norm[solve_exclusion] = 0.0
        camera_weight_sums = pixel_weights.sum(dim=(1, 2)).detach()
        camera_active_pixel_counts = torch.count_nonzero(
            pixel_weights > 0.0, dim=(1, 2)
        ).detach()
        pixels_per_camera = pixel_weights.shape[1] * pixel_weights.shape[2]
        camera_mask_coverage_fractions = (
            camera_active_pixel_counts.to(dtype=torch.float32)
            / float(pixels_per_camera)
        )

        for _ in range(settings.iterations):
            optimizer.zero_grad(set_to_none=True)
            corrected_positions = physical_positions + residual
            gaussian_means, gaussian_covariances = (
                self._candidate_gaussian_geometry(
                    physical_positions=physical_positions,
                    residual=residual,
                    base_gaussian_means=base_gaussian_means,
                )
            )
            visual, _ = self._visual_loss(
                gaussian_means,
                render_colors,
                target_colors,
                pixel_weights,
                gaussian_covariances,
            )
            constraints = self._constraint_losses(
                corrected_positions, residual
            )
            temporal = torch.mean(
                (residual - temporal_reference) ** 2
            ) / (settings.maximum_residual_m**2)
            total = (
                settings.data_weight * visual
                + settings.distance_weight * constraints["distance"]
                + settings.volume_weight * constraints["volume"]
                + settings.shape_weight * constraints["shape"]
                + settings.spatial_weight * constraints["spatial"]
                + settings.temporal_weight * temporal
                + settings.magnitude_weight * constraints["magnitude"]
            )
            total.backward()
            if residual.grad is None or not bool(
                torch.isfinite(residual.grad).all().item()
            ):
                raise RuntimeError("Visual residual gradient is non-finite")
            residual.grad[solve_exclusion] = 0.0
            optimizer.step()
            with torch.no_grad():
                residual.copy_(
                    self._clip_vectors(residual, settings.maximum_residual_m)
                )
                residual[solve_exclusion] = 0.0
        candidate = residual.detach()
        backtrack_count = 0
        with torch.no_grad():
            initial_volume_ratio = (
                _tetra_volumes(physical_positions, self.tet_indices)
                / self.rest_volumes
            )
            relative_floor = initial_volume_ratio * (
                1.0 - settings.maximum_relative_volume_loss_below_floor
            )
            required_volume_ratio = torch.where(
                initial_volume_ratio >= settings.minimum_volume_ratio,
                torch.full_like(
                    initial_volume_ratio, settings.minimum_volume_ratio
                ),
                relative_floor,
            )
            locally_frozen_mask = torch.zeros_like(self.fixed_mask)
            local_projection_passes = 0
            for local_projection_passes in range(
                settings.maximum_local_volume_projection_passes + 1
            ):
                corrected_positions = physical_positions + candidate
                volume_ratio = (
                    _tetra_volumes(corrected_positions, self.tet_indices)
                    / self.rest_volumes
                )
                violating_tetrahedra = (
                    ~torch.isfinite(volume_ratio)
                    | (volume_ratio < required_volume_ratio)
                    | (
                        (initial_volume_ratio > 0.0)
                        & (volume_ratio <= 0.0)
                    )
                )
                if not bool(violating_tetrahedra.any().item()):
                    break
                if (
                    local_projection_passes
                    == settings.maximum_local_volume_projection_passes
                ):
                    break
                violating_particles = torch.unique(
                    self.tet_indices[violating_tetrahedra].reshape(-1)
                )
                locally_frozen_mask[violating_particles] = True
                candidate = candidate.clone()
                candidate[locally_frozen_mask] = 0.0
            for backtrack_count in range(settings.maximum_backtracks + 1):
                corrected_positions = physical_positions + candidate
                volumes = _tetra_volumes(corrected_positions, self.tet_indices)
                volume_ratio = volumes / self.rest_volumes
                newly_inverted = (initial_volume_ratio > 0.0) & (
                    volume_ratio <= 0.0
                )
                valid = bool(
                    torch.isfinite(volume_ratio).all().item()
                    and not torch.any(newly_inverted).item()
                    and torch.all(
                        volume_ratio >= required_volume_ratio
                    ).item()
                )
                if valid:
                    break
                candidate = candidate * 0.5
            else:
                candidate = torch.zeros_like(candidate)
                backtrack_count = settings.maximum_backtracks + 1

            candidate[solve_exclusion] = 0.0
            corrected_positions = physical_positions + candidate
            corrected_means, corrected_covariances = (
                self._candidate_gaussian_geometry(
                    physical_positions=physical_positions,
                    residual=candidate,
                    base_gaussian_means=base_gaussian_means,
                )
            )
            final_visual, final_camera_visual = self._visual_loss(
                corrected_means,
                render_colors,
                target_colors,
                pixel_weights,
                corrected_covariances,
            )
            final_constraints = self._constraint_losses(
                corrected_positions, candidate
            )
            temporal = torch.mean(
                (candidate - temporal_reference) ** 2
            ) / (settings.maximum_residual_m**2)
            final_total = (
                settings.data_weight * final_visual
                + settings.distance_weight * final_constraints["distance"]
                + settings.volume_weight * final_constraints["volume"]
                + settings.shape_weight * final_constraints["shape"]
                + settings.spatial_weight * final_constraints["spatial"]
                + settings.temporal_weight * temporal
                + settings.magnitude_weight * final_constraints["magnitude"]
            )
            final_volume_ratio = final_constraints["volume_ratio"]
            residual_norm = torch.linalg.vector_norm(candidate, dim=1)
            inverted = int(torch.count_nonzero(final_volume_ratio <= 0.0).item())
            initial_inverted = int(
                torch.count_nonzero(initial_volume_ratio <= 0.0).item()
            )
            newly_inverted = int(
                torch.count_nonzero(
                    (initial_volume_ratio > 0.0) & (final_volume_ratio <= 0.0)
                ).item()
            )

        return VisualTissueResidualMappingResult(
            residual=candidate,
            corrected_positions=corrected_positions,
            corrected_gaussian_means=corrected_means,
            initial_visual_loss=float(initial_visual.item()),
            final_visual_loss=float(final_visual.item()),
            initial_camera_visual_losses=tuple(
                float(value)
                for value in initial_camera_visual.detach().cpu().tolist()
            ),
            final_camera_visual_losses=tuple(
                float(value)
                for value in final_camera_visual.detach().cpu().tolist()
            ),
            camera_weight_sums=tuple(
                float(value)
                for value in camera_weight_sums.cpu().tolist()
            ),
            camera_active_pixel_counts=tuple(
                int(value)
                for value in camera_active_pixel_counts.cpu().tolist()
            ),
            camera_mask_coverage_fractions=tuple(
                float(value)
                for value in camera_mask_coverage_fractions.cpu().tolist()
            ),
            visual_gradient_norm=visual_gradient_norm,
            final_total_loss=float(final_total.item()),
            final_distance_loss=float(final_constraints["distance"].item()),
            final_volume_loss=float(final_constraints["volume"].item()),
            final_shape_loss=float(final_constraints["shape"].item()),
            final_spatial_loss=float(final_constraints["spatial"].item()),
            final_temporal_loss=float(temporal.item()),
            final_magnitude_loss=float(final_constraints["magnitude"].item()),
            maximum_residual_m=float(residual_norm.max().item()),
            rms_residual_m=float(torch.sqrt(torch.mean(residual_norm**2)).item()),
            minimum_volume_ratio=float(final_volume_ratio.min().item()),
            initial_minimum_volume_ratio=float(
                initial_volume_ratio.min().item()
            ),
            inverted_tetrahedra=inverted,
            initial_inverted_tetrahedra=initial_inverted,
            newly_inverted_tetrahedra=newly_inverted,
            dynamically_excluded_particles=int(
                torch.count_nonzero(
                    dynamic_exclusion & ~self.fixed_mask
                ).item()
            ),
            locally_frozen_particles=int(
                torch.count_nonzero(locally_frozen_mask).item()
            ),
            local_volume_projection_passes=local_projection_passes,
            backtrack_count=backtrack_count,
        )
