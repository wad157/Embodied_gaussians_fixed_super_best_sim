#!/usr/bin/env python3
"""验证绑定三角面形变会更新高斯尺度，且运行时与可微映射一致。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import warp as wp


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embodied_gaussians.embodied_simulator.builder import (  # noqa: E402
    EmbodiedGaussiansBuilder,
)
from embodied_gaussians.embodied_simulator.simulator import (  # noqa: E402
    update_gaussian_transforms,
)
from embodied_gaussians.physics_simulator.visual_tissue_residual_mapping import (  # noqa: E402
    TetrahedralGaussianVisualResidualMapper,
)
from embodied_gaussians.scene_builders.domain import SoftBody  # noqa: E402


DEFAULT_ASSET = (
    REPO_ROOT
    / "data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm/"
    "gui_assets/tissue_fixedsuperbest.npz"
)


def covariance(quats: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    rotations = TetrahedralGaussianVisualResidualMapper._quaternion_matrices_wxyz(
        quats
    )
    return (
        rotations
        @ torch.diag_embed(scales.square())
        @ rotations.transpose(-1, -2)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument(
        "--warp-cache-dir",
        type=Path,
        default=Path("/tmp/warp-sim-gaussian-triangle-scale"),
    )
    args = parser.parse_args()
    wp.config.kernel_cache_dir = str(args.warp_cache_dir)
    wp.init()

    body = SoftBody.from_npz(args.asset, name="sim_tissue")
    builder = EmbodiedGaussiansBuilder()
    builder.add_soft_body(
        body,
        young_modulus_pa=950.0,
        poisson_ratio=0.45,
        anchor_mode="support_candidate",
        add_gaussians=True,
    )
    model = builder.finalize(device="cpu")
    gaussian_model = builder.gaussian_model
    state = gaussian_model.state()
    physics_state = model.state()
    particle_q = wp.to_torch(physics_state.particle_q)
    soft_ids = gaussian_model.soft_gaussian_ids.long()

    mapper = TetrahedralGaussianVisualResidualMapper.from_visual_face_centroid_bindings(
        rest_positions=wp.to_torch(model.particle_q).detach().clone(),
        tet_indices=wp.to_torch(model.tet_indices).long().reshape(-1, 4),
        fixed_mask=wp.to_torch(model.particle_inv_mass) == 0.0,
        soft_gaussian_ids=soft_ids,
        visual_vertex_particle_indices=(
            gaussian_model.soft_gaussian_visual_vertex_particle_indices
        ),
        visual_vertex_weights=gaussian_model.soft_gaussian_visual_vertex_weights,
        visual_vertex_rest_offsets=(
            gaussian_model.soft_gaussian_visual_vertex_rest_offsets
        ),
        visual_vertex_rest_physical_frames=(
            gaussian_model.soft_gaussian_visual_vertex_rest_physical_frames
        ),
        rest_visual_face_poses=(
            gaussian_model.soft_gaussian_rest_visual_face_poses
        ),
        rest_gaussian_quats=gaussian_model.quats[soft_ids],
        rest_gaussian_scales=gaussian_model.scales[soft_ids],
    )

    def skin() -> None:
        update_gaussian_transforms(
            gaussian_model,
            physics_state.body_q,
            state,
            particle_q=physics_state.particle_q,
        )
        wp.synchronize_device(model.device)

    skin()
    rest_means, rest_covariances = mapper.deformed_gaussian_geometry(particle_q)
    runtime_rest_covariances = covariance(
        state.quats[soft_ids], state.scales[soft_ids]
    )
    rest_mean_error = torch.linalg.vector_norm(
        state.means[soft_ids] - rest_means, dim=1
    ).max()
    rest_covariance_error = torch.linalg.matrix_norm(
        runtime_rest_covariances - rest_covariances, dim=(-2, -1)
    ).max()

    center = particle_q.mean(dim=0, keepdim=True)
    deformation = torch.tensor(
        ((1.12, 0.08, 0.00), (0.00, 0.91, 0.00), (0.00, 0.00, 1.00)),
        dtype=torch.float32,
    )
    deformed_particles = (particle_q - center) @ deformation.T + center
    particle_q.copy_(deformed_particles)
    skin()
    mapped_means, mapped_covariances = mapper.deformed_gaussian_geometry(
        deformed_particles
    )
    runtime_covariances = covariance(
        state.quats[soft_ids], state.scales[soft_ids]
    )
    deformed_mean_error = torch.linalg.vector_norm(
        state.means[soft_ids] - mapped_means, dim=1
    ).max()
    deformed_covariance_error = torch.linalg.matrix_norm(
        runtime_covariances - mapped_covariances, dim=(-2, -1)
    ).max()
    scale_change = torch.linalg.vector_norm(
        state.scales[soft_ids] - gaussian_model.scales[soft_ids], dim=1
    )

    gates = {
        "mapper_uses_deformation_covariance": mapper.deformation_covariance_enabled,
        "rest_geometry_matches_runtime": bool(
            rest_mean_error < 2.0e-7 and rest_covariance_error < 1.0e-9
        ),
        "deformed_geometry_matches_runtime": bool(
            deformed_mean_error < 2.0e-7
            and deformed_covariance_error < 1.0e-9
        ),
        "triangle_deformation_changes_scales": bool(
            scale_change.mean() > 1.0e-6
        ),
        "all_geometry_finite": bool(
            torch.isfinite(state.means).all()
            and torch.isfinite(state.quats).all()
            and torch.isfinite(state.scales).all()
            and torch.isfinite(mapped_covariances).all()
        ),
    }
    report = {
        "stage": "sim_gaussian_triangle_scale_gate",
        "asset": str(args.asset.resolve()),
        "gaussians": int(len(soft_ids)),
        "rest_mean_max_error_m": float(rest_mean_error.item()),
        "rest_covariance_max_error_m2": float(rest_covariance_error.item()),
        "deformed_mean_max_error_m": float(deformed_mean_error.item()),
        "deformed_covariance_max_error_m2": float(
            deformed_covariance_error.item()
        ),
        "scale_change_mean_m": float(scale_change.mean().item()),
        "scale_change_max_m": float(scale_change.max().item()),
        "particle_velocity_update": False,
        "gates": gates,
        "passed": all(gates.values()),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
