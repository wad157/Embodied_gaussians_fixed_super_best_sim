#!/usr/bin/env python3
"""Deterministic synthetic gate for Gaussian visual residual mapping."""

from __future__ import annotations

import json

import torch

from embodied_gaussians.physics_simulator.visual_tissue_residual_mapping import (
    TetrahedralGaussianVisualResidualMapper,
    VisualTissueResidualMappingSettings,
    strict_one_tetrahedron_ring_mask,
)


def main() -> None:
    torch.manual_seed(420)
    rest = torch.tensor(
        (
            (0.000, 0.000, 0.000),
            (0.004, 0.000, 0.000),
            (0.000, 0.004, 0.000),
            (0.000, 0.000, 0.004),
        ),
        dtype=torch.float32,
    )
    tets = torch.tensor(((0, 1, 2, 3),), dtype=torch.long)
    fixed = torch.tensor((True, False, False, False))
    gaussian_ids = torch.tensor((0,), dtype=torch.long)
    # Three visual vertices, each embedded into the same top physical face.
    supports = torch.tensor(((0, 1, 2, 0, 1, 2, 0, 1, 2),), dtype=torch.long)
    vertex_weights = torch.tensor(
        ((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),),
        dtype=torch.float32,
    )
    settings = VisualTissueResidualMappingSettings(
        iterations=12,
        learning_rate_m=4.0e-5,
        image_scale=1.0,
        robust_loss_beta=0.0,
        distance_weight=0.01,
        volume_weight=0.01,
        shape_weight=0.001,
        spatial_weight=0.001,
        temporal_weight=0.0,
        magnitude_weight=0.0001,
        maximum_residual_m=0.0008,
    )
    mapper = TetrahedralGaussianVisualResidualMapper.from_visual_face_centroid_bindings(
        rest_positions=rest,
        tet_indices=tets,
        fixed_mask=fixed,
        soft_gaussian_ids=gaussian_ids,
        visual_vertex_particle_indices=supports,
        visual_vertex_weights=vertex_weights,
        settings=settings,
    )
    base_means = torch.tensor(((0.0013333333, 0.0013333333, 0.0),))
    target_shift = torch.tensor((0.00030, -0.00018, 0.00010))

    def render_colors(means: torch.Tensor) -> torch.Tensor:
        # A small differentiable stand-in for a calibrated image projection.
        value = 0.5 + 120.0 * means[0]
        return value.reshape(1, 1, 1, 3)

    target_colors = render_colors(base_means + target_shift).detach()
    weights = torch.ones((1, 1, 1), dtype=torch.float32)
    result = mapper.solve(
        physical_positions=rest,
        base_gaussian_means=base_means,
        target_colors=target_colors,
        pixel_weights=weights,
        render_colors=render_colors,
    )
    dynamic_exclusion = torch.tensor((False, True, False, False))
    dynamically_excluded_result = mapper.solve(
        physical_positions=rest,
        base_gaussian_means=base_means,
        target_colors=target_colors,
        pixel_weights=weights,
        render_colors=render_colors,
        dynamic_exclusion_mask=dynamic_exclusion,
    )
    chain_tets = torch.tensor(
        ((0, 1, 2, 3), (3, 4, 5, 6)), dtype=torch.long
    )
    strict_ring = strict_one_tetrahedron_ring_mask(
        torch.tensor((True, False, False, False, False, False, False)),
        chain_tets,
    )
    compressed_positions = rest.clone()
    compressed_positions[3, 2] *= 0.01
    compressed_result = mapper.solve(
        physical_positions=compressed_positions,
        base_gaussian_means=base_means,
        target_colors=target_colors,
        pixel_weights=weights,
        render_colors=render_colors,
    )
    achieved_shift = (
        result.corrected_gaussian_means[0] - base_means[0]
    )
    gates = {
        "visual_loss_decreased": result.final_visual_loss < result.initial_visual_loss,
        "fixed_particle_residual_is_zero": bool(
            torch.equal(result.residual[fixed], torch.zeros_like(result.residual[fixed]))
        ),
        "dynamic_grip_exclusion_is_immutable": bool(
            torch.equal(
                dynamically_excluded_result.residual[dynamic_exclusion],
                torch.zeros_like(
                    dynamically_excluded_result.residual[dynamic_exclusion]
                ),
            )
            and torch.equal(
                dynamically_excluded_result.visual_gradient_norm[
                    dynamic_exclusion
                ],
                torch.zeros_like(
                    dynamically_excluded_result.visual_gradient_norm[
                        dynamic_exclusion
                    ]
                ),
            )
            and dynamically_excluded_result.dynamically_excluded_particles
            == 1
            and dynamically_excluded_result.maximum_residual_m > 0.0
        ),
        "grip_neighborhood_is_exactly_one_tet_ring": bool(
            torch.equal(
                strict_ring,
                torch.tensor(
                    (True, True, True, True, False, False, False)
                ),
            )
        ),
        "residual_cap_respected": result.maximum_residual_m
        <= settings.maximum_residual_m + 1.0e-9,
        "tetrahedron_remains_positive": result.inverted_tetrahedra == 0
        and result.minimum_volume_ratio > 0.0,
        "visual_residual_moves_in_target_direction": float(
            torch.dot(achieved_shift, target_shift)
        )
        > 0.0,
        "precompressed_tet_does_not_disable_global_residual": (
            compressed_result.final_visual_loss
            < compressed_result.initial_visual_loss
            and compressed_result.maximum_residual_m > 0.0
            and compressed_result.newly_inverted_tetrahedra == 0
            and compressed_result.minimum_volume_ratio
            >= compressed_result.initial_minimum_volume_ratio
            * (1.0 - settings.maximum_relative_volume_loss_below_floor)
            - 1.0e-6
        ),
        "per_camera_losses_and_weights_recorded": bool(
            len(result.initial_camera_visual_losses) == 1
            and len(result.final_camera_visual_losses) == 1
            and result.camera_weight_sums == (1.0,)
            and result.camera_active_pixel_counts == (1,)
            and result.camera_mask_coverage_fractions == (1.0,)
        ),
        "visual_supervision_gradient_recorded": bool(
            result.visual_gradient_norm.shape == (4,)
            and torch.isfinite(result.visual_gradient_norm).all()
            and torch.any(result.visual_gradient_norm[~fixed] > 0.0)
        ),
    }
    report = {
        "stage": "synthetic_gaussian_visual_tissue_residual_mapping_gate",
        "method": (
            "optimize physical-particle residual through a differentiable "
            "Gaussian observation with normalized tetrahedral constraints"
        ),
        "settings": settings.__dict__,
        "initial_visual_loss": result.initial_visual_loss,
        "final_visual_loss": result.final_visual_loss,
        "target_shift_mm": (target_shift * 1000.0).tolist(),
        "achieved_gaussian_shift_mm": (achieved_shift * 1000.0).tolist(),
        "maximum_particle_residual_mm": result.maximum_residual_m * 1000.0,
        "minimum_volume_ratio": result.minimum_volume_ratio,
        "precompressed_tet": {
            "initial_minimum_volume_ratio": (
                compressed_result.initial_minimum_volume_ratio
            ),
            "final_minimum_volume_ratio": (
                compressed_result.minimum_volume_ratio
            ),
            "maximum_particle_residual_mm": (
                compressed_result.maximum_residual_m * 1000.0
            ),
        },
        "backtrack_count": result.backtrack_count,
        "gates": gates,
        "passed": all(gates.values()),
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Visual tissue residual mapping gate failed")


if __name__ == "__main__":
    main()
