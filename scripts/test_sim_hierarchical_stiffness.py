#!/usr/bin/env python3
"""CPU gate for joint global plus zero-mean regional stiffness."""

from __future__ import annotations

import json

import torch

from embodied_gaussians.physics_simulator.online_tissue_stiffness import (
    OnlineTissueStiffnessSettings,
    ResidualDrivenPaperStiffnessUpdater,
)


def main() -> None:
    rest = torch.tensor(
        (
            (0.000, 0.000, 0.000),
            (0.001, 0.000, 0.000),
            (0.002, 0.000, 0.000),
            (0.000, 0.001, 0.000),
            (0.001, 0.001, 0.000),
            (0.002, 0.001, 0.000),
        ),
        dtype=torch.float32,
    )
    edges = torch.tensor(
        ((0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)),
        dtype=torch.long,
    )
    updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=torch.tensor((True, False, False, False, False, False)),
        edges=edges,
        distance_stiffness=torch.full((6,), 0.20),
        shape_stiffness=torch.full((6,), 0.004),
        track_particle_ids=torch.arange(6, dtype=torch.long)[:, None],
        track_particle_weights=torch.ones((6, 1), dtype=torch.float32),
        track_valid_mask=torch.ones(6, dtype=torch.bool),
        settings=OnlineTissueStiffnessSettings(
            update_mode="differentiable_hierarchical_relative",
            autograd_unroll_steps=5,
            autograd_region_count=3,
            spatial_smoothing_iterations=1,
            autograd_material_iterations=1,
            autograd_integration_substeps=1,
        ),
    )
    coefficients = torch.tensor((0.10, 0.20, -0.05, -0.15))
    normalized = updater.normalize_local_coefficients(coefficients)
    local = updater.local_zero_mean_log_field(normalized)
    distance = updater.local_distance_from_coefficients(normalized)
    dynamic = ~updater.fixed_mask
    weighted_local_mean = torch.mean(local[dynamic])

    raw_gradient = torch.tensor((0.7, 0.3, -0.1, 0.8))
    projected_gradient = updater.project_local_gradient(raw_gradient)
    candidate = None
    for frame in range(5):
        prediction = rest.clone()
        prediction[1:, 2] += 0.00025 + frame * 0.00002
        accepted = torch.zeros_like(rest)
        accepted[1:, 2] -= torch.linspace(0.00002, 0.00010, 5)
        candidate = updater.propose(
            physical_prediction=prediction,
            accepted_residual=accepted,
            rollout_start_positions=prediction,
            physical_velocities=torch.zeros_like(rest),
            track_target_positions=prediction + accepted,
            track_valid_mask=torch.ones(6, dtype=torch.bool),
            frame_index=frame,
        )
        if frame < 4:
            updater.reject("cpu_warmup", candidate)
    assert candidate is not None
    assert candidate.autograd_parameter_gradient is not None
    updater.apply_gradient_consistency_gate(
        candidate,
        candidate.autograd_parameter_gradient,
    )
    short_gradient = updater.project_local_gradient(
        torch.tensor((1.0, 0.50, -0.25, -0.25))
    )
    long_gradient = -short_gradient
    updater.replace_candidate_step_with_warp_gradient(
        candidate,
        long_gradient,
        short_horizon_gradient=short_gradient,
    )
    short_pre = float(
        candidate.metrics["short_constraint_pre_directional_derivative"]
    )
    short_target = float(
        candidate.metrics["short_constraint_target_directional_derivative"]
    )
    short_post = float(
        candidate.metrics["short_constraint_post_directional_derivative"]
    )
    gates = {
        "one_global_plus_three_regions": updater.low_dimensional_log_coefficients.shape
        == (4,),
        "global_axis_preserved": bool(torch.isclose(normalized[0], coefficients[0])),
        "regional_coefficients_zero_mean": bool(
            torch.abs(normalized[1:].mean()) < 1.0e-7
        ),
        "particle_local_field_zero_mean": bool(
            torch.abs(weighted_local_mean) < 1.0e-6
        ),
        "global_and_regions_both_change_field": bool(
            torch.unique(torch.round(distance[dynamic] * 1.0e7)).numel() > 1
            and torch.exp(torch.mean(torch.log(distance[dynamic] / 0.20)))
            > 1.0
        ),
        "fixed_particle_keeps_common_initial_value": bool(
            torch.isclose(distance[0], torch.tensor(0.20))
        ),
        "gradient_global_axis_not_centered_away": bool(
            torch.isclose(projected_gradient[0], raw_gradient[0])
        ),
        "gradient_regional_axes_zero_mean": bool(
            torch.abs(projected_gradient[1:].mean()) < 1.0e-7
        ),
        "causal_candidate_has_joint_gradient": bool(
            candidate.autograd_parameter_gradient is not None
            and candidate.autograd_parameter_gradient.shape == (4,)
            and torch.isfinite(candidate.autograd_parameter_gradient).all()
        ),
        "causal_candidate_has_joint_parameter_step": bool(
            candidate.low_dimensional_log_coefficients is not None
            and candidate.low_dimensional_log_coefficients.shape == (4,)
            and updater.candidate_parameter_step_maximum(candidate) > 0.0
        ),
        "long_horizon_gradient_drives_adam": bool(
            torch.dot(short_gradient, long_gradient) < 0.0
        ),
        "h1_projection_makes_step_short_descent": bool(
            short_post <= short_target + 1.0e-7 and short_post < 0.0
        ),
        "projected_step_respects_cap": bool(
            candidate.metrics["warp_adam_step_maximum"]
            <= updater.settings.maximum_log_step + 1.0e-7
        ),
    }
    report = {
        "normalized_coefficients": normalized.tolist(),
        "distance_stiffness": distance.tolist(),
        "weighted_local_log_mean": float(weighted_local_mean.item()),
        "projected_gradient": projected_gradient.tolist(),
        "long_horizon_gradient": long_gradient.tolist(),
        "causal_candidate_coefficients": (
            None
            if candidate.low_dimensional_log_coefficients is None
            else candidate.low_dimensional_log_coefficients.tolist()
        ),
        "causal_candidate_gradient": (
            None
            if candidate.autograd_parameter_gradient is None
            else candidate.autograd_parameter_gradient.tolist()
        ),
        "causal_candidate_metrics": candidate.metrics,
        "short_constraint": {
            "pre_directional_derivative": short_pre,
            "target_directional_derivative": short_target,
            "post_directional_derivative": short_post,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
