#!/usr/bin/env python3
"""CPU gate for SUPER-inspired local zero-mean causal stiffness updates."""

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
            (0.0000, 0.0000, 0.0000),
            (0.0010, 0.0000, 0.0000),
            (0.0000, 0.0010, 0.0000),
            (0.0000, 0.0000, 0.0010),
            (0.0010, 0.0010, 0.0010),
        ),
        dtype=torch.float32,
    )
    edges = torch.tensor(
        (
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3), (2, 3),
            (1, 4), (2, 4), (3, 4),
        ),
        dtype=torch.long,
    )
    tets = torch.tensor(((0, 1, 2, 3), (1, 2, 3, 4)), dtype=torch.long)
    fixed = torch.tensor((True, False, False, False, False))
    inverse_mass = torch.tensor((0.0, 2.0e5, 1.8e5, 2.2e5, 2.0e5))
    distance = torch.full((5,), 0.20)
    shape = torch.full((5,), 0.004)
    settings = OnlineTissueStiffnessSettings(
        update_mode="differentiable_local_relative",
        log_learning_rate=0.03,
        maximum_log_step=0.02,
        signal_ema_decay=0.90,
        spatial_smoothing_iterations=3,
        spatial_smoothing_blend=0.35,
        strain_signal_weight=0.20,
        autograd_unroll_steps=5,
        autograd_region_count=3,
        autograd_prior_weight=0.02,
        autograd_frame_dt=1.0 / 30.0,
        autograd_material_iterations=2,
        local_update_interval_frames=5,
        local_spatial_prior_weight=0.05,
    )
    updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=fixed,
        edges=edges,
        distance_stiffness=distance,
        shape_stiffness=shape,
        inverse_mass=inverse_mass,
        tet_indices=tets,
        track_particle_ids=torch.arange(5, dtype=torch.long)[:, None],
        track_particle_weights=torch.ones((5, 1), dtype=torch.float32),
        track_valid_mask=torch.ones(5, dtype=torch.bool),
        initial_velocity_damping_per_second=12.0,
        initial_coupling_gain=1.0,
        settings=settings,
    )
    residual = torch.zeros_like(rest)
    residual[1:4, 2] = torch.tensor((0.00002, 0.00005, 0.00008))
    control = torch.tensor((False, False, False, False, True))
    candidate = None
    for frame in range(5):
        start = rest.clone()
        start[1:4, 1] += 0.00001 * frame
        velocity = torch.zeros_like(rest)
        velocity[1:4, 1] = 0.0003
        prediction = start + velocity / 30.0
        target = prediction + residual
        candidate = updater.propose(
            physical_prediction=prediction,
            accepted_residual=residual,
            quality_valid_mask=torch.ones(5, dtype=torch.bool),
            supervision_valid_mask=torch.ones(5, dtype=torch.bool),
            control_exclusion_mask=control,
            control_frozen_mask=control,
            control_coupling_base_positions=start,
            control_coupling_displacements=torch.zeros_like(rest),
            track_target_positions=target,
            track_valid_mask=torch.ones(5, dtype=torch.bool),
            rollout_start_positions=start,
            physical_velocities=velocity,
            frame_index=frame,
        )
        if frame < 4:
            assert candidate.metrics["autograd_history_status"].startswith(
                "warming_up"
            )
            updater.reject("causal_history_warmup", candidate)

    assert candidate is not None
    gradient = candidate.autograd_parameter_gradient
    assert gradient is not None
    assert torch.isfinite(gradient).all() and torch.linalg.vector_norm(gradient) > 0
    before_distance = distance.clone()
    before_shape = shape.clone()
    before_damping = updater.global_velocity_damping_per_second
    allowed, cosine = updater.apply_gradient_consistency_gate(
        candidate, gradient.clone()
    )
    assert allowed and abs(cosine - 1.0) < 1.0e-6
    updater.replace_candidate_step_with_warp_gradient(candidate, gradient.clone())
    log_field = updater.local_zero_mean_log_field(
        candidate.low_dimensional_log_coefficients
    )
    dynamic_weights = torch.where(
        ~updater.local_parameter_exclusion_mask,
        inverse_mass,
        torch.zeros_like(inverse_mass),
    )
    weighted_mean = torch.sum(log_field * dynamic_weights) / dynamic_weights.sum()
    updater.commit(candidate)
    gates = {
        "five_frame_h1_h3_h5_pre_residual_window": bool(
            candidate.metrics["autograd_temporal_steps"] == 5
            and candidate.metrics["local_active_horizons"] == "1,3,5"
            and "pre_residual_h1_h3_h5"
            in candidate.metrics["local_rollout_objective"]
        ),
        "regional_warp_gradient_drives_adam": bool(
            candidate.metrics["optimizer_gradient_source"]
            == "warp_finite_difference_local_h1_h3_h5"
            and updater.adam_step == 1
        ),
        "particle_log_field_has_zero_weighted_mean": bool(
            abs(float(weighted_mean.item())) < 1.0e-6
            and abs(candidate.metrics["local_weighted_particle_log_mean"])
            < 1.0e-6
        ),
        "fixed_and_grasp_particles_keep_reset_material": bool(
            distance[0] == before_distance[0]
            and distance[4] == before_distance[4]
        ),
        "local_dynamic_material_changes": bool(
            not torch.equal(distance[1:4], before_distance[1:4])
            and float(distance[1:4].max() - distance[1:4].min()) > 0.0
        ),
        "shape_and_damping_are_frozen": bool(
            torch.equal(shape, before_shape)
            and updater.global_velocity_damping_per_second == before_damping
            and candidate.metrics["local_velocity_damping_frozen"] == 1
        ),
        "single_causal_optimizer_without_branch_selection": bool(
            settings.local_update_interval_frames == 5
            and candidate.metrics["local_global_mean_frozen"] == 1
        ),
    }
    report = {
        "stage": "differentiable_local_relative_gate",
        "candidate_metrics": candidate.metrics,
        "weighted_particle_log_mean": float(weighted_mean.item()),
        "distance": distance.tolist(),
        "gates": gates,
        "passed": all(gates.values()),
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
