#!/usr/bin/env python3
"""CPU gate for global paper-stiffness/damping/coupling identification."""

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
    tets = torch.tensor(((0, 1, 2, 3), (1, 2, 3, 4)), dtype=torch.long)
    edges = torch.tensor(
        (
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3), (2, 3),
            (1, 4), (2, 4), (3, 4),
        ),
        dtype=torch.long,
    )
    fixed = torch.tensor((True, False, False, False, False))
    distance = torch.full((5,), 0.20)
    shape = torch.full((5,), 0.004)
    updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=fixed,
        edges=edges,
        distance_stiffness=distance,
        shape_stiffness=shape,
        inverse_mass=torch.tensor((0.0, 2.0e5, 2.0e5, 2.0e5, 2.0e5)),
        tet_indices=tets,
        track_particle_ids=torch.arange(5, dtype=torch.long)[:, None],
        track_particle_weights=torch.ones((5, 1), dtype=torch.float32),
        track_valid_mask=torch.ones(5, dtype=torch.bool),
        initial_velocity_damping_per_second=12.0,
        initial_coupling_gain=1.0,
        settings=OnlineTissueStiffnessSettings(
            update_mode="differentiable_global",
            log_learning_rate=0.03,
            maximum_log_step=0.02,
            signal_ema_decay=0.90,
            strain_signal_weight=0.20,
            autograd_unroll_steps=4,
            autograd_region_count=3,
            autograd_prior_weight=0.0,
            autograd_frame_dt=1.0 / 30.0,
            autograd_material_iterations=2,
        ),
    )
    candidate = None
    supplied_start = rest.clone()
    supplied_start[1:, 0] += 0.00005
    start_velocity = torch.zeros_like(rest)
    start_velocity[1:, 1] = 0.0005
    residual = torch.zeros_like(rest)
    residual[1:4, 2] = 0.00004
    control = torch.tensor((False, False, False, False, True))
    for frame in range(3):
        prediction = supplied_start + start_velocity * (frame + 1) / 30.0
        candidate = updater.propose(
            physical_prediction=prediction,
            accepted_residual=residual,
            quality_valid_mask=torch.ones(5, dtype=torch.bool),
            supervision_valid_mask=torch.ones(5, dtype=torch.bool),
            control_exclusion_mask=control,
            control_frozen_mask=control,
            rollout_start_positions=supplied_start,
            physical_velocities=start_velocity,
            track_target_positions=prediction + residual,
            track_valid_mask=torch.ones(5, dtype=torch.bool),
            frame_index=frame,
        )
        if frame < 2:
            updater.reject("warmup", candidate)
    assert candidate is not None
    gradient = candidate.autograd_parameter_gradient
    assert gradient is not None
    assert candidate.global_log_coefficients is not None

    # The global validity gate must not mistake a damping-only proposal for an
    # empty update merely because its particle-space distance log_step is zero.
    saved_coefficients = candidate.global_log_coefficients.clone()
    saved_log_step = candidate.log_step.clone()
    candidate.global_log_coefficients.copy_(updater.global_log_coefficients)
    candidate.global_log_coefficients[1] += 0.01
    candidate.log_step.zero_()
    damping_only_step_maximum = updater.candidate_parameter_step_maximum(
        candidate
    )
    candidate.global_log_coefficients.copy_(saved_coefficients)
    candidate.log_step.copy_(saved_log_step)
    full_candidate_step_maximum = updater.candidate_parameter_step_maximum(
        candidate
    )

    unit_gradient = gradient / torch.linalg.vector_norm(gradient)
    basis = torch.tensor((1.0, 0.0, 0.0), dtype=gradient.dtype)
    if torch.abs(torch.dot(unit_gradient, basis)) > 0.90:
        basis = torch.tensor((0.0, 1.0, 0.0), dtype=gradient.dtype)
    orthogonal = basis - torch.dot(basis, unit_gradient) * unit_gradient
    orthogonal = orthogonal / torch.linalg.vector_norm(orthogonal)
    borderline_cosine = 0.92
    borderline_gradient = torch.linalg.vector_norm(gradient) * (
        borderline_cosine * unit_gradient
        + (1.0 - borderline_cosine**2) ** 0.5 * orthogonal
    )
    borderline_allowed, measured_borderline_cosine = (
        updater.apply_gradient_consistency_gate(candidate, borderline_gradient)
    )
    allowed, cosine = updater.apply_gradient_consistency_gate(
        candidate, gradient.clone()
    )
    updater.replace_candidate_step_with_warp_gradient(candidate, gradient.clone())
    before_shape = shape.clone()
    updater.commit(candidate)
    gates = {
        "rollout_start_is_stored_and_used": bool(
            torch.equal(
                updater.causal_material_history[-1].rollout_start_positions,
                supplied_start,
            )
        ),
        "pre_correction_velocity_is_stored": bool(
            torch.equal(
                updater.causal_material_history[-1].physical_velocities,
                start_velocity,
            )
        ),
        "three_global_log_parameters": bool(
            candidate.global_log_coefficients is not None
            and candidate.global_log_coefficients.shape == (3,)
        ),
        "gradient_is_finite_and_nonzero": bool(
            torch.isfinite(gradient).all().item()
            and torch.linalg.vector_norm(gradient) > 0.0
        ),
        "matching_gradient_passes_cosine_gate": bool(
            allowed and abs(cosine - 1.0) < 1.0e-6
        ),
        "cosine_0p92_fails_fixed_0p95_gate": bool(
            not borderline_allowed
            and abs(measured_borderline_cosine - borderline_cosine) < 1.0e-5
            and updater.settings.gradient_cosine_minimum == 0.95
        ),
        "damping_only_candidate_is_valid": bool(
            abs(damping_only_step_maximum - 0.01) < 1.0e-6
            and full_candidate_step_maximum > 0.0
            and candidate.metrics["candidate_validity_step_source"]
            == "global_distance+damping"
        ),
        "paper_shape_remains_fixed": bool(torch.equal(shape, before_shape)),
        "global_state_commits": bool(
            updater.global_adam_step == 1
            and updater.update_count == 1
            and candidate.global_velocity_damping_per_second is not None
            and candidate.global_coupling_gain is not None
        ),
        "balanced_distribution_loss_is_active": bool(
            candidate.metrics["global_position_tail_region_loss"]
            >= candidate.metrics["global_position_point_mean_loss"]
            and candidate.metrics["global_position_median_band_loss"] >= 0.0
        ),
        "single_h1p5_h2_h3_policy_is_active": bool(
            tuple(updater.settings.global_horizon_weights) == (1.5, 2.0, 3.0)
            and candidate.metrics["global_horizon_weight_h1"] == 1.5
            and candidate.metrics["global_horizon_weight_h2"] == 2.0
            and candidate.metrics["global_horizon_weight_h3"] == 3.0
        ),
        "warp_gradient_drives_committed_adam": bool(
            candidate.metrics["optimizer_gradient_source"]
            == "warp_finite_difference"
        ),
        "grasp_coupling_is_fixed_to_one": bool(
            candidate.metrics["global_coupling_fixed"] == 1
            and candidate.global_coupling_gain == 1.0
            and candidate.global_log_coefficients is not None
            and candidate.global_log_coefficients[2] == 0.0
        ),
        "fixed_surface_tracks_drive_loss": bool(
            candidate.metrics["global_track_loss_count"] == 9
            and updater.track_particle_ids is not None
            and updater.track_region_basis is not None
        ),
    }
    report = {
        "stage": "differentiable_global_paper_system_id_gate",
        "candidate_metrics": candidate.metrics,
        "autograd_gradient": gradient.tolist(),
        "gates": gates,
        "passed": all(gates.values()),
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
