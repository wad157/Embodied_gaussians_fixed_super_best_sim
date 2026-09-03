#!/usr/bin/env python3
"""CPU regression gate for causal observable global paper-PBD MHE."""

from __future__ import annotations

import json

import torch

from embodied_gaussians.physics_simulator.online_tissue_stiffness import (
    OnlineTissueStiffnessSettings,
    ResidualDrivenPaperStiffnessUpdater,
)


def build_updater() -> ResidualDrivenPaperStiffnessUpdater:
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
    return ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=torch.tensor((True, False, False, False, False)),
        edges=edges,
        distance_stiffness=torch.full((5,), 0.20),
        shape_stiffness=torch.full((5,), 0.004),
        inverse_mass=torch.tensor((0.0, 2.0e5, 2.0e5, 2.0e5, 2.0e5)),
        tet_indices=torch.tensor(((0, 1, 2, 3), (1, 2, 3, 4))),
        track_particle_ids=torch.arange(5, dtype=torch.long)[:, None],
        track_particle_weights=torch.ones((5, 1), dtype=torch.float32),
        track_valid_mask=torch.ones(5, dtype=torch.bool),
        initial_velocity_damping_per_second=10.0,
        initial_coupling_gain=1.0,
        settings=OnlineTissueStiffnessSettings(
            update_mode="differentiable_global_mhe",
            log_learning_rate=0.03,
            maximum_log_step=0.02,
            strain_signal_weight=0.20,
            autograd_region_count=3,
            observable_window_size=4,
            observable_minimum_transitions=3,
            observable_update_interval_frames=1,
            observable_minimum_singular_ratio=0.02,
            observable_lm_damping=0.05,
            observable_parameter_prior_weight=0.01,
        ),
    )


def main() -> None:
    updater = build_updater()
    center = updater.global_log_coefficients.clone()
    residual = torch.tensor((0.8, -0.4, 0.3, -0.2, 0.6, -0.5))
    observable_jacobian = torch.tensor(
        (
            (1.0, 0.0),
            (0.7, 0.2),
            (0.2, 1.0),
            (0.0, 0.8),
            (-0.5, 0.3),
            (0.3, -0.6),
        )
    )
    step, allowed, reason, diagnostics = updater.observable_lm_step(
        center_coefficients=center,
        residual=residual,
        jacobian=observable_jacobian,
        wide_jacobian=observable_jacobian * 1.001,
        short_residual_count=3,
    )

    collinear = torch.stack(
        (observable_jacobian[:, 0], observable_jacobian[:, 0] * 1.001),
        dim=1,
    )
    _bad_step, collinear_allowed, collinear_reason, collinear_metrics = (
        updater.observable_lm_step(
            center_coefficients=center,
            residual=residual,
            jacobian=collinear,
            wide_jacobian=collinear,
            short_residual_count=3,
        )
    )
    _scale_step, scale_allowed, scale_reason, scale_metrics = (
        updater.observable_lm_step(
            center_coefficients=center,
            residual=residual,
            jacobian=observable_jacobian,
            wide_jacobian=-observable_jacobian,
            short_residual_count=3,
        )
    )
    projected_step, projected_allowed, _projected_reason, projected_metrics = (
        updater.observable_lm_step(
            center_coefficients=center,
            residual=torch.tensor((1.0, -3.0, 1.0)),
            jacobian=torch.tensor(
                ((1.0, 0.0), (1.0, 0.0), (0.0, 1.0))
            ),
            wide_jacobian=torch.tensor(
                ((1.001, 0.0), (1.001, 0.0), (0.0, 1.001))
            ),
            short_residual_count=1,
        )
    )

    # MHE history must retain non-consecutive, already observed transitions;
    # periodic 7:1 holdouts may create gaps but do not authorize future data.
    candidate = None
    rest = updater.rest_positions
    for frame in (0, 2, 4, 6, 8):
        prediction = rest.clone()
        prediction[1:, 2] += 0.00020
        accepted = torch.zeros_like(rest)
        accepted[1:, 1] += 0.00005
        candidate = updater.propose(
            physical_prediction=prediction,
            accepted_residual=accepted,
            rollout_start_positions=rest,
            physical_velocities=torch.zeros_like(rest),
            track_target_positions=prediction + accepted,
            track_valid_mask=torch.ones(5, dtype=torch.bool),
            frame_index=frame,
        )
        if frame != 8:
            updater.reject("history_test", candidate)
    assert candidate is not None
    history_frames = [
        observation.frame_index for observation in updater.causal_material_history
    ]
    before_shape = updater.shape_stiffness.clone()
    updater.replace_candidate_step_with_observable_lm(
        candidate, step, diagnostics
    )
    candidate_step_maximum = updater.candidate_parameter_step_maximum(candidate)
    updater.commit(candidate)

    gates = {
        "observable_columns_pass": bool(
            allowed
            and reason == "observable_lm_ready"
            and diagnostics["observable_singular_value_ratio"] >= 0.02
        ),
        "step_is_finite_bounded_and_nonzero": bool(
            torch.isfinite(step).all().item()
            and 0.0 < float(step[:2].abs().max()) <= 0.0200001
        ),
        "collinear_material_damping_is_rejected": bool(
            not collinear_allowed
            and collinear_reason == "observable_rank_or_sensitivity_below_gate"
            and collinear_metrics["observable_singular_value_ratio"] < 0.02
        ),
        "fd_scale_direction_mismatch_is_rejected": bool(
            not scale_allowed
            and scale_reason == "observable_fd_scale_inconsistent"
            and scale_metrics["observable_fd_scale_cosine"] < 0.0
        ),
        "h1_increase_is_projected_to_nonincrease": bool(
            projected_allowed
            and projected_metrics["observable_short_descent_projected"] == 1
            and projected_metrics["observable_short_directional_before"] > 0.0
            and projected_metrics["observable_short_directional_after"]
            <= projected_metrics["observable_short_directional_uncertainty"]
            + 1.0e-7
            and projected_metrics["observable_long_directional_after"] <= 0.0
            and torch.isfinite(projected_step).all().item()
        ),
        "holdout_gaps_survive_in_bounded_window": bool(
            history_frames == [2, 4, 6, 8]
        ),
        "only_distance_and_damping_commit": bool(
            candidate_step_maximum > 0.0
            and torch.equal(updater.shape_stiffness, before_shape)
            and updater.global_log_coefficients[2] == 0.0
            and updater.global_coupling_gain == 1.0
            and candidate.metrics["optimizer_gradient_source"]
            == "warp_multishooting_observable_lm"
        ),
    }
    report = {
        "stage": "observable_global_paper_mhe_cpu_gate",
        "observable_diagnostics": diagnostics,
        "history_frames": history_frames,
        "committed_coefficients": updater.global_log_coefficients.tolist(),
        "gates": gates,
        "passed": all(gates.values()),
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
