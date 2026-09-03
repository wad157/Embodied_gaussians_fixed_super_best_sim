#!/usr/bin/env python3
"""CPU gate for the causal low-dimensional differentiable PBD updater."""

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
    inverse_mass = torch.tensor((0.0, 2.0e5, 2.0e5, 2.0e5, 2.0e5))
    distance = torch.full((5,), 0.20)
    shape = torch.full((5,), 0.004)
    settings = OnlineTissueStiffnessSettings(
        update_mode="differentiable_low_dim",
        log_learning_rate=0.03,
        maximum_log_step=0.02,
        signal_ema_decay=0.90,
        spatial_smoothing_iterations=3,
        spatial_smoothing_blend=0.35,
        strain_signal_weight=0.20,
        autograd_unroll_steps=4,
        autograd_region_count=3,
        autograd_prior_weight=0.0,
        autograd_frame_dt=1.0 / 30.0,
        autograd_material_iterations=2,
        autograd_pbd_dt=1.0 / 120.0,
        autograd_pbd_relaxation=0.25,
    )
    updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=fixed,
        edges=edges,
        distance_stiffness=distance,
        shape_stiffness=shape,
        inverse_mass=inverse_mass,
        tet_indices=tets,
        settings=settings,
    )
    rollout_start = rest.clone()
    rollout_start[:, 0] *= 1.10
    residual = torch.zeros_like(rest)
    residual[1:, 2] = 0.00004
    quality = torch.ones(5, dtype=torch.bool)
    quality[3] = False
    supervision = torch.ones(5, dtype=torch.bool)
    control_excluded = torch.tensor((False, False, False, False, True))
    candidate = None
    for frame_index in range(4):
        corrected = rollout_start.clone()
        corrected[1:, 2] += 0.00001 * frame_index
        prediction = corrected - residual
        candidate = updater.propose(
            physical_prediction=prediction,
            accepted_residual=residual,
            quality_valid_mask=quality,
            supervision_valid_mask=supervision,
            control_exclusion_mask=control_excluded,
            control_frozen_mask=torch.zeros(5, dtype=torch.bool),
            rollout_start_positions=prediction,
            physical_velocities=torch.zeros_like(rest),
            frame_index=frame_index,
        )
        if frame_index < 3:
            assert candidate.metrics["autograd_history_status"] == "warming_up"
            updater.reject("causal_history_warmup", candidate)
    assert candidate is not None
    before_distance = distance.clone()
    before_shape = shape.clone()
    gates = {
        "candidate_preserves_verified_until_commit": bool(
            torch.equal(distance, before_distance)
            and torch.equal(shape, before_shape)
        ),
        "causal_full_pbd_autograd_is_recorded": bool(
            candidate.metrics["update_mode"] == "differentiable_low_dim"
            and candidate.metrics["autograd_unroll_steps"] == 4
            and candidate.metrics["autograd_temporal_steps"] == 3
            and candidate.metrics["autograd_material_iterations"] == 2
            and candidate.metrics["autograd_history_status"] == "ready"
        ),
        "adam_produces_finite_nonzero_bounded_step": bool(
            torch.isfinite(candidate.log_step).all().item()
            and float(candidate.log_step.abs().max()) > 0.0
            and float(candidate.log_step.abs().max()) <= 0.0200001
        ),
        "distance_field_is_low_dimensional_and_smooth": bool(
            candidate.metrics["autograd_region_count"] == 3
            and float(
                torch.max(
                    torch.abs(
                        candidate.distance_stiffness[1:]
                        - candidate.distance_stiffness[:-1]
                    )
                )
            )
            < 0.01
        ),
        "edge_strain_is_in_differentiable_objective": bool(
            candidate.metrics["autograd_edge_strain_weight"] == 0.20
            and candidate.metrics["autograd_edge_strain_loss"] >= 0.0
        ),
        "excluded_particle_stiffness_is_exactly_unchanged": bool(
            candidate.log_step[4] == 0.0
            and candidate.distance_stiffness[4] == before_distance[4]
            and candidate.log_step[3] == 0.0
            and candidate.distance_stiffness[3] == before_distance[3]
        ),
        "shape_is_frozen_in_first_identifiability_trial": bool(
            torch.equal(candidate.shape_stiffness, before_shape)
            and candidate.metrics["autograd_shape_frozen"] == 1
        ),
    }
    updater.commit(candidate)
    gates["commit_persists_material_and_adam_state"] = bool(
        not torch.equal(distance, before_distance)
        and updater.adam_step == 1
        and torch.count_nonzero(updater.adam_first_moment) > 0
    )
    report = {
        "stage": "differentiable_low_dimensional_stiffness_gate",
        "configuration": {
            "pbd_unroll_steps": settings.autograd_unroll_steps,
            "region_count": settings.autograd_region_count,
            "adam_lr": settings.log_learning_rate,
            "log_step_cap": settings.maximum_log_step,
            "adam_beta1_ema": settings.signal_ema_decay,
            "spatial_smoothing_iterations": settings.spatial_smoothing_iterations,
            "edge_strain_fallback_weight": settings.strain_signal_weight,
        },
        "candidate_metrics": candidate.metrics,
        "candidate_log_step": candidate.log_step.tolist(),
        "candidate_distance": candidate.distance_stiffness.tolist(),
        "gates": gates,
        "passed": all(gates.values()),
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Differentiable low-dimensional stiffness gate failed")


if __name__ == "__main__":
    main()
