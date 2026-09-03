#!/usr/bin/env python3
"""CPU gate for global plus per-particle graph-regularized stiffness."""

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
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3),
            (2, 3), (1, 4), (2, 4), (3, 4),
        ),
        dtype=torch.long,
    )
    updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=torch.tensor((True, False, False, False, False)),
        edges=edges,
        distance_stiffness=torch.full((5,), 0.20),
        shape_stiffness=torch.full((5,), 0.004),
        inverse_mass=torch.tensor((0.0, 2.0e5, 1.8e5, 2.2e5, 2.0e5)),
        tet_indices=torch.tensor(((0, 1, 2, 3), (1, 2, 3, 4))),
        track_particle_ids=torch.arange(5, dtype=torch.long)[:, None],
        track_particle_weights=torch.ones((5, 1), dtype=torch.float32),
        track_valid_mask=torch.ones(5, dtype=torch.bool),
        settings=OnlineTissueStiffnessSettings(
            update_mode="differentiable_particle_graph_lm",
            log_learning_rate=0.03,
            maximum_log_step=0.02,
            signal_ema_decay=0.90,
            strain_signal_weight=0.20,
            autograd_unroll_steps=5,
            autograd_region_count=3,
            autograd_material_iterations=1,
            autograd_integration_substeps=1,
            local_update_interval_frames=5,
        ),
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
        candidate = updater.propose(
            physical_prediction=prediction,
            accepted_residual=residual,
            quality_valid_mask=torch.ones(5, dtype=torch.bool),
            supervision_valid_mask=torch.ones(5, dtype=torch.bool),
            control_exclusion_mask=control,
            control_frozen_mask=control,
            control_coupling_base_positions=start,
            control_coupling_displacements=torch.zeros_like(rest),
            track_target_positions=prediction + residual,
            track_valid_mask=torch.ones(5, dtype=torch.bool),
            rollout_start_positions=start,
            physical_velocities=velocity,
            frame_index=frame,
        )
        if frame < 4:
            updater.reject("causal_history_warmup", candidate)

    assert candidate is not None
    gradient = candidate.autograd_parameter_gradient
    assert gradient is not None and torch.isfinite(gradient).all()
    before_distance = updater.distance_stiffness.clone()
    before_shape = updater.shape_stiffness.clone()
    predicted = torch.tensor((1.0, -0.4))
    allowed, cosine = updater.apply_particle_graph_directional_gate(
        candidate, predicted, predicted.clone()
    )
    assert allowed and abs(cosine - 1.0) < 1.0e-6
    updater.replace_candidate_step_with_particle_graph_lm(candidate)
    field = updater.local_zero_mean_log_field(
        candidate.low_dimensional_log_coefficients
    )
    active = ~updater.local_parameter_exclusion_mask
    weighted_mean = torch.sum(field * updater.inverse_mass) / (
        updater.inverse_mass[active].sum()
    )
    updater.commit(candidate)
    gates = {
        "one_global_plus_one_value_per_particle": bool(
            updater.low_dimensional_log_coefficients.shape == (6,)
        ),
        "excluded_particles_keep_initial_material": bool(
            updater.distance_stiffness[0] == before_distance[0]
            and updater.distance_stiffness[4] == before_distance[4]
            and updater.shape_stiffness[0] == before_shape[0]
            and updater.shape_stiffness[4] == before_shape[4]
        ),
        "weighted_local_field_is_zero_mean": bool(abs(weighted_mean) < 1.0e-6),
        "distance_and_weak_shape_field_update": bool(
            not torch.equal(updater.distance_stiffness[1:4], before_distance[1:4])
            and not torch.equal(updater.shape_stiffness[1:4], before_shape[1:4])
        ),
        "bounded_graph_lm_step": bool(
            candidate.metrics["graph_lm_step_maximum"]
            <= updater.settings.maximum_log_step + 1.0e-7
        ),
        "directional_warp_gate_recorded": bool(
            candidate.metrics["gradient_check_kind"]
            == "deterministic_warp_directional_fd"
        ),
        "no_initial_estimation": bool(torch.all(before_distance == 0.20)),
    }
    report = {
        "stage": "differentiable_particle_graph_lm_gate",
        "distance": updater.distance_stiffness.tolist(),
        "shape": updater.shape_stiffness.tolist(),
        "metrics": candidate.metrics,
        "gates": gates,
        "passed": all(gates.values()),
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
