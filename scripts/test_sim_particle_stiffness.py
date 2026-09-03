#!/usr/bin/env python3
"""CPU gate proving particle residual mode does not collapse to one scalar."""

from __future__ import annotations

import json

import torch

from embodied_gaussians.physics_simulator.online_tissue_stiffness import (
    OnlineTissueStiffnessSettings,
    ResidualDrivenPaperStiffnessUpdater,
)


def main() -> None:
    rest = torch.tensor(
        ((0.0, 0.0, 0.0), (0.001, 0.0, 0.0), (0.002, 0.0, 0.0),
         (0.003, 0.0, 0.0), (0.004, 0.0, 0.0)),
        dtype=torch.float32,
    )
    distance = torch.full((5,), 0.20)
    shape = torch.full((5,), 0.004)
    updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=torch.tensor((True, False, False, False, False)),
        edges=torch.tensor(((0, 1), (1, 2), (2, 3), (3, 4))),
        distance_stiffness=distance,
        shape_stiffness=shape,
        settings=OnlineTissueStiffnessSettings(
            update_mode="particle_residual",
            log_learning_rate=0.10,
            maximum_log_step=0.10,
            signal_ema_decay=0.0,
            spatial_smoothing_iterations=0,
            strain_signal_weight=0.0,
            hardening_bias=0.0,
        ),
    )
    prediction = rest.clone()
    prediction[1:, 2] = torch.tensor((0.0004, 0.0005, 0.0006, 0.0007))
    residual = torch.zeros_like(rest)
    residual[1, 2] = -0.00020
    residual[2, 2] = 0.00015
    residual[3, 2] = -0.00005
    residual[4, 2] = 0.00020
    excluded = torch.tensor((False, False, False, False, True))
    candidate = updater.propose(
        physical_prediction=prediction,
        accepted_residual=residual,
        control_exclusion_mask=excluded,
    )
    free = candidate.distance_stiffness[1:4]
    gates = {
        "explicit_particle_mode": updater.settings.update_mode
        == "particle_residual",
        "free_particles_are_not_one_scalar": int(torch.unique(free).numel()) > 1,
        "fixed_particle_unchanged": bool(candidate.distance_stiffness[0] == 0.20),
        "control_particle_unchanged": bool(candidate.distance_stiffness[4] == 0.20),
        "particle_array_shape_preserved": candidate.distance_stiffness.shape == (5,),
    }
    report = {
        "candidate_distance_stiffness": candidate.distance_stiffness.tolist(),
        "candidate_log_step": candidate.log_step.tolist(),
        "gates": gates,
        "passed": all(gates.values()),
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
