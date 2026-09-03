#!/usr/bin/env python3
"""Deterministic gate for signed, bounded online paper stiffness updates."""

from __future__ import annotations

import json

import torch

from embodied_gaussians.physics_simulator.online_tissue_stiffness import (
    OnlineTissueStiffnessSettings,
    ResidualDrivenPaperStiffnessUpdater,
)


def main() -> None:
    rest = torch.tensor(
        ((0.0, 0.0, 0.0), (0.001, 0.0, 0.0), (0.002, 0.0, 0.0)),
        dtype=torch.float32,
    )
    fixed = torch.tensor((True, False, False))
    edges = torch.tensor(((0, 1), (1, 2)), dtype=torch.long)
    distance = torch.full((3,), 0.4)
    shape = torch.full((3,), 0.008)
    updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=fixed,
        edges=edges,
        distance_stiffness=distance,
        shape_stiffness=shape,
        settings=OnlineTissueStiffnessSettings(
            log_learning_rate=0.10,
            signal_ema_decay=0.0,
            spatial_smoothing_iterations=0,
            hardening_bias=0.0,
        ),
    )
    prediction = rest.clone()
    prediction[1:, 2] = -0.001
    correction_toward_rest = torch.zeros_like(rest)
    correction_toward_rest[1:, 2] = 0.00030
    hard_metrics = updater.update(
        physical_prediction=prediction,
        accepted_residual=correction_toward_rest,
    )
    hardened_distance = distance.clone()
    hardened_shape = shape.clone()
    correction_farther_from_rest = -correction_toward_rest
    soft_metrics = updater.update(
        physical_prediction=prediction,
        accepted_residual=correction_farther_from_rest,
    )
    updater.reset()
    quality_metrics = updater.update(
        physical_prediction=prediction,
        accepted_residual=correction_toward_rest,
        quality_valid_mask=torch.tensor((False, False, True)),
    )
    quality_gated_distance = distance.clone()
    quality_gated_shape = shape.clone()
    updater.reset()
    candidate_distance = torch.full((3,), 0.4)
    candidate_shape = torch.full((3,), 0.008)
    candidate_updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=fixed,
        edges=edges,
        distance_stiffness=candidate_distance,
        shape_stiffness=candidate_shape,
        settings=OnlineTissueStiffnessSettings(
            log_learning_rate=0.10,
            signal_ema_decay=0.0,
            spatial_smoothing_iterations=1,
            spatial_smoothing_blend=0.5,
            hardening_bias=0.0,
        ),
    )
    proposed = candidate_updater.propose(
        physical_prediction=prediction,
        accepted_residual=correction_toward_rest,
        supervision_valid_mask=torch.tensor((False, True, True)),
        control_exclusion_mask=torch.tensor((False, True, False)),
    )
    proposal_does_not_mutate_verified = bool(
        torch.equal(candidate_distance, torch.full((3,), 0.4))
        and torch.equal(candidate_shape, torch.full((3,), 0.008))
    )
    hard_exclusion_blocks_smoothing_and_ema = bool(
        proposed.log_step[1] == 0.0
        and proposed.signal_ema[1] == 0.0
        and candidate_updater.signal_ema[1] == 0.0
        and proposed.log_step[2] > 0.0
    )
    proposal_diagnostics_are_complete = bool(
        candidate_updater.last_proposal_candidate_count == 1
        and candidate_updater.last_proposal_diagnostics is not None
        and {
            "residual_norm_m",
            "deformation_norm_m",
            "vector_signal",
            "strain_signal",
            "strain_confidence",
            "blended_signal_before_smoothing",
            "smoothed_signal",
            "candidate_ema",
            "log_step",
            "quality_valid_mask",
            "supervision_valid_mask",
            "control_exclusion_mask",
            "eligible_mask",
            "vector_active_mask",
            "strain_active_mask",
            "material_active_mask",
        }.issubset(candidate_updater.last_proposal_diagnostics)
        and torch.equal(
            candidate_updater.last_proposal_diagnostics["log_step"],
            proposed.log_step,
        )
    )
    rejected = candidate_updater.reject("synthetic_rejection", proposed)
    rejection_preserves_verified = bool(
        torch.equal(candidate_distance, torch.full((3,), 0.4))
        and torch.equal(candidate_shape, torch.full((3,), 0.008))
        and rejected["status"] == "rejected"
        and candidate_updater.pending_candidate is None
    )
    candidate_updater.signal_ema.fill_(1.0)
    candidate_updater.invalidate_signal_history(
        torch.tensor((False, True, False))
    )
    selective_ema_invalidation_works = bool(
        candidate_updater.signal_ema.tolist() == [1.0, 0.0, 1.0]
    )
    candidate_updater.invalidate_signal_history()
    global_ema_invalidation_works = bool(
        torch.all(candidate_updater.signal_ema == 0.0)
    )

    # Production-response probe: use a spatially uniform signal so the real
    # one-ring smoother preserves its magnitude.  The current GUI baseline is
    # intentionally above the new online floor and must now move both ways.
    production_fixed = torch.zeros(3, dtype=torch.bool)
    production_prediction = rest.clone()
    production_prediction[:, 2] = -0.001
    production_toward_rest = torch.zeros_like(rest)
    production_toward_rest[:, 2] = 0.00030
    production_farther_from_rest = -production_toward_rest
    production_distance = torch.full((3,), 0.20)
    production_shape = torch.full((3,), 0.004)
    production_updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=production_fixed,
        edges=edges,
        distance_stiffness=production_distance,
        shape_stiffness=production_shape,
    )
    production_softening = production_updater.propose(
        physical_prediction=production_prediction,
        accepted_residual=production_farther_from_rest,
    )
    first_soft_distance = float(
        production_softening.distance_stiffness[0].item()
    )
    first_soft_shape = float(
        production_softening.shape_stiffness[0].item()
    )
    production_updater.reject("first_softening_probe", production_softening)
    production_hardening = production_updater.propose(
        physical_prediction=production_prediction,
        accepted_residual=production_toward_rest,
    )
    first_hard_distance = float(
        production_hardening.distance_stiffness[0].item()
    )
    first_hard_shape = float(
        production_hardening.shape_stiffness[0].item()
    )
    production_updater.reject("first_hardening_probe", production_hardening)

    # Reproduce the immediately preceding production policy in the same probe
    # so the report contains measured before/after response, not hand-derived
    # percentages only.
    previous_distance = torch.full((3,), 0.20)
    previous_shape = torch.full((3,), 0.004)
    previous_updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=production_fixed,
        edges=edges,
        distance_stiffness=previous_distance,
        shape_stiffness=previous_shape,
        settings=OnlineTissueStiffnessSettings(
            signal_ema_decay=0.80,
            distance_minimum=0.20,
            shape_minimum=0.004,
        ),
    )
    previous_softening = previous_updater.propose(
        physical_prediction=production_prediction,
        accepted_residual=production_farther_from_rest,
    )
    previous_first_soft_distance = float(
        previous_softening.distance_stiffness[0].item()
    )
    previous_first_soft_shape = float(
        previous_softening.shape_stiffness[0].item()
    )
    previous_updater.reject("previous_softening_probe", previous_softening)
    previous_hardening = previous_updater.propose(
        physical_prediction=production_prediction,
        accepted_residual=production_toward_rest,
    )
    previous_first_hard_distance = float(
        previous_hardening.distance_stiffness[0].item()
    )
    previous_first_hard_shape = float(
        previous_hardening.shape_stiffness[0].item()
    )
    previous_updater.reject("previous_hardening_probe", previous_hardening)

    # Opposite evidence at the two ends of the real one-ring graph must be
    # able to create a soft and a hard region simultaneously.  This keeps the
    # production spatial smoother enabled and exercises the regional use case,
    # rather than only probing uniform global updates.
    regional_distance = torch.full((3,), 0.20)
    regional_shape = torch.full((3,), 0.004)
    regional_updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=production_fixed,
        edges=edges,
        distance_stiffness=regional_distance,
        shape_stiffness=regional_shape,
    )
    regional_residual = torch.zeros_like(rest)
    regional_residual[0] = production_farther_from_rest[0]
    regional_residual[2] = production_toward_rest[2]
    regional_soft_floor_update = 0
    regional_hard_ceiling_update = 0
    for update_index in range(1, 41):
        regional_updater.update(
            physical_prediction=production_prediction,
            accepted_residual=regional_residual,
        )
        if regional_soft_floor_update == 0 and torch.isclose(
            regional_distance[0], torch.tensor(0.10)
        ):
            regional_soft_floor_update = update_index
        if regional_hard_ceiling_update == 0 and torch.isclose(
            regional_distance[2], torch.tensor(2.00)
        ):
            regional_hard_ceiling_update = update_index

    reconfigured_distance = torch.tensor((0.08, 0.20, 2.20))
    reconfigured_shape = torch.tensor((0.002, 0.004, 0.025))
    reconfigured_updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=production_fixed,
        edges=edges,
        distance_stiffness=reconfigured_distance,
        shape_stiffness=reconfigured_shape,
    )
    reconfigured_updater.signal_ema.fill_(0.5)
    runtime_settings = OnlineTissueStiffnessSettings(
        log_learning_rate=0.12,
        signal_ema_decay=0.60,
        distance_minimum=0.10,
        distance_maximum=2.00,
        shape_minimum=0.003,
        shape_maximum=0.020,
    )
    reconfigured_updater.reconfigure(runtime_settings)

    repeated_distance = torch.full((3,), 0.20)
    repeated_shape = torch.full((3,), 0.004)
    repeated_updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=production_fixed,
        edges=edges,
        distance_stiffness=repeated_distance,
        shape_stiffness=repeated_shape,
    )
    repeated_softening: list[dict[str, float]] = []
    distance_updates_to_floor = 0
    shape_updates_to_floor = 0
    for update_index in range(1, 21):
        repeated_updater.update(
            physical_prediction=production_prediction,
            accepted_residual=production_farther_from_rest,
        )
        repeated_softening.append(
            {
                "update": float(update_index),
                "distance": float(repeated_distance[0].item()),
                "shape": float(repeated_shape[0].item()),
            }
        )
        if distance_updates_to_floor == 0 and torch.allclose(
            repeated_distance, torch.full((3,), 0.10)
        ):
            distance_updates_to_floor = update_index
        if shape_updates_to_floor == 0 and torch.allclose(
            repeated_shape, torch.full((3,), 0.003)
        ):
            shape_updates_to_floor = update_index

    repeated_hard_distance = torch.full((3,), 0.20)
    repeated_hard_shape = torch.full((3,), 0.004)
    repeated_hard_updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=production_fixed,
        edges=edges,
        distance_stiffness=repeated_hard_distance,
        shape_stiffness=repeated_hard_shape,
    )
    repeated_hardening: list[dict[str, float]] = []
    distance_updates_to_ceiling = 0
    shape_updates_to_ceiling = 0
    for update_index in range(1, 21):
        repeated_hard_updater.update(
            physical_prediction=production_prediction,
            accepted_residual=production_toward_rest,
        )
        repeated_hardening.append(
            {
                "update": float(update_index),
                "distance": float(repeated_hard_distance[0].item()),
                "shape": float(repeated_hard_shape[0].item()),
            }
        )
        if distance_updates_to_ceiling == 0 and torch.allclose(
            repeated_hard_distance, torch.full((3,), 2.00)
        ):
            distance_updates_to_ceiling = update_index
        if shape_updates_to_ceiling == 0 and torch.allclose(
            repeated_hard_shape, torch.full((3,), 0.020)
        ):
            shape_updates_to_ceiling = update_index

    floor_distance = torch.full((3,), 0.10)
    floor_shape = torch.full((3,), 0.003)
    floor_updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=fixed,
        edges=edges,
        distance_stiffness=floor_distance,
        shape_stiffness=floor_shape,
        settings=OnlineTissueStiffnessSettings(
            log_learning_rate=0.18,
            signal_ema_decay=0.0,
            spatial_smoothing_iterations=0,
            hardening_bias=0.15,
        ),
    )
    floor_softening = floor_updater.propose(
        physical_prediction=prediction,
        accepted_residual=correction_farther_from_rest,
    )
    lower_bound_saturates = bool(
        torch.equal(
            floor_softening.distance_stiffness,
            torch.full((3,), 0.10),
        )
        and torch.equal(
            floor_softening.shape_stiffness,
            torch.full((3,), 0.003),
        )
        and torch.all(floor_softening.log_step[1:] < 0.0).item()
    )
    floor_updater.reject("lower_bound_probe", floor_softening)
    floor_hardening = floor_updater.propose(
        physical_prediction=prediction,
        accepted_residual=correction_toward_rest,
    )
    lower_bound_can_harden = bool(
        torch.all(floor_hardening.distance_stiffness[1:] > 0.10).item()
        and torch.all(floor_hardening.shape_stiffness[1:] > 0.003).item()
    )
    floor_updater.reject("hardening_probe", floor_hardening)
    gates = {
        "toward_rest_hardens_dynamic_nodes": bool(
            torch.all(hardened_distance[1:] > 0.4).item()
            and torch.all(hardened_shape[1:] > 0.008).item()
        ),
        "away_from_rest_softens_after_hardening": bool(
            torch.all(distance[1:] < hardened_distance[1:]).item()
            and torch.all(shape[1:] < hardened_shape[1:]).item()
        ),
        "fixed_node_never_changes": bool(
            hardened_distance[0] == 0.4
            and hardened_shape[0] == 0.008
        ),
        "reset_restores_baseline": bool(
            torch.equal(distance, torch.full((3,), 0.4))
            and torch.equal(shape, torch.full((3,), 0.008))
        ),
        "local_bad_tet_masks_only_its_incident_particle": bool(
            quality_gated_distance[1] == 0.4
            and quality_gated_shape[1] == 0.008
            and quality_gated_distance[2] > 0.4
            and quality_gated_shape[2] > 0.008
            and quality_metrics["quality_masked_particles"] == 1
            and quality_metrics["quality_valid_particles"] == 1
        ),
        "large_update_is_bounded": bool(
            hard_metrics["maximum_log_step"] <= 0.12 + 1.0e-8
        ),
        "candidate_does_not_mutate_verified": proposal_does_not_mutate_verified,
        "proposal_diagnostics_are_complete": proposal_diagnostics_are_complete,
        "u_t_exclusion_blocks_smoothing_and_ema": hard_exclusion_blocks_smoothing_and_ema,
        "rejection_preserves_verified": rejection_preserves_verified,
        "selective_ema_invalidation_works": (
            selective_ema_invalidation_works
        ),
        "global_ema_invalidation_works": global_ema_invalidation_works,
        "current_soft_baseline_can_soften_below_0p20_0p004": bool(
            first_soft_distance < 0.20
            and first_soft_distance >= 0.10
            and first_soft_shape < 0.004
            and first_soft_shape >= 0.003
        ),
        "current_soft_baseline_can_harden_more_responsively": bool(
            first_hard_distance > previous_first_hard_distance
            and first_hard_shape > previous_first_hard_shape
        ),
        "previous_policy_reference_is_reproduced": bool(
            abs(previous_first_soft_distance - 0.20) < 1.0e-7
            and abs(previous_first_soft_shape - 0.004) < 1.0e-8
            and previous_first_hard_distance > 0.20
            and previous_first_hard_shape > 0.004
        ),
        "repeated_softening_reaches_new_safe_floors": bool(
            distance_updates_to_floor > 0
            and shape_updates_to_floor > 0
            and torch.allclose(
                repeated_distance, torch.full((3,), 0.10)
            )
            and torch.allclose(repeated_shape, torch.full((3,), 0.003))
        ),
        "repeated_hardening_reaches_new_safe_ceilings": bool(
            distance_updates_to_ceiling > 0
            and shape_updates_to_ceiling > 0
            and torch.allclose(
                repeated_hard_distance, torch.full((3,), 2.00)
            )
            and torch.allclose(
                repeated_hard_shape, torch.full((3,), 0.020)
            )
        ),
        "distance_range_supports_20x_regional_contrast": bool(
            regional_soft_floor_update > 0
            and regional_hard_ceiling_update > 0
            and torch.isclose(regional_distance[0], torch.tensor(0.10))
            and torch.isclose(regional_distance[2], torch.tensor(2.00))
            and abs(
                float(regional_distance[2] / regional_distance[0]) - 20.0
            )
            < 1.0e-5
        ),
        "runtime_reconfigure_clips_bounds_and_clears_ema": bool(
            reconfigured_updater.settings == runtime_settings
            and torch.allclose(
                reconfigured_distance, torch.tensor((0.10, 0.20, 2.00))
            )
            and torch.allclose(
                reconfigured_shape, torch.tensor((0.003, 0.004, 0.020))
            )
            and torch.count_nonzero(reconfigured_updater.signal_ema) == 0
        ),
        "new_online_floor_saturates": lower_bound_saturates,
        "new_online_floor_can_still_harden": lower_bound_can_harden,
    }
    report = {
        "stage": "online_visual_residual_paper_stiffness_gate",
        "hardening_metrics": hard_metrics,
        "softening_metrics": soft_metrics,
        "hardened_distance": hardened_distance.tolist(),
        "hardened_shape": hardened_shape.tolist(),
        "quality_gated_distance": quality_gated_distance.tolist(),
        "quality_gated_shape": quality_gated_shape.tolist(),
        "regional_contrast": {
            "distance": regional_distance.tolist(),
            "shape": regional_shape.tolist(),
            "soft_floor_update": regional_soft_floor_update,
            "hard_ceiling_update": regional_hard_ceiling_update,
            "distance_maximum_to_minimum_ratio": float(
                regional_distance.max() / regional_distance.min()
            ),
        },
        "soft_baseline": {
            "initial_distance": 0.20,
            "initial_shape": 0.004,
            "online_distance_floor": 0.10,
            "online_distance_ceiling": 2.00,
            "online_shape_floor": 0.003,
            "first_soft_distance": first_soft_distance,
            "first_soft_shape": first_soft_shape,
            "first_hard_distance": first_hard_distance,
            "first_hard_shape": first_hard_shape,
            "previous_policy": {
                "ema_new_weight": 0.20,
                "first_soft_distance": previous_first_soft_distance,
                "first_soft_shape": previous_first_soft_shape,
                "first_hard_distance": previous_first_hard_distance,
                "first_hard_shape": previous_first_hard_shape,
            },
            "distance_updates_to_floor": distance_updates_to_floor,
            "shape_updates_to_floor": shape_updates_to_floor,
            "repeated_softening": repeated_softening,
            "distance_updates_to_ceiling": distance_updates_to_ceiling,
            "shape_updates_to_ceiling": shape_updates_to_ceiling,
            "repeated_hardening": repeated_hardening,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Online stiffness gate failed")


if __name__ == "__main__":
    main()
