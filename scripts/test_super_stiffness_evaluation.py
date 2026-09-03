#!/usr/bin/env python3
"""Headless gate for phase grouping and persistent stiffness metrics."""

from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import warp as wp

from example_embodied_super_offline import (
    CommittedStiffnessEvaluation,
    PendingStiffnessValidation,
    STIFFNESS_CANDIDATE_LOG_STEP_SCALES,
    STIFFNESS_COMMIT_VALIDATION_HORIZON_FRAMES,
    STIFFNESS_MAXIMUM_ADOPTED_STATE_RMS_M,
    STIFFNESS_MAXIMUM_PENETRATION_M,
    STIFFNESS_PREDICTION_ABSOLUTE_MARGIN,
    STIFFNESS_PREDICTION_RELATIVE_MARGIN,
    STIFFNESS_TRANSITION_COOLDOWN_UPDATES,
    StiffnessToolCommand,
    SuperPlaybackControls,
)
from embodied_gaussians.physics_simulator.stiffness_evaluation import (
    StiffnessActionPhaseClassifier,
    StiffnessMetricsRecorder,
    parse_stiffness_evaluation_horizons,
)


def main() -> None:
    wp.init()
    parsed = parse_stiffness_evaluation_horizons("10, 1,3,5")
    invalid_horizons_rejected = 0
    for value in ("", "0,1", "1,1", "one,3"):
        try:
            parse_stiffness_evaluation_horizons(value)
        except ValueError:
            invalid_horizons_rejected += 1

    classifier = StiffnessActionPhaseClassifier(
        vertical_deadband_m=1.0e-4,
        transition_hold_samples=1,
    )
    phases = [
        classifier.classify({}, 1.0000),
        classifier.classify(
            {"contact_count": 4, "persistent_grip_active": False},
            0.9990,
        ),
        classifier.classify(
            {
                "persistent_grip_q7_motion_state": "closing",
                "persistent_grip_active": False,
            },
            0.9990,
        ),
        classifier.classify(
            {"persistent_grip_active": True},
            0.9990,
        ),
        classifier.classify(
            {"persistent_grip_active": True},
            1.0000,
        ),
        classifier.classify(
            {"persistent_grip_active": True},
            1.0010,
        ),
        classifier.classify(
            {"persistent_grip_active": True},
            1.0000,
        ),
        classifier.classify(
            {
                "persistent_grip_active": False,
                "persistent_grip_q7_motion_state": "opening",
            },
            1.0000,
        ),
    ]

    gate_controls = object.__new__(SuperPlaybackControls)
    gate_controls._last_stiffness_gate_timestep = None
    gate_controls._last_stiffness_gate_jaw_angle = None
    gate_controls._stiffness_transition_cooldown = 0
    gate_controls._last_stiffness_grip_active = False
    gate_controls._last_stiffness_contact_count = None
    gate_controls.stiffness_updater = None

    def contact(*, timestamp, jaw, grip=False, penetration=0.0027, count=4):
        return {
            "persistent_grip_q7_timestamp_s": timestamp,
            "persistent_grip_q7_angle_rad": jaw,
            "persistent_grip_active": grip,
            "maximum_penetration_m": penetration,
            "contact_count": count,
        }

    stable_gate = gate_controls._stiffness_global_gate(
        contact(timestamp=0.0, jaw=0.0)
    )
    fast_jaw_gate = gate_controls._stiffness_global_gate(
        contact(timestamp=0.01, jaw=1.0)
    )
    capture_transition_gate = gate_controls._stiffness_global_gate(
        contact(timestamp=0.02, jaw=1.1, grip=True)
    )
    post_transition_gate = gate_controls._stiffness_global_gate(
        contact(timestamp=0.03, jaw=1.1, grip=True)
    )
    excessive_penetration_gate = gate_controls._stiffness_global_gate(
        contact(timestamp=0.04, jaw=1.1, grip=True, penetration=0.0029)
    )

    pending_controls = object.__new__(SuperPlaybackControls)
    pending_candidate = SimpleNamespace(metrics={"status": "candidate"})
    pending_controls._pending_stiffness_validation = PendingStiffnessValidation(
        candidate=pending_candidate,
        rollout_state="proposal-state",
        frame_index=10,
        grip_active=False,
        history_baseline_rms_m=0.0,
        history_candidate_rms_m=0.0,
        previous_residual=torch.tensor((0.001,)),
        commands=[SimpleNamespace()] * 9,
    )
    pending_controls.stiffness_updater = object()
    pending_controls.current_frame_index = 13
    pending_controls._last_stiffness_validation_metrics = None
    pending_metrics = pending_controls._validate_pending_stiffness(
        gate_paused=False,
        gate_reason="",
        grip_active=False,
    )
    pending_shadow_call = {}
    def fake_pending_shadow(**kwargs):
        pending_shadow_call.update(kwargs)
        return {
            "visual_loss": 0.50,
            "camera_losses": (0.40, 0.60),
            "trajectory_visual_loss": 0.30,
            "trajectory_camera_losses": (0.20, 0.40),
        }

    pending_controls._run_stiffness_rollout_shadow = fake_pending_shadow
    pending_shadow_result = pending_controls._run_stiffness_prediction_shadow(
        pending_controls._pending_stiffness_validation,
        use_candidate=True,
    )
    scale_controls = object.__new__(SuperPlaybackControls)
    verified_distance = torch.full((2,), 0.20)
    verified_shape = torch.full((2,), 0.004)
    scale_controls.stiffness_updater = SimpleNamespace(
        settings=SimpleNamespace(
            maximum_log_step=0.18,
            distance_minimum=0.10,
            distance_maximum=2.0,
            shape_update_gain=1.5,
            shape_minimum=0.003,
            shape_maximum=0.020,
        ),
        distance_stiffness=verified_distance,
        shape_stiffness=verified_shape,
    )
    scale_base = SimpleNamespace(
        log_step=torch.tensor((0.10, -0.10)),
        metrics={"status": "candidate"},
        signal_ema=torch.tensor((0.5, -0.5)),
        eligible_mask=torch.ones(2, dtype=torch.bool),
        source_mask=torch.ones(2, dtype=torch.bool),
    )
    scaled_candidate = scale_controls._scaled_stiffness_candidate(
        scale_base, 2.0
    )

    class FakeAdoptionSimulator:
        def __init__(self) -> None:
            self.state_0 = SimpleNamespace(
                particle_q=wp.array(
                    np.zeros((2, 3), dtype=np.float32),
                    dtype=wp.vec3,
                    device="cpu",
                )
            )
            self.copied_states = []
            self.update_count = 0

        def copy_embodied_gaussian_rollout_state(self, state) -> None:
            self.copied_states.append(state)

        def update_gaussian_transforms(self) -> None:
            self.update_count += 1

    adoption_controls = object.__new__(SuperPlaybackControls)
    adoption_sim = FakeAdoptionSimulator()
    adoption_controls.environment = SimpleNamespace(sim=adoption_sim)
    adoption_controls._previous_visual_residual = None
    adoption_candidate = {
        "particle_positions": torch.full((2, 3), 5.0e-5),
        "rollout_state": "verified-candidate-state",
        "rollout_previous_residual": torch.tensor((0.002,)),
    }
    adoption_metrics = adoption_controls._adopt_validated_stiffness_rollout(
        adoption_candidate
    )
    rejected_adoption_metrics = (
        adoption_controls._adopt_validated_stiffness_rollout(
            {
                **adoption_candidate,
                "particle_positions": torch.full((2, 3), 1.0e-3),
                "rollout_state": "too-large-state",
            }
        )
    )

    with TemporaryDirectory(prefix="super_stiffness_eval_") as directory:
        output = Path(directory) / "run"
        recorder = StiffnessMetricsRecorder(
            output,
            metadata={
                "horizons": parsed,
                "protocols": ("material_isolation", "end_to_end"),
            },
            summary_interval_events=2,
        )
        recorder.record(
            event="visual_update",
            frame_index=10,
            timestamp_s=0.33,
            phase="press",
            image={
                "accepted": True,
                "mean_loss_before": 0.20,
                "mean_loss_after": 0.15,
            },
            physical={"minimum_volume_ratio": 0.80},
            material={"status": "committed", "distance_median": 0.20},
        )
        recorder.record(
            event="open_loop_prediction",
            frame_index=15,
            timestamp_s=0.50,
            phase="lift",
            prediction={
                "horizon_frames": 5,
                "baseline_gap": 0.10,
                "candidate_gap": 0.08,
                "gap_improvement": 0.02,
            },
            force_summary=True,
        )
        recorder.close()

        events = [
            json.loads(line)
            for line in (output / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        summary = json.loads(
            (output / "summary.json").read_text(encoding="utf-8")
        )
        metadata = json.loads(
            (output / "metadata.json").read_text(encoding="utf-8")
        )
        recorder_outputs_are_complete = bool(
            len(events) == 2
            and summary["event_count"] == 2
            and summary["phase_counts"]["press"] == 1
            and summary["phase_counts"]["lift"] == 1
            and summary["event_type_counts"]["visual_update"] == 1
            and summary["event_type_counts"]["open_loop_prediction"] == 1
            and summary["phase_categorical_counts"]["press"]["image"]
            ["accepted"]["true"]
            == 1
            and summary["phase_categorical_counts"]["press"]["material"]
            ["status"]["committed"]
            == 1
            and summary["phase_metrics"]["lift"]["prediction"]
            ["gap_improvement"]["mean"]
            == 0.02
            and metadata["horizons"] == [1, 3, 5, 10]
        )

    class FakeSimulator:
        def __init__(self) -> None:
            self.restored: list[object] = []
            self.update_count = 0

        def clone_embodied_gaussian_rollout_state(self):
            return "live-state"

        def copy_embodied_gaussian_rollout_state(self, state) -> None:
            self.restored.append(state)

        def update_gaussian_transforms(self) -> None:
            self.update_count += 1

    class FakeDataset:
        def __init__(self) -> None:
            self.requested_timestamps: list[float] = []

        def update_frames(self, timestamp: float) -> None:
            self.requested_timestamps.append(float(timestamp))

    with TemporaryDirectory(prefix="super_stiffness_horizon_") as directory:
        controls = object.__new__(SuperPlaybackControls)
        fake_sim = FakeSimulator()
        fake_dataset = FakeDataset()
        controls.environment = SimpleNamespace(sim=fake_sim)
        controls.dataset_manager = fake_dataset
        controls.playback_timestamps = np.arange(8, dtype=np.float64) / 30.0
        controls.current_frame_index = 5
        controls.current_timestep = float(controls.playback_timestamps[5])
        controls._last_state_index = 5
        controls._last_q_full = torch.zeros(1)
        controls._current_action_phase = "lift"
        controls._stiffness_evaluation_epoch = 1
        controls.stiffness_metrics_recorder = StiffnessMetricsRecorder(
            Path(directory) / "run",
            metadata={"horizons": (1, 3, 5, 10)},
        )
        controls._mass_weighted_particle_rms = lambda reference, current: float(
            torch.sqrt(torch.mean((current - reference) ** 2)).item()
        )

        rollout_state = SimpleNamespace(
            embodied_state=SimpleNamespace(
                physics_state=SimpleNamespace(
                    particle_q=wp.array(
                        np.zeros((2, 3), dtype=np.float32),
                        dtype=wp.vec3,
                        device="cpu",
                    )
                )
            ),
            auxiliary_state=SimpleNamespace(
                paper_distance_stiffness=torch.full((2,), 0.20),
                paper_shape_stiffness=torch.full((2,), 0.004),
            ),
        )
        candidate_stub = SimpleNamespace(
            distance_stiffness=torch.full((2,), 0.22),
            shape_stiffness=torch.full((2,), 0.0044),
            metrics={"status": "committed"},
        )
        calls: list[dict] = []

        def fake_shadow(**kwargs):
            calls.append(kwargs)
            use_candidate = bool(kwargs["use_candidate"])
            return {
                "visual_loss": 0.08 if use_candidate else 0.10,
                "camera_losses": (0.07, 0.09)
                if use_candidate
                else (0.09, 0.11),
                "camera_weight_sums": (10.0, 10.0),
                "camera_active_pixel_counts": (10, 10),
                "camera_mask_coverage_fractions": (0.5, 0.5),
                "distance_loss": 0.01,
                "volume_loss": 0.02,
                "shape_loss": 0.03,
                "minimum_volume_ratio": 0.8,
                "inverted_tetrahedra": 0,
                "maximum_penetration_m": 0.0001,
                "anchor_error_rms_m": 0.00002,
                "anchor_error_maximum_m": 0.00003,
                "grip_active": True,
                "open_loop_residual_rms_m": 0.0001
                if use_candidate
                else 0.0002,
                "open_loop_residual_maximum_m": 0.0002
                if use_candidate
                else 0.0003,
                "particle_positions": torch.full(
                    (2, 3), 0.002 if use_candidate else 0.001
                ),
            }

        controls._run_stiffness_rollout_shadow = fake_shadow
        evaluation = CommittedStiffnessEvaluation(
            evaluation_id=7,
            candidate=candidate_stub,
            rollout_state=rollout_state,
            start_frame_index=0,
            start_phase="press",
            commands=[
                StiffnessToolCommand(index, index / 30.0, index, "lift")
                for index in range(1, 6)
            ],
            pending_horizons={3},
        )
        controls._evaluate_committed_stiffness_horizon(evaluation, 3)
        controls.stiffness_metrics_recorder.close()
        horizon_events = [
            json.loads(line)
            for line in controls.stiffness_metrics_recorder.events_path
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        exact_horizon_rollout_is_complete = bool(
            len(calls) == 4
            and {call["freeze_grip_state_machine"] for call in calls}
            == {True, False}
            and all(len(call["commands"]) == 3 for call in calls)
            and all(call["measure_open_loop_residual"] for call in calls)
            and len(horizon_events) == 2
            and {event["prediction"]["protocol"] for event in horizon_events}
            == {"material_isolation", "end_to_end"}
            and all(
                event["prediction"]["horizon_frames"] == 3
                and event["prediction"]["target_frame_index"] == 3
                and np.isclose(
                    event["prediction"]["gap_improvement"], 0.02
                )
                for event in horizon_events
            )
            and fake_dataset.requested_timestamps
            == [
                float(controls.playback_timestamps[3]),
                float(controls.playback_timestamps[5]),
            ]
            and fake_sim.restored[-1] == "live-state"
        )

    gates = {
        "default_horizons_parse_and_sort": parsed == (1, 3, 5, 10),
        "invalid_horizons_are_rejected": invalid_horizons_rejected == 4,
        "all_required_action_phases_are_classified": phases
        == [
            "idle",
            "press",
            "close",
            "capture",
            "capture",
            "lift",
            "place",
            "release",
        ],
        "nearly_open_gate_allows_stable_contact_and_fast_q7": (
            stable_gate[0] is False and fast_jaw_gate[0] is False
        ),
        "capture_transition_pauses_only_its_current_update": (
            capture_transition_gate[0] is True
            and capture_transition_gate[1] == "capture_or_release"
            and post_transition_gate[0] is False
            and STIFFNESS_TRANSITION_COOLDOWN_UPDATES == 1
        ),
        "capture_aligned_penetration_limit_remains_enforced": (
            excessive_penetration_gate[0] is True
            and excessive_penetration_gate[1] == "excessive_penetration"
            and np.isclose(STIFFNESS_MAXIMUM_PENETRATION_M, 0.0028)
        ),
        "commit_margin_keeps_absolute_positive_floor_only": (
            np.isclose(STIFFNESS_PREDICTION_ABSOLUTE_MARGIN, 1.0e-5)
            and STIFFNESS_PREDICTION_RELATIVE_MARGIN == 0.0
        ),
        "candidate_waits_for_five_frame_evidence": (
            STIFFNESS_COMMIT_VALIDATION_HORIZON_FRAMES == 5
            and pending_metrics is not None
            and pending_metrics["status"] == "pending_horizon"
            and pending_metrics["required_prediction_horizon_frames"] == 5
            and pending_controls._pending_stiffness_validation is not None
        ),
        "precommit_shadow_replays_live_closed_loop": (
            pending_shadow_call.get("freeze_grip_state_machine") is False
            and pending_shadow_call.get("replay_visual_residuals") is True
            and pending_shadow_call.get("replay_after_frame_index") == 10
            and torch.equal(
                pending_shadow_call.get("initial_previous_residual"),
                torch.tensor((0.001,)),
            )
            and pending_shadow_result["visual_loss"] == 0.30
            and pending_shadow_result["endpoint_visual_loss"] == 0.50
            and pending_shadow_result["camera_losses"] == (0.20, 0.40)
        ),
        "horizon_line_search_scales_steps_without_mutating_verified": (
            STIFFNESS_CANDIDATE_LOG_STEP_SCALES == (0.5, 1.0, 1.5, 2.0)
            and torch.allclose(
                scaled_candidate.log_step,
                torch.tensor((0.18, -0.18)),
            )
            and torch.equal(verified_distance, torch.full((2,), 0.20))
            and torch.equal(verified_shape, torch.full((2,), 0.004))
            and scaled_candidate.distance_stiffness[0] > 0.20
            and scaled_candidate.distance_stiffness[1] < 0.20
        ),
        "validated_fixed_lag_state_adoption_is_bounded": (
            adoption_metrics["validated_rollout_adopted"] is True
            and adoption_metrics["validated_rollout_state_rms_m"]
            < STIFFNESS_MAXIMUM_ADOPTED_STATE_RMS_M
            and adoption_sim.copied_states == ["verified-candidate-state"]
            and adoption_sim.update_count == 1
            and torch.equal(
                adoption_controls._previous_visual_residual,
                torch.tensor((0.002,)),
            )
            and rejected_adoption_metrics["validated_rollout_adopted"]
            is False
        ),
        "jsonl_metadata_and_phase_summary_are_complete": (
            recorder_outputs_are_complete
        ),
        "exact_horizon_runs_old_new_in_both_protocols": (
            exact_horizon_rollout_is_complete
        ),
    }
    print(gates)
    if not all(gates.values()):
        raise SystemExit("Stiffness evaluation gate failed")


if __name__ == "__main__":
    main()
