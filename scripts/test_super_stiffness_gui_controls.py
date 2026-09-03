#!/usr/bin/env python3
"""Headless gate for the SUPER runtime stiffness tuning panel."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import torch

from example_embodied_super_offline import (
    ContactGripGuiSettings,
    STIFFNESS_TRANSITION_COOLDOWN_UPDATES,
    SuperPlaybackControls,
)
from embodied_gaussians.physics_simulator.online_tissue_stiffness import (
    OnlineTissueStiffnessSettings,
    ResidualDrivenPaperStiffnessUpdater,
)


class FakeImgui:
    def __init__(self) -> None:
        self.labels: list[str] = []
        self.parameter_labels: list[str] = []
        self.float_ranges: dict[str, tuple[float, float]] = {}
        self.lines: list[str] = []

    def text(self, value: str) -> None:
        self.lines.append(str(value))

    def collapsing_header(self, label: str) -> bool:
        self.labels.append(label)
        return True

    def slider_float(self, label: str, value: float, *args):
        self.labels.append(label)
        self.parameter_labels.append(label)
        if len(args) >= 2:
            self.float_ranges[label] = (float(args[0]), float(args[1]))
        return False, value

    def slider_int(self, label: str, value: int, *args):
        self.labels.append(label)
        self.parameter_labels.append(label)
        return False, value

    def checkbox(self, label: str, value: bool):
        self.labels.append(label)
        self.parameter_labels.append(label)
        return False, value

    def button(self, label: str) -> bool:
        self.labels.append(label)
        return False

    def same_line(self) -> None:
        pass

    def is_item_hovered(self) -> bool:
        return False

    def set_tooltip(self, text: str) -> None:
        raise AssertionError("Tooltip should not be requested in headless gate")


class FakeContactSimulator:
    def __init__(self, projector) -> None:
        self.triangle_skin_contact_projector = projector
        self.configure_kwargs: dict | None = None

    def configure_triangle_skin_contacts(self, tool_shape_ids, **kwargs) -> None:
        self.configure_kwargs = {
            "tool_shape_ids": tuple(tool_shape_ids),
            **kwargs,
        }
        # A real apply replaces the projector and therefore clears all
        # persistent-grip arrays.  A distinct object is enough for this gate.
        replacement = SimpleNamespace(**vars(self.triangle_skin_contact_projector))
        replacement.sample_spacing_m = kwargs["sample_spacing_m"]
        replacement.spread_layers = kwargs["spread_layers"]
        replacement.persistent_grip_support_radius_m = kwargs[
            "persistent_grip_support_radius_m"
        ]
        self.triangle_skin_contact_projector = replacement

    def triangle_skin_contact_metrics(self) -> dict:
        return {"rebuilt": True}


def fake_contact_runtime():
    physics = SimpleNamespace(
        triangle_skin_contact_margin_m=0.0004,
        triangle_skin_query_distance_m=0.010,
        triangle_skin_ccd_velocity_scale=1.0,
        triangle_skin_contact_relaxation=1.0,
        triangle_skin_contact_max_correction_m=0.00003,
        triangle_skin_top_barrier_max_correction_m=0.000008,
        triangle_skin_contact_iterations=1,
        triangle_skin_post_contact_material_iterations=0,
        triangle_skin_final_barrier_max_correction_m=0.0,
        triangle_skin_contact_min_volume_ratio=0.03,
        triangle_skin_contact_substep_stride=1,
        contact_projection_velocity_scale=0.35,
        material_projection_velocity_scale=0.12,
        particle_velocity_damping_per_second=10.0,
    )
    projector = SimpleNamespace(
        tool_shape_ids=(10, 11),
        sample_spacing_m=0.00065,
        spread_layers=0,
        top_support_lateral_radius_m=0.001,
        top_support_depth_m=0.008,
        top_support_weight_scale=0.0,
        top_pressure_shoulder_lateral_radius_m=0.0,
        top_pressure_shoulder_depth_m=0.0,
        top_pressure_shoulder_upward_scale=0.0,
        top_pressure_shoulder_outward_scale=0.0,
        top_pressure_shoulder_bias_direction_world=(0.0, 0.0, 0.0),
        top_pressure_shoulder_bias_start_m=0.0008,
        top_barrier_lateral_tolerance_m=0.00030,
        top_barrier_contact_patch_radius_m=0.0,
        top_barrier_clearance_m=0.00025,
        top_barrier_shape_ids=(10, 11),
        jaw_contact_shape_ids=(10, 11),
        jaw_contact_distal_length_m=0.005,
        top_barrier_distal_length_m=0.003,
        top_barrier_tip_allowance_m=0.0025,
        jaw_friction_coefficient=1.5,
        persistent_grip_enabled=True,
        persistent_grip_minimum_contact_samples_per_jaw=8,
        persistent_grip_nearest_surface_particles=4,
        persistent_grip_maximum_jaw_patch_separation_m=0.006,
        persistent_grip_activation_steps=3,
        persistent_grip_maximum_capture_penetration_m=0.001,
        persistent_grip_minimum_capture_volume_ratio=0.20,
        persistent_grip_closed_angle_max_rad=0.13,
        persistent_grip_release_angle_min_rad=0.15,
        persistent_grip_release_angle_delta_rad=0.08,
        persistent_grip_wide_open_angle_rad=0.50,
        persistent_grip_angle_motion_epsilon_rad=1.0e-4,
        persistent_grip_compliance_m_per_n=0.05,
        persistent_grip_relaxation=1.0,
        persistent_grip_maximum_correction_m=0.0001,
        persistent_grip_transfer_layers=1,
        persistent_grip_minimum_volume_ratio=0.03,
        persistent_grip_support_radius_m=0.004,
        persistent_grip_support_generations=1,
    )
    sim = FakeContactSimulator(projector)
    environment = SimpleNamespace(
        sim=sim,
        physics_settings=physics,
        time=lambda: 12.5,
    )
    return environment, projector


def main() -> None:
    rest = torch.tensor(
        ((0.0, 0.0, 0.0), (0.001, 0.0, 0.0), (0.002, 0.0, 0.0)),
        dtype=torch.float32,
    )
    distance = torch.tensor((0.08, 0.20, 2.20), dtype=torch.float32)
    shape = torch.tensor((0.002, 0.004, 0.025), dtype=torch.float32)
    updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=torch.zeros(3, dtype=torch.bool),
        edges=torch.tensor(((0, 1), (1, 2)), dtype=torch.long),
        distance_stiffness=distance,
        shape_stiffness=shape,
    )
    controls = object.__new__(SuperPlaybackControls)
    controls.stiffness_updater = updater
    controls._startup_stiffness_settings = updater.settings
    controls._stiffness_gui_draft = replace(
        updater.settings,
        log_learning_rate=0.12,
        signal_ema_decay=0.60,
    )
    controls._stiffness_gui_message = ""
    controls.playing = False
    controls._pending_stiffness_validation = None
    controls._stiffness_history = deque((object(), object()))
    controls._last_stiffness_validation_metrics = {"stale": True}
    controls._previous_visual_residual = torch.ones(1)
    controls._stiffness_transition_cooldown = 0
    updater.signal_ema.fill_(0.75)

    controls._apply_stiffness_gui_draft()
    applied_safely = bool(
        updater.settings.log_learning_rate == 0.12
        and updater.settings.signal_ema_decay == 0.60
        and torch.allclose(distance, torch.tensor((0.10, 0.20, 2.00)))
        and torch.allclose(shape, torch.tensor((0.003, 0.004, 0.020)))
        and torch.count_nonzero(updater.signal_ema) == 0
        and len(controls._stiffness_history) == 0
        and controls._last_stiffness_validation_metrics is None
        and controls._previous_visual_residual is None
        and controls._stiffness_transition_cooldown
        == STIFFNESS_TRANSITION_COOLDOWN_UPDATES
    )

    fake = FakeImgui()
    controls._draw_stiffness_panel(fake)
    expected_controls = {
        "Distance lower bound",
        "Distance upper bound",
        "Shape lower bound",
        "Shape upper bound",
        "Learning rate (log space)",
        "EMA new-evidence weight",
        "Hardening bias",
        "Shape update gain",
        "Maximum log step",
        "Minimum residual (mm)",
        "Residual full scale (mm)",
        "Minimum deformation (mm)",
        "Deformation full scale (mm)",
        "Spatial smoothing passes",
        "Neighbor smoothing blend",
        "Rejected EMA keep ratio",
    }
    all_formula_controls_exposed = expected_controls.issubset(fake.labels)
    shape_lower_range_is_expanded = bool(
        fake.float_ranges.get("Shape lower bound") == (0.0001, 0.030)
        and fake.float_ranges.get("Shape upper bound") == (0.001, 0.100)
        and fake.float_ranges.get("Distance lower bound") == (0.01, 1.00)
        and fake.float_ranges.get("Distance upper bound") == (0.10, 10.00)
    )

    controls.playing = True
    blocked_draft = replace(updater.settings, log_learning_rate=0.21)
    controls._stiffness_gui_draft = blocked_draft
    controls._apply_stiffness_gui_draft()
    running_apply_is_blocked = bool(
        updater.settings.log_learning_rate == 0.12
        and "Pause playback" in controls._stiffness_gui_message
    )

    environment, old_projector = fake_contact_runtime()
    contact_controls = object.__new__(SuperPlaybackControls)
    contact_controls.environment = environment
    contact_controls.stiffness_updater = None
    contact_controls.playing = False
    contact_controls._pending_stiffness_validation = None
    contact_controls._stiffness_history = deque((object(),))
    contact_controls._last_stiffness_gate_timestep = 1.0
    contact_controls._last_stiffness_gate_jaw_angle = 0.1
    contact_controls._last_stiffness_grip_active = True
    contact_controls._last_stiffness_contact_count = 12
    contact_controls._last_stiffness_validation_metrics = {"stale": True}
    contact_controls._previous_visual_residual = torch.ones(1)
    contact_controls._stiffness_transition_cooldown = 0
    contact_controls._contact_ui_metrics = None
    contact_controls._last_contact_ui_refresh_time = 0.0
    startup_contact = ContactGripGuiSettings.from_runtime(
        environment.physics_settings,
        old_projector,
    )
    contact_controls._startup_contact_grip_settings = startup_contact
    contact_controls._contact_grip_gui_draft = replace(
        startup_contact,
        top_barrier_tip_allowance_m=0.0027,
        grip_maximum_capture_penetration_m=0.0015,
        grip_support_radius_m=0.005,
        grip_maximum_correction_m=0.00005,
        grip_compliance_m_per_n=0.20,
        material_projection_velocity_scale=0.04,
        particle_velocity_damping_per_second=30.0,
    )
    contact_controls._contact_grip_gui_message = ""
    contact_controls._apply_contact_grip_gui_draft()
    contact_apply_is_atomic = bool(
        environment.sim.triangle_skin_contact_projector is not old_projector
        and environment.sim.configure_kwargs[
            "top_barrier_tip_allowance_m"
        ]
        == 0.0027
        and environment.sim.configure_kwargs[
            "persistent_grip_maximum_capture_penetration_m"
        ]
        == 0.0015
        and environment.sim.configure_kwargs[
            "persistent_grip_support_radius_m"
        ]
        == 0.005
        and environment.sim.configure_kwargs[
            "persistent_grip_maximum_correction_m"
        ]
        == 0.00005
        and environment.sim.configure_kwargs[
            "persistent_grip_compliance_m_per_n"
        ]
        == 0.20
        and environment.physics_settings.material_projection_velocity_scale
        == 0.04
        and environment.physics_settings.particle_velocity_damping_per_second
        == 30.0
        and len(contact_controls._stiffness_history) == 0
        and contact_controls._last_stiffness_grip_active is False
        and contact_controls._contact_ui_metrics == {"rebuilt": True}
        and "old grip anchors cleared"
        in contact_controls._contact_grip_gui_message
    )

    contact_fake = FakeImgui()
    contact_controls._draw_contact_grip_tuning_panel(contact_fake)
    expected_contact_controls = {
        "Tip entry allowance (mm)",
        "Capture max penetration (mm)",
        "Grip support radius (mm)",
        "Grip correction cap (mm/substep)",
        "Grip compliance (m/N)",
        "Material/grip velocity transfer",
        "Particle velocity damping (/s)",
    }
    only_requested_contact_controls_are_exposed = bool(
        set(contact_fake.parameter_labels) == expected_contact_controls
        and contact_fake.float_ranges.get("Tip entry allowance (mm)")
        == (0.0, 4.99)
        and contact_fake.float_ranges.get("Capture max penetration (mm)")
        == (0.10, 10.00)
        and contact_fake.float_ranges.get("Grip support radius (mm)")
        == (0.0, 30.0)
        and contact_fake.float_ranges.get(
            "Grip correction cap (mm/substep)"
        )
        == (0.005, 2.000)
        and contact_fake.float_ranges.get("Grip compliance (m/N)")
        == (0.0, 2.0)
        and contact_fake.float_ranges.get(
            "Material/grip velocity transfer"
        )
        == (0.0, 1.0)
        and contact_fake.float_ranges.get(
            "Particle velocity damping (/s)"
        )
        == (0.0, 100.0)
    )

    contact_controls.playing = True
    old_capture_penetration = environment.sim.configure_kwargs[
        "persistent_grip_maximum_capture_penetration_m"
    ]
    contact_controls._contact_grip_gui_draft = replace(
        contact_controls._contact_grip_gui_draft,
        grip_maximum_capture_penetration_m=0.0020,
    )
    contact_controls._apply_contact_grip_gui_draft()
    running_contact_apply_is_blocked = bool(
        environment.sim.configure_kwargs[
            "persistent_grip_maximum_capture_penetration_m"
        ]
        == old_capture_penetration
        and "Pause playback" in contact_controls._contact_grip_gui_message
    )
    rejected_invalid_contact_combinations = 0
    invalid_drafts = (
        replace(
            startup_contact,
            query_distance_m=startup_contact.contact_margin_m * 0.5,
        ),
        replace(
            startup_contact,
            top_barrier_tip_allowance_m=(
                startup_contact.top_barrier_distal_length_m
            ),
        ),
        replace(
            startup_contact,
            grip_release_angle_min_rad=(
                startup_contact.grip_closed_angle_max_rad
            ),
        ),
    )
    for invalid_draft in invalid_drafts:
        try:
            invalid_draft.validate()
        except ValueError:
            rejected_invalid_contact_combinations += 1

    gates = {
        "paused_apply_clips_verified_and_clears_old_evidence": applied_safely,
        "all_formula_settings_are_exposed": all_formula_controls_exposed,
        "stiffness_bound_gui_ranges_are_materially_wider": (
            shape_lower_range_is_expanded
        ),
        "apply_is_blocked_while_playing": running_apply_is_blocked,
        "contact_grip_apply_rebuilds_and_clears_old_anchors": (
            contact_apply_is_atomic
        ),
        "entry_grip_and_jump_stabilization_controls_are_exposed": (
            only_requested_contact_controls_are_exposed
        ),
        "contact_grip_apply_is_blocked_while_playing": (
            running_contact_apply_is_blocked
        ),
        "invalid_contact_geometry_and_grip_angles_are_rejected": (
            rejected_invalid_contact_combinations == len(invalid_drafts)
        ),
    }
    print(gates)
    if not all(gates.values()):
        raise SystemExit("Stiffness GUI control gate failed")


if __name__ == "__main__":
    main()
