#!/usr/bin/env python3
"""Smoke gate for the SUPER soft-tissue triangle-skin contact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from embodied_environments.super_embodied.super_embodied import (  # noqa: E402
    ADAPTIVE_TISSUE_CONTACT_SPREAD_LAYERS,
    ADAPTIVE_TISSUE_TOP_CONTACT_SUPPORT_DEPTH_M,
    ADAPTIVE_TISSUE_TOP_CONTACT_SUPPORT_RADIUS_M,
    PAPER_SOFT_TISSUE_CONTACT_SPREAD_LAYERS,
    PAPER_SOFT_TISSUE_PRESSURE_SHOULDER_BIAS_DIRECTION_WORLD,
    PAPER_SOFT_TISSUE_PRESSURE_SHOULDER_BIAS_START_M,
    PAPER_SOFT_TISSUE_PRESSURE_SHOULDER_RADIUS_M,
    PAPER_SOFT_TISSUE_PRESSURE_SHOULDER_UPWARD_SCALE,
    PAPER_SOFT_TISSUE_TOP_BARRIER_CONTACT_PATCH_RADIUS_M,
    PAPER_SOFT_TISSUE_TOP_BARRIER_LATERAL_TOLERANCE_M,
    PAPER_SOFT_TISSUE_TOP_SUPPORT_DEPTH_M,
    PAPER_SOFT_TISSUE_TOP_SUPPORT_RADIUS_M,
    PAPER_SOFT_TISSUE_TOP_SUPPORT_WEIGHT_SCALE,
    PSM_POSE_DRIVER_PATHS,
    apply_psm_lnd_pose,
    build_environment,
    set_psm_tissue_collisions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate real PSM-to-soft-tissue triangle-skin contact."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--tissue-mode",
        choices=("paper_pbd", "paper_soft", "adaptive_soft"),
        default="adaptive_soft",
    )
    parser.add_argument(
        "--psm-pose-driver",
        choices=tuple(PSM_POSE_DRIVER_PATHS),
        default="depth_then_visual",
    )
    parser.add_argument("--contact-state-index", type=int, default=895)
    parser.add_argument("--deep-state-index", type=int, default=1065)
    parser.add_argument("--physics-steps", type=int, default=10)
    parser.add_argument(
        "--trajectory-stride",
        type=int,
        default=20,
        help="Pose-state stride for the progressive video-path regression.",
    )
    parser.add_argument("--roll-offset-deg", type=float, default=-27.0)
    parser.add_argument(
        "--warp-cache-dir",
        type=Path,
        default=Path("/tmp/warp-super-triangle-skin-contact"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=(
            REPO_ROOT
            / "data/super/grasp5_native/tissue_multiview_v1/"
            "soft_tissue_adaptive_v1/triangle_skin_contact_smoke_report.json"
        ),
    )
    return parser.parse_args()


def apply_pose(environment, state_index: int, offsets: np.ndarray) -> None:
    apply_psm_lnd_pose(
        environment,
        state_index,
        joint_offsets=offsets,
        update_gaussians=False,
    )


def detect(environment) -> dict:
    settings = environment.physics_settings
    projector = environment.sim.triangle_skin_contact_projector
    projector.detect(
        environment.sim.model,
        environment.sim.state_0,
        settings.dt / settings.substeps,
        contact_margin_m=settings.triangle_skin_contact_margin_m,
        query_distance_m=settings.triangle_skin_query_distance_m,
        ccd_velocity_scale=0.0,
        friction_coefficient=0.0,
        relaxation=1.0,
    )
    return projector.metrics()


def run_steps(
    environment,
    state_index: int,
    offsets: np.ndarray,
    count: int,
) -> tuple[float, dict]:
    projector = environment.sim.triangle_skin_contact_projector
    started = time.perf_counter()
    for _ in range(count):
        environment.step(compute_visual_forces=False)
        apply_pose(environment, state_index, offsets)
    wp.synchronize()
    elapsed = time.perf_counter() - started
    return elapsed, projector.metrics()


def particle_metrics(environment, initial_positions: torch.Tensor) -> dict:
    model = environment.sim.model
    state = environment.sim.state_0
    positions = wp.to_torch(state.particle_q)
    velocities = wp.to_torch(state.particle_qd)
    inverse_mass = wp.to_torch(model.particle_inv_mass)
    tetrahedra = wp.to_torch(model.tet_indices).long()
    rest_inverse = wp.to_torch(model.tet_poses)
    deformation_matrix = torch.stack(
        (
            positions[tetrahedra[:, 1]] - positions[tetrahedra[:, 0]],
            positions[tetrahedra[:, 2]] - positions[tetrahedra[:, 0]],
            positions[tetrahedra[:, 3]] - positions[tetrahedra[:, 0]],
        ),
        dim=2,
    )
    volume_ratio = torch.linalg.det(deformation_matrix @ rest_inverse)
    dynamic = inverse_mass > 0.0
    static = ~dynamic
    displacement = torch.linalg.vector_norm(
        positions - initial_positions, dim=1
    )
    return {
        "finite": bool(
            torch.isfinite(positions).all()
            and torch.isfinite(velocities).all()
        ),
        "inverted_tetrahedra": int((volume_ratio <= 0.0).sum().item()),
        "minimum_tetrahedron_volume_ratio": float(
            volume_ratio.min().item()
        ),
        "maximum_dynamic_speed_m_s": float(
            torch.linalg.vector_norm(velocities[dynamic], dim=1).max().item()
        ),
        "maximum_dynamic_displacement_m": float(
            displacement[dynamic].max().item()
        ),
        "p99_dynamic_displacement_m": float(
            torch.quantile(displacement[dynamic], 0.99).item()
        ),
        "maximum_anchor_drift_m": float(
            displacement[static].max().item()
        ),
    }


def main() -> None:
    args = parse_args()
    if args.physics_steps < 1:
        raise ValueError("--physics-steps must be positive")
    if args.trajectory_stride < 1:
        raise ValueError("--trajectory-stride must be positive")
    args.warp_cache_dir.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(args.warp_cache_dir)
    wp.init()

    environment = build_environment(
        add_gaussians=True,
        device=args.device,
        tissue_mode=args.tissue_mode,
        psm_pose_driver_path=PSM_POSE_DRIVER_PATHS[args.psm_pose_driver],
        psm_visual_tip_only=True,
    )
    model = environment.sim.model
    projector = environment.sim.triangle_skin_contact_projector
    offsets = np.zeros(7, dtype=np.float64)
    offsets[3] = np.deg2rad(args.roll_offset_deg)
    initial_state = environment.sim.clone_embodied_gaussian_state()
    initial_positions = wp.to_torch(
        initial_state.physics_state.particle_q
    ).clone()

    apply_pose(environment, args.deep_state_index, offsets)
    deep_pose_metrics = detect(environment)
    deep_pose_enable_result = set_psm_tissue_collisions(
        environment, True
    )
    set_psm_tissue_collisions(environment, False)

    environment.sim.copy_embodied_gaussian_state(initial_state)
    apply_pose(environment, args.contact_state_index, offsets)
    before_contact = detect(environment)
    precontact_state = environment.sim.clone_embodied_gaussian_state()

    # Compile and time the same runtime with contact disabled.
    environment.step(compute_visual_forces=False)
    wp.synchronize()
    environment.sim.copy_embodied_gaussian_state(precontact_state)
    apply_pose(environment, args.contact_state_index, offsets)
    baseline_seconds, _ = run_steps(
        environment,
        args.contact_state_index,
        offsets,
        args.physics_steps,
    )

    environment.sim.copy_embodied_gaussian_state(precontact_state)
    apply_pose(environment, args.contact_state_index, offsets)
    contact_enable_result = set_psm_tissue_collisions(
        environment, True
    )
    # Compile the contact-enabled graph, then restore the exact initial state.
    environment.step(compute_visual_forces=False)
    wp.synchronize()
    environment.sim.copy_embodied_gaussian_state(precontact_state)
    apply_pose(environment, args.contact_state_index, offsets)
    psm_body_ids = environment.super_psm_lnd_body_ids[0]
    expected_psm_pose = wp.to_torch(
        environment.sim.state_0.body_q
    )[psm_body_ids].clone()
    contact_seconds, solver_last_pass = run_steps(
        environment,
        args.contact_state_index,
        offsets,
        args.physics_steps,
    )
    after_contact = detect(environment)
    actual_psm_pose = wp.to_torch(
        environment.sim.state_0.body_q
    )[psm_body_ids]
    maximum_psm_pose_component_error = float(
        torch.max(torch.abs(actual_psm_pose - expected_psm_pose)).item()
    )
    maximum_psm_translation_error_m = float(
        torch.linalg.vector_norm(
            actual_psm_pose[:, :3] - expected_psm_pose[:, :3], dim=1
        ).max().item()
    )
    dynamics = particle_metrics(environment, initial_positions)

    # Regression for the original failure: contact is enabled at a valid
    # pre-contact pose, then the kinematic tool advances to a pose that would
    # begin almost 3 mm below the undeformed top surface.  The oriented top
    # barrier must move tissue instead of modifying the tool pose or letting
    # the tool tunnel through and become "outside" again.
    environment.sim.copy_embodied_gaussian_state(precontact_state)
    apply_pose(environment, args.contact_state_index, offsets)
    active_deep_enable_result = set_psm_tissue_collisions(
        environment, True
    )
    apply_pose(environment, args.deep_state_index, offsets)
    deep_expected_psm_pose = wp.to_torch(
        environment.sim.state_0.body_q
    )[psm_body_ids].clone()
    active_deep_seconds, active_deep_solver_last_pass = run_steps(
        environment,
        args.deep_state_index,
        offsets,
        args.physics_steps,
    )
    active_deep_after = detect(environment)
    deep_actual_psm_pose = wp.to_torch(
        environment.sim.state_0.body_q
    )[psm_body_ids]
    active_deep_pose_error = float(
        torch.max(
            torch.abs(deep_actual_psm_pose - deep_expected_psm_pose)
        ).item()
    )
    active_deep_dynamics = particle_metrics(
        environment, initial_positions
    )

    # Normal playback advances through intermediate absolute video poses
    # instead of teleporting 170 pose states in one physics frame.
    environment.sim.copy_embodied_gaussian_state(initial_state)
    apply_pose(environment, args.contact_state_index, offsets)
    environment.sim.sync_kinematic_body_interpolation()
    trajectory_enable_result = set_psm_tissue_collisions(
        environment, True
    )
    for _ in range(args.physics_steps):
        environment.step(compute_visual_forces=False)
        apply_pose(environment, args.contact_state_index, offsets)
    environment.sim.sync_kinematic_body_interpolation()
    trajectory_state_indices = list(
        range(
            args.contact_state_index,
            args.deep_state_index + 1,
            args.trajectory_stride,
        )
    )
    if trajectory_state_indices[-1] != args.deep_state_index:
        trajectory_state_indices.append(args.deep_state_index)
    trajectory_expected_psm_pose = wp.to_torch(
        environment.sim.state_0.body_q
    )[psm_body_ids].clone()
    trajectory_started = time.perf_counter()
    for trajectory_state_index in trajectory_state_indices[1:]:
        apply_pose(environment, trajectory_state_index, offsets)
        trajectory_expected_psm_pose = wp.to_torch(
            environment.sim.state_0.body_q
        )[psm_body_ids].clone()
        environment.step(compute_visual_forces=False)
    wp.synchronize()
    trajectory_seconds = time.perf_counter() - trajectory_started
    trajectory_after = detect(environment)
    trajectory_dynamics = particle_metrics(
        environment, initial_positions
    )
    trajectory_actual_psm_pose = wp.to_torch(
        environment.sim.state_0.body_q
    )[psm_body_ids]
    trajectory_pose_error = float(
        torch.max(
            torch.abs(
                trajectory_actual_psm_pose
                - trajectory_expected_psm_pose
            )
        ).item()
    )
    particle_radius = wp.to_torch(model.particle_radius)
    paper_tissue_mode = args.tissue_mode in {"paper_pbd", "paper_soft"}
    expected_support_radius_m = (
        PAPER_SOFT_TISSUE_TOP_SUPPORT_RADIUS_M
        if paper_tissue_mode
        else ADAPTIVE_TISSUE_TOP_CONTACT_SUPPORT_RADIUS_M
    )
    expected_support_depth_m = (
        PAPER_SOFT_TISSUE_TOP_SUPPORT_DEPTH_M
        if paper_tissue_mode
        else ADAPTIVE_TISSUE_TOP_CONTACT_SUPPORT_DEPTH_M
    )
    expected_support_weight_scale = (
        PAPER_SOFT_TISSUE_TOP_SUPPORT_WEIGHT_SCALE
        if paper_tissue_mode
        else 0.15
    )
    support_entry_counts_match_configuration = (
        after_contact["top_support_entry_count"] == 0
        and active_deep_after["top_support_entry_count"] == 0
        if expected_support_weight_scale == 0.0
        else after_contact["top_support_entry_count"] > 0
        and active_deep_after["top_support_entry_count"] > 0
    )

    gates = {
        "selected_soft_tissue_mode_is_active": (
            environment.super_tissue_mode == args.tissue_mode
        ),
        "mechanics_particles_have_zero_collision_radius": bool(
            torch.count_nonzero(particle_radius).item() == 0
        ),
        "tool_proxy_sample_hard_limit": projector.sample_count <= 15000,
        "deep_initial_overlap_can_enable_recovery": bool(
            deep_pose_enable_result
            and deep_pose_metrics["top_minimum_signed_distance_m"] < 0.0
        ),
        "gentle_contact_can_enable": bool(contact_enable_result),
        "contact_candidates_generated": (
            before_contact["contact_count"] > 0
        ),
        "postsolve_skin_not_crossed": (
            after_contact["top_minimum_signed_distance_m"]
            >= -environment.physics_settings.triangle_skin_contact_margin_m
            - 1.0e-5
        ),
        "pose_driver_is_preserved": bool(
            not environment.super_psm_tissue_kinematic_gap_guard_enabled
            and maximum_psm_pose_component_error <= 1.0e-7
            and maximum_psm_translation_error_m <= 1.0e-7
        ),
        "active_deep_pose_top_skin_not_crossed": bool(
            active_deep_enable_result
            and active_deep_after["top_minimum_signed_distance_m"]
            >= -environment.physics_settings.triangle_skin_contact_margin_m
            - 1.0e-5
            and active_deep_after["contact_count"] > 0
        ),
        "jaw_uses_triangle_skin_contact_without_persistent_grip": bool(
            not after_contact["persistent_grip_constraint_enabled"]
            and before_contact["jaw_triangle_contact_count"] > 0
            and deep_pose_metrics["jaw_triangle_contact_count"] > 0
            and after_contact["jaw_friction_coefficient"] >= 1.0
            and before_contact["top_barrier_contact_count"] > 0
            and deep_pose_metrics["top_barrier_contact_count"] > 0
            and set(after_contact["top_barrier_shape_ids"])
            == set(after_contact["jaw_contact_shape_ids"])
        ),
        "jaw_top_plane_uses_configured_neighbor_support": bool(
            support_entry_counts_match_configuration
            and after_contact["top_support_weight_scale"]
            == expected_support_weight_scale
            and after_contact["top_support_lateral_radius_m"]
            == expected_support_radius_m
            and after_contact["top_support_depth_m"]
            == expected_support_depth_m
            and after_contact["contact_spread_layers"]
            == (
                PAPER_SOFT_TISSUE_CONTACT_SPREAD_LAYERS
                if paper_tissue_mode
                else ADAPTIVE_TISSUE_CONTACT_SPREAD_LAYERS
            )
        ),
        "paper_top_press_uses_narrow_barrier_without_active_shoulder": bool(
            not paper_tissue_mode
            or (
                after_contact["top_barrier_lateral_tolerance_m"]
                == PAPER_SOFT_TISSUE_TOP_BARRIER_LATERAL_TOLERANCE_M
                and after_contact["top_barrier_contact_patch_radius_m"]
                == PAPER_SOFT_TISSUE_TOP_BARRIER_CONTACT_PATCH_RADIUS_M
                and PAPER_SOFT_TISSUE_PRESSURE_SHOULDER_RADIUS_M == 0.0
                and PAPER_SOFT_TISSUE_PRESSURE_SHOULDER_UPWARD_SCALE == 0.0
                and tuple(
                    after_contact[
                        "top_pressure_shoulder_bias_direction_world"
                    ]
                )
                == PAPER_SOFT_TISSUE_PRESSURE_SHOULDER_BIAS_DIRECTION_WORLD
                and after_contact["top_pressure_shoulder_bias_start_m"]
                == PAPER_SOFT_TISSUE_PRESSURE_SHOULDER_BIAS_START_M
            )
        ),
        "jaw_query_is_wider_than_contact_margin": bool(
            environment.physics_settings.triangle_skin_query_distance_m
            > environment.physics_settings.triangle_skin_contact_margin_m
        ),
        "active_deep_pose_driver_is_preserved": (
            active_deep_pose_error <= 1.0e-7
        ),
        "progressive_video_pose_top_skin_not_crossed": bool(
            trajectory_enable_result
            and trajectory_after["top_minimum_signed_distance_m"]
            >= -environment.physics_settings.triangle_skin_contact_margin_m
            - 1.0e-5
            and trajectory_pose_error <= 2.0e-7
        ),
        "progressive_video_pose_tissue_not_inverted": (
            trajectory_dynamics["inverted_tetrahedra"] == 0
        ),
        "finite": dynamics["finite"],
        "anchors_fixed": dynamics["maximum_anchor_drift_m"] <= 1.0e-9,
        "bounded_speed": dynamics["maximum_dynamic_speed_m_s"] <= 1.0,
        "contact_runtime_below_60ms": (
            contact_seconds / args.physics_steps < 0.060
            if args.device.startswith("cuda")
            else True
        ),
    }
    report = {
        "device": str(wp.get_device(args.device)),
        "tissue": {
            "mode": environment.super_tissue_mode,
            "young_modulus_pa": environment.super_tissue_young_modulus_pa,
            "poisson_ratio": environment.super_tissue_poisson_ratio,
            "velocity_damping_per_second": (
                environment.physics_settings
                .particle_velocity_damping_per_second
            ),
            "gravity_m_s2": environment.super_tissue_gravity_m_s2,
            "particles": model.particle_count,
            "tetrahedra": model.tet_count,
            "collision_skin_triangles": len(
                environment.sim.builder.soft_collision_skin_faces
            ),
            "collision_skin_nodes": projector.skin_node_count,
            "contact_safety_tetrahedra": projector.contact_tet_count,
        },
        "contact_configuration": {
            "tool_samples": projector.sample_count,
            "samples_per_shape": projector.samples_per_shape,
            "margin_m": (
                environment.physics_settings.triangle_skin_contact_margin_m
            ),
            "jaw_query_distance_m": (
                environment.physics_settings.triangle_skin_query_distance_m
            ),
            "friction": environment.physics_settings.triangle_skin_friction,
            "substep_stride": (
                environment.physics_settings
                .triangle_skin_contact_substep_stride
            ),
            "maximum_correction_m": (
                environment.physics_settings
                .triangle_skin_contact_max_correction_m
            ),
            "surface_constraint_iterations": (
                environment.physics_settings
                .triangle_skin_contact_iterations
            ),
            "top_barrier_clearance_m": (
                projector.top_barrier_clearance_m
            ),
            "contact_minimum_volume_ratio": (
                environment.physics_settings
                .triangle_skin_contact_min_volume_ratio
            ),
            "material_minimum_volume_ratio": (
                environment.physics_settings.material_min_volume_ratio
            ),
            "kinematic_minimum_gap_m": getattr(
                environment,
                "super_psm_tissue_kinematic_min_gap_m",
                None,
            ),
            "kinematic_gap_guard_enabled": getattr(
                environment,
                "super_psm_tissue_kinematic_gap_guard_enabled",
                False,
            ),
            "particle_shape_contacts": (
                environment.physics_settings.enable_particle_shape_contacts
            ),
            "particle_particle_contacts": (
                environment.physics_settings.enable_particle_particle_contacts
            ),
        },
        "deep_pose_preflight": {
            "pose_driver": args.psm_pose_driver,
            "state_index": args.deep_state_index,
            "metrics": deep_pose_metrics,
            "enable_result": deep_pose_enable_result,
        },
        "gentle_contact": {
            "pose_driver": args.psm_pose_driver,
            "state_index": args.contact_state_index,
            "before": before_contact,
            "after": after_contact,
            "solver_last_pass": solver_last_pass,
            "kinematic_guard_gap_m": getattr(
                environment,
                "super_psm_tissue_kinematic_guard_gap_m",
                None,
            ),
            "kinematic_guard_last_offset_m": getattr(
                environment,
                "super_psm_tissue_kinematic_guard_offset_m",
                0.0,
            ),
            "maximum_pose_component_error": (
                maximum_psm_pose_component_error
            ),
            "maximum_translation_error_m": (
                maximum_psm_translation_error_m
            ),
        },
        "active_deep_contact": {
            "state_index": args.deep_state_index,
            "enable_result": active_deep_enable_result,
            "after": active_deep_after,
            "solver_last_pass": active_deep_solver_last_pass,
            "maximum_pose_component_error": active_deep_pose_error,
            "dynamics": active_deep_dynamics,
            "seconds": active_deep_seconds,
        },
        "progressive_video_contact": {
            "state_indices": trajectory_state_indices,
            "enable_result": trajectory_enable_result,
            "after": trajectory_after,
            "maximum_pose_component_error": trajectory_pose_error,
            "dynamics": trajectory_dynamics,
            "seconds": trajectory_seconds,
        },
        "dynamics": dynamics,
        "performance": {
            "steps": args.physics_steps,
            "contact_disabled_ms_per_frame": (
                baseline_seconds / args.physics_steps * 1000.0
            ),
            "contact_enabled_ms_per_frame": (
                contact_seconds / args.physics_steps * 1000.0
            ),
            "contact_overhead_ms_per_frame": (
                (contact_seconds - baseline_seconds)
                / args.physics_steps
                * 1000.0
            ),
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
