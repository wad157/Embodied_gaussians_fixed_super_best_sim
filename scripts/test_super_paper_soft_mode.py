#!/usr/bin/env python3
"""Gate the fixed-parameter paper-style SUPER soft-tissue baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from embodied_environments.super_embodied.super_embodied import (  # noqa: E402
    ADAPTIVE_TISSUE_CONTACT_QUERY_DISTANCE_M,
    PAPER_SOFT_TISSUE_GRAVITY_M_S2,
    PAPER_SOFT_TISSUE_GRIP_ACTIVATION_STEPS,
    PAPER_SOFT_TISSUE_GRIP_COMPLIANCE_M_PER_N,
    PAPER_SOFT_TISSUE_GRIP_MAX_CAPTURE_PENETRATION_M,
    PAPER_SOFT_TISSUE_GRIP_MAX_CORRECTION_M,
    PAPER_SOFT_TISSUE_GRIP_MAX_PATCH_SEPARATION_M,
    PAPER_SOFT_TISSUE_GRIP_MIN_CONTACT_SAMPLES_PER_JAW,
    PAPER_SOFT_TISSUE_GRIP_NEAREST_SURFACE_PARTICLES,
    PAPER_SOFT_TISSUE_GRIP_RELAXATION,
    PAPER_SOFT_TISSUE_GRIP_SUPPORT_RADIUS_M,
    PAPER_SOFT_TISSUE_GRIP_SUPPORT_GENERATIONS,
    PAPER_SOFT_TISSUE_MATERIAL_ITERATIONS,
    PAPER_SOFT_TISSUE_MATERIAL_PROJECTION_VELOCITY_SCALE,
    PAPER_SOFT_TISSUE_MATERIAL_RELAXATION,
    PAPER_PBD_TISSUE_PATH,
    PAPER_SOFT_TISSUE_POISSON_RATIO,
    PAPER_SOFT_TISSUE_SURFACE_ITERATIONS,
    PAPER_SOFT_TISSUE_SURFACE_MAX_CORRECTION_M,
    PAPER_SOFT_TISSUE_TOP_BARRIER_DISTAL_LENGTH_M,
    PAPER_SOFT_TISSUE_TOP_BARRIER_CONTACT_PATCH_RADIUS_M,
    PAPER_SOFT_TISSUE_TOP_BARRIER_MAX_CORRECTION_M,
    PAPER_SOFT_TISSUE_TOP_BARRIER_TIP_ALLOWANCE_M,
    PAPER_SOFT_TISSUE_CONTACT_SUBSTEP_STRIDE,
    PAPER_SOFT_TISSUE_CONTACT_SPREAD_LAYERS,
    PAPER_SOFT_TISSUE_TOP_CLEARANCE_M,
    PAPER_SOFT_TISSUE_TOP_SUPPORT_DEPTH_M,
    PAPER_SOFT_TISSUE_TOP_SUPPORT_RADIUS_M,
    PAPER_SOFT_TISSUE_VELOCITY_DAMPING_PER_SECOND,
    PAPER_SOFT_TISSUE_YOUNG_MODULUS_PA,
    PSM_RAW_PAPER_LND_DENSE_CONTACT_UNBOUNDED_XYZ_SURFACE_GAUSSIANS_PATH,
    PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_UNBOUNDED_XYZ_POSE_DRIVER_PATH,
    SUPER_SOFT_VISUAL_FORCE_MAX_PARTICLE_ACCELERATION_M_S2,
    build_environment,
    set_psm_tissue_collisions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify paper_soft uses fixed XPBD material parameters, preserves "
            "soft Gaussian skinning and stronger visual forces, supports "
            "tool contact, and excludes depth-residual/stiffness optimization."
        )
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument(
        "--warp-cache-dir",
        type=Path,
        default=Path("/tmp/warp-super-paper-soft-cache"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.warp_cache_dir.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(args.warp_cache_dir)
    wp.init()

    environment = build_environment(
        num_envs=1,
        add_gaussians=True,
        device=args.device,
        psm_pose_driver_path=(
            PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_UNBOUNDED_XYZ_POSE_DRIVER_PATH
        ),
        psm_visual_tip_only=False,
        tissue_mode="paper_soft",
    )
    simulator = environment.sim
    model = simulator.model
    gaussian_model = simulator.gaussian_model
    handle = environment.super_tissue_soft_handle
    if handle is None:
        raise RuntimeError("paper_soft did not create a soft-body handle")

    tet_materials = wp.to_torch(model.tet_materials).detach()
    mu_expected = PAPER_SOFT_TISSUE_YOUNG_MODULUS_PA / (
        2.0 * (1.0 + PAPER_SOFT_TISSUE_POISSON_RATIO)
    )
    lambda_expected = (
        PAPER_SOFT_TISSUE_YOUNG_MODULUS_PA
        * PAPER_SOFT_TISSUE_POISSON_RATIO
        / (
            (1.0 + PAPER_SOFT_TISSUE_POISSON_RATIO)
            * (1.0 - 2.0 * PAPER_SOFT_TISSUE_POISSON_RATIO)
        )
    )
    gradient_enabled = (
        ~simulator.visual_forces._gaussians_not_involved_in_visual_forces
    )
    expected_gradient_enabled = torch.zeros_like(gradient_enabled)
    expected_gradient_enabled[gaussian_model.soft_gaussian_ids] = True
    with np.load(PAPER_PBD_TISSUE_PATH, allow_pickle=False) as tissue_asset:
        expected_tissue_gaussians = len(
            tissue_asset["gaussian_rest_means_table"]
        )
        tissue_rest_means = tissue_asset[
            "gaussian_rest_means_table"
        ].astype(np.float32)
        tissue_scales = tissue_asset["gaussian_scales"].astype(np.float32)
        tissue_first_pair_observed = tissue_asset[
            "gaussian_first_pair_observed"
        ].astype(bool)
        tissue_later_only_observed = tissue_asset[
            "gaussian_later_only_observed"
        ].astype(bool)
        tissue_observation_count = tissue_asset[
            "gaussian_observation_count"
        ].astype(np.uint8)
        tissue_visual_vertices = tissue_asset[
            "visual_surface_rest_vertices_table"
        ].astype(np.float32)
        tissue_visual_faces = tissue_asset["visual_surface_faces"].astype(np.int32)
        tissue_visual_source_class = tissue_asset[
            "visual_surface_vertex_source_class"
        ].astype(np.uint8)
        tissue_visual_density_zone = tissue_asset[
            "visual_surface_face_density_zone"
        ].astype(np.uint8)
        tissue_visual_face_ids = tissue_asset[
            "gaussian_visual_face_ids"
        ].astype(np.int32)
        tissue_binding_mode = str(tissue_asset["gaussian_binding_mode"].item())
        tissue_face_weights = tissue_asset[
            "gaussian_face_barycentric_weights"
        ].astype(np.float32)
        tissue_tet_weights = tissue_asset[
            "gaussian_barycentric_weights"
        ].astype(np.float32)
        tissue_rest_offsets = tissue_asset[
            "gaussian_rest_offset_table"
        ].astype(np.float32)
        tissue_visual_vertex_weights = tissue_asset[
            "visual_vertex_barycentric_weights"
        ].astype(np.float32)
        tissue_visual_vertex_offsets = tissue_asset[
            "visual_vertex_rest_offset_table"
        ].astype(np.float32)
    with np.load(
        PSM_RAW_PAPER_LND_DENSE_CONTACT_UNBOUNDED_XYZ_SURFACE_GAUSSIANS_PATH,
        allow_pickle=False,
    ) as psm_gaussians:
        psm_link_names = psm_gaussians["link_names"].tolist()
        psm_link_ids = psm_gaussians["link_ids"].astype(np.int64)
        psm_colors = psm_gaussians["colors"].astype(np.float32)
        psm_observed = psm_gaussians["color_directly_observed"].astype(bool)
        psm_black = psm_gaussians["color_unobserved_black"].astype(bool)
        psm_color_source = str(psm_gaussians["color_source"].item())
    psm_shaft_id = psm_link_names.index("PSM1_tool_main_link")
    psm_shaft_unobserved = (psm_link_ids == psm_shaft_id) & ~psm_observed
    contact_enable_result = set_psm_tissue_collisions(environment, True)
    contact_enabled_after_request = bool(
        environment.physics_settings.enable_triangle_skin_contacts
    )
    contact_metrics = simulator.triangle_skin_contact_metrics()
    if contact_metrics is None:
        raise RuntimeError("paper_soft has no triangle-skin contact metrics")
    simulator.set_triangle_skin_jaw_signal(0.12, 1.0)
    closing_grip_metrics = simulator.triangle_skin_contact_metrics()
    simulator.set_triangle_skin_jaw_signal(0.20, 2.0)
    opening_grip_metrics = simulator.triangle_skin_contact_metrics()
    if closing_grip_metrics is None or opening_grip_metrics is None:
        raise RuntimeError("paper_soft q7 grip metrics are unavailable")

    # A stiffness shadow rollout is only trustworthy when every stateful
    # component can be restored, not just Warp's particle/body state.  Mutate
    # one representative value from each mutable subsystem and verify that the
    # combined rollout snapshot restores it exactly.
    rollout_snapshot = simulator.clone_embodied_gaussian_rollout_state()
    projector = simulator.triangle_skin_contact_projector
    material_projector = simulator.material_projector
    if projector is None or material_projector is None:
        raise RuntimeError("paper_soft rollout auxiliaries are unavailable")
    with torch.no_grad():
        wp.to_torch(simulator.state_0.particle_q)[
            handle.particle_start, 0
        ] += 0.001
        simulator.gaussian_state.means[0, 0] += 0.001
        wp.to_torch(projector.persistent_grip_current_jaw_angle).fill_(
            -123.0
        )
        wp.to_torch(material_projector.paper_distance_stiffness).fill_(
            0.123
        )
        wp.to_torch(material_projector.paper_shape_stiffness).fill_(0.456)
        if simulator.kinematic_interpolation_start_q is not None:
            wp.to_torch(simulator.kinematic_interpolation_start_q).zero_()
        if simulator.kinematic_interpolation_target_q is not None:
            wp.to_torch(simulator.kinematic_interpolation_target_q).zero_()
    simulator.sim_time += 17.0
    projector.persistent_grip_previous_jaw_angle = -456.0
    projector.persistent_grip_jaw_motion_state = "mutated"
    simulator.copy_embodied_gaussian_rollout_state(rollout_snapshot)

    restored_auxiliary = simulator.clone_rollout_auxiliary_state()
    snapshot_auxiliary = rollout_snapshot.auxiliary_state
    restored_grip = restored_auxiliary.persistent_grip
    snapshot_grip = snapshot_auxiliary.persistent_grip
    rollout_snapshot_restores_all_state = bool(
        torch.equal(
            wp.to_torch(simulator.state_0.particle_q),
            wp.to_torch(
                rollout_snapshot.embodied_state.physics_state.particle_q
            ),
        )
        and torch.equal(
            simulator.gaussian_state.means,
            rollout_snapshot.embodied_state.gaussian_state.means,
        )
        and restored_auxiliary.sim_time == snapshot_auxiliary.sim_time
        and restored_grip is not None
        and snapshot_grip is not None
        and restored_grip.previous_jaw_angle
        == snapshot_grip.previous_jaw_angle
        and restored_grip.jaw_motion_state == snapshot_grip.jaw_motion_state
        and all(
            torch.equal(restored_grip.arrays[name], value)
            for name, value in snapshot_grip.arrays.items()
        )
        and torch.equal(
            restored_auxiliary.paper_distance_stiffness,
            snapshot_auxiliary.paper_distance_stiffness,
        )
        and torch.equal(
            restored_auxiliary.paper_shape_stiffness,
            snapshot_auxiliary.paper_shape_stiffness,
        )
        and (
            snapshot_auxiliary.kinematic_interpolation_start_q is None
            or torch.equal(
                restored_auxiliary.kinematic_interpolation_start_q,
                snapshot_auxiliary.kinematic_interpolation_start_q,
            )
        )
        and (
            snapshot_auxiliary.kinematic_interpolation_target_q is None
            or torch.equal(
                restored_auxiliary.kinematic_interpolation_target_q,
                snapshot_auxiliary.kinematic_interpolation_target_q,
            )
        )
    )
    contact_disable_result = set_psm_tissue_collisions(environment, False)

    rest_positions = wp.to_torch(model.particle_q).detach().clone()
    for _ in range(20):
        environment.step(compute_visual_forces=False)
    positions = wp.to_torch(simulator.state_0.particle_q).detach()
    velocities = wp.to_torch(simulator.state_0.particle_qd).detach()
    inverse_mass = wp.to_torch(model.particle_inv_mass).detach()
    tetrahedra = wp.to_torch(model.tet_indices).long()
    rest_inverse = wp.to_torch(model.tet_poses).detach()
    deformation = torch.stack(
        (
            positions[tetrahedra[:, 1]] - positions[tetrahedra[:, 0]],
            positions[tetrahedra[:, 2]] - positions[tetrahedra[:, 0]],
            positions[tetrahedra[:, 3]] - positions[tetrahedra[:, 0]],
        ),
        dim=2,
    )
    volume_ratio = torch.linalg.det(deformation @ rest_inverse)
    displacement = positions - rest_positions
    dynamic = inverse_mass > 0.0
    fixed = ~dynamic
    reference_equilibrium_smoke = {
        "steps": 20,
        "all_finite": bool(
            torch.isfinite(positions).all()
            and torch.isfinite(velocities).all()
            and torch.isfinite(volume_ratio).all()
        ),
        "minimum_tetrahedron_volume_ratio": float(volume_ratio.min().item()),
        "maximum_dynamic_displacement_m": float(
            torch.linalg.vector_norm(displacement[dynamic], dim=1).max().item()
        ),
        "maximum_anchor_drift_m": float(
            torch.linalg.vector_norm(displacement[fixed], dim=1).max().item()
        ),
    }

    gates = {
        "mode_is_paper_soft": environment.super_tissue_mode == "paper_soft",
        "paper_pbd_asset_selected": (
            environment.super_tissue_asset_path.resolve()
            == PAPER_PBD_TISSUE_PATH.resolve()
        ),
        "particle_count_matches_asset": (
            handle.particle_end - handle.particle_start == 4059
        ),
        "tetrahedron_count_matches_asset": (
            handle.tet_end - handle.tet_start == 15830
        ),
        "dense_multiview_soft_gaussians_loaded": bool(
            gaussian_model.num_soft_gaussians == expected_tissue_gaussians
            and expected_tissue_gaussians == 26754
        ),
        "tissue_gaussians_are_smaller_and_denser": bool(
            expected_tissue_gaussians >= 4 * 2863
            and np.median(tissue_scales) < 0.0005
            and tissue_scales.max() <= 0.000701
        ),
        "later_frames_fill_first_pair_occlusion": bool(
            tissue_first_pair_observed.sum() == 23845
            and tissue_later_only_observed.sum() == 1872
            and not np.any(
                tissue_first_pair_observed & tissue_later_only_observed
            )
        ),
        "stereo_union_visual_vertices_are_retained": bool(
            np.count_nonzero(tissue_visual_source_class == 1) == 1444
            and np.count_nonzero(tissue_visual_source_class == 2) == 544
            and np.count_nonzero(tissue_visual_source_class == 3) == 11390
            and np.count_nonzero(tissue_visual_source_class == 4) == 205
            and np.count_nonzero(tissue_visual_density_zone == 0) == 14815
            and np.count_nonzero(tissue_visual_density_zone == 1) == 7244
            and np.count_nonzero(tissue_visual_density_zone == 2) == 4695
        ),
        "visual_face_centroid_ellipsoid_binding_loaded": bool(
            tissue_binding_mode == "visual_surface_face_centroid"
            and tissue_face_weights.shape == (expected_tissue_gaussians, 3)
            and np.allclose(tissue_face_weights.sum(axis=1), 1.0)
            and np.all(np.count_nonzero(tissue_tet_weights > 0.0, axis=1) <= 3)
            and np.all(tissue_rest_offsets == 0.0)
            and np.array_equal(
                tissue_visual_face_ids, np.arange(expected_tissue_gaussians)
            )
            and np.allclose(
                tissue_rest_means,
                tissue_visual_vertices[tissue_visual_faces].mean(axis=1),
                atol=2.0e-8,
            )
            and np.allclose(tissue_visual_vertex_weights.sum(axis=1), 1.0)
            and np.isfinite(tissue_visual_vertex_offsets).all()
            and torch.all(gaussian_model.soft_gaussian_binding_modes == 2)
        ),
        "dense_full_psm_gaussians_loaded": bool(
            environment.super_psm_gaussian_count == 76798
            and environment.super_psm_surface_gaussians_path.resolve()
            == (
                PSM_RAW_PAPER_LND_DENSE_CONTACT_UNBOUNDED_XYZ_SURFACE_GAUSSIANS_PATH.resolve()
            )
        ),
        "psm_uses_first_stereo_real_rgb": bool(
            psm_color_source == "first_confirmed_stereo_pair_real_rgb"
            and psm_observed.sum() == 3025
            and np.isfinite(psm_colors).all()
        ),
        "unseen_long_shaft_gaussians_are_black": bool(
            psm_shaft_unobserved.sum() == 70742
            and np.array_equal(psm_black, psm_shaft_unobserved)
            and np.all(psm_colors[psm_shaft_unobserved] == 0.0)
        ),
        "gaussian_four_slot_storage_preserved_for_runtime_compatibility": (
            tuple(gaussian_model.soft_gaussian_particle_indices.shape)
            == (expected_tissue_gaussians, 4)
        ),
        "only_soft_gaussians_receive_visual_gradients": bool(
            torch.equal(gradient_enabled, expected_gradient_enabled)
        ),
        "soft_visual_force_enabled": bool(
            environment.visual_forces_settings.enable_soft_particle_forces
        ),
        "soft_visual_force_has_mass_aware_acceleration_cap": bool(
            environment.visual_forces_settings.soft_max_particle_acceleration
            == SUPER_SOFT_VISUAL_FORCE_MAX_PARTICLE_ACCELERATION_M_S2
            and SUPER_SOFT_VISUAL_FORCE_MAX_PARTICLE_ACCELERATION_M_S2 > 0.0
        ),
        "visual_force_has_positive_iteration_default": (
            environment.visual_forces_settings.iterations > 0
        ),
        "triangle_skin_projector_configured": (
            simulator.triangle_skin_contact_projector is not None
        ),
        "shadow_rollout_snapshot_restores_all_state": (
            rollout_snapshot_restores_all_state
        ),
        "collision_skin_registered": (
            len(simulator.builder.soft_collision_skin_faces) > 0
        ),
        "triangle_skin_contact_can_enable": (
            contact_enable_result and contact_enabled_after_request
        ),
        "only_two_jaws_generate_contact": bool(
            len(contact_metrics["jaw_contact_shape_ids"]) == 2
            and len(contact_metrics["samples_per_shape"]) == 2
            and set(contact_metrics["top_barrier_shape_ids"])
            == set(contact_metrics["jaw_contact_shape_ids"])
            and contact_metrics["top_barrier_face_count"] > 0
        ),
        "jaw_friction_and_persistent_grip_configured": bool(
            contact_metrics["jaw_friction_coefficient"] == 1.5
            and contact_metrics[
                "persistent_grip_constraint_enabled"
            ]
        ),
        "bilateral_four_point_jaw_frame_u_t_is_configured": bool(
            contact_metrics[
                "persistent_grip_maximum_capture_penetration_m"
            ]
            == PAPER_SOFT_TISSUE_GRIP_MAX_CAPTURE_PENETRATION_M
            and contact_metrics["persistent_grip_support_radius_m"]
            == PAPER_SOFT_TISSUE_GRIP_SUPPORT_RADIUS_M
            and contact_metrics["persistent_grip_relaxation"]
            == PAPER_SOFT_TISSUE_GRIP_RELAXATION
            and contact_metrics[
                "persistent_grip_maximum_correction_m"
            ]
            == PAPER_SOFT_TISSUE_GRIP_MAX_CORRECTION_M
            and contact_metrics[
                "persistent_grip_support_candidate_count"
            ]
            > 0
            and contact_metrics[
                "persistent_grip_minimum_contact_samples_per_jaw"
            ]
            == PAPER_SOFT_TISSUE_GRIP_MIN_CONTACT_SAMPLES_PER_JAW
            and contact_metrics[
                "persistent_grip_nearest_surface_particles"
            ]
            == PAPER_SOFT_TISSUE_GRIP_NEAREST_SURFACE_PARTICLES
            and contact_metrics[
                "persistent_grip_maximum_jaw_patch_separation_m"
            ]
            == PAPER_SOFT_TISSUE_GRIP_MAX_PATCH_SEPARATION_M
            and contact_metrics["persistent_grip_activation_steps"]
            == PAPER_SOFT_TISSUE_GRIP_ACTIVATION_STEPS
            and contact_metrics[
                "persistent_grip_compliance_m_per_n"
            ]
            == PAPER_SOFT_TISSUE_GRIP_COMPLIANCE_M_PER_N
            and contact_metrics["persistent_grip_support_generations"]
            == PAPER_SOFT_TISSUE_GRIP_SUPPORT_GENERATIONS
        ),
        "observed_equilibrium_gravity_is_compensated": bool(
            environment.super_tissue_gravity_m_s2
            == PAPER_SOFT_TISSUE_GRAVITY_M_S2
            and PAPER_SOFT_TISSUE_GRAVITY_M_S2 == 0.0
        ),
        "reference_equilibrium_stays_finite_and_inversion_free": bool(
            reference_equilibrium_smoke["all_finite"]
            and reference_equilibrium_smoke[
                "minimum_tetrahedron_volume_ratio"
            ] > 0.999
            and reference_equilibrium_smoke[
                "maximum_dynamic_displacement_m"
            ] <= 1.0e-7
            and reference_equilibrium_smoke[
                "maximum_anchor_drift_m"
            ] <= 1.0e-7
        ),
        "q7_closing_allows_grip_capture": bool(
            closing_grip_metrics["persistent_grip_capture_allowed"]
            and closing_grip_metrics["persistent_grip_q7_motion_state"]
            == "closing"
        ),
        "q7_opening_requests_immediate_release": bool(
            opening_grip_metrics["persistent_grip_release_requested"]
            and opening_grip_metrics["persistent_grip_q7_motion_state"]
            == "opening"
            and opening_grip_metrics[
                "persistent_grip_release_angle_delta_rad"
            ]
            == 0.08
        ),
        "press_contact_has_local_tetrahedral_spread": (
            contact_metrics["contact_spread_layers"]
            == PAPER_SOFT_TISSUE_CONTACT_SPREAD_LAYERS
        ),
        "jaw_top_plane_has_no_synthetic_support": bool(
            contact_metrics["top_support_entry_count"] == 0
            and contact_metrics["top_support_weight_scale"] == 0.0
            and contact_metrics["top_barrier_clearance_m"]
            == PAPER_SOFT_TISSUE_TOP_CLEARANCE_M
            and contact_metrics["contact_spread_layers"]
            == PAPER_SOFT_TISSUE_CONTACT_SPREAD_LAYERS
        ),
        "direct_press_core_is_2p5mm_and_velocity_setting_is_applied": bool(
            PAPER_SOFT_TISSUE_TOP_BARRIER_CONTACT_PATCH_RADIUS_M
            == 0.0025
            and contact_metrics["top_barrier_contact_patch_radius_m"]
            == PAPER_SOFT_TISSUE_TOP_BARRIER_CONTACT_PATCH_RADIUS_M
            and PAPER_SOFT_TISSUE_MATERIAL_PROJECTION_VELOCITY_SCALE
            == 0.14
            and environment.physics_settings
            .material_projection_velocity_scale
            == PAPER_SOFT_TISSUE_MATERIAL_PROJECTION_VELOCITY_SCALE
        ),
        "jaw_surface_correction_limit_is_fixed": bool(
            environment.physics_settings
            .triangle_skin_contact_max_correction_m
            == PAPER_SOFT_TISSUE_SURFACE_MAX_CORRECTION_M
            and environment.physics_settings
            .triangle_skin_top_barrier_max_correction_m
            == PAPER_SOFT_TISSUE_TOP_BARRIER_MAX_CORRECTION_M
            and contact_metrics["top_barrier_distal_length_m"]
            == PAPER_SOFT_TISSUE_TOP_BARRIER_DISTAL_LENGTH_M
            and contact_metrics["top_barrier_tip_allowance_m"]
            == PAPER_SOFT_TISSUE_TOP_BARRIER_TIP_ALLOWANCE_M
            and PAPER_SOFT_TISSUE_TOP_BARRIER_MAX_CORRECTION_M
            < PAPER_SOFT_TISSUE_SURFACE_MAX_CORRECTION_M
            and PAPER_SOFT_TISSUE_TOP_BARRIER_TIP_ALLOWANCE_M
            < PAPER_SOFT_TISSUE_TOP_BARRIER_DISTAL_LENGTH_M
            < contact_metrics["jaw_contact_distal_length_m"]
            and environment.physics_settings
            .triangle_skin_contact_iterations
            == PAPER_SOFT_TISSUE_SURFACE_ITERATIONS
            and environment.physics_settings
            .triangle_skin_contact_substep_stride
            == PAPER_SOFT_TISSUE_CONTACT_SUBSTEP_STRIDE
        ),
        "jaw_query_is_decoupled_from_contact_margin": bool(
            environment.physics_settings.triangle_skin_query_distance_m
            == ADAPTIVE_TISSUE_CONTACT_QUERY_DISTANCE_M
            and (
                environment.physics_settings.triangle_skin_query_distance_m
                > environment.physics_settings.triangle_skin_contact_margin_m
            )
        ),
        "particle_shape_contact_disabled": (
            not environment.physics_settings.enable_particle_shape_contacts
        ),
        "particle_particle_contact_disabled": (
            not environment.physics_settings.enable_particle_particle_contacts
        ),
        "triangle_skin_contact_can_disable": (
            contact_disable_result is False
            and not environment.physics_settings.enable_triangle_skin_contacts
        ),
        "depth_residual_mapping_disabled": (
            not environment.super_tissue_residual_mapping_enabled
        ),
        "stiffness_optimization_disabled": (
            not environment.super_tissue_stiffness_optimization_enabled
        ),
        "material_parameters_are_fixed": bool(
            environment.super_tissue_fixed_material_parameters
        ),
        "connected_soft_material_solver_is_fixed": bool(
            environment.physics_settings.material_iterations
            == PAPER_SOFT_TISSUE_MATERIAL_ITERATIONS
            and environment.physics_settings.material_relaxation
            == PAPER_SOFT_TISSUE_MATERIAL_RELAXATION
            and (
                environment.physics_settings
                .particle_velocity_damping_per_second
            )
            == PAPER_SOFT_TISSUE_VELOCITY_DAMPING_PER_SECOND
        ),
        "uniform_fixed_mu": bool(
            torch.allclose(
                tet_materials[:, 0],
                torch.full_like(tet_materials[:, 0], mu_expected),
                rtol=1.0e-6,
                atol=1.0e-6,
            )
        ),
        "uniform_fixed_lambda": bool(
            torch.allclose(
                tet_materials[:, 1],
                torch.full_like(tet_materials[:, 1], lambda_expected),
                rtol=1.0e-6,
                atol=1.0e-6,
            )
        ),
    }
    report = {
        "mode": environment.super_tissue_mode,
        "device": str(model.device),
        "particles": int(handle.particle_end - handle.particle_start),
        "tetrahedra": int(handle.tet_end - handle.tet_start),
        "soft_gaussians": int(gaussian_model.num_soft_gaussians),
        "psm_gaussians": int(environment.super_psm_gaussian_count),
        "young_modulus_pa": float(environment.super_tissue_young_modulus_pa),
        "poisson_ratio": float(environment.super_tissue_poisson_ratio),
        "material_iterations": int(
            environment.physics_settings.material_iterations
        ),
        "material_relaxation": float(
            environment.physics_settings.material_relaxation
        ),
        "velocity_damping_per_second": float(
            environment.physics_settings.particle_velocity_damping_per_second
        ),
        "contact_margin_m": float(
            environment.physics_settings.triangle_skin_contact_margin_m
        ),
        "jaw_query_distance_m": float(
            environment.physics_settings.triangle_skin_query_distance_m
        ),
        "visual_force_iterations_default": int(
            environment.visual_forces_settings.iterations
        ),
        "reference_equilibrium_smoke": reference_equilibrium_smoke,
        "gates": gates,
        "passed": all(gates.values()),
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("paper_soft gates failed")


if __name__ == "__main__":
    main()
