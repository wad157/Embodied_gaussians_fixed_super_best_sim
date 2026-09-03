#!/usr/bin/env python3
"""Profile the dense paper-soft scene and replay the contact/lift trajectory.

This diagnostic deliberately excludes image residual/visual-force optimization.
It reports physics, Gaussian skinning, persistent-grip capture, the global
no-inversion fallback, and final tetrahedron volume ratios separately.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from embodied_environments.super_embodied.super_embodied import (  # noqa: E402
    PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_UNBOUNDED_XYZ_POSE_DRIVER_PATH,
    apply_psm_lnd_pose,
    build_environment,
    set_psm_tissue_collisions,
)
from embodied_gaussians import DatasetManager  # noqa: E402
from embodied_gaussians.embodied_simulator.visual_force_masks import (  # noqa: E402
    MultiCameraPackedTissueVisualForceWeights,
)


LEFT_MASK_DIR = REPO_ROOT / "data/super/grasp5_native/visual_force_masks_v1"
RIGHT_MASK_DIR = REPO_ROOT / "data/super/grasp5_native/visual_force_masks_right_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument(
        "--tissue-mode",
        default="paper_soft",
        choices=("paper_pbd", "paper_soft"),
    )
    parser.add_argument(
        "--world-translation-mm",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
    )
    parser.add_argument("--start-state", type=int, default=1807)
    parser.add_argument("--end-state", type=int, default=2500)
    parser.add_argument("--state-stride", type=int, default=10)
    parser.add_argument("--baseline-steps", type=int, default=5)
    parser.add_argument("--grip-max-correction-mm", type=float, default=None)
    parser.add_argument("--grip-compliance-m-per-n", type=float, default=None)
    parser.add_argument(
        "--grip-max-capture-penetration-mm", type=float, default=None
    )
    parser.add_argument(
        "--grip-min-capture-volume-ratio", type=float, default=None
    )
    parser.add_argument("--surface-max-correction-mm", type=float, default=None)
    parser.add_argument(
        "--top-barrier-max-correction-mm", type=float, default=None
    )
    parser.add_argument("--jaw-friction-coefficient", type=float, default=None)
    parser.add_argument("--material-iterations", type=int, default=None)
    parser.add_argument(
        "--disable-jaw-tangential-contact", action="store_true"
    )
    parser.add_argument("--disable-top-barrier", action="store_true")
    parser.add_argument("--include-trace", action="store_true")
    parser.add_argument("--young-modulus-pa", type=float, default=None)
    parser.add_argument("--poisson-ratio", type=float, default=None)
    parser.add_argument("--contact-substep-stride", type=int, default=None)
    parser.add_argument("--profile-stereo-visual-force", action="store_true")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "data/super/grasp5_offline_demo",
    )
    return parser.parse_args()


def synchronized_seconds(function) -> float:
    wp.synchronize()
    start = time.perf_counter()
    function()
    wp.synchronize()
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()
    wp.config.kernel_cache_dir = "/tmp/warp-super-contact-offset-diagnostic"
    wp.init()
    environment = build_environment(
        num_envs=1,
        add_gaussians=True,
        device=args.device,
        psm_pose_driver_path=(
            PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_UNBOUNDED_XYZ_POSE_DRIVER_PATH
        ),
        psm_visual_tip_only=False,
        tissue_mode=args.tissue_mode,
    )
    simulator = environment.sim
    model = simulator.model
    settings = environment.physics_settings
    if args.grip_max_correction_mm is not None:
        simulator.triangle_skin_contact_projector.persistent_grip_maximum_correction_m = (
            float(args.grip_max_correction_mm) / 1000.0
        )
    if args.grip_compliance_m_per_n is not None:
        simulator.triangle_skin_contact_projector.persistent_grip_compliance_m_per_n = (
            float(args.grip_compliance_m_per_n)
        )
    if args.grip_max_capture_penetration_mm is not None:
        simulator.triangle_skin_contact_projector.persistent_grip_maximum_capture_penetration_m = (
            float(args.grip_max_capture_penetration_mm) / 1000.0
        )
    if args.grip_min_capture_volume_ratio is not None:
        simulator.triangle_skin_contact_projector.persistent_grip_minimum_capture_volume_ratio = (
            float(args.grip_min_capture_volume_ratio)
        )
    if args.surface_max_correction_mm is not None:
        settings.triangle_skin_contact_max_correction_m = (
            float(args.surface_max_correction_mm) / 1000.0
        )
    if args.top_barrier_max_correction_mm is not None:
        settings.triangle_skin_top_barrier_max_correction_m = (
            float(args.top_barrier_max_correction_mm) / 1000.0
        )
    contact_projector = simulator.triangle_skin_contact_projector
    if args.jaw_friction_coefficient is not None:
        contact_projector.jaw_friction_coefficient = float(
            args.jaw_friction_coefficient
        )
    if args.material_iterations is not None:
        if args.material_iterations < 0:
            raise ValueError("--material-iterations must be non-negative")
        settings.material_iterations = int(args.material_iterations)
    if args.disable_jaw_tangential_contact:
        contact_projector.jaw_surface_sample_enabled.zero_()
    if args.disable_top_barrier:
        contact_projector.top_barrier_sample_enabled.zero_()
    if args.young_modulus_pa is not None or args.poisson_ratio is not None:
        current_materials = wp.to_torch(model.tet_materials)
        young = 25.0 if args.young_modulus_pa is None else float(args.young_modulus_pa)
        poisson = 0.0 if args.poisson_ratio is None else float(args.poisson_ratio)
        mu = young / (2.0 * (1.0 + poisson))
        lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        current_materials[:, 0] = mu
        current_materials[:, 1] = lam
    if args.contact_substep_stride is not None:
        settings.triangle_skin_contact_substep_stride = int(
            args.contact_substep_stride
        )
    offset_world = np.asarray(args.world_translation_mm, dtype=np.float64) / 1000.0

    rest_positions = wp.to_torch(model.particle_q).detach().clone()
    dynamic_mask = wp.to_torch(model.particle_inv_mass).detach() > 0.0

    # Compile once, then measure the actual no-contact mechanics and Gaussian
    # skinning paths independently. The rest pose is equilibrium because this
    # baseline intentionally compensates gravity.
    synchronized_seconds(lambda: simulator.physics_step(settings))
    baseline_physics_s = [
        synchronized_seconds(lambda: simulator.physics_step(settings))
        for _ in range(args.baseline_steps)
    ]
    baseline_skinning_s = [
        synchronized_seconds(simulator.update_gaussian_transforms)
        for _ in range(args.baseline_steps)
    ]

    start_state = max(0, min(args.start_state, len(environment.super_psm_lnd_timestamps) - 1))
    end_state = max(start_state, min(args.end_state, len(environment.super_psm_lnd_timestamps) - 1))
    apply_psm_lnd_pose(
        environment,
        start_state,
        translation_offset=offset_world,
        update_gaussians=False,
    )
    simulator.sync_kinematic_body_interpolation()
    set_psm_tissue_collisions(environment, True)
    initial_metrics = simulator.triangle_skin_contact_metrics()

    state_indices = [start_state]
    state_indices.extend(
        range(start_state + args.state_stride, end_state + 1, args.state_stride)
    )
    if state_indices[-1] != end_state:
        state_indices.append(end_state)

    contact_physics_s: list[float] = []
    skinning_s: list[float] = []
    zero_safety_states: list[int] = []
    minimum_safety_scale = 1.0
    maximum_unsafe_tetrahedra = 0
    minimum_grip_support_scale = 1.0
    minimum_grip_direct_scale = 1.0
    minimum_grip_jaw_scales = [1.0, 1.0]
    minimum_grip_support_scale_state: int | None = None
    minimum_grip_direct_scale_state: int | None = None
    minimum_grip_jaw_scale_states: list[int | None] = [None, None]
    maximum_particle_speed_m_s = 0.0
    maximum_particle_speed_state: int | None = None
    maximum_particle_speed_id: int | None = None
    maximum_direct_particle_speed_m_s = 0.0
    maximum_direct_particle_speed_state: int | None = None
    capture_state: int | None = None
    capture_metrics: dict | None = None
    direct_particle_ids: np.ndarray | None = None
    direct_positions_at_capture: torch.Tensor | None = None
    selected_incident_volume_ratios_at_capture: dict[int, dict] | None = None
    grip_proposal_audit_at_capture: dict | None = None
    direct_local_at_capture: np.ndarray | None = None
    direct_body_ids_at_capture: np.ndarray | None = None
    virtual_frame_position_at_capture: np.ndarray | None = None
    trace: list[dict] = []
    for state_index in state_indices:
        apply_psm_lnd_pose(
            environment,
            state_index,
            translation_offset=offset_world,
            update_gaussians=False,
        )
        contact_physics_s.append(
            synchronized_seconds(lambda: simulator.physics_step(settings))
        )
        skinning_s.append(
            synchronized_seconds(simulator.update_gaussian_transforms)
        )
        metrics = simulator.triangle_skin_contact_metrics()
        assert metrics is not None
        velocities = wp.to_torch(simulator.state_0.particle_qd).detach()
        speeds = torch.linalg.vector_norm(velocities, dim=1)
        dynamic_speeds = torch.where(
            dynamic_mask,
            speeds,
            torch.full_like(speeds, -1.0),
        )
        frame_speed, frame_speed_id = torch.max(dynamic_speeds, dim=0)
        if float(frame_speed.item()) > maximum_particle_speed_m_s:
            maximum_particle_speed_m_s = float(frame_speed.item())
            maximum_particle_speed_state = state_index
            maximum_particle_speed_id = int(frame_speed_id.item())
        direct_mask = (
            wp.to_torch(
                simulator.triangle_skin_contact_projector
                .persistent_grip_particle_direct
            ).detach()
            != 0
        )
        if torch.any(direct_mask):
            frame_direct_speed = float(speeds[direct_mask].max().item())
            if frame_direct_speed > maximum_direct_particle_speed_m_s:
                maximum_direct_particle_speed_m_s = frame_direct_speed
                maximum_direct_particle_speed_state = state_index
        safety_scale = float(metrics["material_safety_step_scale"])
        unsafe = int(metrics["material_safety_unsafe_tetrahedra"])
        grip_support_scale = float(
            metrics["persistent_grip_support_safe_scale"]
        )
        grip_direct_scale = float(
            metrics["persistent_grip_direct_safe_scale"]
        )
        grip_jaw_scales = tuple(
            float(value)
            for value in metrics["persistent_grip_jaw_safe_scales"]
        )
        if grip_support_scale < minimum_grip_support_scale:
            minimum_grip_support_scale = grip_support_scale
            minimum_grip_support_scale_state = state_index
        if grip_direct_scale < minimum_grip_direct_scale:
            minimum_grip_direct_scale = grip_direct_scale
            minimum_grip_direct_scale_state = state_index
        for jaw_index, jaw_scale in enumerate(grip_jaw_scales):
            if jaw_scale < minimum_grip_jaw_scales[jaw_index]:
                minimum_grip_jaw_scales[jaw_index] = jaw_scale
                minimum_grip_jaw_scale_states[jaw_index] = state_index
        minimum_safety_scale = min(minimum_safety_scale, safety_scale)
        maximum_unsafe_tetrahedra = max(maximum_unsafe_tetrahedra, unsafe)
        if safety_scale == 0.0:
            zero_safety_states.append(state_index)
        if bool(metrics["persistent_grip_active"]) and capture_state is None:
            capture_state = state_index
            capture_metrics = metrics.copy()
            direct_particle_ids = np.flatnonzero(
                simulator.triangle_skin_contact_projector
                .persistent_grip_particle_direct.numpy()
                != 0
            )
            direct_positions_at_capture = (
                wp.to_torch(simulator.state_0.particle_q)
                .detach()[
                    torch.as_tensor(
                        direct_particle_ids,
                        device=wp.to_torch(simulator.state_0.particle_q).device,
                    )
                ]
                .clone()
            )
            capture_positions_all = wp.to_torch(
                simulator.state_0.particle_q
            ).detach()
            tetrahedra_at_capture = wp.to_torch(model.tet_indices).long()
            rest_inverse_at_capture = wp.to_torch(model.tet_poses).detach()
            capture_deformation = torch.stack(
                (
                    capture_positions_all[tetrahedra_at_capture[:, 1]]
                    - capture_positions_all[tetrahedra_at_capture[:, 0]],
                    capture_positions_all[tetrahedra_at_capture[:, 2]]
                    - capture_positions_all[tetrahedra_at_capture[:, 0]],
                    capture_positions_all[tetrahedra_at_capture[:, 3]]
                    - capture_positions_all[tetrahedra_at_capture[:, 0]],
                ),
                dim=2,
            )
            capture_volume_ratios = torch.linalg.det(
                capture_deformation @ rest_inverse_at_capture
            )
            projector = simulator.triangle_skin_contact_projector
            if projector.persistent_grip_transfer_layers == 0:
                proposal_delta_array = projector.persistent_grip_deltas
                proposal_weight_array = projector.persistent_grip_delta_counts
            elif projector.persistent_grip_transfer_layers % 2 == 1:
                proposal_delta_array = projector.persistent_grip_spread_deltas_a
                proposal_weight_array = projector.persistent_grip_spread_weights_a
            else:
                proposal_delta_array = projector.persistent_grip_spread_deltas_b
                proposal_weight_array = projector.persistent_grip_spread_weights_b
            proposal_deltas = wp.to_torch(proposal_delta_array).detach()
            proposal_weights = wp.to_torch(proposal_weight_array).detach()
            proposed_positions = capture_positions_all + proposal_deltas
            proposed_deformation = torch.stack(
                (
                    proposed_positions[tetrahedra_at_capture[:, 1]]
                    - proposed_positions[tetrahedra_at_capture[:, 0]],
                    proposed_positions[tetrahedra_at_capture[:, 2]]
                    - proposed_positions[tetrahedra_at_capture[:, 0]],
                    proposed_positions[tetrahedra_at_capture[:, 3]]
                    - proposed_positions[tetrahedra_at_capture[:, 0]],
                ),
                dim=2,
            )
            proposed_volume_ratios = torch.linalg.det(
                proposed_deformation @ rest_inverse_at_capture
            )
            affected_tets = torch.any(
                proposal_weights[tetrahedra_at_capture] > 0.0, dim=1
            )
            affected_ids = torch.nonzero(
                affected_tets, as_tuple=False
            ).flatten()
            affected_current = capture_volume_ratios[affected_tets]
            affected_proposed = proposed_volume_ratios[affected_tets]
            finite = torch.isfinite(affected_proposed)
            proposed_order = torch.argsort(
                torch.nan_to_num(
                    affected_proposed, nan=-float("inf")
                )
            )[:8]
            grip_proposal_audit_at_capture = {
                "affected_tetrahedra": int(affected_ids.numel()),
                "affected_particles": int(
                    torch.count_nonzero(proposal_weights > 0.0).item()
                ),
                "maximum_proposed_delta_mm": 1.0e3
                * float(
                    torch.linalg.vector_norm(proposal_deltas, dim=1)
                    .max()
                    .item()
                ),
                "current_ratio_minimum": float(
                    affected_current.min().item()
                ),
                "proposed_ratio_minimum": float(
                    affected_proposed[finite].min().item()
                ) if torch.any(finite) else None,
                "current_nonpositive": int(
                    torch.count_nonzero(affected_current <= 0.0).item()
                ),
                "proposed_nonpositive": int(
                    torch.count_nonzero(affected_proposed <= 0.0).item()
                ),
                "proposed_nonfinite": int(
                    torch.count_nonzero(~finite).item()
                ),
                "worst_proposals": [
                    {
                        "tetrahedron_id": int(
                            affected_ids[local_id].item()
                        ),
                        "current_ratio": float(
                            affected_current[local_id].item()
                        ),
                        "proposed_ratio": float(
                            affected_proposed[local_id].item()
                        ),
                    }
                    for local_id in proposed_order
                ],
            }
            selected_incident_volume_ratios_at_capture = {}
            for particle_id in direct_particle_ids:
                incident = torch.any(
                    tetrahedra_at_capture == int(particle_id), dim=1
                )
                values = capture_volume_ratios[incident]
                selected_incident_volume_ratios_at_capture[int(particle_id)] = {
                    "minimum": float(values.min().item()),
                    "median": float(values.median().item()),
                    "maximum": float(values.max().item()),
                }
            direct_local_at_capture = (
                simulator.triangle_skin_contact_projector
                .persistent_grip_particle_local.numpy()[direct_particle_ids]
                .copy()
            )
            direct_body_ids_at_capture = (
                simulator.triangle_skin_contact_projector
                .persistent_grip_particle_body.numpy()[direct_particle_ids]
                .copy()
            )
            virtual_body_id = (
                simulator.triangle_skin_contact_projector
                .persistent_grip_virtual_body_id
            )
            body_transforms = simulator.state_0.body_q.numpy()
            virtual_frame_position_at_capture = body_transforms[
                virtual_body_id, :3
            ].copy()
        trace.append(
            {
                "state": state_index,
                "q7_rad": float(metrics["persistent_grip_q7_angle_rad"]),
                "q7_motion": metrics["persistent_grip_q7_motion_state"],
                "capture_allowed": bool(metrics["persistent_grip_capture_allowed"]),
                "grip_active": bool(metrics["persistent_grip_active"]),
                "direct_particles": int(metrics["persistent_grip_direct_particle_count"]),
                "selected_particle_ids": list(
                    metrics["persistent_grip_selected_particle_ids"]
                ),
                "activation_counter": int(
                    metrics["persistent_grip_activation_counter"]
                ),
                "jaw_patch_separation_mm": 1.0e3
                * float(metrics["persistent_grip_jaw_patch_separation_m"]),
                "support_particles": int(metrics["persistent_grip_support_particle_count"]),
                "contacts": int(metrics["contact_count"]),
                "contact_count_by_shape": metrics["contact_count_by_shape"],
                "maximum_penetration_mm": 1.0e3 * float(metrics["maximum_penetration_m"]),
                "minimum_signed_distance_mm": 1.0e3 * float(metrics["minimum_signed_distance_m"]),
                "material_safety_scale": safety_scale,
                "unsafe_tetrahedra": unsafe,
                "grip_support_safe_scale": grip_support_scale,
                "grip_direct_safe_scale": grip_direct_scale,
                "grip_jaw_safe_scales": list(grip_jaw_scales),
                "maximum_particle_speed_m_s": float(frame_speed.item()),
                "maximum_particle_speed_id": int(frame_speed_id.item()),
                "maximum_direct_particle_speed_m_s": (
                    float(speeds[direct_mask].max().item())
                    if torch.any(direct_mask)
                    else 0.0
                ),
            }
        )

    positions = wp.to_torch(simulator.state_0.particle_q).detach()
    displacement = positions - rest_positions
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
    volume_ratios = torch.linalg.det(deformation @ rest_inverse)
    dynamic_z_displacement = displacement[dynamic_mask, 2]
    final_contact_metrics = simulator.triangle_skin_contact_metrics()
    press_roi_z_mm = None
    left_shoulder_z_mm = None
    if final_contact_metrics is not None:
        grasp_center = torch.as_tensor(
            final_contact_metrics["persistent_grip_grasp_center_m"],
            device=positions.device,
            dtype=positions.dtype,
        )
        press_roi_mask = dynamic_mask & (
            torch.linalg.vector_norm(
                rest_positions[:, :2] - grasp_center[:2], dim=1
            )
            <= 0.003
        )
        press_roi_z = displacement[press_roi_mask, 2]
        if press_roi_z.numel() > 0:
            press_roi_z_mm = {
                "particle_count": int(press_roi_z.numel()),
                "minimum": 1.0e3 * float(press_roi_z.min().item()),
                "median": 1.0e3 * float(press_roi_z.median().item()),
                "maximum": 1.0e3 * float(press_roi_z.max().item()),
                "depressed_beyond_0p1mm": int(
                    torch.count_nonzero(press_roi_z <= -1.0e-4).item()
                ),
            }
        # Report the actual stereo-left side of the pressure footprint rather
        # than inferring "left" from an arbitrary world x/y axis.
        diagnostic_dataset = DatasetManager(args.dataset, load_frames=False)
        stereo_left = next(
            camera
            for camera in diagnostic_dataset.cameras
            if camera.name == "stereo_left"
        )
        world_to_camera = torch.as_tensor(
            np.linalg.inv(stereo_left.X_WC),
            device=positions.device,
            dtype=positions.dtype,
        )
        rest_homogeneous = torch.cat(
            (
                rest_positions,
                torch.ones(
                    (len(rest_positions), 1),
                    device=positions.device,
                    dtype=positions.dtype,
                ),
            ),
            dim=1,
        )
        center_homogeneous = torch.cat(
            (
                grasp_center,
                torch.ones(1, device=positions.device, dtype=positions.dtype),
            )
        )
        rest_camera = rest_homogeneous @ world_to_camera.T
        center_camera = world_to_camera @ center_homogeneous
        radial_distance = torch.linalg.vector_norm(
            rest_positions[:, :2] - grasp_center[:2], dim=1
        )
        top_mask = torch.zeros_like(dynamic_mask)
        top_ids = torch.as_tensor(
            simulator.triangle_skin_contact_projector.top_nodes.numpy(),
            device=positions.device,
            dtype=torch.long,
        )
        top_mask[top_ids] = True
        left_shoulder_mask = (
            dynamic_mask
            & top_mask
            & (radial_distance >= 0.0015)
            & (radial_distance <= 0.0050)
            & (rest_camera[:, 0] < center_camera[0])
        )
        left_shoulder_z = displacement[left_shoulder_mask, 2]
        if left_shoulder_z.numel() > 0:
            left_shoulder_z_mm = {
                "particle_count": int(left_shoulder_z.numel()),
                "minimum": 1.0e3 * float(left_shoulder_z.min().item()),
                "median": 1.0e3 * float(left_shoulder_z.median().item()),
                "maximum": 1.0e3 * float(left_shoulder_z.max().item()),
                "raised_beyond_0p1mm": int(
                    torch.count_nonzero(left_shoulder_z >= 1.0e-4).item()
                ),
                "depressed_beyond_0p1mm": int(
                    torch.count_nonzero(left_shoulder_z <= -1.0e-4).item()
                ),
            }
    direct_z_mm = None
    direct_since_capture_mm = None
    virtual_frame_motion_since_capture_mm = None
    direct_target_motion_since_capture_mm = None
    direct_target_tracking_error_mm = None
    if direct_particle_ids is not None and len(direct_particle_ids):
        values = displacement[torch.as_tensor(direct_particle_ids, device=positions.device), 2]
        direct_z_mm = {
            "minimum": 1.0e3 * float(values.min().item()),
            "median": 1.0e3 * float(values.median().item()),
            "maximum": 1.0e3 * float(values.max().item()),
        }
        if direct_positions_at_capture is not None:
            direct_since_capture = (
                positions[
                    torch.as_tensor(
                        direct_particle_ids, device=positions.device
                    )
                ]
                - direct_positions_at_capture
            )
            direct_since_capture_mm = {
                "minimum_xyz": (
                    1.0e3 * direct_since_capture.min(dim=0).values
                ).tolist(),
                "median_xyz": (
                    1.0e3 * direct_since_capture.median(dim=0).values
                ).tolist(),
                "maximum_xyz": (
                    1.0e3 * direct_since_capture.max(dim=0).values
                ).tolist(),
                "median_distance": 1.0e3
                * float(
                    torch.linalg.vector_norm(
                        direct_since_capture, dim=1
                    ).median().item()
                ),
            }
        if (
            direct_local_at_capture is not None
            and direct_body_ids_at_capture is not None
            and direct_positions_at_capture is not None
        ):
            virtual_body_id = (
                simulator.triangle_skin_contact_projector
                .persistent_grip_virtual_body_id
            )
            body_transforms = simulator.state_0.body_q.numpy()
            final_virtual_transform = body_transforms[virtual_body_id]
            virtual_frame_motion_since_capture_mm = (
                1.0e3
                * (
                    final_virtual_transform[:3]
                    - virtual_frame_position_at_capture
                )
            ).tolist()
            direct_targets = np.stack(
                [
                    body_transforms[body_id, :3]
                    + Rotation.from_quat(
                        body_transforms[body_id, 3:7]
                    ).apply(local)
                    for body_id, local in zip(
                        direct_body_ids_at_capture,
                        direct_local_at_capture,
                        strict=True,
                    )
                ],
                axis=0,
            )
            capture_positions_np = direct_positions_at_capture.cpu().numpy()
            final_positions_np = positions[
                torch.as_tensor(direct_particle_ids, device=positions.device)
            ].cpu().numpy()
            direct_target_motion_since_capture_mm = (
                1.0e3 * (direct_targets - capture_positions_np)
            ).tolist()
            direct_target_tracking_error_mm = (
                1.0e3 * (direct_targets - final_positions_np)
            ).tolist()

    stereo_visual_force_ms = None
    if args.profile_stereo_visual_force:
        dataset = DatasetManager(args.dataset)
        dataset.keep_only_cameras(["stereo_left", "stereo_right"])
        dataset.set_visual_force_weight_provider(
            MultiCameraPackedTissueVisualForceWeights(
                {
                    "stereo_left": LEFT_MASK_DIR,
                    "stereo_right": RIGHT_MASK_DIR,
                },
                erosion_radius_px=7,
                highlight_weight=0.10,
            )
        )
        timestamp = float(
            dataset.offline_cameras.cameras["stereo_left"].timestamps[0]
        )
        dataset.update_frames(timestamp)
        settings_visual = environment.visual_forces_settings
        settings_visual.iterations = 1
        environment.frames = dataset.frames
        synchronized_seconds(
            lambda: simulator.compute_visual_forces(
                settings_visual,
                environment.frames,
                settings.dt / settings.substeps,
            )
        )
        simulator.state_0.particle_f.zero_()
        stereo_visual_force_ms = 1.0e3 * synchronized_seconds(
            lambda: simulator.compute_visual_forces(
                settings_visual,
                environment.frames,
                settings.dt / settings.substeps,
            )
        )

    physics_array = np.asarray(contact_physics_s)
    skinning_array = np.asarray(skinning_s)
    result = {
        "tissue_mode": args.tissue_mode,
        "world_translation_mm": list(map(float, args.world_translation_mm)),
        "jaw_mode": "raw_q7_without_artificial_minimum_opening",
        "grip_max_correction_mm": 1.0e3
        * float(
            simulator.triangle_skin_contact_projector
            .persistent_grip_maximum_correction_m
        ),
        "grip_compliance_m_per_n": float(
            simulator.triangle_skin_contact_projector
            .persistent_grip_compliance_m_per_n
        ),
        "grip_max_capture_penetration_mm": 1.0e3
        * float(
            simulator.triangle_skin_contact_projector
            .persistent_grip_maximum_capture_penetration_m
        ),
        "grip_min_capture_volume_ratio": float(
            simulator.triangle_skin_contact_projector
            .persistent_grip_minimum_capture_volume_ratio
        ),
        "surface_max_correction_mm": 1.0e3
        * float(settings.triangle_skin_contact_max_correction_m),
        "top_barrier_max_correction_mm": 1.0e3
        * float(settings.triangle_skin_top_barrier_max_correction_m),
        "jaw_friction_coefficient": float(
            simulator.triangle_skin_contact_projector
            .jaw_friction_coefficient
        ),
        "jaw_tangential_contact_enabled": not bool(
            args.disable_jaw_tangential_contact
        ),
        "top_barrier_enabled": not bool(args.disable_top_barrier),
        "young_modulus_pa_override": args.young_modulus_pa,
        "poisson_ratio_override": args.poisson_ratio,
        "contact_substep_stride": settings.triangle_skin_contact_substep_stride,
        "counts": {
            "particles": int(model.particle_count),
            "tetrahedra": int(model.tet_count),
            "gaussians": int(simulator.gaussian_model.num_gaussians),
            "trajectory_steps": len(state_indices),
        },
        "solver": {
            "substeps": settings.substeps,
            "material_iterations_per_substep": settings.material_iterations,
            "triangle_contact_iterations_per_substep": settings.triangle_skin_contact_iterations,
        },
        "timing_ms": {
            "physics_without_contact_median": 1.0e3 * float(np.median(baseline_physics_s)),
            "physics_with_contact_median": 1.0e3 * float(np.median(physics_array)),
            "physics_with_contact_p95": 1.0e3 * float(np.quantile(physics_array, 0.95)),
            "gaussian_skinning_without_contact_median": 1.0e3 * float(np.median(baseline_skinning_s)),
            "gaussian_skinning_trajectory_median": 1.0e3 * float(np.median(skinning_array)),
            "stereo_visual_force_one_iteration": stereo_visual_force_ms,
        },
        "initial_contact": initial_metrics,
        "capture_state": capture_state,
        "capture_metrics": capture_metrics,
        "selected_incident_volume_ratios_at_capture": (
            selected_incident_volume_ratios_at_capture
        ),
        "grip_proposal_audit_at_capture": grip_proposal_audit_at_capture,
        "final_grip_active": bool(trace[-1]["grip_active"]),
        "direct_particle_z_displacement_mm": direct_z_mm,
        "direct_particle_motion_since_capture_mm": direct_since_capture_mm,
        "virtual_frame_motion_since_capture_mm": (
            virtual_frame_motion_since_capture_mm
        ),
        "direct_target_motion_since_capture_mm": (
            direct_target_motion_since_capture_mm
        ),
        "direct_target_tracking_error_mm": direct_target_tracking_error_mm,
        "direct_particle_jaw_body_ids": (
            direct_body_ids_at_capture.tolist()
            if direct_body_ids_at_capture is not None
            else None
        ),
        "maximum_dynamic_displacement_mm": 1.0e3
        * float(torch.linalg.vector_norm(displacement[dynamic_mask], dim=1).max().item()),
        "dynamic_z_displacement_mm": {
            "minimum": 1.0e3 * float(dynamic_z_displacement.min().item()),
            "median": 1.0e3 * float(dynamic_z_displacement.median().item()),
            "maximum": 1.0e3 * float(dynamic_z_displacement.max().item()),
        },
        "press_roi_3mm_z_displacement_mm": press_roi_z_mm,
        "stereo_left_shoulder_1p5_to_5mm_z_displacement_mm": (
            left_shoulder_z_mm
        ),
        "zero_material_safety_states": zero_safety_states,
        "minimum_material_safety_scale": minimum_safety_scale,
        "minimum_grip_support_safe_scale": minimum_grip_support_scale,
        "minimum_grip_support_safe_scale_state": (
            minimum_grip_support_scale_state
        ),
        "minimum_grip_direct_safe_scale": minimum_grip_direct_scale,
        "minimum_grip_direct_safe_scale_state": (
            minimum_grip_direct_scale_state
        ),
        "minimum_grip_jaw_safe_scales": minimum_grip_jaw_scales,
        "minimum_grip_jaw_safe_scale_states": (
            minimum_grip_jaw_scale_states
        ),
        "maximum_particle_speed_m_s": maximum_particle_speed_m_s,
        "maximum_particle_speed_state": maximum_particle_speed_state,
        "maximum_particle_speed_id": maximum_particle_speed_id,
        "maximum_direct_particle_speed_m_s": (
            maximum_direct_particle_speed_m_s
        ),
        "maximum_direct_particle_speed_state": (
            maximum_direct_particle_speed_state
        ),
        "final_contact_metrics": final_contact_metrics,
        "maximum_unsafe_tetrahedra": maximum_unsafe_tetrahedra,
        "minimum_tetrahedron_volume_ratio": float(volume_ratios.min().item()),
        "inverted_tetrahedra": int(torch.count_nonzero(volume_ratios <= 0.0).item()),
    }
    if args.include_trace:
        result["trace"] = trace
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
