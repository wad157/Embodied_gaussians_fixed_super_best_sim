#!/usr/bin/env python3
"""Evaluate a short exact-timestamp masked visual-force sequence against baseline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
EXAMPLES_DIR = REPO_ROOT / "examples"
for path in (SRC_DIR, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from embodied_environments.super_embodied.super_embodied import (  # noqa: E402
    PSM_ARTICULATION_INDEX,
    apply_psm_lnd_pose,
    build_environment,
    expand_psm_q7_to_urdf_order,
    load_mimic_config,
    urdf_actuated_joint_order,
)
from embodied_gaussians import DatasetManager  # noqa: E402
from embodied_gaussians.embodied_simulator.visual_force_masks import (  # noqa: E402
    PackedTissueVisualForceWeights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate masked visual forces.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "data/super/grasp5_offline_demo",
    )
    parser.add_argument(
        "--mask-asset",
        type=Path,
        default=REPO_ROOT / "data/super/grasp5_native/visual_force_masks_v1",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--physics-steps-per-frame", type=int, default=2)
    parser.add_argument("--visual-update-interval", type=int, default=3)
    parser.add_argument("--visual-iterations", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "data/super/grasp5_native/soft_tissue_v1/visual_sequence_report.json",
    )
    return parser.parse_args()


def pose_driver(dataset: DatasetManager, environment, timestamp: float) -> int:
    robot = dataset.robots["PSM1"]
    state_index = int(
        np.searchsorted(robot.states_timestamps, timestamp, side="right") - 1
    )
    state_index = max(0, min(state_index, len(robot.states) - 1))
    q7 = np.asarray(robot.states[state_index]["q"], dtype=np.float64)
    q_full = expand_psm_q7_to_urdf_order(
        q7,
        mimic_cfg=load_mimic_config(),
        joint_order=urdf_actuated_joint_order(
            REPO_ROOT / "data/super/psm_robot/psm.urdf"
        ),
    )
    q_tensor = torch.from_numpy(q_full).float()
    environment.set_robot_q(PSM_ARTICULATION_INDEX, q_tensor)
    environment.set_robot_desired_q(PSM_ARTICULATION_INDEX, q_tensor)
    apply_psm_lnd_pose(environment, state_index)
    return state_index


def particle_metrics(environment, rest_positions, rest_volume, support_mask):
    positions = wp.to_torch(environment.sim.state_0.particle_q)
    velocities = wp.to_torch(environment.sim.state_0.particle_qd)
    tet_indices = wp.to_torch(environment.sim.model.tet_indices).long()
    tet_positions = positions[tet_indices]
    ds = torch.stack(
        (
            tet_positions[:, 1] - tet_positions[:, 0],
            tet_positions[:, 2] - tet_positions[:, 0],
            tet_positions[:, 3] - tet_positions[:, 0],
        ),
        dim=-1,
    )
    volume = torch.linalg.det(ds) / 6.0
    displacement = torch.linalg.vector_norm(positions - rest_positions, dim=1)
    dynamic = ~support_mask
    return {
        "positions": positions.clone(),
        "center": positions[dynamic].mean(dim=0),
        "finite": bool(
            torch.isfinite(positions).all().item()
            and torch.isfinite(velocities).all().item()
            and torch.isfinite(volume).all().item()
            and torch.isfinite(environment.sim.gaussian_state.means).all().item()
        ),
        "inverted_tetrahedra": int((volume <= 0.0).sum().item()),
        "total_volume_ratio": float((volume.sum() / rest_volume.sum()).item()),
        "anchor_drift_max_m": float(displacement[support_mask].max().item()),
        "particle_speed_max_m_s": float(
            torch.linalg.vector_norm(velocities, dim=1).max().item()
        ),
        "particle_displacement_max_m": float(displacement.max().item()),
    }


def particle_shape_contact_metrics(environment):
    """Measure generated jaw candidates and contacts still penetrating post-solve."""
    model = environment.sim.model
    if model.soft_contact_count is None:
        return {
            "candidate_count": 0,
            "active_count": 0,
            "penetration_max_m": 0.0,
        }
    count = min(
        model.soft_contact_max,
        int(wp.to_torch(model.soft_contact_count)[0].item()),
    )
    if count == 0:
        return {
            "candidate_count": 0,
            "active_count": 0,
            "penetration_max_m": 0.0,
        }
    contact_particle = wp.to_torch(model.soft_contact_particle)[:count].long()
    contact_shape = wp.to_torch(model.soft_contact_shape)[:count].long()
    shape_body = wp.to_torch(model.shape_body)[contact_shape].long()
    if bool((shape_body < 0).any().item()):
        raise RuntimeError("SUPER jaw soft contacts must belong to PSM bodies")
    body_q = wp.to_torch(environment.sim.state_0.body_q)[shape_body]
    body_local = wp.to_torch(model.soft_contact_body_pos)[:count]
    quaternion_xyz = body_q[:, 3:6]
    quaternion_w = body_q[:, 6:7]
    quaternion_cross = torch.linalg.cross(quaternion_xyz, body_local)
    body_world = body_q[:, :3] + body_local + 2.0 * (
        quaternion_w * quaternion_cross
        + torch.linalg.cross(quaternion_xyz, quaternion_cross)
    )
    particle_position = wp.to_torch(environment.sim.state_0.particle_q)[
        contact_particle
    ]
    particle_radius = wp.to_torch(model.particle_radius)[contact_particle]
    normal = wp.to_torch(model.soft_contact_normal)[:count]
    signed_separation = (
        (normal * (particle_position - body_world)).sum(dim=1) - particle_radius
    )
    active = signed_separation <= model.particle_adhesion
    penetration = torch.clamp(-signed_separation[active], min=0.0)
    return {
        "candidate_count": count,
        "active_count": int(active.sum().item()),
        "penetration_max_m": (
            float(penetration.max().item()) if len(penetration) else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    if min(
        args.frames,
        args.physics_steps_per_frame,
        args.visual_update_interval,
        args.visual_iterations,
    ) <= 0:
        raise ValueError("frame/step/iteration arguments must be positive")
    wp.init()
    environment = build_environment(num_envs=1, add_gaussians=True, device=args.device)
    dataset = DatasetManager(args.dataset)
    dataset.keep_only_cameras(["stereo_left"])
    provider = PackedTissueVisualForceWeights(
        args.mask_asset,
        camera_name="stereo_left",
        erosion_radius_px=7,
        highlight_weight=0.10,
    )
    dataset.set_visual_force_weight_provider(provider)
    environment.frames = dataset.frames
    timestamps = np.asarray(
        dataset.offline_cameras.cameras["stereo_left"].timestamps,
        dtype=np.float64,
    )
    end_frame = args.start_frame + args.frames
    if args.start_frame < 0 or end_frame > len(timestamps):
        raise ValueError("Requested camera frame range is outside the dataset")
    selected_timestamps = timestamps[args.start_frame:end_frame]

    model = environment.sim.model
    rest_positions = wp.to_torch(model.particle_q).clone()
    inverse_mass = wp.to_torch(model.particle_inv_mass)
    support_mask = inverse_mass == 0.0
    inverse_rest = torch.as_tensor(
        np.asarray(environment.builder().tet_poses),
        device=rest_positions.device,
        dtype=torch.float32,
    )
    rest_volume = torch.linalg.det(inverse_rest).reciprocal() / 6.0
    initial = environment.sim.clone_embodied_gaussian_state()

    # Warm and capture the physics graph before either measured branch.
    pose_driver(dataset, environment, float(selected_timestamps[0]))
    environment.sim.physics_step(environment.physics_settings)
    wp.synchronize_device(model.device)
    environment.sim.copy_embodied_gaussian_state(initial)
    environment.sim.reset()

    baseline_positions = []
    for timestamp in selected_timestamps:
        pose_driver(dataset, environment, float(timestamp))
        for _ in range(args.physics_steps_per_frame):
            environment.sim.physics_step(environment.physics_settings)
            pose_driver(dataset, environment, float(timestamp))
            environment.sim.update_gaussian_transforms()
        baseline_positions.append(wp.to_torch(environment.sim.state_0.particle_q).clone())
    baseline_positions = torch.stack(baseline_positions)

    environment.sim.copy_embodied_gaussian_state(initial)
    environment.sim.reset()
    settings = environment.visual_forces_settings
    settings.iterations = args.visual_iterations
    settings.enable_soft_particle_forces = True
    settings.require_loss_weights_for_soft = True

    samples = []
    visual_updates = []
    physics_step_index = 0
    wp.synchronize_device(model.device)
    start_time = time.perf_counter()
    for local_frame, timestamp in enumerate(selected_timestamps):
        camera_frame = args.start_frame + local_frame
        dataset.update_frames(float(timestamp))
        state_index = pose_driver(dataset, environment, float(timestamp))
        for _ in range(args.physics_steps_per_frame):
            environment.sim.physics_step(environment.physics_settings)
            pose_driver(dataset, environment, float(timestamp))
            environment.sim.update_gaussian_transforms()
            physics_step_index += 1
            if physics_step_index % args.visual_update_interval == 0:
                environment.sim.compute_visual_forces(
                    settings,
                    environment.frames,
                    environment.physics_settings.dt
                    / environment.physics_settings.substeps,
                )
                wp.synchronize_device(model.device)
                particle_force = wp.to_torch(environment.sim.state_0.particle_f)
                force_norm = torch.linalg.vector_norm(particle_force, dim=1)
                body_force = wp.to_torch(environment.sim.state_0.body_f)
                soft_ids = environment.sim.gaussian_model.soft_gaussian_ids.long()
                target_delta = (
                    environment.sim.visual_forces.means[soft_ids]
                    - environment.sim.gaussian_state.means[soft_ids]
                )
                visual_updates.append(
                    {
                        "camera_frame": camera_frame,
                        "physics_step": physics_step_index,
                        "loss": float(environment.sim.last_visual_force_loss.item()),
                        "particle_force_budget_n": float(force_norm.sum().item()),
                        "particle_force_max_n": float(force_norm.max().item()),
                        "psm_body_force_max": float(
                            torch.linalg.vector_norm(body_force, dim=1).max().item()
                        ),
                        "soft_target_delta_max_m": float(
                            torch.linalg.vector_norm(target_delta, dim=1).max().item()
                        ),
                    }
                )
        metrics = particle_metrics(
            environment,
            rest_positions,
            rest_volume,
            support_mask,
        )
        contact_metrics = particle_shape_contact_metrics(environment)
        baseline = baseline_positions[local_frame]
        visual_minus_baseline = metrics["positions"] - baseline
        dynamic = ~support_mask
        baseline_center = baseline[dynamic].mean(dim=0)
        center_delta = metrics["center"] - baseline_center
        nonrigid_delta = visual_minus_baseline - center_delta
        samples.append(
            {
                "camera_frame": camera_frame,
                "timestamp": float(timestamp),
                "robot_state_index": state_index,
                "mask_frame_index": provider.last_frame_index,
                "mask_valid_pixels": provider.last_valid_pixels,
                "mask_highlight_pixels": provider.last_highlight_pixels,
                "finite": metrics["finite"],
                "inverted_tetrahedra": metrics["inverted_tetrahedra"],
                "total_volume_ratio": metrics["total_volume_ratio"],
                "anchor_drift_max_m": metrics["anchor_drift_max_m"],
                "particle_speed_max_m_s": metrics["particle_speed_max_m_s"],
                "particle_displacement_max_m": metrics[
                    "particle_displacement_max_m"
                ],
                "jaw_contact_candidate_count": contact_metrics["candidate_count"],
                "jaw_contact_active_count": contact_metrics["active_count"],
                "jaw_contact_penetration_max_m": contact_metrics[
                    "penetration_max_m"
                ],
                "center_drift_vs_baseline_m": float(
                    torch.linalg.vector_norm(center_delta).item()
                ),
                "nonrigid_change_mean_vs_baseline_m": float(
                    torch.linalg.vector_norm(nonrigid_delta[dynamic], dim=1)
                    .mean()
                    .item()
                ),
                "nonrigid_change_max_vs_baseline_m": float(
                    torch.linalg.vector_norm(nonrigid_delta[dynamic], dim=1)
                    .max()
                    .item()
                ),
            }
        )
    wp.synchronize_device(model.device)
    elapsed = time.perf_counter() - start_time

    # The project-local material projector accumulates tetrahedron corrections
    # into shared particles with GPU atomics.  Replaying the same initial state
    # is therefore numerically stable but not bitwise deterministic.  Samples
    # completed before the first scattered force can be consumed provide an
    # honest A/A replay floor; branch differences below that floor must not be
    # attributed to visual forces.
    first_force_consumed_step = visual_updates[0]["physics_step"] + 1
    pre_force_samples = [
        sample
        for local_frame, sample in enumerate(samples)
        if (local_frame + 1) * args.physics_steps_per_frame
        < first_force_consumed_step
    ]
    if not pre_force_samples:
        raise RuntimeError("Sequence configuration has no pre-force replay sample")
    replay_floor_nonrigid_m = max(
        sample["nonrigid_change_max_vs_baseline_m"]
        for sample in pre_force_samples
    )
    post_force_samples = samples[len(pre_force_samples) :]
    post_force_branch_nonrigid_m = max(
        sample["nonrigid_change_max_vs_baseline_m"]
        for sample in post_force_samples
    )

    gates = {
        "all_finite": all(sample["finite"] for sample in samples),
        "no_inverted_tetrahedra": max(
            sample["inverted_tetrahedra"] for sample in samples
        )
        == 0,
        "volume_error_below_2_percent": max(
            abs(sample["total_volume_ratio"] - 1.0) for sample in samples
        )
        < 0.02,
        "anchor_drift_below_1_um": max(
            sample["anchor_drift_max_m"] for sample in samples
        )
        < 1.0e-6,
        "speed_below_0_25_m_s": max(
            sample["particle_speed_max_m_s"] for sample in samples
        )
        < 0.25,
        "center_drift_vs_baseline_below_1_mm": max(
            sample["center_drift_vs_baseline_m"] for sample in samples
        )
        < 1.0e-3,
        "mask_never_empty": min(sample["mask_valid_pixels"] for sample in samples)
        > 0,
        "visual_loss_finite": bool(
            visual_updates
            and np.isfinite([update["loss"] for update in visual_updates]).all()
        ),
        "force_budget_enforced": max(
            update["particle_force_budget_n"] for update in visual_updates
        )
        <= settings.soft_max_total_force + 1.0e-8,
        "psm_force_zero": max(
            update["psm_body_force_max"] for update in visual_updates
        )
        == 0.0,
        "pre_force_replay_difference_below_0_25_mm": (
            replay_floor_nonrigid_m < 2.5e-4
        ),
        "jaw_contact_postsolve_penetration_below_0_25_mm": max(
            sample["jaw_contact_penetration_max_m"] for sample in samples
        )
        < 2.5e-4,
    }
    report = {
        "stage": "E_masked_visual_force_short_sequence",
        "device": str(model.device),
        "configuration": {
            "start_frame": args.start_frame,
            "frames": args.frames,
            "physics_steps_per_frame": args.physics_steps_per_frame,
            "visual_update_interval": args.visual_update_interval,
            "visual_iterations": args.visual_iterations,
            "total_visual_updates": len(visual_updates),
        },
        "performance": {
            "elapsed_seconds": elapsed,
            "seconds_per_camera_frame": elapsed / args.frames,
        },
        "summary": {
            "max_volume_error": max(
                abs(sample["total_volume_ratio"] - 1.0) for sample in samples
            ),
            "max_particle_speed_m_s": max(
                sample["particle_speed_max_m_s"] for sample in samples
            ),
            "max_center_drift_vs_baseline_m": max(
                sample["center_drift_vs_baseline_m"] for sample in samples
            ),
            "max_nonrigid_change_vs_baseline_m": max(
                sample["nonrigid_change_max_vs_baseline_m"] for sample in samples
            ),
            "pre_force_replay_nonrigid_floor_m": replay_floor_nonrigid_m,
            "post_force_branch_nonrigid_max_m": post_force_branch_nonrigid_m,
            "post_force_excess_over_replay_floor_m": max(
                0.0, post_force_branch_nonrigid_m - replay_floor_nonrigid_m
            ),
            "loss_first": visual_updates[0]["loss"],
            "loss_last": visual_updates[-1]["loss"],
            "loss_min": min(update["loss"] for update in visual_updates),
            "loss_max": max(update["loss"] for update in visual_updates),
            "max_force_budget_n": max(
                update["particle_force_budget_n"] for update in visual_updates
            ),
            "max_jaw_contact_candidates": max(
                sample["jaw_contact_candidate_count"] for sample in samples
            ),
            "max_jaw_contact_active": max(
                sample["jaw_contact_active_count"] for sample in samples
            ),
            "max_jaw_contact_penetration_m": max(
                sample["jaw_contact_penetration_max_m"] for sample in samples
            ),
        },
        "visual_updates": visual_updates,
        "samples": samples,
        "gates": gates,
        "interpretation": {
            "gate_scope": "closed-loop safety and boundedness, not causal deformation accuracy",
            "replay_floor_source": (
                "GPU atomic accumulation order in tetrahedral material projection"
            ),
            "causal_visual_deformation_isolated": False,
            "reason": (
                "the nonzero pre-force A/A replay floor is comparable to the "
                "post-force branch difference"
            ),
        },
        "passed": bool(all(gates.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Masked visual-force short-sequence gate failed")


if __name__ == "__main__":
    main()
