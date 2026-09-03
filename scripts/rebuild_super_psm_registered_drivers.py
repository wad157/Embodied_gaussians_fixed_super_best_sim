#!/usr/bin/env python3

"""Rebuild GUI drivers with a fixed strict-LND-to-URDF registration.

This is a lightweight adapter pass: it does not rerun segmentation or pose
optimization.  It converts both the registered LND prior and the part-corrected
paper states into the seven current URDF visual-link poses, then resamples the
1441 video poses onto the 5458 robot-state timestamps used by the GUI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


REPO = Path(__file__).resolve().parents[1]
TRACK_ROOT = REPO / "data/super/psm_tracking"
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

from super_psm_tracking_common import (  # noqa: E402
    TrackingInputs,
    _paper_component_transforms,
    ctr_to_matrix,
    save_runtime_driver,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild registered-LND and corrected GUI pose drivers."
    )
    parser.add_argument(
        "--states",
        type=Path,
        default=TRACK_ROOT
        / "part_pose_correction_full/tracking_states_part_corrected.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TRACK_ROOT / "part_pose_correction_full",
    )
    parser.add_argument(
        "--registered-driver",
        type=Path,
        default=TRACK_ROOT / "psm_registered_lnd_pose_driver.npz",
    )
    parser.add_argument(
        "--corrected-driver",
        type=Path,
        default=TRACK_ROOT / "psm_part_corrected_pose_driver.npz",
    )
    return parser.parse_args()


def save_video_poses(
    path: Path,
    inputs: TrackingInputs,
    poses: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        timestamps=inputs.video_timestamps[: len(poses)],
        link_names=np.asarray(inputs.link_names),
        poses_rect_camera_xyz_xyzw=poses.astype(np.float32),
    )


def pose_delta_summary(
    poses: np.ndarray,
    reference: np.ndarray,
) -> dict[str, list[float]]:
    translation_mm = np.linalg.norm(
        poses[:, :3] - reference[:, :3], axis=1
    ) * 1000.0
    dot = np.abs(np.sum(poses[:, 3:] * reference[:, 3:], axis=1))
    rotation_deg = np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))
    return {
        "translation_mm_per_link": translation_mm.tolist(),
        "rotation_deg_per_link": rotation_deg.tolist(),
    }


def strict_current_poses_at_first_video_frame(
    inputs: TrackingInputs,
) -> np.ndarray:
    state_index = int(inputs.video_joint_indices()[0])
    q7 = inputs.q7_states[state_index]
    lnd = inputs.lnd_forward_kinematics(q7)
    poses = np.empty((len(inputs.link_names), 7), dtype=np.float32)
    for link_index, lnd_link_id in enumerate(inputs.lnd_link_ids):
        matrix = (
            inputs.T_rectified_camera_psm_base
            @ lnd[int(lnd_link_id)]
            @ inputs.T_lndlink_urdf_link[link_index]
        )
        poses[link_index, :3] = matrix[:3, 3]
        poses[link_index, 3:] = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return poses


def transform_local_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_quat(pose[3:]).as_matrix()
    return points @ rotation.T + pose[:3]


def estimate_camera_alignment(
    prior_ctr: np.ndarray,
    prior_joints: np.ndarray,
    strict_first: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit the current tip-only Gaussian surface to registered paper CAD."""
    from propagate_super_psm_part_masks import load_paper_meshes

    surface = np.load(
        REPO / "data/super/psm_robot/psm_surface_gaussians.npz"
    )
    link_names = surface["link_names"].tolist()
    main_link_id = link_names.index("PSM1_tool_main_link")
    source_parts = []
    for link_index, pose in enumerate(strict_first):
        if link_index == main_link_id:
            continue
        selected = surface["link_ids"] == link_index
        source_parts.append(
            transform_local_points(surface["means"][selected], pose)
        )
    source = np.concatenate(source_parts).astype(np.float64)

    vertices, _ = load_paper_meshes()
    camera = ctr_to_matrix(prior_ctr[0])
    components = _paper_component_transforms(prior_joints[0])
    # The GUI's default tip mode excludes tool_main.  Match its six distal
    # link surfaces against the paper logo and two jaw meshes; the long paper
    # shaft would otherwise dominate the registration with unrelated points.
    target_parts = []
    for mesh_vertices, component_index in zip(
        vertices[1:], (1, 3, 4), strict=True
    ):
        transform = camera @ components[component_index]
        target_parts.append(
            mesh_vertices @ transform[:3, :3].T + transform[:3, 3]
        )
    target = np.concatenate(target_parts).astype(np.float64)

    tree = cKDTree(target)
    alignment = np.eye(4, dtype=np.float64)
    alignment[:3, 3] = np.median(target, axis=0) - np.median(
        source, axis=0
    )
    iterations = 0
    for iterations in range(100):
        transformed = (
            source @ alignment[:3, :3].T + alignment[:3, 3]
        )
        distances, target_ids = tree.query(transformed)
        keep = distances <= np.percentile(distances, 80.0)
        source_kept = source[keep]
        target_kept = target[target_ids[keep]]
        source_center = source_kept.mean(axis=0)
        target_center = target_kept.mean(axis=0)
        covariance = (
            (source_kept - source_center).T
            @ (target_kept - target_center)
        )
        left, _, right_t = np.linalg.svd(covariance)
        rotation = right_t.T @ left.T
        if np.linalg.det(rotation) < 0.0:
            right_t[-1] *= -1.0
            rotation = right_t.T @ left.T
        translation = target_center - rotation @ source_center
        updated = np.eye(4, dtype=np.float64)
        updated[:3, :3] = rotation
        updated[:3, 3] = translation
        if np.linalg.norm(updated - alignment) < 1e-10:
            alignment = updated
            break
        alignment = updated

    transformed = source @ alignment[:3, :3].T + alignment[:3, 3]
    distances = tree.query(transformed)[0]
    report = {
        "source_current_tip_gaussian_count": int(len(source)),
        "target_paper_distal_vertex_count": int(len(target)),
        "icp_iterations": iterations + 1,
        "trim_percentile": 80.0,
        "nearest_target_distance_mm_p0_p25_p50_p75_p80_p95_max": np.percentile(
            distances * 1000.0, [0, 25, 50, 75, 80, 95, 100]
        ).tolist(),
        "translation_mm": (alignment[:3, 3] * 1000.0).tolist(),
        "rotation_rotvec_deg": np.degrees(
            Rotation.from_matrix(alignment[:3, :3]).as_rotvec()
        ).tolist(),
        "rotation_angle_deg": float(
            np.degrees(
                Rotation.from_matrix(alignment[:3, :3]).magnitude()
            )
        ),
    }
    return alignment, report


def main() -> None:
    args = parse_args()
    if not args.states.exists():
        raise FileNotFoundError(args.states)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = TrackingInputs.load()
    states = np.load(args.states)
    prior_ctr = states["prior_ctr"].astype(np.float32)
    prior_joints = states["prior_joints"].astype(np.float32)
    corrected_ctr = states["corrected_ctr"].astype(np.float32)
    corrected_joints = states["corrected_joints"].astype(np.float32)
    expected = len(inputs.video_timestamps)
    if not (
        len(prior_ctr)
        == len(prior_joints)
        == len(corrected_ctr)
        == len(corrected_joints)
        == expected
    ):
        raise ValueError("Expected complete 1441-frame registered/corrected states")

    strict_first = strict_current_poses_at_first_video_frame(inputs)
    camera_alignment, alignment_report = estimate_camera_alignment(
        prior_ctr, prior_joints, strict_first
    )
    registration_path = args.output_dir / "paper_to_gui_registration.npz"
    np.savez_compressed(
        registration_path,
        T_rectified_camera_alignment=camera_alignment,
        registration_ctr=prior_ctr[0],
        registration_joints=prior_joints[0],
    )
    conversion_options = {
        "registration_ctr": prior_ctr[0],
        "registration_joints": prior_joints[0],
        "camera_alignment": camera_alignment,
    }
    registered_poses = inputs.paper_states_to_visual_poses(
        prior_ctr, prior_joints, **conversion_options
    )
    corrected_poses = inputs.paper_states_to_visual_poses(
        corrected_ctr, corrected_joints, **conversion_options
    )
    registered_gui_wrist_angles = inputs.gui_wrist_angles_from_paper_residual(
        prior_joints, prior_joints
    )
    corrected_gui_wrist_angles = inputs.gui_wrist_angles_from_paper_residual(
        prior_joints, corrected_joints
    )
    registered_gui_jaw_angles = inputs.gui_jaw_angles_from_paper_residual(
        prior_joints, prior_joints
    )
    corrected_gui_jaw_angles = inputs.gui_jaw_angles_from_paper_residual(
        prior_joints, corrected_joints
    )
    registered_poses = inputs.enforce_urdf_distal_kinematics(
        registered_poses,
        registered_gui_wrist_angles,
        registered_gui_jaw_angles,
    )
    corrected_poses = inputs.enforce_urdf_distal_kinematics(
        corrected_poses,
        corrected_gui_wrist_angles,
        corrected_gui_jaw_angles,
    )
    registered_video_path = (
        args.output_dir / "visual_poses_registered_lnd.npz"
    )
    corrected_video_path = (
        args.output_dir / "visual_poses_part_corrected.npz"
    )
    save_video_poses(registered_video_path, inputs, registered_poses)
    save_video_poses(corrected_video_path, inputs, corrected_poses)
    save_runtime_driver(args.registered_driver, inputs, registered_poses)
    save_runtime_driver(args.corrected_driver, inputs, corrected_poses)

    gui_state_index = int(
        np.searchsorted(
            inputs.joint_timestamps,
            inputs.video_timestamps[0],
            side="right",
        )
        - 1
    )
    with np.load(args.registered_driver) as registered_runtime:
        registered_gui_first = registered_runtime[
            "poses_rect_camera_xyz_xyzw"
        ][gui_state_index]
    with np.load(args.corrected_driver) as corrected_runtime:
        corrected_gui_first = corrected_runtime[
            "poses_rect_camera_xyz_xyzw"
        ][gui_state_index]
    report = {
        "method": (
            "Paper-component to current-URDF-link registration uses the "
            "registered-LND reference, followed by a fixed camera-space SE(3) "
            "fit from the current tip Gaussians to the registered paper CAD; "
            "the corrected sequence uses the same immutable registration."
        ),
        "frame_count": expected,
        "runtime_state_count": len(inputs.joint_timestamps),
        "geometry_alignment": alignment_report,
        "jaw_kinematics": {
            "method": (
                "The complete distal chain is rebuilt from the tracked "
                "tool_wrist_link anchor with literal URDF wrist-pitch, "
                "wrist-yaw, and jaw parent/child transforms. Both jaws use "
                "origin=(0,0,0), axis=(0,0,1), and symmetric +/-jaw/2 rigid "
                "rotations. CAD registration is not inserted between URDF "
                "joints. The "
                "encoder jaw is the baseline and twice the common paper "
                "half-angle residual supplies the image correction."
            ),
            "registered_gui_jaw_deg_min_p05_p50_p95_max": np.percentile(
                np.degrees(registered_gui_jaw_angles), [0, 5, 50, 95, 100]
            ).tolist(),
            "corrected_gui_jaw_deg_min_p05_p50_p95_max": np.percentile(
                np.degrees(corrected_gui_jaw_angles), [0, 5, 50, 95, 100]
            ).tolist(),
            "image_jaw_correction_deg_min_p05_p50_p95_max": np.percentile(
                np.degrees(
                    corrected_gui_jaw_angles - registered_gui_jaw_angles
                ),
                [0, 5, 50, 95, 100],
            ).tolist(),
        },
        "wrist_kinematics": {
            "method": (
                "URDF wrist-pitch q4 stays on the encoder to preserve the "
                "shaft-in-jaw-plane constraint. URDF wrist-yaw q5 uses the "
                "encoder baseline plus only corrected-minus-prior paper q5; "
                "the paper static joint-zero offset is not applied twice."
            ),
            "registered_gui_wrist_deg_min_p05_p50_p95_max": np.percentile(
                np.degrees(registered_gui_wrist_angles),
                [0, 5, 50, 95, 100],
                axis=0,
            ).T.tolist(),
            "corrected_gui_wrist_deg_min_p05_p50_p95_max": np.percentile(
                np.degrees(corrected_gui_wrist_angles),
                [0, 5, 50, 95, 100],
                axis=0,
            ).T.tolist(),
        },
        "registration_asset": str(registration_path),
        "first_video_timestamp": float(inputs.video_timestamps[0]),
        "first_gui_state_index": gui_state_index,
        "first_gui_state_timestamp": float(
            inputs.joint_timestamps[gui_state_index]
        ),
        "registered_first_vs_strict": pose_delta_summary(
            registered_gui_first, strict_first
        ),
        "corrected_first_vs_strict": pose_delta_summary(
            corrected_gui_first, strict_first
        ),
        "corrected_first_vs_registered": pose_delta_summary(
            corrected_gui_first, registered_gui_first
        ),
        "registered_video_poses": str(registered_video_path),
        "corrected_video_poses": str(corrected_video_path),
        "registered_runtime_driver": str(args.registered_driver),
        "corrected_runtime_driver": str(args.corrected_driver),
    }
    report_path = args.output_dir / "driver_registration_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
