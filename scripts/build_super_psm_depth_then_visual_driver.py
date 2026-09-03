#!/usr/bin/env python3

"""Build a full candidate driver from accepted depth-then-visual keyframes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation


REPO = Path(__file__).resolve().parents[1]
TRACK_ROOT = REPO / "data/super/psm_tracking"
OLD_ROOT = (
    TRACK_ROOT / "part_pose_correction_expanded_3mm_8deg_12deg_25deg"
)
KEYFRAME_ROOT = TRACK_ROOT / "psm_depth_then_visual_v1"
OUTPUT_ROOT = TRACK_ROOT / "psm_depth_then_visual_full_v1"
EXPECTED_SHA256 = {
    "table_frame": "6dddc2178cdf816f5dada5febdd528f80e42d52e631076e1f5f4a952297adecf",
    "cameras": "e1e7b7e7e21ca8a9409c88a29409d2e9ad6b783a85b277e71cec0c84340ce4ef",
    "old_driver": "21df09849b08d6cef1ae47c694e7400b6df5228e0805c5c35ecf3d68e2ef648a",
}

sys.path[:0] = [str(REPO), str(REPO / "scripts")]

from super_psm_tracking_common import (  # noqa: E402
    TrackingInputs,
    ctr_to_matrix,
    matrix_to_ctr,
    pose_to_matrix,
    resample_pose_sequence,
    save_runtime_driver,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Propagate accepted depth-then-visual keyframes safely."
    )
    parser.add_argument(
        "--old-states",
        type=Path,
        default=OLD_ROOT / "tracking_states_part_corrected.npz",
    )
    parser.add_argument(
        "--old-video-poses",
        type=Path,
        default=OLD_ROOT / "visual_poses_part_corrected.npz",
    )
    parser.add_argument(
        "--registration",
        type=Path,
        default=OLD_ROOT / "paper_to_gui_registration.npz",
    )
    parser.add_argument(
        "--old-driver",
        type=Path,
        default=TRACK_ROOT / "psm_part_corrected_pose_driver.npz",
    )
    parser.add_argument(
        "--keyframe-states",
        type=Path,
        default=KEYFRAME_ROOT / "keyframe_states.npz",
    )
    parser.add_argument(
        "--keyframe-report",
        type=Path,
        default=KEYFRAME_ROOT / "report.json",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--runtime-driver",
        type=Path,
        default=TRACK_ROOT / "psm_depth_then_visual_pose_driver_candidate.npz",
    )
    parser.add_argument("--max-linear-speed-mm-s", type=float, default=5.0)
    parser.add_argument("--max-angular-speed-deg-s", type=float, default=10.0)
    parser.add_argument("--max-linear-accel-mm-s2", type=float, default=100.0)
    parser.add_argument("--max-angular-accel-deg-s2", type=float, default=300.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_residual(
    new_ctr: np.ndarray,
    new_joints: np.ndarray,
    old_ctr: np.ndarray,
    old_joints: np.ndarray,
) -> np.ndarray:
    new = ctr_to_matrix(new_ctr)
    old = ctr_to_matrix(old_ctr)
    rotation = new[:3, :3] @ old[:3, :3].T
    return np.concatenate(
        [
            Rotation.from_matrix(rotation).as_rotvec(),
            new[:3, 3] - old[:3, 3],
            np.asarray(new_joints, dtype=np.float64)
            - np.asarray(old_joints, dtype=np.float64),
        ]
    )


def apply_residuals(
    old_ctr: np.ndarray,
    old_joints: np.ndarray,
    residuals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    output_ctr = np.empty_like(old_ctr, dtype=np.float32)
    rotations = Rotation.from_rotvec(residuals[:, :3]).as_matrix()
    for index in range(len(old_ctr)):
        old = ctr_to_matrix(old_ctr[index])
        new = old.copy()
        new[:3, :3] = rotations[index] @ old[:3, :3]
        new[:3, 3] = old[:3, 3] + residuals[index, 3:6]
        output_ctr[index] = matrix_to_ctr(new)
    output_joints = np.asarray(old_joints, dtype=np.float64) + residuals[:, 6:10]
    output_joints[:, 0] = np.clip(output_joints[:, 0], -1.5707, 1.5707)
    output_joints[:, 1] = np.clip(output_joints[:, 1], -1.3963, 1.3963)
    output_joints[:, 2:] = np.clip(output_joints[:, 2:], 0.0, math.pi / 2.0)
    return output_ctr, output_joints.astype(np.float32)


def pose_errors(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    max_translation = 0.0
    max_rotation = 0.0
    for pose_a, pose_b in zip(
        first.reshape(-1, 7), second.reshape(-1, 7), strict=True
    ):
        matrix_a = pose_to_matrix(pose_a)
        matrix_b = pose_to_matrix(pose_b)
        max_translation = max(
            max_translation,
            float(np.linalg.norm(matrix_a[:3, 3] - matrix_b[:3, 3])),
        )
        relative = matrix_a[:3, :3] @ matrix_b[:3, :3].T
        max_rotation = max(
            max_rotation,
            math.degrees(float(Rotation.from_matrix(relative).magnitude())),
        )
    return max_translation, max_rotation


def correction_motion(
    timestamps: np.ndarray, residuals: np.ndarray
) -> dict[str, float]:
    dt = np.diff(timestamps)
    if np.any(dt <= 0.0):
        raise ValueError("Video timestamps must be strictly increasing")
    linear_velocity = np.diff(residuals[:, 3:6], axis=0) / dt[:, None]
    angular_velocity = np.diff(residuals[:, :3], axis=0) / dt[:, None]
    midpoint_dt = 0.5 * (dt[1:] + dt[:-1])
    linear_accel = np.diff(linear_velocity, axis=0) / midpoint_dt[:, None]
    angular_accel = np.diff(angular_velocity, axis=0) / midpoint_dt[:, None]
    return {
        "max_linear_speed_mm_s": float(
            np.max(np.linalg.norm(linear_velocity, axis=1)) * 1000.0
        ),
        "max_angular_speed_deg_s": float(
            np.degrees(np.max(np.linalg.norm(angular_velocity, axis=1)))
        ),
        "max_linear_accel_mm_s2": float(
            np.max(np.linalg.norm(linear_accel, axis=1)) * 1000.0
        ),
        "max_angular_accel_deg_s2": float(
            np.degrees(np.max(np.linalg.norm(angular_accel, axis=1)))
        ),
    }


def main() -> None:
    args = parse_args()
    paths = (
        args.old_states,
        args.old_video_poses,
        args.registration,
        args.old_driver,
        args.keyframe_states,
        args.keyframe_report,
        REPO / "data/super/table_frame.json",
        REPO / "data/super/grasp5_offline_demo/cameras.json",
    )
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    frozen_paths = {
        "table_frame": REPO / "data/super/table_frame.json",
        "cameras": REPO / "data/super/grasp5_offline_demo/cameras.json",
        "old_driver": args.old_driver,
    }
    frozen_hashes = {name: sha256(path) for name, path in frozen_paths.items()}
    if frozen_hashes != EXPECTED_SHA256:
        raise RuntimeError(
            f"Frozen coordinate/old-driver hash mismatch: {frozen_hashes}"
        )

    inputs = TrackingInputs.load()
    old = np.load(args.old_states)
    old_video_asset = np.load(args.old_video_poses)
    keyframes = np.load(args.keyframe_states)
    keyframe_report = json.loads(args.keyframe_report.read_text(encoding="utf-8"))
    registration = np.load(args.registration)
    frame_count = len(old["corrected_ctr"])
    if frame_count != len(inputs.video_timestamps):
        raise ValueError("Old corrected states must cover every video frame")

    conversion_options = {
        "registration_ctr": registration["registration_ctr"],
        "registration_joints": registration["registration_joints"],
        "camera_alignment": registration["T_rectified_camera_alignment"],
    }
    rebuilt_old_video = inputs.paper_states_to_visual_poses(
        old["corrected_ctr"], old["corrected_joints"], **conversion_options
    )
    old_wrist = inputs.gui_wrist_angles_from_paper_residual(
        old["prior_joints"], old["corrected_joints"]
    )
    old_jaw = inputs.gui_jaw_angles_from_paper_residual(
        old["prior_joints"], old["corrected_joints"]
    )
    rebuilt_old_video = inputs.enforce_urdf_distal_kinematics(
        rebuilt_old_video, old_wrist, old_jaw
    )
    baseline_video_translation_error, baseline_video_rotation_error = pose_errors(
        rebuilt_old_video, old_video_asset["poses_rect_camera_xyz_xyzw"]
    )
    if baseline_video_translation_error > 1.0e-9 or baseline_video_rotation_error > 1.0e-6:
        raise RuntimeError("Frozen old video-pose reconstruction is not exact")
    rebuilt_old_runtime = resample_pose_sequence(
        inputs.video_timestamps, rebuilt_old_video, inputs.joint_timestamps
    )
    old_runtime = np.load(args.old_driver)["poses_rect_camera_xyz_xyzw"]
    baseline_runtime_translation_error, baseline_runtime_rotation_error = pose_errors(
        rebuilt_old_runtime, old_runtime
    )
    if baseline_runtime_translation_error > 1.0e-7 or baseline_runtime_rotation_error > 1.0e-4:
        raise RuntimeError("Frozen old runtime-driver reconstruction is not exact")

    frame_indices = keyframes["frame_indices"].astype(np.int64)
    accepted_frames = [
        int(frame)
        for frame in frame_indices
        if keyframe_report["frame_metrics"][str(int(frame))][
            "visual_candidate_accepted"
        ]
    ]
    rejected_frames = [int(frame) for frame in frame_indices if int(frame) not in accepted_frames]
    local_by_frame = {int(frame): index for index, frame in enumerate(frame_indices)}
    anchor_frames = sorted(set([0, frame_count - 1, *frame_indices.tolist()]))
    anchor_residuals = np.zeros((len(anchor_frames), 10), dtype=np.float64)
    for anchor_index, frame in enumerate(anchor_frames):
        if frame not in accepted_frames:
            continue
        local_index = local_by_frame[frame]
        anchor_residuals[anchor_index] = state_residual(
            keyframes["accepted_ctr"][local_index],
            keyframes["accepted_joints"][local_index],
            old["corrected_ctr"][frame],
            old["corrected_joints"][frame],
        )
    anchor_times = inputs.video_timestamps[anchor_frames]
    residuals = PchipInterpolator(
        anchor_times, anchor_residuals, axis=0
    )(inputs.video_timestamps)
    candidate_ctr, candidate_joints = apply_residuals(
        old["corrected_ctr"], old["corrected_joints"], residuals
    )

    anchor_errors = {}
    for frame in frame_indices:
        local_index = local_by_frame[int(frame)]
        expected_ctr = (
            keyframes["accepted_ctr"][local_index]
            if int(frame) in accepted_frames
            else old["corrected_ctr"][frame]
        )
        expected_joints = (
            keyframes["accepted_joints"][local_index]
            if int(frame) in accepted_frames
            else old["corrected_joints"][frame]
        )
        expected_matrix = ctr_to_matrix(expected_ctr)
        candidate_matrix = ctr_to_matrix(candidate_ctr[frame])
        anchor_errors[str(int(frame))] = {
            "translation_m": float(
                np.linalg.norm(
                    expected_matrix[:3, 3] - candidate_matrix[:3, 3]
                )
            ),
            "rotation_deg": math.degrees(
                float(
                    Rotation.from_matrix(
                        expected_matrix[:3, :3]
                        @ candidate_matrix[:3, :3].T
                    ).magnitude()
                )
            ),
            "joints_rad": float(
                np.max(np.abs(expected_joints - candidate_joints[frame]))
            ),
        }
    max_anchor_error = max(
        max(value.values()) for value in anchor_errors.values()
    )
    if max_anchor_error > 1.0e-5:
        raise RuntimeError(f"Keyframe interpolation is not exact: {anchor_errors}")

    candidate_video = inputs.paper_states_to_visual_poses(
        candidate_ctr, candidate_joints, **conversion_options
    )
    candidate_wrist = inputs.gui_wrist_angles_from_paper_residual(
        old["prior_joints"], candidate_joints
    )
    candidate_jaw = inputs.gui_jaw_angles_from_paper_residual(
        old["prior_joints"], candidate_joints
    )
    candidate_video = inputs.enforce_urdf_distal_kinematics(
        candidate_video, candidate_wrist, candidate_jaw
    )

    jaw_names = (
        "PSM1_tool_wrist_sca_ee_link_1",
        "PSM1_tool_wrist_sca_ee_link_2",
    )
    jaw_indices = tuple(inputs.link_names.index(name) for name in jaw_names)
    max_pivot_gap = 0.0
    max_axis_error = 0.0
    for frame in range(frame_count):
        left = pose_to_matrix(candidate_video[frame, jaw_indices[0]])
        right = pose_to_matrix(candidate_video[frame, jaw_indices[1]])
        max_pivot_gap = max(
            max_pivot_gap, float(np.linalg.norm(left[:3, 3] - right[:3, 3]))
        )
        cosine = float(np.clip(np.dot(left[:3, 2], right[:3, 2]), -1.0, 1.0))
        max_axis_error = max(max_axis_error, math.degrees(math.acos(cosine)))
    jaw_gate = max_pivot_gap < 1.0e-7 and max_axis_error < 1.0e-3
    if not jaw_gate:
        raise RuntimeError("Full-sequence URDF jaw gate failed")

    motion = correction_motion(inputs.video_timestamps, residuals)
    motion_gate = bool(
        motion["max_linear_speed_mm_s"] <= args.max_linear_speed_mm_s
        and motion["max_angular_speed_deg_s"] <= args.max_angular_speed_deg_s
        and motion["max_linear_accel_mm_s2"] <= args.max_linear_accel_mm_s2
        and motion["max_angular_accel_deg_s2"] <= args.max_angular_accel_deg_s2
    )
    if not motion_gate:
        raise RuntimeError(f"Full-sequence correction motion gate failed: {motion}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "tracking_states_depth_then_visual_full.npz",
        video_timestamps=inputs.video_timestamps,
        corrected_ctr=candidate_ctr,
        corrected_joints=candidate_joints,
        old_corrected_ctr=old["corrected_ctr"],
        old_corrected_joints=old["corrected_joints"],
        interpolated_residuals=residuals,
        anchor_frames=np.asarray(anchor_frames, dtype=np.int64),
        anchor_residuals=anchor_residuals,
        accepted_frames=np.asarray(accepted_frames, dtype=np.int64),
        rejected_frames=np.asarray(rejected_frames, dtype=np.int64),
    )
    np.savez_compressed(
        args.output_dir / "visual_poses_depth_then_visual_full.npz",
        timestamps=inputs.video_timestamps,
        link_names=np.asarray(inputs.link_names),
        poses_rect_camera_xyz_xyzw=candidate_video,
        gui_wrist_angles=candidate_wrist,
        gui_jaw_angles=candidate_jaw,
    )
    save_runtime_driver(args.runtime_driver, inputs, candidate_video)
    report = {
        "method": (
            "PCHIP interpolation on measured video timestamps of accepted "
            "candidate-minus-old-corrected camera pose/joint residuals; all "
            "rejected keyframes and sequence endpoints are exact zero anchors"
        ),
        "accepted_frames": accepted_frames,
        "rejected_zero_anchor_frames": rejected_frames,
        "all_anchor_frames": anchor_frames,
        "anchor_errors": anchor_errors,
        "baseline_reconstruction": {
            "video_translation_error_m": baseline_video_translation_error,
            "video_rotation_error_deg": baseline_video_rotation_error,
            "runtime_translation_error_m": baseline_runtime_translation_error,
            "runtime_rotation_error_deg": baseline_runtime_rotation_error,
        },
        "correction_motion": motion,
        "correction_motion_limits": {
            "max_linear_speed_mm_s": args.max_linear_speed_mm_s,
            "max_angular_speed_deg_s": args.max_angular_speed_deg_s,
            "max_linear_accel_mm_s2": args.max_linear_accel_mm_s2,
            "max_angular_accel_deg_s2": args.max_angular_accel_deg_s2,
        },
        "correction_motion_gate_passed": motion_gate,
        "jaw_kinematics": {
            "shared_pivot_gap_max_m": max_pivot_gap,
            "hinge_axis_error_max_deg": max_axis_error,
            "passed": jaw_gate,
        },
        "frozen_input_sha256": frozen_hashes,
        "runtime_driver": str(args.runtime_driver),
        "runtime_driver_sha256": sha256(args.runtime_driver),
        "runtime_state_count": len(inputs.joint_timestamps),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
