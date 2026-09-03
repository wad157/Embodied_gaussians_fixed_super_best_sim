#!/usr/bin/env python3
"""Validate GUI roll/jaw offsets on strict and corrected PSM pose drivers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from examples.embodied_environments.super_embodied.psm_lnd_kinematics import (  # noqa: E402
    PSMLNDKinematics,
)
from examples.embodied_environments.super_embodied.super_embodied import (  # noqa: E402
    PSM_LND_MODEL_PATH,
    PSM_LND_POSE_REPORT_PATH,
    TABLE_FRAME_PATH,
    apply_psm_pose_driver_joint_offsets,
    load_psm_lnd_pose_driver,
)


DRIVER = REPO / "data/super/psm_tracking/psm_part_corrected_pose_driver.npz"
ROBOTS = REPO / "data/super/grasp5_offline_demo/robots.json"


def pose_matrices(poses: np.ndarray) -> np.ndarray:
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None], len(poses), axis=0)
    matrices[:, :3, :3] = Rotation.from_quat(poses[:, 3:]).as_matrix()
    matrices[:, :3, 3] = poses[:, :3]
    return matrices


def pose_error(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    actual_matrices = pose_matrices(actual)
    expected_matrices = pose_matrices(expected)
    translation_mm = float(
        np.max(
            np.linalg.norm(
                actual_matrices[:, :3, 3] - expected_matrices[:, :3, 3], axis=1
            )
        )
        * 1000.0
    )
    rotation_deg = float(
        np.max(
            Rotation.from_matrix(
                actual_matrices[:, :3, :3]
                @ expected_matrices[:, :3, :3].transpose(0, 2, 1)
            ).magnitude()
        )
        * 180.0
        / np.pi
    )
    return translation_mm, rotation_deg


def main() -> None:
    _, link_names, corrected_poses = load_psm_lnd_pose_driver(
        np.asarray(
            json.loads(TABLE_FRAME_PATH.read_text(encoding="utf-8"))["X_table_camera"],
            dtype=np.float64,
        ),
        path=DRIVER,
    )
    kinematics = PSMLNDKinematics.from_files(
        PSM_LND_MODEL_PATH,
        PSM_LND_POSE_REPORT_PATH,
        TABLE_FRAME_PATH,
        link_names,
    )
    robot_data = json.loads(ROBOTS.read_text(encoding="utf-8"))["PSM1"]
    q7_states = np.asarray([state["q"] for state in robot_data["states"]])
    state_index = 1000
    q7 = q7_states[state_index]
    offsets = np.zeros(7, dtype=np.float64)
    offsets[3] = np.deg2rad(-27.0)
    offsets[6] = np.deg2rad(17.0)

    corrected = corrected_poses[state_index]
    zero = apply_psm_pose_driver_joint_offsets(
        corrected, q7, kinematics, np.zeros(7, dtype=np.float64)
    )
    zero_translation_mm, zero_rotation_deg = pose_error(zero, corrected)

    roll_offsets = np.zeros(7, dtype=np.float64)
    roll_offsets[3] = offsets[3]
    corrected_roll = apply_psm_pose_driver_joint_offsets(
        corrected, q7, kinematics, roll_offsets
    )
    corrected_matrices = pose_matrices(corrected)
    roll_matrices = pose_matrices(corrected_roll)
    main_translation_mm, main_rotation_deg = pose_error(
        corrected_roll[:1], corrected[:1]
    )
    distal_distance_before = np.linalg.norm(
        corrected_matrices[1:, None, :3, 3]
        - corrected_matrices[None, 1:, :3, 3],
        axis=-1,
    )
    distal_distance_after = np.linalg.norm(
        roll_matrices[1:, None, :3, 3] - roll_matrices[None, 1:, :3, 3],
        axis=-1,
    )
    roll_rigidity_error_mm = float(
        np.max(np.abs(distal_distance_after - distal_distance_before)) * 1000.0
    )
    roll_delta_deg = np.degrees(
        Rotation.from_matrix(
            roll_matrices[1:, :3, :3]
            @ corrected_matrices[1:, :3, :3].transpose(0, 2, 1)
        ).magnitude()
    )

    jaw_offsets = np.zeros(7, dtype=np.float64)
    jaw_offsets[6] = offsets[6]
    corrected_jaw = apply_psm_pose_driver_joint_offsets(
        corrected, q7, kinematics, jaw_offsets
    )
    non_jaw_translation_mm, non_jaw_rotation_deg = pose_error(
        corrected_jaw[:5], corrected[:5]
    )
    jaw_matrices = pose_matrices(corrected_jaw)
    jaw_delta_deg = np.degrees(
        Rotation.from_matrix(
            jaw_matrices[5:, :3, :3]
            @ corrected_matrices[5:, :3, :3].transpose(0, 2, 1)
        ).magnitude()
    )
    parent_index = link_names.index("PSM1_tool_wrist_sca_shaft_link")
    jaw_indices = [
        link_names.index("PSM1_tool_wrist_sca_ee_link_1"),
        link_names.index("PSM1_tool_wrist_sca_ee_link_2"),
    ]
    jaw_pivot_gap_mm = float(
        np.max(
            np.linalg.norm(
                jaw_matrices[jaw_indices, :3, 3]
                - jaw_matrices[parent_index, :3, 3],
                axis=1,
            )
        )
        * 1000.0
    )
    parent_axis = jaw_matrices[parent_index, :3, 2]
    jaw_axis_parallel_error_deg = np.degrees(
        np.arccos(
            np.clip(
                np.abs(jaw_matrices[jaw_indices, :3, 2] @ parent_axis),
                0.0,
                1.0,
            )
        )
    )

    report = {
        "state_index": state_index,
        "requested_roll_offset_deg": -27.0,
        "requested_jaw_opening_offset_deg": 17.0,
        "corrected_zero_offset_max_translation_mm": zero_translation_mm,
        "corrected_zero_offset_max_rotation_deg": zero_rotation_deg,
        "roll_keeps_main_max_translation_mm": main_translation_mm,
        "roll_keeps_main_max_rotation_deg": main_rotation_deg,
        "roll_distal_rigidity_error_mm": roll_rigidity_error_mm,
        "roll_distal_rotation_deg": roll_delta_deg.tolist(),
        "jaw_keeps_non_jaw_max_translation_mm": non_jaw_translation_mm,
        "jaw_keeps_non_jaw_max_rotation_deg": non_jaw_rotation_deg,
        "jaw_half_rotation_left_right_deg": jaw_delta_deg.tolist(),
        "jaw_shared_pivot_max_gap_mm": jaw_pivot_gap_mm,
        "jaw_axis_parallel_error_left_right_deg": (
            jaw_axis_parallel_error_deg.tolist()
        ),
    }
    print(json.dumps(report, indent=2))

    assert zero_translation_mm == 0.0
    assert zero_rotation_deg < 1.0e-6
    assert main_translation_mm == 0.0
    assert main_rotation_deg < 1.0e-6
    assert roll_rigidity_error_mm < 1.0e-3
    assert np.allclose(roll_delta_deg, 27.0, atol=1.0e-3)
    assert non_jaw_translation_mm == 0.0
    assert non_jaw_rotation_deg < 1.0e-6
    assert np.allclose(jaw_delta_deg, 8.5, atol=1.0e-3)
    assert jaw_pivot_gap_mm < 1.0e-3
    assert np.all(jaw_axis_parallel_error_deg < 1.0e-3)


if __name__ == "__main__":
    main()
