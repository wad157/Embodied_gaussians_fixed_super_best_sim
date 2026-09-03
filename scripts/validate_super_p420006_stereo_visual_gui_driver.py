#!/usr/bin/env python3
"""Validate the full raw-to-stereo-visual P420006 GUI pose chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import build_super_raw_psm_gui_driver as raw_builder


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = (
    REPO_ROOT / "data/super/psm_raw_kinematics_v1（纯机器人学版本）"
)
P420_ROOT = RAW_ROOT / "gui_p420006_v1"
VISUAL_ROOT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_p420006_stereo_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the 5458-state stereo-visual P420006 GUI driver."
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--p420-root", type=Path, default=P420_ROOT)
    parser.add_argument("--visual-root", type=Path, default=VISUAL_ROOT)
    parser.add_argument(
        "--driver",
        type=Path,
        default=(
            VISUAL_ROOT
            / "gui_v1/psm_p420006_stereo_visual_gui_pose_driver.npz"
        ),
    )
    parser.add_argument(
        "--registration",
        type=Path,
        default=VISUAL_ROOT / "gui_v1/registration_report.json",
    )
    parser.add_argument(
        "--table-frame",
        type=Path,
        default=REPO_ROOT / "data/super/table_frame.json",
    )
    parser.add_argument(
        "--camera-manifest",
        type=Path,
        default=REPO_ROOT
        / "data/super/grasp5_offline_demo/cameras.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=VISUAL_ROOT / "gui_v1/gui_validation_report.json",
    )
    return parser.parse_args()


def poses_to_matrices(poses: np.ndarray) -> np.ndarray:
    matrices = np.broadcast_to(
        np.eye(4, dtype=np.float64),
        (*poses.shape[:-1], 4, 4),
    ).copy()
    matrices[..., :3, 3] = poses[..., :3]
    matrices[..., :3, :3] = Rotation.from_quat(
        poses[..., 3:].reshape(-1, 4)
    ).as_matrix().reshape(*poses.shape[:-1], 3, 3)
    return matrices


def rotation_error_deg(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    relative = actual @ np.swapaxes(expected, -1, -2)
    return np.degrees(
        Rotation.from_matrix(relative.reshape(-1, 3, 3))
        .magnitude()
        .reshape(relative.shape[:-2])
    )


def main() -> None:
    args = parse_args()
    raw_model_path = args.raw_root / "model.json"
    raw_kinematics_path = args.raw_root / "kinematics.npz"
    p420_driver_path = (
        args.p420_root / "psm_p420006_gui_pose_driver.npz"
    )
    optimization_report_path = args.visual_root / "optimization_report.json"
    corrections_path = args.visual_root / "keyframe_corrections.npz"
    for path in (
        raw_model_path,
        raw_kinematics_path,
        p420_driver_path,
        optimization_report_path,
        corrections_path,
        args.driver,
        args.registration,
        args.table_frame,
        args.camera_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    registration = json.loads(
        args.registration.read_text(encoding="utf-8")
    )
    optimization = json.loads(
        optimization_report_path.read_text(encoding="utf-8")
    )
    raw_model = json.loads(raw_model_path.read_text(encoding="utf-8"))
    X_gui_left_current = np.asarray(
        json.loads(args.table_frame.read_text(encoding="utf-8"))[
            "X_table_camera"
        ],
        dtype=np.float64,
    )
    camera_manifest = json.loads(
        args.camera_manifest.read_text(encoding="utf-8")
    )
    X_opencv_to_blender = np.diag([1.0, -1.0, -1.0, 1.0])
    camera_manifest_error = float(
        np.max(
            np.abs(
                X_gui_left_current @ X_opencv_to_blender
                - np.asarray(
                    camera_manifest["stereo_left"]["X_WC"],
                    dtype=np.float64,
                )
            )
        )
    )

    with np.load(raw_kinematics_path, allow_pickle=False) as raw:
        raw_timestamps_s = raw["joint_timestamps_s"].astype(np.float64)
        raw_timestamps_ros_ns = raw[
            "joint_timestamps_ros_ns"
        ].astype(np.int64)
        raw_q7 = raw["q7"].astype(np.float64)
        raw_T_left_lnd = raw[
            "T_rectified_left_camera_lnd_link"
        ].astype(np.float64)
    with np.load(p420_driver_path, allow_pickle=False) as source:
        source_link_names = source["link_names"].tolist()
        source_lnd_link_ids = source["lnd_link_ids"].astype(np.int64)
        source_link_offsets = source[
            "T_lndlink_urdf_link"
        ].astype(np.float64)
        source_X_gui_left = source[
            "X_gui_world_rectified_left_camera"
        ].astype(np.float64)
        source_X_gui_base = source[
            "X_gui_world_urdf_base"
        ].astype(np.float64)
    with np.load(args.driver, allow_pickle=False) as driver:
        driver_schema = str(driver["schema"].item())
        driver_timestamps_s = driver["timestamps"].astype(np.float64)
        driver_timestamps_ros_ns = driver[
            "timestamps_ros_ns"
        ].astype(np.int64)
        driver_q7 = driver["q7"].astype(np.float64)
        driver_q7_raw = driver["q7_raw"].astype(np.float64)
        driver_link_names = driver["link_names"].tolist()
        driver_lnd_link_ids = driver["lnd_link_ids"].astype(np.int64)
        driver_link_offsets = driver[
            "T_lndlink_urdf_link"
        ].astype(np.float64)
        driver_X_gui_left = driver[
            "X_gui_world_rectified_left_camera"
        ].astype(np.float64)
        driver_X_gui_base = driver[
            "X_gui_world_urdf_base"
        ].astype(np.float64)
        T_right_left = driver[
            "T_rectified_right_camera_rectified_left_camera"
        ].astype(np.float64)
        global_rotvec = driver[
            "visual_global_rotvec_camera_rad"
        ].astype(np.float64)
        global_translation = driver[
            "visual_global_translation_camera_m"
        ].astype(np.float64)
        global_q = driver[
            "visual_global_q3_q6_rad"
        ].astype(np.float64)
        registration_rotation = driver[
            "visual_global_registration_rotation_camera"
        ].astype(np.float64)
        local_rotvec = driver[
            "visual_local_rotvec_camera_rad"
        ].astype(np.float64)
        local_translation = driver[
            "visual_local_translation_camera_m"
        ].astype(np.float64)
        local_q = driver[
            "visual_local_q3_q6_rad"
        ].astype(np.float64)
        jaw_scale = float(driver["visual_jaw_affine_scale"])
        jaw_offset = float(driver["visual_jaw_affine_offset_rad"])
        saved_poses = driver[
            "poses_gui_world_xyz_xyzw"
        ].astype(np.float64)

    state_count = len(raw_timestamps_s)
    link_count = len(driver_link_names)
    shapes_valid = bool(
        driver_q7.shape == (state_count, 7)
        and driver_q7_raw.shape == (state_count, 7)
        and saved_poses.shape == (state_count, link_count, 7)
        and local_rotvec.shape == (state_count, 3)
        and local_translation.shape == (state_count, 3)
        and local_q.shape == (state_count, 4)
    )
    if not shapes_valid:
        raise RuntimeError("Visual driver array shapes are inconsistent")

    expected_q7 = raw_q7.copy()
    expected_q7[:, 3:6] += global_q[None, :3] + local_q[:, :3]
    expected_q7[:, 6] = jaw_scale * raw_q7[:, 6] + jaw_offset
    q7_reconstruction_error = float(
        np.max(np.abs(driver_q7 - expected_q7))
    )
    jaw_affine_error = float(
        np.max(
            np.abs(
                driver_q7[:, 6]
                - (jaw_scale * raw_q7[:, 6] + jaw_offset)
            )
        )
    )

    expected_left = np.empty(
        (state_count, link_count, 4, 4),
        dtype=np.float64,
    )
    global_rotation = Rotation.from_rotvec(global_rotvec).as_matrix()
    local_rotations = Rotation.from_rotvec(local_rotvec).as_matrix()
    delta_rotations = (
        local_rotations
        @ global_rotation[None]
        @ registration_rotation[None]
    )
    T_left_base = raw_T_left_lnd[:, 0]
    for state_index in range(state_count):
        fk = raw_builder.lnd_fk(
            raw_model["lnd"],
            driver_q7[state_index],
        )
        links = (
            T_left_base[state_index][None]
            @ fk[driver_lnd_link_ids]
            @ driver_link_offsets
        )
        anchor = (
            T_left_base[state_index] @ fk[6]
        )[:3, 3]
        rotation = delta_rotations[state_index]
        links[:, :3, :3] = rotation[None] @ links[:, :3, :3]
        links[:, :3, 3] = (
            (links[:, :3, 3] - anchor[None]) @ rotation.T
            + anchor[None]
            + global_translation[None]
            + local_translation[state_index][None]
        )
        expected_left[state_index] = links
    expected_gui = driver_X_gui_left[None, None] @ expected_left
    actual_gui = poses_to_matrices(saved_poses)
    chain_translation_error_m = float(
        np.linalg.norm(
            actual_gui[..., :3, 3] - expected_gui[..., :3, 3],
            axis=-1,
        ).max()
    )
    chain_rotation_error_deg = float(
        rotation_error_deg(
            actual_gui[..., :3, :3],
            expected_gui[..., :3, :3],
        ).max()
    )

    jaw_indices = [
        driver_link_names.index("PSM1_tool_wrist_sca_ee_link_1"),
        driver_link_names.index("PSM1_tool_wrist_sca_ee_link_2"),
    ]
    jaw_hinge_origin_separation_m = float(
        np.linalg.norm(
            expected_left[:, jaw_indices[0], :3, 3]
            - expected_left[:, jaw_indices[1], :3, 3],
            axis=1,
        ).max()
    )
    jaw_hinge_axis_abs_dot_min = float(
        np.abs(
            np.einsum(
                "ni,ni->n",
                expected_left[:, jaw_indices[0], :3, 0],
                expected_left[:, jaw_indices[1], :3, 0],
            )
        ).min()
    )

    dt = np.diff(raw_timestamps_s)
    local_rotvec_speed = np.max(
        np.abs(np.diff(local_rotvec, axis=0) / dt[:, None]),
        axis=0,
    )
    local_translation_speed = np.max(
        np.abs(np.diff(local_translation, axis=0) / dt[:, None]),
        axis=0,
    )
    local_wrist_speed = np.max(
        np.abs(np.diff(local_q[:, :3], axis=0) / dt[:, None]),
        axis=0,
    )
    jaw_increment_product = (
        np.diff(driver_q7[:, 6]) * np.diff(raw_q7[:, 6])
    )
    stereo_rotation_error = float(
        np.max(np.abs(T_right_left[:3, :3] - np.eye(3)))
    )
    stereo_baseline_m = float(np.linalg.norm(T_right_left[:3, 3]))

    with np.load(corrections_path, allow_pickle=False) as corrections:
        global_bounds = corrections[
            "global_parameter_bounds"
        ].astype(np.float64)
        local_bounds = corrections[
            "local_parameter_bounds"
        ].astype(np.float64)
    global_values = np.r_[
        global_rotvec,
        global_translation,
        global_q,
    ]
    local_values = np.c_[local_rotvec, local_translation, local_q]
    global_bound_fraction = float(
        np.max(np.abs(global_values / global_bounds))
    )
    local_bound_fraction = float(
        np.max(np.abs(local_values / local_bounds[None]))
    )
    high_confidence_jaw_fit_error = float(
        registration["correction_statistics"][
            "jaw_affine_calibration"
        ]["maximum_high_confidence_fit_residual_deg"]
    )

    gates = {
        "registration_report_passed": registration.get("passed") is True,
        "optimization_report_passed": optimization.get("passed") is True,
        "driver_schema": (
            driver_schema
            == "super_psm_raw_p420006_stereo_visual_gui_pose_driver_v1"
        ),
        "array_shapes": shapes_valid,
        "all_values_finite": bool(
            np.isfinite(saved_poses).all()
            and np.isfinite(driver_q7).all()
            and np.isfinite(local_values).all()
        ),
        "timestamps_are_all_5458_raw_states": bool(
            state_count == 5458
            and np.array_equal(driver_timestamps_s, raw_timestamps_s)
            and np.array_equal(
                driver_timestamps_ros_ns, raw_timestamps_ros_ns
            )
        ),
        "raw_q7_preserved_as_provenance": bool(
            np.array_equal(driver_q7_raw, raw_q7)
        ),
        "raw_arm_q0_q2_unchanged": bool(
            np.array_equal(driver_q7[:, :3], raw_q7[:, :3])
        ),
        "corrected_q7_reconstructs": q7_reconstruction_error <= 1.0e-12,
        "jaw_affine_exact": jaw_affine_error <= 1.0e-12,
        "jaw_affine_monotonic": bool(
            jaw_scale > 0.0
            and np.min(jaw_increment_product) >= -1.0e-14
        ),
        "jaw_range_physical": bool(
            driver_q7[:, 6].min() >= 0.0
            and driver_q7[:, 6].max() <= 1.6
        ),
        "high_confidence_jaw_fit_residual": (
            high_confidence_jaw_fit_error <= 5.0
        ),
        "source_p420006_adapters_unchanged": bool(
            driver_link_names == source_link_names
            and np.array_equal(driver_lnd_link_ids, source_lnd_link_ids)
            and np.array_equal(driver_link_offsets, source_link_offsets)
        ),
        "gui_transform_is_current": bool(
            np.max(np.abs(driver_X_gui_left - X_gui_left_current))
            <= 1.0e-12
            and np.array_equal(driver_X_gui_left, source_X_gui_left)
            and np.array_equal(driver_X_gui_base, source_X_gui_base)
        ),
        "gui_camera_manifest_matches": camera_manifest_error <= 1.0e-12,
        "full_chain_translation": chain_translation_error_m <= 1.0e-6,
        "full_chain_rotation": chain_rotation_error_deg <= 1.0e-4,
        "global_correction_not_clipped": global_bound_fraction <= 0.995,
        "local_correction_not_clipped": local_bound_fraction <= 0.98,
        "local_rotation_is_smooth": bool(
            np.max(local_rotvec_speed) <= 0.02
        ),
        "local_translation_is_smooth": bool(
            np.max(local_translation_speed) <= 0.001
        ),
        "local_wrist_is_smooth": bool(
            np.max(local_wrist_speed) <= 0.05
        ),
        "jaw_hinge_origins_coincident": (
            jaw_hinge_origin_separation_m <= 1.0e-6
        ),
        "jaw_hinge_axes_parallel": (
            jaw_hinge_axis_abs_dot_min >= 1.0 - 1.0e-8
        ),
        "stereo_extrinsic_is_fixed_rectified_pair": bool(
            stereo_rotation_error <= 1.0e-12
            and 0.005 <= stereo_baseline_m <= 0.006
        ),
    }
    report = {
        "schema": "super_psm_p420006_stereo_visual_gui_validation_v1",
        "passed": all(gates.values()),
        "gates": gates,
        "states": state_count,
        "links": link_count,
        "errors": {
            "q7_reconstruction_max_abs": q7_reconstruction_error,
            "jaw_affine_max_abs": jaw_affine_error,
            "gui_camera_manifest_max_abs": camera_manifest_error,
            "full_chain_translation_max_m": chain_translation_error_m,
            "full_chain_rotation_max_deg": chain_rotation_error_deg,
            "jaw_hinge_origin_separation_max_m": (
                jaw_hinge_origin_separation_m
            ),
            "jaw_hinge_axis_abs_dot_min": jaw_hinge_axis_abs_dot_min,
            "global_boundary_fraction": global_bound_fraction,
            "local_boundary_fraction": local_bound_fraction,
            "high_confidence_jaw_fit_residual_deg": (
                high_confidence_jaw_fit_error
            ),
        },
        "smoothness": {
            "local_rotation_component_max_rad_s": (
                local_rotvec_speed.tolist()
            ),
            "local_translation_component_max_m_s": (
                local_translation_speed.tolist()
            ),
            "local_wrist_component_max_rad_s": (
                local_wrist_speed.tolist()
            ),
        },
        "jaw": {
            "affine_scale": jaw_scale,
            "affine_offset_deg": float(np.degrees(jaw_offset)),
            "corrected_range_deg": np.degrees(
                [driver_q7[:, 6].min(), driver_q7[:, 6].max()]
            ).tolist(),
        },
        "stereo": {
            "baseline_m": stereo_baseline_m,
            "rectified_rotation_max_abs": stereo_rotation_error,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise RuntimeError("Stereo-visual P420006 GUI validation failed")


if __name__ == "__main__":
    main()
