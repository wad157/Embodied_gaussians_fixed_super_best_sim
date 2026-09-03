#!/usr/bin/env python3
"""Validate the raw-only PSM asset chain consumed by the SUPER GUI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from embodied_environments.super_embodied.psm_lnd_kinematics import (  # noqa: E402
    PSMLNDKinematics,
)


def parse_args() -> argparse.Namespace:
    raw_root = (
        REPO_ROOT
        / "data/super/psm_raw_kinematics_v1（纯机器人学版本）"
    )
    parser = argparse.ArgumentParser(
        description="Validate raw LND -> fresh CAD -> current GUI coordinates."
    )
    parser.add_argument("--raw-root", type=Path, default=raw_root)
    parser.add_argument(
        "--gui-root",
        type=Path,
        default=None,
        help="GUI asset directory. Defaults to RAW_ROOT/gui_v1.",
    )
    parser.add_argument(
        "--registration",
        type=Path,
        default=None,
        help="Registration report. Defaults to GUI_ROOT/registration_report.json.",
    )
    parser.add_argument(
        "--driver",
        type=Path,
        default=None,
        help="Pose driver. Defaults to GUI_ROOT/psm_raw_gui_pose_driver.npz.",
    )
    parser.add_argument(
        "--gaussians",
        type=Path,
        default=None,
        help="Surface asset. Defaults to GUI_ROOT/psm_raw_surface_gaussians.npz.",
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
        default=None,
        help="Validation report. Defaults to GUI_ROOT/gui_validation_report.json.",
    )
    return parser.parse_args()


def poses_to_matrices(poses: np.ndarray) -> np.ndarray:
    matrices = np.repeat(
        np.eye(4, dtype=np.float64)[None, None],
        poses.shape[0],
        axis=0,
    )
    matrices = np.repeat(matrices, poses.shape[1], axis=1)
    matrices[..., :3, :3] = Rotation.from_quat(
        poses[..., 3:].reshape(-1, 4)
    ).as_matrix().reshape(*poses.shape[:2], 3, 3)
    matrices[..., :3, 3] = poses[..., :3]
    return matrices


def rotation_error_deg(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    relative = (
        actual[..., :3, :3]
        @ np.swapaxes(expected[..., :3, :3], -1, -2)
    )
    return np.degrees(
        Rotation.from_matrix(relative.reshape(-1, 3, 3))
        .magnitude()
        .reshape(relative.shape[:-2])
    )


def main() -> None:
    args = parse_args()
    gui_root = (
        args.gui_root
        if args.gui_root is not None
        else args.raw_root / "gui_v1"
    )
    raw_model_path = args.raw_root / "model.json"
    raw_kinematics_path = args.raw_root / "kinematics.npz"
    registration_path = (
        args.registration
        if args.registration is not None
        else gui_root / "registration_report.json"
    )
    driver_path = (
        args.driver
        if args.driver is not None
        else gui_root / "psm_raw_gui_pose_driver.npz"
    )
    gaussian_path = (
        args.gaussians
        if args.gaussians is not None
        else gui_root / "psm_raw_surface_gaussians.npz"
    )
    output_path = (
        args.output
        if args.output is not None
        else gui_root / "gui_validation_report.json"
    )
    for path in (
        raw_model_path,
        raw_kinematics_path,
        registration_path,
        driver_path,
        gaussian_path,
        args.table_frame,
        args.camera_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    registration = json.loads(
        registration_path.read_text(encoding="utf-8")
    )
    if registration.get("passed") is not True:
        raise RuntimeError("Raw GUI registration report has not passed")
    if registration.get("uses_historical_psm_derivatives") is not False:
        raise RuntimeError("Raw GUI registration does not exclude old derivatives")

    X_gui_camera = np.asarray(
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
                X_gui_camera @ X_opencv_to_blender
                - np.asarray(
                    camera_manifest["stereo_left"]["X_WC"],
                    dtype=np.float64,
                )
            )
        )
    )

    with np.load(raw_kinematics_path, allow_pickle=False) as raw:
        raw_timestamps = raw["joint_timestamps_s"].astype(np.float64)
        raw_q7 = raw["q7"].astype(np.float64)
        raw_T_rectified_left_lnd = raw[
            "T_rectified_left_camera_lnd_link"
        ].astype(np.float64)
    with np.load(driver_path, allow_pickle=False) as driver:
        driver_timestamps = driver["timestamps"].astype(np.float64)
        driver_q7 = driver["q7"].astype(np.float64)
        driver_link_names = driver["link_names"].tolist()
        lnd_link_ids = driver["lnd_link_ids"].astype(np.int64)
        link_offsets = driver["T_lndlink_urdf_link"].astype(np.float64)
        saved_X_gui_camera = driver[
            "X_gui_world_rectified_left_camera"
        ].astype(np.float64)
        driver_matrices = poses_to_matrices(
            driver["poses_gui_world_xyz_xyzw"].astype(np.float64)
        )

    selected = np.unique(
        np.linspace(0, len(driver_timestamps) - 1, 17).round().astype(int)
    )
    expected_matrices = np.einsum(
        "ij,nljk->nlik",
        X_gui_camera,
        raw_T_rectified_left_lnd[selected][:, lnd_link_ids]
        @ link_offsets[None],
    )
    selected_actual = driver_matrices[selected]
    chain_translation_error_m = float(
        np.linalg.norm(
            selected_actual[..., :3, 3]
            - expected_matrices[..., :3, 3],
            axis=-1,
        ).max()
    )
    chain_rotation_error_deg = float(
        rotation_error_deg(selected_actual, expected_matrices).max()
    )

    runtime_kinematics = PSMLNDKinematics.from_files(
        raw_model_path,
        registration_path,
        args.table_frame,
        driver_link_names,
    )
    runtime_matrices = np.stack(
        [
            runtime_kinematics.visual_matrices_table(driver_q7[index])
            for index in selected
        ]
    )
    runtime_translation_error_m = float(
        np.linalg.norm(
            runtime_matrices[..., :3, 3]
            - selected_actual[..., :3, 3],
            axis=-1,
        ).max()
    )
    runtime_rotation_error_deg = float(
        rotation_error_deg(runtime_matrices, selected_actual).max()
    )

    jaw_link_names = (
        "PSM1_tool_wrist_sca_ee_link_1",
        "PSM1_tool_wrist_sca_ee_link_2",
    )
    missing_jaw_links = [
        name for name in jaw_link_names if name not in driver_link_names
    ]
    if missing_jaw_links:
        raise KeyError(
            f"Pose driver is missing jaw links: {missing_jaw_links}"
        )
    jaw_indices = [driver_link_names.index(name) for name in jaw_link_names]
    jaw_hinge_origin_separation_m = float(
        np.linalg.norm(
            driver_matrices[:, jaw_indices[0], :3, 3]
            - driver_matrices[:, jaw_indices[1], :3, 3],
            axis=-1,
        ).max()
    )
    jaw_axis_local = np.asarray(
        registration.get("manual_offset_conventions", {})
        .get("jaw", {})
        .get("axis_xyz", [0.0, 0.0, 1.0]),
        dtype=np.float64,
    )
    jaw_axis_local /= np.linalg.norm(jaw_axis_local)
    jaw_axes_world = [
        np.einsum(
            "nij,j->ni",
            driver_matrices[:, jaw_index, :3, :3],
            jaw_axis_local,
        )
        for jaw_index in jaw_indices
    ]
    jaw_hinge_axis_abs_dot_min = float(
        np.abs(
            np.einsum("ni,ni->n", jaw_axes_world[0], jaw_axes_world[1])
        ).min()
    )

    with np.load(gaussian_path, allow_pickle=False) as gaussians:
        gaussian_link_names = gaussians["link_names"].tolist()
        gaussian_link_ids = gaussians["link_ids"].astype(np.int64)
        gaussian_count = int(len(gaussians["means"]))
        gaussian_finite = bool(
            all(
                np.isfinite(gaussians[name]).all()
                for name in (
                    "means",
                    "quats_wxyz",
                    "scales",
                    "opacities",
                    "colors",
                )
            )
        )

    gates = {
        "registration_report_passed": registration.get("passed") is True,
        "historical_psm_derivatives_excluded": (
            registration.get("uses_historical_psm_derivatives") is False
        ),
        "timestamps_are_raw_exact": bool(
            np.array_equal(driver_timestamps, raw_timestamps)
        ),
        "q7_is_raw_exact": bool(np.array_equal(driver_q7, raw_q7)),
        "saved_gui_transform_matches_current": bool(
            np.max(np.abs(saved_X_gui_camera - X_gui_camera)) <= 1.0e-12
        ),
        "gui_camera_manifest_matches": camera_manifest_error <= 1.0e-12,
        "coordinate_chain_translation": chain_translation_error_m <= 1.0e-6,
        "coordinate_chain_rotation": chain_rotation_error_deg <= 1.0e-4,
        "runtime_fk_translation": runtime_translation_error_m <= 1.0e-6,
        "runtime_fk_rotation": runtime_rotation_error_deg <= 1.0e-4,
        "gaussian_links_match_driver": (
            gaussian_link_names == driver_link_names
        ),
        "gaussians_are_finite": gaussian_finite,
        "gaussian_link_ids_valid": bool(
            gaussian_link_ids.min() >= 0
            and gaussian_link_ids.max() < len(gaussian_link_names)
        ),
    }
    jaw_geometry_gate_required = (
        registration.get("version") == "raw_p420006"
    )
    if jaw_geometry_gate_required:
        gates.update(
            {
                "jaw_hinge_origins_coincident": (
                    jaw_hinge_origin_separation_m <= 1.0e-6
                ),
                "jaw_hinge_axes_parallel": (
                    jaw_hinge_axis_abs_dot_min >= 1.0 - 1.0e-8
                ),
            }
        )
    report = {
        "schema": "super_psm_raw_gui_validation_v1",
        "passed": all(gates.values()),
        "gates": gates,
        "jaw_geometry_gate_required": jaw_geometry_gate_required,
        "states": len(driver_timestamps),
        "links": len(driver_link_names),
        "gaussians": gaussian_count,
        "selected_state_indices": selected.tolist(),
        "errors": {
            "gui_camera_manifest_max_abs": camera_manifest_error,
            "coordinate_chain_translation_max_m": (
                chain_translation_error_m
            ),
            "coordinate_chain_rotation_max_deg": (
                chain_rotation_error_deg
            ),
            "runtime_fk_translation_max_m": runtime_translation_error_m,
            "runtime_fk_rotation_max_deg": runtime_rotation_error_deg,
            "jaw_hinge_origin_separation_max_m": (
                jaw_hinge_origin_separation_m
            ),
            "jaw_hinge_axis_abs_dot_min": jaw_hinge_axis_abs_dot_min,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("Raw PSM GUI validation failed")


if __name__ == "__main__":
    main()
