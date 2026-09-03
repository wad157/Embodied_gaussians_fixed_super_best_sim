#!/usr/bin/env python3
"""Build a GUI driver from a first-pair static stereo calibration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import build_super_raw_psm_gui_driver as raw_builder
from build_super_paper_lnd_sam2_online_gui import (
    PAPER_LINK_NAMES,
    PAPER_LND_LINK_IDS,
    PAPER_REPO,
    PAPER_T5_MESH,
    PAPER_VISIBLE_LINKS,
    RAW_ROOT,
    UPSTREAM_COMMIT,
    build_fresh_urdf_and_mimic,
    current_gui_base_transform,
    sha256,
    static_helper_registration,
)
from super_psm_tracking_common import _paper_component_transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_ROOT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/"
    "raw_paper_lnd_first_stereo_static_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lift one first-frame stereo SE(3)+q5 calibration over every "
            "raw robot state and build an isolated paper-LND GUI asset."
        )
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument(
        "--calibration-root", type=Path, default=CALIBRATION_ROOT
    )
    parser.add_argument(
        "--dvrk-root",
        type=Path,
        default=REPO_ROOT / "data/dvrk_model",
    )
    parser.add_argument("--paper-repo", type=Path, default=PAPER_REPO)
    parser.add_argument(
        "--table-frame",
        type=Path,
        default=REPO_ROOT / "data/super/table_frame.json",
    )
    parser.add_argument(
        "--camera-manifest",
        type=Path,
        default=REPO_ROOT / "data/super/grasp5_offline_demo/cameras.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=CALIBRATION_ROOT / "gui_v1"
    )
    parser.add_argument(
        "--pose-version", default="raw_paper_lnd_first_stereo_static_q5"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_model_path = args.raw_root / "model.json"
    raw_kinematics_path = args.raw_root / "kinematics.npz"
    raw_report_path = args.raw_root / "report.json"
    correction_candidates = [
        args.calibration_root / "first_stereo_se3_fixed_q5.npz",
        args.calibration_root / "first_stereo_static_correction.npz",
    ]
    present_corrections = [p for p in correction_candidates if p.is_file()]
    if len(present_corrections) != 1:
        raise RuntimeError(
            "Expected exactly one supported first-stereo correction in "
            f"{args.calibration_root}, got {present_corrections}"
        )
    correction_path = present_corrections[0]
    optimization_report_path = args.calibration_root / "report.json"
    paper_mesh_dir = args.paper_repo / "urdfs/dVRK/meshes"
    paper_mesh_paths = [
        paper_mesh_dir / name for name in PAPER_VISIBLE_LINKS.values()
    ]
    required = [
        raw_model_path,
        raw_kinematics_path,
        raw_report_path,
        correction_path,
        optimization_report_path,
        args.table_frame,
        args.camera_manifest,
        *paper_mesh_paths,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    commit = subprocess.run(
        ["git", "-C", str(args.paper_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"Unexpected upstream commit {commit}")
    raw_report = json.loads(raw_report_path.read_text(encoding="utf-8"))
    optimization_report = json.loads(
        optimization_report_path.read_text(encoding="utf-8")
    )
    if raw_report.get("passed") is not True:
        raise RuntimeError("Frozen raw-kinematics report has not passed")
    accepted_optimization_schemas = {
        "super_paper_lnd_first_stereo_static_optimization_v1",
        "super_paper_lnd_first_stereo_se3_fixed_q5_optimization_v1",
    }
    if (
        optimization_report.get("schema")
        not in accepted_optimization_schemas
        or optimization_report.get("passed") is not True
        or optimization_report.get("hard_constraints", {}).get(
            "per_frame_visual_state_count"
        )
        != 0
    ):
        raise RuntimeError("Static first-stereo calibration is not accepted")

    raw_model = json.loads(raw_model_path.read_text(encoding="utf-8"))
    table_frame = json.loads(args.table_frame.read_text(encoding="utf-8"))
    camera_manifest = json.loads(
        args.camera_manifest.read_text(encoding="utf-8")
    )
    X_table_camera = np.asarray(
        table_frame["X_table_camera"], dtype=np.float64
    )
    X_blender_from_opencv = np.diag([1.0, -1.0, -1.0, 1.0])
    manifest_camera = np.asarray(
        camera_manifest["stereo_left"]["X_WC"], dtype=np.float64
    )
    camera_error = float(
        np.max(
            np.abs(
                X_table_camera @ X_blender_from_opencv - manifest_camera
            )
        )
    )
    if camera_error > 1.0e-12:
        raise RuntimeError(
            f"Current GUI camera/table frame gate failed: {camera_error:.3e}"
        )

    with np.load(raw_kinematics_path, allow_pickle=False) as raw:
        timestamps_s = raw["joint_timestamps_s"].astype(np.float64)
        timestamps_ros_ns = raw["joint_timestamps_ros_ns"].astype(np.int64)
        q7_raw = raw["q7"].astype(np.float64)
        T_left_lnd_raw = raw[
            "T_rectified_left_camera_lnd_link"
        ].astype(np.float64)
    T_right_left = np.asarray(
        raw_model["calibration_and_static_transforms"][
            "T_rectified_right_camera_rectified_left_camera"
        ],
        dtype=np.float64,
    )
    fixed_q5_mode = (
        optimization_report["schema"]
        == "super_paper_lnd_first_stereo_se3_fixed_q5_optimization_v1"
    )
    with np.load(correction_path, allow_pickle=False) as correction_data:
        schema = correction_data["schema"].item()
        correction = correction_data["correction"].astype(np.float64)
        bounds = correction_data["parameter_bounds"].astype(np.float64)
        delta_T_left = correction_data[
            "delta_T_rectified_left_camera"
        ].astype(np.float64)
        q5_offset = float(correction_data["q5_zero_offset_rad"])
        per_frame = correction_data[
            "per_frame_visual_corrections"
        ].astype(np.float64)
    expected_correction_schema = (
        "super_paper_lnd_first_stereo_se3_fixed_q5_v1"
        if fixed_q5_mode
        else "super_paper_lnd_first_stereo_static_se3_q5_v1"
    )
    if (
        schema != expected_correction_schema
        or correction.shape != (7,)
        or bounds.shape != (7,)
        or delta_T_left.shape != (4, 4)
        or per_frame.shape != (0, 7)
        or not np.isclose(q5_offset, correction[6], atol=0.0, rtol=0.0)
    ):
        raise RuntimeError("Unexpected static correction contract")
    active_parameter_count = 6 if fixed_q5_mode else 7
    if (
        float(
            np.max(
                np.abs(
                    correction[:active_parameter_count]
                    / bounds[:active_parameter_count]
                )
            )
        )
        >= 0.95
    ):
        raise RuntimeError("Refusing a static correction at its search bound")
    if fixed_q5_mode and not np.isclose(q5_offset, 0.0, atol=0.0, rtol=0.0):
        raise RuntimeError("Fixed-q5 calibration contains a nonzero q5 offset")

    q7_corrected = q7_raw.copy()
    q7_corrected[:, 4] += q5_offset
    if not np.array_equal(q7_corrected[:, [0, 1, 2, 3, 5, 6]], q7_raw[:, [0, 1, 2, 3, 5, 6]]):
        raise RuntimeError("Static calibration modified a joint other than q5")
    q5_error = float(
        np.max(np.abs((q7_corrected[:, 4] - q7_raw[:, 4]) - q5_offset))
    )
    if q5_error > 1.0e-12:
        raise RuntimeError(f"q5 zero-offset invariance failed: {q5_error:.3e}")
    if fixed_q5_mode and not np.array_equal(q7_corrected, q7_raw):
        raise RuntimeError("Fixed-q5 calibration did not preserve raw q7 exactly")

    state_count = len(q7_raw)
    T_left_frame4_raw = np.empty((state_count, 4, 4), dtype=np.float64)
    T_left_frame4_corrected = np.empty_like(T_left_frame4_raw)
    T_left_links = np.empty(
        (state_count, len(PAPER_LINK_NAMES), 4, 4), dtype=np.float64
    )
    for state_index in range(state_count):
        # The common frame4 remains pure raw q7/LND/hand-eye.  q5 is applied
        # only inside the exact paper component FK below.
        raw_fk = raw_builder.lnd_fk(raw_model["lnd"], q7_raw[state_index])
        T_frame4_raw = T_left_lnd_raw[state_index, 0] @ raw_fk[4]
        T_frame4 = delta_T_left @ T_frame4_raw
        paper_joints = np.asarray(
            [
                q7_corrected[state_index, 4],
                q7_corrected[state_index, 5],
                0.5 * q7_corrected[state_index, 6],
                0.5 * q7_corrected[state_index, 6],
            ],
            dtype=np.float64,
        )
        components = _paper_component_transforms(paper_joints)
        T45 = components[1] @ np.linalg.inv(PAPER_T5_MESH)
        links = np.empty((len(PAPER_LINK_NAMES), 4, 4), dtype=np.float64)
        links[0] = T_frame4 @ components[0]
        links[1] = T_frame4
        links[2] = T_frame4 @ T45
        links[3] = T_frame4 @ components[2]
        links[4] = T_frame4 @ components[1]
        links[5] = T_frame4 @ components[3]
        links[6] = T_frame4 @ components[4]
        T_left_frame4_raw[state_index] = T_frame4_raw
        T_left_frame4_corrected[state_index] = T_frame4
        T_left_links[state_index] = links

    recovered_delta = T_left_frame4_corrected @ np.linalg.inv(
        T_left_frame4_raw
    )
    static_delta_error = float(
        np.max(np.abs(recovered_delta - delta_T_left[None]))
    )
    raw_relative = np.linalg.inv(T_left_frame4_raw[0]) @ T_left_frame4_raw
    corrected_relative = (
        np.linalg.inv(T_left_frame4_corrected[0])
        @ T_left_frame4_corrected
    )
    relative_motion_error = float(
        np.max(np.abs(raw_relative - corrected_relative))
    )
    T_right_links = T_right_left[None, None] @ T_left_links
    right_chain_error = float(
        np.max(
            np.abs(
                T_right_links - T_right_left[None, None] @ T_left_links
            )
        )
    )
    T_gui_links = X_table_camera[None, None] @ T_left_links
    poses_gui_world = raw_builder.matrix_series_to_poses(T_gui_links)
    if not (
        np.isfinite(T_left_links).all()
        and np.isfinite(T_right_links).all()
        and np.isfinite(poses_gui_world).all()
    ):
        raise RuntimeError("Static paper-LND GUI poses are not finite")
    if max(static_delta_error, relative_motion_error, right_chain_error) > 1.0e-10:
        raise RuntimeError(
            "Static transform invariance failed: "
            f"delta={static_delta_error:.3e}, "
            f"relative={relative_motion_error:.3e}, "
            f"right={right_chain_error:.3e}"
        )

    raw_builder.prepare_output_dir(args.output_dir, args.overwrite)
    urdf_path, mimic_path, mesh_reports, _classic_sources = (
        build_fresh_urdf_and_mimic(
            dvrk_root=args.dvrk_root,
            paper_mesh_dir=paper_mesh_dir,
            output_dir=args.output_dir,
        )
    )
    registration_robot = raw_builder.expand_root_psm_xacro(args.dvrk_root)
    registration_mimic = raw_builder.extract_mimic_map(registration_robot)
    X_gui_urdf_base, X_lnd_base_urdf_base = current_gui_base_transform(
        robot=registration_robot,
        mimic=registration_mimic,
        raw_model=raw_model,
        X_table_camera=X_table_camera,
    )
    link_mapping, helper_offsets = static_helper_registration(raw_model["lnd"])
    helper_offset_array = np.stack(
        [
            np.asarray(helper_offsets[name], dtype=np.float64)
            for name in PAPER_LINK_NAMES
        ]
    )
    driver_path = args.output_dir / "psm_paper_lnd_gui_pose_driver.npz"
    driver_schema = (
        "super_psm_raw_paper_lnd_first_stereo_se3_fixed_q5_gui_driver_v1"
        if fixed_q5_mode
        else "super_psm_raw_paper_lnd_first_stereo_static_gui_driver_v1"
    )
    np.savez_compressed(
        driver_path,
        schema=np.asarray(driver_schema),
        coordinate_frame=np.asarray("current_gui_table_world"),
        transform_convention=np.asarray(
            "X_GUI_rectLeft @ delta_T_rectLeft @ raw frame4 @ exact "
            + (
                "paper component FK(q5_raw exactly)"
                if fixed_q5_mode
                else "paper component FK(q5_raw + fixed zero offset)"
            )
        ),
        timestamps=timestamps_s,
        timestamps_ros_ns=timestamps_ros_ns,
        q7=q7_corrected,
        q7_raw=q7_raw,
        link_names=np.asarray(PAPER_LINK_NAMES),
        lnd_link_ids=PAPER_LND_LINK_IDS,
        poses_gui_world_xyz_xyzw=poses_gui_world,
        T_lndlink_urdf_link=helper_offset_array,
        X_gui_world_rectified_left_camera=X_table_camera,
        X_gui_world_urdf_base=X_gui_urdf_base,
        T_rectified_right_camera_rectified_left_camera=T_right_left,
        first_stereo_delta_T_rectified_left_camera=delta_T_left,
        first_stereo_q5_zero_offset_rad=np.asarray(q5_offset),
        per_frame_visual_correction_count=np.asarray(0, dtype=np.int64),
    )

    surface_path = args.output_dir / "psm_paper_lnd_surface_gaussians.npz"
    surface_report_path = (
        args.output_dir / "psm_paper_lnd_surface_gaussians_report.json"
    )
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/build_psm_surface_gaussians.py"),
            "--urdf",
            str(urdf_path),
            "--output",
            str(surface_path),
            "--report",
            str(surface_report_path),
            "--density",
            "30000",
            "--min-samples-per-mesh",
            "64",
            "--max-samples-per-mesh",
            "12000",
            "--seed",
            "420",
        ],
        check=True,
    )

    report_schema = (
        "super_psm_raw_paper_lnd_first_stereo_se3_fixed_q5_gui_registration_v1"
        if fixed_q5_mode
        else "super_psm_raw_paper_lnd_first_stereo_static_gui_registration_v1"
    )
    report = {
        "schema": report_schema,
        "passed": True,
        "version": args.pose_version,
        "method": (
            "raw q7/LND/calibration/hand-eye backbone + exact paper LND "
            "CAD/FK + one fixed first-pair shared stereo SE(3) + "
            + (
                "bitwise-exact raw q5; "
                if fixed_q5_mode
                else "one fixed mechanical q5 zero offset; "
            )
            + "no later visual correction"
        ),
        "upstream": {
            "repository": "https://github.com/hanyang-hu/online_dvrk_tracking",
            "commit": commit,
            "paper_fk": "diffcali/eval_dvrk/LND_fk.py",
        },
        "coordinate_chain": (
            "T_GUI_component(t) = X_GUI_rectLeft @ delta_T_rectLeft @ "
            "T_rectLeft_PSMbase(raw hand-eye) @ LND_FK_frame4(raw q7) @ "
            + (
                "paper_component_FK(q5_raw, q6_raw, jaw_raw)"
                if fixed_q5_mode
                else "paper_component_FK(q5_raw+delta_q5, q6_raw, jaw_raw)"
            )
        ),
        "state_rule": (
            f"all {state_count} raw robot states retained; after pair 0 no "
            "image, SAM2 mask, tip detector, CMA-ES or Kalman state is read"
        ),
        "static_correction": {
            "rotation_vector_deg": optimization_report["optimization"][
                "rotation_vector_deg"
            ],
            "translation_mm": optimization_report["optimization"][
                "translation_mm"
            ],
            "q5_zero_offset_deg": float(np.degrees(q5_offset)),
            "delta_T_rectified_left_camera": delta_T_left.tolist(),
        },
        "hard_invariants": {
            "all_q7_raw_exact": bool(np.array_equal(q7_corrected, q7_raw)),
            "q0_q3_q5_q6_raw_exact": bool(
                np.array_equal(
                    q7_corrected[:, [0, 1, 2, 3, 5, 6]],
                    q7_raw[:, [0, 1, 2, 3, 5, 6]],
                )
            ),
            "q5_fixed_zero_offset_max_error_rad": q5_error,
            "per_frame_visual_correction_count": 0,
            "fixed_left_delta_max_error": static_delta_error,
            "frame4_relative_motion_max_error": relative_motion_error,
            "right_from_left_chain_max_error": right_chain_error,
            "jaw_exactly_raw": bool(
                np.array_equal(q7_corrected[:, 6], q7_raw[:, 6])
            ),
        },
        "canonical_registration": {
            "link_mapping": link_mapping,
            "T_lndlink_urdf_link": helper_offsets,
            "X_lnd_base_urdf_base": X_lnd_base_urdf_base.tolist(),
        },
        "manual_offset_conventions": {
            "paper_wrist_pitch": {
                "model": "paper_lnd_exact_component_fk",
                "joint_index": 4,
                "anchor_link": "PSM1_tool_wrist_link",
                "moving_links": [
                    "PSM1_tool_wrist_shaft_link",
                    "PSM1_tool_wrist_sca_link",
                    "PSM1_tool_wrist_sca_shaft_link",
                    "PSM1_tool_wrist_sca_ee_link_1",
                    "PSM1_tool_wrist_sca_ee_link_2",
                ],
            }
        },
        "fresh_gui_assets": {
            "urdf": str(urdf_path.resolve()),
            "urdf_sha256": sha256(urdf_path),
            "mimic_map": str(mimic_path.resolve()),
            "mimic_map_sha256": sha256(mimic_path),
            "surface_gaussians": str(surface_path.resolve()),
            "surface_gaussians_sha256": sha256(surface_path),
            "converted_meshes": mesh_reports,
        },
        "inputs": {
            "raw_kinematics": {
                "path": str(raw_kinematics_path.resolve()),
                "sha256": sha256(raw_kinematics_path),
            },
            "static_correction": {
                "path": str(correction_path.resolve()),
                "sha256": sha256(correction_path),
            },
            "optimization_report": {
                "path": str(optimization_report_path.resolve()),
                "sha256": sha256(optimization_report_path),
            },
        },
        "driver": {
            "path": str(driver_path.resolve()),
            "sha256": sha256(driver_path),
            "states": state_count,
            "links": len(PAPER_LINK_NAMES),
            "poses_shape": list(poses_gui_world.shape),
        },
    }
    report_path = args.output_dir / "registration_report.json"
    raw_builder.write_json(report_path, report)
    print(
        json.dumps(
            {
                "passed": True,
                "version": args.pose_version,
                "states": state_count,
                "q5_zero_offset_deg": float(np.degrees(q5_offset)),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
