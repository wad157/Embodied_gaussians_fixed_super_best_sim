#!/usr/bin/env python3
"""Lift accepted stereo P420006 keyframe corrections to all raw q7 states.

This builder never reads a historical tracked pose sequence.  It recomputes
every link transform from the frozen raw q7/LND backbone, then applies:

1. one stereo-estimated global camera-frame correction; and
2. a shape-preserving interpolation of the small keyframe residuals.

The existing raw P420006 driver is read only for its fixed CAD-link adapters
and current-GUI coordinate transform.  Its saved link poses are not consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator
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
        description=(
            "Build a 5458-state GUI driver from the accepted raw-P420006 "
            "stereo keyframe correction."
        )
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--p420-root", type=Path, default=P420_ROOT)
    parser.add_argument("--visual-root", type=Path, default=VISUAL_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VISUAL_ROOT / "gui_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def rotation_error_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    relative = first @ np.swapaxes(second, -1, -2)
    return np.degrees(
        Rotation.from_matrix(relative.reshape(-1, 3, 3))
        .magnitude()
        .reshape(relative.shape[:-2])
    )


def interpolate_clamped_pchip(
    key_timestamps_ros_ns: np.ndarray,
    key_values: np.ndarray,
    query_timestamps_ros_ns: np.ndarray,
) -> np.ndarray:
    origin = int(key_timestamps_ros_ns[0])
    key_time = (
        key_timestamps_ros_ns.astype(np.float64) - float(origin)
    ) * 1.0e-9
    query_time = (
        query_timestamps_ros_ns.astype(np.float64) - float(origin)
    ) * 1.0e-9
    if np.any(np.diff(key_time) <= 0.0):
        raise ValueError("Visual keyframe timestamps are not strictly increasing")
    clamped_time = np.clip(query_time, key_time[0], key_time[-1])
    interpolator = PchipInterpolator(
        key_time,
        np.asarray(key_values, dtype=np.float64),
        axis=0,
        extrapolate=False,
    )
    values = np.asarray(interpolator(clamped_time), dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("PCHIP produced non-finite correction values")
    return values


def finite_difference_maximum(
    values: np.ndarray,
    timestamps_s: np.ndarray,
) -> np.ndarray:
    dt = np.diff(timestamps_s)
    if np.any(dt <= 0.0):
        raise ValueError("Raw joint timestamps are not strictly increasing")
    return np.max(np.abs(np.diff(values, axis=0) / dt[:, None]), axis=0)


def main() -> None:
    args = parse_args()
    raw_model_path = args.raw_root / "model.json"
    raw_kinematics_path = args.raw_root / "kinematics.npz"
    raw_report_path = args.raw_root / "report.json"
    source_driver_path = (
        args.p420_root / "psm_p420006_gui_pose_driver.npz"
    )
    source_registration_path = args.p420_root / "registration_report.json"
    keyframes_path = args.visual_root / "keyframes.npz"
    corrections_path = args.visual_root / "keyframe_corrections.npz"
    optimization_report_path = args.visual_root / "optimization_report.json"
    confidence_path = args.visual_root / "annotation_confidence.json"
    for path in (
        raw_model_path,
        raw_kinematics_path,
        raw_report_path,
        source_driver_path,
        source_registration_path,
        keyframes_path,
        corrections_path,
        optimization_report_path,
        confidence_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    raw_report = json.loads(raw_report_path.read_text(encoding="utf-8"))
    source_registration = json.loads(
        source_registration_path.read_text(encoding="utf-8")
    )
    optimization_report = json.loads(
        optimization_report_path.read_text(encoding="utf-8")
    )
    if raw_report.get("passed") is not True:
        raise RuntimeError("The frozen raw-kinematics report has not passed")
    if (
        source_registration.get("passed") is not True
        or source_registration.get("version") != "raw_p420006"
        or source_registration.get("uses_historical_psm_derivatives")
        is not False
    ):
        raise RuntimeError("The raw-only P420006 registration is not accepted")
    if (
        optimization_report.get("schema")
        != "super_p420006_stereo_keyframe_optimization_v2"
        or optimization_report.get("passed") is not True
    ):
        raise RuntimeError("The stereo keyframe optimization has not passed")

    raw_model = json.loads(raw_model_path.read_text(encoding="utf-8"))
    with np.load(raw_kinematics_path, allow_pickle=False) as raw:
        timestamps_s = raw["joint_timestamps_s"].astype(np.float64)
        timestamps_ros_ns = raw["joint_timestamps_ros_ns"].astype(np.int64)
        q7_raw = raw["q7"].astype(np.float64)
        T_left_lnd_raw = raw[
            "T_rectified_left_camera_lnd_link"
        ].astype(np.float64)
    with np.load(source_driver_path, allow_pickle=False) as source:
        source_schema = str(source["schema"].item())
        source_timestamps = source["timestamps"].astype(np.float64)
        source_timestamps_ros_ns = source[
            "timestamps_ros_ns"
        ].astype(np.int64)
        source_q7 = source["q7"].astype(np.float64)
        link_names = source["link_names"].tolist()
        lnd_link_ids = source["lnd_link_ids"].astype(np.int64)
        link_offsets = source["T_lndlink_urdf_link"].astype(np.float64)
        X_gui_left = source[
            "X_gui_world_rectified_left_camera"
        ].astype(np.float64)
        X_gui_urdf_base = source[
            "X_gui_world_urdf_base"
        ].astype(np.float64)
    if source_schema != "super_psm_raw_p420006_gui_pose_driver_v1":
        raise RuntimeError(f"Unexpected raw P420006 driver schema: {source_schema}")
    if not (
        np.array_equal(source_timestamps, timestamps_s)
        and np.array_equal(source_timestamps_ros_ns, timestamps_ros_ns)
        and np.array_equal(source_q7, q7_raw)
    ):
        raise RuntimeError(
            "The raw P420006 adapter carrier no longer matches frozen raw q7"
        )

    with np.load(keyframes_path, allow_pickle=False) as keyframes:
        keyframe_pair_timestamps = keyframes[
            "pair_timestamp_ros_ns"
        ].astype(np.int64)
        keyframe_q7_mid = keyframes["q7_mid"].astype(np.float64)
        T_right_left = keyframes[
            "T_rectified_right_camera_rectified_left_camera"
        ].astype(np.float64)
    with np.load(corrections_path, allow_pickle=False) as correction:
        correction_schema = str(correction["schema"].item())
        correction_timestamps = correction[
            "pair_timestamp_ros_ns"
        ].astype(np.int64)
        global_rotvec = correction[
            "global_correction_rotvec_camera_rad"
        ].astype(np.float64)
        global_translation = correction[
            "global_correction_translation_camera_m"
        ].astype(np.float64)
        global_q = correction[
            "global_correction_q3_q6_rad"
        ].astype(np.float64)
        local_rotvec_key = correction[
            "local_correction_rotvec_camera_rad"
        ].astype(np.float64)
        local_translation_key = correction[
            "local_correction_translation_camera_m"
        ].astype(np.float64)
        local_q_key = correction[
            "local_correction_q3_q6_rad"
        ].astype(np.float64)
        registration_rotation = correction[
            "global_registration_rotation_camera"
        ].astype(np.float64)
    if correction_schema != "super_p420006_stereo_keyframe_corrections_v2":
        raise RuntimeError(f"Unexpected correction schema: {correction_schema}")
    if not (
        np.array_equal(correction_timestamps, keyframe_pair_timestamps)
        and len(correction_timestamps) == 10
    ):
        raise RuntimeError("Corrections do not match the ten stereo keyframes")
    if (
        np.max(np.abs(registration_rotation.T @ registration_rotation - np.eye(3)))
        > 1.0e-7
        or np.linalg.det(registration_rotation) < 0.999999
    ):
        raise RuntimeError("The stored global CAD registration is not a rotation")

    local_rotvec = interpolate_clamped_pchip(
        correction_timestamps,
        local_rotvec_key,
        timestamps_ros_ns,
    )
    local_translation = interpolate_clamped_pchip(
        correction_timestamps,
        local_translation_key,
        timestamps_ros_ns,
    )
    local_q = interpolate_clamped_pchip(
        correction_timestamps,
        local_q_key,
        timestamps_ros_ns,
    )
    keyframe_reconstruction_error = max(
        float(
            np.max(
                np.abs(
                    interpolate_clamped_pchip(
                        correction_timestamps,
                        values,
                        correction_timestamps,
                    )
                    - values
                )
            )
        )
        for values in (
            local_rotvec_key,
            local_translation_key,
            local_q_key[:, :3],
        )
    )
    if keyframe_reconstruction_error > 1.0e-10:
        raise RuntimeError(
            "PCHIP does not reproduce keyframe corrections exactly: "
            f"{keyframe_reconstruction_error:.3e}"
        )

    confidence = json.loads(confidence_path.read_text(encoding="utf-8"))
    jaw_weights = np.empty(len(correction_timestamps), dtype=np.float64)
    for keyframe_index in range(len(correction_timestamps)):
        frame_confidence = dict(confidence["default"])
        frame_confidence.update(
            confidence.get("keyframes", {}).get(
                str(keyframe_index), {}
            )
        )
        jaw_weights[keyframe_index] = 0.5 * (
            float(frame_confidence["jaw_masks"])
            + float(frame_confidence["jaw_tips"])
        )
    raw_jaw_key = keyframe_q7_mid[:, 6]
    optimized_jaw_key = (
        raw_jaw_key + global_q[3] + local_q_key[:, 3]
    )
    jaw_design = np.column_stack(
        [raw_jaw_key, np.ones_like(raw_jaw_key)]
    )
    weighted_design = jaw_weights[:, None] * jaw_design
    jaw_scale, jaw_offset = np.linalg.solve(
        jaw_design.T @ weighted_design,
        jaw_design.T @ (jaw_weights * optimized_jaw_key),
    )
    if not (0.25 <= jaw_scale <= 2.0):
        raise RuntimeError(
            "Stereo jaw calibration is not physically monotonic: "
            f"scale={jaw_scale:.6f}"
        )
    fitted_jaw_key = jaw_scale * raw_jaw_key + jaw_offset
    jaw_fit_residual = fitted_jaw_key - optimized_jaw_key

    q7_corrected = q7_raw.copy()
    q7_corrected[:, 3:6] += (
        global_q[None, :3] + local_q[:, :3]
    )
    q7_corrected[:, 6] = jaw_scale * q7_raw[:, 6] + jaw_offset
    # Store the affine jaw result in the same residual layout as the other
    # joints.  It is deliberately not the time interpolation of the occluded
    # frame estimates.
    local_q[:, 3] = (
        q7_corrected[:, 6] - q7_raw[:, 6] - global_q[3]
    )
    if not np.array_equal(q7_corrected[:, :3], q7_raw[:, :3]):
        raise RuntimeError("Visual correction modified the raw arm q0..q2 backbone")
    # The raw-only P420006 URDF was deliberately widened to this measured
    # encoder range; reject a visually fitted jaw that leaves that contract.
    if (
        float(q7_corrected[:, 6].min()) < -1.2
        or float(q7_corrected[:, 6].max()) > 1.6
    ):
        raise RuntimeError(
            "Corrected jaw leaves the raw P420006 URDF limits: "
            f"{q7_corrected[:, 6].min():.6f}.."
            f"{q7_corrected[:, 6].max():.6f} rad"
        )

    state_count = len(timestamps_s)
    link_count = len(link_names)
    T_left_links = np.empty(
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
    T_left_base = T_left_lnd_raw[:, 0]
    for state_index in range(state_count):
        fk = raw_builder.lnd_fk(
            raw_model["lnd"],
            q7_corrected[state_index],
        )
        links = (
            T_left_base[state_index][None]
            @ fk[lnd_link_ids]
            @ link_offsets
        )
        anchor = (
            T_left_base[state_index] @ fk[6]
        )[:3, 3]
        delta_rotation = delta_rotations[state_index]
        links[:, :3, :3] = delta_rotation[None] @ links[:, :3, :3]
        links[:, :3, 3] = (
            (links[:, :3, 3] - anchor[None]) @ delta_rotation.T
            + anchor[None]
            + global_translation[None]
            + local_translation[state_index][None]
        )
        T_left_links[state_index] = links

    T_right_links = T_right_left[None, None] @ T_left_links
    T_gui_links = X_gui_left[None, None] @ T_left_links
    poses_gui_world = raw_builder.matrix_series_to_poses(T_gui_links)
    if not (
        np.isfinite(T_left_links).all()
        and np.isfinite(T_right_links).all()
        and np.isfinite(poses_gui_world).all()
    ):
        raise RuntimeError("The lifted stereo-visual driver is not finite")

    jaw_indices = [
        link_names.index("PSM1_tool_wrist_sca_ee_link_1"),
        link_names.index("PSM1_tool_wrist_sca_ee_link_2"),
    ]
    jaw_origin_separation = np.linalg.norm(
        T_left_links[:, jaw_indices[0], :3, 3]
        - T_left_links[:, jaw_indices[1], :3, 3],
        axis=1,
    )
    jaw_axis_abs_dot = np.abs(
        np.einsum(
            "ni,ni->n",
            T_left_links[:, jaw_indices[0], :3, 0],
            T_left_links[:, jaw_indices[1], :3, 0],
        )
    )
    if (
        float(jaw_origin_separation.max()) > 1.0e-9
        or float(jaw_axis_abs_dot.min()) < 1.0 - 1.0e-10
    ):
        raise RuntimeError("Visual lifting broke the shared parallel jaw hinge")

    raw_p420_left = (
        T_left_lnd_raw[:, lnd_link_ids] @ link_offsets[None]
    )
    position_change_m = np.linalg.norm(
        T_left_links[..., :3, 3] - raw_p420_left[..., :3, 3],
        axis=-1,
    )
    rotation_change_deg = rotation_error_deg(
        T_left_links[..., :3, :3],
        raw_p420_left[..., :3, :3],
    )

    raw_builder.prepare_output_dir(args.output_dir, args.overwrite)
    driver_path = (
        args.output_dir
        / "psm_p420006_stereo_visual_gui_pose_driver.npz"
    )
    np.savez_compressed(
        driver_path,
        schema=np.asarray(
            "super_psm_raw_p420006_stereo_visual_gui_pose_driver_v1"
        ),
        coordinate_frame=np.asarray("current_gui_table_world"),
        transform_convention=np.asarray(
            "raw q7/LND FK + stereo global correction + clamped PCHIP "
            "keyframe residual, left-camera state converted by current GUI "
            "X_gui_world_rectified_left_camera"
        ),
        timestamps=timestamps_s,
        timestamps_ros_ns=timestamps_ros_ns,
        q7=q7_corrected,
        q7_raw=q7_raw,
        link_names=np.asarray(link_names),
        lnd_link_ids=lnd_link_ids,
        poses_gui_world_xyz_xyzw=poses_gui_world,
        T_lndlink_urdf_link=link_offsets,
        X_gui_world_rectified_left_camera=X_gui_left,
        X_gui_world_urdf_base=X_gui_urdf_base,
        T_rectified_right_camera_rectified_left_camera=T_right_left,
        visual_keyframe_timestamps_ros_ns=correction_timestamps,
        visual_global_rotvec_camera_rad=global_rotvec,
        visual_global_translation_camera_m=global_translation,
        visual_global_q3_q6_rad=global_q,
        visual_global_registration_rotation_camera=(
            registration_rotation
        ),
        visual_local_rotvec_camera_rad=local_rotvec,
        visual_local_translation_camera_m=local_translation,
        visual_local_q3_q6_rad=local_q,
        visual_jaw_affine_scale=np.asarray(jaw_scale),
        visual_jaw_affine_offset_rad=np.asarray(jaw_offset),
    )

    correction_velocity = {
        "local_rotvec_rad_s": finite_difference_maximum(
            local_rotvec, timestamps_s
        ).tolist(),
        "local_translation_m_s": finite_difference_maximum(
            local_translation, timestamps_s
        ).tolist(),
        "local_q3_q6_rad_s": finite_difference_maximum(
            local_q, timestamps_s
        ).tolist(),
    }
    report: dict[str, Any] = {
        "schema": (
            "super_psm_raw_p420006_stereo_visual_gui_registration_v1"
        ),
        "passed": True,
        "version": "raw_p420006_stereo_visual",
        "method": (
            "frozen original q7/LND/calibration/hand-eye backbone; exact "
            "P420006 CAD adapters; one shared two-camera global correction; "
            "clamped shape-preserving interpolation of ten small residuals"
        ),
        "uses_historical_psm_derivatives": False,
        "uses_source_p420006_saved_poses": False,
        "state_rule": (
            "all 5458 states are recomputed from q7_raw; q0..q2 remain exact; "
            "q3..q5 use the small interpolated visual residual; q6 uses one "
            "confidence-weighted monotonic affine calibration; a "
            "distal-hinge-centered camera-frame residual is also applied"
        ),
        "stereo_rule": (
            "one 3-D state is estimated jointly; right-camera poses are "
            "always T_rectRight_rectLeft @ T_rectLeft_link"
        ),
        "coordinate_chain": (
            "T_GUIworld_link(t) = X_GUIworld_rectLeftCamera @ "
            "visual_correct(T_rectLeft_LNDbase(raw,t) @ "
            "FK_LND(q7_corrected,t) @ T_LNDlink_P420006link)"
        ),
        "interpolation": {
            "method": "component-wise PCHIP",
            "keyframes": len(correction_timestamps),
            "outside_keyframe_range": "clamp to nearest endpoint",
            "jaw_rule": (
                "do not time-interpolate occluded jaw estimates; fit one "
                "confidence-weighted monotonic affine map from raw q6"
            ),
            "keyframe_reconstruction_max_abs": (
                keyframe_reconstruction_error
            ),
            "derivative_maximum": correction_velocity,
        },
        "correction_statistics": {
            "global_rotation_vector_deg": np.degrees(
                global_rotvec
            ).tolist(),
            "global_translation_mm": (
                global_translation * 1.0e3
            ).tolist(),
            "global_q3_q6_deg": np.degrees(global_q).tolist(),
            "jaw_affine_calibration": {
                "scale": float(jaw_scale),
                "offset_rad": float(jaw_offset),
                "offset_deg": float(np.degrees(jaw_offset)),
                "keyframe_weights": jaw_weights.tolist(),
                "keyframe_fit_residual_deg": np.degrees(
                    jaw_fit_residual
                ).tolist(),
                "maximum_high_confidence_fit_residual_deg": float(
                    np.max(
                        np.abs(
                            np.degrees(
                                jaw_fit_residual[jaw_weights >= 0.999]
                            )
                        )
                    )
                ),
            },
            "local_rotation_component_range_deg": [
                np.degrees(local_rotvec.min(axis=0)).tolist(),
                np.degrees(local_rotvec.max(axis=0)).tolist(),
            ],
            "local_translation_component_range_mm": [
                (local_translation.min(axis=0) * 1.0e3).tolist(),
                (local_translation.max(axis=0) * 1.0e3).tolist(),
            ],
            "local_q3_q6_component_range_deg": [
                np.degrees(local_q.min(axis=0)).tolist(),
                np.degrees(local_q.max(axis=0)).tolist(),
            ],
            "raw_to_visual_position_change_mm": {
                "maximum": float(position_change_m.max() * 1.0e3),
                "median": float(np.median(position_change_m) * 1.0e3),
            },
            "raw_to_visual_rotation_change_deg": {
                "maximum": float(rotation_change_deg.max()),
                "median": float(np.median(rotation_change_deg)),
            },
            "corrected_jaw_range_deg": np.degrees(
                [
                    q7_corrected[:, 6].min(),
                    q7_corrected[:, 6].max(),
                ]
            ).tolist(),
        },
        "geometry_gates": {
            "jaw_hinge_origin_separation_max_m": float(
                jaw_origin_separation.max()
            ),
            "jaw_hinge_axis_abs_dot_min": float(jaw_axis_abs_dot.min()),
        },
        "canonical_registration": source_registration[
            "canonical_registration"
        ],
        "manual_offset_conventions": source_registration[
            "manual_offset_conventions"
        ],
        "inputs": {
            "raw_model": {
                "path": str(raw_model_path.resolve()),
                "sha256": sha256(raw_model_path),
            },
            "raw_kinematics": {
                "path": str(raw_kinematics_path.resolve()),
                "sha256": sha256(raw_kinematics_path),
            },
            "raw_report": {
                "path": str(raw_report_path.resolve()),
                "sha256": sha256(raw_report_path),
            },
            "raw_p420006_fixed_adapters": {
                "driver_path": str(source_driver_path.resolve()),
                "driver_sha256": sha256(source_driver_path),
                "registration_path": str(
                    source_registration_path.resolve()
                ),
                "registration_sha256": sha256(
                    source_registration_path
                ),
            },
            "stereo_keyframes": {
                "path": str(keyframes_path.resolve()),
                "sha256": sha256(keyframes_path),
            },
            "accepted_corrections": {
                "path": str(corrections_path.resolve()),
                "sha256": sha256(corrections_path),
            },
            "optimization_report": {
                "path": str(optimization_report_path.resolve()),
                "sha256": sha256(optimization_report_path),
            },
            "annotation_confidence": {
                "path": str(confidence_path.resolve()),
                "sha256": sha256(confidence_path),
            },
        },
        "reused_gui_assets_without_changes": {
            "urdf": str(
                (args.p420_root / "psm_p420006.urdf").resolve()
            ),
            "mimic_map": str(
                (args.p420_root / "psm_p420006_mimic_map.json").resolve()
            ),
            "surface_gaussians": str(
                (
                    args.p420_root
                    / "psm_p420006_surface_gaussians.npz"
                ).resolve()
            ),
        },
        "driver": {
            "path": str(driver_path.resolve()),
            "sha256": sha256(driver_path),
            "states": state_count,
            "links": link_count,
            "poses_shape": list(poses_gui_world.shape),
            "q7_raw_preserved_shape": list(q7_raw.shape),
            "q7_corrected_shape": list(q7_corrected.shape),
        },
    }
    report_path = args.output_dir / "registration_report.json"
    raw_builder.write_json(report_path, report)
    print(
        json.dumps(
            {
                "passed": True,
                "version": report["version"],
                "states": state_count,
                "keyframes": len(correction_timestamps),
                "jaw_range_deg": report["correction_statistics"][
                    "corrected_jaw_range_deg"
                ],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
