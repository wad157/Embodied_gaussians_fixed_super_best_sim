#!/usr/bin/env python3
"""Lift the full stereo SurgicalSAM2 online correction to all raw q7 times.

The 1631 strict stereo observations estimate bounded residuals.  This builder
linearly samples those already Kalman-filtered residuals at all 5458 native
robot timestamps, then recomputes every P420006 link from:

    raw bag q7 + calibration + hand-eye + LND FK + fixed CAD adapters.

No historical tracked pose sequence is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

import build_super_raw_psm_gui_driver as raw_builder


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = (
    REPO_ROOT / "data/super/psm_raw_kinematics_v1（纯机器人学版本）"
)
P420_ROOT = RAW_ROOT / "gui_p420006_v1"
SAM2_ROOT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_p420006_stereo_v1"
    / "surgicalsam2_stereo_sequence_v1"
)
OPTIMIZATION_ROOT = SAM2_ROOT / "online_stereo_cma_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a complete 5458-state GUI driver from the accepted "
            "1631-pair first-frame-only stereo SurgicalSAM2 correction."
        )
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--p420-root", type=Path, default=P420_ROOT)
    parser.add_argument(
        "--optimization-root",
        type=Path,
        default=OPTIMIZATION_ROOT,
    )
    parser.add_argument(
        "--visual-root",
        type=Path,
        default=SAM2_ROOT.parent,
        help="Directory containing keyframes.npz and pair_manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OPTIMIZATION_ROOT / "gui_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def interpolate_clamped_linear(
    source_timestamps_ns: np.ndarray,
    source_values: np.ndarray,
    query_timestamps_ns: np.ndarray,
) -> np.ndarray:
    source_timestamps_ns = np.asarray(source_timestamps_ns, dtype=np.int64)
    source_values = np.asarray(source_values, dtype=np.float64)
    query_timestamps_ns = np.asarray(query_timestamps_ns, dtype=np.int64)
    if np.any(np.diff(source_timestamps_ns) <= 0):
        raise ValueError("Stereo pair timestamps must be strictly increasing")
    origin = int(source_timestamps_ns[0])
    source_s = (
        source_timestamps_ns.astype(np.float64) - float(origin)
    ) * 1.0e-9
    query_s = (
        query_timestamps_ns.astype(np.float64) - float(origin)
    ) * 1.0e-9
    result = np.column_stack(
        [
            np.interp(
                query_s,
                source_s,
                source_values[:, component],
                left=float(source_values[0, component]),
                right=float(source_values[-1, component]),
            )
            for component in range(source_values.shape[1])
        ]
    )
    if not np.isfinite(result).all():
        raise RuntimeError("Linear correction interpolation is not finite")
    return result


def rotation_error_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    relative = first @ np.swapaxes(second, -1, -2)
    return np.degrees(
        Rotation.from_matrix(relative.reshape(-1, 3, 3))
        .magnitude()
        .reshape(relative.shape[:-2])
    )


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
    manifest_path = args.visual_root / "pair_manifest.json"
    correction_path = (
        args.optimization_root / "online_stereo_corrections.npz"
    )
    optimization_report_path = args.optimization_root / "report.json"
    for path in (
        raw_model_path,
        raw_kinematics_path,
        raw_report_path,
        source_driver_path,
        source_registration_path,
        keyframes_path,
        manifest_path,
        correction_path,
        optimization_report_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    raw_report = json.loads(raw_report_path.read_text(encoding="utf-8"))
    source_registration = json.loads(
        source_registration_path.read_text(encoding="utf-8")
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
        raise RuntimeError("The raw-only P420006 adapter carrier is not accepted")
    if (
        optimization_report.get("schema")
        != "super_p420006_stereo_sam2_online_optimization_v1"
        or optimization_report.get("passed") is not True
        or optimization_report.get("sequence", {}).get(
            "complete_strict_pair_sequence"
        )
        is not True
    ):
        raise RuntimeError("The complete stereo SAM2 online optimization failed")

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
        raise RuntimeError(f"Unexpected source driver: {source_schema}")
    if not (
        np.array_equal(source_timestamps, timestamps_s)
        and np.array_equal(source_timestamps_ros_ns, timestamps_ros_ns)
        and np.array_equal(source_q7, q7_raw)
    ):
        raise RuntimeError("The fixed P420006 adapters no longer match raw q7")

    with np.load(keyframes_path, allow_pickle=False) as keyframes:
        T_right_left = keyframes[
            "T_rectified_right_camera_rectified_left_camera"
        ].astype(np.float64)
    with np.load(correction_path, allow_pickle=False) as correction:
        schema = str(correction["schema"].item())
        pair_timestamps_ns = correction[
            "pair_timestamp_ros_ns"
        ].astype(np.int64)
        pair_slots = correction["strict_pair_slot"].astype(np.int64)
        global_correction = correction[
            "global_correction"
        ].astype(np.float64)
        local_pair_correction = correction[
            "local_filtered_corrections"
        ].astype(np.float64)
        global_bounds = correction[
            "global_parameter_bounds"
        ].astype(np.float64)
        local_bounds = correction[
            "local_parameter_bounds"
        ].astype(np.float64)
    expected_pairs = int(manifest["source_pair_count"])
    if (
        schema != "super_p420006_stereo_sam2_online_corrections_v1"
        or len(pair_timestamps_ns) != expected_pairs
        or not np.array_equal(
            pair_slots,
            np.arange(expected_pairs, dtype=np.int64),
        )
        or local_pair_correction.shape != (expected_pairs, 10)
        or global_correction.shape != (10,)
    ):
        raise RuntimeError("Online corrections do not cover all strict pairs")
    if (
        float(np.max(np.abs(global_correction / global_bounds))) >= 0.995
        or float(
            np.max(
                np.abs(local_pair_correction / local_bounds[None])
            )
        )
        >= 0.995
    ):
        raise RuntimeError("Refusing a correction that saturates its bounds")

    local_correction = interpolate_clamped_linear(
        pair_timestamps_ns,
        local_pair_correction,
        timestamps_ros_ns,
    )
    q7_corrected = q7_raw.copy()
    q7_corrected[:, 3:7] += (
        global_correction[None, 6:10] + local_correction[:, 6:10]
    )
    if not np.array_equal(q7_corrected[:, :3], q7_raw[:, :3]):
        raise RuntimeError("Visual correction modified raw q0..q2")
    if (
        float(q7_corrected[:, 6].min()) < -1.2
        or float(q7_corrected[:, 6].max()) > 1.6
    ):
        raise RuntimeError(
            "Corrected jaw leaves the P420006 limits: "
            f"{q7_corrected[:, 6].min():.6f}.."
            f"{q7_corrected[:, 6].max():.6f}"
        )

    state_count = len(timestamps_s)
    link_count = len(link_names)
    T_left_links = np.empty(
        (state_count, link_count, 4, 4),
        dtype=np.float64,
    )
    global_rotation = Rotation.from_rotvec(
        global_correction[:3]
    ).as_matrix()
    local_rotations = Rotation.from_rotvec(
        local_correction[:, :3]
    ).as_matrix()
    delta_rotations = local_rotations @ global_rotation[None]
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
            + global_correction[None, 3:6]
            + local_correction[state_index][None, 3:6]
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
        raise RuntimeError("Lifted GUI poses are not finite")

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
        / "psm_p420006_sam2_online_gui_pose_driver.npz"
    )
    np.savez_compressed(
        driver_path,
        schema=np.asarray(
            "super_psm_raw_p420006_sam2_online_gui_pose_driver_v1"
        ),
        coordinate_frame=np.asarray("current_gui_table_world"),
        transform_convention=np.asarray(
            "raw q7/LND FK + fixed P420006 adapters + first-pair-only "
            "stereo SurgicalSAM2 global and timestamp-linear online residuals"
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
        visual_pair_timestamps_ros_ns=pair_timestamps_ns,
        visual_global_correction=global_correction,
        visual_local_pair_correction=local_pair_correction,
        visual_local_q7_time_correction=local_correction,
    )

    report: dict[str, Any] = {
        "schema": (
            "super_psm_raw_p420006_sam2_online_gui_registration_v1"
        ),
        "passed": True,
        "version": "raw_p420006_sam2_online",
        "method": (
            "complete 1631-pair first-frame-only causal stereo SurgicalSAM2 "
            "online correction lifted to all native q7 times"
        ),
        "uses_historical_psm_derivatives": False,
        "uses_source_p420006_saved_poses": False,
        "state_rule": (
            "all 5458 states are recomputed from raw q7/LND; q0..q2 remain "
            "exact; only bounded q3..q6 and distal-hinge-centered residuals "
            "are added"
        ),
        "interpolation": {
            "method": "component-wise linear",
            "observations": len(pair_timestamps_ns),
            "outside_pair_range": "clamp to nearest endpoint",
            "query_states": state_count,
        },
        "correction_statistics": {
            "global_rotation_vector_deg": np.degrees(
                global_correction[:3]
            ).tolist(),
            "global_translation_mm": (
                global_correction[3:6] * 1.0e3
            ).tolist(),
            "global_q3_q6_deg": np.degrees(
                global_correction[6:10]
            ).tolist(),
            "maximum_global_boundary_fraction": float(
                np.max(np.abs(global_correction / global_bounds))
            ),
            "maximum_local_boundary_fraction": float(
                np.max(
                    np.abs(local_pair_correction / local_bounds[None])
                )
            ),
            "corrected_jaw_range_deg": np.degrees(
                [q7_corrected[:, 6].min(), q7_corrected[:, 6].max()]
            ).tolist(),
            "raw_to_visual_position_change_mm": {
                "maximum": float(position_change_m.max() * 1.0e3),
                "median": float(np.median(position_change_m) * 1.0e3),
            },
            "raw_to_visual_rotation_change_deg": {
                "maximum": float(rotation_change_deg.max()),
                "median": float(np.median(rotation_change_deg)),
            },
        },
        "geometry_gates": {
            "jaw_hinge_origin_separation_max_m": float(
                jaw_origin_separation.max()
            ),
            "jaw_hinge_axis_abs_dot_min": float(jaw_axis_abs_dot.min()),
        },
        "inputs": {
            "raw_model": {
                "path": str(raw_model_path.resolve()),
                "sha256": sha256(raw_model_path),
            },
            "raw_kinematics": {
                "path": str(raw_kinematics_path.resolve()),
                "sha256": sha256(raw_kinematics_path),
            },
            "raw_p420006_adapter_driver": {
                "path": str(source_driver_path.resolve()),
                "sha256": sha256(source_driver_path),
            },
            "online_corrections": {
                "path": str(correction_path.resolve()),
                "sha256": sha256(correction_path),
            },
            "optimization_report": {
                "path": str(optimization_report_path.resolve()),
                "sha256": sha256(optimization_report_path),
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
                "strict_pairs": len(pair_timestamps_ns),
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
