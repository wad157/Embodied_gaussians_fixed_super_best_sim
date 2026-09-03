#!/usr/bin/env python3
"""Select and extract ten raw stereo anchors for P420006 visual calibration.

The inputs are deliberately limited to the frozen raw kinematic backbone, the
fresh P420006 registration/surface asset, the original bag, and the original
stereo calibration.  Historical image-corrected pose drivers are never read.

The selected observations are strict monotonic stereo pairs from the raw bag.
Selection covers time, distal 3-D position, roll, wrist pitch/yaw, and jaw
opening while requiring the P420006 distal surface to be visible in both
rectified cameras.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

from build_super_raw_psm_kinematics import (
    LEFT_TOPIC,
    RIGHT_TOPIC,
    image_message_to_bgr,
    load_stereo_calibration,
    nearest_indices,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = (
    REPO_ROOT / "data/super/psm_raw_kinematics_v1（纯机器人学版本）"
)
P420_ROOT = RAW_ROOT / "gui_p420006_v1"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_p420006_stereo_v1"
)
UPSTREAM_URL = "https://github.com/hanyang-hu/online_dvrk_tracking"
UPSTREAM_COMMIT = "cb2a264167aaf05b5a9c20da885d48568f78311f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select representative strict raw stereo pairs and extract their "
            "rectified images for P420006 manual annotation."
        )
    )
    parser.add_argument(
        "--bag",
        type=Path,
        default=REPO_ROOT / "data/grasp5/grasp5.bag",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO_ROOT / "data/camera_calibration.yaml",
    )
    parser.add_argument(
        "--kinematics",
        type=Path,
        default=RAW_ROOT / "kinematics.npz",
    )
    parser.add_argument(
        "--raw-model",
        type=Path,
        default=RAW_ROOT / "model.json",
    )
    parser.add_argument(
        "--raw-report",
        type=Path,
        default=RAW_ROOT / "report.json",
    )
    parser.add_argument(
        "--registration",
        type=Path,
        default=P420_ROOT / "registration_report.json",
    )
    parser.add_argument(
        "--surface-gaussians",
        type=Path,
        default=P420_ROOT / "psm_p420006_surface_gaussians.npz",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--minimum-visibility", type=float, default=0.70)
    parser.add_argument("--minimum-time-gap-s", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
        shutil.rmtree(path)
    (path / "images").mkdir(parents=True)
    (path / "annotations").mkdir()
    (path / "previews").mkdir()


def verify_frozen_raw_input(
    path: Path,
    frozen: dict[str, Any],
    label: str,
) -> None:
    stat = path.stat()
    if int(frozen["size_bytes"]) != stat.st_size:
        raise RuntimeError(
            f"{label} size differs from frozen raw report: "
            f"{stat.st_size} != {frozen['size_bytes']}"
        )
    if int(frozen["mtime_ns"]) != stat.st_mtime_ns:
        raise RuntimeError(
            f"{label} mtime differs from frozen raw report: "
            f"{stat.st_mtime_ns} != {frozen['mtime_ns']}"
        )


def transform_surface(
    local_means: np.ndarray,
    gaussian_link_ids: np.ndarray,
    link_transforms: np.ndarray,
) -> np.ndarray:
    rotations = link_transforms[gaussian_link_ids, :3, :3]
    translations = link_transforms[gaussian_link_ids, :3, 3]
    return np.einsum("nij,nj->ni", rotations, local_means) + translations


def project_visibility(
    points_camera: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> tuple[float, float]:
    positive = points_camera[:, 2] > 1.0e-6
    uv = np.empty((len(points_camera), 2), dtype=np.float64)
    uv[:, 0] = (
        K[0, 0] * points_camera[:, 0] / points_camera[:, 2] + K[0, 2]
    )
    uv[:, 1] = (
        K[1, 1] * points_camera[:, 1] / points_camera[:, 2] + K[1, 2]
    )
    visible = (
        positive
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < height)
    )
    visibility = float(np.mean(visible))
    if np.count_nonzero(visible) < 2:
        return visibility, 0.0
    extent = np.ptp(uv[visible], axis=0)
    return visibility, float(np.linalg.norm(extent))


def robust_unit_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low = np.percentile(values, 2.0, axis=0)
    high = np.percentile(values, 98.0, axis=0)
    scale = np.maximum(high - low, 1.0e-9)
    return np.clip((values - low) / scale, 0.0, 1.0)


def choose_representative_pairs(
    features: np.ndarray,
    timestamps: np.ndarray,
    q7: np.ndarray,
    eligible: np.ndarray,
    count: int,
    minimum_time_gap_s: float,
) -> np.ndarray:
    eligible_indices = np.flatnonzero(eligible)
    if len(eligible_indices) < count:
        raise RuntimeError(
            f"Only {len(eligible_indices)} eligible pairs for {count} anchors"
        )

    # With only ten anchors, forcing every distal-joint extremum consumes the
    # full budget and can select visually redundant neighbors.  Keep only the
    # sequence boundaries and jaw extrema mandatory; normalized farthest-point
    # sampling then spends the remaining budget on image-relevant pose/time
    # diversity, including roll and wrist motion.
    seed_candidates = [
        int(eligible_indices[0]),
        int(eligible_indices[-1]),
    ]
    jaw_values = q7[eligible_indices, 6]
    seed_candidates.extend(
        [
            int(eligible_indices[int(np.argmin(jaw_values))]),
            int(eligible_indices[int(np.argmax(jaw_values))]),
        ]
    )

    selected: list[int] = []

    def far_enough(candidate: int, gap: float) -> bool:
        return all(
            abs(float(timestamps[candidate] - timestamps[index])) >= gap
            for index in selected
        )

    for candidate in seed_candidates:
        if candidate not in selected and far_enough(
            candidate, minimum_time_gap_s
        ):
            selected.append(candidate)
        if len(selected) == count:
            break

    # Farthest-point completion in the normalized observation/kinematic space.
    while len(selected) < count:
        remaining = np.asarray(
            [index for index in eligible_indices if index not in selected],
            dtype=np.int64,
        )
        if len(remaining) == 0:
            raise RuntimeError("Representative selection exhausted candidates")
        distances = np.linalg.norm(
            features[remaining, None] - features[np.asarray(selected)][None],
            axis=-1,
        ).min(axis=1)
        order = remaining[np.argsort(distances)[::-1]]
        candidate = next(
            (
                int(index)
                for index in order
                if far_enough(int(index), minimum_time_gap_s)
            ),
            int(order[0]),
        )
        selected.append(candidate)

    return np.asarray(sorted(selected, key=timestamps.__getitem__), dtype=np.int64)


def extract_selected_images(
    bag_path: Path,
    calibration: Any,
    selected_left_indices: np.ndarray,
    selected_right_indices: np.ndarray,
) -> dict[tuple[str, int], np.ndarray]:
    left_targets = {int(value) for value in selected_left_indices}
    right_targets = {int(value) for value in selected_right_indices}
    targets = {LEFT_TOPIC: left_targets, RIGHT_TOPIC: right_targets}
    counters = {LEFT_TOPIC: 0, RIGHT_TOPIC: 0}
    images: dict[tuple[str, int], np.ndarray] = {}
    typestore = get_typestore(Stores.ROS1_NOETIC)

    with Reader(bag_path) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic in targets
        ]
        if {connection.topic for connection in connections} != set(targets):
            raise RuntimeError("Original bag is missing a stereo image topic")
        for connection, _timestamp_ns, raw in reader.messages(
            connections=connections
        ):
            side = "left" if connection.topic == LEFT_TOPIC else "right"
            index = counters[connection.topic]
            if index in targets[connection.topic]:
                message = typestore.deserialize_ros1(raw, connection.msgtype)
                image = image_message_to_bgr(message)
                maps = (
                    calibration.left_maps
                    if side == "left"
                    else calibration.right_maps
                )
                rectified = cv2.remap(
                    image,
                    maps[0],
                    maps[1],
                    interpolation=cv2.INTER_LINEAR,
                )
                images[(side, index)] = rectified
            counters[connection.topic] += 1
            if (
                len(images)
                == len(selected_left_indices) + len(selected_right_indices)
            ):
                break
    return images


def make_contact_sheet(
    output_path: Path,
    rows: list[dict[str, Any]],
    images: dict[tuple[str, int], np.ndarray],
) -> None:
    panels: list[np.ndarray] = []
    panel_width = 720
    for keyframe_index, row in enumerate(rows):
        pair: list[np.ndarray] = []
        for side in ("left", "right"):
            source_index = int(row[f"{side}_index"])
            image = images[(side, source_index)]
            scale = panel_width / (2.0 * image.shape[1])
            thumb = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
            pair.append(thumb)
        panel = np.concatenate(pair, axis=1)
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 38), (0, 0, 0), -1)
        text = (
            f"{keyframe_index:02d}  t={row['pair_time_s']:.3f}s  "
            f"dt={row['stereo_delta_ms']:+.2f}ms  "
            f"roll={row['q7_mid'][3]:+.3f}  jaw={row['q7_mid'][6]:+.3f}"
        )
        cv2.putText(
            panel,
            text,
            (8, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
    sheet = np.concatenate(panels, axis=0)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Failed to write {output_path}")


def main() -> None:
    args = parse_args()
    if args.count < 2:
        raise ValueError("--count must be at least two")
    if not 0.0 <= args.minimum_visibility <= 1.0:
        raise ValueError("--minimum-visibility must be in [0, 1]")
    for path in (
        args.bag,
        args.calibration,
        args.kinematics,
        args.raw_model,
        args.raw_report,
        args.registration,
        args.surface_gaussians,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    registration = json.loads(
        args.registration.read_text(encoding="utf-8")
    )
    if (
        registration.get("passed") is not True
        or registration.get("version") != "raw_p420006"
    ):
        raise RuntimeError("P420006 registration has not passed")

    raw_report = json.loads(args.raw_report.read_text(encoding="utf-8"))
    if raw_report.get("passed") is not True:
        raise RuntimeError("Frozen raw kinematic report has not passed")
    verify_frozen_raw_input(
        args.bag,
        raw_report["inputs"]["bag"],
        "Original bag",
    )
    verify_frozen_raw_input(
        args.calibration,
        raw_report["inputs"]["camera_calibration"],
        "Original camera calibration",
    )
    raw_model = json.loads(args.raw_model.read_text(encoding="utf-8"))
    T_right_left = np.asarray(
        raw_model["calibration_and_static_transforms"][
            "T_rectified_right_camera_rectified_left_camera"
        ],
        dtype=np.float64,
    )
    if T_right_left.shape != (4, 4):
        raise RuntimeError("Invalid frozen rectified stereo transform")

    calibration = load_stereo_calibration(args.calibration)
    width, height = calibration.image_size
    with np.load(args.kinematics, allow_pickle=False) as raw:
        stereo_left_index = raw["stereo_left_index"].astype(np.int64)
        stereo_right_index = raw["stereo_right_index"].astype(np.int64)
        left_timestamps_ns = raw["left_timestamps_ros_ns"].astype(np.int64)
        right_timestamps_ns = raw["right_timestamps_ros_ns"].astype(np.int64)
        joint_timestamps_ns = raw["joint_timestamps_ros_ns"].astype(np.int64)
        q7 = raw["q7"].astype(np.float64)
        T_left_lnd = raw[
            "T_rectified_left_camera_lnd_link"
        ].astype(np.float64)
        T_right_lnd = raw[
            "T_rectified_right_camera_lnd_link"
        ].astype(np.float64)

    left_pair_ns = left_timestamps_ns[stereo_left_index]
    right_pair_ns = right_timestamps_ns[stereo_right_index]
    pair_ns = left_pair_ns + (right_pair_ns - left_pair_ns) // 2
    pair_q_index = nearest_indices(pair_ns, joint_timestamps_ns)
    left_q_index = nearest_indices(left_pair_ns, joint_timestamps_ns)
    right_q_index = nearest_indices(right_pair_ns, joint_timestamps_ns)
    time_origin_ns = int(joint_timestamps_ns[0])
    pair_time_s = (pair_ns - time_origin_ns).astype(np.float64) * 1.0e-9

    with np.load(args.surface_gaussians, allow_pickle=False) as surface:
        local_means_all = surface["means"].astype(np.float64)
        gaussian_link_ids_all = surface["link_ids"].astype(np.int64)
        gaussian_link_names = surface["link_names"].tolist()
    distal_gaussians = gaussian_link_ids_all != gaussian_link_names.index(
        "PSM1_tool_main_link"
    )
    local_means = local_means_all[distal_gaussians]
    gaussian_link_ids = gaussian_link_ids_all[distal_gaussians]

    driver_path = P420_ROOT / "psm_p420006_gui_pose_driver.npz"
    with np.load(driver_path, allow_pickle=False) as driver:
        lnd_link_ids = driver["lnd_link_ids"].astype(np.int64)
        link_offsets = driver["T_lndlink_urdf_link"].astype(np.float64)
        driver_link_names = driver["link_names"].tolist()
    if driver_link_names != gaussian_link_names:
        raise RuntimeError("P420006 driver/surface link names disagree")

    visibility_left = np.empty(len(pair_ns), dtype=np.float64)
    visibility_right = np.empty(len(pair_ns), dtype=np.float64)
    extent_left = np.empty(len(pair_ns), dtype=np.float64)
    extent_right = np.empty(len(pair_ns), dtype=np.float64)
    distal_position = np.empty((len(pair_ns), 3), dtype=np.float64)
    distal_link_index = driver_link_names.index(
        "PSM1_tool_wrist_sca_shaft_link"
    )
    for pair_index in range(len(pair_ns)):
        T_left_links = (
            T_left_lnd[left_q_index[pair_index], lnd_link_ids]
            @ link_offsets
        )
        T_right_links = (
            T_right_lnd[right_q_index[pair_index], lnd_link_ids]
            @ link_offsets
        )
        left_points = transform_surface(
            local_means, gaussian_link_ids, T_left_links
        )
        right_points = transform_surface(
            local_means, gaussian_link_ids, T_right_links
        )
        (
            visibility_left[pair_index],
            extent_left[pair_index],
        ) = project_visibility(
            left_points, calibration.K_left_rect, width, height
        )
        (
            visibility_right[pair_index],
            extent_right[pair_index],
        ) = project_visibility(
            right_points, calibration.K_right_rect, width, height
        )
        distal_position[pair_index] = T_left_links[
            distal_link_index, :3, 3
        ]

    minimum_extent_px = 30.0
    eligible = (
        (visibility_left >= args.minimum_visibility)
        & (visibility_right >= args.minimum_visibility)
        & (extent_left >= minimum_extent_px)
        & (extent_right >= minimum_extent_px)
    )
    raw_features = np.column_stack(
        [
            pair_time_s,
            distal_position,
            q7[pair_q_index, 3:7],
        ]
    )
    features = robust_unit_scale(raw_features)
    # Do not allow time alone to dominate geometric diversity.
    features[:, 0] *= 0.5
    selected_pair_slots = choose_representative_pairs(
        features,
        pair_time_s,
        q7[pair_q_index],
        eligible,
        args.count,
        args.minimum_time_gap_s,
    )
    selected_left = stereo_left_index[selected_pair_slots]
    selected_right = stereo_right_index[selected_pair_slots]

    print(
        f"Extracting {len(selected_pair_slots)} anchors from "
        f"{len(pair_ns)} strict stereo pairs"
    )
    selected_images = extract_selected_images(
        args.bag,
        calibration,
        selected_left,
        selected_right,
    )
    expected_image_count = 2 * len(selected_pair_slots)
    if len(selected_images) != expected_image_count:
        raise RuntimeError(
            f"Extracted {len(selected_images)}/{expected_image_count} images"
        )

    prepare_output(args.output_dir, args.overwrite)
    manifest_rows: list[dict[str, Any]] = []
    for keyframe_index, pair_slot in enumerate(selected_pair_slots):
        left_index = int(stereo_left_index[pair_slot])
        right_index = int(stereo_right_index[pair_slot])
        row = {
            "keyframe_index": keyframe_index,
            "strict_pair_slot": int(pair_slot),
            "left_index": left_index,
            "right_index": right_index,
            "left_timestamp_ros_ns": int(left_pair_ns[pair_slot]),
            "right_timestamp_ros_ns": int(right_pair_ns[pair_slot]),
            "pair_timestamp_ros_ns": int(pair_ns[pair_slot]),
            "pair_time_s": float(pair_time_s[pair_slot]),
            "stereo_delta_ms": float(
                (left_pair_ns[pair_slot] - right_pair_ns[pair_slot]) * 1.0e-6
            ),
            "left_q_index": int(left_q_index[pair_slot]),
            "right_q_index": int(right_q_index[pair_slot]),
            "pair_q_index": int(pair_q_index[pair_slot]),
            "q7_mid": q7[pair_q_index[pair_slot]].tolist(),
            "left_visibility": float(visibility_left[pair_slot]),
            "right_visibility": float(visibility_right[pair_slot]),
            "left_extent_px": float(extent_left[pair_slot]),
            "right_extent_px": float(extent_right[pair_slot]),
            "left_image": (
                f"images/keyframe_{keyframe_index:02d}_left_rectified.png"
            ),
            "right_image": (
                f"images/keyframe_{keyframe_index:02d}_right_rectified.png"
            ),
        }
        for side, source_index in (
            ("left", left_index),
            ("right", right_index),
        ):
            output_path = (
                args.output_dir
                / "images"
                / f"keyframe_{keyframe_index:02d}_{side}_rectified.png"
            )
            if not cv2.imwrite(
                str(output_path),
                selected_images[(side, source_index)],
                [cv2.IMWRITE_PNG_COMPRESSION, 3],
            ):
                raise RuntimeError(f"Failed to write {output_path}")
            row[f"{side}_image_sha256"] = sha256(output_path)
        manifest_rows.append(row)

    np.savez_compressed(
        args.output_dir / "keyframes.npz",
        schema=np.asarray("super_p420006_stereo_keyframes_v1"),
        strict_pair_slot=selected_pair_slots,
        left_index=selected_left,
        right_index=selected_right,
        left_timestamp_ros_ns=left_pair_ns[selected_pair_slots],
        right_timestamp_ros_ns=right_pair_ns[selected_pair_slots],
        pair_timestamp_ros_ns=pair_ns[selected_pair_slots],
        pair_time_s=pair_time_s[selected_pair_slots],
        left_q_index=left_q_index[selected_pair_slots],
        right_q_index=right_q_index[selected_pair_slots],
        pair_q_index=pair_q_index[selected_pair_slots],
        q7_mid=q7[pair_q_index[selected_pair_slots]],
        visibility_left=visibility_left[selected_pair_slots],
        visibility_right=visibility_right[selected_pair_slots],
        K_left_rect=calibration.K_left_rect,
        K_right_rect=calibration.K_right_rect,
        T_rectified_right_camera_rectified_left_camera=T_right_left,
    )
    make_contact_sheet(
        args.output_dir / "previews/keyframe_contact_sheet.png",
        manifest_rows,
        selected_images,
    )

    report = {
        "schema": "super_p420006_stereo_keyframe_manifest_v1",
        "passed": True,
        "method": (
            "strict raw stereo pairs + bilateral P420006 visibility gate + "
            "joint/pose/time extrema seeds + normalized farthest-point sampling"
        ),
        "count": len(manifest_rows),
        "source_pair_count": len(pair_ns),
        "eligible_pair_count": int(np.count_nonzero(eligible)),
        "pairing_policy": (
            "maximum-cardinality monotonic raw pairing with |dt| <= 20 ms"
        ),
        "selection": {
            "minimum_visibility": args.minimum_visibility,
            "minimum_extent_px": minimum_extent_px,
            "minimum_time_gap_s": args.minimum_time_gap_s,
            "feature_order": [
                "pair_time_s",
                "distal_x_m",
                "distal_y_m",
                "distal_z_m",
                "raw_roll_rad",
                "raw_wrist_pitch_rad",
                "raw_wrist_yaw_rad",
                "raw_jaw_rad",
            ],
            "forced_extrema": [
                "sequence endpoints",
                "raw jaw min/max",
            ],
        },
        "upstream_method": {
            "repository": UPSTREAM_URL,
            "commit": UPSTREAM_COMMIT,
            "adaptation_boundary": (
                "Use batched rendering/CMA-ES with shared-state calibrated "
                "stereo P420006. Masks and jaw tips are manually annotated; "
                "this pipeline does not run SurgicalSAM2."
            ),
        },
        "inputs": {
            "bag": {
                "path": str(args.bag.resolve()),
                "size_bytes": args.bag.stat().st_size,
                "mtime_ns": args.bag.stat().st_mtime_ns,
                "sha256": raw_report["inputs"]["bag"]["sha256"],
                "sha256_source": str(args.raw_report.resolve()),
            },
            "calibration": {
                "path": str(args.calibration.resolve()),
                "sha256": sha256(args.calibration),
            },
            "kinematics": {
                "path": str(args.kinematics.resolve()),
                "sha256": sha256(args.kinematics),
            },
            "raw_model": {
                "path": str(args.raw_model.resolve()),
                "sha256": sha256(args.raw_model),
            },
            "raw_report": {
                "path": str(args.raw_report.resolve()),
                "sha256": sha256(args.raw_report),
            },
            "registration": {
                "path": str(args.registration.resolve()),
                "sha256": sha256(args.registration),
            },
            "surface_gaussians": {
                "path": str(args.surface_gaussians.resolve()),
                "sha256": sha256(args.surface_gaussians),
            },
        },
        "coordinate_frame": "rectified OpenCV stereo cameras",
        "image_size_wh": [width, height],
        "keyframes": manifest_rows,
        "annotation_status": "pending_manual_stereo_masks_and_jaw_tips",
    }
    write_json(args.output_dir / "pair_manifest.json", report)
    print(
        json.dumps(
            {
                "passed": True,
                "selected": len(manifest_rows),
                "eligible_pairs": int(np.count_nonzero(eligible)),
                "output_dir": str(args.output_dir),
                "contact_sheet": str(
                    args.output_dir
                    / "previews/keyframe_contact_sheet.png"
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
