#!/usr/bin/env python3
"""Build the SUPER PSM kinematic backbone from root inputs only.

This script deliberately does not read anything under ``data/super``.  Its input
allowlist is:

* the original ROS1 bag (stereo images and q7);
* the original OpenCV stereo calibration;
* the original hand-eye calibration; and
* the original LND model.

The output is an LND-native kinematic backbone.  It does not contain an URDF/CAD
registration, image-derived correction, depth-derived correction, table-frame
transform, or a previously generated pose driver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore


LEFT_TOPIC = "/stereo/slave/left/image"
RIGHT_TOPIC = "/stereo/slave/right/image"
JOINT_TOPIC = "/dvrk/PSM1/slave/state_joint_current"
TARGET_TOPICS = {LEFT_TOPIC, RIGHT_TOPIC, JOINT_TOPIC}

EXPECTED_JOINT_NAMES = (
    "outer_yaw",
    "outer_pitch",
    "outer_insertion",
    "outer_roll",
    "outer_wrist_pitch",
    "outer_wrist_yaw",
    "jaw",
)
LND_LINK_NAMES = (
    "base",
    "outer_yaw",
    "outer_pitch",
    "outer_insertion",
    "outer_roll",
    "outer_wrist_pitch",
    "outer_wrist_yaw",
    "jaw_positive_half",
    "jaw_negative_half",
)


@dataclass(frozen=True)
class StereoCalibration:
    image_size: tuple[int, int]
    K1: np.ndarray
    K2: np.ndarray
    D1: np.ndarray
    D2: np.ndarray
    R_raw_right_raw_left: np.ndarray
    t_raw_right_raw_left_m: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q_m: np.ndarray
    K_left_rect: np.ndarray
    K_right_rect: np.ndarray
    baseline_m: float
    left_maps: tuple[np.ndarray, np.ndarray]
    right_maps: tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class BagData:
    time_origin_ros_ns: int
    bag_start_ros_ns: int
    bag_end_ros_ns: int
    left_timestamps_ns: np.ndarray
    right_timestamps_ns: np.ndarray
    joint_timestamps_ns: np.ndarray
    q7: np.ndarray
    q7_velocity: np.ndarray
    q7_effort: np.ndarray
    joint_names: tuple[str, ...]
    validation_images: dict[tuple[str, int], np.ndarray]


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Build an isolated q7 + hand-eye + stereo calibration + LND "
            "kinematic backbone directly from the original ROS bag."
        )
    )
    parser.add_argument(
        "--bag",
        type=Path,
        default=repo / "data/grasp5/grasp5.bag",
        help="Original ROS1 bag containing stereo images and PSM1 q7.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=repo / "data/camera_calibration.yaml",
        help="Original OpenCV stereo calibration. Translation is in millimetres.",
    )
    parser.add_argument(
        "--handeye",
        type=Path,
        default=repo / "data/handeye.yaml",
        help="Original hand-eye calibration. Translation is in millimetres.",
    )
    parser.add_argument(
        "--lnd",
        type=Path,
        default=repo / "data/LND.json",
        help="Original LND Modified-DH model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo
        / "data/super/psm_raw_kinematics_v1（纯机器人学版本）",
        help="New isolated output directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    parser.add_argument(
        "--hash-bag",
        action="store_true",
        help="Compute a full SHA256 of the large bag in addition to its immutable identity.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def input_identity(path: Path, *, full_hash: bool) -> dict[str, Any]:
    stat = path.stat()
    identity: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if full_hash:
        identity["sha256"] = sha256_file(path)
    return identity


def read_opencv_matrix(path: Path, key: str) -> np.ndarray:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(f"Could not open OpenCV YAML: {path}")
    value = storage.getNode(key).mat()
    storage.release()
    if value is None:
        raise KeyError(f"Missing OpenCV matrix {key!r} in {path}")
    return np.asarray(value, dtype=np.float64)


def read_opencv_sequence(path: Path, key: str) -> np.ndarray:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(f"Could not open OpenCV YAML: {path}")
    node = storage.getNode(key)
    if node.empty():
        storage.release()
        raise KeyError(f"Missing OpenCV sequence {key!r} in {path}")
    values = np.asarray([node.at(i).real() for i in range(node.size())], dtype=np.float64)
    storage.release()
    return values


def load_stereo_calibration(path: Path) -> StereoCalibration:
    K1 = read_opencv_matrix(path, "K1")
    K2 = read_opencv_matrix(path, "K2")
    D1 = read_opencv_matrix(path, "D1")
    D2 = read_opencv_matrix(path, "D2")
    R = read_opencv_matrix(path, "R")
    # The root SUPER calibration stores stereo translation in millimetres.
    t_mm = read_opencv_sequence(path, "T").reshape(3, 1)
    image_size_hw = read_opencv_sequence(path, "ImageSize")
    height, width = (int(image_size_hw[0]), int(image_size_hw[1]))

    R1, R2, P1_mm, P2_mm, Q_per_mm, _, _ = cv2.stereoRectify(
        K1,
        D1,
        K2,
        D2,
        (width, height),
        R,
        t_mm,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    baseline_m = float(np.linalg.norm(t_mm) * 1e-3)
    if not 0.001 <= baseline_m <= 0.02:
        raise ValueError(
            f"Unexpected SUPER stereo baseline {baseline_m:.6f} m; "
            "the root calibration translation must be in millimetres."
        )

    # P2 and Q inherit the calibration translation unit.  Convert their metric
    # components so every exported 3-D transform consistently uses metres.
    P1 = P1_mm.copy()
    P2 = P2_mm.copy()
    P2[:, 3] *= 1e-3
    Q_m = Q_per_mm.copy()
    Q_m[3, 2] *= 1e3

    left_maps = cv2.initUndistortRectifyMap(
        K1, D1, R1, P1_mm, (width, height), cv2.CV_32FC1
    )
    right_maps = cv2.initUndistortRectifyMap(
        K2, D2, R2, P2_mm, (width, height), cv2.CV_32FC1
    )
    return StereoCalibration(
        image_size=(width, height),
        K1=K1,
        K2=K2,
        D1=D1.reshape(-1),
        D2=D2.reshape(-1),
        R_raw_right_raw_left=R,
        t_raw_right_raw_left_m=t_mm.reshape(3) * 1e-3,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q_m=Q_m,
        K_left_rect=P1[:, :3],
        K_right_rect=P2[:, :3],
        baseline_m=baseline_m,
        left_maps=left_maps,
        right_maps=right_maps,
    )


def parse_handeye(path: Path) -> tuple[np.ndarray, np.ndarray]:
    text = path.read_text(encoding="utf-8")
    arrays: dict[str, np.ndarray] = {}
    for match in re.finditer(r"(\w+):\s*\[(.*?)\]", text, flags=re.DOTALL):
        values = re.findall(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
            match.group(2),
        )
        arrays[match.group(1)] = np.asarray([float(value) for value in values])
    for key in ("PSM1_rvec", "PSM1_tvec"):
        if key not in arrays or arrays[key].shape != (3,):
            raise ValueError(f"Expected three values for {key} in {path}")
    rotation, _ = cv2.Rodrigues(arrays["PSM1_rvec"].reshape(3, 1))
    translation_m = arrays["PSM1_tvec"] * 1e-3
    return rotation, translation_m


def strip_json_comments(text: str) -> str:
    # LND.json contains C++-style end-of-line comments but no string values with
    # literal //, so a line-level removal keeps parsing deterministic.
    return re.sub(r"//.*$", "", text, flags=re.MULTILINE)


def load_lnd(path: Path) -> dict[str, Any]:
    text = strip_json_comments(path.read_text(encoding="utf-8"))
    text = re.sub(r",\s*([}\]])", r"\1", text)
    lnd = json.loads(text)
    dh_params = lnd.get("DH_params", [])
    if len(dh_params) != 6:
        raise ValueError(f"Expected six LND DH joints, found {len(dh_params)}")
    if not bool(lnd.get("has_gripper")):
        raise ValueError("This pipeline expects the LND PSM gripper definition")
    return lnd


def validation_indices(count: int) -> set[int]:
    if count <= 0:
        return set()
    return {0, count // 2, count - 1}


def image_message_to_bgr(message: Any) -> np.ndarray:
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size != height * step:
        raise ValueError(
            f"Image payload has {raw.size} bytes, expected height*step={height * step}"
        )
    rows = raw.reshape(height, step)
    encoding = str(message.encoding).lower()
    if encoding in {"rgb8", "bgr8"}:
        image = rows[:, : width * 3].reshape(height, width, 3)
        if encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image.copy()
    if encoding in {"mono8", "8uc1"}:
        mono = rows[:, :width].reshape(height, width)
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unsupported source image encoding {message.encoding!r}")


def read_raw_bag(path: Path) -> BagData:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    timestamps: dict[str, list[int]] = {
        LEFT_TOPIC: [],
        RIGHT_TOPIC: [],
        JOINT_TOPIC: [],
    }
    positions: list[list[float]] = []
    velocities: list[list[float]] = []
    efforts: list[list[float]] = []
    joint_names: tuple[str, ...] | None = None
    images: dict[tuple[str, int], np.ndarray] = {}

    with Reader(path) as reader:
        connections_by_topic = {
            connection.topic: connection
            for connection in reader.connections
            if connection.topic in TARGET_TOPICS
        }
        if set(connections_by_topic) != TARGET_TOPICS:
            missing = sorted(TARGET_TOPICS - set(connections_by_topic))
            raise RuntimeError(f"Missing required bag topics: {missing}")
        image_targets = {
            LEFT_TOPIC: validation_indices(connections_by_topic[LEFT_TOPIC].msgcount),
            RIGHT_TOPIC: validation_indices(connections_by_topic[RIGHT_TOPIC].msgcount),
        }
        counters = {LEFT_TOPIC: 0, RIGHT_TOPIC: 0}
        connections = list(connections_by_topic.values())
        bag_start_ns = int(reader.start_time)
        bag_end_ns = int(reader.end_time)

        for connection, timestamp_ns_raw, raw in reader.messages(connections=connections):
            timestamp_ns = int(timestamp_ns_raw)
            topic = connection.topic
            timestamps[topic].append(timestamp_ns)
            if topic in (LEFT_TOPIC, RIGHT_TOPIC):
                index = counters[topic]
                if index in image_targets[topic]:
                    message = typestore.deserialize_ros1(raw, connection.msgtype)
                    side = "left" if topic == LEFT_TOPIC else "right"
                    images[(side, index)] = image_message_to_bgr(message)
                counters[topic] += 1
                continue

            message = typestore.deserialize_ros1(raw, connection.msgtype)
            names = tuple(str(name) for name in message.name)
            if joint_names is None:
                joint_names = names
            elif names != joint_names:
                raise RuntimeError("Joint names changed within the raw bag")
            positions.append([float(value) for value in message.position])
            velocities.append([float(value) for value in message.velocity])
            efforts.append([float(value) for value in message.effort])

    if joint_names is None:
        raise RuntimeError("The raw bag contains no PSM1 joint states")
    if joint_names != EXPECTED_JOINT_NAMES:
        raise ValueError(
            f"Unexpected q7 joint order {joint_names}; expected {EXPECTED_JOINT_NAMES}"
        )
    q7 = np.asarray(positions, dtype=np.float64)
    q7_velocity = np.asarray(velocities, dtype=np.float64)
    q7_effort = np.asarray(efforts, dtype=np.float64)
    if q7.shape[1:] != (7,) or q7_velocity.shape != q7.shape or q7_effort.shape != q7.shape:
        raise ValueError(
            "Expected position, velocity and effort arrays with identical (N, 7) shape"
        )

    left_ns = np.asarray(timestamps[LEFT_TOPIC], dtype=np.int64)
    right_ns = np.asarray(timestamps[RIGHT_TOPIC], dtype=np.int64)
    joint_ns = np.asarray(timestamps[JOINT_TOPIC], dtype=np.int64)
    time_origin_ns = int(min(left_ns[0], right_ns[0], joint_ns[0]))
    return BagData(
        time_origin_ros_ns=time_origin_ns,
        bag_start_ros_ns=bag_start_ns,
        bag_end_ros_ns=bag_end_ns,
        left_timestamps_ns=left_ns,
        right_timestamps_ns=right_ns,
        joint_timestamps_ns=joint_ns,
        q7=q7,
        q7_velocity=q7_velocity,
        q7_effort=q7_effort,
        joint_names=joint_names,
        validation_images=images,
    )


def rotx(theta: float) -> np.ndarray:
    cosine, sine = math.cos(theta), math.sin(theta)
    return np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cosine, -sine, 0.0],
            [0.0, sine, cosine, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotz(theta: float) -> np.ndarray:
    cosine, sine = math.cos(theta), math.sin(theta)
    return np.asarray(
        [
            [cosine, -sine, 0.0, 0.0],
            [sine, cosine, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def transx(distance: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = distance
    return transform


def transz(distance: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[2, 3] = distance
    return transform


def modified_dh(alpha: float, a: float, theta: float, d: float) -> np.ndarray:
    """Craig modified DH: Rx(alpha) Tx(a) Rz(theta) Tz(d)."""
    return rotx(alpha) @ transx(a) @ rotz(theta) @ transz(d)


def lnd_forward_kinematics(lnd: dict[str, Any], q7: np.ndarray) -> np.ndarray:
    if q7.shape != (7,):
        raise ValueError(f"Expected q7 shape (7,), got {q7.shape}")
    transforms = np.empty((9, 4, 4), dtype=np.float64)
    transforms[0] = np.eye(4, dtype=np.float64)
    current = transforms[0].copy()
    for index, dh in enumerate(lnd["DH_params"], start=1):
        theta0 = float(dh.get("theta", 0.0))
        d0 = float(dh.get("D", 0.0))
        offset = float(dh.get("offset", 0.0))
        if dh["type"] == "revolute":
            theta = theta0 + float(q7[index - 1]) + offset
            d = d0
        elif dh["type"] == "prismatic":
            theta = theta0
            d = d0 + float(q7[index - 1]) + offset
        else:
            raise ValueError(f"Unsupported LND joint type {dh['type']!r}")
        current = current @ modified_dh(
            float(dh.get("alpha", 0.0)),
            float(dh.get("A", 0.0)),
            theta,
            d,
        )
        transforms[index] = current

    # This is the gripper convention in the original LND RobotLink model:
    # parallel children of link 6 with +jaw/2 and -jaw/2 rotations.
    transforms[7] = transforms[6] @ rotz(0.5 * float(q7[6]))
    transforms[8] = transforms[6] @ rotz(-0.5 * float(q7[6]))
    return transforms


def build_lnd_transforms(lnd: dict[str, Any], q7: np.ndarray) -> np.ndarray:
    result = np.empty((len(q7), 9, 4, 4), dtype=np.float64)
    for index, state in enumerate(q7):
        result[index] = lnd_forward_kinematics(lnd, state)
    return result


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def transform_series(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.einsum("ij,nljk->nlik", left, right)


def nearest_indices(query_ns: np.ndarray, reference_ns: np.ndarray) -> np.ndarray:
    right = np.searchsorted(reference_ns, query_ns, side="left")
    right = np.clip(right, 0, len(reference_ns) - 1)
    left = np.clip(right - 1, 0, len(reference_ns) - 1)
    choose_left = np.abs(query_ns - reference_ns[left]) <= np.abs(
        reference_ns[right] - query_ns
    )
    return np.where(choose_left, left, right).astype(np.int64)


def zoh_indices(query_ns: np.ndarray, reference_ns: np.ndarray) -> np.ndarray:
    return np.clip(
        np.searchsorted(reference_ns, query_ns, side="right") - 1,
        0,
        len(reference_ns) - 1,
    ).astype(np.int64)


def paired_stereo_indices(
    left_ns: np.ndarray,
    right_ns: np.ndarray,
    *,
    maximum_delta_ns: int = 20_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Maximum-cardinality monotonic pairing under a timestamp gate.

    The raw grasp5 bag starts the right stream one message before the left
    stream.  Equal array indices are therefore not stereo pairs.  For sorted
    streams, consuming the earliest mutually feasible pair gives a
    maximum-cardinality monotonic matching under the fixed time gate.
    """
    left_indices: list[int] = []
    right_indices: list[int] = []
    left_index = 0
    right_index = 0
    while left_index < len(left_ns) and right_index < len(right_ns):
        left_time = int(left_ns[left_index])
        right_time = int(right_ns[right_index])
        if right_time < left_time - maximum_delta_ns:
            right_index += 1
        elif left_time < right_time - maximum_delta_ns:
            left_index += 1
        else:
            left_indices.append(left_index)
            right_indices.append(right_index)
            left_index += 1
            right_index += 1
    if not left_indices:
        raise RuntimeError(
            f"No stereo pairs satisfy the {maximum_delta_ns * 1e-6:.3f} ms gate"
        )
    return (
        np.asarray(left_indices, dtype=np.int64),
        np.asarray(right_indices, dtype=np.int64),
    )


def relative_seconds(timestamps_ns: np.ndarray, origin_ns: int) -> np.ndarray:
    return (timestamps_ns - origin_ns).astype(np.float64) * 1e-9


def point_feature_positions(
    lnd: dict[str, Any], link_transforms: np.ndarray
) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    positions: list[np.ndarray] = []
    for feature in lnd.get("point_features", []):
        names.append(str(feature["name"]))
        link = int(feature["link"])
        local = np.r_[np.asarray(feature["position"], dtype=np.float64), 1.0]
        positions.append((link_transforms[link] @ local)[:3])
    return names, np.asarray(positions, dtype=np.float64)


def skeleton_positions(
    lnd: dict[str, Any], link_transforms: np.ndarray
) -> np.ndarray:
    segments: list[np.ndarray] = []
    for segment in lnd.get("skeleton_structure", []):
        endpoints = []
        for suffix in ("1", "2"):
            link = int(segment[f"link{suffix}"])
            local = np.r_[np.asarray(segment[f"position{suffix}"], dtype=np.float64), 1.0]
            endpoints.append((link_transforms[link] @ local)[:3])
        segments.append(np.stack(endpoints))
    return np.asarray(segments, dtype=np.float64)


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def project_points(K: np.ndarray, points_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive = points_camera[:, 2] > 1e-9
    pixels = np.full((len(points_camera), 2), np.nan, dtype=np.float64)
    pixels[positive] = (
        points_camera[positive, :2] / points_camera[positive, 2, None]
    ) * np.asarray([K[0, 0], K[1, 1]])
    pixels[positive, 0] += K[0, 2]
    pixels[positive, 1] += K[1, 2]
    return pixels, positive


def draw_validation_overlay(
    image: np.ndarray,
    K: np.ndarray,
    T_camera_psm: np.ndarray,
    link_transforms: np.ndarray,
    lnd: dict[str, Any],
    title: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    overlay = image.copy()
    width, height = image.shape[1], image.shape[0]
    feature_names, features_psm = point_feature_positions(lnd, link_transforms)
    features_camera = transform_points(T_camera_psm, features_psm)
    feature_pixels, feature_positive = project_points(K, features_camera)

    skeleton_psm = skeleton_positions(lnd, link_transforms)
    skeleton_camera = transform_points(
        T_camera_psm, skeleton_psm.reshape(-1, 3)
    ).reshape(skeleton_psm.shape)
    skeleton_pixels, skeleton_positive = project_points(
        K, skeleton_camera.reshape(-1, 3)
    )
    skeleton_pixels = skeleton_pixels.reshape(len(skeleton_psm), 2, 2)
    skeleton_positive = skeleton_positive.reshape(len(skeleton_psm), 2)

    for index, segment in enumerate(skeleton_pixels):
        if not bool(np.all(skeleton_positive[index])):
            continue
        point_a = tuple(np.rint(segment[0]).astype(int))
        point_b = tuple(np.rint(segment[1]).astype(int))
        cv2.line(overlay, point_a, point_b, (0, 255, 255), 2, cv2.LINE_AA)

    visible = []
    for name, pixel, positive in zip(feature_names, feature_pixels, feature_positive):
        in_image = bool(
            positive and 0 <= pixel[0] < width and 0 <= pixel[1] < height
        )
        if not in_image:
            continue
        visible.append(name)
        point = tuple(np.rint(pixel).astype(int))
        cv2.circle(overlay, point, 5, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            name,
            (point[0] + 7, point[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        overlay,
        title,
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (30, 220, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay, {
        "feature_count": len(feature_names),
        "visible_feature_count": len(visible),
        "visible_features": visible,
        "minimum_feature_depth_m": float(np.min(features_camera[:, 2])),
        "maximum_feature_depth_m": float(np.max(features_camera[:, 2])),
        "feature_pixels": {
            name: pixel.tolist() for name, pixel in zip(feature_names, feature_pixels)
        },
    }


def rotation_gate(transforms: np.ndarray) -> dict[str, float]:
    rotations = transforms[..., :3, :3].reshape(-1, 3, 3)
    identity = np.eye(3, dtype=np.float64)
    orthogonality = np.max(
        np.abs(np.swapaxes(rotations, 1, 2) @ rotations - identity)
    )
    determinant_error = np.max(np.abs(np.linalg.det(rotations) - 1.0))
    return {
        "max_orthogonality_error": float(orthogonality),
        "max_determinant_error": float(determinant_error),
    }


def motion_continuity_gate(
    timestamps_s: np.ndarray,
    q7: np.ndarray,
    q7_velocity: np.ndarray,
    transforms: np.ndarray,
) -> dict[str, Any]:
    delta_time = np.diff(timestamps_s)
    translations = transforms[:, :, :3, 3]
    translation_step_m = np.linalg.norm(np.diff(translations, axis=0), axis=-1)
    previous_rotation = transforms[:-1, :, :3, :3]
    next_rotation = transforms[1:, :, :3, :3]
    relative_rotation = np.einsum(
        "nlji,nljk->nlik", previous_rotation, next_rotation
    )
    rotation_step_deg = np.degrees(
        np.arccos(
            np.clip(
                (np.trace(relative_rotation, axis1=-2, axis2=-1) - 1.0) * 0.5,
                -1.0,
                1.0,
            )
        )
    )
    finite_difference_velocity = np.diff(q7, axis=0) / delta_time[:, None]
    reported_midpoint_velocity = 0.5 * (q7_velocity[:-1] + q7_velocity[1:])
    velocity_correlation = [
        float(
            np.corrcoef(
                finite_difference_velocity[:, index],
                reported_midpoint_velocity[:, index],
            )[0, 1]
        )
        for index in range(q7.shape[1])
    ]
    jaw_delta = np.abs(np.diff(q7[:, 6]))
    largest_jaw_step_index = int(np.argmax(jaw_delta))
    serial_rotation = rotation_step_deg[:, :7]
    jaw_rotation = rotation_step_deg[:, 7:]
    passed = bool(
        np.all(delta_time > 0)
        and np.max(translation_step_m) < 0.001
        and np.max(serial_rotation) < 5.0
        and np.max(jaw_rotation) < 20.0
        and min(velocity_correlation) > 0.85
    )
    return {
        "passed": passed,
        "gate_limits": {
            "maximum_link_translation_step_mm": 1.0,
            "maximum_serial_link_rotation_step_deg": 5.0,
            "maximum_jaw_link_rotation_step_deg": 20.0,
            "minimum_q_finite_difference_vs_reported_velocity_correlation": 0.85,
        },
        "q_timestamp_interval_ms": {
            "minimum": float(np.min(delta_time) * 1e3),
            "p50": float(np.percentile(delta_time, 50) * 1e3),
            "p95": float(np.percentile(delta_time, 95) * 1e3),
            "maximum": float(np.max(delta_time) * 1e3),
        },
        "link_translation_step_mm": {
            "p50": float(np.percentile(translation_step_m, 50) * 1e3),
            "p95": float(np.percentile(translation_step_m, 95) * 1e3),
            "maximum": float(np.max(translation_step_m) * 1e3),
            "maximum_per_link": (
                np.max(translation_step_m, axis=0) * 1e3
            ).tolist(),
        },
        "link_rotation_step_deg": {
            "p50": float(np.percentile(rotation_step_deg, 50)),
            "p95": float(np.percentile(rotation_step_deg, 95)),
            "maximum_serial_links_0_to_6": float(np.max(serial_rotation)),
            "maximum_jaw_links_7_to_8": float(np.max(jaw_rotation)),
            "maximum_per_link": np.max(rotation_step_deg, axis=0).tolist(),
        },
        "q_finite_difference_vs_reported_velocity_correlation": dict(
            zip(EXPECTED_JOINT_NAMES, velocity_correlation)
        ),
        "largest_raw_jaw_step": {
            "state_index_before": largest_jaw_step_index,
            "state_index_after": largest_jaw_step_index + 1,
            "timestamp_before_s": float(timestamps_s[largest_jaw_step_index]),
            "timestamp_after_s": float(timestamps_s[largest_jaw_step_index + 1]),
            "jaw_before_rad": float(q7[largest_jaw_step_index, 6]),
            "jaw_after_rad": float(q7[largest_jaw_step_index + 1, 6]),
            "jaw_delta_rad": float(
                q7[largest_jaw_step_index + 1, 6]
                - q7[largest_jaw_step_index, 6]
            ),
            "reported_velocity_before_rad_s": float(
                q7_velocity[largest_jaw_step_index, 6]
            ),
            "reported_velocity_after_rad_s": float(
                q7_velocity[largest_jaw_step_index + 1, 6]
            ),
            "interpretation": (
                "Retained raw fast jaw motion: adjacent reported velocities have "
                "the same sign and the full jaw velocity correlation gate passes."
            ),
        },
    }


def percentile_ms(delta_ns: np.ndarray) -> dict[str, float]:
    values = np.abs(delta_ns).astype(np.float64) * 1e-6
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)),
    }


def calibration_to_json(
    calibration: StereoCalibration,
    T_raw_left_psm: np.ndarray,
    T_left_rect_psm: np.ndarray,
    T_right_rect_psm: np.ndarray,
    T_right_rect_left_rect: np.ndarray,
) -> dict[str, Any]:
    return {
        "source_translation_unit": "millimetre",
        "exported_length_unit": "metre",
        "image_size": list(calibration.image_size),
        "K1": calibration.K1.tolist(),
        "K2": calibration.K2.tolist(),
        "D1": calibration.D1.tolist(),
        "D2": calibration.D2.tolist(),
        "R_raw_right_raw_left": calibration.R_raw_right_raw_left.tolist(),
        "t_raw_right_raw_left_m": calibration.t_raw_right_raw_left_m.tolist(),
        "R1_raw_left_to_rectified_left": calibration.R1.tolist(),
        "R2_raw_right_to_rectified_right": calibration.R2.tolist(),
        "P1_m": calibration.P1.tolist(),
        "P2_m": calibration.P2.tolist(),
        "Q_per_m": calibration.Q_m.tolist(),
        "K_left_rect": calibration.K_left_rect.tolist(),
        "K_right_rect": calibration.K_right_rect.tolist(),
        "baseline_m": calibration.baseline_m,
        "T_raw_left_camera_psm_base": T_raw_left_psm.tolist(),
        "T_rectified_left_camera_psm_base": T_left_rect_psm.tolist(),
        "T_rectified_right_camera_psm_base": T_right_rect_psm.tolist(),
        "T_rectified_right_camera_rectified_left_camera": (
            T_right_rect_left_rect.tolist()
        ),
    }


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root_inputs = (args.bag, args.calibration, args.handeye, args.lnd)
    for path in root_inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    # Reject accidental use of any prior data/super derivative as an input.
    super_root = (Path(__file__).resolve().parents[1] / "data/super").resolve()
    for path in root_inputs:
        resolved = path.resolve()
        if resolved == super_root or super_root in resolved.parents:
            raise ValueError(
                f"Refusing derived input under data/super: {resolved}. "
                "Use only the four root source files."
            )

    prepare_output_dir(args.output_dir, args.overwrite)
    snapshots = args.output_dir / "root_inputs_snapshot"
    snapshots.mkdir()
    shutil.copy2(args.calibration, snapshots / args.calibration.name)
    shutil.copy2(args.handeye, snapshots / args.handeye.name)
    shutil.copy2(args.lnd, snapshots / args.lnd.name)

    print("[1/5] Reading root calibration, hand-eye and LND")
    calibration = load_stereo_calibration(args.calibration)
    handeye_rotation, handeye_translation_m = parse_handeye(args.handeye)
    lnd = load_lnd(args.lnd)

    print("[2/5] Scanning the original bag (all stereo timestamps and all q7)")
    bag = read_raw_bag(args.bag)
    for timestamps, name in (
        (bag.left_timestamps_ns, "left image"),
        (bag.right_timestamps_ns, "right image"),
        (bag.joint_timestamps_ns, "q7"),
    ):
        if len(timestamps) < 2 or not np.all(np.diff(timestamps) > 0):
            raise RuntimeError(f"{name} timestamps are not strictly increasing")

    print("[3/5] Computing LND FK and raw-camera/rectified-camera transforms")
    T_psm_links = build_lnd_transforms(lnd, bag.q7)
    T_raw_left_psm = make_transform(handeye_rotation, handeye_translation_m)
    T_left_rect_raw_left = make_transform(calibration.R1, np.zeros(3))
    T_raw_right_raw_left = make_transform(
        calibration.R_raw_right_raw_left,
        calibration.t_raw_right_raw_left_m,
    )
    T_right_rect_raw_right = make_transform(calibration.R2, np.zeros(3))
    T_left_rect_psm = T_left_rect_raw_left @ T_raw_left_psm
    T_right_rect_psm = (
        T_right_rect_raw_right @ T_raw_right_raw_left @ T_raw_left_psm
    )
    T_right_rect_left_rect = (
        T_right_rect_raw_right
        @ T_raw_right_raw_left
        @ np.linalg.inv(T_left_rect_raw_left)
    )
    T_left_rect_links = transform_series(T_left_rect_psm, T_psm_links)
    T_right_rect_links = transform_series(T_right_rect_psm, T_psm_links)

    stereo_left_index, stereo_right_index = paired_stereo_indices(
        bag.left_timestamps_ns, bag.right_timestamps_ns
    )
    unpaired_left_index = np.setdiff1d(
        np.arange(len(bag.left_timestamps_ns), dtype=np.int64),
        stereo_left_index,
        assume_unique=True,
    )
    unpaired_right_index = np.setdiff1d(
        np.arange(len(bag.right_timestamps_ns), dtype=np.int64),
        stereo_right_index,
        assume_unique=True,
    )
    left_q_nearest = nearest_indices(
        bag.left_timestamps_ns, bag.joint_timestamps_ns
    )
    right_q_nearest = nearest_indices(
        bag.right_timestamps_ns, bag.joint_timestamps_ns
    )
    left_q_zoh = zoh_indices(bag.left_timestamps_ns, bag.joint_timestamps_ns)
    right_q_zoh = zoh_indices(bag.right_timestamps_ns, bag.joint_timestamps_ns)

    calibration_json = calibration_to_json(
        calibration,
        T_raw_left_psm,
        T_left_rect_psm,
        T_right_rect_psm,
        T_right_rect_left_rect,
    )
    model_json = {
        "schema": "super_psm_raw_kinematics_model_v1",
        "scope": "LND-native kinematic backbone only",
        "input_policy": {
            "allowed": [
                "original ROS bag stereo image topics",
                "original ROS bag PSM1 q7 topic",
                "root camera_calibration.yaml",
                "root handeye.yaml",
                "root LND.json",
            ],
            "forbidden": [
                "data/super derived images, videos, timestamps or joints",
                "prior LND intermediates or pose drivers",
                "URDF/CAD registration",
                "image/depth/manual pose correction",
                "table/world coordinate transforms",
            ],
        },
        "length_unit": "metre",
        "angle_unit": "radian",
        "transform_convention": (
            "T_A_B maps homogeneous coordinates from frame B into frame A"
        ),
        "dh_convention": "Craig modified DH: Rx(alpha) Tx(A) Rz(theta) Tz(D)",
        "q7_joint_names": list(bag.joint_names),
        "lnd_link_names": list(LND_LINK_NAMES),
        "lnd_link_index_semantics": {
            "0": "PSM base before the first DH transform",
            "1..6": "serial Modified-DH links after q1..q6",
            "7": "parallel gripper child of link 6 at +jaw/2",
            "8": "parallel gripper child of link 6 at -jaw/2",
        },
        "lnd": lnd,
        "calibration_and_static_transforms": calibration_json,
    }
    write_json(args.output_dir / "model.json", model_json)

    np.savez_compressed(
        args.output_dir / "kinematics.npz",
        schema=np.asarray("super_psm_raw_kinematics_v1"),
        time_origin_ros_ns=np.asarray(bag.time_origin_ros_ns, dtype=np.int64),
        bag_start_ros_ns=np.asarray(bag.bag_start_ros_ns, dtype=np.int64),
        bag_end_ros_ns=np.asarray(bag.bag_end_ros_ns, dtype=np.int64),
        joint_names=np.asarray(bag.joint_names),
        lnd_link_names=np.asarray(LND_LINK_NAMES),
        joint_timestamps_ros_ns=bag.joint_timestamps_ns,
        joint_timestamps_s=relative_seconds(
            bag.joint_timestamps_ns, bag.time_origin_ros_ns
        ),
        q7=bag.q7,
        q7_velocity=bag.q7_velocity,
        q7_effort=bag.q7_effort,
        T_psm_base_lnd_link=T_psm_links,
        T_rectified_left_camera_lnd_link=T_left_rect_links,
        T_rectified_right_camera_lnd_link=T_right_rect_links,
        left_timestamps_ros_ns=bag.left_timestamps_ns,
        left_timestamps_s=relative_seconds(
            bag.left_timestamps_ns, bag.time_origin_ros_ns
        ),
        right_timestamps_ros_ns=bag.right_timestamps_ns,
        right_timestamps_s=relative_seconds(
            bag.right_timestamps_ns, bag.time_origin_ros_ns
        ),
        stereo_left_index=stereo_left_index,
        stereo_right_index=stereo_right_index,
        unpaired_left_index=unpaired_left_index,
        unpaired_right_index=unpaired_right_index,
        left_q_nearest_index=left_q_nearest,
        right_q_nearest_index=right_q_nearest,
        left_q_zoh_index=left_q_zoh,
        right_q_zoh_index=right_q_zoh,
    )

    print("[4/5] Rendering raw-bag projection checks")
    validation_dir = args.output_dir / "validation"
    validation_dir.mkdir()
    validation_frames: list[dict[str, Any]] = []
    side_configuration = {
        "left": (
            bag.left_timestamps_ns,
            left_q_nearest,
            calibration.left_maps,
            calibration.K_left_rect,
            T_left_rect_psm,
        ),
        "right": (
            bag.right_timestamps_ns,
            right_q_nearest,
            calibration.right_maps,
            calibration.K_right_rect,
            T_right_rect_psm,
        ),
    }
    for (side, frame_index), raw_image in sorted(bag.validation_images.items()):
        timestamps_ns, q_indices, maps, K, T_camera_psm = side_configuration[side]
        q_index = int(q_indices[frame_index])
        rectified = cv2.remap(
            raw_image, maps[0], maps[1], interpolation=cv2.INTER_LINEAR
        )
        title = (
            f"RAW bag {side} frame={frame_index} q={q_index} "
            f"dt={(int(timestamps_ns[frame_index]) - int(bag.joint_timestamps_ns[q_index])) * 1e-6:+.3f} ms"
        )
        overlay, frame_report = draw_validation_overlay(
            rectified,
            K,
            T_camera_psm,
            T_psm_links[q_index],
            lnd,
            title,
        )
        raw_path = validation_dir / f"{side}_{frame_index:06d}_raw.png"
        rectified_path = validation_dir / f"{side}_{frame_index:06d}_rectified.png"
        overlay_path = validation_dir / f"{side}_{frame_index:06d}_lnd_overlay.png"
        if not cv2.imwrite(str(raw_path), raw_image):
            raise RuntimeError(f"Failed to write {raw_path}")
        if not cv2.imwrite(str(rectified_path), rectified):
            raise RuntimeError(f"Failed to write {rectified_path}")
        if not cv2.imwrite(str(overlay_path), overlay):
            raise RuntimeError(f"Failed to write {overlay_path}")
        validation_frames.append(
            {
                "side": side,
                "image_index": frame_index,
                "image_timestamp_ros_ns": int(timestamps_ns[frame_index]),
                "nearest_q_index": q_index,
                "nearest_q_delta_ms": (
                    int(timestamps_ns[frame_index])
                    - int(bag.joint_timestamps_ns[q_index])
                )
                * 1e-6,
                "raw_image": raw_path.name,
                "rectified_image": rectified_path.name,
                "overlay_image": overlay_path.name,
                **frame_report,
            }
        )

    rotation_psm_gate = rotation_gate(T_psm_links)
    rotation_left_gate = rotation_gate(T_left_rect_links)
    rotation_right_gate = rotation_gate(T_right_rect_links)
    continuity_gate = motion_continuity_gate(
        relative_seconds(bag.joint_timestamps_ns, bag.time_origin_ros_ns),
        bag.q7,
        bag.q7_velocity,
        T_psm_links,
    )
    expected_rectified_stereo = np.eye(4, dtype=np.float64)
    expected_rectified_stereo[0, 3] = -calibration.baseline_m
    stereo_transform_error = float(
        np.max(np.abs(T_right_rect_left_rect - expected_rectified_stereo))
    )
    if stereo_transform_error > 1e-10:
        raise RuntimeError(
            "Rectified right-from-left transform is not a pure baseline translation: "
            f"max error {stereo_transform_error:.3e}"
        )

    epipolar_errors: list[np.ndarray] = []
    disparity_errors: list[np.ndarray] = []
    for q_index in (0, len(T_psm_links) // 2, len(T_psm_links) - 1):
        _, features_psm = point_feature_positions(lnd, T_psm_links[q_index])
        left_points = transform_points(T_left_rect_psm, features_psm)
        right_points = transform_points(T_right_rect_psm, features_psm)
        left_pixels, left_positive = project_points(
            calibration.K_left_rect, left_points
        )
        right_pixels, right_positive = project_points(
            calibration.K_right_rect, right_points
        )
        jointly_positive = left_positive & right_positive
        epipolar_errors.append(
            np.abs(
                left_pixels[jointly_positive, 1]
                - right_pixels[jointly_positive, 1]
            )
        )
        expected_disparity = (
            calibration.K_left_rect[0, 0]
            * calibration.baseline_m
            / left_points[jointly_positive, 2]
        )
        measured_disparity = (
            left_pixels[jointly_positive, 0]
            - right_pixels[jointly_positive, 0]
        )
        disparity_errors.append(np.abs(measured_disparity - expected_disparity))
    maximum_epipolar_error_px = float(np.max(np.concatenate(epipolar_errors)))
    maximum_disparity_identity_error_px = float(
        np.max(np.concatenate(disparity_errors))
    )

    left_feature_visibility = [
        frame["visible_feature_count"]
        for frame in validation_frames
        if frame["side"] == "left"
    ]
    right_feature_visibility = [
        frame["visible_feature_count"]
        for frame in validation_frames
        if frame["side"] == "right"
    ]
    feature_count = len(lnd.get("point_features", []))
    projection_gate_passed = bool(
        left_feature_visibility
        and right_feature_visibility
        and min(left_feature_visibility) == feature_count
        and min(right_feature_visibility) == feature_count
    )

    print("[5/5] Writing provenance and numerical gate report")
    inputs = {
        "bag": input_identity(args.bag, full_hash=args.hash_bag),
        "camera_calibration": input_identity(args.calibration, full_hash=True),
        "handeye": input_identity(args.handeye, full_hash=True),
        "lnd": input_identity(args.lnd, full_hash=True),
    }
    report = {
        "schema": "super_psm_raw_kinematics_report_v1",
        "passed": bool(
            projection_gate_passed
            and continuity_gate["passed"]
            and rotation_psm_gate["max_orthogonality_error"] < 1e-10
            and rotation_psm_gate["max_determinant_error"] < 1e-10
            and stereo_transform_error < 1e-10
            and maximum_epipolar_error_px < 1e-9
            and maximum_disparity_identity_error_px < 1e-9
        ),
        "inputs": inputs,
        "raw_bag": {
            "topics": {
                "left": LEFT_TOPIC,
                "right": RIGHT_TOPIC,
                "q7": JOINT_TOPIC,
            },
            "time_origin_ros_ns": bag.time_origin_ros_ns,
            "bag_start_ros_ns": bag.bag_start_ros_ns,
            "bag_end_ros_ns": bag.bag_end_ros_ns,
            "bag_duration_s": (
                bag.bag_end_ros_ns - bag.bag_start_ros_ns
            )
            * 1e-9,
            "left_image_count": len(bag.left_timestamps_ns),
            "right_image_count": len(bag.right_timestamps_ns),
            "joint_state_count": len(bag.joint_timestamps_ns),
            "left_time_range_s": [
                float(
                    relative_seconds(
                        bag.left_timestamps_ns[[0, -1]], bag.time_origin_ros_ns
                    )[0]
                ),
                float(
                    relative_seconds(
                        bag.left_timestamps_ns[[0, -1]], bag.time_origin_ros_ns
                    )[1]
                ),
            ],
            "right_time_range_s": [
                float(
                    relative_seconds(
                        bag.right_timestamps_ns[[0, -1]], bag.time_origin_ros_ns
                    )[0]
                ),
                float(
                    relative_seconds(
                        bag.right_timestamps_ns[[0, -1]], bag.time_origin_ros_ns
                    )[1]
                ),
            ],
            "q7_time_range_s": [
                float(
                    relative_seconds(
                        bag.joint_timestamps_ns[[0, -1]], bag.time_origin_ros_ns
                    )[0]
                ),
                float(
                    relative_seconds(
                        bag.joint_timestamps_ns[[0, -1]], bag.time_origin_ros_ns
                    )[1]
                ),
            ],
        },
        "synchronization": {
            "policy": {
                "backbone": "native q7 timestamps; no image resampling",
                "stereo_pairs": (
                    "maximum-cardinality monotonic matching with |dt| <= 20 ms"
                ),
                "image_q_nearest": "nearest q7 index, validation only",
                "image_q_zoh": "searchsorted(right)-1, exported for causal playback",
            },
            "stereo_pair_count": len(stereo_left_index),
            "unpaired_left_count": len(unpaired_left_index),
            "unpaired_right_count": len(unpaired_right_index),
            "unpaired_left_index": unpaired_left_index.tolist(),
            "unpaired_right_index": unpaired_right_index.tolist(),
            "stereo_pair_delta": percentile_ms(
                bag.left_timestamps_ns[stereo_left_index]
                - bag.right_timestamps_ns[stereo_right_index]
            ),
            "left_nearest_q_delta": percentile_ms(
                bag.left_timestamps_ns
                - bag.joint_timestamps_ns[left_q_nearest]
            ),
            "right_nearest_q_delta": percentile_ms(
                bag.right_timestamps_ns
                - bag.joint_timestamps_ns[right_q_nearest]
            ),
            "left_zoh_q_age": percentile_ms(
                bag.left_timestamps_ns - bag.joint_timestamps_ns[left_q_zoh]
            ),
            "right_zoh_q_age": percentile_ms(
                bag.right_timestamps_ns - bag.joint_timestamps_ns[right_q_zoh]
            ),
        },
        "calibration": {
            "baseline_m": calibration.baseline_m,
            "rectified_right_from_left_max_error_vs_pure_baseline": (
                stereo_transform_error
            ),
            "maximum_same_q_epipolar_y_error_px": maximum_epipolar_error_px,
            "maximum_same_q_disparity_identity_error_px": (
                maximum_disparity_identity_error_px
            ),
        },
        "kinematics": {
            "joint_names": list(bag.joint_names),
            "q7_shape": list(bag.q7.shape),
            "link_transform_shape": list(T_psm_links.shape),
            "length_unit": "metre",
            "angle_unit": "radian",
            "psm_base_rotation_gate": rotation_psm_gate,
            "left_camera_rotation_gate": rotation_left_gate,
            "right_camera_rotation_gate": rotation_right_gate,
            "all_finite": bool(
                np.isfinite(T_psm_links).all()
                and np.isfinite(T_left_rect_links).all()
                and np.isfinite(T_right_rect_links).all()
            ),
            "motion_continuity_gate": continuity_gate,
        },
        "projection_validation": {
            "passed": projection_gate_passed,
            "minimum_required_visible_features_per_view": feature_count,
            "frames": validation_frames,
        },
        "outputs": {
            "model": "model.json",
            "kinematics": "kinematics.npz",
            "validation": "validation/",
            "root_input_snapshots": "root_inputs_snapshot/",
        },
        "explicit_non_inputs": [
            "data/super/grasp5_native/*",
            "data/super/grasp5_offline_demo/*",
            "data/super/psm_robot/*",
            "data/super/psm_tracking/*",
            "all historical strict/registered/corrected/depth pose drivers",
        ],
    }
    write_json(args.output_dir / "report.json", report)
    if not report["passed"]:
        raise RuntimeError(
            f"Raw kinematic backbone gates failed; inspect {args.output_dir / 'report.json'}"
        )
    print(json.dumps({
        "passed": report["passed"],
        "left_images": len(bag.left_timestamps_ns),
        "right_images": len(bag.right_timestamps_ns),
        "q7_states": len(bag.joint_timestamps_ns),
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
