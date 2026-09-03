#!/usr/bin/env python3

"""Refine the active PSM driver with timestamp-aware dense stereo depth.

The active driver remains the kinematic source of truth.  This script estimates
one bounded camera-frame rigid correction at reliable video keyframes, smooths
those corrections in time, and left-multiplies the same correction onto every
PSM link.  Consequently all source parent/child transforms, including the two
shared-pivot jaws, are preserved exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


REPO = Path(__file__).resolve().parents[1]
TRACK_ROOT = REPO / "data/super/psm_tracking"
NATIVE_ROOT = REPO / "data/super/grasp5_native"
DEPTH_ROOT = TRACK_ROOT / "psm_stereo_3d_corrected_v1/depth_observations"
OUTPUT_ROOT = TRACK_ROOT / "psm_stereo_3d_corrected_v2"
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

from super_psm_tracking_common import (  # noqa: E402
    matrix_to_pose,
    pose_to_matrix,
    resample_pose_sequence,
)


PART_NAMES = ("body", "jaw_left", "jaw_right")
PART_COLORS = ((255, 255, 0), (255, 0, 255), (0, 255, 255))
DEFAULT_FRAMES = (
    "0,80,141,160,240,320,400,480,560,640,720,800,880,960,"
    "1040,1120,1200,1280,1360,1440"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bounded dense-stereo 3D refinement of the SUPER PSM driver."
    )
    parser.add_argument("--frames", default=DEFAULT_FRAMES)
    parser.add_argument(
        "--depth-dir", type=Path, default=DEPTH_ROOT
    )
    parser.add_argument(
        "--source-driver",
        type=Path,
        default=TRACK_ROOT / "psm_part_corrected_pose_driver.npz",
    )
    parser.add_argument(
        "--part-masks",
        type=Path,
        default=TRACK_ROOT / "part_masks_full_sequence/part_masks_full.npz",
    )
    parser.add_argument(
        "--gaussians",
        type=Path,
        default=REPO / "data/super/psm_robot/psm_surface_gaussians.npz",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=NATIVE_ROOT / "calib_rectified.json",
    )
    parser.add_argument(
        "--left-metadata",
        type=Path,
        default=REPO / "data/super/grasp5_offline_demo/videos/stereo_left.json",
    )
    parser.add_argument(
        "--right-metadata",
        type=Path,
        default=REPO / "data/super/grasp5_offline_demo/videos/stereo_right.json",
    )
    parser.add_argument("--rgb-dir", type=Path, default=NATIVE_ROOT / "rgb")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--roll-offset-deg", type=float, default=-27.0)
    parser.add_argument("--fallback-initial-z-mm", type=float, default=-5.7106)
    parser.add_argument("--max-rotation-deg", type=float, default=4.0)
    parser.add_argument("--max-xy-mm", type=float, default=3.0)
    parser.add_argument("--max-z-residual-mm", type=float, default=3.5)
    parser.add_argument("--nearest-depth-radius-px", type=float, default=7.0)
    parser.add_argument("--model-agreement-mm", type=float, default=4.0)
    parser.add_argument("--min-body-samples", type=int, default=100)
    parser.add_argument("--max-body-mad-mm", type=float, default=3.0)
    parser.add_argument("--max-evaluations", type=int, default=180)
    parser.add_argument("--max-correction-linear-speed-mm-s", type=float, default=5.0)
    parser.add_argument("--max-correction-angular-speed-deg-s", type=float, default=10.0)
    parser.add_argument("--max-correction-linear-accel-mm-s2", type=float, default=100.0)
    parser.add_argument("--max-correction-angular-accel-deg-s2", type=float, default=300.0)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path.resolve())


def five_number(values: np.ndarray) -> list[float] | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return np.percentile(values, [0, 5, 50, 95, 100]).tolist()


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    threshold = 0.5 * float(np.sum(weights))
    return float(values[np.searchsorted(np.cumsum(weights), threshold)])


def unpack_mask(asset: np.lib.npyio.NpzFile, frame: int) -> np.ndarray:
    height, width = asset["mask_shape"].astype(int)
    count = int(height * width)
    masks = np.unpackbits(
        asset["masks_packbits"][frame], axis=1, count=count
    ).reshape(len(PART_NAMES), height, width)
    return masks.astype(bool)


def apply_roll_offset(
    poses: np.ndarray, names: tuple[str, ...], angle_deg: float
) -> np.ndarray:
    if abs(angle_deg) < 1.0e-12:
        return poses.copy()
    output = poses.copy()
    main_index = names.index("PSM1_tool_main_link")
    for frame in range(len(output)):
        matrices = np.stack([pose_to_matrix(pose) for pose in output[frame]])
        local = np.eye(4, dtype=np.float64)
        local[:3, :3] = Rotation.from_euler(
            "z", math.radians(angle_deg)
        ).as_matrix()
        delta = matrices[main_index] @ local @ np.linalg.inv(matrices[main_index])
        for link_index, name in enumerate(names):
            if name != "PSM1_tool_main_link":
                matrices[link_index] = delta @ matrices[link_index]
                output[frame, link_index] = matrix_to_pose(matrices[link_index])
    return output


def transform_surface(
    local_points: np.ndarray,
    link_ids: np.ndarray,
    poses: np.ndarray,
) -> np.ndarray:
    output = np.empty_like(local_points, dtype=np.float64)
    for link_id in np.unique(link_ids):
        selected = link_ids == link_id
        matrix = pose_to_matrix(poses[int(link_id)])
        output[selected] = (
            local_points[selected] @ matrix[:3, :3].T + matrix[:3, 3]
        )
    return output


def project(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    normalized = points / points[:, 2:3]
    return (normalized @ K.T)[:, :2]


def front_surface_indices(
    points: np.ndarray,
    pixels: np.ndarray,
    parts: np.ndarray,
    cell_size: int = 4,
    tolerance_m: float = 0.0015,
) -> np.ndarray:
    visible = np.zeros(len(points), dtype=bool)
    for part in range(len(PART_NAMES)):
        selected = np.flatnonzero(parts == part)
        cell_depth: dict[tuple[int, int], float] = {}
        for index in selected:
            u, v = pixels[index]
            key = (int(math.floor(u / cell_size)), int(math.floor(v / cell_size)))
            cell_depth[key] = min(cell_depth.get(key, np.inf), float(points[index, 2]))
        for index in selected:
            u, v = pixels[index]
            key = (int(math.floor(u / cell_size)), int(math.floor(v / cell_size)))
            visible[index] = points[index, 2] <= cell_depth[key] + tolerance_m
    return np.flatnonzero(visible)


def bilinear_sample(image: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    height, width = image.shape
    x = pixels[:, 0]
    y = pixels[:, 1]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    inside = (x0 >= 0) & (y0 >= 0) & (x1 < width) & (y1 < height)
    x0c = np.clip(x0, 0, width - 1)
    x1c = np.clip(x1, 0, width - 1)
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y1, 0, height - 1)
    wx = x - x0
    wy = y - y0
    result = (
        image[y0c, x0c] * (1.0 - wx) * (1.0 - wy)
        + image[y0c, x1c] * wx * (1.0 - wy)
        + image[y1c, x0c] * (1.0 - wx) * wy
        + image[y1c, x1c] * wx * wy
    )
    result = np.asarray(result, dtype=np.float64)
    result[~inside] = np.nan
    return result


@dataclass
class PartMaps:
    foundation_depth: np.ndarray
    raft_depth: np.ndarray
    valid_distance: np.ndarray
    outside_distance: np.ndarray
    valid_pixel_count: int
    confidence_tier: str


@dataclass
class FrameProblem:
    frame: int
    points: np.ndarray
    parts: np.ndarray
    anchor: np.ndarray
    K: np.ndarray
    maps: tuple[PartMaps, ...]
    global_z_m: float
    nearest_radius_px: float
    max_rotation_rad: float
    max_xy_m: float
    max_z_residual_m: float

    def corrected_points(self, residual: np.ndarray) -> np.ndarray:
        rotation = Rotation.from_rotvec(residual[:3]).as_matrix()
        translation = residual[3:6] + np.array(
            [0.0, 0.0, self.global_z_m], dtype=np.float64
        )
        return (
            (self.points - self.anchor) @ rotation.T
            + self.anchor
            + translation
        )

    def samples(
        self, residual: np.ndarray, *, raft: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        points = self.corrected_points(residual)
        pixels = project(points, self.K)
        model_depth: list[np.ndarray] = []
        observed_depth: list[np.ndarray] = []
        part_values: list[np.ndarray] = []
        pixel_distance: list[np.ndarray] = []
        for part in range(len(PART_NAMES)):
            selected = self.parts == part
            if not np.any(selected):
                continue
            maps = self.maps[part]
            distance = bilinear_sample(maps.valid_distance, pixels[selected])
            depth_map = maps.raft_depth if raft else maps.foundation_depth
            observed = bilinear_sample(depth_map, pixels[selected])
            valid = (
                np.isfinite(distance)
                & np.isfinite(observed)
                & (distance <= self.nearest_radius_px)
                & (points[selected, 2] > 0.0)
            )
            model_depth.append(points[selected, 2][valid])
            observed_depth.append(observed[valid])
            part_values.append(np.full(np.count_nonzero(valid), part, dtype=np.int8))
            pixel_distance.append(distance[valid])
        if not model_depth:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty, empty.astype(np.int8), empty
        return (
            np.concatenate(model_depth),
            np.concatenate(observed_depth),
            np.concatenate(part_values),
            np.concatenate(pixel_distance),
        )

    def metrics(self, residual: np.ndarray, *, raft: bool = False) -> dict:
        model, observed, parts, distance = self.samples(residual, raft=raft)
        report: dict[str, object] = {}
        for part, name in enumerate(PART_NAMES):
            selected = parts == part
            values = (observed[selected] - model[selected]) * 1000.0
            if len(values):
                median = float(np.median(values))
                mad = float(np.median(np.abs(values - median)))
            else:
                median = float("nan")
                mad = float("nan")
            report[name] = {
                "sample_count": int(len(values)),
                "stereo_minus_model_mm_min_p05_p50_p95_max": five_number(values),
                "median_mm": median,
                "mad_mm": mad,
                "nearest_valid_pixel_distance_px_p50_p95": (
                    np.percentile(distance[selected], [50, 95]).tolist()
                    if np.any(selected)
                    else None
                ),
                "valid_pixel_count": self.maps[part].valid_pixel_count,
                "confidence_tier": self.maps[part].confidence_tier,
            }
        return report

    def objective(self, residual: np.ndarray) -> np.ndarray:
        points = self.corrected_points(residual)
        pixels = project(points, self.K)
        values: list[np.ndarray] = []
        depth_weights = (1.0, 0.35, 0.35)
        silhouette_weights = (0.45, 0.65, 0.65)
        for part in range(len(PART_NAMES)):
            selected = self.parts == part
            maps = self.maps[part]
            part_pixels = pixels[selected]
            part_points = points[selected]
            distance = bilinear_sample(maps.valid_distance, part_pixels)
            observed = bilinear_sample(maps.foundation_depth, part_pixels)
            outside = bilinear_sample(maps.outside_distance, part_pixels)
            finite = np.isfinite(distance) & np.isfinite(observed)
            gate = np.zeros(len(part_pixels), dtype=np.float64)
            gate[finite] = np.clip(
                1.0 - distance[finite] / self.nearest_radius_px, 0.0, 1.0
            )
            depth_residual_mm = np.zeros(len(part_pixels), dtype=np.float64)
            depth_residual_mm[finite] = (
                part_points[finite, 2] - observed[finite]
            ) * 1000.0
            values.append(
                math.sqrt(depth_weights[part]) * gate * depth_residual_mm
            )
            outside = np.nan_to_num(
                outside, nan=self.nearest_radius_px, posinf=self.nearest_radius_px
            )
            mm_per_pixel = (
                np.clip(part_points[:, 2], 0.03, 0.25)
                / float(self.K[0, 0])
                * 1000.0
            )
            values.append(
                math.sqrt(silhouette_weights[part]) * outside * mm_per_pixel
            )

        prior = np.concatenate(
            [
                residual[:3] / max(self.max_rotation_rad, 1.0e-9),
                residual[3:5] / max(self.max_xy_m, 1.0e-9),
                residual[5:6] / max(self.max_z_residual_m, 1.0e-9),
            ]
        )
        values.append(0.45 * prior)
        return np.concatenate(values)


def build_part_maps(
    mask: np.ndarray,
    foundation_depth: np.ndarray,
    raft_depth: np.ndarray,
    confidence: np.lib.npyio.NpzFile,
    agreement_mm: float,
) -> PartMaps:
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=2) > 0
    high = confidence["high_confidence"] & eroded
    relaxed = (
        confidence["foundation_lr_valid"]
        & confidence["raft_valid"]
        & np.isfinite(confidence["absolute_model_difference_mm"])
        & (confidence["absolute_model_difference_mm"] <= agreement_mm)
        & eroded
    )
    if np.count_nonzero(high) >= 40:
        valid = high
        tier = "foundation_raft_high_confidence"
    else:
        valid = relaxed
        tier = "foundation_raft_relaxed_agreement"
    valid &= np.isfinite(foundation_depth) & np.isfinite(raft_depth)
    outside = cv2.distanceTransform(
        (~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    ).astype(np.float32)
    if not np.any(valid):
        shape = foundation_depth.shape
        return PartMaps(
            foundation_depth=np.full(shape, np.nan, dtype=np.float32),
            raft_depth=np.full(shape, np.nan, dtype=np.float32),
            valid_distance=np.full(shape, 1.0e6, dtype=np.float32),
            outside_distance=outside,
            valid_pixel_count=0,
            confidence_tier="none",
        )
    distance, indices = distance_transform_edt(~valid, return_indices=True)
    nearest_foundation = foundation_depth[indices[0], indices[1]].astype(np.float32)
    nearest_raft = raft_depth[indices[0], indices[1]].astype(np.float32)
    return PartMaps(
        foundation_depth=nearest_foundation,
        raft_depth=nearest_raft,
        valid_distance=distance.astype(np.float32),
        outside_distance=outside,
        valid_pixel_count=int(np.count_nonzero(valid)),
        confidence_tier=tier,
    )


def make_problem(
    frame: int,
    global_z_m: float,
    video_poses_displayed: np.ndarray,
    local_points: np.ndarray,
    link_ids: np.ndarray,
    point_parts: np.ndarray,
    mask_asset: np.lib.npyio.NpzFile,
    depth_dir: Path,
    K: np.ndarray,
    args: argparse.Namespace,
    anchor_index: int,
) -> FrameProblem:
    foundation_path = depth_dir / f"{frame:06d}-depth.npy"
    raft_path = depth_dir / f"{frame:06d}-raft_depth.npy"
    confidence_path = depth_dir / f"{frame:06d}-confidence.npz"
    for path in (foundation_path, raft_path, confidence_path):
        if not path.exists():
            raise FileNotFoundError(path)
    foundation = np.load(foundation_path, mmap_mode="r")
    raft = np.load(raft_path, mmap_mode="r")
    masks = unpack_mask(mask_asset, frame)
    if masks.shape[1:] != foundation.shape:
        masks = np.stack(
            [
                cv2.resize(
                    mask.astype(np.uint8),
                    (foundation.shape[1], foundation.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
                for mask in masks
            ]
        )
    with np.load(confidence_path, allow_pickle=False) as confidence:
        maps = tuple(
            build_part_maps(
                masks[part],
                foundation,
                raft,
                confidence,
                args.model_agreement_mm,
            )
            for part in range(len(PART_NAMES))
        )
    frame_points = transform_surface(
        local_points, link_ids, video_poses_displayed[frame]
    )
    frame_pixels = project(frame_points, K)
    visible = front_surface_indices(frame_points, frame_pixels, point_parts)
    anchor = video_poses_displayed[frame, anchor_index, :3].astype(np.float64)
    return FrameProblem(
        frame=frame,
        points=frame_points[visible],
        parts=point_parts[visible],
        anchor=anchor,
        K=K,
        maps=maps,
        global_z_m=global_z_m,
        nearest_radius_px=args.nearest_depth_radius_px,
        max_rotation_rad=math.radians(args.max_rotation_deg),
        max_xy_m=args.max_xy_mm / 1000.0,
        max_z_residual_m=args.max_z_residual_mm / 1000.0,
    )


def correction_matrix(
    anchor: np.ndarray, residual: np.ndarray, global_z_m: float
) -> np.ndarray:
    rotation = Rotation.from_rotvec(residual[:3]).as_matrix()
    translation = residual[3:6] + np.array([0.0, 0.0, global_z_m])
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = anchor + translation - rotation @ anchor
    return matrix


def apply_corrections(
    poses: np.ndarray,
    residuals: np.ndarray,
    global_z_m: float,
    anchor_index: int,
) -> np.ndarray:
    output = np.empty_like(poses, dtype=np.float32)
    for frame in range(len(poses)):
        anchor = poses[frame, anchor_index, :3].astype(np.float64)
        correction = correction_matrix(anchor, residuals[frame], global_z_m)
        for link in range(poses.shape[1]):
            output[frame, link] = matrix_to_pose(
                correction @ pose_to_matrix(poses[frame, link])
            )
    return output


def relative_transform_gate(
    source: np.ndarray, candidate: np.ndarray, anchor_index: int
) -> dict:
    max_translation = 0.0
    max_rotation = 0.0
    max_jaw_pivot_gap = 0.0
    max_source_jaw_axis_error = 0.0
    max_candidate_jaw_axis_error = 0.0
    parent_index = 4
    jaw_indices = (5, 6)
    for frame in range(len(source)):
        source_anchor_inv = np.linalg.inv(
            pose_to_matrix(source[frame, anchor_index])
        )
        candidate_anchor_inv = np.linalg.inv(
            pose_to_matrix(candidate[frame, anchor_index])
        )
        for link in range(source.shape[1]):
            source_relative = source_anchor_inv @ pose_to_matrix(source[frame, link])
            candidate_relative = candidate_anchor_inv @ pose_to_matrix(
                candidate[frame, link]
            )
            delta = candidate_relative @ np.linalg.inv(source_relative)
            max_translation = max(max_translation, float(np.linalg.norm(delta[:3, 3])))
            max_rotation = max(
                max_rotation,
                float(Rotation.from_matrix(delta[:3, :3]).magnitude()),
            )
        source_jaws = [pose_to_matrix(source[frame, index]) for index in jaw_indices]
        jaws = [pose_to_matrix(candidate[frame, index]) for index in jaw_indices]
        max_jaw_pivot_gap = max(
            max_jaw_pivot_gap,
            float(np.linalg.norm(jaws[0][:3, 3] - jaws[1][:3, 3])),
        )
        source_axis_dot = abs(
            float(source_jaws[0][:3, 2] @ source_jaws[1][:3, 2])
        )
        max_source_jaw_axis_error = max(
            max_source_jaw_axis_error,
            float(math.acos(np.clip(source_axis_dot, 0.0, 1.0))),
        )
        candidate_axis_dot = abs(float(jaws[0][:3, 2] @ jaws[1][:3, 2]))
        max_candidate_jaw_axis_error = max(
            max_candidate_jaw_axis_error,
            float(math.acos(np.clip(candidate_axis_dot, 0.0, 1.0))),
        )
        parent = pose_to_matrix(candidate[frame, parent_index])
        for jaw in jaws:
            relative_translation = (np.linalg.inv(parent) @ jaw)[:3, 3]
            max_jaw_pivot_gap = max(
                max_jaw_pivot_gap, float(np.linalg.norm(relative_translation))
            )
    report = {
        "max_source_relative_translation_error_m": max_translation,
        "max_source_relative_rotation_error_deg": math.degrees(max_rotation),
        "max_shared_jaw_pivot_gap_m": max_jaw_pivot_gap,
        "source_max_jaw_hinge_axis_parallel_error_deg": math.degrees(
            max_source_jaw_axis_error
        ),
        "candidate_max_jaw_hinge_axis_parallel_error_deg": math.degrees(
            max_candidate_jaw_axis_error
        ),
        "relative_kinematics_preserved": bool(
            max_translation <= 1.0e-6 and math.degrees(max_rotation) <= 1.0e-3
        ),
        "jaw_shared_pivot_preserved": bool(max_jaw_pivot_gap <= 1.0e-6),
        "jaw_hinge_axis_error_not_increased": bool(
            math.degrees(max_candidate_jaw_axis_error - max_source_jaw_axis_error)
            <= 1.0e-3
        ),
    }
    if not all(
        report[key]
        for key in (
            "relative_kinematics_preserved",
            "jaw_shared_pivot_preserved",
            "jaw_hinge_axis_error_not_increased",
        )
    ):
        raise RuntimeError(f"Candidate failed the jaw/kinematic gate: {report}")
    return report


def temporal_correction_gate(
    timestamps: np.ndarray, residuals: np.ndarray, args: argparse.Namespace
) -> dict[str, float | bool]:
    dt = np.diff(timestamps.astype(np.float64))
    if len(dt) < 2 or np.any(dt <= 0.0):
        raise ValueError("Video timestamps must be strictly increasing")

    linear_velocity = np.diff(residuals[:, 3:6], axis=0) / dt[:, None]
    rotations = Rotation.from_rotvec(residuals[:, :3])
    angular_velocity = (
        (rotations[:-1].inv() * rotations[1:]).as_rotvec() / dt[:, None]
    )
    midpoint_dt = 0.5 * (dt[:-1] + dt[1:])
    linear_acceleration = np.diff(linear_velocity, axis=0) / midpoint_dt[:, None]
    angular_acceleration = np.diff(angular_velocity, axis=0) / midpoint_dt[:, None]

    max_linear_speed = float(
        np.max(np.linalg.norm(linear_velocity, axis=1)) * 1000.0
    )
    max_angular_speed = float(
        np.degrees(np.max(np.linalg.norm(angular_velocity, axis=1)))
    )
    max_linear_acceleration = float(
        np.max(np.linalg.norm(linear_acceleration, axis=1)) * 1000.0
    )
    max_angular_acceleration = float(
        np.degrees(np.max(np.linalg.norm(angular_acceleration, axis=1)))
    )
    passed = bool(
        max_linear_speed <= args.max_correction_linear_speed_mm_s
        and max_angular_speed <= args.max_correction_angular_speed_deg_s
        and max_linear_acceleration <= args.max_correction_linear_accel_mm_s2
        and max_angular_acceleration <= args.max_correction_angular_accel_deg_s2
    )
    report: dict[str, float | bool] = {
        "max_linear_speed_mm_s": max_linear_speed,
        "max_angular_speed_deg_s": max_angular_speed,
        "max_linear_acceleration_mm_s2": max_linear_acceleration,
        "max_angular_acceleration_deg_s2": max_angular_acceleration,
        "linear_speed_limit_mm_s": args.max_correction_linear_speed_mm_s,
        "angular_speed_limit_deg_s": args.max_correction_angular_speed_deg_s,
        "linear_acceleration_limit_mm_s2": args.max_correction_linear_accel_mm_s2,
        "angular_acceleration_limit_deg_s2": args.max_correction_angular_accel_deg_s2,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"Candidate failed the temporal correction gate: {report}")
    return report


def draw_surface(
    image: np.ndarray,
    points: np.ndarray,
    parts: np.ndarray,
    K: np.ndarray,
    *,
    prior: bool,
) -> np.ndarray:
    output = image.copy()
    pixels = project(points, K)
    for pixel, part in zip(pixels, parts, strict=True):
        u, v = np.rint(pixel).astype(int)
        if not (0 <= u < output.shape[1] and 0 <= v < output.shape[0]):
            continue
        if prior:
            cv2.circle(output, (u, v), 3, (20, 110, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(output, (u, v), 2, PART_COLORS[int(part)], -1, cv2.LINE_AA)
    return output


def annotate(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 46), (0, 0, 0), -1)
    cv2.putText(
        output,
        text,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def crop_panel(image: np.ndarray, center: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    x0 = int(np.clip(round(center[0] - width / 2), 0, image.shape[1] - width))
    y0 = int(np.clip(round(center[1] - height / 2), 0, image.shape[0] - height))
    return image[y0 : y0 + height, x0 : x0 + width]


def depth_residual_panel(
    image: np.ndarray,
    problem: FrameProblem,
    residual: np.ndarray,
    label: str,
) -> np.ndarray:
    output = image.copy()
    points = problem.corrected_points(residual)
    pixels = project(points, problem.K)
    for part in range(len(PART_NAMES)):
        selected = problem.parts == part
        distance = bilinear_sample(problem.maps[part].valid_distance, pixels[selected])
        observed = bilinear_sample(problem.maps[part].foundation_depth, pixels[selected])
        valid = (
            np.isfinite(distance)
            & np.isfinite(observed)
            & (distance <= problem.nearest_radius_px)
        )
        residual_mm = (observed[valid] - points[selected, 2][valid]) * 1000.0
        for pixel, value in zip(pixels[selected][valid], residual_mm, strict=True):
            u, v = np.rint(pixel).astype(int)
            if abs(value) <= 1.0:
                color = (50, 255, 50)
            elif value < 0.0:
                color = (255, 80, 20)
            else:
                color = (30, 30, 255)
            if 0 <= u < output.shape[1] and 0 <= v < output.shape[0]:
                cv2.circle(output, (u, v), 4, color, 1, cv2.LINE_AA)
    return annotate(output, label)


def comparison_image(
    frame: int,
    problem: FrameProblem,
    smoothed_residual: np.ndarray,
    source_displayed: np.ndarray,
    candidate_displayed: np.ndarray,
    source_right_displayed: np.ndarray,
    candidate_right_displayed: np.ndarray,
    local_points: np.ndarray,
    link_ids: np.ndarray,
    point_parts: np.ndarray,
    K: np.ndarray,
    baseline_m: float,
    left: np.ndarray,
    right: np.ndarray,
    right_index: int,
) -> np.ndarray:
    prior_left_points = transform_surface(local_points, link_ids, source_displayed)
    candidate_left_points = transform_surface(
        local_points, link_ids, candidate_displayed
    )
    prior_right_points = transform_surface(
        local_points, link_ids, source_right_displayed
    )
    candidate_right_points = transform_surface(
        local_points, link_ids, candidate_right_displayed
    )
    prior_right_points[:, 0] -= baseline_m
    candidate_right_points[:, 0] -= baseline_m

    left_before = draw_surface(left, prior_left_points, point_parts, K, prior=True)
    left_after = draw_surface(left, candidate_left_points, point_parts, K, prior=False)
    right_before = draw_surface(right, prior_right_points, point_parts, K, prior=True)
    right_after = draw_surface(right, candidate_right_points, point_parts, K, prior=False)
    before_metrics = problem.metrics(np.zeros(6))["body"]
    after_metrics = problem.metrics(smoothed_residual)["body"]
    depth_before = depth_residual_panel(
        left,
        problem,
        np.zeros(6),
        f"Foundation residual before: {before_metrics['median_mm']:+.2f} mm",
    )
    depth_after = depth_residual_panel(
        left,
        problem,
        smoothed_residual,
        f"Foundation residual after: {after_metrics['median_mm']:+.2f} mm",
    )
    center = np.median(project(candidate_left_points, K), axis=0)
    labels = (
        f"left {frame}: active corrected prior",
        f"left {frame}: stereo 3D candidate",
        f"right {right_index}: timestamp prior",
        f"right {right_index}: stereo 3D candidate",
        f"Foundation before: {before_metrics['median_mm']:+.2f} mm",
        f"Foundation after: {after_metrics['median_mm']:+.2f} mm",
    )
    panels = [
        annotate(crop_panel(panel, center, (900, 600)), label)
        for panel, label in zip(
            (
                left_before,
                left_after,
                right_before,
                right_after,
                depth_before,
                depth_after,
            ),
            labels,
            strict=True,
        )
    ]
    return np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:])))


def main() -> None:
    args = parse_args()
    frames = sorted({int(value) for value in args.frames.split(",") if value.strip()})
    if not frames:
        raise ValueError("At least one keyframe is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir = args.output_dir / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    calibration = read_json(args.calibration)
    left_metadata = read_json(args.left_metadata)
    right_metadata = read_json(args.right_metadata)
    K = np.asarray(calibration["K_left_rect"], dtype=np.float64)
    baseline_m = float(calibration["baseline_m"])
    left_timestamps = np.asarray(left_metadata["timestamps"], dtype=np.float64)
    right_timestamps = np.asarray(right_metadata["timestamps"], dtype=np.float64)
    if any(frame < 0 or frame >= len(left_timestamps) for frame in frames):
        raise IndexError("Requested keyframe lies outside the left video")

    with np.load(args.source_driver, allow_pickle=False) as source_asset:
        runtime_timestamps = source_asset["timestamps"].astype(np.float64)
        link_names = tuple(source_asset["link_names"].tolist())
        source_runtime_poses = source_asset[
            "poses_rect_camera_xyz_xyzw"
        ].astype(np.float64)
    source_video_poses = resample_pose_sequence(
        runtime_timestamps, source_runtime_poses, left_timestamps
    ).astype(np.float64)
    source_video_displayed = apply_roll_offset(
        source_video_poses, link_names, args.roll_offset_deg
    )
    anchor_index = link_names.index("PSM1_tool_wrist_link")

    with np.load(args.gaussians, allow_pickle=False) as gaussian_asset:
        all_local_points = gaussian_asset["means"].astype(np.float64)
        all_link_ids = gaussian_asset["link_ids"].astype(np.int64)
        gaussian_names = tuple(gaussian_asset["link_names"].tolist())
    if gaussian_names != link_names:
        raise ValueError("Gaussian and driver link-name orders differ")
    visible = all_link_ids != link_names.index("PSM1_tool_main_link")
    local_points = all_local_points[visible]
    link_ids = all_link_ids[visible]
    point_parts = np.where(link_ids == 5, 1, np.where(link_ids == 6, 2, 0)).astype(
        np.int8
    )

    mask_asset = np.load(args.part_masks, allow_pickle=False)
    global_reports: list[dict] = []
    shifts: list[float] = []
    weights: list[float] = []
    print("estimating the sequence-wide camera-z initialization")
    for frame in frames:
        problem = make_problem(
            frame,
            0.0,
            source_video_displayed,
            local_points,
            link_ids,
            point_parts,
            mask_asset,
            args.depth_dir,
            K,
            args,
            anchor_index,
        )
        foundation = problem.metrics(np.zeros(6))
        raft = problem.metrics(np.zeros(6), raft=True)
        body = foundation["body"]
        right_after = int(np.searchsorted(right_timestamps, left_timestamps[frame]))
        timestamp_bracketed = bool(0 < right_after < len(right_timestamps))
        reliable = bool(
            timestamp_bracketed
            and body["sample_count"] >= args.min_body_samples
            and np.isfinite(body["mad_mm"])
            and body["mad_mm"] <= args.max_body_mad_mm
            and abs(body["median_mm"]) <= 15.0
        )
        if reliable:
            shifts.append(float(body["median_mm"]))
            weights.append(float(body["sample_count"]) / max(float(body["mad_mm"]), 0.5))
        global_reports.append(
            {
                "frame": frame,
                "foundation": foundation,
                "raft": raft,
                "right_timestamp_bracketed": timestamp_bracketed,
                "reliable_for_global_z": reliable,
            }
        )
        print(
            f"frame {frame:4d}: n={body['sample_count']:3d} "
            f"shift={body['median_mm']:+.3f}mm mad={body['mad_mm']:.3f} "
            f"reliable={reliable}"
        )
        del problem
    if shifts:
        global_z_mm = weighted_median(np.asarray(shifts), np.asarray(weights))
    else:
        global_z_mm = float(args.fallback_initial_z_mm)
    global_z_m = global_z_mm / 1000.0
    print(f"global camera-z initialization: {global_z_mm:+.4f} mm")

    max_rotation = math.radians(args.max_rotation_deg)
    lower = np.asarray(
        [-max_rotation] * 3
        + [-args.max_xy_mm / 1000.0] * 2
        + [-args.max_z_residual_mm / 1000.0]
    )
    upper = -lower
    keyframe_residuals: list[np.ndarray] = []
    accepted_frames: list[int] = []
    optimization_reports: list[dict] = []
    print("optimizing reliable stereo keyframes")
    for global_report in global_reports:
        frame = int(global_report["frame"])
        if not global_report["reliable_for_global_z"]:
            optimization_reports.append(
                {"frame": frame, "accepted": False, "reason": "unreliable depth"}
            )
            continue
        problem = make_problem(
            frame,
            global_z_m,
            source_video_displayed,
            local_points,
            link_ids,
            point_parts,
            mask_asset,
            args.depth_dir,
            K,
            args,
            anchor_index,
        )
        raw_shift_mm = float(global_report["foundation"]["body"]["median_mm"])
        initial = np.zeros(6, dtype=np.float64)
        initial[5] = np.clip(
            (raw_shift_mm - global_z_mm) / 1000.0,
            lower[5] * 0.95,
            upper[5] * 0.95,
        )
        before = problem.metrics(np.zeros(6))
        result = least_squares(
            problem.objective,
            initial,
            bounds=(lower, upper),
            loss="huber",
            f_scale=1.0,
            max_nfev=args.max_evaluations,
            x_scale="jac",
        )
        after = problem.metrics(result.x)
        after_raft = problem.metrics(result.x, raft=True)
        before_abs = abs(float(before["body"]["median_mm"]))
        after_abs = abs(float(after["body"]["median_mm"]))
        accepted = bool(
            result.success
            and after["body"]["sample_count"] >= args.min_body_samples
            and after_abs <= before_abs + 0.25
            and float(after["body"]["mad_mm"]) <= args.max_body_mad_mm + 0.5
        )
        if accepted:
            accepted_frames.append(frame)
            keyframe_residuals.append(result.x.copy())
        optimization_reports.append(
            {
                "frame": frame,
                "accepted": accepted,
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "function_evaluations": int(result.nfev),
                "residual_rotation_deg_xyz": np.degrees(result.x[:3]).tolist(),
                "residual_translation_mm_xyz": (result.x[3:] * 1000.0).tolist(),
                "total_camera_z_mm": float(global_z_mm + result.x[5] * 1000.0),
                "before_foundation": before,
                "after_foundation": after,
                "after_raft": after_raft,
            }
        )
        print(
            f"frame {frame:4d}: accepted={accepted} "
            f"body {before['body']['median_mm']:+.3f} -> "
            f"{after['body']['median_mm']:+.3f} mm, "
            f"dxyz={(result.x[3:] * 1000.0).round(3).tolist()}"
        )
        del problem
    if len(accepted_frames) < 3:
        raise RuntimeError(
            f"Only {len(accepted_frames)} reliable keyframes survived; refusing to write a driver"
        )

    accepted_frames_np = np.asarray(accepted_frames, dtype=np.int64)
    keyframe_residuals_np = np.stack(keyframe_residuals)
    full_residuals = np.empty((len(left_timestamps), 6), dtype=np.float64)
    accepted_timestamps = left_timestamps[accepted_frames_np]
    for axis in range(6):
        interpolator = PchipInterpolator(
            accepted_timestamps,
            keyframe_residuals_np[:, axis],
            extrapolate=False,
        )
        full_residuals[:, axis] = interpolator(left_timestamps)
        full_residuals[left_timestamps < accepted_timestamps[0], axis] = (
            keyframe_residuals_np[0, axis]
        )
        full_residuals[left_timestamps > accepted_timestamps[-1], axis] = (
            keyframe_residuals_np[-1, axis]
        )
    full_residuals = np.clip(full_residuals, lower, upper)
    temporal_gate = temporal_correction_gate(left_timestamps, full_residuals, args)

    runtime_residuals = np.empty((len(runtime_timestamps), 6), dtype=np.float64)
    for axis in range(6):
        runtime_residuals[:, axis] = np.interp(
            runtime_timestamps,
            left_timestamps,
            full_residuals[:, axis],
            left=full_residuals[0, axis],
            right=full_residuals[-1, axis],
        )
    candidate_runtime_poses = apply_corrections(
        source_runtime_poses, runtime_residuals, global_z_m, anchor_index
    )
    candidate_video_poses = apply_corrections(
        source_video_poses, full_residuals, global_z_m, anchor_index
    )
    kinematic_gate = relative_transform_gate(
        source_runtime_poses, candidate_runtime_poses, anchor_index
    )

    candidate_driver = args.output_dir / "psm_stereo_3d_corrected_pose_driver_candidate.npz"
    np.savez_compressed(
        candidate_driver,
        timestamps=runtime_timestamps,
        link_names=np.asarray(link_names),
        poses_rect_camera_xyz_xyzw=candidate_runtime_poses,
    )
    residual_path = args.output_dir / "stereo_pose_residuals_full.npz"
    np.savez_compressed(
        residual_path,
        video_timestamps=left_timestamps,
        global_camera_z_initialization_m=np.asarray(global_z_m),
        full_residual_rotvec_rad_translation_m=full_residuals,
        accepted_keyframes=accepted_frames_np,
        keyframe_residual_rotvec_rad_translation_m=keyframe_residuals_np,
    )
    video_pose_path = args.output_dir / "visual_poses_stereo_3d.npz"
    np.savez_compressed(
        video_pose_path,
        video_timestamps=left_timestamps,
        link_names=np.asarray(link_names),
        poses_rect_camera_xyz_xyzw=candidate_video_poses,
    )

    candidate_video_displayed = apply_roll_offset(
        candidate_video_poses, link_names, args.roll_offset_deg
    )
    comparison_images: list[np.ndarray] = []
    final_reports: list[dict] = []
    for frame in accepted_frames:
        problem = make_problem(
            frame,
            global_z_m,
            source_video_displayed,
            local_points,
            link_ids,
            point_parts,
            mask_asset,
            args.depth_dir,
            K,
            args,
            anchor_index,
        )
        smoothed = full_residuals[frame]
        final_foundation = problem.metrics(smoothed)
        final_raft = problem.metrics(smoothed, raft=True)
        final_reports.append(
            {
                "frame": frame,
                "smoothed_residual_rotation_deg_xyz": np.degrees(
                    smoothed[:3]
                ).tolist(),
                "smoothed_residual_translation_mm_xyz": (
                    smoothed[3:] * 1000.0
                ).tolist(),
                "total_camera_z_mm": float(global_z_mm + smoothed[5] * 1000.0),
                "foundation": final_foundation,
                "raft": final_raft,
            }
        )
        right_index = int(np.argmin(np.abs(right_timestamps - left_timestamps[frame])))
        right_source_pose = resample_pose_sequence(
            runtime_timestamps,
            source_runtime_poses,
            right_timestamps[right_index : right_index + 1],
        )[0]
        right_candidate_pose = resample_pose_sequence(
            runtime_timestamps,
            candidate_runtime_poses,
            right_timestamps[right_index : right_index + 1],
        )[0]
        right_source_displayed = apply_roll_offset(
            right_source_pose[None], link_names, args.roll_offset_deg
        )[0]
        right_candidate_displayed = apply_roll_offset(
            right_candidate_pose[None], link_names, args.roll_offset_deg
        )[0]
        left = cv2.imread(
            str(args.rgb_dir / f"{frame:06d}-left.png"), cv2.IMREAD_COLOR
        )
        right = cv2.imread(
            str(args.rgb_dir / f"{right_index:06d}-right.png"), cv2.IMREAD_COLOR
        )
        if left is None or right is None:
            raise FileNotFoundError(f"Missing comparison RGB for frame {frame}")
        image = comparison_image(
            frame,
            problem,
            smoothed,
            source_video_displayed[frame],
            candidate_video_displayed[frame],
            right_source_displayed,
            right_candidate_displayed,
            local_points,
            link_ids,
            point_parts,
            K,
            baseline_m,
            left,
            right,
            right_index,
        )
        image_path = comparison_dir / f"frame{frame:06d}_stereo_3d_comparison.png"
        cv2.imwrite(str(image_path), image)
        comparison_images.append(
            cv2.resize(image, (1350, 600), interpolation=cv2.INTER_AREA)
        )
        del problem
    rows = []
    for start in range(0, len(comparison_images), 2):
        row = comparison_images[start : start + 2]
        if len(row) == 1:
            row.append(np.zeros_like(row[0]))
        rows.append(np.hstack(row))
    contact_sheet = comparison_dir / "contact_sheet.png"
    cv2.imwrite(str(contact_sheet), np.vstack(rows))

    table_path = REPO / "data/super/table_frame.json"
    cameras_path = REPO / "data/super/grasp5_offline_demo/cameras.json"
    report = {
        "method": (
            "Timestamp-synchronized FoundationStereo depth with RAFT agreement, "
            "bounded per-keyframe camera-frame SE(3) residuals, and timestamp-domain "
            "shape-preserving PCHIP interpolation"
        ),
        "source_driver": relative_path(args.source_driver),
        "source_driver_sha256": sha256(args.source_driver),
        "candidate_driver": relative_path(candidate_driver),
        "candidate_driver_sha256": sha256(candidate_driver),
        "coordinate_invariants": {
            "table_frame_sha256": sha256(table_path),
            "cameras_sha256": sha256(cameras_path),
            "driver_frame": "rectified left OpenCV camera",
            "table_or_camera_calibration_modified": False,
        },
        "global_camera_z_initialization_mm": global_z_mm,
        "requested_keyframes": frames,
        "accepted_keyframes": accepted_frames,
        "bounds": {
            "rotation_deg_per_axis": args.max_rotation_deg,
            "xy_mm_per_axis": args.max_xy_mm,
            "z_residual_around_global_mm": args.max_z_residual_mm,
        },
        "temporal_interpolation": {
            "method": "PCHIP over measured left-camera timestamps",
            "outside_anchor_range": "hold nearest accepted keyframe constant",
        },
        "global_initialization_frames": global_reports,
        "optimization_frames": optimization_reports,
        "final_smoothed_frames": final_reports,
        "kinematic_and_jaw_gate": kinematic_gate,
        "temporal_correction_gate": temporal_gate,
        "jaw_policy": (
            "No wrist or jaw joint is re-estimated from stereo. The same common "
            "rigid correction is applied to all seven source links at every runtime "
            "timestamp, preserving the validated encoder-based shared-pivot jaw motion."
        ),
        "residual_asset": relative_path(residual_path),
        "video_pose_asset": relative_path(video_pose_path),
        "comparison_directory": relative_path(comparison_dir),
        "comparison_contact_sheet": relative_path(contact_sheet),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    mask_asset.close()
    print(json.dumps({
        "global_camera_z_initialization_mm": global_z_mm,
        "accepted_keyframes": accepted_frames,
        "candidate_driver": relative_path(candidate_driver),
        "kinematic_and_jaw_gate": kinematic_gate,
        "temporal_correction_gate": temporal_gate,
        "report": relative_path(report_path),
        "contact_sheet": relative_path(contact_sheet),
    }, indent=2))


if __name__ == "__main__":
    main()
