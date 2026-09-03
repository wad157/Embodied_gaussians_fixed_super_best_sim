#!/usr/bin/env python3
"""Track a small set of grasp-region tissue keypoints for stage B.

Sparse Shi-Tomasi features are seeded near the distal tool, tracked with
forward/backward pyramidal LK, rejected inside tool occlusions, and associated
with the nearest saved stage-B surface point.  The result is a set of
tracklets, not permission to move the frozen PSM trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "data/super/tissue_calibration_v1/stage_a_frozen_manifest.json"
)
DEFAULT_SURFACE_ROOT = (
    REPO_ROOT
    / "data/super/tissue_calibration_v1/stage_b_surface_observations"
)
DEFAULT_OUTPUT = DEFAULT_SURFACE_ROOT / "keypoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--surface-root", type=Path, default=DEFAULT_SURFACE_ROOT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-frame", type=int, default=270)
    parser.add_argument("--end-frame", type=int, default=1300)
    parser.add_argument("--tracking-scale", type=float, default=0.5)
    parser.add_argument("--reseed-interval", type=int, default=30)
    parser.add_argument("--target-active-tracks", type=int, default=60)
    parser.add_argument("--maximum-output-tracks", type=int, default=32)
    parser.add_argument("--minimum-track-length", type=int, default=45)
    parser.add_argument("--seed-outer-radius-px", type=int, default=160)
    parser.add_argument("--tool-exclusion-radius-px", type=int, default=30)
    parser.add_argument("--maximum-fb-error-px", type=float, default=2.0)
    parser.add_argument("--maximum-surface-pixel-distance", type=float, default=30.0)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def unpack_tissue_mask(
    packed: np.ndarray, frame: int, width: int
) -> np.ndarray:
    return np.unpackbits(
        packed[frame], axis=1, count=width
    ).astype(bool, copy=False)


class DistalToolMasks:
    def __init__(self, path: Path):
        self.archive = np.load(path, allow_pickle=False)
        self.shape = tuple(
            int(value) for value in self.archive["mask_shape"].tolist()
        )
        self.bitorder = str(self.archive["bitorder"].item())
        self.native_indices = np.asarray(
            self.archive["stereo_left_index"], dtype=np.int64
        )
        self.packed = self.archive["left_distal_masks_packbits"]
        self.slot_by_frame = {
            int(frame): slot
            for slot, frame in enumerate(self.native_indices)
        }

    def close(self) -> None:
        self.archive.close()

    def mask(self, frame: int) -> tuple[np.ndarray, int]:
        slot = self.slot_by_frame.get(frame)
        if slot is None:
            slot = int(np.argmin(np.abs(self.native_indices - frame)))
        source_frame = int(self.native_indices[slot])
        mask = np.unpackbits(
            self.packed[slot],
            bitorder=self.bitorder,
            count=int(np.prod(self.shape)),
        ).reshape(self.shape)
        return mask.astype(bool), source_frame


def disk_kernel(radius: int) -> np.ndarray:
    size = radius * 2 + 1
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (size, size)
    )


def scaled_mask(
    mask: np.ndarray, shape_hw: tuple[int, int]
) -> np.ndarray:
    return cv2.resize(
        mask.astype(np.uint8),
        (shape_hw[1], shape_hw[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def read_gray(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.resize(
        image,
        (shape_hw[1], shape_hw[0]),
        interpolation=cv2.INTER_AREA,
    )


def add_seeds(
    gray: np.ndarray,
    tissue: np.ndarray,
    distal: np.ndarray,
    active_points: np.ndarray,
    maximum_new: int,
    outer_radius: int,
    exclusion_radius: int,
) -> np.ndarray:
    outer = cv2.dilate(
        distal.astype(np.uint8), disk_kernel(outer_radius)
    ).astype(bool)
    excluded = cv2.dilate(
        distal.astype(np.uint8), disk_kernel(exclusion_radius)
    ).astype(bool)
    roi = tissue & outer & ~excluded
    if len(active_points):
        active_mask = np.zeros(roi.shape, dtype=np.uint8)
        for x, y in active_points:
            cv2.circle(
                active_mask, (int(round(x)), int(round(y))), 10, 255, -1
            )
        roi &= active_mask == 0
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=maximum_new,
        qualityLevel=0.01,
        minDistance=10.0,
        mask=roi.astype(np.uint8) * 255,
        blockSize=7,
        useHarrisDetector=False,
    )
    if corners is None:
        return np.empty((0, 2), dtype=np.float32)
    return corners.reshape(-1, 2).astype(np.float32)


def track_forward_backward(
    previous: np.ndarray,
    current: np.ndarray,
    points: np.ndarray,
    maximum_fb_error: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(points):
        return points.copy(), np.empty(0, dtype=bool), np.empty(0)
    source = points.reshape(-1, 1, 2).astype(np.float32)
    forward, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        source,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )
    backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(
        current,
        previous,
        forward,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )
    tracked = forward.reshape(-1, 2)
    fb_error = np.linalg.norm(
        backward.reshape(-1, 2) - points, axis=1
    )
    valid = (
        (status_forward.reshape(-1) > 0)
        & (status_backward.reshape(-1) > 0)
        & np.isfinite(tracked).all(axis=1)
        & np.isfinite(fb_error)
        & (fb_error <= maximum_fb_error)
    )
    return tracked, valid, fb_error


def mask_contains(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    if not len(points):
        return np.empty(0, dtype=bool)
    x = np.rint(points[:, 0]).astype(np.int64)
    y = np.rint(points[:, 1]).astype(np.int64)
    inside = (
        (x >= 0)
        & (x < mask.shape[1])
        & (y >= 0)
        & (y < mask.shape[0])
    )
    result = np.zeros(len(points), dtype=bool)
    result[inside] = mask[y[inside], x[inside]]
    return result


def select_tracks(
    observations: dict[int, list[tuple[int, np.ndarray, float]]],
    minimum_length: int,
    maximum_tracks: int,
) -> list[int]:
    candidates = [
        track_id
        for track_id, values in observations.items()
        if len(values) >= minimum_length
    ]
    candidates.sort(
        key=lambda track_id: (
            -len(observations[track_id]),
            observations[track_id][0][0],
            track_id,
        )
    )
    return candidates[:maximum_tracks]


def associate_surface_points(
    *,
    selected_ids: list[int],
    observations: dict[int, list[tuple[int, np.ndarray, float]]],
    surface_root: Path,
    scale_to_full: float,
    maximum_distance: float,
) -> dict[str, np.ndarray]:
    by_frame: dict[int, list[tuple[int, int, np.ndarray, float]]] = defaultdict(
        list
    )
    for track_id in selected_ids:
        for local_index, (frame, uv, fb_error) in enumerate(
            observations[track_id]
        ):
            by_frame[frame].append((track_id, local_index, uv, fb_error))

    records: list[tuple[int, int, np.ndarray, float, np.ndarray, float, bool]] = []
    for frame in sorted(by_frame):
        surface_path = surface_root / "frames" / f"{frame:06d}.npz"
        if not surface_path.is_file():
            for track_id, _, uv, fb_error in by_frame[frame]:
                records.append(
                    (
                        track_id,
                        frame,
                        uv * scale_to_full,
                        fb_error * scale_to_full,
                        np.full(3, np.nan, dtype=np.float32),
                        float("inf"),
                        False,
                    )
                )
            continue
        with np.load(surface_path, allow_pickle=False) as surface:
            pixels = np.asarray(surface["pixels_uv"], dtype=np.float64)
            points = np.asarray(surface["points_table"], dtype=np.float32)
        tree = cKDTree(pixels)
        frame_items = by_frame[frame]
        queries = np.stack(
            [item[2] * scale_to_full for item in frame_items]
        )
        distances, indices = tree.query(queries, k=1)
        for item, query, distance, index in zip(
            frame_items, queries, distances, indices, strict=True
        ):
            track_id, _, _, fb_error = item
            valid_3d = bool(distance <= maximum_distance)
            point = (
                points[int(index)]
                if valid_3d
                else np.full(3, np.nan, dtype=np.float32)
            )
            records.append(
                (
                    track_id,
                    frame,
                    query,
                    fb_error * scale_to_full,
                    point,
                    float(distance),
                    valid_3d,
                )
            )

    return {
        "track_ids": np.asarray([item[0] for item in records], dtype=np.int32),
        "frames": np.asarray([item[1] for item in records], dtype=np.int32),
        "pixels_uv": np.asarray([item[2] for item in records], dtype=np.float32),
        "fb_error_px": np.asarray(
            [item[3] for item in records], dtype=np.float32
        ),
        "points_table": np.asarray(
            [item[4] for item in records], dtype=np.float32
        ),
        "surface_pixel_distance": np.asarray(
            [item[5] for item in records], dtype=np.float32
        ),
        "has_3d": np.asarray([item[6] for item in records], dtype=bool),
    }


def draw_track_preview(
    *,
    frame: int,
    rgb_dir: Path,
    records: dict[str, np.ndarray],
    output: Path,
) -> None:
    image = cv2.imread(
        str(rgb_dir / f"{frame:06d}-left.png"), cv2.IMREAD_COLOR
    )
    if image is None:
        raise FileNotFoundError(frame)
    selected = np.flatnonzero(records["frames"] == frame)
    for index in selected:
        track_id = int(records["track_ids"][index])
        u, v = records["pixels_uv"][index]
        color = (
            int(60 + (track_id * 83) % 196),
            int(60 + (track_id * 47) % 196),
            int(60 + (track_id * 131) % 196),
        )
        cv2.circle(image, (int(round(u)), int(round(v))), 5, color, -1)
        cv2.putText(
            image,
            str(track_id),
            (int(round(u)) + 5, int(round(v)) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output), image)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.tracking_scale <= 1.0:
        raise ValueError("tracking-scale must be in (0, 1]")
    if args.end_frame < args.start_frame:
        raise ValueError("end-frame must not precede start-frame")

    manifest = load_json(args.manifest)
    if (
        manifest.get("stage") != "A_frozen_inputs"
        or not manifest.get("passed", False)
    ):
        raise RuntimeError("Stage-A manifest is not complete")
    surface_report_path = args.surface_root / "stage_b_report.json"
    surface_report = load_json(surface_report_path)
    if not surface_report.get("passed", False):
        raise RuntimeError("Complete stage-B surface observations are required")

    left_metadata_path = repo_path(
        manifest["sequence"]["offline_stereo"]["left_metadata"]["path"]
    )
    left_metadata = load_json(left_metadata_path)
    timestamps = np.asarray(left_metadata["timestamps"], dtype=np.float64)
    width, height = (int(value) for value in left_metadata["resolution"])
    tracking_shape = (
        int(round(height * args.tracking_scale)),
        int(round(width * args.tracking_scale)),
    )
    scale_to_full = 1.0 / args.tracking_scale
    rgb_dir = REPO_ROOT / "data/super/grasp5_native/rgb"
    tissue_path = repo_path(
        manifest["observation_inputs_for_stage_b"]["left_tissue_masks"]["path"]
    )
    tool_path = repo_path(
        manifest["observation_inputs_for_stage_b"][
            "stereo_tool_part_masks"
        ]["path"]
    )
    tissue_packed = np.load(tissue_path, mmap_mode="r")
    tool_masks = DistalToolMasks(tool_path)

    observations: dict[int, list[tuple[int, np.ndarray, float]]] = defaultdict(
        list
    )
    active_ids: list[int] = []
    active_points = np.empty((0, 2), dtype=np.float32)
    next_track_id = 0
    previous_gray: np.ndarray | None = None
    source_gap_frames: list[int] = []
    outer_radius = max(
        1, int(round(args.seed_outer_radius_px * args.tracking_scale))
    )
    exclusion_radius = max(
        1, int(round(args.tool_exclusion_radius_px * args.tracking_scale))
    )
    maximum_fb_error = (
        args.maximum_fb_error_px * args.tracking_scale
    )

    try:
        for frame in range(args.start_frame, args.end_frame + 1):
            gray = read_gray(
                rgb_dir / f"{frame:06d}-left.png", tracking_shape
            )
            tissue_full = unpack_tissue_mask(tissue_packed, frame, width)
            tissue = scaled_mask(tissue_full, tracking_shape)
            distal_native, source_frame = tool_masks.mask(frame)
            source_gap_frames.append(abs(source_frame - frame))
            distal = scaled_mask(distal_native, tracking_shape)
            excluded = cv2.dilate(
                distal.astype(np.uint8), disk_kernel(exclusion_radius)
            ).astype(bool)

            if previous_gray is not None and len(active_points):
                tracked, valid, fb_error = track_forward_backward(
                    previous_gray,
                    gray,
                    active_points,
                    maximum_fb_error,
                )
                valid &= mask_contains(tissue, tracked)
                valid &= ~mask_contains(excluded, tracked)
                active_points = tracked[valid]
                active_ids = [
                    track_id
                    for track_id, keep in zip(active_ids, valid, strict=True)
                    if keep
                ]
                kept_errors = fb_error[valid]
                for track_id, point, error in zip(
                    active_ids,
                    active_points,
                    kept_errors,
                    strict=True,
                ):
                    observations[track_id].append(
                        (frame, point.copy(), float(error))
                    )

            should_reseed = (
                frame == args.start_frame
                or (
                    (frame - args.start_frame) % args.reseed_interval == 0
                    and len(active_ids) < args.target_active_tracks
                )
            )
            if should_reseed:
                maximum_new = max(
                    0, args.target_active_tracks - len(active_ids)
                )
                new_points = add_seeds(
                    gray,
                    tissue,
                    distal,
                    active_points,
                    maximum_new,
                    outer_radius,
                    exclusion_radius,
                )
                for point in new_points:
                    track_id = next_track_id
                    next_track_id += 1
                    active_ids.append(track_id)
                    observations[track_id].append(
                        (frame, point.copy(), 0.0)
                    )
                if len(new_points):
                    active_points = np.concatenate(
                        (active_points, new_points), axis=0
                    )

            previous_gray = gray
            if (frame - args.start_frame) % 100 == 0:
                print(
                    f"frame={frame} active={len(active_ids)} "
                    f"total_tracks={next_track_id}",
                    flush=True,
                )
    finally:
        tool_masks.close()

    selected_ids = select_tracks(
        observations,
        args.minimum_track_length,
        args.maximum_output_tracks,
    )
    if len(selected_ids) < 8:
        raise RuntimeError(
            f"Only {len(selected_ids)} tracklets passed the length gate"
        )
    records = associate_surface_points(
        selected_ids=selected_ids,
        observations=observations,
        surface_root=args.surface_root,
        scale_to_full=scale_to_full,
        maximum_distance=args.maximum_surface_pixel_distance,
    )
    has_3d_fraction = float(np.mean(records["has_3d"]))
    track_lengths = np.asarray(
        [len(observations[track_id]) for track_id in selected_ids],
        dtype=np.int32,
    )
    if has_3d_fraction < 0.60:
        raise RuntimeError(
            f"Only {has_3d_fraction:.3f} of keypoints have surface 3D"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    previews = args.output_dir / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    output_npz = args.output_dir / "keypoint_tracks.npz"
    np.savez_compressed(
        output_npz,
        schema=np.asarray("super_tissue_grasp_keypoint_tracklets_v1"),
        selected_track_ids=np.asarray(selected_ids, dtype=np.int32),
        track_lengths=track_lengths,
        **records,
    )
    preview_frames = sorted(
        {
            args.start_frame,
            548,
            700,
            906,
            1276,
            args.end_frame,
        }
    )
    for frame in preview_frames:
        if args.start_frame <= frame <= args.end_frame:
            draw_track_preview(
                frame=frame,
                rgb_dir=rgb_dir,
                records=records,
                output=previews / f"{frame:06d}.jpg",
            )

    report = {
        "schema": "super_tissue_grasp_keypoint_tracklets_report_v1",
        "stage": "B_grasp_region_keypoints",
        "passed": True,
        "frame_range_inclusive": [args.start_frame, args.end_frame],
        "tracking_scale": args.tracking_scale,
        "created_track_count": next_track_id,
        "selected_track_count": len(selected_ids),
        "selected_track_ids": selected_ids,
        "track_length_min_p50_max": [
            int(track_lengths.min()),
            float(np.median(track_lengths)),
            int(track_lengths.max()),
        ],
        "observation_count": int(len(records["frames"])),
        "surface_3d_observation_count": int(
            np.count_nonzero(records["has_3d"])
        ),
        "surface_3d_fraction": has_3d_fraction,
        "tool_mask_source_gap_frames_max": int(max(source_gap_frames)),
        "parameters": {
            "reseed_interval": args.reseed_interval,
            "target_active_tracks": args.target_active_tracks,
            "maximum_output_tracks": args.maximum_output_tracks,
            "minimum_track_length": args.minimum_track_length,
            "seed_outer_radius_px": args.seed_outer_radius_px,
            "tool_exclusion_radius_px": args.tool_exclusion_radius_px,
            "maximum_fb_error_px": args.maximum_fb_error_px,
            "maximum_surface_pixel_distance": (
                args.maximum_surface_pixel_distance
            ),
        },
        "inputs": {
            "stage_a_manifest": {
                "path": str(args.manifest.resolve()),
                "sha256": sha256(args.manifest),
            },
            "stage_b_surface_report": {
                "path": str(surface_report_path.resolve()),
                "sha256": sha256(surface_report_path),
            },
        },
        "output": {
            "path": str(output_npz.resolve()),
            "sha256": sha256(output_npz),
        },
        "usage": (
            "Sparse tissue-state loss only. These tracks must never modify "
            "the frozen PSM trajectory."
        ),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
