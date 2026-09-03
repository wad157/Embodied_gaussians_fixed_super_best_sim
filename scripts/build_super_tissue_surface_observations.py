#!/usr/bin/env python3
"""Build manifest-driven tissue surface observations for calibration stage B.

FoundationStereo estimates left-reference depth from the nearest frozen right
frame.  The propagated tissue mask is eroded, the dense-contact tool mask is
conservatively dilated and removed, and the remaining surface is transformed
to the table/world frame.  Only voxelized, statistically filtered point clouds
are saved; dense depth is optional.

The script supports disjoint multi-GPU shards.  Each shard writes independent
per-frame files.  Run once with ``--finalize-only`` after all shards complete
to create the combined stage report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "data/super/tissue_calibration_v1/stage_a_frozen_manifest.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data/super/tissue_calibration_v1/stage_b_surface_observations"
)
FOUNDATION_CHECKPOINT = (
    REPO_ROOT
    / "third_party/FoundationStereo/pretrained_models/23-51-11/"
    "model_best_bp2.pth"
)

os.environ.setdefault("XFORMERS_DISABLED", "1")
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from generate_super_depth_foundation_timestamped import (  # noqa: E402
    depth_from_disparity,
    disparity_with_lr_consistency,
    image_tensor,
    infer_foundation,
    load_foundation_model,
)


LANDMARK_NAMES = (
    "initial_reference",
    "first_contact",
    "grasp_closed",
    "traction_reference",
    "maximum_retraction",
    "release_complete",
    "rebound_reference",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--frames",
        default="eligible",
        help=(
            "eligible, landmarks, or comma-separated frames/ranges such as "
            "0,270,540-550"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--foundation-checkpoint",
        type=Path,
        default=FOUNDATION_CHECKPOINT,
    )
    parser.add_argument("--foundation-iters", type=int, default=32)
    parser.add_argument(
        "--foundation-hierarchical",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--lr-consistency",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--stereo-semantic-consistency",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Require every accepted left tissue point to reproject into the "
            "right tissue mask outside the right tool mask."
        ),
    )
    parser.add_argument("--lr-threshold-px", type=float, default=1.5)
    parser.add_argument("--min-depth-mm", type=float, default=35.0)
    parser.add_argument("--max-depth-mm", type=float, default=250.0)
    parser.add_argument("--tissue-erosion-px", type=int, default=2)
    parser.add_argument("--tool-dilation-px", type=int, default=30)
    parser.add_argument("--voxel-size-mm", type=float, default=1.0)
    parser.add_argument("--outlier-neighbors", type=int, default=12)
    parser.add_argument("--outlier-std-ratio", type=float, default=2.5)
    parser.add_argument("--minimum-points", type=int, default=1000)
    parser.add_argument("--maximum-points", type=int, default=9000)
    parser.add_argument(
        "--save-dense-depth",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace selected frame outputs with this configuration.",
    )
    parser.add_argument(
        "--overwrite-mismatched",
        action="store_true",
        help=(
            "When resuming, keep outputs with the requested processing "
            "signature and atomically replace only partial or mismatched ones."
        ),
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def processing_settings(args: argparse.Namespace) -> dict[str, Any]:
    settings = {
        "foundation_checkpoint": str(args.foundation_checkpoint.resolve()),
        "foundation_iters": args.foundation_iters,
        "foundation_hierarchical": args.foundation_hierarchical,
        "lr_consistency": args.lr_consistency,
        "lr_threshold_px": args.lr_threshold_px,
        "depth_range_mm": [args.min_depth_mm, args.max_depth_mm],
        "tissue_erosion_px": args.tissue_erosion_px,
        "tool_dilation_px": args.tool_dilation_px,
        "voxel_size_mm": args.voxel_size_mm,
        "outlier_neighbors": args.outlier_neighbors,
        "outlier_std_ratio": args.outlier_std_ratio,
        "minimum_points": args.minimum_points,
        "maximum_points": args.maximum_points,
        "save_dense_depth": args.save_dense_depth,
    }
    # Keep the pre-semantic-baseline signature stable when this optional
    # filter is disabled. Formal stage-B outputs explicitly enable it.
    if args.stereo_semantic_consistency:
        settings["stereo_semantic_consistency"] = True
    return settings


def settings_signature(settings: dict[str, Any]) -> str:
    canonical = json.dumps(
        settings, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def resolve_from_repo(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def parse_frame_spec(
    spec: str, eligible: list[int], landmarks: dict[str, Any]
) -> list[int]:
    if spec == "eligible":
        return eligible
    if spec == "landmarks":
        return sorted(
            {
                int(landmarks[name]["left_frame"])
                for name in LANDMARK_NAMES
            }
        )
    frames: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid frame range: {item}")
            frames.update(range(start, end + 1))
        else:
            frames.add(int(item))
    if not frames:
        raise ValueError("No frames selected")
    ineligible = sorted(frames - set(eligible))
    if ineligible:
        raise ValueError(f"Frames are excluded by stage A: {ineligible}")
    return sorted(frames)


def nearest_indices(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    upper = np.searchsorted(reference, query, side="left")
    upper = np.clip(upper, 1, len(reference) - 1)
    lower = upper - 1
    choose_upper = np.abs(reference[upper] - query) < np.abs(
        reference[lower] - query
    )
    return np.where(choose_upper, upper, lower)


def unpack_tissue_mask(
    packed: np.ndarray, frame: int, width: int
) -> np.ndarray:
    return np.unpackbits(
        packed[frame], axis=1, count=width
    ).astype(bool, copy=False)


class ToolMasks:
    def __init__(self, path: Path, output_shape_hw: tuple[int, int]):
        archive = np.load(path, allow_pickle=False)
        self.archive = archive
        self.mask_shape = tuple(
            int(value) for value in archive["mask_shape"].tolist()
        )
        self.bitorder = str(archive["bitorder"].item())
        self.packed = {
            "left": archive["left_masks_packbits"],
            "right": archive["right_masks_packbits"],
        }
        self.native_indices = {
            "left": np.asarray(
                archive["stereo_left_index"], dtype=np.int64
            ),
            "right": np.asarray(
                archive["stereo_right_index"], dtype=np.int64
            ),
        }
        self.output_shape_hw = output_shape_hw
        self.slot_by_native_index = {
            side: {
                int(native_index): slot
                for slot, native_index in enumerate(indices)
            }
            for side, indices in self.native_indices.items()
        }

    def close(self) -> None:
        self.archive.close()

    def mask(
        self, frame: int, side: str = "left"
    ) -> tuple[np.ndarray, int, int]:
        if side not in self.packed:
            raise ValueError(f"Unknown stereo side: {side}")
        indices = self.native_indices[side]
        slot = self.slot_by_native_index[side].get(frame)
        if slot is None:
            nearest = int(np.argmin(np.abs(indices - frame)))
            slot = nearest
        source_frame = int(indices[slot])
        flat = np.unpackbits(
            self.packed[side][slot],
            bitorder=self.bitorder,
            count=int(np.prod(self.mask_shape)),
        )
        half_resolution = flat.reshape(self.mask_shape).astype(bool)
        full_resolution = cv2.resize(
            half_resolution.astype(np.uint8),
            (self.output_shape_hw[1], self.output_shape_hw[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        return full_resolution, source_frame, abs(source_frame - frame)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def voxel_downsample(
    points_camera: np.ndarray,
    points_table: np.ndarray,
    pixels_uv: np.ndarray,
    colors_rgb: np.ndarray,
    lr_error_px: np.ndarray,
    voxel_size_m: float,
) -> tuple[np.ndarray, ...]:
    keys = np.floor(points_table / voxel_size_m).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    ordered_keys = keys[order]
    changes = np.empty(len(order), dtype=bool)
    changes[0] = True
    changes[1:] = np.any(ordered_keys[1:] != ordered_keys[:-1], axis=1)
    starts = np.flatnonzero(changes)
    counts = np.diff(np.append(starts, len(order))).astype(np.float64)

    def means(values: np.ndarray) -> np.ndarray:
        values_float = values.astype(np.float64, copy=False)
        return np.add.reduceat(
            values_float[order], starts, axis=0
        ) / counts.reshape((-1,) + (1,) * (values.ndim - 1))

    representative = order[starts]
    return (
        means(points_camera).astype(np.float32),
        means(points_table).astype(np.float32),
        pixels_uv[representative].astype(np.int16),
        np.clip(np.rint(means(colors_rgb)), 0, 255).astype(np.uint8),
        means(lr_error_px).reshape(-1).astype(np.float32),
    )


def remove_statistical_outliers(
    points_table: np.ndarray,
    neighbors: int,
    std_ratio: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if len(points_table) <= neighbors:
        return np.ones(len(points_table), dtype=bool), {
            "mean_neighbor_distance_m": 0.0,
            "threshold_m": float("inf"),
        }
    k = min(neighbors + 1, len(points_table))
    distances, _ = cKDTree(points_table).query(
        points_table, k=k, workers=-1
    )
    mean_distance = distances[:, 1:].mean(axis=1)
    threshold = float(
        mean_distance.mean() + std_ratio * mean_distance.std()
    )
    keep = mean_distance <= threshold
    return keep, {
        "mean_neighbor_distance_m": float(mean_distance.mean()),
        "threshold_m": threshold,
    }


def cap_points(
    arrays: tuple[np.ndarray, ...], maximum: int
) -> tuple[np.ndarray, ...]:
    count = len(arrays[0])
    if count <= maximum:
        return arrays
    selected = np.linspace(0, count - 1, maximum, dtype=np.int64)
    return tuple(array[selected] for array in arrays)


def five_number(values: np.ndarray, scale: float = 1.0) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)] * scale
    if not len(values):
        return [float("nan")] * 5
    return [
        float(value) for value in np.percentile(values, [0, 5, 50, 95, 100])
    ]


def draw_preview(
    image_bgr: np.ndarray,
    tissue_mask: np.ndarray,
    tool_mask: np.ndarray,
    pixels_uv: np.ndarray,
    output: Path,
) -> None:
    preview = image_bgr.copy()
    tint = preview.copy()
    tint[tissue_mask] = (40, 170, 40)
    tint[tool_mask] = (30, 30, 220)
    preview = cv2.addWeighted(preview, 0.68, tint, 0.32, 0.0)
    for u, v in pixels_uv[:: max(1, len(pixels_uv) // 2500)]:
        cv2.circle(preview, (int(u), int(v)), 1, (255, 220, 20), -1)
    cv2.imwrite(str(output), preview)


def output_paths(output_dir: Path, frame: int) -> tuple[Path, Path]:
    stem = f"{frame:06d}"
    return (
        output_dir / "frames" / f"{stem}.npz",
        output_dir / "frames" / f"{stem}.json",
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.shard_count <= 0:
        raise ValueError("shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    if args.voxel_size_mm <= 0.0:
        raise ValueError("voxel-size-mm must be positive")
    if args.minimum_points <= 0:
        raise ValueError("minimum-points must be positive")
    if args.maximum_points < args.minimum_points:
        raise ValueError("maximum-points must be >= minimum-points")


def load_stage_inputs(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_json(manifest_path)
    if (
        manifest.get("stage") != "A_frozen_inputs"
        or not manifest.get("passed", False)
    ):
        raise RuntimeError("Stage-A manifest is not complete")
    stage = {
        "left_video_metadata": resolve_from_repo(
            manifest["sequence"]["offline_stereo"]["left_metadata"]["path"]
        ),
        "right_video_metadata": resolve_from_repo(
            manifest["sequence"]["offline_stereo"]["right_metadata"]["path"]
        ),
        "calibration": resolve_from_repo(
            manifest["camera"]["rectified_calibration"]["path"]
        ),
        "table_frame": resolve_from_repo(
            manifest["camera"]["table_frame"]["path"]
        ),
        "left_tissue_masks": resolve_from_repo(
            manifest["observation_inputs_for_stage_b"][
                "left_tissue_masks"
            ]["path"]
        ),
        "right_tissue_masks": resolve_from_repo(
            manifest["observation_inputs_for_stage_b"][
                "right_tissue_masks"
            ]["path"]
        ),
        "tool_masks": resolve_from_repo(
            manifest["observation_inputs_for_stage_b"][
                "stereo_tool_part_masks"
            ]["path"]
        ),
        "rgb_dir": REPO_ROOT / "data/super/grasp5_native/rgb",
    }
    return manifest, stage


def finalize_report(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    selected_frames: list[int],
    output_dir: Path,
) -> dict[str, Any]:
    expected_settings = processing_settings(args)
    expected_signature = settings_signature(expected_settings)
    reports: list[dict[str, Any]] = []
    missing: list[int] = []
    failed: list[int] = []
    mismatched: list[int] = []
    artifact_digest = hashlib.sha256()
    artifact_bytes = 0
    for frame in selected_frames:
        npz_path, json_path = output_paths(output_dir, frame)
        if not npz_path.is_file() or not json_path.is_file():
            missing.append(frame)
            continue
        artifact_digest.update(f"{frame:06d}".encode())
        for artifact_path in (npz_path, json_path):
            artifact_digest.update(sha256(artifact_path).encode())
            artifact_bytes += artifact_path.stat().st_size
        report = load_json(json_path)
        reports.append(report)
        if not report.get("passed", False):
            failed.append(frame)
        if report.get("processing_signature") != expected_signature:
            mismatched.append(frame)

    complete = not missing and not failed and not mismatched
    point_counts = np.asarray(
        [report["output_point_count"] for report in reports], dtype=np.int64
    )
    visible_fractions = np.asarray(
        [report["visible_depth_fraction"] for report in reports],
        dtype=np.float64,
    )
    elapsed = np.asarray(
        [report["elapsed_seconds"] for report in reports], dtype=np.float64
    )
    combined = {
        "schema": "super_tissue_surface_observations_stage_b_v1",
        "stage": "B_visible_surface_observations",
        "status": "complete" if complete else "partial",
        "passed": complete,
        "stage_a_manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256(args.manifest),
        },
        "requested_frame_spec": args.frames,
        "selected_frames": (
            selected_frames if len(selected_frames) <= 256 else None
        ),
        "selected_frame_count": len(selected_frames),
        "completed_frame_count": len(reports),
        "missing_frames": missing,
        "failed_frames": failed,
        "configuration_mismatched_frames": mismatched,
        "ordered_frame_artifact_digest_sha256": artifact_digest.hexdigest(),
        "frame_artifact_bytes": artifact_bytes,
        "processing_signature": expected_signature,
        "parameters": expected_settings,
        "summary": {
            "point_count_min_p05_p50_p95_max": (
                five_number(point_counts) if len(point_counts) else None
            ),
            "visible_depth_fraction_min_p05_p50_p95_max": (
                five_number(visible_fractions)
                if len(visible_fractions)
                else None
            ),
            "elapsed_seconds_min_p05_p50_p95_max": (
                five_number(elapsed) if len(elapsed) else None
            ),
        },
        "coordinate_frame": "dense-ground-aligned table/world frame",
        "surface_observation": (
            "visible left-reference tissue surface only; tool occlusion "
            "removed before backprojection"
        ),
        "residual_mapping_enabled": False,
        "stiffness_optimization_enabled": False,
        "next_step": (
            "add tracked grasp-region surface keypoints, then use these "
            "observations in deterministic no-visual-force replay"
        ),
        "frozen_driver_name": manifest["psm_trajectory"]["driver_name"],
    }
    atomic_json(output_dir / "stage_b_report.json", combined)
    return combined


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    manifest, stage = load_stage_inputs(args.manifest)

    offline = manifest["sequence"]["offline_stereo"]
    excluded = set(int(value) for value in offline["stereo_excluded_left_frames"])
    full_range = range(
        int(offline["stereo_calibration_left_frame_range_inclusive"][0]),
        int(offline["stereo_calibration_left_frame_range_inclusive"][1]) + 1,
    )
    eligible = [frame for frame in full_range if frame not in excluded]
    selected_frames = parse_frame_spec(
        args.frames, eligible, manifest["sequence"]["landmarks"]
    )
    settings = processing_settings(args)
    signature = settings_signature(settings)

    (output_dir / "frames").mkdir(parents=True, exist_ok=True)
    (output_dir / "previews").mkdir(parents=True, exist_ok=True)
    if args.save_dense_depth:
        (output_dir / "dense_depth").mkdir(parents=True, exist_ok=True)

    if args.finalize_only:
        report = finalize_report(
            args=args,
            manifest=manifest,
            selected_frames=selected_frames,
            output_dir=output_dir,
        )
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise SystemExit(2)
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    left_metadata = load_json(stage["left_video_metadata"])
    right_metadata = load_json(stage["right_video_metadata"])
    calibration = load_json(stage["calibration"])
    table_frame = load_json(stage["table_frame"])
    left_timestamps = np.asarray(
        left_metadata["timestamps"], dtype=np.float64
    )
    right_timestamps = np.asarray(
        right_metadata["timestamps"], dtype=np.float64
    )
    right_for_left = nearest_indices(right_timestamps, left_timestamps)
    image_width, image_height = (
        int(value) for value in left_metadata["resolution"]
    )
    K = np.asarray(calibration["K_left_rect"], dtype=np.float64)
    X_table_camera = np.asarray(
        table_frame["X_table_camera"], dtype=np.float64
    )
    baseline = float(calibration["baseline_m"])
    cx_delta = float(
        calibration["K_right_rect"][0][2]
        - calibration["K_left_rect"][0][2]
    )
    tissue_packed = np.load(stage["left_tissue_masks"], mmap_mode="r")
    right_tissue_packed = (
        np.load(stage["right_tissue_masks"], mmap_mode="r")
        if args.stereo_semantic_consistency
        else None
    )
    tool_masks = ToolMasks(
        stage["tool_masks"], (image_height, image_width)
    )

    frames_this_shard = selected_frames[
        args.shard_index :: args.shard_count
    ]
    pending: list[int] = []
    for frame in frames_this_shard:
        npz_path, json_path = output_paths(output_dir, frame)
        outputs_exist = npz_path.is_file() and json_path.is_file()
        if outputs_exist and args.overwrite:
            pending.append(frame)
            continue
        if outputs_exist and args.resume:
            previous = load_json(json_path)
            if previous.get("processing_signature") == signature:
                continue
            if args.overwrite_mismatched:
                pending.append(frame)
                continue
            raise RuntimeError(
                f"Frame {frame} exists with a different processing "
                "configuration; pass --overwrite or --overwrite-mismatched"
            )
        if (
            npz_path.exists() or json_path.exists()
        ) and (args.overwrite or args.overwrite_mismatched):
            pending.append(frame)
            continue
        if npz_path.exists() or json_path.exists():
            raise FileExistsError(
                f"Incomplete/existing outputs for frame {frame}; "
                "pass --overwrite or --overwrite-mismatched"
            )
        pending.append(frame)

    if not pending:
        print(
            f"shard {args.shard_index}/{args.shard_count}: "
            "all selected outputs already exist"
        )
        tool_masks.close()
        return

    torch.manual_seed(0)
    np.random.seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    print(
        f"loading FoundationStereo on {device}; "
        f"shard={args.shard_index}/{args.shard_count}; "
        f"pending={len(pending)}"
    )
    model, model_metadata = load_foundation_model(
        args.foundation_checkpoint, device
    )
    model_record = {
        "checkpoint": str(args.foundation_checkpoint.resolve()),
        "checkpoint_sha256": model_metadata["checkpoint_sha256"],
    }
    landmark_frames = {
        int(manifest["sequence"]["landmarks"][name]["left_frame"])
        for name in LANDMARK_NAMES
    }

    try:
        for progress_index, frame in enumerate(pending, start=1):
            started = time.perf_counter()
            right_frame = int(right_for_left[frame])
            stereo_delta_ms = float(
                abs(right_timestamps[right_frame] - left_timestamps[frame])
                * 1e3
            )
            if stereo_delta_ms > 20.0:
                raise RuntimeError(
                    f"Frame {frame} violates frozen stereo limit: "
                    f"{stereo_delta_ms:.3f} ms"
                )

            left_path = stage["rgb_dir"] / f"{frame:06d}-left.png"
            right_path = (
                stage["rgb_dir"] / f"{right_frame:06d}-right.png"
            )
            left_bgr = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
            right_bgr = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
            if left_bgr is None or right_bgr is None:
                raise FileNotFoundError(
                    f"Missing stereo images: {left_path}, {right_path}"
                )

            left_tensor = image_tensor(left_bgr, device)
            right_tensor = image_tensor(right_bgr, device)
            if args.lr_consistency:
                disparity, lr_valid, lr_error = (
                    disparity_with_lr_consistency(
                        infer_foundation,
                        model,
                        left_tensor,
                        right_tensor,
                        args.foundation_iters,
                        args.foundation_hierarchical,
                        device,
                        args.lr_threshold_px,
                    )
                )
            else:
                disparity = infer_foundation(
                    model,
                    left_tensor,
                    right_tensor,
                    args.foundation_iters,
                    args.foundation_hierarchical,
                    device,
                )
                lr_valid = np.ones(disparity.shape, dtype=bool)
                lr_error = np.zeros(disparity.shape, dtype=np.float32)

            depth, depth_valid = depth_from_disparity(
                disparity,
                lr_valid,
                float(K[0, 0]),
                baseline,
                cx_delta,
                args.min_depth_mm / 1000.0,
                args.max_depth_mm / 1000.0,
            )
            tissue_mask = unpack_tissue_mask(
                tissue_packed, frame, image_width
            )
            if args.tissue_erosion_px > 0:
                size = args.tissue_erosion_px * 2 + 1
                tissue_mask = cv2.erode(
                    tissue_mask.astype(np.uint8),
                    np.ones((size, size), dtype=np.uint8),
                ).astype(bool)
            tool_mask, tool_source_frame, tool_source_gap = tool_masks.mask(
                frame, side="left"
            )
            if args.tool_dilation_px > 0:
                size = args.tool_dilation_px * 2 + 1
                tool_mask = cv2.dilate(
                    tool_mask.astype(np.uint8),
                    np.ones((size, size), dtype=np.uint8),
                ).astype(bool)
            visible_semantic = tissue_mask & ~tool_mask
            selected = visible_semantic & depth_valid
            right_tissue_pixels = 0
            right_tool_pixels = 0
            right_tool_source_frame: int | None = None
            right_tool_source_gap: int | None = None
            if args.stereo_semantic_consistency:
                assert right_tissue_packed is not None
                right_tissue_mask = unpack_tissue_mask(
                    right_tissue_packed, right_frame, image_width
                )
                if args.tissue_erosion_px > 0:
                    size = args.tissue_erosion_px * 2 + 1
                    right_tissue_mask = cv2.erode(
                        right_tissue_mask.astype(np.uint8),
                        np.ones((size, size), dtype=np.uint8),
                    ).astype(bool)
                (
                    right_tool_mask,
                    right_tool_source_frame,
                    right_tool_source_gap,
                ) = tool_masks.mask(right_frame, side="right")
                if args.tool_dilation_px > 0:
                    size = args.tool_dilation_px * 2 + 1
                    right_tool_mask = cv2.dilate(
                        right_tool_mask.astype(np.uint8),
                        np.ones((size, size), dtype=np.uint8),
                    ).astype(bool)
                right_visible_semantic = (
                    right_tissue_mask & ~right_tool_mask
                )
                candidate_rows, candidate_columns = np.nonzero(selected)
                right_columns = np.rint(
                    candidate_columns
                    - disparity[candidate_rows, candidate_columns]
                ).astype(np.int64)
                in_right_image = (
                    (right_columns >= 0)
                    & (right_columns < image_width)
                )
                stereo_semantic_valid = np.zeros(
                    len(candidate_rows), dtype=bool
                )
                stereo_semantic_valid[in_right_image] = (
                    right_visible_semantic[
                        candidate_rows[in_right_image],
                        right_columns[in_right_image],
                    ]
                )
                selected[:] = False
                selected[
                    candidate_rows[stereo_semantic_valid],
                    candidate_columns[stereo_semantic_valid],
                ] = True
                right_tissue_pixels = int(
                    np.count_nonzero(right_tissue_mask)
                )
                right_tool_pixels = int(
                    np.count_nonzero(right_tool_mask)
                )
            rows, columns = np.nonzero(selected)
            if len(rows) == 0:
                raise RuntimeError(f"No visible tissue depth for frame {frame}")

            z = depth[rows, columns]
            points_camera = np.stack(
                (
                    (columns - K[0, 2]) * z / K[0, 0],
                    (rows - K[1, 2]) * z / K[1, 1],
                    z,
                ),
                axis=1,
            )
            points_table = transform_points(
                points_camera, X_table_camera
            )
            pixels_uv = np.column_stack((columns, rows)).astype(np.int16)
            colors_rgb = left_bgr[rows, columns, ::-1]
            selected_lr_error = lr_error[rows, columns, None]

            arrays = voxel_downsample(
                points_camera,
                points_table,
                pixels_uv,
                colors_rgb,
                selected_lr_error,
                args.voxel_size_mm / 1000.0,
            )
            keep, outlier_metrics = remove_statistical_outliers(
                arrays[1],
                args.outlier_neighbors,
                args.outlier_std_ratio,
            )
            before_outlier_count = len(arrays[0])
            arrays = tuple(array[keep] for array in arrays)
            arrays = cap_points(arrays, args.maximum_points)
            (
                points_camera_ds,
                points_table_ds,
                pixels_uv_ds,
                colors_rgb_ds,
                lr_error_ds,
            ) = arrays
            point_count = len(points_table_ds)
            finite = bool(
                np.isfinite(points_camera_ds).all()
                and np.isfinite(points_table_ds).all()
                and np.isfinite(lr_error_ds).all()
            )
            passed = finite and point_count >= args.minimum_points

            npz_path, json_path = output_paths(output_dir, frame)
            atomic_npz(
                npz_path,
                schema=np.asarray(
                    "super_tissue_visible_surface_observation_v1"
                ),
                left_frame=np.asarray(frame, dtype=np.int32),
                left_timestamp_s=np.asarray(
                    left_timestamps[frame], dtype=np.float64
                ),
                right_frame=np.asarray(right_frame, dtype=np.int32),
                right_timestamp_s=np.asarray(
                    right_timestamps[right_frame], dtype=np.float64
                ),
                points_table=points_table_ds,
                points_left_camera=points_camera_ds,
                pixels_uv=pixels_uv_ds,
                colors_rgb=colors_rgb_ds,
                lr_error_px=lr_error_ds,
            )
            if args.save_dense_depth:
                depth_path = (
                    output_dir / "dense_depth" / f"{frame:06d}.npy"
                )
                np.save(depth_path, depth)
            if frame in landmark_frames:
                draw_preview(
                    left_bgr,
                    tissue_mask,
                    tool_mask,
                    pixels_uv_ds,
                    output_dir / "previews" / f"{frame:06d}.jpg",
                )

            report = {
                "schema": "super_tissue_visible_surface_frame_report_v1",
                "passed": passed,
                "processing_signature": signature,
                "processing_settings": settings,
                "left_frame": frame,
                "left_timestamp_s": float(left_timestamps[frame]),
                "right_frame": right_frame,
                "right_timestamp_s": float(right_timestamps[right_frame]),
                "stereo_abs_delta_ms": stereo_delta_ms,
                "tool_mask_source_left_frame": tool_source_frame,
                "tool_mask_source_gap_frames": tool_source_gap,
                "right_tool_mask_source_frame": right_tool_source_frame,
                "right_tool_mask_source_gap_frames": right_tool_source_gap,
                "tissue_pixels_after_erosion": int(
                    np.count_nonzero(tissue_mask)
                ),
                "tool_pixels_after_dilation": int(
                    np.count_nonzero(tool_mask)
                ),
                "right_tissue_pixels_after_erosion": right_tissue_pixels,
                "right_tool_pixels_after_dilation": right_tool_pixels,
                "visible_semantic_pixels": int(
                    np.count_nonzero(visible_semantic)
                ),
                "visible_depth_pixels": int(len(rows)),
                "visible_depth_fraction": float(
                    len(rows) / max(np.count_nonzero(visible_semantic), 1)
                ),
                "voxel_point_count": before_outlier_count,
                "outlier_removed_count": int(
                    before_outlier_count - np.count_nonzero(keep)
                ),
                "output_point_count": point_count,
                "depth_mm_min_p05_p50_p95_max": five_number(z, 1000.0),
                "table_z_mm_min_p05_p50_p95_max": five_number(
                    points_table_ds[:, 2], 1000.0
                ),
                "lr_error_px_min_p05_p50_p95_max": five_number(
                    lr_error_ds
                ),
                "outlier_filter": outlier_metrics,
                "finite": finite,
                "model": model_record,
                "elapsed_seconds": float(time.perf_counter() - started),
            }
            atomic_json(json_path, report)
            print(
                f"[{progress_index}/{len(pending)}] frame={frame} "
                f"right={right_frame} dt={stereo_delta_ms:.2f}ms "
                f"points={point_count} "
                f"visible={report['visible_depth_fraction']:.3f} "
                f"elapsed={report['elapsed_seconds']:.2f}s "
                f"passed={passed}",
                flush=True,
            )
            if not passed:
                raise RuntimeError(
                    f"Frame {frame} failed surface observation gates"
                )
    finally:
        tool_masks.close()

    atomic_json(
        output_dir
        / f"shard_{args.shard_index:02d}_of_{args.shard_count:02d}.json",
        {
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "device": str(device),
            "frames": frames_this_shard,
            "pending_processed": pending,
            "foundation_checkpoint": model_record,
            "processing_signature": signature,
            "complete": True,
        },
    )
    print(
        f"shard {args.shard_index}/{args.shard_count} complete: "
        f"{len(pending)} frames"
    )


if __name__ == "__main__":
    main()
