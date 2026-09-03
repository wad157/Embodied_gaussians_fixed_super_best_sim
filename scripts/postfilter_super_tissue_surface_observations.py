#!/usr/bin/env python3
"""Apply right-view tissue/tool semantics to complete stage-B point clouds.

This reuses saved 3D points and their left pixels, so it does not rerun the
stereo network. The right pixel is recovered from calibrated disparity, and a
point is retained only if it lands inside eroded right tissue and outside the
dilated right tool mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from build_super_tissue_surface_observations import (
    ToolMasks,
    atomic_json,
    atomic_npz,
    draw_preview,
    five_number,
    load_json,
    output_paths,
    sha256,
    unpack_tissue_mask,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_ROOT = REPO_ROOT / "data/super/tissue_calibration_v1"
DEFAULT_MANIFEST = CALIBRATION_ROOT / "stage_a_frozen_manifest.json"
DEFAULT_SOURCE = CALIBRATION_ROOT / "stage_b_dense_baseline"
DEFAULT_OUTPUT = CALIBRATION_ROOT / "stage_b_surface_observations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tissue-erosion-px", type=int, default=2)
    parser.add_argument("--tool-dilation-px", type=int, default=30)
    parser.add_argument("--minimum-points", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def settings_signature(settings: dict[str, Any]) -> str:
    canonical = json.dumps(
        settings, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.source_root = args.source_root.resolve()
    args.output_dir = args.output_dir.resolve()
    manifest = load_json(args.manifest)
    source_report_path = args.source_root / "stage_b_report.json"
    source_report = load_json(source_report_path)
    if not (
        source_report.get("passed", False)
        and source_report.get("selected_frame_count") == 1433
        and source_report["parameters"]["lr_consistency"] is False
    ):
        raise RuntimeError("A complete dense stage-B baseline is required")

    offline = manifest["sequence"]["offline_stereo"]
    excluded = set(
        int(frame) for frame in offline["stereo_excluded_left_frames"]
    )
    start, end = (
        int(value)
        for value in offline[
            "stereo_calibration_left_frame_range_inclusive"
        ]
    )
    frames = [
        frame for frame in range(start, end + 1) if frame not in excluded
    ]
    right_metadata = load_json(
        repo_path(offline["right_metadata"]["path"])
    )
    width, height = (
        int(value) for value in right_metadata["resolution"]
    )
    calibration = load_json(
        repo_path(manifest["camera"]["rectified_calibration"]["path"])
    )
    fx = float(calibration["K_left_rect"][0][0])
    baseline = float(calibration["baseline_m"])
    cx_delta = float(
        calibration["K_right_rect"][0][2]
        - calibration["K_left_rect"][0][2]
    )
    observation_inputs = manifest["observation_inputs_for_stage_b"]
    right_tissue = np.load(
        repo_path(observation_inputs["right_tissue_masks"]["path"]),
        mmap_mode="r",
    )
    tool_masks = ToolMasks(
        repo_path(observation_inputs["stereo_tool_part_masks"]["path"]),
        (height, width),
    )
    source_report_sha = sha256(source_report_path)
    settings = {
        "source_report": str(source_report_path),
        "source_report_sha256": source_report_sha,
        "source_processing_signature": source_report[
            "processing_signature"
        ],
        "foundation_checkpoint": source_report["parameters"][
            "foundation_checkpoint"
        ],
        "foundation_iters": source_report["parameters"][
            "foundation_iters"
        ],
        "foundation_hierarchical": source_report["parameters"][
            "foundation_hierarchical"
        ],
        "lr_consistency": False,
        "stereo_semantic_consistency": True,
        "semantic_filter_mode": (
            "calibrated right reprojection of saved left-reference 3D points"
        ),
        "tissue_erosion_px": args.tissue_erosion_px,
        "tool_dilation_px": args.tool_dilation_px,
        "minimum_points": args.minimum_points,
        "maximum_points": source_report["parameters"]["maximum_points"],
    }
    signature = settings_signature(settings)
    (args.output_dir / "frames").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "previews").mkdir(parents=True, exist_ok=True)
    landmark_frames = {
        int(item["left_frame"])
        for item in manifest["sequence"]["landmarks"].values()
    }

    reports: list[dict[str, Any]] = []
    try:
        for index, frame in enumerate(frames, start=1):
            source_npz, source_json = output_paths(
                args.source_root, frame
            )
            output_npz, output_json = output_paths(args.output_dir, frame)
            if (
                not args.overwrite
                and output_npz.is_file()
                and output_json.is_file()
            ):
                existing = load_json(output_json)
                if existing.get("processing_signature") == signature:
                    reports.append(existing)
                    continue
                raise RuntimeError(
                    f"Frame {frame} has a different output configuration"
                )

            source_frame_report = load_json(source_json)
            with np.load(source_npz, allow_pickle=False) as archive:
                arrays = {
                    name: np.asarray(archive[name]) for name in archive.files
                }
            points_camera = np.asarray(
                arrays["points_left_camera"], dtype=np.float64
            )
            pixels = np.asarray(arrays["pixels_uv"], dtype=np.int64)
            right_frame = int(np.asarray(arrays["right_frame"]).item())
            disparity = (
                fx * baseline / points_camera[:, 2] - cx_delta
            )
            right_columns = np.rint(
                pixels[:, 0] - disparity
            ).astype(np.int64)

            tissue_mask = unpack_tissue_mask(
                right_tissue, right_frame, width
            )
            if args.tissue_erosion_px:
                size = args.tissue_erosion_px * 2 + 1
                tissue_mask = cv2.erode(
                    tissue_mask.astype(np.uint8),
                    np.ones((size, size), dtype=np.uint8),
                ).astype(bool)
            tool_mask, tool_source_frame, tool_source_gap = tool_masks.mask(
                right_frame, side="right"
            )
            if args.tool_dilation_px:
                size = args.tool_dilation_px * 2 + 1
                tool_mask = cv2.dilate(
                    tool_mask.astype(np.uint8),
                    np.ones((size, size), dtype=np.uint8),
                ).astype(bool)
            right_visible = tissue_mask & ~tool_mask
            inside = (
                (right_columns >= 0)
                & (right_columns < width)
                & (pixels[:, 1] >= 0)
                & (pixels[:, 1] < height)
            )
            keep = np.zeros(len(points_camera), dtype=bool)
            keep[inside] = right_visible[
                pixels[inside, 1], right_columns[inside]
            ]
            output_count = int(np.count_nonzero(keep))
            if output_count < args.minimum_points:
                raise RuntimeError(
                    f"Frame {frame} retained only {output_count} points"
                )

            filtered: dict[str, np.ndarray] = {}
            for name, value in arrays.items():
                if value.ndim > 0 and value.shape[0] == len(keep):
                    filtered[name] = value[keep]
                else:
                    filtered[name] = value
            atomic_npz(output_npz, **filtered)

            report = dict(source_frame_report)
            report.update(
                {
                    "schema": (
                        "super_tissue_visible_surface_observation_"
                        "bilateral_semantic_v1"
                    ),
                    "passed": True,
                    "processing_settings": settings,
                    "processing_signature": signature,
                    "source_output_point_count": len(keep),
                    "output_point_count": output_count,
                    "visible_depth_fraction": float(
                        output_count / max(len(keep), 1)
                    ),
                    "right_tissue_pixels_after_erosion": int(
                        np.count_nonzero(tissue_mask)
                    ),
                    "right_tool_pixels_after_dilation": int(
                        np.count_nonzero(tool_mask)
                    ),
                    "right_tool_mask_source_frame": tool_source_frame,
                    "right_tool_mask_source_gap_frames": tool_source_gap,
                    "right_semantic_retained_fraction": float(
                        output_count / max(len(keep), 1)
                    ),
                    "source_frame_npz_sha256": sha256(source_npz),
                    "source_frame_json_sha256": sha256(source_json),
                }
            )
            atomic_json(output_json, report)
            reports.append(report)
            if frame in landmark_frames:
                image_path = (
                    REPO_ROOT
                    / "data/super/grasp5_native/rgb"
                    / f"{frame:06d}-left.png"
                )
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                left_tissue_pixels = np.zeros((height, width), dtype=bool)
                left_tool_pixels = np.zeros((height, width), dtype=bool)
                draw_preview(
                    image,
                    left_tissue_pixels,
                    left_tool_pixels,
                    filtered["pixels_uv"],
                    args.output_dir
                    / "previews"
                    / f"{frame:06d}.jpg",
                )
            if index % 100 == 0 or index == len(frames):
                print(
                    f"[{index}/{len(frames)}] frame={frame} "
                    f"retained={output_count}/{len(keep)}",
                    flush=True,
                )
    finally:
        tool_masks.close()

    artifact_digest = hashlib.sha256()
    artifact_bytes = 0
    for frame in frames:
        for path in output_paths(args.output_dir, frame):
            artifact_digest.update(sha256(path).encode())
            artifact_bytes += path.stat().st_size
    point_counts = np.asarray(
        [report["output_point_count"] for report in reports]
    )
    retained = np.asarray(
        [report["right_semantic_retained_fraction"] for report in reports]
    )
    combined = {
        "schema": (
            "super_tissue_surface_observations_bilateral_semantic_stage_b_v1"
        ),
        "stage": "B_visible_surface_observations",
        "status": "complete",
        "passed": len(reports) == len(frames),
        "stage_a_manifest": {
            "path": str(args.manifest),
            "sha256": sha256(args.manifest),
        },
        "source_report": {
            "path": str(source_report_path),
            "sha256": source_report_sha,
        },
        "selected_frame_count": len(frames),
        "completed_frame_count": len(reports),
        "processing_signature": signature,
        "parameters": settings,
        "ordered_frame_artifact_digest_sha256": (
            artifact_digest.hexdigest()
        ),
        "frame_artifact_bytes": artifact_bytes,
        "summary": {
            "point_count_min_p05_p50_p95_max": five_number(point_counts),
            "right_semantic_retained_fraction_min_p05_p50_p95_max": (
                five_number(retained)
            ),
        },
        "coordinate_frame": "dense-ground-aligned table/world frame",
        "surface_observation": (
            "full-sequence left-reference stereo points with bilateral "
            "tissue/tool semantic consistency"
        ),
        "lr_consistency": (
            "validated separately on a stratified full-sequence audit"
        ),
        "residual_mapping_enabled": False,
        "stiffness_optimization_enabled": False,
    }
    atomic_json(args.output_dir / "stage_b_report.json", combined)
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
