#!/usr/bin/env python3
"""Validate complete stage-B surfaces, stratified LR audit, and keypoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_ROOT = REPO_ROOT / "data/super/tissue_calibration_v1"
DEFAULT_MANIFEST = CALIBRATION_ROOT / "stage_a_frozen_manifest.json"
DEFAULT_PRIMARY = CALIBRATION_ROOT / "stage_b_surface_observations"
DEFAULT_BASELINE = CALIBRATION_ROOT / "stage_b_dense_baseline"
DEFAULT_AUDIT = CALIBRATION_ROOT / "stage_b_lr_stratified_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--primary-root", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument(
        "--baseline-root", type=Path, default=DEFAULT_BASELINE
    )
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument(
        "--keypoint-report",
        type=Path,
        default=DEFAULT_PRIMARY / "keypoints/report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CALIBRATION_ROOT / "stage_b_validation_report.json",
    )
    parser.add_argument("--maximum-chamfer-p95-mm", type=float, default=3.0)
    parser.add_argument(
        "--minimum-visible-depth-fraction", type=float, default=0.50
    )
    parser.add_argument(
        "--maximum-initial-rest-p95-mm", type=float, default=2.0
    )
    parser.add_argument("--minimum-lift-gain-mm", type=float, default=5.0)
    parser.add_argument("--minimum-release-drop-mm", type=float, default=5.0)
    parser.add_argument("--maximum-rebound-error-mm", type=float, default=5.0)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def five_number(values: np.ndarray, scale: float = 1.0) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)] * scale
    return [
        float(value) for value in np.percentile(values, [0, 5, 50, 95, 100])
    ]


def load_points(root: Path, frame: int) -> np.ndarray:
    path = root / "frames" / f"{frame:06d}.npz"
    with np.load(path, allow_pickle=False) as archive:
        points = np.asarray(archive["points_table"], dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not np.isfinite(points).all()
    ):
        raise RuntimeError(f"Invalid points: {path}")
    return points


def symmetric_distance(
    first: np.ndarray, second: np.ndarray
) -> np.ndarray:
    first_to_second = cKDTree(second).query(first, k=1, workers=-1)[0]
    second_to_first = cKDTree(first).query(second, k=1, workers=-1)[0]
    return np.concatenate((first_to_second, second_to_first))


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    primary_report_path = args.primary_root / "stage_b_report.json"
    baseline_report_path = args.baseline_root / "stage_b_report.json"
    audit_report_path = args.audit_root / "stage_b_report.json"
    primary_report = read_json(primary_report_path)
    baseline_report = read_json(baseline_report_path)
    audit_report = read_json(audit_report_path)
    keypoint_report = read_json(args.keypoint_report)
    offline = manifest["sequence"]["offline_stereo"]
    excluded = set(
        int(frame) for frame in offline["stereo_excluded_left_frames"]
    )
    frame_start, frame_end = (
        int(value)
        for value in offline[
            "stereo_calibration_left_frame_range_inclusive"
        ]
    )
    eligible_frames = [
        frame
        for frame in range(frame_start, frame_end + 1)
        if frame not in excluded
    ]
    frame_quality_violations: dict[str, list[int]] = {
        "not_passed_or_nonfinite": [],
        "point_count_out_of_range": [],
        "visible_depth_fraction_too_low": [],
        "stereo_delta_over_20ms": [],
        "tool_mask_missing_or_semantic_mask_expanded": [],
        "right_semantic_mask_missing": [],
        "baseline_chamfer_p95_over_threshold": [],
    }
    full_sequence_chamfer_p95_mm: list[float] = []
    for frame in eligible_frames:
        frame_report = read_json(
            args.primary_root / "frames" / f"{frame:06d}.json"
        )
        if not (
            frame_report.get("passed", False)
            and frame_report.get("finite", False)
        ):
            frame_quality_violations["not_passed_or_nonfinite"].append(frame)
        if not 1000 <= int(frame_report["output_point_count"]) <= 9000:
            frame_quality_violations["point_count_out_of_range"].append(frame)
        if (
            float(frame_report["visible_depth_fraction"])
            < args.minimum_visible_depth_fraction
        ):
            frame_quality_violations[
                "visible_depth_fraction_too_low"
            ].append(frame)
        if float(frame_report["stereo_abs_delta_ms"]) > 20.0:
            frame_quality_violations["stereo_delta_over_20ms"].append(frame)
        if not (
            int(frame_report["tool_pixels_after_dilation"]) > 0
            and int(frame_report["visible_semantic_pixels"])
            <= int(frame_report["tissue_pixels_after_erosion"])
        ):
            frame_quality_violations[
                "tool_mask_missing_or_semantic_mask_expanded"
            ].append(frame)
        if not (
            int(frame_report.get("right_tissue_pixels_after_erosion", 0)) > 0
            and int(frame_report.get("right_tool_pixels_after_dilation", 0))
            > 0
            and int(
                frame_report.get(
                    "right_tool_mask_source_gap_frames", 999
                )
            )
            <= 1
        ):
            frame_quality_violations[
                "right_semantic_mask_missing"
            ].append(frame)
        primary_points = load_points(args.primary_root, frame)
        baseline_points = load_points(args.baseline_root, frame)
        frame_chamfer_p95_mm = float(
            np.percentile(
                symmetric_distance(primary_points, baseline_points),
                95,
            )
            * 1000.0
        )
        full_sequence_chamfer_p95_mm.append(frame_chamfer_p95_mm)
        if frame_chamfer_p95_mm > args.maximum_chamfer_p95_mm:
            frame_quality_violations[
                "baseline_chamfer_p95_over_threshold"
            ].append(frame)

    audit_frames = [
        int(frame) for frame in audit_report.get("selected_frames") or []
    ]
    audit_lr_threshold = float(
        audit_report["parameters"]["lr_threshold_px"]
    )
    audit_quality_violations: dict[str, list[int]] = {
        "not_passed_or_nonfinite": [],
        "lr_error_over_threshold": [],
        "right_semantic_mask_missing": [],
    }
    for frame in audit_frames:
        frame_report = read_json(
            args.audit_root / "frames" / f"{frame:06d}.json"
        )
        if not (
            frame_report.get("passed", False)
            and frame_report.get("finite", False)
        ):
            audit_quality_violations[
                "not_passed_or_nonfinite"
            ].append(frame)
        if (
            float(
                frame_report[
                    "lr_error_px_min_p05_p50_p95_max"
                ][4]
            )
            > audit_lr_threshold + 1.0e-5
        ):
            audit_quality_violations[
                "lr_error_over_threshold"
            ].append(frame)
        if not (
            int(frame_report.get("right_tissue_pixels_after_erosion", 0)) > 0
            and int(frame_report.get("right_tool_pixels_after_dilation", 0))
            > 0
        ):
            audit_quality_violations[
                "right_semantic_mask_missing"
            ].append(frame)

    landmark_items = [
        (name, int(item["left_frame"]))
        for name, item in manifest["sequence"]["landmarks"].items()
    ]
    landmark_reports: dict[str, Any] = {}
    chamfer_p95_values: list[float] = []
    maximum_z: dict[str, float] = {}
    point_count_gate = True
    tool_removal_gate = True
    for name, frame in landmark_items:
        primary = load_points(args.primary_root, frame)
        baseline = load_points(args.baseline_root, frame)
        audit = load_points(args.audit_root, frame)
        baseline_distances = symmetric_distance(primary, baseline)
        audit_distances = symmetric_distance(primary, audit)
        baseline_distance_summary = five_number(
            baseline_distances, 1000.0
        )
        audit_distance_summary = five_number(audit_distances, 1000.0)
        chamfer_p95_values.append(audit_distance_summary[3])
        maximum_z[name] = float(primary[:, 2].max() * 1000.0)
        primary_frame_report = read_json(
            args.primary_root / "frames" / f"{frame:06d}.json"
        )
        point_count_gate &= 1000 <= len(primary) <= 9000
        tool_removal_gate &= (
            primary_frame_report["tool_pixels_after_dilation"] > 0
            and primary_frame_report["visible_semantic_pixels"]
            < primary_frame_report["tissue_pixels_after_erosion"]
        )
        landmark_reports[name] = {
            "frame": frame,
            "primary_point_count": len(primary),
            "dense_baseline_point_count": len(baseline),
            "lr_audit_point_count": len(audit),
            "primary_to_dense_baseline_symmetric_distance_mm_"
            "min_p05_p50_p95_max": baseline_distance_summary,
            "primary_to_lr_audit_symmetric_distance_mm_"
            "min_p05_p50_p95_max": audit_distance_summary,
            "primary_table_z_mm_min_p05_p50_p95_max": five_number(
                primary[:, 2], 1000.0
            ),
        }

    initial_z = maximum_z["initial_reference"]
    peak_z = maximum_z["maximum_retraction"]
    release_z = maximum_z["release_complete"]
    rebound_z = maximum_z["rebound_reference"]
    lift_gain = peak_z - initial_z
    release_drop = peak_z - release_z
    rebound_error = abs(rebound_z - initial_z)
    initial_points = load_points(
        args.primary_root,
        int(manifest["sequence"]["landmarks"]["initial_reference"]["left_frame"]),
    )
    tissue_asset_path = REPO_ROOT / manifest["tissue"]["asset"]["path"]
    with np.load(tissue_asset_path, allow_pickle=False) as tissue:
        rest_positions = np.asarray(
            tissue["rest_positions_table"], dtype=np.float64
        )
        top_mask = np.asarray(tissue["top_node_mask"], dtype=bool)
    initial_to_rest_mm = (
        cKDTree(rest_positions[top_mask])
        .query(initial_points, k=1, workers=-1)[0]
        * 1000.0
    )
    initial_rest_summary = five_number(initial_to_rest_mm)
    gates = {
        "primary_full_sequence_complete": bool(
            primary_report.get("passed", False)
            and primary_report.get("selected_frame_count")
            == len(eligible_frames)
            and primary_report.get("completed_frame_count")
            == len(eligible_frames)
        ),
        "dense_baseline_full_sequence_complete": bool(
            baseline_report.get("passed", False)
            and baseline_report.get("selected_frame_count")
            == len(eligible_frames)
            and baseline_report.get("completed_frame_count")
            == len(eligible_frames)
        ),
        "primary_uses_complete_dense_stereo_sequence": (
            primary_report["parameters"]["lr_consistency"] is False
        ),
        "primary_uses_bilateral_semantic_consistency": (
            primary_report["parameters"].get(
                "stereo_semantic_consistency", False
            )
            is True
        ),
        "baseline_uses_dense_foundation": (
            baseline_report["parameters"]["lr_consistency"] is False
        ),
        "stratified_lr_audit_complete": bool(
            audit_report.get("passed", False)
            and len(audit_frames) >= 64
            and set(frame for _, frame in landmark_items)
            <= set(audit_frames)
            and audit_report["parameters"]["lr_consistency"] is True
            and audit_report["parameters"].get(
                "stereo_semantic_consistency", False
            )
            is True
        ),
        "all_stratified_lr_audit_frames_pass_quality_gates": not any(
            audit_quality_violations.values()
        ),
        "all_primary_frames_pass_quality_gates": not any(
            frame_quality_violations.values()
        ),
        "landmark_point_counts_in_range": bool(point_count_gate),
        "tool_occlusion_removed_at_landmarks": bool(tool_removal_gate),
        "primary_to_lr_audit_landmark_chamfer_p95_bounded": bool(
            max(chamfer_p95_values) <= args.maximum_chamfer_p95_mm
        ),
        "full_sequence_semantic_to_dense_baseline_chamfer_p95_bounded": (
            not frame_quality_violations[
                "baseline_chamfer_p95_over_threshold"
            ]
        ),
        "lift_is_observed": bool(lift_gain >= args.minimum_lift_gain_mm),
        "release_drop_is_observed": bool(
            release_drop >= args.minimum_release_drop_mm
        ),
        "post_release_rebound_near_initial_height": bool(
            rebound_error <= args.maximum_rebound_error_mm
        ),
        "initial_observation_matches_rest_surface": bool(
            initial_rest_summary[3] <= args.maximum_initial_rest_p95_mm
        ),
        "grasp_keypoint_tracks_complete": bool(
            keypoint_report.get("passed", False)
            and keypoint_report.get("selected_track_count", 0) >= 8
            and keypoint_report.get("surface_3d_fraction", 0.0) >= 0.60
        ),
    }
    passed = all(gates.values())
    report = {
        "schema": "super_tissue_surface_observations_stage_b_validation_v1",
        "stage": "B_visible_surface_observations_validation",
        "passed": passed,
        "gates": gates,
        "landmarks": landmark_reports,
        "temporal_shape_evidence": {
            "initial_max_z_mm": initial_z,
            "maximum_retraction_max_z_mm": peak_z,
            "release_complete_max_z_mm": release_z,
            "rebound_reference_max_z_mm": rebound_z,
            "lift_gain_mm": lift_gain,
            "release_drop_mm": release_drop,
            "rebound_max_height_error_mm": rebound_error,
        },
        "initial_observation_to_rest_surface_mm_min_p05_p50_p95_max": (
            initial_rest_summary
        ),
        "full_sequence_baseline_chamfer_p95_mm_min_p05_p50_p95_max": (
            five_number(np.asarray(full_sequence_chamfer_p95_mm))
        ),
        "thresholds": {
            "maximum_chamfer_p95_mm": args.maximum_chamfer_p95_mm,
            "minimum_visible_depth_fraction": (
                args.minimum_visible_depth_fraction
            ),
            "maximum_initial_rest_p95_mm": args.maximum_initial_rest_p95_mm,
            "minimum_lift_gain_mm": args.minimum_lift_gain_mm,
            "minimum_release_drop_mm": args.minimum_release_drop_mm,
            "maximum_rebound_error_mm": args.maximum_rebound_error_mm,
        },
        "full_sequence_quality_violations": frame_quality_violations,
        "stratified_lr_audit_quality_violations": (
            audit_quality_violations
        ),
        "inputs": {
            "stage_a_manifest": {
                "path": str(args.manifest.resolve()),
                "sha256": sha256(args.manifest),
            },
            "primary_report": {
                "path": str(primary_report_path.resolve()),
                "sha256": sha256(primary_report_path),
            },
            "dense_baseline_report": {
                "path": str(baseline_report_path.resolve()),
                "sha256": sha256(baseline_report_path),
            },
            "stratified_lr_audit_report": {
                "path": str(audit_report_path.resolve()),
                "sha256": sha256(audit_report_path),
            },
            "keypoint_report": {
                "path": str(args.keypoint_report.resolve()),
                "sha256": sha256(args.keypoint_report),
            },
        },
        "next_stage": (
            "C: deterministic no-visual-force replay using the frozen PSM "
            "trajectory; no residual mapping or stiffness updates yet."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
