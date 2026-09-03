#!/usr/bin/env python3

"""Validate the shared-pivot GUI jaw reconstruction against the legacy adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


REPO = Path(__file__).resolve().parents[1]
TRACK_ROOT = REPO / "data/super/psm_tracking"
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

from super_psm_tracking_common import (  # noqa: E402
    TrackingInputs,
    pose_to_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare legacy independent jaw links with shared-pivot jaws."
    )
    parser.add_argument(
        "--states",
        type=Path,
        default=TRACK_ROOT
        / "part_pose_correction_full/tracking_states_part_corrected.npz",
    )
    parser.add_argument(
        "--registration",
        type=Path,
        default=TRACK_ROOT
        / "part_pose_correction_full/paper_to_gui_registration.npz",
    )
    parser.add_argument(
        "--fixed-poses",
        type=Path,
        default=TRACK_ROOT
        / "part_pose_correction_full/visual_poses_part_corrected.npz",
    )
    parser.add_argument(
        "--gaussians",
        type=Path,
        default=REPO / "data/super/psm_robot/psm_surface_gaussians.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TRACK_ROOT / "jaw_shared_pivot_validation",
    )
    parser.add_argument("--frames", default="540,560,800,1280")
    return parser.parse_args()


def percentile_summary(values: np.ndarray) -> list[float]:
    return np.percentile(values, [0, 5, 50, 95, 100]).tolist()


def transformed_points(
    local_points: np.ndarray, pose: np.ndarray
) -> np.ndarray:
    transform = pose_to_matrix(pose)
    return (
        local_points @ transform[:3, :3].T + transform[:3, 3]
    )


def jaw_metrics(
    poses: np.ndarray,
    local_jaws: tuple[np.ndarray, np.ndarray],
) -> dict[str, np.ndarray]:
    frame_count = len(poses)
    pivot_gap_mm = np.empty(frame_count, dtype=np.float64)
    centroid_gap_mm = np.empty(frame_count, dtype=np.float64)
    camera_depth_gap_mm = np.empty(frame_count, dtype=np.float64)
    surface_gap_mm = np.empty(frame_count, dtype=np.float64)
    hinge_axis_parallel_error_deg = np.empty(frame_count, dtype=np.float64)
    shaft_in_jaw_plane_error_deg = np.empty(frame_count, dtype=np.float64)
    for frame_index in range(frame_count):
        points = tuple(
            transformed_points(local_jaws[index], poses[frame_index, 5 + index])
            for index in range(2)
        )
        jaw_matrices = tuple(
            pose_to_matrix(poses[frame_index, 5 + index]) for index in range(2)
        )
        # Both URDF jaw child-link origins are the shared hinge point and both
        # local z axes are the physical hinge axis.
        pivots = tuple(matrix[:3, 3] for matrix in jaw_matrices)
        pivot_gap_mm[frame_index] = (
            np.linalg.norm(pivots[0] - pivots[1]) * 1000.0
        )
        centroid_delta = points[0].mean(axis=0) - points[1].mean(axis=0)
        centroid_gap_mm[frame_index] = (
            np.linalg.norm(centroid_delta) * 1000.0
        )
        camera_depth_gap_mm[frame_index] = abs(
            centroid_delta[2] * 1000.0
        )
        surface_gap_mm[frame_index] = (
            cKDTree(points[1]).query(points[0])[0].min() * 1000.0
        )
        hinge_axis_parallel_error_deg[frame_index] = np.degrees(
            np.arccos(
                np.clip(
                    abs(float(jaw_matrices[0][:3, 2] @ jaw_matrices[1][:3, 2])),
                    0.0,
                    1.0,
                )
            )
        )
        shaft_direction = (
            poses[frame_index, 1, :3] - poses[frame_index, 0, :3]
        )
        shaft_direction /= np.linalg.norm(shaft_direction)
        hinge_axis = jaw_matrices[0][:3, 2]
        shaft_in_jaw_plane_error_deg[frame_index] = np.degrees(
            np.arcsin(
                np.clip(abs(float(shaft_direction @ hinge_axis)), 0.0, 1.0)
            )
        )
    return {
        "pivot_gap_mm": pivot_gap_mm,
        "centroid_gap_mm": centroid_gap_mm,
        "camera_depth_gap_mm": camera_depth_gap_mm,
        "surface_gap_mm": surface_gap_mm,
        "hinge_axis_parallel_error_deg": hinge_axis_parallel_error_deg,
        "shaft_in_jaw_plane_error_deg": shaft_in_jaw_plane_error_deg,
    }


def plot_comparison(
    path: Path,
    frames: list[int],
    legacy: np.ndarray,
    fixed: np.ndarray,
    local_jaws: tuple[np.ndarray, np.ndarray],
    legacy_metrics: dict[str, np.ndarray],
    fixed_metrics: dict[str, np.ndarray],
) -> None:
    figure = plt.figure(figsize=(4.2 * len(frames), 8.0), dpi=180)
    for row, (name, poses, metrics) in enumerate(
        (
            ("Legacy independent links", legacy, legacy_metrics),
            ("Fixed shared pivot", fixed, fixed_metrics),
        )
    ):
        for column, frame_index in enumerate(frames):
            axis = figure.add_subplot(
                2, len(frames), row * len(frames) + column + 1, projection="3d"
            )
            points = tuple(
                transformed_points(
                    local_jaws[index], poses[frame_index, 5 + index]
                )
                for index in range(2)
            )
            pivots = tuple(
                pose_to_matrix(poses[frame_index, 5 + index])[:3, 3]
                for index in range(2)
            )
            center = np.mean(np.concatenate(points), axis=0)
            for jaw_points, color, label in zip(
                points,
                ("#d000ff", "#ff3030"),
                ("jaw 1", "jaw 2"),
                strict=True,
            ):
                local = (jaw_points - center) * 1000.0
                axis.scatter(
                    local[:, 0],
                    local[:, 1],
                    local[:, 2],
                    s=5,
                    color=color,
                    alpha=0.8,
                    label=label,
                )
            for pivot, color in zip(
                pivots, ("#6a00a8", "#8b0000"), strict=True
            ):
                local_pivot = (pivot - center) * 1000.0
                axis.scatter(
                    *local_pivot,
                    marker="x",
                    s=70,
                    linewidth=2.5,
                    color=color,
                )
            axis.set_title(
                f"{name} | frame {frame_index}\n"
                f"pivot gap={metrics['pivot_gap_mm'][frame_index]:.3f} mm, "
                f"plane error={metrics['shaft_in_jaw_plane_error_deg'][frame_index]:.2f} deg",
                fontsize=9,
            )
            axis.set_xlabel("camera x / mm", fontsize=7)
            axis.set_ylabel("camera y / mm", fontsize=7)
            axis.set_zlabel("camera depth / mm", fontsize=7)
            axis.set_xlim(-8, 8)
            axis.set_ylim(-8, 8)
            axis.set_zlim(-8, 8)
            axis.set_box_aspect((1, 1, 1))
            axis.view_init(elev=22, azim=-58)
            axis.tick_params(labelsize=6)
            if row == 0 and column == 0:
                axis.legend(loc="upper left", fontsize=7)
    figure.suptitle(
        "GUI jaw geometry: separate first-frame adapters vs one physical pivot",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = TrackingInputs.load()
    with np.load(args.states) as states:
        prior_ctr = states["prior_ctr"].astype(np.float32)
        prior_joints = states["prior_joints"].astype(np.float32)
        corrected_ctr = states["corrected_ctr"].astype(np.float32)
        corrected_joints = states["corrected_joints"].astype(np.float32)
    with np.load(args.registration) as registration:
        conversion_options = {
            "registration_ctr": registration["registration_ctr"],
            "registration_joints": registration["registration_joints"],
            "camera_alignment": registration["T_rectified_camera_alignment"],
        }
    legacy = inputs.paper_states_to_visual_poses(
        corrected_ctr, corrected_joints, **conversion_options
    )
    with np.load(args.fixed_poses) as fixed_asset:
        fixed = fixed_asset["poses_rect_camera_xyz_xyzw"].astype(np.float32)
    with np.load(args.gaussians) as gaussian_asset:
        means = gaussian_asset["means"].astype(np.float64)
        link_ids = gaussian_asset["link_ids"].astype(np.int64)
    local_jaws = (means[link_ids == 5], means[link_ids == 6])
    legacy_metrics = jaw_metrics(legacy, local_jaws)
    fixed_metrics = jaw_metrics(fixed, local_jaws)
    closed = np.max(prior_joints[:, 2:], axis=1) <= 1.0e-6
    closed_indices = np.flatnonzero(closed)
    closed_runs = np.split(
        closed_indices,
        np.flatnonzero(np.diff(closed_indices) > 1) + 1,
    )
    stable_closed_indices = max(closed_runs, key=len)
    stable_closed = np.zeros_like(closed)
    stable_closed[stable_closed_indices] = True
    frames = sorted(
        {
            int(value)
            for value in args.frames.split(",")
            if value.strip()
        }
    )
    if any(frame < 0 or frame >= len(fixed) for frame in frames):
        raise IndexError("Requested comparison frame is outside the sequence")
    figure_path = args.output_dir / "jaw_shared_pivot_comparison.png"
    plot_comparison(
        figure_path,
        frames,
        legacy,
        fixed,
        local_jaws,
        legacy_metrics,
        fixed_metrics,
    )
    report = {
        "frame_count": len(fixed),
        "stable_closed_frame_start": int(stable_closed_indices[0]),
        "stable_closed_frame_end": int(stable_closed_indices[-1]),
        "stable_closed_frame_count": int(len(stable_closed_indices)),
        "summary_order": ["min", "p05", "p50", "p95", "max"],
        "legacy_closed": {
            name: percentile_summary(values[stable_closed])
            for name, values in legacy_metrics.items()
        },
        "fixed_closed": {
            name: percentile_summary(values[stable_closed])
            for name, values in fixed_metrics.items()
        },
        "comparison_frames": frames,
        "comparison_figure": str(figure_path),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
