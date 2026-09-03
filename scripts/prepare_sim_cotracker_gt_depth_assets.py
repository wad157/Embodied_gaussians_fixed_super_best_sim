#!/usr/bin/env python3
"""为仿真数据构建 CoTracker + 指定深度的固定范围物理观测资产。

二维运动只来自 RGB 上的 CoTracker 轨迹；三维提升采样显式指定的逐帧
float32 深度。默认保持旧的GT消融行为，也可传入RGB估计深度目录。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embodied_gaussians.physics_simulator.flow_depth_particle_observer import (  # noqa: E402
    FlowDepthObservationSettings,
    FlowDepthRangeBindingSettings,
    build_fixed_particle_range_bindings,
    observe_flow_depth_tracks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera", default="stereo_left")
    parser.add_argument(
        "--depth-dir",
        type=Path,
        help=(
            "可选的逐帧深度目录；未指定时保持原有仿真GT深度。"
            "RGB估计深度必须显式传入该参数。"
        ),
    )
    parser.add_argument(
        "--depth-filename-pattern",
        default=None,
        help="Python format，例如 {frame:06d}-depth.npy；默认按深度来源选择",
    )
    parser.add_argument(
        "--depth-source-label",
        default=None,
        help="写入资产的可审计深度来源标签",
    )
    parser.add_argument("--radius-mm", type=float, default=6.0)
    parser.add_argument("--fallback-radius-mm", type=float, default=8.0)
    parser.add_argument("--sigma-mm", type=float, default=3.0)
    parser.add_argument("--minimum-movable-particles", type=int, default=3)
    parser.add_argument("--maximum-depth-change-mm", type=float, default=15.0)
    parser.add_argument("--maximum-observed-flow-mm", type=float, default=20.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_bindings(
    path: Path,
    bindings,
    *,
    depth_source: str,
    depth_estimated_from_rgb: bool,
) -> None:
    np.savez_compressed(
        path,
        schema=np.asarray("fixedsuperbest.sim_cotracker_depth_bindings.v2"),
        depth_source=np.asarray(depth_source),
        depth_estimated_from_rgb=np.asarray(depth_estimated_from_rgb),
        track_valid=bindings.track_valid,
        particle_ids=bindings.particle_ids,
        particle_weights=bindings.particle_weights,
        support_counts=bindings.support_counts,
        movable_support_counts=bindings.movable_support_counts,
        binding_radius_m=bindings.binding_radius_m,
        initial_depth_m=bindings.initial_depth_m,
        depth_sampling_radius_px=bindings.depth_sampling_radius_px,
        initial_points_table=bindings.initial_points_table,
        nearest_surface_distance_m=bindings.nearest_surface_distance_m,
        binding_method=np.asarray(bindings.binding_method),
        surface_face_ids=(
            bindings.surface_face_ids
            if bindings.surface_face_ids is not None
            else np.full(len(bindings.track_valid), -1, dtype=np.int32)
        ),
        surface_projection_points_table=(
            bindings.surface_projection_points_table
            if bindings.surface_projection_points_table is not None
            else np.full((len(bindings.track_valid), 3), np.nan, dtype=np.float32)
        ),
    )


def five_number(values: np.ndarray, scale: float = 1.0) -> list[float] | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)] * scale
    if not len(finite):
        return None
    return np.percentile(finite, [0, 5, 50, 95, 100]).tolist()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    tracks_path = args.tracks.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    uses_estimated_depth = args.depth_dir is not None
    depth_dir = (
        args.depth_dir.expanduser().resolve()
        if uses_estimated_depth
        else dataset / "ground_truth/depth" / args.camera
    )
    depth_pattern = args.depth_filename_pattern or (
        "{frame:06d}-depth.npy" if uses_estimated_depth else "{frame:06d}.npy"
    )
    depth_source_label = args.depth_source_label or (
        "external_rgb_estimated_depth"
        if uses_estimated_depth
        else "simulator_ground_truth_float32_depth_map"
    )
    if uses_estimated_depth and args.depth_source_label is None:
        raise ValueError("RGB估计深度必须显式指定 --depth-source-label")

    def depth_path(frame: int) -> Path:
        return depth_dir / depth_pattern.format(frame=int(frame))

    bindings_path = output / "bindings.npz"
    observations_path = output / "observations.npz"
    report_path = output / "report.json"
    collisions = [
        path for path in (bindings_path, observations_path, report_path) if path.exists()
    ]
    if collisions:
        raise FileExistsError(
            "拒绝覆盖已有光流深度资产：\n- "
            + "\n- ".join(str(path) for path in collisions)
        )
    required = (
        tracks_path,
        dataset / "gui_assets/tissue_fixedsuperbest.npz",
        dataset / "videos" / f"{args.camera}.json",
        dataset / "cameras.json",
        depth_path(0),
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(tracks_path, allow_pickle=False) as loaded:
        source_frames = loaded["source_frame_indices"].astype(np.int32)
        pixels = loaded["tracks_original_px"].astype(np.float64)
        tracker_valid = loaded["visibility"].astype(bool) & loaded[
            "dynamic_tissue_valid"
        ].astype(bool)
        tracker_confidence = (
            loaded["tracking_confidence"].astype(np.float64)
            if "tracking_confidence" in loaded.files
            else np.ones_like(tracker_valid, dtype=np.float64)
        )
    if len(source_frames) < 2 or int(source_frames[0]) != 0:
        raise ValueError("轨迹必须从第0帧开始并至少包含两帧")
    if np.any(np.diff(source_frames) <= 0):
        raise ValueError("轨迹源帧必须严格递增")
    if pixels.shape[:2] != tracker_valid.shape:
        raise ValueError("轨迹像素和有效性形状不一致")
    if tracker_confidence.shape != tracker_valid.shape:
        raise ValueError("轨迹置信度形状不一致")

    camera_metadata = json.loads(
        (dataset / "videos" / f"{args.camera}.json").read_text(encoding="utf-8")
    )
    camera_poses = json.loads(
        (dataset / "cameras.json").read_text(encoding="utf-8")
    )
    intrinsic = np.asarray(camera_metadata["K"], dtype=np.float64)
    x_world_camera = np.asarray(
        camera_poses[args.camera]["X_WC_ros_optical"], dtype=np.float64
    )
    with np.load(
        dataset / "gui_assets/tissue_fixedsuperbest.npz", allow_pickle=False
    ) as asset:
        rest_positions = asset["rest_positions_table"].astype(np.float64)
        surface_mask = asset["surface_node_mask"].astype(bool)
        fixed_mask = asset["fixed_mask"].astype(bool)

    binding_settings = FlowDepthRangeBindingSettings(
        radius_m=args.radius_mm * 1.0e-3,
        fallback_radius_m=args.fallback_radius_mm * 1.0e-3,
        sigma_m=args.sigma_mm * 1.0e-3,
        minimum_movable_particles=args.minimum_movable_particles,
        maximum_depth_sampling_radius_px=2,
    )
    first_depth = np.load(depth_path(0), allow_pickle=False)
    bindings = build_fixed_particle_range_bindings(
        initial_pixels_uv=pixels[0],
        depth=first_depth,
        intrinsic=intrinsic,
        x_table_camera=x_world_camera,
        rest_positions_table=rest_positions,
        surface_mask=surface_mask,
        fixed_mask=fixed_mask,
        initial_track_valid=tracker_valid[0],
        settings=binding_settings,
    )
    output.mkdir(parents=True, exist_ok=True)
    save_bindings(
        bindings_path,
        bindings,
        depth_source=depth_source_label,
        depth_estimated_from_rgb=uses_estimated_depth,
    )

    observation_settings = FlowDepthObservationSettings(
        maximum_depth_sampling_radius_px=2,
        minimum_depth_m=0.035,
        maximum_depth_m=0.250,
        maximum_depth_change_m=args.maximum_depth_change_mm * 1.0e-3,
        maximum_observed_flow_m=args.maximum_observed_flow_mm * 1.0e-3,
    )
    observations = []
    for pair_index in range(len(source_frames) - 1):
        current_frame = int(source_frames[pair_index])
        next_frame = int(source_frames[pair_index + 1])
        current_depth_file = depth_path(current_frame)
        next_depth_file = depth_path(next_frame)
        if not current_depth_file.is_file() or not next_depth_file.is_file():
            raise FileNotFoundError(
                f"缺少深度帧：{current_depth_file} / {next_depth_file}"
            )
        current_depth = np.load(current_depth_file, allow_pickle=False)
        next_depth = np.load(next_depth_file, allow_pickle=False)
        confidence = np.minimum(
            tracker_confidence[pair_index], tracker_confidence[pair_index + 1]
        )
        observations.append(
            observe_flow_depth_tracks(
                current_pixels_uv=pixels[pair_index],
                next_pixels_uv=pixels[pair_index + 1],
                current_depth=current_depth,
                next_depth=next_depth,
                intrinsic=intrinsic,
                x_table_camera=x_world_camera,
                current_track_valid=(
                    tracker_valid[pair_index] & bindings.track_valid
                ),
                next_track_valid=(
                    tracker_valid[pair_index + 1] & bindings.track_valid
                ),
                confidence=confidence,
                settings=observation_settings,
            )
        )

    track_valid = np.stack([item.track_valid for item in observations])
    confidence = np.stack([item.confidence for item in observations])
    observed_flow = np.stack([item.observed_flow_table for item in observations])
    current_points = np.stack([item.current_points_table for item in observations])
    next_points = np.stack([item.next_points_table for item in observations])
    current_depth = np.stack([item.current_depth_m for item in observations])
    next_depth = np.stack([item.next_depth_m for item in observations])
    np.savez_compressed(
        observations_path,
        schema=np.asarray("fixedsuperbest.sim_cotracker_depth_observations.v2"),
        depth_source=np.asarray(depth_source_label),
        depth_estimated_from_rgb=np.asarray(uses_estimated_depth),
        tracker_source=np.asarray("CoTracker3_RGB_offline"),
        current_source_frames=source_frames[:-1],
        next_source_frames=source_frames[1:],
        track_valid=track_valid,
        confidence=confidence,
        observed_flow_table=observed_flow,
        current_points_table=current_points,
        next_points_table=next_points,
        current_depth_m=current_depth,
        next_depth_m=next_depth,
    )

    valid_flow_mm = np.linalg.norm(observed_flow[track_valid], axis=1) * 1.0e3
    report = {
        "schema": "fixedsuperbest.sim_cotracker_depth_assets_report.v2",
        "dataset": str(dataset),
        "experiment_class": (
            "rgb_estimated_depth" if uses_estimated_depth else "oracle_gt_depth"
        ),
        "rgb_motion_source": "CoTracker3 offline tracks",
        "depth_source": depth_source_label,
        "depth_directory": str(depth_dir),
        "depth_filename_pattern": depth_pattern,
        "depth_estimated_from_rgb": uses_estimated_depth,
        "uses_tissue_trajectory_ground_truth": False,
        "camera": args.camera,
        "source_frame_count": int(len(source_frames)),
        "pair_count": int(len(observations)),
        "requested_track_count": int(pixels.shape[1]),
        "bound_track_count": int(bindings.track_valid.sum()),
        "valid_observations": int(track_valid.sum()),
        "mean_valid_tracks_per_pair": float(track_valid.sum(axis=1).mean()),
        "support_count_five_number": five_number(
            bindings.support_counts[bindings.track_valid]
        ),
        "flow_norm_mm_five_number": five_number(valid_flow_mm),
        "parameters": {
            "radius_mm": args.radius_mm,
            "fallback_radius_mm": args.fallback_radius_mm,
            "sigma_mm": args.sigma_mm,
            "minimum_movable_particles": args.minimum_movable_particles,
            "maximum_depth_change_mm": args.maximum_depth_change_mm,
            "maximum_observed_flow_mm": args.maximum_observed_flow_mm,
        },
        "inputs": {
            "tracks": str(tracks_path),
            "tracks_sha256": sha256(tracks_path),
        },
        "outputs": {
            "bindings": str(bindings_path),
            "observations": str(observations_path),
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
