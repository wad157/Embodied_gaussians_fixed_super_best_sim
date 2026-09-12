#!/usr/bin/env python3
"""Fail-closed audit for the current EndoGaussian SIM baseline protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = REPO_ROOT / "baselines" / "endogaussian_sim"
import sys

sys.path.insert(0, str(ADAPTER_ROOT))

from protocol import (  # noqa: E402
    CAMERAS,
    DATASETS,
    HOLDOUT_OFFSET,
    HOLDOUT_STRIDE,
    INTERNAL_UNITS_PER_METER,
    QUERY_FRAME,
    UPSTREAM_COMMIT,
    assert_exact_frame_files,
    dataset_spec,
    future_frames,
    future_start,
    load_json,
    project_world_points,
    reconstruction_holdouts,
    resolve_dataset,
    sha256_file,
    training_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument(
        "--baseline-root", type=Path, default=REPO_ROOT / "baselines" / "EndoGaussian"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def git_output(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments], text=True
    ).strip()


def audit(args: argparse.Namespace) -> dict[str, object]:
    spec = dataset_spec(args.dataset_key)
    dataset = resolve_dataset(REPO_ROOT, args.dataset_key, args.dataset)
    baseline_root = args.baseline_root.expanduser().resolve()
    if not (baseline_root / ".git").exists():
        raise FileNotFoundError("缺少官方 EndoGaussian checkout：{}".format(baseline_root))
    commit = git_output(baseline_root, "rev-parse", "HEAD")
    if commit != UPSTREAM_COMMIT:
        raise ValueError("EndoGaussian commit 错误：{} != {}".format(commit, UPSTREAM_COMMIT))
    tracked_changes = git_output(
        baseline_root,
        "diff",
        "--name-only",
        "HEAD",
        "--",
        "gaussian_renderer",
        "scene",
        "arguments",
        "utils",
        "train.py",
        "submodules/depth-diff-gaussian-rasterization/*.cu",
        "submodules/depth-diff-gaussian-rasterization/*.cpp",
        "submodules/depth-diff-gaussian-rasterization/cuda_rasterizer",
        "submodules/simple-knn/*.cu",
        "submodules/simple-knn/*.cpp",
        "submodules/simple-knn/*.h",
    )
    if tracked_changes:
        raise ValueError("官方 EndoGaussian 算法源码被修改：\n{}".format(tracked_changes))

    episode = load_json(dataset / "episode.json")
    frames = int(spec["frames"])
    if episode.get("name") != spec["name"] or int(episode.get("frames", -1)) != frames:
        raise ValueError("episode.json 与固定数据集定义不一致")
    if int(episode.get("fps", -1)) != 30 or episode.get("resolution") != [1536, 1536]:
        raise ValueError("正式数据必须保持 30 FPS、1536x1536")
    privileged = episode.get("privileged_evaluation_only", [])
    if "ground_truth" not in privileged:
        raise ValueError("episode.json 未把 ground_truth 标为 evaluation-only")

    cameras = load_json(dataset / "cameras.json")
    timestamps = None
    calibration_hashes: dict[str, str] = {}
    intrinsics: dict[str, np.ndarray] = {}
    world_from_cameras: dict[str, np.ndarray] = {}
    for camera in CAMERAS:
        if camera not in cameras or "X_WC_ros_optical" not in cameras[camera]:
            raise ValueError("{} 缺少固定 ROS optical 外参".format(camera))
        metadata_path = dataset / "videos" / "{}.json".format(camera)
        metadata = load_json(metadata_path)
        current_timestamps = np.asarray(metadata["timestamps"], dtype=np.float64)
        if current_timestamps.shape != (frames,):
            raise ValueError("{} 时间戳数量错误".format(camera))
        expected_timestamps = np.arange(frames, dtype=np.float64) / 30.0
        if not np.allclose(current_timestamps, expected_timestamps, atol=1.0e-12):
            raise ValueError("{} 时间戳不是固定 30 Hz 序列".format(camera))
        if timestamps is not None and not np.array_equal(current_timestamps, timestamps):
            raise ValueError("左右相机时间戳不一致")
        timestamps = current_timestamps
        intrinsic = np.asarray(metadata["K"], dtype=np.float64)
        if intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic)):
            raise ValueError("{} 内参非法".format(camera))
        intrinsics[camera] = intrinsic
        world_from_cameras[camera] = np.asarray(
            cameras[camera]["X_WC_ros_optical"], dtype=np.float64
        )
        assert_exact_frame_files(dataset / "rgb" / camera, frames, ".png")
        assert_exact_frame_files(
            dataset / "ground_truth" / "masks" / "tissue" / camera,
            frames,
            ".png",
        )
        calibration_hashes[camera] = sha256_file(metadata_path)

    depth_root = dataset / "estimated_depth" / str(spec["depth"])
    summary_path = depth_root / "depth_generation_summary.json"
    summary = load_json(summary_path)
    if summary.get("outputs", {}).get("depth_units") != "meters":
        raise ValueError("FoundationStereo 深度必须为米")
    if bool(summary.get("inputs", {}).get("uses_ground_truth_depth_for_estimation", True)):
        raise ValueError("深度生成记录显示使用了真值深度")
    for camera in CAMERAS:
        assert_exact_frame_files(
            depth_root / camera, frames, ".npy", stem_suffix="-depth"
        )
        mask_dir = dataset / "gui_assets" / "visual_force_masks" / camera
        report = load_json(mask_dir / "report.json")
        if not bool(report.get("passed")) or int(report.get("frames", -1)) != frames:
            raise ValueError("{} 当前公共组织 mask 资产不完整".format(camera))
        packed = np.load(mask_dir / "tissue_masks_packbits.npy", mmap_mode="r")
        if packed.shape != (frames, 1536, 192) or packed.dtype != np.uint8:
            raise ValueError("{} packed tissue mask 形状或类型错误".format(camera))

    manifest_path = dataset / "evaluation" / "evaluation_points_30_non_grasp.json"
    manifest = load_json(manifest_path)
    evaluation_ids = np.asarray(manifest["tissue_node_ids"], dtype=np.int64)
    if evaluation_ids.shape != (30,) or len(np.unique(evaluation_ids)) != 30:
        raise ValueError("固定评估清单必须恰好包含 30 个不重复节点")
    boundary_path = dataset / "task_inputs" / "known_grasp_region_boundary.npz"
    with np.load(boundary_path, allow_pickle=False) as boundary:
        excluded = np.asarray(
            boundary["evaluation_exclusion_tissue_node_ids"], dtype=np.int64
        )
    overlap = np.intersect1d(evaluation_ids, excluded)
    if len(overlap):
        raise ValueError("30 点清单与控制区重叠：{}".format(overlap.tolist()))

    # This is an evaluation-interface audit only. The training loader never opens
    # ground_truth. Query pixels become available only after the checkpoint freezes.
    with np.load(
        dataset / "ground_truth" / "trajectories_3d.npz", allow_pickle=False
    ) as reference_3d:
        all_ids_3d = np.asarray(reference_3d["tissue_node_ids"], dtype=np.int64)
        lookup_3d = {int(node): index for index, node in enumerate(all_ids_3d)}
        columns_3d = np.asarray([lookup_3d[int(node)] for node in evaluation_ids])
        query_points_world = np.asarray(
            reference_3d["tissue_positions_world"][QUERY_FRAME, columns_3d],
            dtype=np.float64,
        )

    visible_at_query: dict[str, int] = {}
    projection_recheck_max_px: dict[str, float] = {}
    for camera in CAMERAS:
        with np.load(
            dataset / "ground_truth" / "trajectories_2d" / "{}.npz".format(camera),
            allow_pickle=False,
        ) as reference:
            all_ids = np.asarray(reference["tissue_node_ids"], dtype=np.int64)
            lookup = {int(node): index for index, node in enumerate(all_ids)}
            columns = np.asarray([lookup[int(node)] for node in evaluation_ids])
            visible = np.asarray(reference["tissue_visible"])[QUERY_FRAME, columns]
            visible_at_query[camera] = int(visible.sum())
            if not bool(np.all(visible)):
                raise ValueError("{} 的 frame-0 查询点并非全部可见".format(camera))
            expected_uv = np.asarray(
                reference["tissue_uv_pixels"][QUERY_FRAME, columns], dtype=np.float64
            )
        projected_uv, projection_valid, _ = project_world_points(
            query_points_world,
            intrinsics[camera],
            world_from_cameras[camera],
            (1536, 1536),
        )
        error_px = np.linalg.norm(projected_uv.astype(np.float64) - expected_uv, axis=1)
        projection_recheck_max_px[camera] = float(np.max(error_px))
        if not bool(np.all(projection_valid)) or float(np.max(error_px)) > 1.0e-4:
            raise ValueError(
                "{} 的世界坐标到像素投影与固定真值不一致；max={:.6g}px".format(
                    camera, float(np.max(error_px))
                )
            )

    train = training_frames(frames)
    holdout = reconstruction_holdouts(frames)
    future = future_frames(frames)
    if set(train) & set(holdout) or set(train) & set(future) or set(holdout) & set(future):
        raise AssertionError("协议帧集合发生重叠")
    if sorted(train + holdout + future) != list(range(frames)):
        raise AssertionError("协议帧集合没有完整覆盖序列")

    return {
        "schema": "fixedsuperbest.endogaussian_protocol_audit.v1",
        "passed": True,
        "dataset_key": args.dataset_key,
        "dataset": str(dataset),
        "dataset_name": spec["name"],
        "frames": frames,
        "future_start": future_start(frames),
        "split": {
            "definition": "train: t < floor(4T/5) and t % 8 != 7; reconstruction: t % 8 == 7; future: final 20%",
            "holdout_stride": HOLDOUT_STRIDE,
            "holdout_offset": HOLDOUT_OFFSET,
            "training_frame_count": len(train),
            "training_view_count": len(train) * len(CAMERAS),
            "reconstruction_holdout_count": len(holdout),
            "future_frame_count": len(future),
        },
        "inputs": {
            "rgb": "rgb/stereo_{left,right}/%06d.png",
            "depth": str(spec["depth"]),
            "depth_summary_sha256": sha256_file(summary_path),
            "tissue_mask": "gui_assets/visual_force_masks/<camera>/tissue_masks_packbits.npy",
            "camera_calibration_sha256": calibration_hashes,
            "forbidden_during_training": [
                "ground_truth",
                "evaluation/evaluation_points_30_non_grasp.json",
                "future RGB/depth/masks",
            ],
            "fixed_internal_units_per_meter": INTERNAL_UNITS_PER_METER,
        },
        "evaluation_points": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "count": 30,
            "query_frame": QUERY_FRAME,
            "visible_at_query": visible_at_query,
            "projection_recheck_max_px": projection_recheck_max_px,
            "controlled_boundary": str(boundary_path),
            "controlled_boundary_sha256": sha256_file(boundary_path),
        },
        "endogaussian": {
            "root": str(baseline_root),
            "commit": commit,
            "algorithm_source_files_clean": True,
        },
    }


def main() -> None:
    args = parse_args()
    report = audit(args)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError("拒绝覆盖协议审计结果：{}".format(output))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
