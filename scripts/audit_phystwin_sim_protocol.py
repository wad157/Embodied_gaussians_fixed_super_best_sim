#!/usr/bin/env python3
"""Fail-closed audit for the pinned PhysTwin SIM evaluation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = REPO_ROOT / "baselines" / "phystwin_sim"
sys.path.insert(0, str(ADAPTER_ROOT))

from protocol import (  # noqa: E402
    CAMERAS,
    COTRACKER_CHECKPOINT,
    DATASETS,
    HOLDOUT_OFFSET,
    HOLDOUT_STRIDE,
    QUERY_FRAME,
    UPSTREAM_COMMIT,
    dataset_spec,
    future_frames,
    future_start,
    project_world_points,
    reconstruction_holdouts,
    resolve_dataset,
    sha256_file,
    training_frames,
)


EXPECTED_HEADLESS_PATCH_SHA256 = (
    "4ad745d0baf8410672b95817f529bb4d9ac0cfbfe45bc6cb1626d0d08df693b8"
)
EXPECTED_COTRACKER_SHA256 = (
    "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834"
)
DEFAULT_COTRACKER_WEIGHT = Path(
    "/home/jwshan/.cache/torch/hub/checkpoints/scaled_offline.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument(
        "--baseline-root", type=Path, default=REPO_ROOT / "baselines" / "PhysTwin"
    )
    parser.add_argument(
        "--cotracker-weight", type=Path, default=DEFAULT_COTRACKER_WEIGHT
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def git_output(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments], text=True
    ).strip()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_frame_files(root: Path, frames: int, suffix: str, stem_suffix: str = ""):
    paths = sorted(root.glob("*{}".format(suffix)))
    expected = ["{:06d}{}{}".format(i, stem_suffix, suffix) for i in range(frames)]
    actual = [path.name for path in paths]
    if actual != expected:
        raise ValueError("帧文件不完整或含额外文件：{}".format(root))


def audit(args: argparse.Namespace) -> dict[str, object]:
    spec = dataset_spec(args.dataset_key)
    frames = int(spec["frames"])
    dataset = resolve_dataset(REPO_ROOT, args.dataset_key, args.dataset)
    baseline_root = args.baseline_root.expanduser().resolve()
    if not (baseline_root / ".git").is_dir():
        raise FileNotFoundError("缺少官方 PhysTwin checkout：{}".format(baseline_root))
    commit = git_output(baseline_root, "rev-parse", "HEAD")
    if commit != UPSTREAM_COMMIT:
        raise ValueError("PhysTwin commit 错误：{} != {}".format(commit, UPSTREAM_COMMIT))
    diff = git_output(
        baseline_root, "diff", "--", "qqtt/__init__.py", "qqtt/utils/__init__.py"
    )
    if text_sha256(diff + ("\n" if diff else "")) != EXPECTED_HEADLESS_PATCH_SHA256:
        raise ValueError("PhysTwin headless import 补丁不匹配")
    other_changes = git_output(
        baseline_root,
        "diff",
        "--name-only",
        "--",
        ".",
        ":(exclude)qqtt/__init__.py",
        ":(exclude)qqtt/utils/__init__.py",
    )
    if other_changes:
        raise ValueError("PhysTwin 算法源码被修改：\n{}".format(other_changes))

    episode = load_json(dataset / "episode.json")
    if episode.get("name") != spec["name"] or int(episode.get("frames", -1)) != frames:
        raise ValueError("episode.json 与固定数据集定义不一致")
    if int(episode.get("fps", -1)) != 30 or episode.get("resolution") != [1536, 1536]:
        raise ValueError("正式数据必须保持 30 FPS、1536x1536")
    if "ground_truth" not in episode.get("privileged_evaluation_only", []):
        raise ValueError("ground_truth 未标为 evaluation-only")

    cameras_json = load_json(dataset / "cameras.json")
    calibration_hashes = {}
    intrinsics = {}
    world_from_cameras = {}
    for camera in CAMERAS:
        metadata_path = dataset / "videos" / "{}.json".format(camera)
        metadata = load_json(metadata_path)
        timestamps = np.asarray(metadata["timestamps"], dtype=np.float64)
        if not np.allclose(timestamps, np.arange(frames) / 30.0, atol=1.0e-12):
            raise ValueError("{} 时间戳不符合固定 30 Hz".format(camera))
        intrinsics[camera] = np.asarray(metadata["K"], dtype=np.float64)
        world_from_cameras[camera] = np.asarray(
            cameras_json[camera]["X_WC_ros_optical"], dtype=np.float64
        )
        assert_frame_files(dataset / "rgb" / camera, frames, ".png")
        assert_frame_files(
            dataset / "ground_truth" / "masks" / "tissue" / camera,
            frames,
            ".png",
        )
        calibration_hashes[camera] = sha256_file(metadata_path)

    depth_root = dataset / "estimated_depth" / str(spec["depth"])
    depth_summary_path = depth_root / "depth_generation_summary.json"
    depth_summary = load_json(depth_summary_path)
    if depth_summary.get("outputs", {}).get("depth_units") != "meters":
        raise ValueError("FoundationStereo 深度单位必须是米")
    if bool(depth_summary.get("inputs", {}).get("uses_ground_truth_depth_for_estimation", True)):
        raise ValueError("深度生成记录显示使用了真值深度")
    for camera in CAMERAS:
        assert_frame_files(depth_root / camera, frames, ".npy", "-depth")
        mask_root = dataset / "gui_assets" / "visual_force_masks" / camera
        mask_report = load_json(mask_root / "report.json")
        packed = np.load(mask_root / "tissue_masks_packbits.npy", mmap_mode="r")
        if not bool(mask_report.get("passed")) or packed.shape != (frames, 1536, 192):
            raise ValueError("{} 公共组织 mask 不完整".format(camera))

    tracker_weight = args.cotracker_weight.expanduser().resolve()
    if not tracker_weight.is_file() or sha256_file(tracker_weight) != EXPECTED_COTRACKER_SHA256:
        raise ValueError("CoTracker3 checkpoint 缺失或哈希不匹配")
    pose_path = dataset / "task_inputs" / "psm_link_poses.npz"
    mesh_path = dataset / "gui_assets" / "official_psm_tip_meshes_v2.npz"
    if not pose_path.is_file() or not mesh_path.is_file():
        raise FileNotFoundError("缺少协议允许的 PSM 已知控制轨迹或官方工具网格")

    manifest_path = dataset / "evaluation" / "evaluation_points_30_non_grasp.json"
    manifest = load_json(manifest_path)
    evaluation_ids = np.asarray(manifest["tissue_node_ids"], dtype=np.int64)
    if evaluation_ids.shape != (30,) or len(np.unique(evaluation_ids)) != 30:
        raise ValueError("固定评估清单必须含 30 个不重复节点")
    boundary_path = dataset / "task_inputs" / "known_grasp_region_boundary.npz"
    with np.load(boundary_path, allow_pickle=False) as boundary:
        excluded = np.asarray(
            boundary["evaluation_exclusion_tissue_node_ids"], dtype=np.int64
        )
    if len(np.intersect1d(evaluation_ids, excluded)):
        raise ValueError("固定评估点与控制区重叠")

    with np.load(dataset / "ground_truth" / "trajectories_3d.npz") as reference:
        lookup = {int(node): i for i, node in enumerate(reference["tissue_node_ids"])}
        columns = np.asarray([lookup[int(node)] for node in evaluation_ids])
        query_world = np.asarray(
            reference["tissue_positions_world"][QUERY_FRAME, columns], dtype=np.float64
        )
    projection_errors = {}
    for camera in CAMERAS:
        with np.load(dataset / "ground_truth" / "trajectories_2d" / f"{camera}.npz") as ref:
            lookup = {int(node): i for i, node in enumerate(ref["tissue_node_ids"])}
            columns = np.asarray([lookup[int(node)] for node in evaluation_ids])
            if not bool(np.all(ref["tissue_visible"][QUERY_FRAME, columns])):
                raise ValueError("{} 查询帧不能看到全部固定点".format(camera))
            expected_uv = np.asarray(
                ref["tissue_uv_pixels"][QUERY_FRAME, columns], dtype=np.float64
            )
        projected, valid, _ = project_world_points(
            query_world,
            intrinsics[camera],
            world_from_cameras[camera],
            (1536, 1536),
        )
        error = np.linalg.norm(projected.astype(np.float64) - expected_uv, axis=1)
        projection_errors[camera] = float(error.max())
        if not bool(np.all(valid)) or float(error.max()) > 1.0e-4:
            raise ValueError("{} 固定投影复核失败".format(camera))

    train = training_frames(frames)
    holdout = reconstruction_holdouts(frames)
    future = future_frames(frames)
    if sorted(train + holdout + future) != list(range(frames)):
        raise AssertionError("协议帧集合没有无重叠地覆盖完整序列")
    return {
        "schema": "fixedsuperbest.phystwin_protocol_audit.v1",
        "passed": True,
        "dataset_key": args.dataset_key,
        "dataset": str(dataset),
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
            "estimated_depth": str(spec["depth"]),
            "depth_summary_sha256": sha256_file(depth_summary_path),
            "tissue_mask": "gui_assets/visual_force_masks/<camera>/tissue_masks_packbits.npy",
            "camera_calibration_sha256": calibration_hashes,
            "known_control": {
                "psm_link_poses_sha256": sha256_file(pose_path),
                "official_psm_mesh_sha256": sha256_file(mesh_path),
            },
            "forbidden_during_training": [
                "ground_truth",
                "evaluation/evaluation_points_30_non_grasp.json",
                "holdout RGB/depth/masks",
                "future RGB/depth/masks",
            ],
        },
        "evaluation_points": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "count": 30,
            "query_frame": QUERY_FRAME,
            "projection_recheck_max_px": projection_errors,
            "controlled_boundary": str(boundary_path),
            "controlled_boundary_sha256": sha256_file(boundary_path),
        },
        "phystwin": {
            "root": str(baseline_root),
            "commit": commit,
            "headless_import_patch_sha256": EXPECTED_HEADLESS_PATCH_SHA256,
            "algorithm_source_files_clean": True,
            "physics_core_modified": False,
            "trajectory_source": "native persistent spring-mass particles plus upstream Gaussian LBS",
            "shape_of_motion_used": False,
        },
        "tracker": {
            "name": COTRACKER_CHECKPOINT,
            "checkpoint_sha256": EXPECTED_COTRACKER_SHA256,
        },
    }


def main() -> None:
    args = parse_args()
    report = audit(args)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError("拒绝覆盖协议审计：{}".format(output))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
