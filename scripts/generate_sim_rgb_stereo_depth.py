#!/usr/bin/env python3
"""使用与 SUPER 相同的 FoundationStereo 主模型生成 SIM RGB 双目深度。

SIM 的双目相机带会聚角，输入首先依据数据集内参和外参严格校正。不同版本
数据集的左右相机排列并不一致：程序根据 stereoRectify 的 P2[0,3] 自动判断
校正后 x_R-x_L 的符号，只在需要时同时水平翻转左右校正图，使
FoundationStereo 始终工作在训练时的正视差约定下。输出会重新采样到原始
左右 RGB 像素坐标，供 CoTracker 轨迹直接查询。

仿真 GT 深度只用于生成后的独立误差审计，不参与推理、尺度或后处理。
现有 ground_truth 目录永远不会被修改。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm"
FOUNDATION_CHECKPOINT = (
    ROOT
    / "third_party/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth"
)
sys.path.insert(0, str(ROOT / "scripts"))

from generate_super_depth_foundation_timestamped import (  # noqa: E402
    image_tensor,
    infer_foundation,
    load_foundation_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "默认写入 <dataset>/estimated_depth/"
            "foundation_stereo_rgb_v1；不会覆盖 GT 深度"
        ),
    )
    parser.add_argument(
        "--frames",
        default="all",
        help="all、逗号列表（0,7,15）或闭区间（0:359）",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument(
        "--hierarchical", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--lr-threshold-px", type=float, default=1.5)
    parser.add_argument("--min-depth-mm", type=float, default=35.0)
    parser.add_argument("--max-depth-mm", type=float, default=250.0)
    parser.add_argument("--preview-stride", type=int, default=30)
    parser.add_argument(
        "--rectification-alpha",
        type=float,
        default=0.0,
        help=(
            "OpenCV stereoRectify视野参数；0裁剪到公共区域，1保留完整视野。"
            "这是相机几何预处理，不使用深度或轨迹真值。"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="跳过已有且包含深度、置信度和逐帧报告的完整帧",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="显式允许覆盖已有对应帧"
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_frames(specification: str, count: int) -> list[int]:
    if specification.strip().lower() == "all":
        return list(range(count))
    frames: list[int] = []
    for item in specification.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            start_text, end_text = item.split(":", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"非法帧区间：{item}")
            frames.extend(range(start, end + 1))
        else:
            frames.append(int(item))
    frames = sorted(set(frames))
    if not frames or frames[0] < 0 or frames[-1] >= count:
        raise ValueError(f"帧范围必须位于 [0, {count - 1}]")
    return frames


def camera_geometry(
    dataset: Path, *, rectification_alpha: float = 0.0
) -> dict[str, np.ndarray | float | tuple[int, int]]:
    cameras = read_json(dataset / "cameras.json")
    left_metadata = read_json(dataset / "videos/stereo_left.json")
    right_metadata = read_json(dataset / "videos/stereo_right.json")
    K_left = np.asarray(left_metadata["K"], dtype=np.float64)
    K_right = np.asarray(right_metadata["K"], dtype=np.float64)
    resolution_left = tuple(int(value) for value in left_metadata["resolution"])
    resolution_right = tuple(int(value) for value in right_metadata["resolution"])
    if resolution_left != resolution_right:
        raise ValueError("左右相机分辨率不一致")
    width, height = resolution_left
    X_WL = np.asarray(
        cameras["stereo_left"]["X_WC_ros_optical"], dtype=np.float64
    )
    X_WR = np.asarray(
        cameras["stereo_right"]["X_WC_ros_optical"], dtype=np.float64
    )
    rotation_left_to_right = X_WR[:3, :3].T @ X_WL[:3, :3]
    translation_left_to_right = (
        X_WR[:3, :3].T @ (X_WL[:3, 3] - X_WR[:3, 3])
    ).reshape(3, 1)
    distortion = np.zeros((5, 1), dtype=np.float64)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_left,
        distortion,
        K_right,
        distortion,
        (width, height),
        rotation_left_to_right,
        translation_left_to_right,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=float(rectification_alpha),
    )
    baseline = float(np.linalg.norm(translation_left_to_right))
    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        K_left, distortion, R1, P1[:, :3], (width, height), cv2.CV_32FC1
    )
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        K_right, distortion, R2, P2[:, :3], (width, height), cv2.CV_32FC1
    )
    return {
        "K_left": K_left,
        "K_right": K_right,
        "X_WL": X_WL,
        "X_WR": X_WR,
        "rotation_left_to_right": rotation_left_to_right,
        "translation_left_to_right": translation_left_to_right,
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "roi1": np.asarray(roi1, dtype=np.int32),
        "roi2": np.asarray(roi2, dtype=np.int32),
        "baseline_m": baseline,
        "resolution": (width, height),
        "map_left_x": map_left_x,
        "map_left_y": map_left_y,
        "map_right_x": map_right_x,
        "map_right_y": map_right_y,
    }


def original_to_rectified_maps(
    intrinsic: np.ndarray,
    rectification: np.ndarray,
    projection: np.ndarray,
    resolution: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width, height = resolution
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    points = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 1, 2)
    rectified = cv2.undistortPoints(
        points,
        intrinsic,
        np.zeros((5, 1), dtype=np.float64),
        R=rectification,
        P=projection,
    ).reshape(height, width, 2)
    normalized_x = (grid_x - intrinsic[0, 2]) / intrinsic[0, 0]
    normalized_y = (grid_y - intrinsic[1, 2]) / intrinsic[1, 1]
    rectified_ray_z = (
        rectification[2, 0] * normalized_x
        + rectification[2, 1] * normalized_y
        + rectification[2, 2]
    ).astype(np.float32)
    return (
        rectified[..., 0].astype(np.float32),
        rectified[..., 1].astype(np.float32),
        rectified_ray_z,
    )


def sample_original_depth(
    rectified_depth: np.ndarray,
    original_to_rectified_x: np.ndarray,
    original_to_rectified_y: np.ndarray,
    rectified_ray_z: np.ndarray,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    sampled = cv2.remap(
        rectified_depth,
        original_to_rectified_x,
        original_to_rectified_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )
    depth = sampled / rectified_ray_z
    valid = (
        np.isfinite(depth)
        & np.isfinite(rectified_ray_z)
        & (rectified_ray_z > 1.0e-6)
        & (depth >= minimum)
        & (depth <= maximum)
    )
    output = np.full(depth.shape, np.nan, dtype=np.float32)
    output[valid] = depth[valid].astype(np.float32)
    return output


def colorize_depth(depth: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    if valid is None:
        valid = np.isfinite(depth)
    selected = depth[valid & np.isfinite(depth)]
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    if len(selected):
        lo, hi = np.percentile(selected, [2, 98])
        finite = np.where(np.isfinite(depth), depth, lo)
        scaled = np.clip(
            (finite - lo) / max(float(hi - lo), 1.0e-9) * 255.0, 0, 255
        ).astype(np.uint8)
        scaled[~valid] = 0
    return cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)


def preview(
    rgb: np.ndarray,
    estimated: np.ndarray,
    lr_valid: np.ndarray,
    gt: np.ndarray | None,
) -> np.ndarray:
    size = (768, 768)
    panels = [
        cv2.resize(rgb, size, interpolation=cv2.INTER_AREA),
        cv2.resize(colorize_depth(estimated), size, interpolation=cv2.INTER_AREA),
        cv2.resize(
            colorize_depth(estimated, lr_valid), size, interpolation=cv2.INTER_AREA
        ),
    ]
    labels = ["original RGB", "FoundationStereo depth", "LR-consistent audit"]
    if gt is not None:
        error = np.abs(estimated - gt)
        panels.append(
            cv2.resize(
                colorize_depth(error, np.isfinite(error)),
                size,
                interpolation=cv2.INTER_AREA,
            )
        )
        labels.append("GT audit |estimated-GT|")
    for panel, label in zip(panels, labels, strict=True):
        cv2.putText(
            panel,
            label,
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return np.concatenate(panels, axis=1)


def five_number(values: np.ndarray, scale: float = 1.0) -> list[float] | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)] * scale
    if not len(finite):
        return None
    return np.percentile(finite, [0, 5, 50, 95, 100]).tolist()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else dataset / "estimated_depth/foundation_stereo_rgb_v1"
    )
    episode = read_json(dataset / "episode.json")
    frame_count = int(episode["frames"])
    frames = parse_frames(args.frames, frame_count)
    if args.iterations < 1:
        raise ValueError("--iterations 必须为正数")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了CUDA，但当前进程无法访问GPU")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if not 0.0 <= args.rectification_alpha <= 1.0:
        raise ValueError("--rectification-alpha必须位于[0,1]")
    geometry = camera_geometry(
        dataset, rectification_alpha=args.rectification_alpha
    )
    width, height = geometry["resolution"]
    assert isinstance(width, int) and isinstance(height, int)
    left_original_map = original_to_rectified_maps(
        geometry["K_left"], geometry["R1"], geometry["P1"], (width, height)
    )
    right_original_map = original_to_rectified_maps(
        geometry["K_right"], geometry["R2"], geometry["P2"], (width, height)
    )
    minimum = args.min_depth_mm * 1.0e-3
    maximum = args.max_depth_mm * 1.0e-3
    fx_rectified = float(geometry["P1"][0, 0])
    baseline = float(geometry["baseline_m"])
    rectified_translation_px_m = float(geometry["P2"][0, 3])
    if abs(rectified_translation_px_m) <= 1.0e-12:
        raise ValueError("校正双目P2[0,3]为零，无法确定视差方向")
    # 对同一三维点，P2的平移项决定 x_R-x_L 的符号。这里只读取相机
    # 标定，不使用GT深度、组织轨迹或任何评估信息。
    right_minus_left_sign = float(np.sign(rectified_translation_px_m))

    for name in (
        "stereo_left",
        "stereo_right",
        "confidence",
        "frame_reports",
        "previews",
    ):
        (output / name).mkdir(parents=True, exist_ok=True)
    pending_frames: list[int] = []
    reused_reports: list[dict] = []
    for frame in frames:
        targets = (
            output / "stereo_left" / f"{frame:06d}-depth.npy",
            output / "stereo_right" / f"{frame:06d}-depth.npy",
            output / "confidence" / f"{frame:06d}.npz",
            output / "frame_reports" / f"{frame:06d}.json",
        )
        exists = [path.exists() for path in targets]
        if args.overwrite or not any(exists):
            pending_frames.append(frame)
        elif args.resume and all(exists):
            reused_reports.append(read_json(targets[-1]))
        else:
            raise FileExistsError(
                f"第{frame}帧输出不完整或禁止续跑；拒绝静默覆盖：\n- "
                + "\n- ".join(str(path) for path in targets if path.exists())
            )

    if not pending_frames:
        existing_summary = output / "depth_generation_summary.json"
        if existing_summary.is_file():
            print(f"all requested frames already complete: {existing_summary}")
            return
        raise RuntimeError("全部帧已存在，但缺少总报告")
    print(
        f"loading FoundationStereo on {device}; "
        f"pending={len(pending_frames)}, reused={len(reused_reports)}",
        flush=True,
    )
    model, model_metadata = load_foundation_model(FOUNDATION_CHECKPOINT, device)
    reports: list[dict] = list(reused_reports)
    for order, frame in enumerate(pending_frames, start=1):
        started = time.perf_counter()
        left_path = dataset / "rgb/stereo_left" / f"{frame:06d}.png"
        right_path = dataset / "rgb/stereo_right" / f"{frame:06d}.png"
        left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left is None or right is None:
            raise FileNotFoundError(f"缺少第{frame}帧双目RGB")
        if left.shape[:2] != (height, width) or right.shape[:2] != (height, width):
            raise ValueError(f"第{frame}帧RGB分辨率错误")

        left_rectified = cv2.remap(
            left,
            geometry["map_left_x"],
            geometry["map_left_y"],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        right_rectified = cv2.remap(
            right,
            geometry["map_right_x"],
            geometry["map_right_y"],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        if right_minus_left_sign > 0.0:
            # x_R-x_L>0: mirror the left-reference pair so that the model sees
            # x'_L-x'_R>0. The right-reference pair is already positive.
            left_disparity_rectified = np.flip(
                infer_foundation(
                    model,
                    image_tensor(np.flip(left_rectified, axis=1).copy(), device),
                    image_tensor(np.flip(right_rectified, axis=1).copy(), device),
                    args.iterations,
                    args.hierarchical,
                    device,
                ),
                axis=1,
            ).copy()
            right_disparity_rectified = infer_foundation(
                model,
                image_tensor(right_rectified, device),
                image_tensor(left_rectified, device),
                args.iterations,
                args.hierarchical,
                device,
            )
        else:
            # x_R-x_L<0: the left-reference pair already has x_L-x_R>0;
            # mirror only the swapped right-reference pair.
            left_disparity_rectified = infer_foundation(
                model,
                image_tensor(left_rectified, device),
                image_tensor(right_rectified, device),
                args.iterations,
                args.hierarchical,
                device,
            )
            right_disparity_rectified = np.flip(
                infer_foundation(
                    model,
                    image_tensor(np.flip(right_rectified, axis=1).copy(), device),
                    image_tensor(np.flip(left_rectified, axis=1).copy(), device),
                    args.iterations,
                    args.hierarchical,
                    device,
                ),
                axis=1,
            ).copy()

        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        corresponding_right_x = (
            grid_x + right_minus_left_sign * left_disparity_rectified
        )
        sampled_right_disparity = cv2.remap(
            right_disparity_rectified,
            corresponding_right_x,
            grid_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        )
        lr_error_rectified = np.abs(
            left_disparity_rectified - sampled_right_disparity
        ).astype(np.float32)
        left_lr_valid_rectified = (
            np.isfinite(left_disparity_rectified)
            & np.isfinite(sampled_right_disparity)
            & (left_disparity_rectified > 0.0)
            & (corresponding_right_x >= 0.0)
            & (corresponding_right_x < width - 1)
            & (lr_error_rectified <= args.lr_threshold_px)
        )

        corresponding_left_x = (
            grid_x - right_minus_left_sign * right_disparity_rectified
        )
        sampled_left_disparity = cv2.remap(
            left_disparity_rectified,
            corresponding_left_x,
            grid_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        )
        right_lr_error_rectified = np.abs(
            right_disparity_rectified - sampled_left_disparity
        ).astype(np.float32)
        right_lr_valid_rectified = (
            np.isfinite(right_disparity_rectified)
            & np.isfinite(sampled_left_disparity)
            & (right_disparity_rectified > 0.0)
            & (corresponding_left_x >= 0.0)
            & (corresponding_left_x < width - 1)
            & (right_lr_error_rectified <= args.lr_threshold_px)
        )

        left_depth_rectified = np.full((height, width), np.nan, dtype=np.float32)
        right_depth_rectified = np.full((height, width), np.nan, dtype=np.float32)
        left_positive = np.isfinite(left_disparity_rectified) & (
            left_disparity_rectified > 0.0
        )
        right_positive = np.isfinite(right_disparity_rectified) & (
            right_disparity_rectified > 0.0
        )
        left_depth_rectified[left_positive] = (
            fx_rectified * baseline / left_disparity_rectified[left_positive]
        )
        right_depth_rectified[right_positive] = (
            fx_rectified * baseline / right_disparity_rectified[right_positive]
        )
        left_depth = sample_original_depth(
            left_depth_rectified, *left_original_map, minimum, maximum
        )
        right_depth = sample_original_depth(
            right_depth_rectified, *right_original_map, minimum, maximum
        )
        left_lr_valid = cv2.remap(
            left_lr_valid_rectified.astype(np.uint8),
            left_original_map[0],
            left_original_map[1],
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        ).astype(bool)
        right_lr_valid = cv2.remap(
            right_lr_valid_rectified.astype(np.uint8),
            right_original_map[0],
            right_original_map[1],
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        ).astype(bool)
        left_lr_valid &= np.isfinite(left_depth)
        right_lr_valid &= np.isfinite(right_depth)

        np.save(output / "stereo_left" / f"{frame:06d}-depth.npy", left_depth)
        np.save(output / "stereo_right" / f"{frame:06d}-depth.npy", right_depth)
        np.savez_compressed(
            output / "confidence" / f"{frame:06d}.npz",
            schema=np.asarray("fixedsuperbest.sim_foundation_depth_confidence.v1"),
            left_lr_valid_original=left_lr_valid,
            right_lr_valid_original=right_lr_valid,
            left_valid_original=np.isfinite(left_depth),
            right_valid_original=np.isfinite(right_depth),
            lr_threshold_px=np.asarray(args.lr_threshold_px, dtype=np.float32),
        )

        gt_left_path = dataset / "ground_truth/depth/stereo_left" / f"{frame:06d}.npy"
        gt_left = (
            np.load(gt_left_path, allow_pickle=False)
            if gt_left_path.is_file()
            else None
        )
        audit_valid = (
            np.isfinite(left_depth) & np.isfinite(gt_left)
            if gt_left is not None
            else np.zeros(left_depth.shape, dtype=bool)
        )
        absolute_error = (
            np.abs(left_depth - gt_left)
            if gt_left is not None
            else np.full(left_depth.shape, np.nan, dtype=np.float32)
        )
        report = {
            "frame": frame,
            "left_valid_fraction": float(np.mean(np.isfinite(left_depth))),
            "right_valid_fraction": float(np.mean(np.isfinite(right_depth))),
            "left_lr_consistent_fraction": float(np.mean(left_lr_valid)),
            "right_lr_consistent_fraction": float(np.mean(right_lr_valid)),
            "left_depth_m_min_p05_p50_p95_max": five_number(left_depth),
            "right_depth_m_min_p05_p50_p95_max": five_number(right_depth),
            "privileged_gt_audit_only": {
                "valid_fraction": float(np.mean(audit_valid)),
                "absolute_error_mm_min_p05_p50_p95_max": five_number(
                    absolute_error[audit_valid], 1000.0
                ),
                "mean_absolute_error_mm": (
                    float(np.mean(absolute_error[audit_valid]) * 1000.0)
                    if np.any(audit_valid)
                    else None
                ),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        reports.append(report)
        (output / "frame_reports" / f"{frame:06d}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if args.preview_stride > 0 and (
            frame == frames[0]
            or frame == frames[-1]
            or frame % args.preview_stride == 0
        ):
            cv2.imwrite(
                str(output / "previews" / f"{frame:06d}-left.png"),
                preview(left, left_depth, left_lr_valid, gt_left),
            )
        print(
            f"[{order:03d}/{len(pending_frames):03d}] frame={frame:03d} "
            f"valid={report['left_valid_fraction']:.2%} "
            f"LR={report['left_lr_consistent_fraction']:.2%} "
            f"MAE={report['privileged_gt_audit_only']['mean_absolute_error_mm']:.3f}mm "
            f"time={report['elapsed_seconds']:.2f}s",
            flush=True,
        )

    reports.sort(key=lambda row: int(row["frame"]))
    all_audit_means = np.asarray(
        [row["privileged_gt_audit_only"]["mean_absolute_error_mm"] for row in reports],
        dtype=np.float64,
    )
    summary = {
        "schema": "fixedsuperbest.sim_rgb_stereo_depth.v2",
        "dataset": str(dataset),
        "purpose": (
            "RGB-only stereo depth for replacing privileged simulator GT depth "
            "in trajectory observation"
        ),
        "estimator": "FoundationStereo primary model, matching current SUPER v4",
        "model": model_metadata,
        "runtime": {
            "python": sys.executable,
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "device": str(device),
        },
        "inputs": {
            "left_rgb_pattern": "rgb/stereo_left/NNNNNN.png",
            "right_rgb_pattern": "rgb/stereo_right/NNNNNN.png",
            "camera_geometry": "cameras.json + videos/stereo_{left,right}.json",
            "uses_ground_truth_depth_for_estimation": False,
            "ground_truth_usage": "post-generation numerical audit only",
            "rectified_right_minus_left_sign": right_minus_left_sign,
            "disparity_sign_source": "stereoRectify P2[0,3] from camera calibration",
            "rectification_alpha": float(args.rectification_alpha),
        },
        "rectification": {
            "required": True,
            "reason": "synthetic stereo cameras converge on a common target",
            "left_network_input_mirrored_for_positive_disparity": True,
            "baseline_m": baseline,
            "fx_rectified_px": fx_rectified,
            "R_left_to_right": geometry["rotation_left_to_right"].tolist(),
            "T_left_to_right_m": geometry["translation_left_to_right"].reshape(3).tolist(),
            "R1": geometry["R1"].tolist(),
            "R2": geometry["R2"].tolist(),
            "P1": geometry["P1"].tolist(),
            "P2": geometry["P2"].tolist(),
        },
        "parameters": {
            "frames": frames,
            "iterations": args.iterations,
            "hierarchical": args.hierarchical,
            "lr_threshold_px": args.lr_threshold_px,
            "depth_range_mm": [args.min_depth_mm, args.max_depth_mm],
            "seed": args.seed,
        },
        "outputs": {
            "left_depth_pattern": "stereo_left/NNNNNN-depth.npy",
            "right_depth_pattern": "stereo_right/NNNNNN-depth.npy",
            "confidence_pattern": "confidence/NNNNNN.npz",
            "depth_units": "meters",
            "pixel_coordinates": "original unrectified RGB coordinates",
            "invalid_value": "NaN",
        },
        "aggregate": {
            "generated_frames": len(reports),
            "left_valid_fraction_mean": float(
                np.mean([row["left_valid_fraction"] for row in reports])
            ),
            "left_lr_consistent_fraction_mean": float(
                np.mean([row["left_lr_consistent_fraction"] for row in reports])
            ),
            "privileged_gt_audit_frame_mean_absolute_error_mm": {
                "mean": float(np.nanmean(all_audit_means)),
                "min_p05_p50_p95_max": five_number(all_audit_means),
            },
        },
        "frames": reports,
    }
    summary_path = output / "depth_generation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    print(f"done: {summary_path}")


if __name__ == "__main__":
    main()
