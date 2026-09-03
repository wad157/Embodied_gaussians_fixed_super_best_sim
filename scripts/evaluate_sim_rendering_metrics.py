#!/usr/bin/env python3
"""计算重建渲染与仿真 RGB 真值之间的 PSNR、SSIM 和 LPIPS。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


CAMERAS = ("stereo_left", "stereo_right")
DEFAULT_LPIPS_PACKAGE_ROOT = Path(
    "/Media_HDD/jwshan/conda_envs/Deform3DGS/lib/python3.7/site-packages"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dataset", type=Path, required=True)
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        required=True,
        help="预测根目录；支持 <root>/rgb/<camera> 或 <root>/<camera>。",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lpips-package-root", type=Path, default=DEFAULT_LPIPS_PACKAGE_ROOT)
    parser.add_argument(
        "--split-frame-index",
        type=int,
        default=-1,
        help="可选训练/未来预测分界帧；正式 80/20 协议使用 240。",
    )
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument(
        "--frame-end-exclusive", type=int, default=-1
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--frame-offset", type=int, default=0)
    return parser.parse_args()


def load_lpips(package_root: Path, device: str):
    try:
        import lpips
    except ModuleNotFoundError:
        if not (package_root / "lpips/__init__.py").is_file():
            raise ModuleNotFoundError(
                "未找到 LPIPS。当前机器可用的离线包路径应通过 "
                "--lpips-package-root 指定。"
            )
        sys.path.append(str(package_root))
        import lpips
    model = lpips.LPIPS(net="alex", version="0.1", verbose=False)
    model = model.to(device).eval()
    return model


def resolve_prediction_camera(root: Path, camera: str) -> Path:
    candidates = (root / "rgb" / camera, root / camera)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"预测目录中没有 {camera}：尝试过 {[str(path) for path in candidates]}"
    )


def tissue_union_crop(
    reference_mask: np.ndarray,
    prediction_alpha: np.ndarray,
    *,
    padding: int = 16,
) -> tuple[slice, slice]:
    union = reference_mask | (prediction_alpha > (1.0 / 255.0))
    y, x = np.nonzero(union)
    if not len(x):
        raise ValueError("真值与预测组织 mask 的并集为空")
    height, width = union.shape
    x0 = max(int(x.min()) - padding, 0)
    x1 = min(int(x.max()) + padding + 1, width)
    y0 = max(int(y.min()) - padding, 0)
    y1 = min(int(y.max()) + padding + 1, height)
    return slice(y0, y1), slice(x0, x1)


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def psnr(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None) -> float:
    error = (reference - prediction) ** 2
    if mask is not None:
        if not np.any(mask):
            return float("nan")
        mse = float(error[mask].mean())
    else:
        mse = float(error.mean())
    return float("inf") if mse == 0.0 else -10.0 * math.log10(mse)


def ssim_map(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Wang 等人的 11×11、sigma=1.5 RGB SSIM 图。"""

    c1 = 0.01**2
    c2 = 0.03**2
    mu_ref = cv2.GaussianBlur(reference, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)
    mu_pred = cv2.GaussianBlur(prediction, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)
    mu_ref_sq = mu_ref * mu_ref
    mu_pred_sq = mu_pred * mu_pred
    mu_cross = mu_ref * mu_pred
    sigma_ref = cv2.GaussianBlur(
        reference * reference, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT
    ) - mu_ref_sq
    sigma_pred = cv2.GaussianBlur(
        prediction * prediction, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT
    ) - mu_pred_sq
    sigma_cross = cv2.GaussianBlur(
        reference * prediction, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT
    ) - mu_cross
    channel_map = ((2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)) / (
        (mu_ref_sq + mu_pred_sq + c1) * (sigma_ref + sigma_pred + c2)
    )
    return channel_map.mean(axis=2)


def lpips_tensor(image: np.ndarray, device: str) -> torch.Tensor:
    tensor = (
        torch.from_numpy(np.ascontiguousarray(image))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    return tensor * 2.0 - 1.0


def safe_mean(values: list[float]) -> float | None:
    finite_or_inf = [value for value in values if not math.isnan(value)]
    if not finite_or_inf:
        return None
    return float(np.mean(np.asarray(finite_or_inf, dtype=np.float64)))


def main() -> None:
    args = parse_args()
    reference_root = args.reference_dataset.expanduser().resolve()
    prediction_root = args.prediction_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() or output.with_suffix(".csv").exists():
        raise FileExistsError(f"拒绝覆盖已有渲染评估结果：{output}")
    episode = json.loads((reference_root / "episode.json").read_text(encoding="utf-8"))
    dataset_frame_count = int(episode["frames"])
    if args.frame_stride < 1:
        raise ValueError("frame_stride 必须为正整数")
    if not 0 <= args.frame_offset < args.frame_stride:
        raise ValueError("frame_offset 必须位于 [0, frame_stride)")
    if args.frame_start < 0:
        raise ValueError("frame_start 不能为负")
    if (
        args.frame_end_exclusive >= 0
        and args.frame_end_exclusive <= args.frame_start
    ):
        raise ValueError("frame_end_exclusive 必须大于 frame_start")
    lpips_model = load_lpips(args.lpips_package_root.expanduser().resolve(), args.device)

    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for camera in CAMERAS:
            reference_dir = reference_root / "rgb" / camera
            prediction_dir = resolve_prediction_camera(prediction_root, camera)
            prediction_alpha_dir = prediction_root / "alpha" / camera
            if not prediction_alpha_dir.is_dir():
                raise FileNotFoundError(
                    f"预测缺少独立组织 alpha：{prediction_alpha_dir}"
                )
            all_prediction_paths = sorted(prediction_dir.glob("*.png"))
            if not all_prediction_paths:
                raise ValueError(f"{camera} 没有预测 PNG")
            try:
                all_frame_indices = [
                    int(path.stem) for path in all_prediction_paths
                ]
            except ValueError as error:
                raise ValueError(f"{camera} 预测 PNG 必须使用六位帧号命名") from error
            if (
                len(set(all_frame_indices)) != len(all_frame_indices)
                or min(all_frame_indices) < 0
                or max(all_frame_indices) >= dataset_frame_count
            ):
                raise ValueError(f"{camera} 预测帧号重复或超出数据集范围")
            selected = [
                args.frame_start <= frame_index
                and (
                    args.frame_end_exclusive < 0
                    or frame_index < args.frame_end_exclusive
                )
                and frame_index % args.frame_stride == args.frame_offset
                for frame_index in all_frame_indices
            ]
            prediction_paths = [
                path
                for path, keep in zip(all_prediction_paths, selected)
                if keep
            ]
            frame_indices = [
                frame_index
                for frame_index, keep in zip(all_frame_indices, selected)
                if keep
            ]
            if not prediction_paths:
                raise ValueError(f"{camera} 的帧选择规则没有留下预测 PNG")
            reference_paths = [
                reference_dir / f"{frame_index:06d}.png"
                for frame_index in frame_indices
            ]
            if any(not path.is_file() for path in reference_paths):
                raise ValueError(
                    f"{camera} 存在找不到对应真值的预测帧"
                )
            for local_index, (frame_index, reference_path, prediction_path) in enumerate(
                zip(frame_indices, reference_paths, prediction_paths)
            ):
                if reference_path.name != prediction_path.name:
                    raise ValueError(
                        f"{camera} 文件名未对齐：{reference_path.name} != {prediction_path.name}"
                    )
                reference = read_rgb(reference_path)
                prediction = read_rgb(prediction_path)
                if prediction.shape != reference.shape:
                    raise ValueError(
                        f"{camera}/{reference_path.name} 尺寸不一致："
                        f"{reference.shape} != {prediction.shape}"
                    )
                tissue_mask = np.asarray(
                    Image.open(
                        reference_root
                        / "ground_truth/masks/tissue"
                        / camera
                        / reference_path.name
                    ).convert("L")
                ) > 0
                prediction_alpha = np.asarray(
                    Image.open(
                        prediction_alpha_dir / prediction_path.name
                    ).convert("L"),
                    dtype=np.float32,
                ) / 255.0
                if prediction_alpha.shape != tissue_mask.shape:
                    raise ValueError(
                        f"{camera}/{reference_path.name} alpha 尺寸不一致"
                    )
                structural_map = ssim_map(reference, prediction)
                tissue_float = tissue_mask.astype(np.float32)[..., None]
                reference_tissue_layer = reference * tissue_float
                prediction_tissue_layer = prediction
                crop_y, crop_x = tissue_union_crop(
                    tissue_mask, prediction_alpha
                )
                reference_tissue_crop = reference_tissue_layer[crop_y, crop_x]
                prediction_tissue_crop = prediction_tissue_layer[crop_y, crop_x]
                tissue_layer_structural_map = ssim_map(
                    reference_tissue_crop, prediction_tissue_crop
                )
                reference_tissue_layer_tensor = lpips_tensor(
                    reference_tissue_crop, args.device
                )
                prediction_tissue_layer_tensor = lpips_tensor(
                    prediction_tissue_crop, args.device
                )
                tissue_layer_lpips = float(
                    lpips_model(
                        reference_tissue_layer_tensor,
                        prediction_tissue_layer_tensor,
                    ).item()
                )
                rows.append(
                    {
                        "camera": camera,
                        "frame": frame_index,
                        "segment": (
                            "future_open_loop_20_percent"
                            if args.split_frame_index >= 0
                            and frame_index >= args.split_frame_index
                            else "assimilation_0_79_percent"
                        ),
                        "filename": reference_path.name,
                        "psnr_full_db": psnr(reference, prediction, None),
                        "ssim_full": float(structural_map.mean()),
                        "psnr_tissue_db": psnr(reference, prediction, tissue_mask),
                        "ssim_tissue": float(structural_map[tissue_mask].mean()),
                        "psnr_tissue_layer_db": psnr(
                            reference_tissue_crop,
                            prediction_tissue_crop,
                            None,
                        ),
                        "ssim_tissue_layer": float(
                            tissue_layer_structural_map.mean()
                        ),
                        "lpips_tissue_layer_alex_v0.1": tissue_layer_lpips,
                        "tissue_pixels": int(tissue_mask.sum()),
                        "tissue_union_crop_width": int(
                            crop_x.stop - crop_x.start
                        ),
                        "tissue_union_crop_height": int(
                            crop_y.stop - crop_y.start
                        ),
                    }
                )
                print(
                    f"[渲染评估] {camera} {local_index + 1:04d}/"
                    f"{len(prediction_paths):04d} (frame={frame_index})",
                    flush=True,
                )

    metric_names = (
        "psnr_full_db",
        "ssim_full",
        "psnr_tissue_db",
        "ssim_tissue",
        "psnr_tissue_layer_db",
        "ssim_tissue_layer",
        "lpips_tissue_layer_alex_v0.1",
    )
    def summarize(selected_rows: list[dict[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for camera in (*CAMERAS, "all"):
            selected = (
                selected_rows
                if camera == "all"
                else [row for row in selected_rows if row["camera"] == camera]
            )
            result[camera] = {
                name: safe_mean([float(row[name]) for row in selected])
                for name in metric_names
            }
        return result

    summary = summarize(rows)
    segment_summary: dict[str, object] = {"all": summary}
    if args.split_frame_index >= 0:
        for segment in (
            "assimilation_0_79_percent",
            "future_open_loop_20_percent",
        ):
            selected_rows = [row for row in rows if row["segment"] == segment]
            if not selected_rows:
                raise ValueError("训练/未来分界必须把渲染预测划分成两个非空区间")
            segment_summary[segment] = summarize(selected_rows)
    report = {
        "schema": "fixedsuperbest.render_metrics.v2",
        "reference_dataset": str(reference_root),
        "prediction_dir": str(prediction_root),
        "frames_per_camera": len(rows) // len(CAMERAS),
        "split_frame_index": (
            None if args.split_frame_index < 0 else args.split_frame_index
        ),
        "frame_selection": {
            "start": args.frame_start,
            "end_exclusive": (
                None
                if args.frame_end_exclusive < 0
                else args.frame_end_exclusive
            ),
            "stride": args.frame_stride,
            "offset": args.frame_offset,
            "selected_frame_indices": sorted(
                {int(row["frame"]) for row in rows}
            ),
        },
        "cameras": list(CAMERAS),
        "color_space": "sRGB PNG 数值直接归一化到 [0,1]",
        "psnr": "RGB 通道 MSE，峰值 1.0，单位 dB",
        "ssim": "RGB 通道平均；11×11 Gaussian window，sigma=1.5，K1=0.01，K2=0.03",
        "lpips": "官方 LPIPS 0.1，AlexNet/ImageNet，输入缩放到 [-1,1]",
        "primary_render_region": "tissue_layer_union_crop",
        "tissue_layer": (
            "真值 RGB 乘真值 tissue mask 后置黑，与黑背景组织 Gaussian 预测整图比较；"
            "每帧裁到真值 mask 与预测 alpha 并集外扩 16 px；"
            "预测组织落在真值轮廓外也会被惩罚且不被大面积背景稀释"
        ),
        "summary": summary,
        "segments": segment_summary,
        "per_frame_csv": output.with_suffix(".csv").name,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
