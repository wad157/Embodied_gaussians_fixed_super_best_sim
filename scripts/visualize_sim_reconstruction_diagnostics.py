#!/usr/bin/env python3
"""生成旧前80%同化区间诊断；它不是正式 EH-SurGS 7:1 重建评估。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


METHODS = (
    ("pbd", "PBD"),
    ("pbd_visual_residual", "PBD + visual residual"),
    (
        "pbd_visual_residual_stiffness",
        "PBD + visual residual + stiffness",
    ),
)
COLORS = ("#6B7280", "#2563EB", "#DC2626")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="默认写入 <evaluation-root>/reconstruction_diagnostics。",
    )
    parser.add_argument("--camera", default="stereo_left")
    parser.add_argument(
        "--selected-frames",
        default="",
        help="逗号分隔；默认在前 80% 内等距选择 4 帧。",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def union_crop(
    reference_mask: np.ndarray,
    alpha_masks: list[np.ndarray],
    padding: int = 16,
) -> tuple[slice, slice]:
    union = reference_mask.copy()
    for alpha in alpha_masks:
        union |= alpha > (1.0 / 255.0)
    y, x = np.nonzero(union)
    if not len(x):
        raise ValueError("重建诊断的组织并集为空")
    height, width = union.shape
    return (
        slice(max(int(y.min()) - padding, 0), min(int(y.max()) + padding + 1, height)),
        slice(max(int(x.min()) - padding, 0), min(int(x.max()) + padding + 1, width)),
    )


def load_per_frame(
    method_dir: Path,
    split_frame: int,
) -> dict[str, np.ndarray]:
    trajectory_rows = read_csv(method_dir / "trajectory_metrics.csv")
    render_rows = read_csv(method_dir / "render_metrics.csv")
    trajectory_by_frame = {
        int(row["frame"]): row
        for row in trajectory_rows
        if int(row["frame"]) < split_frame
    }
    render_by_frame: dict[int, list[dict[str, str]]] = {}
    for row in render_rows:
        frame = int(row["frame"])
        if frame < split_frame:
            render_by_frame.setdefault(frame, []).append(row)
    frames = np.asarray(sorted(trajectory_by_frame), dtype=np.int32)
    if not len(frames) or set(frames) != set(render_by_frame):
        raise ValueError(f"{method_dir.name} 的轨迹与渲染帧没有完整对齐")
    trajectory = [trajectory_by_frame[int(frame)] for frame in frames]
    rendering = [render_by_frame[int(frame)] for frame in frames]
    return {
        "frame": frames,
        "track_3d_mean_mm": np.asarray(
            [float(row["track_3d_mean_mm"]) for row in trajectory]
        ),
        "track_2d_mean_px": np.asarray(
            [
                0.5
                * (
                    float(row["stereo_left_track_2d_mean_px"])
                    + float(row["stereo_right_track_2d_mean_px"])
                )
                for row in trajectory
            ]
        ),
        "psnr_db": np.asarray(
            [
                np.mean([float(row["psnr_tissue_layer_db"]) for row in rows])
                for rows in rendering
            ]
        ),
        "ssim": np.asarray(
            [
                np.mean([float(row["ssim_tissue_layer"]) for row in rows])
                for rows in rendering
            ]
        ),
        "lpips": np.asarray(
            [
                np.mean(
                    [float(row["lpips_tissue_layer_alex_v0.1"]) for row in rows]
                )
                for rows in rendering
            ]
        ),
    }


def plot_curves(
    series: dict[str, dict[str, np.ndarray]],
    split_frame: int,
    output: Path,
) -> None:
    fields = (
        ("track_3d_mean_mm", "3D tracking mean (mm)", False),
        ("track_2d_mean_px", "Stereo 2D tracking mean (px)", False),
        ("psnr_db", "Tissue-layer PSNR (dB)", True),
        ("ssim", "Tissue-layer SSIM", True),
        ("lpips", "Tissue-layer LPIPS", False),
    )
    figure, axes = plt.subplots(3, 2, figsize=(13.5, 11), sharex=True)
    flat_axes = axes.ravel()
    for axis, (field, label, higher_is_better) in zip(flat_axes, fields):
        for (key, display), color in zip(METHODS, COLORS):
            values = series[key][field]
            axis.plot(series[key]["frame"], values, color=color, label=display, linewidth=1.5)
        axis.set_ylabel(label)
        axis.grid(alpha=0.22)
        direction = "higher is better" if higher_is_better else "lower is better"
        axis.set_title(direction, fontsize=9, color="#4B5563")
    flat_axes[-1].axis("off")
    for axis in flat_axes[:-1]:
        axis.axvline(split_frame - 1, color="#111827", linestyle="--", linewidth=1.0)
        axis.set_xlim(0, split_frame - 1)
        axis.set_xlabel("Frame (front 80% assimilation interval)")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.96, 0.07))
    figure.suptitle("Legacy assimilation diagnostics (not 7:1 reconstruction)", fontsize=15)
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_rgb_panels(
    *,
    evaluation_root: Path,
    dataset: Path,
    camera: str,
    frames: list[int],
    output: Path,
) -> None:
    row_labels = ["Ground truth tissue"]
    for _key, display in METHODS:
        row_labels.extend((display, f"|GT - {display}|"))
    figure, axes = plt.subplots(
        len(row_labels),
        len(frames),
        figsize=(4.3 * len(frames), 2.8 * len(row_labels)),
        squeeze=False,
    )
    for column, frame in enumerate(frames):
        filename = f"{frame:06d}.png"
        reference = load_rgb(dataset / "rgb" / camera / filename)
        reference_mask = load_mask(
            dataset / "ground_truth/masks/tissue" / camera / filename
        )
        predictions = []
        alphas = []
        for key, _display in METHODS:
            render_root = evaluation_root / key / "artifacts/renders"
            predictions.append(load_rgb(render_root / "rgb" / camera / filename))
            alphas.append(load_rgb(render_root / "alpha" / camera / filename)[..., 0])
        crop_y, crop_x = union_crop(reference_mask, alphas)
        reference_layer = reference * reference_mask[..., None]
        axes[0, column].imshow(reference_layer[crop_y, crop_x])
        row = 1
        for prediction in predictions:
            axes[row, column].imshow(prediction[crop_y, crop_x])
            difference = np.mean(
                np.abs(reference_layer[crop_y, crop_x] - prediction[crop_y, crop_x]),
                axis=2,
            )
            axes[row + 1, column].imshow(
                difference,
                cmap="magma",
                vmin=0.0,
                vmax=0.45,
            )
            row += 2
        axes[0, column].set_title(f"frame {frame}")
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9)
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"Reconstruction RGB and absolute-error diagnostics ({camera})",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def mean_metrics(values: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        key: float(np.mean(values[key]))
        for key in (
            "track_3d_mean_mm",
            "track_2d_mean_px",
            "psnr_db",
            "ssim",
            "lpips",
        )
    }


def main() -> None:
    args = parse_args()
    evaluation_root = args.evaluation_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else evaluation_root / "reconstruction_diagnostics"
    )
    if output_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有诊断目录：{output_dir}")
    reports = {
        key: json.loads(
            (evaluation_root / key / "trajectory_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        for key, _display in METHODS
    }
    split_frames = {int(report["split_frame_index"]) for report in reports.values()}
    datasets = {Path(report["reference_dataset"]).resolve() for report in reports.values()}
    if len(split_frames) != 1 or len(datasets) != 1:
        raise ValueError("三组评估的 80/20 分界或参考数据集不一致")
    split_frame = split_frames.pop()
    dataset = datasets.pop()
    series = {
        key: load_per_frame(evaluation_root / key, split_frame)
        for key, _display in METHODS
    }
    if args.selected_frames:
        selected_frames = [
            int(value) for value in args.selected_frames.split(",") if value.strip()
        ]
    else:
        selected_frames = np.linspace(0, split_frame - 1, 4).round().astype(int).tolist()
    if any(frame < 0 or frame >= split_frame for frame in selected_frames):
        raise ValueError("重建诊断帧必须全部位于前 80% 区间")

    output_dir.mkdir(parents=True)
    curves_path = output_dir / "reconstruction_metric_curves.png"
    panels_path = output_dir / "reconstruction_rgb_error_panels.png"
    plot_curves(series, split_frame, curves_path)
    plot_rgb_panels(
        evaluation_root=evaluation_root,
        dataset=dataset,
        camera=args.camera,
        frames=selected_frames,
        output=panels_path,
    )
    summary = {key: mean_metrics(values) for key, values in series.items()}
    pbd = summary["pbd"]
    improvements: dict[str, dict[str, float]] = {}
    for key, _display in METHODS[1:]:
        current = summary[key]
        improvements[key] = {
            "track_3d_mean_reduction_percent": 100.0
            * (pbd["track_3d_mean_mm"] - current["track_3d_mean_mm"])
            / pbd["track_3d_mean_mm"],
            "track_2d_mean_reduction_percent": 100.0
            * (pbd["track_2d_mean_px"] - current["track_2d_mean_px"])
            / pbd["track_2d_mean_px"],
            "psnr_gain_db": current["psnr_db"] - pbd["psnr_db"],
            "ssim_gain": current["ssim"] - pbd["ssim"],
            "lpips_reduction_percent": 100.0
            * (pbd["lpips"] - current["lpips"])
            / pbd["lpips"],
        }
    report = {
        "schema": "fixedsuperbest.reconstruction_diagnostics.v1",
        "evaluation_root": str(evaluation_root),
        "reference_dataset": str(dataset),
        "protocol": (
            "legacy front-80% causal assimilation diagnostic only; "
            "not the formal EH-SurGS 7:1 reconstruction evaluation"
        ),
        "split_frame_index": split_frame,
        "evaluated_frame_range": [0, split_frame - 1],
        "selected_rgb_frames": selected_frames,
        "camera_for_rgb_panels": args.camera,
        "means": summary,
        "relative_to_pbd": improvements,
        "figures": {
            "metric_curves": curves_path.name,
            "rgb_error_panels": panels_path.name,
        },
    }
    (output_dir / "reconstruction_diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
