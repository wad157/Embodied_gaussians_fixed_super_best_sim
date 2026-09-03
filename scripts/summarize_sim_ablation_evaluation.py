#!/usr/bin/env python3
"""汇总 PBD、PBD+视觉残差、PBD+视觉残差+刚度修正三组评估。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


VARIANTS = (
    ("pbd", "PBD"),
    ("pbd_visual_residual", "PBD + 视觉残差"),
    (
        "pbd_visual_residual_stiffness",
        "PBD + 视觉残差 + 刚度修正",
    ),
)
SEGMENTS = (
    ("all", "全序列"),
    ("assimilation_0_79_percent", "前 80% 同化"),
    ("future_open_loop_20_percent", "后 20% 开环预测"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        required=True,
        help="包含三个固定名称消融子目录的评估根目录。",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def weighted_stereo_mean(cameras: dict, field: str) -> float | None:
    numerator = 0.0
    denominator = 0
    for camera in ("stereo_left", "stereo_right"):
        metrics = cameras[camera]
        value = metrics.get(field)
        count = int(metrics.get("count", 0))
        if value is not None and count:
            numerator += float(value) * count
            denominator += count
    return numerator / denominator if denominator else None


def pooled_stereo_rmse(cameras: dict) -> float | None:
    squared_error_sum = 0.0
    denominator = 0
    for camera in ("stereo_left", "stereo_right"):
        metrics = cameras[camera]
        value = metrics.get("rmse")
        count = int(metrics.get("count", 0))
        if value is not None and count:
            squared_error_sum += float(value) ** 2 * count
            denominator += count
    return math.sqrt(squared_error_sum / denominator) if denominator else None


def fmt(value: object, digits: int) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def main() -> None:
    args = parse_args()
    root = args.evaluation_root.expanduser().resolve()
    outputs = (
        root / "comparison.json",
        root / "comparison.csv",
        root / "comparison.md",
    )
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"拒绝覆盖已有三组汇总：{existing}")

    rows: list[dict[str, object]] = []
    inputs: dict[str, object] = {}
    for variant_key, variant_name in VARIANTS:
        variant_root = root / variant_key
        trajectory_path = variant_root / "trajectory_metrics.json"
        rendering_path = variant_root / "render_metrics.json"
        artifact_path = variant_root / "artifacts/artifact_metadata.json"
        trajectory = read_json(trajectory_path)
        rendering = read_json(rendering_path)
        artifact = read_json(artifact_path)
        inputs[variant_key] = {
            "trajectory": str(trajectory_path),
            "rendering": str(rendering_path),
            "artifact": str(artifact_path),
            "initial_embedding_error_mm": artifact[
                "initial_embedding_error_mm"
            ],
        }
        for segment_key, segment_name in SEGMENTS:
            track = trajectory["segments"][segment_key]
            render = rendering["segments"][segment_key]["all"]
            cameras = track["2d"]
            rows.append(
                {
                    "variant": variant_key,
                    "method": variant_name,
                    "segment": segment_key,
                    "segment_zh": segment_name,
                    "frames": int(track["frame_count"]),
                    "track_3d_mean_mm": track["3d"].get("mean"),
                    "track_3d_rmse_mm": track["3d"].get("rmse"),
                    "track_3d_p95_mm": track["3d"].get("p95"),
                    "track_2d_stereo_mean_px": weighted_stereo_mean(
                        cameras, "mean"
                    ),
                    "track_2d_stereo_rmse_px": pooled_stereo_rmse(cameras),
                    "track_2d_left_mean_px": cameras["stereo_left"].get(
                        "mean"
                    ),
                    "track_2d_right_mean_px": cameras["stereo_right"].get(
                        "mean"
                    ),
                    "psnr_tissue_layer_db": render.get(
                        "psnr_tissue_layer_db"
                    ),
                    "ssim_tissue_layer": render.get("ssim_tissue_layer"),
                    "lpips_tissue_layer": render.get(
                        "lpips_tissue_layer_alex_v0.1"
                    ),
                }
            )

    report = {
        "schema": "fixedsuperbest.sim_ablation_comparison.v1",
        "evaluation_root": str(root),
        "variants": [key for key, _ in VARIANTS],
        "segments": [key for key, _ in SEGMENTS],
        "metric_direction": {
            "3d_tracking": "lower_is_better",
            "2d_tracking": "lower_is_better",
            "psnr": "higher_is_better",
            "ssim": "higher_is_better",
            "lpips": "lower_is_better",
        },
        "primary_render_region": "tissue_layer_union_crop",
        "initial_material_protocol": (
            "all variants use the predetermined uniform distance=0.20 and "
            "shape=0.004; no dataset-driven initial-parameter selection"
        ),
        "inputs": inputs,
        "rows": rows,
    }
    outputs[0].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with outputs[1].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 三组仿真重建消融评估",
        "",
        "三组统一使用预先固定的全局均匀初值 distance=0.20、shape=0.004；不进行参数分支选择，只有第三组允许根据 RGB 残差在线更新局部刚度。",
        "",
        "轨迹误差和 LPIPS 越低越好；PSNR、SSIM 越高越好。渲染主指标只比较组织层。",
        "",
        "| 方法 | 区间 | 3D Mean (mm) ↓ | 3D RMSE (mm) ↓ | 3D P95 (mm) ↓ | 双目 2D Mean (px) ↓ | 双目 2D RMSE (px) ↓ | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {segment} | {m3d} | {r3d} | {p3d} | "
            "{m2d} | {r2d} | {psnr} | {ssim} | {lpips} |".format(
                method=row["method"],
                segment=row["segment_zh"],
                m3d=fmt(row["track_3d_mean_mm"], 3),
                r3d=fmt(row["track_3d_rmse_mm"], 3),
                p3d=fmt(row["track_3d_p95_mm"], 3),
                m2d=fmt(row["track_2d_stereo_mean_px"], 3),
                r2d=fmt(row["track_2d_stereo_rmse_px"], 3),
                psnr=fmt(row["psnr_tissue_layer_db"], 3),
                ssim=fmt(row["ssim_tissue_layer"], 4),
                lpips=fmt(row["lpips_tissue_layer"], 4),
            )
        )
    lines.extend(
        (
            "",
            "说明：前 80% 允许各方法按组设定使用视觉观测；后 20% 冻结视觉残差与刚度更新，直接开环预测。已知五点夹持边界不计入轨迹误差。",
            "",
        )
    )
    outputs[2].write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
