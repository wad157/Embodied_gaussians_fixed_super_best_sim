#!/usr/bin/env python3
"""汇总三个 SIM 数据集各三次完整 A/B/C 测评的均值与标准差。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


DATASETS = (
    ("sim01", "SIM-01 Planar-X-Pull"),
    ("sim02", "SIM-02 Planar-Y-Pull"),
    ("sim03", "SIM-03 Edge-Z-Lift"),
)
CAPABILITIES = (
    ("reconstruction_7to1", "7:1 重建"),
    ("future_80to20", "80/20 未来预测"),
)
METHODS = (
    ("pbd", "A：纯 PBD"),
    ("pbd_cotracker_foundation_depth", "B：PBD + RGB轨迹校正"),
    (
        "pbd_cotracker_foundation_depth_global_only_h3w4",
        "C：B + 全局刚度/阻尼更新（H1:H2:H3=1.5:2:4）",
    ),
)
METRICS = (
    ("3d_mean_mm", "3D Tracking (mm)↓"),
    ("2d_mean_px", "2D Tracking (px)↓"),
    ("psnr_db", "PSNR (dB)↑"),
    ("ssim", "SSIM↑"),
    ("lpips", "LPIPS↓"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repeat-count", type=int, default=3)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def fmt(metric: str, mean: float, std: float) -> str:
    digits = 4 if metric in {"ssim", "lpips"} else 3
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    repeat_names = [f"repeat_{index:02d}" for index in range(1, args.repeat_count + 1)]

    raw: dict[str, dict[str, dict[str, dict[str, dict]]]] = {}
    for repeat in repeat_names:
        raw[repeat] = {}
        for dataset_id, _ in DATASETS:
            path = root / repeat / dataset_id / "comparison_complete.json"
            if not path.is_file():
                raise FileNotFoundError(f"缺少完整测评：{path}")
            payload = load(path)
            raw[repeat][dataset_id] = payload["results"]

    aggregate: dict[str, dict[str, dict[str, dict]]] = {}
    for dataset_id, dataset_label in DATASETS:
        aggregate[dataset_id] = {}
        for capability, capability_label in CAPABILITIES:
            aggregate[dataset_id][capability] = {}
            for method, method_label in METHODS:
                rows = [raw[repeat][dataset_id][capability][method] for repeat in repeat_names]
                metric_summary = {}
                for metric, _ in METRICS:
                    values = [float(row[metric]) for row in rows]
                    mean, std = mean_std(values)
                    metric_summary[metric] = {
                        "mean": mean,
                        "sample_std": std,
                        "values": values,
                    }
                aggregate[dataset_id][capability][method] = {
                    "dataset_label": dataset_label,
                    "capability_label": capability_label,
                    "method_label": method_label,
                    "repeat_count": len(rows),
                    "metrics": metric_summary,
                }

    output = {
        "schema": "fixedsuperbest.sim_three_datasets_three_repeats.v1",
        "protocol": {
            "repeat_count": args.repeat_count,
            "initial_distance_stiffness": 0.20,
            "initial_shape_stiffness": 0.004,
            "stiffness_update_mode": "differentiable_global",
            "global_horizon_weights_h1_h2_h3": [1.5, 2.0, 4.0],
            "distance_bounds": [0.01, 2.0],
            "maximum_log_offset": 2.302585093,
            "depth": "FoundationStereo estimated independently from each dataset RGB stereo pair",
            "evaluation_points": "30 fixed non-grasp nodes per dataset",
        },
        "aggregate": aggregate,
    }
    json_path = root / "comparison_mean_std.json"
    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# 三数据集适中初始刚度三次独立复测",
        "",
        "- 每个数据集、每种能力、每种方法均独立运行3次。",
        "- 数值格式为均值 ± 样本标准差；初始距离刚度为0.20，shape为0.004。",
        "- C组统一使用全局距离刚度/阻尼在线更新，H1:H2:H3=1.5:2:4。",
        "- 深度与轨迹资产均来自对应数据集自身，不跨数据集混用。",
        "",
    ]
    for capability, capability_label in CAPABILITIES:
        lines.extend(
            [
                f"## {capability_label}",
                "",
                "| 数据集 | 方法 | n | 3D Tracking (mm)↓ | 2D Tracking (px)↓ | PSNR (dB)↑ | SSIM↑ | LPIPS↓ |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for dataset_id, dataset_label in DATASETS:
            for method, method_label in METHODS:
                row = aggregate[dataset_id][capability][method]
                values = row["metrics"]
                cells = [fmt(metric, values[metric]["mean"], values[metric]["sample_std"]) for metric, _ in METRICS]
                lines.append(
                    f"| {dataset_label} | {method_label} | {row['repeat_count']} | "
                    + " | ".join(cells)
                    + " |"
                )
        lines.append("")

    lines.extend(
        [
            "## 三次原始结果",
            "",
            "| 次数 | 数据集 | 能力 | 方法 | 3D (mm) | 2D (px) | PSNR | SSIM | LPIPS |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for repeat_index, repeat in enumerate(repeat_names, start=1):
        for dataset_id, dataset_label in DATASETS:
            for capability, capability_label in CAPABILITIES:
                for method, method_label in METHODS:
                    row = raw[repeat][dataset_id][capability][method]
                    lines.append(
                        f"| {repeat_index} | {dataset_label} | {capability_label} | {method_label} | "
                        f"{row['3d_mean_mm']:.4f} | {row['2d_mean_px']:.3f} | "
                        f"{row['psnr_db']:.3f} | {row['ssim']:.4f} | {row['lpips']:.4f} |"
                    )
    lines.append("")
    md_path = root / "comparison_mean_std.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"写入：{md_path}", flush=True)


if __name__ == "__main__":
    main()
