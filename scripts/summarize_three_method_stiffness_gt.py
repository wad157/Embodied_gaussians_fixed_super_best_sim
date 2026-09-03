#!/usr/bin/env python3
"""汇总同一数据集三种方法的材料场与十区域真值比较。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


METHOD_NAMES = {
    "pbd": "PBD",
    "pbd_visual_residual": "PBD + 视觉残差",
    "pbd_visual_residual_stiffness": "PBD + 视觉残差 + 刚度更新",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        nargs=3,
        action="append",
        required=True,
        metavar=("KEY", "RUN_DIR", "COMPARISON_DIR"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def weighted_stereo_mean(cameras: dict, field: str) -> float:
    total = sum(int(camera["count"]) for camera in cameras.values())
    return sum(
        float(camera[field]) * int(camera["count"])
        for camera in cameras.values()
    ) / total


def pooled_stereo_rmse(cameras: dict) -> float:
    total = sum(int(camera["count"]) for camera in cameras.values())
    return math.sqrt(
        sum(
            float(camera["rmse"]) ** 2 * int(camera["count"])
            for camera in cameras.values()
        )
        / total
    )


def read_method(key: str, run_arg: str, comparison_arg: str) -> dict[str, object]:
    if key not in METHOD_NAMES:
        raise ValueError(f"未知方法：{key}")
    run_dir = Path(run_arg).expanduser().resolve()
    comparison_dir = Path(comparison_arg).expanduser().resolve()
    trajectory_path = run_dir / "trajectory_metrics.json"
    comparison_path = comparison_dir / "comparison.json"
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    final_frame = int(comparison["analysis_frames"][-1])
    final = comparison["selected_frames"][str(final_frame)]
    region = final["region_level"]
    cameras = trajectory["2d"]
    return {
        "method_key": key,
        "method": METHOD_NAMES[key],
        "run_dir": str(run_dir),
        "comparison_dir": str(comparison_dir),
        "final_frame": final_frame,
        "distance_particle_spearman": final["distance_particle_level"][
            "spearman"
        ],
        "distance_particle_log_pattern_rmse": final[
            "distance_particle_level"
        ]["relative_log_pattern_rmse"],
        "distance_region_spearman": region["distance_region_correlation"][
            "spearman"
        ],
        "distance_region_log_pattern_rmse": region[
            "distance_region_relative_pattern"
        ]["relative_log_pattern_rmse"],
        "distance_floor_fraction": final["distance_particle_level"][
            "floor_fraction"
        ],
        "distance_soft_hard_accuracy": final["distance_particle_level"][
            "soft_hard_accuracy"
        ],
        "shape_particle_spearman": final["shape_particle_level"]["spearman"],
        "shape_particle_log_pattern_rmse": final["shape_particle_level"][
            "relative_log_pattern_rmse"
        ],
        "shape_region_spearman": region["shape_region_correlation"][
            "spearman"
        ],
        "shape_region_log_pattern_rmse": region[
            "shape_region_relative_pattern"
        ]["relative_log_pattern_rmse"],
        "shape_floor_fraction": final["shape_particle_level"][
            "floor_fraction"
        ],
        "shape_soft_hard_accuracy": final["shape_particle_level"][
            "soft_hard_accuracy"
        ],
        "true_softest_region": region["true_softest_region"],
        "true_hardest_region": region["true_hardest_region"],
        "distance_softest_region": region["distance_softest_region"],
        "distance_hardest_region": region["distance_hardest_region"],
        "shape_softest_region": region["shape_softest_region"],
        "shape_hardest_region": region["shape_hardest_region"],
        "trajectory_3d_mean_mm": trajectory["3d"]["mean"],
        "trajectory_3d_rmse_mm": trajectory["3d"]["rmse"],
        "trajectory_2d_stereo_mean_px": weighted_stereo_mean(cameras, "mean"),
        "trajectory_2d_stereo_rmse_px": pooled_stereo_rmse(cameras),
    }


def fmt(value: object, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"拒绝覆盖三方法材料汇总：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [read_method(*method) for method in args.method]
    if [row["method_key"] for row in rows] != list(METHOD_NAMES):
        raise ValueError("三方法必须按PBD、视觉残差、视觉残差+刚度顺序提供")
    baseline = rows[1]
    updated = rows[2]

    def improvement(field: str) -> float:
        old = float(baseline[field])
        new = float(updated[field])
        return (old - new) / old

    report = {
        "schema": "fixedsuperbest.three_method_material_gt.v1",
        "comparison_policy": (
            "同一lift30mm数据集、相同初值和7:1协议；Pa与PBD参数去除"
            "各自全局几何均值后比较相对空间软硬分布"
        ),
        "relative_pattern_metric": (
            "RMSE((log k - mean(log k)) - (log E - mean(log E))); "
            "越低表示相对区域柔软度越接近真值，且不依赖Pa/PBD绝对单位"
        ),
        "rows": rows,
        "stiffness_update_vs_visual_residual": {
            "distance_particle_pattern_improvement_fraction": improvement(
                "distance_particle_log_pattern_rmse"
            ),
            "distance_region_pattern_improvement_fraction": improvement(
                "distance_region_log_pattern_rmse"
            ),
            "shape_particle_pattern_improvement_fraction": improvement(
                "shape_particle_log_pattern_rmse"
            ),
            "shape_region_pattern_improvement_fraction": improvement(
                "shape_region_log_pattern_rmse"
            ),
        },
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    scalar_fields = [
        key for key, value in rows[0].items() if not isinstance(value, list)
    ]
    with (output_dir / "comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# lift30mm三方法柔软度真值比较",
        "",
        "相对分布RMSE去除了Pa与PBD参数的全局尺度，越低越接近真实区域软硬分布。",
        "",
        "| 方法 | D粒子ρ ↑ | D粒子分布RMSE ↓ | D区域ρ ↑ | D区域分布RMSE ↓ | D下限占比 | S粒子ρ ↑ | S粒子分布RMSE ↓ | S区域ρ ↑ | S区域分布RMSE ↓ | S下限占比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {fmt(row['distance_particle_spearman'])} | "
            f"{fmt(row['distance_particle_log_pattern_rmse'])} | "
            f"{fmt(row['distance_region_spearman'])} | "
            f"{fmt(row['distance_region_log_pattern_rmse'])} | "
            f"{fmt(row['distance_floor_fraction'])} | "
            f"{fmt(row['shape_particle_spearman'])} | "
            f"{fmt(row['shape_particle_log_pattern_rmse'])} | "
            f"{fmt(row['shape_region_spearman'])} | "
            f"{fmt(row['shape_region_log_pattern_rmse'])} | "
            f"{fmt(row['shape_floor_fraction'])} |"
        )
    changes = report["stiffness_update_vs_visual_residual"]
    lines.extend(
        (
            "",
            "## 刚度更新相对视觉残差组",
            "",
            f"- Distance粒子相对分布误差改善：{float(changes['distance_particle_pattern_improvement_fraction']) * 100.0:.2f}%",
            f"- Distance区域相对分布误差改善：{float(changes['distance_region_pattern_improvement_fraction']) * 100.0:.2f}%",
            f"- Shape粒子相对分布误差改善：{float(changes['shape_particle_pattern_improvement_fraction']) * 100.0:.2f}%",
            f"- Shape区域相对分布误差改善：{float(changes['shape_region_pattern_improvement_fraction']) * 100.0:.2f}%",
            "",
            "负数表示刚度更新后反而比保持统一初值更偏离真实相对柔软度分布。",
            "",
        )
    )
    (output_dir / "comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
