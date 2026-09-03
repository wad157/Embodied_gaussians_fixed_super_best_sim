#!/usr/bin/env python3
"""汇总 FoundationStereo 下鲁棒相对形变刚度探针的轨迹指标。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CAPABILITIES = ("reconstruction_7to1", "future_80to20")
BASE_METHODS = (
    "pbd",
    "pbd_cotracker_foundation_depth",
)
LABELS = {
    "pbd": "A：纯 PBD",
    "pbd_cotracker_foundation_depth": (
        "B：PBD + CoTracker轨迹校正 + FoundationStereo RGB深度"
    ),
    "pbd_cotracker_foundation_depth_global_relative": (
        "C-R：B + Cauchy相对形变 + 累计Warp刚度更新"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument(
        "--method-c",
        default="pbd_cotracker_foundation_depth_global_relative",
    )
    parser.add_argument(
        "--label-c",
        default="C-R：B + Cauchy相对形变 + 累计Warp刚度更新",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def weighted_eye(metric: dict, field: str) -> float:
    eyes = metric["2d"]
    count = sum(int(value["count"]) for value in eyes.values())
    return sum(
        float(value[field]) * int(value["count"])
        for value in eyes.values()
    ) / max(count, 1)


def improvement(baseline: float, candidate: float) -> float:
    return 100.0 * (baseline - candidate) / max(abs(baseline), 1.0e-12)


def main() -> None:
    args = parse_args()
    root = args.evaluation_root.resolve()
    methods = (*BASE_METHODS, str(args.method_c))
    labels = dict(LABELS)
    labels[str(args.method_c)] = str(args.label_c)
    results: dict[str, dict[str, dict[str, float | int | str]]] = {}
    for capability in CAPABILITIES:
        rows = {}
        for method in methods:
            path = root / capability / method / "trajectory_metrics.json"
            metric = load(path)
            rows[method] = {
                "label": labels[method],
                "frames": int(metric["frames"]),
                "evaluation_points": int(metric["evaluated_nodes"]),
                "3d_mean_mm": float(metric["3d"]["mean"]),
                "3d_rmse_mm": float(metric["3d"]["rmse"]),
                "3d_median_mm": float(metric["3d"]["median"]),
                "3d_p95_mm": float(metric["3d"]["p95"]),
                "3d_max_mm": float(metric["3d"]["max"]),
                "2d_mean_px": weighted_eye(metric, "mean"),
                "2d_rmse_px": weighted_eye(metric, "rmse"),
                "metrics": str(path),
            }
        b = rows[methods[1]]
        c = rows[methods[2]]
        rows["c_relative_vs_b_improvement_percent"] = {
            field: improvement(float(b[field]), float(c[field]))
            for field in (
                "3d_mean_mm",
                "3d_rmse_mm",
                "3d_median_mm",
                "3d_p95_mm",
                "3d_max_mm",
                "2d_mean_px",
                "2d_rmse_px",
            )
        }
        results[capability] = rows

    payload = {
        "schema": "fixedsuperbest.sim_relative_stiffness_probe.v1",
        "protocol": {
            "depth": "FoundationStereo estimated from RGB; no simulator GT depth",
            "optimizer_tracks": "203 CoTracker tracks (200 valid bindings)",
            "evaluation_points": "30 fixed non-grasp points",
            "reconstruction": "7:1 causal holdout",
            "future_prediction": "first 80% adaptation, final 20% open-loop",
            "parameter_selection": False,
            "future_observation_access": False,
            "rendering_metrics_in_probe": False,
        },
        "results": results,
    }
    output = root / "trajectory_probe_comparison.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# 鲁棒相对形变刚度更新：轨迹探针",
        "",
        "本轮只判断2D/3D轨迹是否值得进入完整渲染测评；不计算PSNR/SSIM/LPIPS。",
        "",
    ]
    for capability in CAPABILITIES:
        lines.extend(
            (
                f"## {capability}",
                "",
                "| 方法 | 3D Mean (mm)↓ | RMSE↓ | Median↓ | P95↓ | Max↓ | 2D Mean (px)↓ | 2D RMSE↓ |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            )
        )
        for method in methods:
            row = results[capability][method]
            lines.append(
                f"| {row['label']} | {row['3d_mean_mm']:.4f} | "
                f"{row['3d_rmse_mm']:.4f} | {row['3d_median_mm']:.4f} | "
                f"{row['3d_p95_mm']:.4f} | {row['3d_max_mm']:.4f} | "
                f"{row['2d_mean_px']:.3f} | {row['2d_rmse_px']:.3f} |"
            )
        gain = results[capability]["c_relative_vs_b_improvement_percent"]
        lines.extend(
            (
                "",
                "C-R 相对 B 的改善率（正数为改善）："
                f"3D Mean {gain['3d_mean_mm']:+.2f}%，"
                f"3D Max {gain['3d_max_mm']:+.2f}%，"
                f"2D Mean {gain['2d_mean_px']:+.2f}%。",
                "",
            )
        )
    (root / "trajectory_probe_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
