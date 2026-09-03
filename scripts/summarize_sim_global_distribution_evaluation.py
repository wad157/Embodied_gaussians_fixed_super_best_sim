#!/usr/bin/env python3
"""Summarize the balanced global-system-ID reconstruction/future evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CAPABILITIES = ("reconstruction_7to1", "future_80to20")
METHODS = (
    "pbd",
    "pbd_cotracker_gt_depth",
    "pbd_cotracker_gt_depth_global_distribution",
)
LABELS = {
    "pbd": "纯 PBD",
    "pbd_cotracker_gt_depth": "PBD + CoTracker轨迹校正 + 仿真GT深度",
    "pbd_cotracker_gt_depth_global_distribution": (
        "PBD + CoTracker轨迹校正 + 仿真GT深度 + 分布鲁棒刚度更新"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--pure-pbd-root", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def weighted_eye(metric: dict, field: str) -> float:
    eyes = metric["2d"]
    count = sum(int(value["count"]) for value in eyes.values())
    return sum(
        float(value[field]) * int(value["count"]) for value in eyes.values()
    ) / max(count, 1)


def paths(
    evaluation_root: Path,
    reference_root: Path,
    pure_pbd_root: Path,
    capability: str,
    method: str,
) -> tuple[Path, Path]:
    if method == "pbd":
        trajectory = (
            reference_root
            / "baseline_30"
            / capability
            / method
            / "trajectory_metrics.json"
        )
        rendering = pure_pbd_root / capability / method / "render_metrics.json"
    elif method == "pbd_cotracker_gt_depth":
        directory = reference_root / capability / method
        trajectory = directory / "trajectory_metrics.json"
        rendering = directory / "render_metrics.json"
    else:
        directory = evaluation_root / capability / method
        trajectory = directory / "trajectory_metrics.json"
        rendering = directory / "render_metrics.json"
    return trajectory, rendering


def main() -> None:
    args = parse_args()
    root = args.evaluation_root.resolve()
    reference = args.reference_root.resolve()
    pure_pbd = args.pure_pbd_root.resolve()
    results: dict[str, dict[str, dict]] = {}
    for capability in CAPABILITIES:
        results[capability] = {}
        for method in METHODS:
            trajectory_path, rendering_path = paths(
                root, reference, pure_pbd, capability, method
            )
            trajectory = load(trajectory_path)
            rendering = load(rendering_path)["summary"]["all"]
            results[capability][method] = {
                "label": LABELS[method],
                "frames": int(trajectory["frames"]),
                "evaluation_points": int(trajectory["evaluated_nodes"]),
                "3d_mean_mm": float(trajectory["3d"]["mean"]),
                "3d_rmse_mm": float(trajectory["3d"]["rmse"]),
                "3d_median_mm": float(trajectory["3d"]["median"]),
                "3d_p95_mm": float(trajectory["3d"]["p95"]),
                "3d_max_mm": float(trajectory["3d"]["max"]),
                "2d_mean_px": weighted_eye(trajectory, "mean"),
                "2d_rmse_px": weighted_eye(trajectory, "rmse"),
                "psnr_db": float(rendering["psnr_tissue_layer_db"]),
                "ssim": float(rendering["ssim_tissue_layer"]),
                "lpips": float(rendering["lpips_tissue_layer_alex_v0.1"]),
                "trajectory_metrics": str(trajectory_path),
                "render_metrics": str(rendering_path),
            }

    payload = {
        "schema": "fixedsuperbest.global_trackmean_warpfd_evaluation.v4",
        "protocol": {
            "points": "30 fixed non-grasp farthest-point nodes",
            "reconstruction": "7:1 causal holdout",
            "future": "first 80% adaptation, last 20% open-loop",
            "optimizer_tracks": (
                "203 fixed CoTracker material tracks; 30 evaluation nodes are "
                "not optimizer inputs"
            ),
            "loss": (
                "0.85 track mean + 0.10 equal-region mean + "
                "0.05 worst-quartile regions"
            ),
            "causal_horizon_weights": {"H1": 1.5, "H2": 2.0, "H3": 3.0},
            "optimizer_gradient": (
                "Warp central finite difference after fixed Torch/Warp cosine "
                "gate >= 0.95"
            ),
            "candidate_validity": (
                "max absolute log step over complete distance+damping vector"
            ),
            "grasp_coupling_gain": 1.0,
            "same_policy_for_all_protocols": True,
            "evaluation_gt_not_used_by_optimizer": True,
        },
        "results": results,
    }
    (root / "comparison_complete.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 分布鲁棒全局刚度更新：完整测评",
        "",
        "- 评估点：30 个固定非夹持点；优化器不读取评估真值。",
        "- 重建：7:1 因果留出；未来预测：前 80% 更新，后 20% 开环。",
        "- 状态校正：CoTracker RGB二维轨迹 + 仿真GT深度；不是像素级视觉残差。",
        "- 优化点：203条固定CoTracker表面材料轨迹；30点评估清单不进入优化器。",
        "- 新损失：85%轨迹Mean + 10%空间区域Mean + 5%最差区域。",
        "- 统一时间权重：H1:H2:H3=1.5:2:3；两种协议不切换策略。",
        "- 候选有效性：检查完整distance+damping参数步，不再只检查distance粒子映射。",
        "- 更新梯度：固定余弦门≥0.95；通过后由Warp有限差分梯度驱动Adam。",
        "- 已知夹持边界耦合固定为1，不参与估计。",
        "",
    ]
    for capability in CAPABILITIES:
        title = "7:1 重建" if capability == "reconstruction_7to1" else "80/20 未来预测"
        lines.extend(
            [
                f"## {title}",
                "",
                "| 方法 | 3D Mean↓ | RMSE↓ | Median↓ | P95↓ | Max↓ | 2D Mean↓ | PSNR↑ | SSIM↑ | LPIPS↓ |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            row = results[capability][method]
            lines.append(
                f"| {row['label']} | {row['3d_mean_mm']:.4f} | "
                f"{row['3d_rmse_mm']:.4f} | {row['3d_median_mm']:.4f} | "
                f"{row['3d_p95_mm']:.4f} | {row['3d_max_mm']:.4f} | "
                f"{row['2d_mean_px']:.3f} | {row['psnr_db']:.3f} | "
                f"{row['ssim']:.4f} | {row['lpips']:.4f} |"
            )
        lines.append("")
    (root / "comparison_complete.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
