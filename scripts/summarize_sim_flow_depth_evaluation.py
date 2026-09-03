#!/usr/bin/env python3
"""汇总30个非夹持点上的 PBD/RGB/CoTracker+真值深度对比。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CAPABILITIES = ("reconstruction_7to1", "future_80to20")
METHODS = (
    "pbd",
    "pbd_visual_residual",
    "pbd_cotracker_gt_depth",
    "pbd_cotracker_gt_depth_stiffness",
)
LABELS = {
    "pbd": "PBD",
    "pbd_visual_residual": "PBD + RGB视觉残差",
    "pbd_cotracker_gt_depth": "PBD + CoTracker + GT深度",
    "pbd_cotracker_gt_depth_stiffness": (
        "PBD + CoTracker + GT深度 + 在线刚度"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def method_paths(
    evaluation_root: Path, baseline_root: Path, capability: str, method: str
) -> tuple[Path, Path]:
    if method in {"pbd", "pbd_visual_residual"}:
        trajectory = (
            evaluation_root
            / "baseline_30"
            / capability
            / method
            / "trajectory_metrics.json"
        )
        rendering = baseline_root / capability / method / "render_metrics.json"
    else:
        directory = evaluation_root / capability / method
        trajectory = directory / "trajectory_metrics.json"
        rendering = directory / "render_metrics.json"
    return trajectory, rendering


def weighted_2d_mean(metric: dict) -> float:
    eyes = metric["2d"]
    count = sum(int(eyes[name]["count"]) for name in eyes)
    return sum(
        float(eyes[name]["mean"]) * int(eyes[name]["count"])
        for name in eyes
    ) / max(count, 1)


def main() -> None:
    args = parse_args()
    root = args.evaluation_root.expanduser().resolve()
    baseline = args.baseline_root.expanduser().resolve()
    result: dict[str, dict] = {}
    for capability in CAPABILITIES:
        result[capability] = {}
        for method in METHODS:
            trajectory_path, rendering_path = method_paths(
                root, baseline, capability, method
            )
            trajectory = load(trajectory_path)
            rendering = load(rendering_path)
            render_all = rendering["summary"]["all"]
            result[capability][method] = {
                "label": LABELS[method],
                "evaluation_points": int(trajectory["evaluated_nodes"]),
                "frames": int(trajectory["frames"]),
                "trajectory_3d_mean_mm": float(trajectory["3d"]["mean"]),
                "trajectory_3d_rmse_mm": float(trajectory["3d"]["rmse"]),
                "trajectory_2d_mean_px": weighted_2d_mean(trajectory),
                "psnr_tissue_layer_db": float(
                    render_all["psnr_tissue_layer_db"]
                ),
                "ssim_tissue_layer": float(render_all["ssim_tissue_layer"]),
                "lpips_tissue_layer": float(
                    render_all["lpips_tissue_layer_alex_v0.1"]
                ),
                "trajectory_metrics": str(trajectory_path),
                "render_metrics": str(rendering_path),
            }
        pbd = result[capability]["pbd"]
        for method in METHODS[1:]:
            row = result[capability][method]
            row["improvement_vs_pbd_3d_percent"] = 100.0 * (
                pbd["trajectory_3d_mean_mm"] - row["trajectory_3d_mean_mm"]
            ) / max(pbd["trajectory_3d_mean_mm"], 1.0e-12)
            row["improvement_vs_pbd_2d_percent"] = 100.0 * (
                pbd["trajectory_2d_mean_px"] - row["trajectory_2d_mean_px"]
            ) / max(pbd["trajectory_2d_mean_px"], 1.0e-12)

    payload = {
        "schema": "fixedsuperbest.sim_flow_depth_evaluation_30_points.v1",
        "evaluation_root": str(root),
        "baseline_root": str(baseline),
        "point_protocol": (
            "30 deterministic farthest-point tissue nodes; both-eye visibility "
            ">=80%; outside known grasp region"
        ),
        "depth_protocol": (
            "simulator ground-truth float32 depth; no RGB depth estimation"
        ),
        "tracker_protocol": (
            "CoTracker3 scaled offline on RGB; 2D/3D tissue GT trajectories are "
            "not optimization inputs"
        ),
        "results": result,
    }
    (root / "comparison_30_points.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# CoTracker + 仿真真值深度：30点正式对比",
        "",
        "- 评估点：30个固定非夹持组织点，双目完整序列可见率均≥80%。",
        "- 深度：直接读取仿真器逐帧 float32 真值深度，不从RGB估计。",
        "- 运动：CoTracker3从RGB得到二维轨迹；未把组织2D/3D真值轨迹喂给算法。",
        "- 高斯：只随绑定三角面更新，不直接写高斯中心。",
        "- 注意：当前CoTracker为离线模型，因此结果属于GT深度上限消融，不能直接宣称为严格在线RGB-only结果。",
        "",
    ]
    for capability in CAPABILITIES:
        title = "7:1 重建" if capability == "reconstruction_7to1" else "80/20 未来预测"
        lines.extend(
            [
                f"## {title}",
                "",
                "| 方法 | 3D mean↓ (mm) | 3D RMSE↓ (mm) | 2D mean↓ (px) | PSNR↑ (dB) | SSIM↑ | LPIPS↓ |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            row = result[capability][method]
            lines.append(
                f"| {row['label']} | {row['trajectory_3d_mean_mm']:.4f} | "
                f"{row['trajectory_3d_rmse_mm']:.4f} | "
                f"{row['trajectory_2d_mean_px']:.3f} | "
                f"{row['psnr_tissue_layer_db']:.3f} | "
                f"{row['ssim_tissue_layer']:.4f} | "
                f"{row['lpips_tissue_layer']:.4f} |"
            )
        lines.append("")
    (root / "comparison_30_points.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
