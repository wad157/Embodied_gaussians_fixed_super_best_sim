#!/usr/bin/env python3
"""汇总 RGB FoundationStereo 深度下的 A/B/C 完整测评。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CAPABILITIES = ("reconstruction_7to1", "future_80to20")
DEFAULT_METHODS = (
    "pbd",
    "pbd_cotracker_foundation_depth",
    "pbd_cotracker_foundation_depth_global_distribution",
)
LABELS = {
    "pbd": "A：纯 PBD",
    "pbd_cotracker_foundation_depth": (
        "B：PBD + CoTracker轨迹校正 + FoundationStereo RGB深度"
    ),
    "pbd_cotracker_foundation_depth_global_distribution": (
        "C：B + 分布鲁棒刚度更新"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--depth-summary", type=Path, required=True)
    parser.add_argument("--observation-report", type=Path, required=True)
    parser.add_argument(
        "--method-b",
        default="pbd_cotracker_foundation_depth",
    )
    parser.add_argument(
        "--label-b",
        default=(
            "B：PBD + CoTracker轨迹校正 + FoundationStereo RGB深度"
        ),
    )
    parser.add_argument(
        "--method-c",
        default="pbd_cotracker_foundation_depth_global_distribution",
    )
    parser.add_argument(
        "--label-c",
        default="C：B + 分布鲁棒刚度更新",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def weighted_eye(metric: dict, field: str) -> float:
    eyes = metric["2d"]
    count = sum(int(value["count"]) for value in eyes.values())
    return sum(
        float(value[field]) * int(value["count"]) for value in eyes.values()
    ) / max(count, 1)


def main() -> None:
    args = parse_args()
    root = args.evaluation_root.resolve()
    methods = (DEFAULT_METHODS[0], str(args.method_b), str(args.method_c))
    labels = dict(LABELS)
    labels[str(args.method_b)] = str(args.label_b)
    labels[str(args.method_c)] = str(args.label_c)
    depth_summary = load(args.depth_summary.resolve())
    observation_report = load(args.observation_report.resolve())
    if observation_report.get("depth_estimated_from_rgb") is not True:
        raise ValueError("观测资产没有声明 depth_estimated_from_rgb=true")
    if "foundation" not in str(observation_report.get("depth_source", "")).lower():
        raise ValueError("观测资产不是FoundationStereo深度")

    results: dict[str, dict[str, dict]] = {}
    for capability in CAPABILITIES:
        results[capability] = {}
        for method in methods:
            directory = root / capability / method
            trajectory_path = directory / "trajectory_metrics.json"
            rendering_path = directory / "render_metrics.json"
            trajectory = load(trajectory_path)
            rendering = load(rendering_path)["summary"]["all"]
            results[capability][method] = {
                "label": labels[method],
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
        "schema": "fixedsuperbest.sim_foundation_depth_complete_evaluation.v2",
        "protocol": {
            "evaluation_points": (
                f"{results['reconstruction_7to1'][methods[0]]['evaluation_points']} "
                "fixed non-grasp nodes; never optimizer inputs"
            ),
            "optimizer_tracks": (
                f"{observation_report['requested_track_count']} fixed CoTracker "
                "RGB material tracks"
            ),
            "depth": "FoundationStereo estimated from stereo RGB; simulator GT depth forbidden",
            "reconstruction": (
                "7:1 causal holdout; "
                f"{results['reconstruction_7to1'][methods[0]]['frames']} scored frames"
            ),
            "future_prediction": "first 80% adaptation; final 20% open-loop",
            "same_state_correction_parameters_for_B_and_C": True,
            "same_stiffness_policy_for_both_capabilities": True,
            "no_protocol_branch_or_parameter_selection": True,
        },
        "depth_generation": {
            "summary": str(args.depth_summary.resolve()),
            "uses_ground_truth_depth_for_estimation": depth_summary["inputs"][
                "uses_ground_truth_depth_for_estimation"
            ],
            "aggregate": depth_summary["aggregate"],
        },
        "observation_assets": {
            "report": str(args.observation_report.resolve()),
            "depth_source": observation_report["depth_source"],
            "depth_estimated_from_rgb": observation_report[
                "depth_estimated_from_rgb"
            ],
            "optimizer_track_count": observation_report["requested_track_count"],
            "bound_track_count": observation_report["bound_track_count"],
        },
        "results": results,
    }
    (root / "comparison_complete.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# FoundationStereo RGB深度：完整测评",
        "",
        "- 深度由左右RGB经FoundationStereo估计，仿真GT深度不进入观测或优化。",
        (
            f"- 优化使用{observation_report['requested_track_count']}条CoTracker轨迹"
            f"（有效绑定{observation_report['bound_track_count']}条）；最终指标只使用"
            f"{results['reconstruction_7to1'][methods[0]]['evaluation_points']}个固定非夹持点。"
        ),
        "- 重建采用7:1因果留出；未来预测采用前80%更新、后20%开环。",
        "- A/B/C的夹持边界、PBD参数和状态校正参数一致；C仅增加在线刚度更新。",
        "",
    ]
    for capability in CAPABILITIES:
        title = "7:1 重建" if capability == "reconstruction_7to1" else "80/20 未来预测"
        lines.extend(
            [
                f"## {title}",
                "",
                "| 方法 | 3D Mean (mm)↓ | RMSE↓ | Median↓ | P95↓ | Max↓ | 2D Mean (px)↓ | PSNR↑ | SSIM↑ | LPIPS↓ |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in methods:
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
