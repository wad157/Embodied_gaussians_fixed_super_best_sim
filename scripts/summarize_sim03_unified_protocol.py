#!/usr/bin/env python3
"""汇总 SIM-03 单 rollout 联合重建/未来预测协议的 B/C 指标。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = (
    ("pbd_alltracker_foundation_depth", "B：PBD + AllTracker RGB轨迹校正"),
    (
        "pbd_alltracker_foundation_depth_global_only_h3w4",
        "C：B + 全局刚度/阻尼更新（H1:H2:H3=1.5:2:4）",
    ),
)
CAPABILITIES = (
    ("reconstruction_7to1", "前80%内7:1重建"),
    ("future_80to20", "后20%开环未来预测"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--observation-report", type=Path, required=True)
    parser.add_argument("--future-start", type=int, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def weighted_2d(metric: dict, field: str = "mean") -> float:
    eyes = metric["2d"]
    count = sum(int(value["count"]) for value in eyes.values())
    return sum(
        float(value[field]) * int(value["count"]) for value in eyes.values()
    ) / max(count, 1)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    dataset = args.dataset.resolve()
    observation_report_path = args.observation_report.resolve()
    episode = load(dataset / "episode.json")
    frame_count = int(episode["frames"])
    expected_future_start = frame_count * 4 // 5
    if args.future_start != expected_future_start:
        raise ValueError(
            f"80/20分界错误：{args.future_start} != {expected_future_start}"
        )
    report = load(observation_report_path)
    if "alltracker" not in str(report.get("rgb_motion_source", "")).lower():
        raise ValueError("观测报告不是AllTracker")
    if report.get("depth_estimated_from_rgb") is not True:
        raise ValueError("观测报告没有声明RGB估计深度")

    results: dict[str, dict[str, dict]] = {}
    source_rollouts: dict[str, str] = {}
    holdout_frames = list(range(7, args.future_start, 8))
    future_frames = list(range(args.future_start, frame_count))
    for method, _ in METHODS:
        artifact_root = root / "methods" / method / "artifacts"
        metadata_path = artifact_root / "artifact_metadata.json"
        metadata = load(metadata_path)
        if Path(metadata["reference_dataset"]).resolve() != dataset:
            raise ValueError(
                f"{method} rollout数据集错误："
                f"{metadata['reference_dataset']} != {dataset}"
            )
        if (
            int(metadata["frames"]) != frame_count
            or int(metadata["frame_start"]) != 0
            or int(metadata["frame_end_inclusive"]) != frame_count - 1
        ):
            raise ValueError(f"{method} 没有完整导出{frame_count}帧")
        if int(metadata["open_loop_start_frame"]) != args.future_start:
            raise ValueError(f"{method} 的80/20分界错误")
        if metadata["holdout_frame_indices"] != holdout_frames:
            raise ValueError(f"{method} 的前80% 7:1留出帧错误")
        if metadata["render_frame_mode"] != "holdout_future":
            raise ValueError(f"{method} 未使用联合渲染帧模式")
        source_rollouts[method] = str(
            artifact_root / "predicted_trajectories.npz"
        )

    for capability, _ in CAPABILITIES:
        results[capability] = {}
        for method, label in METHODS:
            metrics_root = root / "metrics" / capability / method
            trajectory_path = metrics_root / "trajectory_metrics.json"
            render_path = metrics_root / "render_metrics.json"
            trajectory = load(trajectory_path)
            rendering = load(render_path)["summary"]["all"]
            results[capability][method] = {
                "label": label,
                "frames": int(trajectory["frames"]),
                "evaluation_points": int(trajectory["evaluated_nodes"]),
                "3d_mean_mm": float(trajectory["3d"]["mean"]),
                "2d_mean_px": weighted_2d(trajectory),
                "psnr_db": float(rendering["psnr_tissue_layer_db"]),
                "ssim": float(rendering["ssim_tissue_layer"]),
                "lpips": float(rendering["lpips_tissue_layer_alex_v0.1"]),
                "trajectory_metrics": str(trajectory_path),
                "render_metrics": str(render_path),
            }
    for method, _ in METHODS:
        reconstruction = results["reconstruction_7to1"][method]
        future = results["future_80to20"][method]
        if reconstruction["frames"] != len(holdout_frames):
            raise ValueError("前80%的7:1重建帧数不正确")
        if future["frames"] != len(future_frames):
            raise ValueError("后20%的未来预测帧数不正确")
        if reconstruction["evaluation_points"] != future["evaluation_points"]:
            raise ValueError("重建和未来预测使用了不同评估点")

    payload = {
        "schema": "fixedsuperbest.sim03_unified_80_20_7to1.v1",
        "protocol": {
            "tracker": "AllTracker official dense RGB flow",
            "depth": "FoundationStereo estimated from stereo RGB",
            "methods": [method for method, _ in METHODS],
            "runs_per_method": 1,
            "single_rollout_per_method": True,
            "frame_count": frame_count,
            "assimilation_interval": [0, args.future_start],
            "reconstruction_holdout_stride": 8,
            "reconstruction_holdout_offset": 7,
            "reconstruction_holdout_frames": holdout_frames,
            "future_open_loop_interval": [args.future_start, frame_count],
            "future_frames": future_frames,
            "no_future_rgb_or_parameter_updates": True,
            "evaluation_points": "30 fixed non-grasp nodes",
            "no_best_run_selection": True,
        },
        "observation_assets": {
            "report": str(observation_report_path),
            "requested_track_count": int(report["requested_track_count"]),
            "bound_track_count": int(report["bound_track_count"]),
        },
        "source_rollouts": source_rollouts,
        "results": results,
    }
    json_path = root / "comparison_unified.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# SIM-03 AllTracker：单rollout联合评估",
        "",
        "- B、C各只运行一次完整360帧物理rollout。",
        "- 前80%为帧0–287，其中每8帧留出第8帧，共36帧评估7:1重建。",
        "- 后20%为帧288–359，共72帧；关闭视觉校正和刚度更新后评估开环未来预测。",
        "- 两项指标来自同一rollout，不再分别初始化和运行两次。",
        "- 使用30个固定非夹持评估点；深度来自FoundationStereo RGB双目估计。",
        "",
        "| 能力 | 方法 | 帧数 | 3D Tracking (mm)↓ | 2D Tracking (px)↓ | PSNR (dB)↑ | SSIM↑ | LPIPS↓ |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for capability, capability_label in CAPABILITIES:
        for method, method_label in METHODS:
            row = results[capability][method]
            lines.append(
                f"| {capability_label} | {method_label} | {row['frames']} | "
                f"{row['3d_mean_mm']:.4f} | {row['2d_mean_px']:.3f} | "
                f"{row['psnr_db']:.3f} | {row['ssim']:.4f} | {row['lpips']:.4f} |"
            )
    lines.append("")
    md_path = root / "comparison_unified.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"写入：{md_path}", flush=True)


if __name__ == "__main__":
    main()
