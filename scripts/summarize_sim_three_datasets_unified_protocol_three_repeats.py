#!/usr/bin/env python3
"""严格校验并汇总三个数据集新联合协议的三次独立结果。"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


DATASETS = (
    ("sim01", "SIM-01 Planar-X-Pull", "tissue_retraction_free_support_front_v2"),
    ("sim02", "SIM-02 Planar-Y-Pull", "tissue_retraction_free_support_side_v2"),
    ("sim03", "SIM-03 Edge-Z-Lift", "tissue_long_edge_lift_return_sufia_v2_lift30mm"),
)
METHODS = (
    ("pbd", "A：纯 PBD"),
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
    parser.add_argument("--sim03-existing-root", type=Path, required=True)
    parser.add_argument("--asset-base", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def weighted_2d(metric: dict) -> float:
    eyes = metric["2d"]
    count = sum(int(value["count"]) for value in eyes.values())
    return sum(
        float(value["mean"]) * int(value["count"]) for value in eyes.values()
    ) / max(count, 1)


def source_root(
    root: Path, existing: Path, repeat: str, dataset_id: str, method: str
) -> Path:
    if dataset_id == "sim03" and repeat == "repeat_01" and method != "pbd":
        return existing
    return root / "runs" / repeat / dataset_id


def validate_and_read(
    evaluation_root: Path,
    dataset: Path,
    method: str,
    frame_count: int,
) -> dict[str, dict]:
    future_start = frame_count * 4 // 5
    holdout_frames = list(range(7, future_start, 8))
    future_frames = list(range(future_start, frame_count))
    artifact_root = evaluation_root / "methods" / method / "artifacts"
    metadata = load(artifact_root / "artifact_metadata.json")
    if Path(metadata["reference_dataset"]).resolve() != dataset:
        raise ValueError(f"rollout数据集错误：{artifact_root}")
    if (
        int(metadata["frames"]) != frame_count
        or int(metadata["frame_start"]) != 0
        or int(metadata["frame_end_inclusive"]) != frame_count - 1
    ):
        raise ValueError(f"rollout不是完整{frame_count}帧：{artifact_root}")
    if int(metadata["open_loop_start_frame"]) != future_start:
        raise ValueError(f"80/20分界错误：{artifact_root}")
    if metadata["holdout_frame_indices"] != holdout_frames:
        raise ValueError(f"前80%的7:1留出帧错误：{artifact_root}")
    if metadata["render_frame_mode"] != "holdout_future":
        raise ValueError(f"没有使用holdout_future联合渲染：{artifact_root}")

    output: dict[str, dict] = {}
    for capability, _ in CAPABILITIES:
        metric_root = evaluation_root / "metrics" / capability / method
        trajectory_path = metric_root / "trajectory_metrics.json"
        render_path = metric_root / "render_metrics.json"
        trajectory = load(trajectory_path)
        rendering = load(render_path)["summary"]["all"]
        expected_frames = holdout_frames if capability == "reconstruction_7to1" else future_frames
        if int(trajectory["frames"]) != len(expected_frames):
            raise ValueError(f"评估帧数错误：{trajectory_path}")
        if int(trajectory["evaluated_nodes"]) != 30:
            raise ValueError(f"没有使用30个固定非夹持点：{trajectory_path}")
        selected = trajectory["frame_selection"]["selected_frame_indices"]
        if selected != expected_frames:
            raise ValueError(f"评估帧序列错误：{trajectory_path}")
        output[capability] = {
            "frames": int(trajectory["frames"]),
            "3d_mean_mm": float(trajectory["3d"]["mean"]),
            "2d_mean_px": weighted_2d(trajectory),
            "psnr_db": float(rendering["psnr_tissue_layer_db"]),
            "ssim": float(rendering["ssim_tissue_layer"]),
            "lpips": float(rendering["lpips_tissue_layer_alex_v0.1"]),
            "trajectory_metrics": str(trajectory_path),
            "render_metrics": str(render_path),
            "rollout": str(artifact_root / "predicted_trajectories.npz"),
        }
    return output


def summarize(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "values": values,
    }


def fmt(metric: str, summary: dict) -> str:
    digits = 4 if metric in {"ssim", "lpips"} else 3
    return f"{summary['mean']:.{digits}f} ± {summary['sample_std']:.{digits}f}"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    existing = args.sim03_existing_root.resolve()
    asset_base = args.asset_base.resolve()
    repo = Path(__file__).resolve().parents[1]
    repeat_names = ["repeat_01", "repeat_02", "repeat_03"]

    reports = {}
    for dataset_id, _, _ in DATASETS:
        report_path = asset_base / dataset_id / "flow_depth_assets" / "report.json"
        report = load(report_path)
        if "alltracker" not in str(report.get("rgb_motion_source", "")).lower():
            raise ValueError(f"不是AllTracker观测资产：{report_path}")
        if report.get("depth_estimated_from_rgb") is not True:
            raise ValueError(f"不是RGB估计深度：{report_path}")
        reports[dataset_id] = {
            "path": str(report_path),
            "requested_track_count": int(report["requested_track_count"]),
            "bound_track_count": int(report["bound_track_count"]),
        }

    raw: dict = {}
    sources: dict = {}
    for dataset_id, dataset_label, dataset_dir in DATASETS:
        dataset = (repo / "data" / "sim" / dataset_dir).resolve()
        frame_count = int(load(dataset / "episode.json")["frames"])
        raw[dataset_id] = {}
        sources[dataset_id] = {}
        for repeat in repeat_names:
            raw[dataset_id][repeat] = {}
            sources[dataset_id][repeat] = {}
            for method, _ in METHODS:
                evaluation_root = source_root(root, existing, repeat, dataset_id, method)
                raw[dataset_id][repeat][method] = validate_and_read(
                    evaluation_root, dataset, method, frame_count
                )
                sources[dataset_id][repeat][method] = str(evaluation_root)

    aggregate: dict = {}
    for dataset_id, dataset_label, _ in DATASETS:
        aggregate[dataset_id] = {}
        for capability, capability_label in CAPABILITIES:
            aggregate[dataset_id][capability] = {}
            for method, method_label in METHODS:
                metric_summaries = {}
                for metric, _ in METRICS:
                    values = [
                        raw[dataset_id][repeat][method][capability][metric]
                        for repeat in repeat_names
                    ]
                    metric_summaries[metric] = summarize(values)
                aggregate[dataset_id][capability][method] = {
                    "dataset_label": dataset_label,
                    "capability_label": capability_label,
                    "method_label": method_label,
                    "repeat_count": 3,
                    "metrics": metric_summaries,
                }

    payload = {
        "schema": "fixedsuperbest.three_datasets_unified_protocol_three_repeats.v1",
        "protocol": {
            "tracker": "AllTracker official dense RGB flow",
            "depth": "FoundationStereo estimated independently from each dataset stereo RGB",
            "repeat_count_per_dataset_method": 3,
            "single_rollout_per_repeat_and_method": True,
            "reconstruction": "first 80%, 7:1 held-out frames (stride=8, offset=7)",
            "future_prediction": "last 20% open loop without RGB or stiffness updates",
            "evaluation_points": "30 fixed non-grasp nodes per dataset",
            "no_best_run_selection": True,
            "sim03_repeat_01_BC_reused": str(existing),
        },
        "observation_assets": reports,
        "sources": sources,
        "raw": raw,
        "aggregate": aggregate,
    }
    (root / "comparison_mean_std.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# 三数据集统一协议三次独立评测",
        "",
        "- 每个方法每次只运行一个连续rollout；前80%同化并以7:1留出帧评估重建，后20%冻结RGB校正和刚度更新后开环预测。",
        "- SIM01/02的A/B/C各新运行三次；SIM03的A新运行三次，B/C将已完成的一次作为第1次并新增两次。",
        "- 三次全部进入均值和样本标准差，不选择最佳运行。",
        "- 每套数据独立使用自己的AllTracker、FoundationStereo和30个固定非夹持评估点。",
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
        for dataset_id, dataset_label, _ in DATASETS:
            for method, method_label in METHODS:
                row = aggregate[dataset_id][capability][method]
                cells = [fmt(metric, row["metrics"][metric]) for metric, _ in METRICS]
                lines.append(
                    f"| {dataset_label} | {method_label} | 3 | " + " | ".join(cells) + " |"
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
        for dataset_id, dataset_label, _ in DATASETS:
            for capability, capability_label in CAPABILITIES:
                for method, method_label in METHODS:
                    row = raw[dataset_id][repeat][method][capability]
                    lines.append(
                        f"| {repeat_index} | {dataset_label} | {capability_label} | {method_label} | "
                        f"{row['3d_mean_mm']:.4f} | {row['2d_mean_px']:.3f} | "
                        f"{row['psnr_db']:.3f} | {row['ssim']:.4f} | {row['lpips']:.4f} |"
                    )
    lines.append("")
    (root / "comparison_mean_std.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"写入：{root / 'comparison_mean_std.md'}", flush=True)


if __name__ == "__main__":
    main()
