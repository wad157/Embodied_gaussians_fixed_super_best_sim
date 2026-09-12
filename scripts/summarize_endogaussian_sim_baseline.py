#!/usr/bin/env python3
"""Validate and summarize three EndoGaussian repeats on all current SIM datasets."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


DATASETS = (
    ("sim01", "SIM-01 Planar-X-Pull", 300),
    ("sim02", "SIM-02 Planar-Y-Pull", 300),
    ("sim03", "SIM-03 Edge-Z-Lift", 360),
)
CAPABILITIES = (
    ("reconstruction_7to1", "前80%离线7:1重建"),
    ("future_80to20", "后20%零样本时间外推"),
)
METRICS = (
    ("3d_mean_mm", "3D Tracking (mm)↓"),
    ("2d_mean_px", "2D Tracking (px)↓"),
    ("psnr_db", "PSNR (dB)↑"),
    ("ssim", "SSIM↑"),
    ("lpips", "LPIPS↓"),
    ("coverage", "3D Coverage↑"),
)
EXPECTED_COMMIT = "8d12793838a1595b299df0696c8149c07329e980"
EXPECTED_DECODER = "som_query_anchored_displacement_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def weighted_2d(trajectory: dict) -> float:
    cameras = trajectory["2d"]
    count = sum(int(value["count"]) for value in cameras.values())
    if count == 0:
        raise ValueError("2D 有效点数量为 0")
    return sum(
        float(value["mean"]) * int(value["count"]) for value in cameras.values()
    ) / count


def expected_frames(frame_count: int, capability: str) -> list[int]:
    split = frame_count * 4 // 5
    if capability == "reconstruction_7to1":
        return [frame for frame in range(split) if frame % 8 == 7]
    return list(range(split, frame_count))


def read_run(run: Path, dataset_key: str, frame_count: int) -> dict:
    status = (run / "status.txt").read_text(encoding="utf-8")
    if "status=complete" not in status:
        raise ValueError("运行未完成：{}".format(run))
    audit = load(run / "protocol_audit.json")
    provenance = load(run / "artifacts" / "provenance.json")
    if not audit.get("passed") or audit.get("dataset_key") != dataset_key:
        raise ValueError("协议审计不匹配：{}".format(run))
    if provenance.get("endogaussian_commit") != EXPECTED_COMMIT:
        raise ValueError("EndoGaussian commit 不匹配：{}".format(run))
    tracks = provenance["tracks"]
    if tracks.get("decoder_version") != EXPECTED_DECODER:
        raise ValueError("轨迹解码版本错误：{}".format(run))
    if int(tracks["query_point_count"]) != 30 or int(tracks["query_valid_count"]) != 30:
        raise ValueError("查询点不是 30/30 完整覆盖：{}".format(run))
    if float(tracks.get("initial_query_reprojection_max_px", float("inf"))) > 1.0e-3:
        raise ValueError("frame-0 轨迹没有严格锚定查询像素：{}".format(run))
    output = {}
    for capability, _ in CAPABILITIES:
        metric_root = run / "metrics" / capability
        trajectory_path = metric_root / "trajectory_metrics.json"
        render_path = metric_root / "render_metrics.json"
        trajectory = load(trajectory_path)
        rendering = load(render_path)
        wanted = expected_frames(frame_count, capability)
        selected_trajectory = trajectory["frame_selection"]["selected_frame_indices"]
        selected_render = rendering["frame_selection"]["selected_frame_indices"]
        if selected_trajectory != wanted or selected_render != wanted:
            raise ValueError("评估帧不符合固定协议：{} {}".format(run, capability))
        if int(trajectory["evaluated_nodes"]) != 30:
            raise ValueError("没有使用固定 30 点：{}".format(trajectory_path))
        render_all = rendering["summary"]["all"]
        output[capability] = {
            "frames": len(wanted),
            "3d_mean_mm": float(trajectory["3d"]["mean"]),
            "2d_mean_px": weighted_2d(trajectory),
            "psnr_db": float(render_all["psnr_tissue_layer_db"]),
            "ssim": float(render_all["ssim_tissue_layer"]),
            "lpips": float(render_all["lpips_tissue_layer_alex_v0.1"]),
            "coverage": float(trajectory["3d"]["valid_fraction"]),
            "trajectory_metrics": str(trajectory_path),
            "render_metrics": str(render_path),
        }
    return output


def summarize(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "values": values,
    }


def format_metric(metric: str, value: dict) -> str:
    if metric == "coverage":
        return "{:.4f} ± {:.4f}".format(value["mean"], value["sample_std"])
    digits = 4 if metric in {"ssim", "lpips"} else 3
    return "{value:.{digits}f} ± {std:.{digits}f}".format(
        value=value["mean"], std=value["sample_std"], digits=digits
    )


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    repeats = ("repeat_01", "repeat_02", "repeat_03")
    raw = {}
    aggregate = {}
    for dataset_key, dataset_label, frame_count in DATASETS:
        raw[dataset_key] = {}
        for repeat in repeats:
            run = root / "runs" / repeat / dataset_key
            raw[dataset_key][repeat] = read_run(run, dataset_key, frame_count)
        aggregate[dataset_key] = {}
        for capability, capability_label in CAPABILITIES:
            aggregate[dataset_key][capability] = {
                "dataset_label": dataset_label,
                "capability_label": capability_label,
                "repeat_count": 3,
                "metrics": {
                    metric: summarize(
                        [raw[dataset_key][repeat][capability][metric] for repeat in repeats]
                    )
                    for metric, _ in METRICS
                },
            }
    payload = {
        "schema": "fixedsuperbest.endogaussian_three_dataset_three_repeat.v1",
        "method": "EndoGaussian pinned upstream + Shape-of-Motion-style decoder",
        "endogaussian_commit": EXPECTED_COMMIT,
        "trajectory_decoder_version": EXPECTED_DECODER,
        "protocol": {
            "training": "only first 80% non-holdout frames; two calibrated stereo views",
            "reconstruction": "offline interpolation on stride=8 offset=7 holdouts",
            "future": "same frozen temporal field extrapolated over final 20%; no future observations or tool control",
            "evaluation_points": "30 immutable non-grasp nodes queried at stereo_left frame 0 after checkpoint freeze; query pixels anchored with EndoGaussian-rendered depth",
            "alignment": "none",
            "repeat_count": 3,
            "aggregation": "arithmetic mean and sample standard deviation; no best-run selection",
        },
        "raw": raw,
        "aggregate": aggregate,
    }
    json_path = root / "comparison_mean_std.json"
    markdown_path = root / "comparison_mean_std.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("拒绝覆盖已有 baseline 汇总")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# EndoGaussian 仿真 baseline：三数据集三次独立评测",
        "",
        "固定官方 commit `{}`。轨迹使用 Shape of Motion 式特征栅格化解码；不训练 Shape of Motion，不做位姿、尺度或时间对齐。".format(EXPECTED_COMMIT),
        "",
        "后 20% 表示 prefix-trained EndoGaussian 时间场的零样本外推，不等同于受工具控制的物理开环 rollout。",
        "",
    ]
    for capability, capability_label in CAPABILITIES:
        lines.extend(
            [
                "## {}".format(capability_label),
                "",
                "| 数据集 | n | 3D Tracking (mm)↓ | 2D Tracking (px)↓ | PSNR (dB)↑ | SSIM↑ | LPIPS↓ | 3D Coverage↑ |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for dataset_key, dataset_label, _ in DATASETS:
            row = aggregate[dataset_key][capability]
            values = [format_metric(metric, row["metrics"][metric]) for metric, _ in METRICS]
            lines.append("| {} | 3 | {} |".format(dataset_label, " | ".join(values)))
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print("写入：{}".format(markdown_path), flush=True)


if __name__ == "__main__":
    main()
