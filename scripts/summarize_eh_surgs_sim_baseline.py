#!/usr/bin/env python3
"""Validate and summarize three EH-SurGS repeats on all fixed SIM datasets."""

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
EXPECTED_COMMIT = "73fa04e6f5c21cc1685f728eccb1332e81ce620c"
EXPECTED_DECODER = "som_query_anchored_displacement_v2"


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


def expected_frames(frame_count: int, capability: str):
    split = frame_count * 4 // 5
    if capability == "reconstruction_7to1":
        return [frame for frame in range(split) if frame % 8 == 7]
    return list(range(split, frame_count))


def read_run(run: Path, dataset_key: str, frame_count: int, seed: int) -> dict:
    status = (run / "status.txt").read_text(encoding="utf-8")
    if "status=complete" not in status or "seed={}".format(seed) not in status:
        raise ValueError("运行状态或 seed 不匹配：{}".format(run))
    audit = load(run / "protocol_audit.json")
    provenance = load(run / "artifacts" / "provenance.json")
    if not audit.get("passed") or audit.get("dataset_key") != dataset_key:
        raise ValueError("协议审计不匹配：{}".format(run))
    if provenance.get("eh_surgs_commit") != EXPECTED_COMMIT:
        raise ValueError("EH-SurGS commit 不匹配：{}".format(run))
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
        trajectory, rendering = load(trajectory_path), load(render_path)
        wanted = expected_frames(frame_count, capability)
        if trajectory["frame_selection"]["selected_frame_indices"] != wanted:
            raise ValueError("轨迹评估帧不符合协议：{}".format(run))
        if rendering["frame_selection"]["selected_frame_indices"] != wanted:
            raise ValueError("渲染评估帧不符合协议：{}".format(run))
        if int(trajectory["evaluated_nodes"]) != 30:
            raise ValueError("没有使用固定 30 点：{}".format(run))
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


def summarize(values):
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "values": values,
    }


def format_metric(metric: str, value: dict) -> str:
    digits = 4 if metric in {"ssim", "lpips", "coverage"} else 3
    return "{value:.{digits}f} ± {std:.{digits}f}".format(
        value=value["mean"], std=value["sample_std"], digits=digits
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    repeats = ("repeat_01", "repeat_02", "repeat_03")
    raw, aggregate = {}, {}
    for dataset_key, dataset_label, frame_count in DATASETS:
        raw[dataset_key] = {}
        for repeat_index, repeat in enumerate(repeats):
            raw[dataset_key][repeat] = read_run(
                root / "runs" / repeat / dataset_key,
                dataset_key,
                frame_count,
                repeat_index,
            )
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
        "schema": "fixedsuperbest.eh_surgs_three_dataset_three_repeat.v1",
        "method": "EH-SurGS pinned upstream + Shape-of-Motion-style decoder",
        "eh_surgs_commit": EXPECTED_COMMIT,
        "trajectory_decoder_version": EXPECTED_DECODER,
        "protocol": {
            "training": "only first 80% non-holdout frames; two calibrated stereo views",
            "reconstruction": "offline interpolation on stride=8 offset=7 holdouts",
            "future": "frozen temporal field extrapolated over final 20%; no future observations or tool control",
            "evaluation_points": "30 immutable non-grasp nodes queried at stereo_left frame 0 after checkpoint freeze",
            "alignment": "none",
            "seeds": [0, 1, 2],
            "aggregation": "arithmetic mean and sample standard deviation; no best-run selection",
        },
        "raw": raw,
        "aggregate": aggregate,
    }
    json_path = root / "comparison_mean_std.json"
    markdown_path = root / "comparison_mean_std.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("拒绝覆盖已有 EH-SurGS 汇总")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# EH-SurGS 仿真 baseline：三数据集三次独立评测",
        "",
        "固定官方 commit `{}`，seeds 0/1/2；均值与样本标准差，不选最优 run。".format(EXPECTED_COMMIT),
        "",
        "后 20% 是 prefix-trained 时间场零样本外推，不等同于受工具控制的物理 rollout。",
        "",
    ]
    for capability, capability_label in CAPABILITIES:
        lines.extend([
            "## {}".format(capability_label),
            "",
            "| 数据集 | n | 3D Tracking (mm)↓ | 2D Tracking (px)↓ | PSNR (dB)↑ | SSIM↑ | LPIPS↓ | 3D Coverage↑ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for dataset_key, dataset_label, _ in DATASETS:
            metrics = aggregate[dataset_key][capability]["metrics"]
            values = [format_metric(metric, metrics[metric]) for metric, _ in METRICS]
            lines.append("| {} | 3 | {} |".format(dataset_label, " | ".join(values)))
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print("写入：{}".format(markdown_path), flush=True)


if __name__ == "__main__":
    main()
