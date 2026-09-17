#!/usr/bin/env python3
"""Validate and summarize three TRACE repeats on all fixed SIM datasets."""

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
    ("future_80to20", "后20%零样本动力学外推"),
)
METRICS = (
    ("3d_mean_mm", "3D Tracking (mm)↓"),
    ("2d_mean_px", "2D Tracking (px)↓"),
    ("psnr_db", "PSNR (dB)↑"),
    ("ssim", "SSIM↑"),
    ("lpips", "LPIPS↓"),
    ("coverage", "3D Coverage↑"),
)
EXPECTED_COMMIT = "a4597585bc0e56c56922abe75be9198eb119c95a"
EXPECTED_DECODER = "query_anchored_gaussian_displacement_v1"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def weighted_2d(trajectory: dict) -> float:
    cameras = trajectory["2d"]
    count = sum(int(value["count"]) for value in cameras.values())
    if count == 0:
        raise ValueError("2D valid count is zero")
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
        raise ValueError("run status/seed mismatch: {}".format(run))
    audit = load(run / "protocol_audit.json")
    provenance = load(run / "artifacts" / "provenance.json")
    if not audit.get("passed") or audit.get("dataset_key") != dataset_key:
        raise ValueError("protocol audit mismatch: {}".format(run))
    if provenance.get("trace_commit") != EXPECTED_COMMIT:
        raise ValueError("TRACE commit mismatch: {}".format(run))
    if provenance.get("external_psm_control") is not False:
        raise ValueError("TRACE run does not certify no-PSM evaluation: {}".format(run))
    tracks = provenance["tracks"]
    if tracks.get("decoder_version") != EXPECTED_DECODER:
        raise ValueError("trajectory decoder mismatch: {}".format(run))
    if int(tracks["query_point_count"]) != 30 or int(tracks["query_valid_count"]) != 30:
        raise ValueError("query coverage is not 30/30: {}".format(run))
    if float(tracks.get("initial_query_reprojection_max_px", float("inf"))) > 1.0e-3:
        raise ValueError("query anchoring failed: {}".format(run))
    output = {}
    for capability, _ in CAPABILITIES:
        metric_root = run / "metrics" / capability
        trajectory_path = metric_root / "trajectory_metrics.json"
        render_path = metric_root / "render_metrics.json"
        trajectory, rendering = load(trajectory_path), load(render_path)
        wanted = expected_frames(frame_count, capability)
        if trajectory["frame_selection"]["selected_frame_indices"] != wanted:
            raise ValueError("trajectory frames mismatch: {}".format(run))
        if rendering["frame_selection"]["selected_frame_indices"] != wanted:
            raise ValueError("render frames mismatch: {}".format(run))
        if int(trajectory["evaluated_nodes"]) != 30:
            raise ValueError("evaluation did not use the frozen 30 points: {}".format(run))
        render_all = rendering["summary"]["all"]
        output[capability] = {
            "frames": len(wanted),
            "3d_mean_mm": float(trajectory["3d"]["mean"]),
            "2d_mean_px": weighted_2d(trajectory),
            "psnr_db": float(render_all["psnr_tissue_layer_db"]),
            "ssim": float(render_all["ssim_tissue_layer"]),
            "lpips": float(render_all["lpips_tissue_layer_alex_v0.1"]),
            "psnr_full_db": float(render_all["psnr_full_db"]),
            "ssim_full": float(render_all["ssim_full"]),
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
                "full_scene_rendering": {
                    metric: summarize(
                        [raw[dataset_key][repeat][capability][metric] for repeat in repeats]
                    )
                    for metric in ("psnr_full_db", "ssim_full")
                },
            }
    payload = {
        "schema": "fixedsuperbest.trace_three_dataset_three_repeat.v1",
        "method": "official TRACE core + full-K RGB-only SIM adapter",
        "trace_commit": EXPECTED_COMMIT,
        "trajectory_decoder_version": EXPECTED_DECODER,
        "protocol": {
            "training": "legal first-80% non-holdout stereo RGB and calibrated cameras only",
            "forbidden": "PSM, depth, masks, GT trajectories, held-out/future RGB",
            "reconstruction": "stride=8 offset=7 held-out interpolation",
            "future": "native TRACE translation-rotation dynamics; no future observations/control",
            "alignment": "none",
            "seeds": [0, 1, 2],
            "aggregation": "arithmetic mean and sample standard deviation; no best-run selection",
        },
        "render_metric_note": (
            "TRACE represents the full scene and exports full opacity. Existing tissue-layer "
            "metrics are retained unchanged for protocol continuity; full-scene PSNR/SSIM are "
            "also preserved because tissue-only alpha is unavailable without adding segmentation."
        ),
        "raw": raw,
        "aggregate": aggregate,
    }
    summary = root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    json_path = summary / "summary.json"
    markdown_path = summary / "summary.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("refusing to overwrite TRACE summary")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# TRACE SIM 基线：三数据集 × 三种子",
        "",
        "官方提交 `{}`；训练只使用 RGB 与相机标定，不使用 PSM、深度、mask、对齐或最优运行筛选。".format(EXPECTED_COMMIT),
        "",
        "TRACE 渲染完整场景。为保持协议连续性，仍报告未经修改的组织层指标；完整场景 PSNR/SSIM 保存在 `summary.json` 中。",
        "",
    ]
    for capability, label in CAPABILITIES:
        lines.extend([
            "## {}".format(label),
            "",
            "| 数据集 | n | 3D (mm)↓ | 2D (px)↓ | PSNR (dB)↑ | SSIM↑ | LPIPS↓ | 3D 覆盖率↑ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for dataset_key, dataset_label, _ in DATASETS:
            metrics = aggregate[dataset_key][capability]["metrics"]
            values = [format_metric(metric, metrics[metric]) for metric, _ in METRICS]
            lines.append("| {} | 3 | {} |".format(dataset_label, " | ".join(values)))
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote {}".format(markdown_path), flush=True)


if __name__ == "__main__":
    main()
