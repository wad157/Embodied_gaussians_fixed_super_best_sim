#!/usr/bin/env python3
"""Validate and summarize the requested EG-Soft SIM repeats without cherry-picking."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np


UPSTREAM_COMMIT = "c97ec671f97af25985e0af8844c0aac8d8119b97"
DATASETS = {
    "sim01": {"label": "SIM-01 Planar-X-Pull", "frames": 300, "repeats": 2},
    "sim02": {"label": "SIM-02 Planar-Y-Pull", "frames": 300, "repeats": 3},
    "sim03": {"label": "SIM-03 Edge-Z-Lift", "frames": 360, "repeats": 3},
}
CAPABILITIES = {
    "reconstruction_7to1": "前80%在线7:1重建",
    "future_80to20": "后20%开环物理预测",
}
METRICS = (
    ("3d_mean_mm", "3D Tracking (mm)↓", 3),
    ("2d_mean_px", "2D Tracking (px)↓", 3),
    ("2d_valid_fraction", "2D Valid Coverage↑", 4),
    ("psnr_db", "PSNR (dB)↑", 3),
    ("ssim", "SSIM↑", 4),
    ("lpips", "LPIPS↓", 4),
    ("3d_valid_fraction", "3D Valid Coverage↑", 4),
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_frames(frame_count: int, capability: str) -> list[int]:
    split = frame_count * 4 // 5
    if capability == "reconstruction_7to1":
        return [frame for frame in range(split) if frame % 8 == 7]
    return list(range(split, frame_count))


def stereo_2d(trajectory: dict) -> tuple[float | None, int, int]:
    cameras = trajectory["2d"]
    valid = sum(int(value["count"]) for value in cameras.values())
    reference = sum(
        int(value["reference_visible_count"]) for value in cameras.values()
    )
    if reference <= 0:
        raise ValueError("2D reference visible count 非法")
    mean = None
    if valid:
        mean = sum(
            float(value["mean"]) * int(value["count"])
            for value in cameras.values()
        ) / valid
    return mean, valid, reference


def read_run(run: Path, dataset_key: str, frame_count: int, seed: int) -> dict:
    status = (run / "status.txt").read_text(encoding="utf-8")
    if "status=complete" not in status or f"seed={seed}" not in status:
        raise ValueError(f"运行状态或 seed 不匹配：{run}")
    audit = load(run / "protocol_audit.json")
    body = load(run / "initialization" / "body.metadata.json")
    rollout = load(run / "artifacts" / "rollout_metadata.json")
    tracks = load(run / "artifacts" / "predicted_trajectories.metadata.json")
    if not audit.get("passed") or audit.get("dataset_key") != dataset_key:
        raise ValueError(f"协议审计不匹配：{run}")
    if audit.get("upstream", {}).get("commit") != UPSTREAM_COMMIT:
        raise ValueError(f"上游 commit 不匹配：{run}")
    if not body.get("formal") or body.get("seed") != seed:
        raise ValueError(f"正式初始化或 seed 不匹配：{run}")
    if body.get("initialization_cameras") != ["stereo_left", "stereo_right"]:
        raise ValueError(f"初始化不是严格双目：{run}")
    if body.get("initialization_frame") != 0:
        raise ValueError(f"初始化不是 frame 0：{run}")
    if not rollout.get("formal") or rollout.get("seed") != seed:
        raise ValueError(f"正式 rollout 或 seed 不匹配：{run}")
    if rollout.get("actuation") != "psm-fk-collision-only":
        raise ValueError(f"主结果不是 collision-only：{run}")
    if int(rollout.get("frame_count", -1)) != frame_count:
        raise ValueError(f"rollout 帧数错误：{run}")
    if len(tracks.get("selected_parent_particle_ids", [])) != 30:
        raise ValueError(f"轨迹导出不是30点：{run}")
    if tracks.get("query_gt_depth_used") or tracks.get("query_gt_3d_used"):
        raise ValueError(f"轨迹导出使用了评测GT几何：{run}")
    if float(tracks.get("initial_query_reprojection_max_px", math.inf)) > 1.0e-3:
        raise ValueError(f"frame-0 查询没有严格锚定：{run}")
    with np.load(run / "artifacts" / "predicted_trajectories.npz", allow_pickle=False) as archive:
        positions = np.asarray(archive["tissue_positions_world"])
        if positions.shape != (frame_count, 30, 3):
            raise ValueError(f"预测轨迹维度错误：{run}")
        if not np.isfinite(positions).all():
            raise ValueError(f"预测轨迹包含非有限值：{run}")

    result = {
        "seed": seed,
        "particle_count": int(body["particle_count"]),
        "gaussian_count": int(body["gaussian_count"]),
        "query_raster_coverage_count": int(tracks["query_raster_coverage_count"]),
        "query_fallback_count": int(
            tracks["query_projected_nearest_gaussian_fallback_count"]
        ),
        "capabilities": {},
    }
    for capability in CAPABILITIES:
        metric_root = run / "metrics" / capability
        trajectory = load(metric_root / "trajectory_metrics.json")
        rendering = load(metric_root / "render_metrics.json")
        wanted = expected_frames(frame_count, capability)
        if trajectory["frame_selection"]["selected_frame_indices"] != wanted:
            raise ValueError(f"轨迹帧选择错误：{run} {capability}")
        if rendering["frame_selection"]["selected_frame_indices"] != wanted:
            raise ValueError(f"渲染帧选择错误：{run} {capability}")
        if int(trajectory["evaluated_nodes"]) != 30:
            raise ValueError(f"评测节点数错误：{run} {capability}")
        two_d_mean, two_d_valid, two_d_reference = stereo_2d(trajectory)
        render_all = rendering["summary"]["all"]
        result["capabilities"][capability] = {
            "frames": len(wanted),
            "3d_mean_mm": float(trajectory["3d"]["mean"]),
            "2d_mean_px": two_d_mean,
            "2d_valid_count": two_d_valid,
            "2d_reference_visible_count": two_d_reference,
            "2d_valid_fraction": two_d_valid / two_d_reference,
            "psnr_db": float(render_all["psnr_tissue_layer_db"]),
            "ssim": float(render_all["ssim_tissue_layer"]),
            "lpips": float(render_all["lpips_tissue_layer_alex_v0.1"]),
            "3d_valid_fraction": float(trajectory["3d"]["valid_fraction"]),
        }
    return result


def summarize(values: list[float | None]) -> dict:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    # A requested-repeat mean is undefined if even one constituent run has no
    # valid samples.  Never hide a failed/out-of-view run by averaging only the
    # surviving runs; coverage is reported as its own always-defined metric.
    if len(finite) != len(values):
        return {
            "count": len(finite),
            "mean": None,
            "sample_std": None,
            "values": values,
        }
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "sample_std": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "values": values,
    }


def formatted(summary: dict, digits: int) -> str:
    if summary["mean"] is None:
        return "N/A"
    return f"{summary['mean']:.{digits}f} ± {summary['sample_std']:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    raw: dict[str, dict[str, object]] = {}
    aggregate: dict[str, dict[str, object]] = {}
    for dataset_key, spec in DATASETS.items():
        repeat_names = [f"repeat_{index:02d}" for index in range(1, spec["repeats"] + 1)]
        raw[dataset_key] = {}
        for seed, repeat in enumerate(repeat_names):
            raw[dataset_key][repeat] = read_run(
                root / repeat / dataset_key,
                dataset_key,
                int(spec["frames"]),
                seed,
            )
        aggregate[dataset_key] = {}
        for capability, label in CAPABILITIES.items():
            aggregate[dataset_key][capability] = {
                "dataset_label": spec["label"],
                "capability_label": label,
                "repeat_count": len(repeat_names),
                "metrics": {
                    metric: summarize(
                        [
                            raw[dataset_key][repeat]["capabilities"][capability][metric]
                            for repeat in repeat_names
                        ]
                    )
                    for metric, _, _ in METRICS
                },
            }

    payload = {
        "schema": "fixedsuperbest.embodied_gaussians_requested_repeat_summary.v1",
        "method": "Embodied Gaussians Soft paper reconstruction; PSM/FK collision-only",
        "upstream_commit": UPSTREAM_COMMIT,
        "protocol": {
            "sim01": "seeds 0/1; two-run arithmetic mean and sample standard deviation",
            "sim02": "seeds 0/1/2; three-run arithmetic mean and sample standard deviation",
            "sim03": "seeds 0/1/2; three-run arithmetic mean and sample standard deviation",
            "selection": "all requested runs included; no best-run selection",
            "cameras": ["stereo_left", "stereo_right"],
            "actuation": "psm-fk-collision-only",
            "alignment": "none",
            "undefined_2d": "N/A when a run has zero valid projected samples; coverage reported separately",
        },
        "raw": raw,
        "aggregate": aggregate,
    }
    json_path = root / "requested_repeats_mean_std.json"
    csv_path = root / "requested_repeats_mean_std.csv"
    markdown_path = root / "requested_repeats_mean_std.md"
    for path in (json_path, csv_path, markdown_path):
        if path.exists():
            raise FileExistsError(f"拒绝覆盖已有汇总：{path}")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rows = []
    for dataset_key, spec in DATASETS.items():
        for capability, label in CAPABILITIES.items():
            entry = aggregate[dataset_key][capability]
            row = {
                "dataset": dataset_key,
                "capability": capability,
                "repeats": entry["repeat_count"],
            }
            for metric, _, _ in METRICS:
                value = entry["metrics"][metric]
                row[f"{metric}_mean"] = value["mean"]
                row[f"{metric}_sample_std"] = value["sample_std"]
            rows.append(row)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Embodied Gaussians–Soft requested SIM repeats",
        "",
        "SIM-01 使用 seeds 0/1；SIM-02/03 使用 seeds 0/1/2。算术均值与样本标准差，不挑最佳运行。",
        "",
        "| Dataset | Capability | Runs | "
        + " | ".join(label for _, label, _ in METRICS)
        + " |",
        "|---|---|---:|" + "---:|" * len(METRICS),
    ]
    for dataset_key, spec in DATASETS.items():
        for capability, capability_label in CAPABILITIES.items():
            entry = aggregate[dataset_key][capability]
            values = [
                formatted(entry["metrics"][metric], digits)
                for metric, _, digits in METRICS
            ]
            lines.append(
                f"| {spec['label']} | {capability_label} | {entry['repeat_count']} | "
                + " | ".join(values)
                + " |"
            )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "markdown": str(markdown_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
