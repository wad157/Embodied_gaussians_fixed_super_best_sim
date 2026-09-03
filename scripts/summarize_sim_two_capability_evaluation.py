#!/usr/bin/env python3
"""汇总 3 方法 ×（7:1 重建、80/20 未来预测）六个正式实验。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


METHODS = (
    ("pbd", "PBD"),
    ("pbd_visual_residual", "PBD + 视觉残差"),
    (
        "pbd_visual_residual_stiffness",
        "PBD + 视觉残差 + 刚度修正",
    ),
)
CAPABILITIES = (
    ("reconstruction_7to1", "重建（7:1 留出测试）"),
    ("future_80to20", "未来预测（后20%开环）"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def weighted_stereo_mean(cameras: dict, field: str) -> float | None:
    numerator = 0.0
    denominator = 0
    for camera in ("stereo_left", "stereo_right"):
        metrics = cameras[camera]
        value = metrics.get(field)
        count = int(metrics.get("count", 0))
        if value is not None and count:
            numerator += float(value) * count
            denominator += count
    return numerator / denominator if denominator else None


def pooled_stereo_rmse(cameras: dict) -> float | None:
    squared_error_sum = 0.0
    denominator = 0
    for camera in ("stereo_left", "stereo_right"):
        metrics = cameras[camera]
        value = metrics.get("rmse")
        count = int(metrics.get("count", 0))
        if value is not None and count:
            squared_error_sum += float(value) ** 2 * count
            denominator += count
    return math.sqrt(squared_error_sum / denominator) if denominator else None


def fmt(value: object, digits: int) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def validate_protocol(
    capability: str,
    trajectory: dict,
    rendering: dict,
    artifact: dict,
) -> None:
    trajectory_frames = trajectory["frame_selection"]["selected_frame_indices"]
    rendering_frames = rendering["frame_selection"]["selected_frame_indices"]
    if trajectory_frames != rendering_frames:
        raise ValueError(f"{capability} 的轨迹与渲染计分帧不一致")
    if capability == "reconstruction_7to1":
        expected = list(range(7, 360, 8))
        if (
            trajectory_frames != expected
            or artifact.get("holdout_stride") != 8
            or artifact.get("holdout_offset") != 7
            or artifact.get("open_loop_start_frame") is not None
            or artifact.get("render_frame_mode") != "holdout"
        ):
            raise ValueError("重建实验不满足因果 7:1 留出协议")
    elif capability == "future_80to20":
        expected = list(range(288, 360))
        if (
            trajectory_frames != expected
            or artifact.get("holdout_stride") is not None
            or artifact.get("open_loop_start_frame") != 288
            or artifact.get("render_frame_mode") != "future"
        ):
            raise ValueError("未来实验不满足连续 80/20 开环协议")


def main() -> None:
    args = parse_args()
    root = args.evaluation_root.expanduser().resolve()
    outputs = (
        root / "comparison.json",
        root / "comparison.csv",
        root / "comparison.md",
    )
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"拒绝覆盖已有六实验汇总：{existing}")

    rows: list[dict[str, object]] = []
    inputs: dict[str, dict[str, object]] = {}
    for capability_key, capability_name in CAPABILITIES:
        inputs[capability_key] = {}
        for method_key, method_name in METHODS:
            method_root = root / capability_key / method_key
            trajectory_path = method_root / "trajectory_metrics.json"
            rendering_path = method_root / "render_metrics.json"
            artifact_path = method_root / "artifacts/artifact_metadata.json"
            trajectory = read_json(trajectory_path)
            rendering = read_json(rendering_path)
            artifact = read_json(artifact_path)
            validate_protocol(
                capability_key, trajectory, rendering, artifact
            )
            track = trajectory["segments"]["all"]
            render = rendering["segments"]["all"]["all"]
            cameras = track["2d"]
            row = {
                "capability": capability_key,
                "capability_zh": capability_name,
                "variant": method_key,
                "method": method_name,
                "frames": int(track["frame_count"]),
                "track_3d_mean_mm": track["3d"].get("mean"),
                "track_3d_rmse_mm": track["3d"].get("rmse"),
                "track_3d_p95_mm": track["3d"].get("p95"),
                "track_2d_stereo_mean_px": weighted_stereo_mean(
                    cameras, "mean"
                ),
                "track_2d_stereo_rmse_px": pooled_stereo_rmse(cameras),
                "psnr_tissue_layer_db": render.get(
                    "psnr_tissue_layer_db"
                ),
                "ssim_tissue_layer": render.get("ssim_tissue_layer"),
                "lpips_tissue_layer": render.get(
                    "lpips_tissue_layer_alex_v0.1"
                ),
            }
            rows.append(row)
            inputs[capability_key][method_key] = {
                "trajectory": str(trajectory_path),
                "rendering": str(rendering_path),
                "artifact": str(artifact_path),
            }

    report = {
        "schema": "fixedsuperbest.two_capability_ablation.v1",
        "evaluation_root": str(root),
        "experiment_count": 6,
        "reconstruction_protocol": (
            "causal 7:1: frames 7,15,...,359 are held out from RGB residual "
            "and stiffness learning and used only for 2D/3D/render evaluation"
        ),
        "future_protocol": (
            "frames 0..287 estimate state/material; frames 288..359 freeze "
            "RGB residual and stiffness and are evaluated open loop"
        ),
        "evaluation_points": (
            "the same fixed 10 non-grasp tissue points for every run"
        ),
        "rows": rows,
        "inputs": inputs,
    }
    outputs[0].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with outputs[1].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 重建与未来预测：六实验正式消融",
        "",
        "三种方法分别独立运行 7:1 重建和 80/20 未来预测，共六次仿真。",
        "重建测试帧不进入视觉残差或刚度更新；未来72帧完全冻结两项更新。",
        "所有实验使用相同10个非夹持关键点和相同初始材料参数。",
        "",
        "| 能力 | 方法 | 帧数 | 3D Mean (mm) ↓ | 3D RMSE ↓ | 3D P95 ↓ | 双目2D Mean (px) ↓ | 双目2D RMSE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {capability} | {method} | {frames} | {m3d} | {r3d} | "
            "{p3d} | {m2d} | {r2d} | {psnr} | {ssim} | {lpips} |".format(
                capability=row["capability_zh"],
                method=row["method"],
                frames=row["frames"],
                m3d=fmt(row["track_3d_mean_mm"], 3),
                r3d=fmt(row["track_3d_rmse_mm"], 3),
                p3d=fmt(row["track_3d_p95_mm"], 3),
                m2d=fmt(row["track_2d_stereo_mean_px"], 3),
                r2d=fmt(row["track_2d_stereo_rmse_px"], 3),
                psnr=fmt(row["psnr_tissue_layer_db"], 3),
                ssim=fmt(row["ssim_tissue_layer"], 4),
                lpips=fmt(row["lpips_tissue_layer"], 4),
            )
        )
    lines.extend(
        (
            "",
            "说明：轨迹指标只使用固定10点；本次所选已知夹持核心边界及其排除区不计分。",
            "渲染指标比较组织层，并惩罚落在真值组织轮廓之外的预测。",
            "",
        )
    )
    outputs[2].write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
