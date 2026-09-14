#!/usr/bin/env python3
"""Validate and summarize three PhysTwin repeats on the fixed SIM protocol."""

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
    ("future_80to20", "后20%已知工具控制开环预测"),
)
METRICS = (
    ("3d_mean_mm", "3D Tracking (mm)↓"),
    ("2d_mean_px", "2D Tracking (px)↓"),
    ("psnr_db", "PSNR (dB)↑"),
    ("ssim", "SSIM↑"),
    ("lpips", "LPIPS↓"),
    ("coverage", "3D Coverage↑"),
)
EXPECTED_COMMIT = "81c718790a37e5e0102eb77af2c6edd34a9db25f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def weighted_2d(trajectory: dict) -> float:
    cameras = trajectory["2d"]
    count = sum(int(value["count"]) for value in cameras.values())
    return sum(
        float(value["mean"]) * int(value["count"]) for value in cameras.values()
    ) / count


def expected_frames(frame_count: int, capability: str) -> list[int]:
    split = frame_count * 4 // 5
    if capability == "reconstruction_7to1":
        return [frame for frame in range(split) if frame % 8 == 7]
    return list(range(split, frame_count))


def first_existing(run: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        candidate = run / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("找不到候选目录：{} {}".format(run, names))


def read_run(run: Path, dataset_key: str, frame_count: int, seed: int) -> dict:
    status = (run / "status.txt").read_text(encoding="utf-8")
    if "status=complete" not in status or "seed={}".format(seed) not in status:
        raise ValueError("运行状态或 seed 不匹配：{}".format(run))
    audit = load(run / "protocol_audit.json")
    preprocess_root = first_existing(
        run, ("preprocess", "preprocess_formal", "preprocess_v2")
    )
    appearance_root = first_existing(run, ("appearance", "appearance_formal"))
    physics_root = first_existing(run, ("physics", "physics_formal"))
    artifacts_root = first_existing(run, ("artifacts", "artifacts_formal"))
    metrics_root = first_existing(run, ("metrics", "metrics_formal"))
    preprocess = load(preprocess_root / "metadata.json")
    appearance = load(appearance_root / "metadata.json")
    physics = load(physics_root / "metadata.json")
    provenance = load(artifacts_root / "provenance.json")
    if not audit.get("passed") or audit.get("dataset_key") != dataset_key:
        raise ValueError("协议审计不匹配：{}".format(run))
    if audit["phystwin"].get("commit") != EXPECTED_COMMIT:
        raise ValueError("PhysTwin commit 不匹配：{}".format(run))
    expected_train = [i for i in range(frame_count * 4 // 5) if i % 8 != 7]
    if preprocess.get("allowed_observation_frames") != expected_train:
        raise ValueError("预处理读取帧不符合固定协议：{}".format(run))
    if preprocess.get("future_observations_opened") or preprocess.get("evaluation_truth_opened"):
        raise ValueError("预处理存在未来帧或评估真值泄漏：{}".format(run))
    if int(appearance.get("iterations", -1)) != 1000 or int(appearance.get("downsample", -1)) != 2:
        raise ValueError("外观训练不是固定正式配置：{}".format(run))
    if not physics.get("full_protocol"):
        raise ValueError("物理优化不是 CMA20/Adam200/667 子步正式配置：{}".format(run))
    if physics.get("shape_of_motion_used") or provenance.get("shape_of_motion_used"):
        raise ValueError("PhysTwin 正式轨迹不得使用 Shape of Motion：{}".format(run))
    if provenance.get("phystwin_commit") != EXPECTED_COMMIT:
        raise ValueError("导出 provenance 的 PhysTwin commit 不匹配：{}".format(run))
    if int(provenance.get("query_count", -1)) != 30 or int(
        provenance.get("query_alpha_valid_count", -1)
    ) != 30:
        raise ValueError("导出没有完整覆盖固定 30 点：{}".format(run))
    if float(provenance.get("initial_query_reprojection_max_px", float("inf"))) > 1.0e-3:
        raise ValueError("查询点首帧锚定误差过大：{}".format(run))
    if not provenance.get("formal_physics_protocol"):
        raise ValueError("导出拒绝标记为正式物理协议：{}".format(run))

    result = {}
    for capability, _ in CAPABILITIES:
        metric_root = metrics_root / capability
        trajectory_path = metric_root / "trajectory_metrics.json"
        render_path = metric_root / "render_metrics.json"
        trajectory = load(trajectory_path)
        rendering = load(render_path)
        wanted = expected_frames(frame_count, capability)
        if trajectory["frame_selection"]["selected_frame_indices"] != wanted:
            raise ValueError("轨迹评估帧错误：{} {}".format(run, capability))
        if rendering["frame_selection"]["selected_frame_indices"] != wanted:
            raise ValueError("渲染评估帧错误：{} {}".format(run, capability))
        if int(trajectory["evaluated_nodes"]) != 30:
            raise ValueError("轨迹评估没有使用固定 30 点：{}".format(run))
        render_all = rendering["summary"]["all"]
        result[capability] = {
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
    return result


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
    runner_layout = (root / "runs").is_dir()
    for dataset_key, dataset_label, frame_count in DATASETS:
        raw[dataset_key] = {}
        for seed, repeat in enumerate(repeats):
            run = (
                root / "runs" / repeat / dataset_key
                if runner_layout
                else root / dataset_key / repeat
            )
            raw[dataset_key][repeat] = read_run(run, dataset_key, frame_count, seed)
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
        "schema": "fixedsuperbest.phystwin_three_dataset_three_repeat.v1",
        "method": "PhysTwin pinned upstream native spring-mass particles + upstream Gaussian LBS",
        "phystwin_commit": EXPECTED_COMMIT,
        "shape_of_motion_used": False,
        "protocol": {
            "training": "only first 80% non-holdout RGB/depth/masks; calibrated stereo views; known PSM control",
            "reconstruction": "offline interpolation on stride=8 offset=7 holdouts",
            "future": "frozen PhysTwin rolled out with protocol-provided known PSM controls; no future observations",
            "evaluation_points": "30 immutable non-grasp nodes queried at stereo_left frame 0 after physics and appearance freeze",
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
        raise FileExistsError("拒绝覆盖已有 PhysTwin 汇总")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# PhysTwin 仿真 baseline：三数据集三次独立评测",
        "",
        "固定官方 commit `{}`；轨迹直接来自持久 spring-mass 粒子并经 PhysTwin 上游 Gaussian LBS 传播，不使用 Shape of Motion，不做对齐。".format(EXPECTED_COMMIT),
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
