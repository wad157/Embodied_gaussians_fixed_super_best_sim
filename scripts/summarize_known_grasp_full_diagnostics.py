#!/usr/bin/env python3
"""汇总统一夹持边界六实验的材料真值与在线更新信号。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


CAPABILITIES = (
    ("reconstruction_7to1", "重建7:1"),
    ("future_80to20", "未来80/20"),
)
METHODS = (
    ("pbd", "PBD"),
    ("pbd_visual_residual", "PBD+视觉残差"),
    (
        "pbd_visual_residual_stiffness",
        "PBD+视觉残差+刚度更新",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def finite_spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if len(a) < 3 or np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        return None
    value = float(spearmanr(a, b).statistic)
    return value if math.isfinite(value) else None


def fmt(value: object, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def final_material_summary(comparison: dict) -> dict[str, object]:
    frame = max(int(key) for key in comparison["selected_frames"])
    selected = comparison["selected_frames"][str(frame)]
    distance = selected["distance_particle_level"]
    shape = selected["shape_particle_level"]
    region = selected["region_level"]
    return {
        "frame": frame,
        "distance_particle_spearman": distance["spearman"],
        "distance_particle_pattern_rmse": distance[
            "relative_log_pattern_rmse"
        ],
        "distance_floor_fraction": distance["floor_fraction"],
        "distance_region_spearman": region["distance_region_correlation"][
            "spearman"
        ],
        "distance_region_pattern_rmse": region[
            "distance_region_relative_pattern"
        ]["relative_log_pattern_rmse"],
        "shape_particle_spearman": shape["spearman"],
        "shape_particle_pattern_rmse": shape[
            "relative_log_pattern_rmse"
        ],
        "shape_floor_fraction": shape["floor_fraction"],
        "shape_region_spearman": region["shape_region_correlation"][
            "spearman"
        ],
        "shape_region_pattern_rmse": region[
            "shape_region_relative_pattern"
        ]["relative_log_pattern_rmse"],
        "true_softest_region": region["true_softest_region"],
        "true_hardest_region": region["true_hardest_region"],
        "distance_softest_region": region["distance_softest_region"],
        "distance_hardest_region": region["distance_hardest_region"],
        "shape_softest_region": region["shape_softest_region"],
        "shape_hardest_region": region["shape_hardest_region"],
    }


def read_final_region_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    frame = max(int(row["frame"]) for row in rows)
    result: list[dict[str, object]] = []
    for row in rows:
        if int(row["frame"]) != frame:
            continue
        result.append(
            {
                key: (
                    int(value)
                    if key in {"frame", "region", "mapped_particle_count"}
                    else float(value)
                )
                for key, value in row.items()
            }
        )
    return sorted(result, key=lambda row: int(row["region"]))


def main() -> None:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    root = args.evaluation_root.expanduser().resolve()
    outputs = {
        "json": root / "comprehensive_analysis.json",
        "md": root / "comprehensive_analysis.md",
        "regions": root / "final_region_stiffness.csv",
        "signals": root / "stiffness_signal_region_timeseries.csv",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"拒绝覆盖已有综合诊断：{existing}")

    base = read_json(root / "comparison.json")
    base_lookup = {
        (row["capability"], row["variant"]): row for row in base["rows"]
    }
    with np.load(dataset / "ground_truth/tissue_state.npz") as gt:
        gt_rest = np.asarray(gt["simulation_rest_local"], dtype=np.float64)
        gt_region = np.asarray(gt["simulation_region_ids"], dtype=np.int16)
        gt_young = np.asarray(
            gt["youngs_modulus_pa_per_simulation_node"], dtype=np.float64
        )
        gt_coupling = np.asarray(
            gt["grasp_coupling_weights"], dtype=np.float64
        )
        displacement = np.asarray(gt["simulation_positions"], dtype=np.float64)
        gt_motion = np.linalg.norm(
            displacement - gt_rest[None], axis=2
        ).max(axis=0)
    with np.load(dataset / "gui_assets/tissue_fixedsuperbest.npz") as asset:
        reconstruction_rest = np.asarray(
            asset["rest_positions_table"], dtype=np.float64
        )
    nearest_distance, nearest_gt = cKDTree(gt_rest).query(
        reconstruction_rest, k=1
    )
    mapped_region = gt_region[nearest_gt]
    mapped_young = gt_young[nearest_gt]
    mapped_coupling = gt_coupling[nearest_gt]
    mapped_motion = gt_motion[nearest_gt]

    material_rows: list[dict[str, object]] = []
    final_region_rows: list[dict[str, object]] = []
    signal_summaries: dict[str, dict[str, object]] = {}
    signal_timeseries_rows: list[dict[str, object]] = []

    for capability, _capability_name in CAPABILITIES:
        for method, _method_name in METHODS:
            method_root = root / capability / method
            artifact = read_json(
                method_root / "artifacts" / "artifact_metadata.json"
            )
            boundary_metadata = artifact.get("known_grasp_boundary", {})
            if (
                boundary_metadata.get("mode") != "known_grasp_region"
                or boundary_metadata.get("schema")
                != "fixedsuperbest.known_grasp_region_boundary.v1"
                or int(boundary_metadata.get("controlled_particle_count", -1))
                != 12
            ):
                raise ValueError(
                    f"{capability}/{method}没有使用统一12点完整夹持边界"
                )
            comparison = read_json(
                method_root / "stiffness_gt_comparison" / "comparison.json"
            )
            material = final_material_summary(comparison)
            material_rows.append(
                {"capability": capability, "method": method, **material}
            )
            regions = read_final_region_rows(
                method_root
                / "stiffness_gt_comparison"
                / "selected_frame_region_stiffness.csv"
            )
            for row in regions:
                final_region_rows.append(
                    {"capability": capability, "method": method, **row}
                )

        signal_path = (
            root
            / capability
            / "pbd_visual_residual_stiffness"
            / "artifacts"
            / "stiffness_signal_diagnostics.npz"
        )
        with np.load(signal_path, allow_pickle=False) as signal:
            frame_indices = np.asarray(signal["frame_indices"], dtype=np.int32)
            proposal_count = np.asarray(
                signal["proposal_candidate_count"], dtype=np.int32
            )
            proposal_slots = np.flatnonzero(proposal_count >= 0)
            status = np.asarray(signal["proposal_status"]).astype(str)
            fields = {
                name: np.asarray(signal[name])
                for name in (
                    "residual_norm_m",
                    "deformation_norm_m",
                    "vector_signal",
                    "strain_signal",
                    "strain_confidence",
                    "blended_signal_before_smoothing",
                    "smoothed_signal",
                    "candidate_ema",
                    "log_step",
                    "eligible_mask",
                    "material_active_mask",
                    "control_exclusion_mask",
                )
            }
            committed_count = int(
                np.asarray(signal["committed_update_count"])[-1]
            )

        aggregate_region: list[dict[str, float | int]] = []
        for region in range(10):
            region_mask = mapped_region == region
            active_values: dict[str, list[np.ndarray]] = {
                name: []
                for name in (
                    "residual_norm_m",
                    "deformation_norm_m",
                    "vector_signal",
                    "strain_signal",
                    "strain_confidence",
                    "blended_signal_before_smoothing",
                    "smoothed_signal",
                    "candidate_ema",
                    "log_step",
                )
            }
            for slot in proposal_slots:
                active = fields["material_active_mask"][slot].astype(bool)
                eligible = fields["eligible_mask"][slot].astype(bool)
                selected = region_mask & active & eligible
                row: dict[str, object] = {
                    "capability": capability,
                    "frame": int(frame_indices[slot]),
                    "proposal": int(proposal_count[slot]),
                    "status": status[slot],
                    "region": region,
                    "region_particle_count": int(np.count_nonzero(region_mask)),
                    "active_particle_count": int(np.count_nonzero(selected)),
                    "control_excluded_particle_count": int(
                        np.count_nonzero(
                            region_mask
                            & fields["control_exclusion_mask"][slot].astype(bool)
                        )
                    ),
                }
                for name in active_values:
                    values = fields[name][slot][selected]
                    row[f"mean_{name}"] = (
                        float(np.mean(values)) if len(values) else math.nan
                    )
                    if len(values):
                        active_values[name].append(values.astype(np.float64))
                signal_timeseries_rows.append(row)
            summary: dict[str, float | int] = {
                "region": region,
                "gt_youngs_pa": float(np.mean(mapped_young[region_mask])),
                "coupling_weight": float(np.mean(mapped_coupling[region_mask])),
                "maximum_motion_m": float(np.mean(mapped_motion[region_mask])),
                "particle_count": int(np.count_nonzero(region_mask)),
            }
            for name, chunks in active_values.items():
                summary[f"mean_{name}"] = (
                    float(np.mean(np.concatenate(chunks)))
                    if chunks
                    else math.nan
                )
            aggregate_region.append(summary)

        gt_by_region = np.asarray(
            [row["gt_youngs_pa"] for row in aggregate_region]
        )
        coupling_by_region = np.asarray(
            [row["coupling_weight"] for row in aggregate_region]
        )
        motion_by_region = np.asarray(
            [row["maximum_motion_m"] for row in aggregate_region]
        )
        region_mean_step = np.asarray(
            [row["mean_log_step"] for row in aggregate_region]
        )
        all_nonzero_steps = fields["log_step"][proposal_slots]
        all_nonzero_steps = all_nonzero_steps[np.isfinite(all_nonzero_steps)]
        nonzero = np.abs(all_nonzero_steps) > 1.0e-12
        active_steps = all_nonzero_steps[nonzero]
        signal_summaries[capability] = {
            "proposal_count": int(len(proposal_slots)),
            "committed_update_count": committed_count,
            "committed_status_count": int(
                np.count_nonzero(status[proposal_slots] == "committed")
            ),
            "active_log_step_count": int(len(active_steps)),
            "softening_step_fraction": (
                float(np.mean(active_steps < 0.0)) if len(active_steps) else None
            ),
            "hardening_step_fraction": (
                float(np.mean(active_steps > 0.0)) if len(active_steps) else None
            ),
            "mean_absolute_log_step": (
                float(np.mean(np.abs(active_steps)))
                if len(active_steps)
                else None
            ),
            "region_mean_log_step_spearman_vs_true_youngs": finite_spearman(
                gt_by_region, region_mean_step
            ),
            "region_mean_log_step_spearman_vs_coupling": finite_spearman(
                coupling_by_region, region_mean_step
            ),
            "region_mean_log_step_spearman_vs_motion": finite_spearman(
                motion_by_region, region_mean_step
            ),
            "per_region": aggregate_region,
        }

    with outputs["regions"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(final_region_rows[0]))
        writer.writeheader()
        writer.writerows(final_region_rows)
    with outputs["signals"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(signal_timeseries_rows[0])
        )
        writer.writeheader()
        writer.writerows(signal_timeseries_rows)

    report = {
        "schema": "fixedsuperbest.known_grasp_comprehensive_analysis.v1",
        "dataset": str(dataset),
        "evaluation_root": str(root),
        "boundary": read_json(
            dataset / "task_inputs" / "known_grasp_region_boundary.json"
        ),
        "mapping": {
            "reconstruction_particles": int(len(reconstruction_rest)),
            "nearest_gt_distance_mean_mm": float(nearest_distance.mean() * 1e3),
            "nearest_gt_distance_p95_mm": float(
                np.percentile(nearest_distance, 95.0) * 1e3
            ),
        },
        "capability_metrics": base["rows"],
        "material_metrics": material_rows,
        "stiffness_signal_summary": signal_summaries,
        "outputs": {
            "final_region_stiffness_csv": str(outputs["regions"]),
            "stiffness_signal_region_timeseries_csv": str(outputs["signals"]),
        },
    }
    outputs["json"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    material_lookup = {
        (row["capability"], row["method"]): row for row in material_rows
    }
    lines = [
        "# 统一已知夹持区域：重建、未来预测与刚度信号综合分析",
        "",
        "三种方法都使用相同的12点完整夹持核心Dirichlet边界；外围组织不读取真值运动。",
        "材料真值只在所有运行结束后用于本文件的比较。",
        "",
        "## 轨迹与渲染",
        "",
        "| 能力 | 方法 | 3D Mean ↓ | 3D RMSE ↓ | 2D Mean ↓ | 2D RMSE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for capability, capability_name in CAPABILITIES:
        for method, method_name in METHODS:
            row = base_lookup[(capability, method)]
            lines.append(
                f"| {capability_name} | {method_name} | "
                f"{fmt(row['track_3d_mean_mm'], 3)} | "
                f"{fmt(row['track_3d_rmse_mm'], 3)} | "
                f"{fmt(row['track_2d_stereo_mean_px'], 3)} | "
                f"{fmt(row['track_2d_stereo_rmse_px'], 3)} | "
                f"{fmt(row['psnr_tissue_layer_db'], 3)} | "
                f"{fmt(row['ssim_tissue_layer'], 4)} | "
                f"{fmt(row['lpips_tissue_layer'], 4)} |"
            )
    lines.extend(
        [
            "",
            "## 最终材料空间分布",
            "",
            "| 能力 | 方法 | D区域ρ ↑ | D区域分布RMSE ↓ | D下限占比 | S区域ρ ↑ | S区域分布RMSE ↓ | S下限占比 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for capability, capability_name in CAPABILITIES:
        for method, method_name in METHODS:
            row = material_lookup[(capability, method)]
            lines.append(
                f"| {capability_name} | {method_name} | "
                f"{fmt(row['distance_region_spearman'])} | "
                f"{fmt(row['distance_region_pattern_rmse'])} | "
                f"{fmt(row['distance_floor_fraction'])} | "
                f"{fmt(row['shape_region_spearman'])} | "
                f"{fmt(row['shape_region_pattern_rmse'])} | "
                f"{fmt(row['shape_floor_fraction'])} |"
            )
    lines.extend(["", "## 引起刚度更新的信号", ""])
    for capability, capability_name in CAPABILITIES:
        signal = signal_summaries[capability]
        lines.extend(
            [
                f"### {capability_name}",
                "",
                f"- 候选/提交次数：{signal['proposal_count']}/{signal['committed_update_count']}",
                f"- 软化/硬化log步占比：{fmt(signal['softening_step_fraction'])}/{fmt(signal['hardening_step_fraction'])}",
                f"- 平均绝对log步：{fmt(signal['mean_absolute_log_step'], 6)}",
                "- 区域平均log步Spearman："
                f"真值E={fmt(signal['region_mean_log_step_spearman_vs_true_youngs'])}，"
                f"夹持耦合={fmt(signal['region_mean_log_step_spearman_vs_coupling'])}，"
                f"真实运动幅度={fmt(signal['region_mean_log_step_spearman_vs_motion'])}",
                "",
            ]
        )
    lines.extend(
        [
            "逐区域最终刚度见`final_region_stiffness.csv`；逐候选、逐区域原始信号见",
            "`stiffness_signal_region_timeseries.csv`。",
            "",
        ]
    )
    outputs["md"].write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
