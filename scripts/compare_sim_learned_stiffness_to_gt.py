#!/usr/bin/env python3
"""将RGB在线更新的逐粒子PBD刚度与仿真十区域材料真值做事后比较。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--material-diagnostics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--analysis-frames",
        default="123,126,132,160,200,240,287,359",
        help="写入逐区域明细的帧号，逗号分隔。",
    )
    parser.add_argument("--distance-lower", type=float, default=0.10)
    parser.add_argument("--distance-upper", type=float, default=2.00)
    parser.add_argument("--shape-lower", type=float, default=0.003)
    parser.add_argument("--shape-upper", type=float, default=0.020)
    return parser.parse_args()


def finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def correlation(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float | None]:
    valid = np.isfinite(reference) & np.isfinite(estimate)
    x = np.asarray(reference[valid], dtype=np.float64)
    y = np.asarray(estimate[valid], dtype=np.float64)
    if len(x) < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return {
            "pearson": None,
            "spearman": None,
            "normalized_rmse": None,
        }
    pearson = pearsonr(x, y).statistic
    spearman = spearmanr(x, y).statistic
    xz = (x - x.mean()) / x.std()
    yz = (y - y.mean()) / y.std()
    return {
        "pearson": finite_or_none(pearson),
        "spearman": finite_or_none(spearman),
        "normalized_rmse": finite_or_none(np.sqrt(np.mean((xz - yz) ** 2))),
    }


def binary_soft_hard_accuracy(
    reference: np.ndarray, estimate: np.ndarray
) -> float | None:
    if np.ptp(estimate) == 0.0:
        return None
    reference_hard = reference > np.median(reference)
    estimate_hard = estimate > np.median(estimate)
    return float(np.mean(reference_hard == estimate_hard))


def relative_log_pattern_error(
    reference: np.ndarray, estimate: np.ndarray
) -> dict[str, float]:
    """Compare only spatial relative variation, independent of units/scale."""
    reference_log = np.log(np.asarray(reference, dtype=np.float64))
    estimate_log = np.log(np.asarray(estimate, dtype=np.float64))
    reference_pattern = reference_log - reference_log.mean()
    estimate_pattern = estimate_log - estimate_log.mean()
    difference = estimate_pattern - reference_pattern
    return {
        "relative_log_pattern_rmse": float(
            np.sqrt(np.mean(difference * difference))
        ),
        "relative_log_pattern_mae": float(np.mean(np.abs(difference))),
        "reference_log_standard_deviation": float(reference_pattern.std()),
        "estimate_log_standard_deviation": float(estimate_pattern.std()),
    }


def field_summary(
    values: np.ndarray,
    reference: np.ndarray,
    lower: float,
    upper: float,
) -> dict[str, object]:
    result: dict[str, object] = {
        "minimum": float(np.min(values)),
        "p05": float(np.percentile(values, 5.0)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "maximum": float(np.max(values)),
        "floor_fraction": float(np.mean(values <= lower + 1.0e-6)),
        "ceiling_fraction": float(np.mean(values >= upper - 1.0e-6)),
        "soft_hard_accuracy": binary_soft_hard_accuracy(reference, values),
    }
    result.update(correlation(reference, values))
    result.update(relative_log_pattern_error(reference, values))
    return result


def region_rows(
    frame: int,
    gt_young: np.ndarray,
    region_ids: np.ndarray,
    distance: np.ndarray,
    shape: np.ndarray,
    distance_lower: float,
    distance_upper: float,
    shape_lower: float,
    shape_upper: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for region in sorted(int(value) for value in np.unique(region_ids)):
        selected = region_ids == region
        d = distance[selected]
        s = shape[selected]
        rows.append(
            {
                "frame": frame,
                "region": region,
                "mapped_particle_count": int(np.count_nonzero(selected)),
                "gt_youngs_pa": float(np.mean(gt_young[selected])),
                "distance_mean": float(np.mean(d)),
                "distance_median": float(np.median(d)),
                "distance_floor_fraction": float(
                    np.mean(d <= distance_lower + 1.0e-6)
                ),
                "distance_ceiling_fraction": float(
                    np.mean(d >= distance_upper - 1.0e-6)
                ),
                "shape_mean": float(np.mean(s)),
                "shape_median": float(np.median(s)),
                "shape_floor_fraction": float(
                    np.mean(s <= shape_lower + 1.0e-6)
                ),
                "shape_ceiling_fraction": float(
                    np.mean(s >= shape_upper - 1.0e-6)
                ),
            }
        )
    return rows


def region_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    gt = np.asarray([row["gt_youngs_pa"] for row in rows], dtype=np.float64)
    distance = np.asarray([row["distance_mean"] for row in rows], dtype=np.float64)
    shape = np.asarray([row["shape_mean"] for row in rows], dtype=np.float64)
    true_softest = int(rows[int(np.argmin(gt))]["region"])
    true_hardest = int(rows[int(np.argmax(gt))]["region"])
    distance_softest = int(rows[int(np.argmin(distance))]["region"])
    distance_hardest = int(rows[int(np.argmax(distance))]["region"])
    shape_softest = int(rows[int(np.argmin(shape))]["region"])
    shape_hardest = int(rows[int(np.argmax(shape))]["region"])
    result = {
        "distance_region_correlation": correlation(gt, distance),
        "shape_region_correlation": correlation(gt, shape),
        "distance_region_relative_pattern": relative_log_pattern_error(
            gt, distance
        ),
        "shape_region_relative_pattern": relative_log_pattern_error(gt, shape),
        "true_softest_region": true_softest,
        "true_hardest_region": true_hardest,
        "distance_softest_region": distance_softest,
        "distance_hardest_region": distance_hardest,
        "shape_softest_region": shape_softest,
        "shape_hardest_region": shape_hardest,
        "distance_softest_correct": distance_softest == true_softest,
        "distance_hardest_correct": distance_hardest == true_hardest,
        "shape_softest_correct": shape_softest == true_softest,
        "shape_hardest_correct": shape_hardest == true_hardest,
    }
    return result


def fmt(value: object, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    diagnostics_path = args.material_diagnostics.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"拒绝覆盖已有刚度真值诊断：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_path = dataset / "ground_truth/tissue_state.npz"
    asset_path = dataset / "gui_assets/tissue_fixedsuperbest.npz"
    with np.load(gt_path, allow_pickle=False) as gt_file:
        gt_rest = np.asarray(gt_file["simulation_rest_local"], dtype=np.float64)
        gt_young_nodes = np.asarray(
            gt_file["youngs_modulus_pa_per_simulation_node"],
            dtype=np.float64,
        )
        gt_region_nodes = np.asarray(
            gt_file["simulation_region_ids"], dtype=np.int16
        )
    with np.load(asset_path, allow_pickle=False) as asset_file:
        reconstruction_rest = np.asarray(
            asset_file["rest_positions_table"], dtype=np.float64
        )
    with np.load(diagnostics_path, allow_pickle=False) as diagnostic_file:
        required = {
            "frame_indices",
            "reconstruction_particle_ids",
            "distance_stiffness_per_particle",
            "shape_stiffness_per_particle",
        }
        missing = required.difference(diagnostic_file.files)
        if missing:
            raise KeyError(
                "旧材料诊断没有完整逐粒子刚度场，必须使用新诊断复跑产物："
                f"{sorted(missing)}"
            )
        frame_indices = np.asarray(diagnostic_file["frame_indices"], dtype=np.int32)
        particle_ids = np.asarray(
            diagnostic_file["reconstruction_particle_ids"], dtype=np.int32
        )
        distance_fields = np.asarray(
            diagnostic_file["distance_stiffness_per_particle"],
            dtype=np.float64,
        )
        shape_fields = np.asarray(
            diagnostic_file["shape_stiffness_per_particle"],
            dtype=np.float64,
        )
    if particle_ids.tolist() != list(range(len(reconstruction_rest))):
        raise ValueError("材料粒子ID与重建资产粒子顺序不一致")
    expected_shape = (len(frame_indices), len(reconstruction_rest))
    if distance_fields.shape != expected_shape or shape_fields.shape != expected_shape:
        raise ValueError("逐粒子刚度场维度与帧/重建粒子不一致")

    mapping_distance, nearest_gt = cKDTree(gt_rest).query(
        reconstruction_rest, k=1
    )
    mapped_young = gt_young_nodes[nearest_gt]
    mapped_region = gt_region_nodes[nearest_gt]
    frame_to_slot = {int(frame): slot for slot, frame in enumerate(frame_indices)}
    requested_frames = [
        int(value.strip())
        for value in args.analysis_frames.split(",")
        if value.strip()
    ]
    absent = [frame for frame in requested_frames if frame not in frame_to_slot]
    if absent:
        raise ValueError(f"诊断帧不在产物中：{absent}")

    per_frame_rows: list[dict[str, object]] = []
    all_region_rows: list[dict[str, object]] = []
    selected_reports: dict[str, object] = {}
    for slot, frame_value in enumerate(frame_indices):
        frame = int(frame_value)
        distance = distance_fields[slot]
        shape = shape_fields[slot]
        distance_report = field_summary(
            distance,
            mapped_young,
            args.distance_lower,
            args.distance_upper,
        )
        shape_report = field_summary(
            shape,
            mapped_young,
            args.shape_lower,
            args.shape_upper,
        )
        per_frame_rows.append(
            {
                "frame": frame,
                "distance_pearson": distance_report["pearson"],
                "distance_spearman": distance_report["spearman"],
                "distance_soft_hard_accuracy": distance_report[
                    "soft_hard_accuracy"
                ],
                "distance_floor_fraction": distance_report["floor_fraction"],
                "distance_ceiling_fraction": distance_report[
                    "ceiling_fraction"
                ],
                "shape_pearson": shape_report["pearson"],
                "shape_spearman": shape_report["spearman"],
                "shape_soft_hard_accuracy": shape_report[
                    "soft_hard_accuracy"
                ],
                "shape_floor_fraction": shape_report["floor_fraction"],
                "shape_ceiling_fraction": shape_report["ceiling_fraction"],
            }
        )
        if frame in requested_frames:
            regions = region_rows(
                frame,
                mapped_young,
                mapped_region,
                distance,
                shape,
                args.distance_lower,
                args.distance_upper,
                args.shape_lower,
                args.shape_upper,
            )
            all_region_rows.extend(regions)
            selected_reports[str(frame)] = {
                "distance_particle_level": distance_report,
                "shape_particle_level": shape_report,
                "region_level": region_summary(regions),
            }

    valid_distance = [
        row for row in per_frame_rows if row["distance_spearman"] is not None
    ]
    valid_shape = [row for row in per_frame_rows if row["shape_spearman"] is not None]
    best_distance = (
        max(valid_distance, key=lambda row: float(row["distance_spearman"]))
        if valid_distance
        else None
    )
    best_shape = (
        max(valid_shape, key=lambda row: float(row["shape_spearman"]))
        if valid_shape
        else None
    )
    gt_unique = np.unique(gt_young_nodes)
    report = {
        "schema": "fixedsuperbest.stiffness_gt_comparison.v1",
        "dataset": str(dataset),
        "material_diagnostics": str(diagnostics_path),
        "comparison_policy": (
            "材料真值仅在运行结束后读取；通过首帧静止空间最近邻将"
            f"{len(reconstruction_rest)}个重建粒子映射到{len(gt_rest)}个"
            "仿真材料节点；高杨氏模量应对应高PBD刚度"
        ),
        "unit_warning": (
            "Pa与PBD distance/shape不是同一单位，只比较空间排序、相关性和"
            "相对区域差异，不比较绝对数值"
        ),
        "mapping": {
            "gt_node_count": int(len(gt_rest)),
            "reconstruction_particle_count": int(len(reconstruction_rest)),
            "nearest_distance_mean_mm": float(mapping_distance.mean() * 1.0e3),
            "nearest_distance_p95_mm": float(
                np.percentile(mapping_distance, 95.0) * 1.0e3
            ),
            "nearest_distance_max_mm": float(mapping_distance.max() * 1.0e3),
        },
        "ground_truth": {
            "region_count": int(len(gt_unique)),
            "youngs_modulus_pa": gt_unique.tolist(),
            "minimum_pa": float(gt_unique.min()),
            "maximum_pa": float(gt_unique.max()),
            "maximum_minimum_ratio": float(gt_unique.max() / gt_unique.min()),
            "coefficient_of_variation": float(
                gt_unique.std() / gt_unique.mean()
            ),
        },
        "analysis_frames": requested_frames,
        "selected_frames": selected_reports,
        "maximum_distance_spearman_frame": best_distance,
        "maximum_shape_spearman_frame": best_shape,
        "per_frame_csv": "per_frame_stiffness_gt_correlation.csv",
        "per_region_csv": "selected_frame_region_stiffness.csv",
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "per_frame_stiffness_gt_correlation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_frame_rows[0]))
        writer.writeheader()
        writer.writerows(per_frame_rows)
    with (output_dir / "selected_frame_region_stiffness.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_region_rows[0]))
        writer.writeheader()
        writer.writerows(all_region_rows)

    final_frame = requested_frames[-1]
    final = selected_reports[str(final_frame)]
    region = final["region_level"]
    lines = [
        "# 在线刚度与十区域材料真值比较",
        "",
        "材料真值只在运行结束后读取。Pa与PBD刚度单位不同，因此只比较空间排序。",
        "",
        "| 帧 | Distance Spearman | Distance相对分布RMSE | Distance下限占比 | Shape Spearman | Shape相对分布RMSE | Shape下限占比 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for frame in requested_frames:
        selected = selected_reports[str(frame)]
        distance = selected["distance_particle_level"]
        shape = selected["shape_particle_level"]
        lines.append(
            f"| {frame} | {fmt(distance['spearman'])} | "
            f"{fmt(distance['relative_log_pattern_rmse'])} | "
            f"{fmt(distance['floor_fraction'])} | "
            f"{fmt(shape['spearman'])} | "
            f"{fmt(shape['relative_log_pattern_rmse'])} | "
            f"{fmt(shape['floor_fraction'])} |"
        )
    lines.extend(
        (
            "",
            f"最终诊断帧：frame {final_frame}。",
            f"Distance区域级Spearman：{fmt(region['distance_region_correlation']['spearman'])}。",
            f"Shape区域级Spearman：{fmt(region['shape_region_correlation']['spearman'])}。",
            f"真实最软/最硬区域：{region['true_softest_region']}/{region['true_hardest_region']}。",
            f"Distance估计最软/最硬区域：{region['distance_softest_region']}/{region['distance_hardest_region']}。",
            f"Shape估计最软/最硬区域：{region['shape_softest_region']}/{region['shape_hardest_region']}。",
            "",
        )
    )
    final_region_rows = [
        row for row in all_region_rows if int(row["frame"]) == final_frame
    ]
    lines.extend(
        (
            "## 最终帧逐区域比较",
            "",
            "| 区域 | 真值E (Pa) | Distance均值 | Distance中位数 | Distance下限占比 | Shape均值 | Shape中位数 | Shape下限占比 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in final_region_rows:
        lines.append(
            f"| {row['region']} | {fmt(row['gt_youngs_pa'], 2)} | "
            f"{fmt(row['distance_mean'], 5)} | "
            f"{fmt(row['distance_median'], 5)} | "
            f"{fmt(row['distance_floor_fraction'])} | "
            f"{fmt(row['shape_mean'], 6)} | "
            f"{fmt(row['shape_median'], 6)} | "
            f"{fmt(row['shape_floor_fraction'])} |"
        )
    lines.append("")
    (output_dir / "comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
