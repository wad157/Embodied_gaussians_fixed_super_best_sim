#!/usr/bin/env python3
"""Summarize joint global-mean and zero-mean regional stiffness updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CAPABILITIES = ("reconstruction_7to1", "future_80to20")


def finite_stats(values: np.ndarray) -> dict[str, float | int | None]:
    selected = np.asarray(values, dtype=np.float64)
    selected = selected[np.isfinite(selected)]
    return {
        "count": int(selected.size),
        "minimum": None if not selected.size else float(selected.min()),
        "median": None if not selected.size else float(np.median(selected)),
        "maximum": None if not selected.size else float(selected.max()),
        "final": None if not selected.size else float(selected[-1]),
    }


def three(values: np.ndarray) -> list[float]:
    return [float(values.min()), float(np.median(values)), float(values.max())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--method", required=True)
    args = parser.parse_args()
    root = args.evaluation_root.resolve()
    results: dict[str, dict[str, object]] = {}
    for capability in CAPABILITIES:
        artifact = root / capability / args.method / "artifacts"
        with np.load(
            artifact / "stiffness_signal_diagnostics.npz", allow_pickle=False
        ) as signal:
            proposal = np.asarray(
                signal["proposal_candidate_count"], dtype=np.int64
            )
            committed = np.asarray(
                signal["committed_update_count"], dtype=np.int64
            )
            cosine = np.asarray(
                signal["gradient_direction_cosine"], dtype=np.float64
            )
            global_log = np.asarray(
                signal["hierarchical_global_log_distance"], dtype=np.float64
            )
            projection_applied = np.asarray(
                signal["short_constraint_projection_applied"],
                dtype=np.float64,
            )
            short_pre = np.asarray(
                signal["short_constraint_pre_directional_derivative"],
                dtype=np.float64,
            )
            short_target = np.asarray(
                signal["short_constraint_target_directional_derivative"],
                dtype=np.float64,
            )
            short_post = np.asarray(
                signal["short_constraint_post_directional_derivative"],
                dtype=np.float64,
            )
            short_long_cosine = np.asarray(
                signal["warp_fd_short_long_gradient_cosine"],
                dtype=np.float64,
            )
        with np.load(artifact / "material_diagnostics.npz") as material:
            distance = np.asarray(
                material["distance_stiffness_per_particle"], dtype=np.float64
            )
        metadata = json.loads(
            (artifact / "artifact_metadata.json").read_text(encoding="utf-8")
        )
        initial = distance[0]
        final = distance[-1]
        log_ratio = np.log(
            np.maximum(final, 1.0e-12) / np.maximum(initial, 1.0e-12)
        )
        global_stats = finite_stats(global_log)
        global_final = float(global_stats["final"] or 0.0)
        regional_deviation = log_ratio - global_final
        constraint_valid = (
            np.isfinite(short_pre)
            & np.isfinite(short_target)
            & np.isfinite(short_post)
        )
        results[capability] = {
            "candidate_frames": int(np.count_nonzero(proposal >= 0)),
            "committed_updates": int(committed.max(initial=0)),
            "gradient_direction_cosine": finite_stats(cosine),
            "global_log_distance": global_stats,
            "global_distance_scale_final": float(np.exp(global_final)),
            "initial_distance_min_median_max": three(initial),
            "final_distance_min_median_max": three(final),
            "regional_log_deviation_min_median_max": three(regional_deviation),
            "short_long_gradient_cosine": finite_stats(short_long_cosine),
            "short_constraint_checked_commits": int(
                np.count_nonzero(constraint_valid)
            ),
            "short_constraint_projected_commits": int(
                np.count_nonzero(
                    constraint_valid & (projection_applied > 0.5)
                )
            ),
            "short_constraint_post_violations": int(
                np.count_nonzero(
                    constraint_valid & (short_post > short_target + 1.0e-7)
                )
            ),
            "short_constraint_pre_directional_derivative": finite_stats(
                short_pre
            ),
            "short_constraint_post_directional_derivative": finite_stats(
                short_post
            ),
            "particles_changed_from_initial": int(
                np.count_nonzero(np.abs(final - initial) > 1.0e-8)
            ),
            "region_count": int(
                metadata["online_stiffness_settings"]["autograd_region_count"]
            ),
            "uses_material_ground_truth": False,
            "common_uniform_initialization": bool(np.ptp(initial) < 1.0e-8),
        }

    payload = {
        "schema": "fixedsuperbest.sim_hierarchical_stiffness_diagnostics.v1",
        "evaluation_root": str(root),
        "method": args.method,
        "parameterization": (
            "log(k_i/k0)=global_alpha+sum_r B_ir beta_r; weighted regional "
            "field is zero mean; all coefficients start at zero; H3/H5 Warp-"
            "Adam step is projected into an H1 reconstruction-descent half-space"
        ),
        "results": results,
    }
    (root / "hierarchical_stiffness_diagnostics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 全局＋区域联合刚度诊断",
        "",
        "- 全局与区域系数均从零开始，只使用当前实验的因果RGB观测。",
        "- 区域基由静止网格构建，不读取真实刚度值或真实区域边界。",
        "- 长期H3/H5更新必须满足H1一阶下降约束，不做分支选择或回滚。",
        "",
        "| 能力 | 区域数 | 提交 | H1投影/检查 | H1违例 | 全局scale | 最终distance min/median/max | 区域log偏差 min/median/max |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for capability in CAPABILITIES:
        value = results[capability]
        final = value["final_distance_min_median_max"]
        regional = value["regional_log_deviation_min_median_max"]
        lines.append(
            f"| {capability} | {value['region_count']} | "
            f"{value['committed_updates']} | "
            f"{value['short_constraint_projected_commits']}/"
            f"{value['short_constraint_checked_commits']} | "
            f"{value['short_constraint_post_violations']} | "
            f"{value['global_distance_scale_final']:.6f} | "
            f"{final[0]:.6f}/{final[1]:.6f}/{final[2]:.6f} | "
            f"{regional[0]:.6f}/{regional[1]:.6f}/{regional[2]:.6f} |"
        )
    (root / "hierarchical_stiffness_diagnostics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
