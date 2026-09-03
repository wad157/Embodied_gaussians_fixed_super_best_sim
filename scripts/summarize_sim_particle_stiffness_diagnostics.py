#!/usr/bin/env python3
"""Summarize the independently stored particle-wise stiffness field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CAPABILITIES = ("reconstruction_7to1", "future_80to20")


def stats(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    return [
        float(values.min()),
        float(np.median(values)),
        float(values.max()),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--method", required=True)
    args = parser.parse_args()
    root = args.evaluation_root.resolve()
    results: dict[str, dict[str, object]] = {}
    for capability in CAPABILITIES:
        artifact = root / capability / args.method / "artifacts"
        with np.load(artifact / "material_diagnostics.npz") as material:
            distance = np.asarray(
                material["distance_stiffness_per_particle"], dtype=np.float64
            )
            particle_ids = np.asarray(
                material["reconstruction_particle_ids"], dtype=np.int64
            )
        with np.load(
            artifact / "stiffness_signal_diagnostics.npz", allow_pickle=False
        ) as signal:
            proposal = np.asarray(
                signal["proposal_candidate_count"], dtype=np.int64
            )
            committed = np.asarray(
                signal["committed_update_count"], dtype=np.int64
            )
            log_step = np.asarray(signal["log_step"], dtype=np.float64)
            eligible = np.asarray(signal["eligible_mask"], dtype=bool)

        initial = distance[0]
        final = distance[-1]
        changed = np.abs(final - initial) > 1.0e-8
        proposal_slots = proposal >= 0
        nonzero_step = np.abs(log_step) > 1.0e-12
        ever_updated = np.any(nonzero_step[proposal_slots], axis=0)
        ever_eligible = np.any(eligible[proposal_slots], axis=0)
        results[capability] = {
            "particle_count": int(len(particle_ids)),
            "candidate_frames": int(np.count_nonzero(proposal_slots)),
            "committed_updates": int(committed.max(initial=0)),
            "initial_distance_min_median_max": stats(initial),
            "final_distance_min_median_max": stats(final),
            "final_distance_standard_deviation": float(np.std(final)),
            "final_unique_values_1e8": int(len(np.unique(np.round(final, 8)))),
            "particles_changed_from_initial": int(np.count_nonzero(changed)),
            "particles_ever_eligible": int(np.count_nonzero(ever_eligible)),
            "particles_with_nonzero_independent_step": int(
                np.count_nonzero(ever_updated)
            ),
            "field_is_spatially_nonuniform": bool(np.ptp(final) > 1.0e-8),
        }

    payload = {
        "schema": "fixedsuperbest.sim_particle_stiffness_diagnostics.v1",
        "evaluation_root": str(root),
        "method": args.method,
        "parameterization": (
            "one distance/shape stiffness value per physical particle; "
            "graph smoothing applies to evidence only"
        ),
        "results": results,
    }
    (root / "particle_stiffness_diagnostics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 逐粒子刚度场诊断",
        "",
        "- 每个自由物理粒子独立保存刚度；固定、夹持控制和无效粒子不更新。",
        "- 空间图只平滑残差信号，不共享或平均最终刚度参数。",
        "",
        "| 能力 | 粒子数 | 提交次数 | 曾独立更新粒子 | 最终distance min/median/max | 唯一值数 |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for capability in CAPABILITIES:
        value = results[capability]
        final = value["final_distance_min_median_max"]
        lines.append(
            f"| {capability} | {value['particle_count']} | "
            f"{value['committed_updates']} | "
            f"{value['particles_with_nonzero_independent_step']} | "
            f"{final[0]:.6f}/{final[1]:.6f}/{final[2]:.6f} | "
            f"{value['final_unique_values_1e8']} |"
        )
    (root / "particle_stiffness_diagnostics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
