#!/usr/bin/env python3
"""Summarize local zero-mean H1/H3/H5 stiffness diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CAPABILITIES = ("reconstruction_7to1", "future_80to20")


def finite(values: np.ndarray) -> dict[str, float | int | None]:
    selected = np.asarray(values, dtype=np.float64)
    selected = selected[np.isfinite(selected)]
    return {
        "count": int(selected.size),
        "minimum": None if not selected.size else float(selected.min()),
        "median": None if not selected.size else float(np.median(selected)),
        "maximum": None if not selected.size else float(selected.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--method", required=True)
    args = parser.parse_args()
    root = args.evaluation_root.resolve()
    results: dict[str, dict] = {}
    for capability in CAPABILITIES:
        artifact = root / capability / args.method / "artifacts"
        with np.load(
            artifact / "stiffness_signal_diagnostics.npz", allow_pickle=True
        ) as signal:
            status = np.asarray(signal["proposal_status"]).astype(str)
            committed = np.asarray(signal["committed_update_count"], dtype=np.int64)
            cosine = np.asarray(signal["gradient_direction_cosine"], dtype=np.float64)
            threshold = np.asarray(signal["gradient_cosine_minimum"], dtype=np.float64)
            warp_norm = np.asarray(signal["warp_fd_gradient_norm"], dtype=np.float64)
            zero_mean = np.asarray(
                signal["local_weighted_particle_log_mean"], dtype=np.float64
            )
            unique_status, status_count = np.unique(status, return_counts=True)
            cosine_checked = np.isfinite(cosine)
            cosine_passed = cosine_checked & np.isfinite(threshold) & (
                cosine >= threshold
            )
            signal_result = {
                "candidate_frames": int(np.count_nonzero(status != "none")),
                "warp_checked_frames": int(np.count_nonzero(np.isfinite(warp_norm))),
                "cosine_checked_frames": int(np.count_nonzero(cosine_checked)),
                "cosine_passed_frames": int(np.count_nonzero(cosine_passed)),
                "committed_updates": int(committed.max(initial=0)),
                "gradient_direction_cosine": finite(cosine),
                "warp_fd_gradient_norm": finite(warp_norm),
                "weighted_particle_log_mean": finite(zero_mean),
                "proposal_status_counts": {
                    key: int(value)
                    for key, value in zip(
                        unique_status.tolist(), status_count.tolist()
                    )
                },
            }
        with np.load(artifact / "material_diagnostics.npz") as material:
            distance = np.asarray(
                material["distance_stiffness_per_particle"], dtype=np.float64
            )
            initial = distance[0]
            final = distance[-1]
            log_ratio = np.log(
                np.maximum(final, 1.0e-12) / np.maximum(initial, 1.0e-12)
            )
            material_result = {
                "initial_distance_min_median_max": [
                    float(initial.min()),
                    float(np.median(initial)),
                    float(initial.max()),
                ],
                "final_distance_min_median_max": [
                    float(final.min()),
                    float(np.median(final)),
                    float(final.max()),
                ],
                "final_log_ratio_min_median_max": [
                    float(log_ratio.min()),
                    float(np.median(log_ratio)),
                    float(log_ratio.max()),
                ],
                "particles_changed_from_initial": int(
                    np.count_nonzero(np.abs(final - initial) > 1.0e-8)
                ),
            }
        metadata = json.loads(
            (artifact / "artifact_metadata.json").read_text(encoding="utf-8")
        )
        results[capability] = {
            **signal_result,
            **material_result,
            "online_stiffness_settings": metadata["online_stiffness_settings"],
        }

    payload = {
        "schema": "fixedsuperbest.sim_local_relative_diagnostics.v1",
        "evaluation_root": str(root),
        "method": args.method,
        "results": results,
    }
    (root / "local_relative_diagnostics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 局部零均值刚度更新诊断",
        "",
        "| 能力 | Warp检查 | 余弦通过 | 刚度提交 | 最终distance min/median/max | 加权log均值最大绝对值 |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for capability in CAPABILITIES:
        value = results[capability]
        final = value["final_distance_min_median_max"]
        zero = value["weighted_particle_log_mean"]
        zero_abs = max(
            abs(zero["minimum"] or 0.0), abs(zero["maximum"] or 0.0)
        )
        lines.append(
            f"| {capability} | {value['warp_checked_frames']} | "
            f"{value['cosine_passed_frames']} | {value['committed_updates']} | "
            f"{final[0]:.6f}/{final[1]:.6f}/{final[2]:.6f} | "
            f"{zero_abs:.3e} |"
        )
    (root / "local_relative_diagnostics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
