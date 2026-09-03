#!/usr/bin/env python3
"""汇总 Foundation 深度 C 组的梯度与刚度提交诊断。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CAPABILITIES = ("reconstruction_7to1", "future_80to20")
METHOD = "pbd_cotracker_foundation_depth_global_distribution"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--method", default=METHOD)
    return parser.parse_args()


def finite_summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {
        "count": int(finite.size),
        "minimum": None if not finite.size else float(finite.min()),
        "median": None if not finite.size else float(np.median(finite)),
        "maximum": None if not finite.size else float(finite.max()),
    }


def main() -> None:
    args = parse_args()
    root = args.evaluation_root.resolve()
    method = str(args.method)
    results: dict[str, dict] = {}
    for capability in CAPABILITIES:
        artifact = root / capability / method / "artifacts"
        with np.load(
            artifact / "stiffness_signal_diagnostics.npz", allow_pickle=True
        ) as signal:
            raw_distance = signal["autograd_raw_gradient_log_distance"]
            raw_damping = signal["autograd_raw_gradient_log_damping"]
            finite_count = signal["autograd_gradient_finite_count"]
            nonfinite_count = signal["autograd_gradient_nonfinite_count"]
            warp_distance = signal["warp_fd_gradient_log_distance"]
            warp_damping = signal["warp_fd_gradient_log_damping"]
            cosine = signal["gradient_direction_cosine"]
            threshold = signal["gradient_cosine_minimum"]
            status = np.asarray(signal["proposal_status"]).astype(str)
            committed = np.asarray(signal["committed_update_count"], dtype=np.int64)
            ready = np.isfinite(finite_count)
            raw_finite = ready & (finite_count == 3)
            raw_nonfinite = ready & (nonfinite_count > 0)
            cosine_finite = np.isfinite(cosine)
            cosine_pass = cosine_finite & np.isfinite(threshold) & (cosine >= threshold)
            unique_status, status_count = np.unique(status, return_counts=True)
            signal_result = {
                "candidate_frames": int(np.count_nonzero(status != "none")),
                "ready_autograd_frames": int(np.count_nonzero(ready)),
                "all_three_autograd_parameters_finite_frames": int(
                    np.count_nonzero(raw_finite)
                ),
                "nonfinite_autograd_frames": int(np.count_nonzero(raw_nonfinite)),
                "nonzero_raw_distance_gradient_frames": int(
                    np.count_nonzero(raw_finite & (raw_distance != 0.0))
                ),
                "nonzero_raw_damping_gradient_frames": int(
                    np.count_nonzero(raw_finite & (raw_damping != 0.0))
                ),
                "warp_fd_frames": int(
                    np.count_nonzero(
                        np.isfinite(warp_distance) & np.isfinite(warp_damping)
                    )
                ),
                "cosine_checked_frames": int(np.count_nonzero(cosine_finite)),
                "cosine_passed_frames": int(np.count_nonzero(cosine_pass)),
                "committed_updates": int(committed.max(initial=0)),
                "proposal_status_counts": {
                    key: int(value)
                    for key, value in zip(unique_status.tolist(), status_count.tolist())
                },
                "raw_distance_gradient": finite_summary(raw_distance),
                "raw_damping_gradient": finite_summary(raw_damping),
                "warp_distance_gradient": finite_summary(warp_distance),
                "warp_damping_gradient": finite_summary(warp_damping),
                "gradient_direction_cosine": finite_summary(cosine),
            }
        with np.load(artifact / "material_diagnostics.npz") as material:
            distance = np.asarray(
                material["distance_stiffness_per_particle"], dtype=np.float64
            )
            initial = distance[0]
            final = distance[-1]
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
                "particles_changed_from_initial": int(
                    np.count_nonzero(np.abs(final - initial) > 1.0e-8)
                ),
            }
        metadata = json.loads(
            (artifact / "artifact_metadata.json").read_text(encoding="utf-8")
        )
        stiffness_metadata = metadata["online_stiffness_settings"]
        results[capability] = {
            **signal_result,
            **material_result,
            "metadata_candidate_count": int(
                stiffness_metadata["candidate_count"]
            ),
            "metadata_committed_update_count": int(
                stiffness_metadata["committed_update_count"]
            ),
            "metadata_rejected_update_count": int(
                stiffness_metadata["rejected_update_count"]
            ),
        }

    payload = {
        "schema": "fixedsuperbest.foundation_c_gradient_fix_diagnostics.v1",
        "evaluation_root": str(root),
        "method": method,
        "results": results,
    }
    (root / "c_gradient_diagnostics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Foundation 深度 C 组梯度修复诊断",
        "",
        "| 能力 | AD就绪帧 | AD非有限帧 | Warp检查 | 余弦通过 | 刚度提交 | 最终距离刚度 min/median/max |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for capability in CAPABILITIES:
        value = results[capability]
        final = value["final_distance_min_median_max"]
        lines.append(
            f"| {capability} | {value['ready_autograd_frames']} | "
            f"{value['nonfinite_autograd_frames']} | {value['cosine_checked_frames']} | "
            f"{value['cosine_passed_frames']} | {value['committed_updates']} | "
            f"{final[0]:.6f}/{final[1]:.6f}/{final[2]:.6f} |"
        )
    (root / "c_gradient_diagnostics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
