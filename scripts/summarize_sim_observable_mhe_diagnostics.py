#!/usr/bin/env python3
"""汇总全局可观测 MHE 刚度更新的真实 Warp 诊断。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CAPABILITIES = ("reconstruction_7to1", "future_80to20")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument(
        "--method",
        default="pbd_cotracker_foundation_depth_global_mhe",
    )
    return parser.parse_args()


def finite_summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
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
            status = np.asarray(signal["proposal_status"]).astype(str)
            unique_status, counts = np.unique(status, return_counts=True)
            committed = np.asarray(
                signal["committed_update_count"], dtype=np.int64
            )
            results[capability] = {
                "candidate_frames": int(np.count_nonzero(status != "none")),
                "proposal_status_counts": {
                    key: int(value)
                    for key, value in zip(unique_status.tolist(), counts.tolist())
                },
                "committed_updates": int(committed.max(initial=0)),
                "observable_allowed_frames": int(
                    np.count_nonzero(signal["observable_allowed"] == 1)
                ),
                "transition_count": finite_summary(
                    signal["observable_transition_count"]
                ),
                "used_transition_count": finite_summary(
                    signal["observable_used_transition_count"]
                ),
                "segment_count": finite_summary(
                    signal["observable_segment_count"]
                ),
                "residual_count": finite_summary(
                    signal["observable_residual_count"]
                ),
                "short_residual_count": finite_summary(
                    signal["observable_short_residual_count"]
                ),
                "used_short_transition_count": finite_summary(
                    signal["observable_used_short_transition_count"]
                ),
                "short_descent_projected_frames": int(
                    np.count_nonzero(
                        signal["observable_short_descent_projected"] == 1
                    )
                ),
                "short_directional_before": finite_summary(
                    signal["observable_short_directional_before"]
                ),
                "short_directional_after": finite_summary(
                    signal["observable_short_directional_after"]
                ),
                "short_directional_uncertainty": finite_summary(
                    signal["observable_short_directional_uncertainty"]
                ),
                "long_directional_after": finite_summary(
                    signal["observable_long_directional_after"]
                ),
                "jacobian_norm": finite_summary(
                    signal["observable_jacobian_norm"]
                ),
                "singular_value_ratio": finite_summary(
                    signal["observable_singular_value_ratio"]
                ),
                "fd_scale_cosine": finite_summary(
                    signal["observable_fd_scale_cosine"]
                ),
                "log_distance_step": finite_summary(
                    signal["observable_log_distance_step"]
                ),
                "log_damping_step": finite_summary(
                    signal["observable_log_damping_step"]
                ),
            }
        with np.load(artifact / "material_diagnostics.npz") as material:
            distance = np.asarray(
                material["distance_stiffness_per_particle"], dtype=np.float64
            )
            results[capability]["distance_initial_final"] = [
                float(np.nanmedian(distance[0])),
                float(np.nanmedian(distance[-1])),
            ]
        metadata = json.loads(
            (artifact / "artifact_metadata.json").read_text(encoding="utf-8")
        )
        settings = metadata["online_stiffness_settings"]
        results[capability]["metadata"] = {
            "candidate_count": int(settings["candidate_count"]),
            "committed_update_count": int(settings["committed_update_count"]),
            "rejected_update_count": int(settings["rejected_update_count"]),
            "window_size": int(settings["observable_window_size"]),
            "minimum_transitions": int(
                settings["observable_minimum_transitions"]
            ),
            "minimum_singular_ratio": float(
                settings["observable_minimum_singular_ratio"]
            ),
        }

    payload = {
        "schema": "fixedsuperbest.sim_observable_global_mhe_diagnostics.v1",
        "method": method,
        "results": results,
    }
    (root / "observable_mhe_diagnostics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 全局可观测 MHE 刚度诊断",
        "",
        "| 能力 | 候选帧 | 可观测通过帧 | 刚度提交 | 奇异值比 median | FD尺度余弦 median | distance 初值→末值 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for capability, value in results.items():
        ratio = value["singular_value_ratio"]["median"]
        cosine = value["fd_scale_cosine"]["median"]
        initial, final = value["distance_initial_final"]
        lines.append(
            f"| {capability} | {value['candidate_frames']} | "
            f"{value['observable_allowed_frames']} | {value['committed_updates']} | "
            f"{float('nan') if ratio is None else ratio:.4f} | "
            f"{float('nan') if cosine is None else cosine:.4f} | "
            f"{initial:.6f}→{final:.6f} |"
        )
    (root / "observable_mhe_diagnostics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
