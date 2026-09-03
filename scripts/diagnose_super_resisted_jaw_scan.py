#!/usr/bin/env python3
"""Scan q7 at the grasp pose and report bilateral closure resistance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import warp as wp


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from embodied_environments.super_embodied.super_embodied import (  # noqa: E402
    PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_UNBOUNDED_XYZ_POSE_DRIVER_PATH,
    apply_psm_lnd_pose,
    build_environment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--state-index", type=int, default=1807)
    parser.add_argument(
        "--world-translation-mm",
        nargs=3,
        type=float,
        default=(-0.2, 0.0, -1.5),
    )
    parser.add_argument("--angles", default="0.50,0.45,0.40,0.35,0.30,0.25,0.20,0.15,0.10,0.05,0.00,-0.05,-0.10")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wp.config.kernel_cache_dir = "/tmp/warp-super-resisted-jaw-scan"
    wp.init()
    environment = build_environment(
        add_gaussians=True,
        device=args.device,
        tissue_mode="paper_soft",
        psm_pose_driver_path=(
            PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_UNBOUNDED_XYZ_POSE_DRIVER_PATH
        ),
        psm_visual_tip_only=False,
    )
    state_index = max(
        0,
        min(
            int(args.state_index),
            len(environment.super_psm_q7_states) - 1,
        ),
    )
    raw_q7 = np.asarray(
        environment.super_psm_q7_states[state_index], dtype=np.float64
    )
    translation = np.asarray(
        args.world_translation_mm, dtype=np.float64
    ) / 1000.0
    angles = [float(value) for value in args.angles.split(",")]
    settings = environment.physics_settings
    projector = environment.sim.triangle_skin_contact_projector
    rows = []
    for actual_angle in angles:
        offsets = np.zeros(7, dtype=np.float64)
        offsets[6] = actual_angle - raw_q7[6]
        apply_psm_lnd_pose(
            environment,
            state_index,
            joint_offsets=offsets,
            translation_offset=translation,
            closing_requested=True,
            update_gaussians=False,
        )
        projector.detect(
            environment.sim.model,
            environment.sim.state_0,
            settings.dt / settings.substeps,
            contact_margin_m=settings.triangle_skin_contact_margin_m,
            query_distance_m=settings.triangle_skin_query_distance_m,
            ccd_velocity_scale=0.0,
            friction_coefficient=settings.triangle_skin_friction,
            relaxation=1.0,
        )
        feedback = projector.jaw_closure_resistance_metrics()
        contact_metrics = projector.metrics()
        jaw_a_counts = projector.persistent_grip_jaw_a_counts.numpy()
        jaw_b_counts = projector.persistent_grip_jaw_b_counts.numpy()
        repeated_hit_counts = {
            str(threshold): int(
                np.count_nonzero(
                    (jaw_a_counts >= threshold)
                    & (jaw_b_counts >= threshold)
                )
            )
            for threshold in (
                1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64,
            )
        }
        rows.append(
            {
                "actual_q7_rad": actual_angle,
                "closing_contacts_left_right": list(
                    feedback["contact_counts"]
                ),
                "resistance_left_right_m2": list(
                    feedback["resistance_m2"]
                ),
                "maximum_penetration_mm": (
                    1000.0 * contact_metrics["maximum_penetration_m"]
                ),
                "between_jaw_particle_candidates": contact_metrics[
                    "persistent_grip_between_jaw_candidate_count"
                ],
                "between_jaw_candidates_by_min_hits_per_jaw": (
                    repeated_hit_counts
                ),
                "grip_active": contact_metrics["persistent_grip_active"],
                "bilateral_block": all(
                    count >= 8 and resistance > 0.0
                    for count, resistance in zip(
                        feedback["contact_counts"],
                        feedback["resistance_m2"],
                        strict=True,
                    )
                ),
            }
        )
    print(
        json.dumps(
            {
                "state_index": state_index,
                "raw_q7_rad": float(raw_q7[6]),
                "world_translation_mm": list(args.world_translation_mm),
                "scan": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
