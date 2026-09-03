#!/usr/bin/env python3
"""验证统一夹持核心边界不含材料或外围运动真值。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset.expanduser().resolve()
    boundary_path = root / "task_inputs" / "known_grasp_region_boundary.npz"
    report_path = root / "task_inputs" / "known_grasp_region_boundary.json"
    tissue_path = root / "gui_assets" / "tissue_fixedsuperbest.npz"
    truth_path = root / "ground_truth" / "tissue_state.npz"

    with np.load(boundary_path, allow_pickle=False) as boundary:
        schema = str(np.asarray(boundary["schema"]).item())
        ids = np.asarray(
            boundary["reconstruction_particle_ids"], dtype=np.int64
        )
        trajectory = np.asarray(
            boundary["trajectory_positions_world"], dtype=np.float64
        )
        grasped = np.asarray(boundary["grasped"], dtype=bool)
        source_ids = np.asarray(
            boundary["source_truth_grasp_node_ids"], dtype=np.int64
        )
        forbidden_fields = {
            "youngs_modulus_pa_per_simulation_node",
            "simulation_region_ids",
            "distance_stiffness",
            "shape_stiffness",
        }.intersection(boundary.files)
    with np.load(tissue_path, allow_pickle=False) as tissue:
        rest = np.asarray(tissue["rest_positions_table"], dtype=np.float64)
        top = np.asarray(tissue["top_node_mask"], dtype=bool)
        fixed = np.asarray(tissue["fixed_mask"], dtype=bool)
    with np.load(truth_path, allow_pickle=False) as truth:
        truth_grasp = np.asarray(truth["grasp_mask"], dtype=bool)

    displacement = trajectory - rest[ids][None]
    grasp_displacement_spread = np.linalg.norm(
        displacement[grasped]
        - displacement[grasped].mean(axis=1, keepdims=True),
        axis=2,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    gates = {
        "schema": schema
        == "fixedsuperbest.known_grasp_region_boundary.v1",
        "twelve_unique_reconstruction_nodes": len(ids) == 12
        and len(np.unique(ids)) == 12,
        "top_and_bottom_are_both_controlled": bool(np.any(top[ids]))
        and not bool(np.all(top[ids])),
        "no_fixed_node_is_controlled": not bool(np.any(fixed[ids])),
        "source_is_exact_truth_grasp_core": np.array_equal(
            np.sort(source_ids), np.flatnonzero(truth_grasp)
        ),
        "closed_grasp_is_uniform_boundary": float(
            grasp_displacement_spread.max()
        )
        < 1.0e-8,
        "no_material_fields_are_exported": not forbidden_fields,
        "report_declares_no_material_truth": report.get(
            "uses_material_ground_truth"
        )
        is False,
        "report_declares_no_non_grasp_motion_truth": report.get(
            "uses_non_grasp_motion_ground_truth"
        )
        is False,
    }
    result = {"gates": gates, "passed": all(gates.values())}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
