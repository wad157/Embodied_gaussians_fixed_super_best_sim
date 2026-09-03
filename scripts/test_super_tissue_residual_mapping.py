#!/usr/bin/env python3
"""CPU gates for deterministic volumetric tissue residual mapping."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embodied_gaussians.physics_simulator.tissue_residual_mapping import (  # noqa: E402
    TetrahedralTissueResidualMapper,
    TissueResidualMappingSettings,
)


def main() -> None:
    # Two 3x3 surface layers with a fixed bottom layer.
    xy = np.asarray(
        [(x, y) for y in range(3) for x in range(3)], dtype=np.float64
    )
    top = np.column_stack((xy * 0.001, np.full(9, 0.001)))
    bottom = np.column_stack((xy * 0.001, np.zeros(9)))
    rest = np.concatenate((top, bottom), axis=0)
    top_mask = np.zeros(18, dtype=bool)
    top_mask[:9] = True
    fixed = np.zeros(18, dtype=bool)
    fixed[9:] = True
    inward = np.concatenate((np.zeros(9), np.full(9, 0.001)))
    edges = []
    for y in range(3):
        for x in range(3):
            node = y * 3 + x
            if x < 2:
                edges.append((node, node + 1))
            if y < 2:
                edges.append((node, node + 3))
    observation0 = top.copy()
    mapper = TetrahedralTissueResidualMapper(
        rest_positions=rest,
        top_node_mask=top_mask,
        fixed_mask=fixed,
        inward_depth=inward,
        top_edges=np.asarray(edges, dtype=np.int32),
        initial_observation=observation0,
        settings=TissueResidualMappingSettings(
            iterations=20,
            maximum_residual_m=0.002,
        ),
    )
    observation1 = observation0.copy()
    observation1[4, 2] += 0.0005
    residual_a, history_a, metrics_a = mapper.map(
        physical_positions=rest,
        observation_points=observation1,
        previous_top_residual=None,
    )
    residual_b, history_b, metrics_b = mapper.map(
        physical_positions=rest,
        observation_points=observation1,
        previous_top_residual=None,
    )
    residual_decay, _history_decay, _metrics_decay = mapper.map(
        physical_positions=rest,
        observation_points=None,
        previous_top_residual=history_a,
    )
    gates = {
        "all_finite": bool(
            np.isfinite(residual_a).all() and np.isfinite(history_a).all()
        ),
        "deterministic_full_residual": bool(
            np.array_equal(residual_a, residual_b)
        ),
        "deterministic_history": bool(np.array_equal(history_a, history_b)),
        "visible_surface_moves_toward_observation": bool(
            residual_a[4, 2] > 0.0
        ),
        "surface_graph_spreads_locally": bool(
            residual_a[1, 2] > 0.0 and residual_a[1, 2] < residual_a[4, 2]
        ),
        "fixed_particles_receive_zero": bool(
            np.count_nonzero(residual_a[fixed]) == 0
        ),
        "missing_observation_decays_history": bool(
            np.linalg.norm(residual_decay)
            < np.linalg.norm(residual_a)
        ),
        "residual_cap_respected": bool(
            metrics_a["maximum_residual_m"] <= 0.002 + 1.0e-12
        ),
        "repeat_metrics_match": bool(metrics_a == metrics_b),
    }
    report = {
        "schema": "super_tissue_residual_mapping_cpu_gate_v1",
        "passed": all(gates.values()),
        "gates": gates,
        "metrics": metrics_a,
    }
    output = (
        REPO_ROOT
        / "data/super/tissue_calibration_v1/"
        "stage_e_residual_mapping_cpu_gate.json"
    )
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Residual mapping CPU gate failed")


if __name__ == "__main__":
    main()
