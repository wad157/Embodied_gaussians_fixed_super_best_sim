#!/usr/bin/env python3
"""将仿真器的完整夹持核心轨迹映射为重建器的统一已知边界。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCHEMA = "fixedsuperbest.known_grasp_region_boundary.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--xy-margin-m",
        type=float,
        default=1.0e-7,
        help="夹持核心XY包围盒的数值容差，默认1e-7 m。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset.expanduser().resolve()
    state_path = root / "ground_truth" / "tissue_state.npz"
    tissue_path = root / "gui_assets" / "tissue_fixedsuperbest.npz"
    phase_path = root / "task_inputs" / "phases.json"
    legacy_boundary_path = root / "task_inputs" / "red_marker_boundary.npz"
    for path in (state_path, tissue_path, phase_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(state_path, allow_pickle=False) as loaded:
        timestamps = np.asarray(loaded["timestamps"], dtype=np.float64)
        truth_positions = np.asarray(
            loaded["simulation_positions"], dtype=np.float64
        )
        truth_rest = np.asarray(
            loaded["simulation_rest_local"], dtype=np.float64
        )
        truth_grasp_mask = np.asarray(loaded["grasp_mask"], dtype=bool)
        coupling_weights = np.asarray(
            loaded["grasp_coupling_weights"], dtype=np.float64
        )
    with np.load(tissue_path, allow_pickle=False) as loaded:
        reconstruction_rest = np.asarray(
            loaded["rest_positions_table"], dtype=np.float64
        )
        reconstruction_surface = np.asarray(
            loaded["surface_node_mask"], dtype=bool
        )
        reconstruction_top = np.asarray(loaded["top_node_mask"], dtype=bool)
        reconstruction_fixed = np.asarray(loaded["fixed_mask"], dtype=bool)

    phases = json.loads(phase_path.read_text(encoding="utf-8"))
    grasped = np.asarray(
        [bool(record["grasped"]) for record in phases], dtype=bool
    )
    if len(grasped) != len(timestamps):
        raise ValueError("grasp phase与真值轨迹帧数不一致")
    if not bool(truth_grasp_mask.any()):
        raise ValueError("真值组织没有夹持核心")
    if not np.array_equal(truth_grasp_mask, coupling_weights == 1.0):
        raise ValueError("夹持核心与权重1.0节点不一致")
    evaluation_exclusion_ids = np.empty(0, dtype=np.int32)
    if legacy_boundary_path.is_file():
        with np.load(legacy_boundary_path, allow_pickle=False) as loaded:
            if "evaluation_exclusion_tissue_node_ids" in loaded.files:
                evaluation_exclusion_ids = np.asarray(
                    loaded["evaluation_exclusion_tissue_node_ids"],
                    dtype=np.int32,
                )

    truth_grasp_ids = np.flatnonzero(truth_grasp_mask).astype(np.int32)
    core_rest = truth_rest[truth_grasp_ids]
    xy_min = core_rest[:, :2].min(axis=0)
    xy_max = core_rest[:, :2].max(axis=0)
    margin = float(args.xy_margin_m)
    reconstruction_core = (
        reconstruction_surface
        & ~reconstruction_fixed
        & np.all(reconstruction_rest[:, :2] >= xy_min[None] - margin, axis=1)
        & np.all(reconstruction_rest[:, :2] <= xy_max[None] + margin, axis=1)
    )
    reconstruction_ids = np.flatnonzero(reconstruction_core).astype(np.int32)
    if len(reconstruction_ids) < 4:
        raise RuntimeError(
            "映射到重建网格的夹持核心节点不足："
            f"{len(reconstruction_ids)}"
        )
    top_count = int(np.count_nonzero(reconstruction_top[reconstruction_ids]))
    bottom_count = int(len(reconstruction_ids) - top_count)
    if top_count == 0 or bottom_count == 0:
        raise RuntimeError("完整夹持核心必须同时包含上下表面节点")

    truth_displacement = (
        truth_positions[:, truth_grasp_ids]
        - truth_rest[None, truth_grasp_ids]
    )
    mean_displacement = truth_displacement.mean(axis=1)
    core_rigid_deviation = np.linalg.norm(
        truth_displacement - mean_displacement[:, None], axis=2
    )
    grasp_deviation = core_rigid_deviation[grasped]
    maximum_rigid_deviation_m = float(
        grasp_deviation.max() if grasp_deviation.size else 0.0
    )
    if maximum_rigid_deviation_m > 1.0e-5:
        raise RuntimeError(
            "真值夹持核心不是统一运动边界：最大节点偏差"
            f"{maximum_rigid_deviation_m * 1e3:.6f} mm"
        )

    trajectory = (
        reconstruction_rest[reconstruction_ids][None]
        + mean_displacement[:, None]
    ).astype(np.float32)
    velocities = np.gradient(trajectory, timestamps, axis=0).astype(np.float32)

    output = root / "task_inputs" / "known_grasp_region_boundary.npz"
    report_path = root / "task_inputs" / "known_grasp_region_boundary.json"
    np.savez_compressed(
        output,
        schema=np.asarray(SCHEMA),
        timestamps=timestamps,
        grasped=grasped,
        reconstruction_particle_ids=reconstruction_ids,
        trajectory_positions_world=trajectory,
        trajectory_velocities_world=velocities,
        grasp_region_center_world=core_rest.mean(axis=0).astype(np.float32),
        grasp_region_xy_min=xy_min.astype(np.float32),
        grasp_region_xy_max=xy_max.astype(np.float32),
        source_truth_grasp_node_ids=truth_grasp_ids,
        source_truth_mean_displacement_world=mean_displacement.astype(np.float32),
        source_truth_grasp_coupling_weights=coupling_weights[
            truth_grasp_ids
        ].astype(np.float32),
        reconstruction_top_mask=reconstruction_top[reconstruction_ids],
        evaluation_exclusion_tissue_node_ids=evaluation_exclusion_ids,
    )
    report = {
        "schema": SCHEMA,
        "runtime_input": str(output.relative_to(root)),
        "construction_source": str(state_path.relative_to(root)),
        "semantics": (
            "只将数据生成器中权重等于1的8x8 mm夹持核心映射为重建Dirichlet"
            "边界；外围余弦耦合区不读取真值位置，仍由PBD、RGB残差和在线刚度决定"
        ),
        "truth_grasp_particle_count": int(len(truth_grasp_ids)),
        "controlled_reconstruction_particle_count": int(len(reconstruction_ids)),
        "controlled_top_particle_count": top_count,
        "controlled_bottom_particle_count": bottom_count,
        "controlled_reconstruction_particle_ids": reconstruction_ids.tolist(),
        "truth_grasp_particle_ids": truth_grasp_ids.tolist(),
        "grasp_region_xy_bounds_m": [xy_min.tolist(), xy_max.tolist()],
        "first_grasp_frame": int(np.flatnonzero(grasped)[0]),
        "last_grasp_frame": int(np.flatnonzero(grasped)[-1]),
        "maximum_truth_core_nonrigid_deviation_mm": (
            maximum_rigid_deviation_m * 1.0e3
        ),
        "uses_material_ground_truth": False,
        "uses_non_grasp_motion_ground_truth": False,
        "evaluation_exclusion_tissue_node_ids": (
            evaluation_exclusion_ids.tolist()
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "[known grasp boundary] 完成："
        f"GT核心{len(truth_grasp_ids)}点 -> 重建{len(reconstruction_ids)}点 "
        f"(top={top_count}, bottom={bottom_count}); {output}"
    )


if __name__ == "__main__":
    main()
