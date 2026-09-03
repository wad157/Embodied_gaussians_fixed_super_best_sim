#!/usr/bin/env python3
"""固定选择覆盖组织表面且远离夹爪接触区的仿真轨迹评估点。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CAMERAS = ("stereo_left", "stereo_right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument(
        "--boundary",
        type=Path,
        default=None,
        help="夹持排除边界；默认优先使用 known_grasp_region_boundary.npz。",
    )
    parser.add_argument(
        "--minimum-visible-fraction",
        type=float,
        default=0.80,
        help="每只眼在完整序列中的最低真值可见比例。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖固定评估点清单：{output}")
    if args.count <= 0:
        raise ValueError("--count 必须为正整数")
    if not 0.0 <= args.minimum_visible_fraction <= 1.0:
        raise ValueError("--minimum-visible-fraction 必须在 [0,1]")

    reference = np.load(root / "ground_truth/trajectories_3d.npz")
    all_node_ids = reference["tissue_node_ids"].astype(np.int64)
    candidate_ids = reference["tissue_evaluation_node_ids"].astype(np.int64)
    positions0 = reference["tissue_positions_world"][0].astype(np.float64)
    node_lookup = {int(node_id): index for index, node_id in enumerate(all_node_ids)}
    candidate_columns = np.asarray(
        [node_lookup[int(node_id)] for node_id in candidate_ids], dtype=np.int64
    )
    candidate_positions = positions0[candidate_columns]

    boundary_path = args.boundary
    if boundary_path is None:
        known = root / "task_inputs/known_grasp_region_boundary.npz"
        boundary_path = (
            known if known.is_file() else root / "task_inputs/red_marker_boundary.npz"
        )
    boundary_path = boundary_path.expanduser().resolve()
    with np.load(boundary_path, allow_pickle=False) as boundary:
        controlled_ids = boundary[
            "evaluation_exclusion_tissue_node_ids"
        ].astype(np.int64)
        if "grasp_region_center_world" in boundary.files:
            contact_center = boundary["grasp_region_center_world"].astype(
                np.float64
            )
            xy_min = boundary["grasp_region_xy_min"].astype(np.float64)
            xy_max = boundary["grasp_region_xy_max"].astype(np.float64)
            half_diagonal = float(
                np.max(
                    np.linalg.norm(
                        np.stack((xy_min, xy_max))
                        - contact_center[:2][None, :],
                        axis=1,
                    )
                )
            )
            # 再排除约一个重建粒子间距，避免评估点落在受控夹持环上。
            exclusion_radius = half_diagonal + 0.0025
        else:
            contact_center = boundary["red_marker_center_world"].astype(
                np.float64
            )
            exclusion_radius = float(boundary["evaluation_exclusion_radius_m"])

    distance_to_contact = np.linalg.norm(
        candidate_positions - contact_center[None, :], axis=1
    )
    allowed = (
        ~np.isin(candidate_ids, controlled_ids)
        & (distance_to_contact > exclusion_radius)
    )

    visibility: dict[str, np.ndarray] = {}
    for camera in CAMERAS:
        tracks = np.load(
            root / "ground_truth/trajectories_2d" / f"{camera}.npz"
        )
        visible = tracks["tissue_visible"][:, candidate_columns].astype(bool)
        visibility[camera] = visible.mean(axis=0)
        allowed &= visibility[camera] >= args.minimum_visible_fraction

    allowed_indices = np.flatnonzero(allowed)
    if len(allowed_indices) < args.count:
        raise ValueError(
            f"夹持/可见性排除后只有 {len(allowed_indices)} 个候选，"
            f"少于要求的 {args.count} 个"
        )

    # 仅使用首帧 xy 几何进行确定性最远点采样，不读取任何方法预测或误差。
    xy = candidate_positions[allowed_indices, :2]
    ids = candidate_ids[allowed_indices]
    center = xy.mean(axis=0)
    center_distance = np.linalg.norm(xy - center[None, :], axis=1)
    first_order = np.lexsort((ids, center_distance))
    selected_local = [int(first_order[0])]
    minimum_distance = np.linalg.norm(
        xy - xy[selected_local[0]][None, :], axis=1
    )
    while len(selected_local) < args.count:
        available = np.ones(len(xy), dtype=bool)
        available[selected_local] = False
        best_distance = minimum_distance.copy()
        best_distance[~available] = -np.inf
        maximum = float(best_distance.max())
        ties = np.flatnonzero(np.isclose(best_distance, maximum, atol=1.0e-12))
        next_local = int(ties[np.argmin(ids[ties])])
        selected_local.append(next_local)
        minimum_distance = np.minimum(
            minimum_distance,
            np.linalg.norm(xy - xy[next_local][None, :], axis=1),
        )

    selected_indices = allowed_indices[np.asarray(selected_local, dtype=np.int64)]
    selected_ids = candidate_ids[selected_indices]
    selected_positions = candidate_positions[selected_indices]
    records = []
    for index, node_id, position in zip(
        selected_indices, selected_ids, selected_positions
    ):
        records.append(
            {
                "tissue_node_id": int(node_id),
                "rest_position_world_m": position.tolist(),
                "distance_to_grasp_center_mm": float(
                    distance_to_contact[index] * 1.0e3
                ),
                "visibility_fraction": {
                    camera: float(visibility[camera][index])
                    for camera in CAMERAS
                },
            }
        )

    payload = {
        "schema": "fixedsuperbest.sim_evaluation_points.v1",
        "dataset": str(root),
        "count": int(len(selected_ids)),
        "tissue_node_ids": selected_ids.tolist(),
        "selection": (
            "deterministic farthest-point sampling on frame-0 world xy; "
            "prediction/error independent"
        ),
        "candidate_nodes_before_exclusion": int(len(candidate_ids)),
        "candidate_nodes_after_exclusion": int(len(allowed_indices)),
        "contact_exclusion": {
            "source": str(boundary_path),
            "controlled_node_ids": controlled_ids.tolist(),
            "center_world_m": contact_center.tolist(),
            "radius_mm": exclusion_radius * 1.0e3,
            "strictly_outside_radius": True,
        },
        "minimum_visible_fraction_per_camera": float(
            args.minimum_visible_fraction
        ),
        "points": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
