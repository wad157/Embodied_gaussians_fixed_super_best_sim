#!/usr/bin/env python3
"""把红色抓取标记的小片区真值轨迹导出为重建位移边界。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--controlled-node-count",
        type=int,
        default=5,
        help="红色标记中心及最近邻中严格受控的重建顶面节点数，默认 5。",
    )
    parser.add_argument(
        "--red-score-threshold",
        type=float,
        default=0.40,
        help="Gaussian 红色分数 R-max(G,B) 门限。",
    )
    parser.add_argument(
        "--marker-face-radius-m",
        type=float,
        default=0.0022,
        help="真值网格红色材质子集使用的面中心半径。",
    )
    parser.add_argument(
        "--maximum-control-radius-m",
        type=float,
        default=0.0038,
        help="受控重建节点到红色标记中心的最大半径，默认 3.8 mm。",
    )
    parser.add_argument(
        "--evaluation-exclusion-radius-m",
        type=float,
        default=0.0050,
        help="轨迹评估中排除五点已知边界小邻域的半径，默认 5 mm。",
    )
    return parser.parse_args()


def barycentric_xy(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Return planar barycentric weights for one nearly horizontal face."""
    a, b, c = np.asarray(triangle, dtype=np.float64)[:, :2]
    v0 = b - a
    v1 = c - a
    v2 = np.asarray(point, dtype=np.float64)[:2] - a
    d00 = float(v0 @ v0)
    d01 = float(v0 @ v1)
    d11 = float(v1 @ v1)
    d20 = float(v2 @ v0)
    d21 = float(v2 @ v1)
    denominator = d00 * d11 - d01 * d01
    if abs(denominator) <= 1.0e-20:
        raise ValueError("红色标记包含退化三角形")
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    return np.asarray((1.0 - v - w, v, w), dtype=np.float64)


def closest_barycentric_xy(
    point: np.ndarray, triangle: np.ndarray
) -> tuple[np.ndarray, float]:
    """Closest point weights and planar distance on a triangle."""
    point_xy = np.asarray(point, dtype=np.float64)[:2]
    triangle_xy = np.asarray(triangle, dtype=np.float64)[:, :2]
    unconstrained = barycentric_xy(point_xy, triangle_xy)
    if float(unconstrained.min()) >= 0.0:
        return unconstrained, 0.0
    best_weights = None
    best_distance = float("inf")
    for start, end in ((0, 1), (1, 2), (2, 0)):
        edge = triangle_xy[end] - triangle_xy[start]
        denominator = float(edge @ edge)
        if denominator <= 1.0e-20:
            continue
        alpha = float(
            np.clip(
                ((point_xy - triangle_xy[start]) @ edge) / denominator,
                0.0,
                1.0,
            )
        )
        projected = triangle_xy[start] + alpha * edge
        distance = float(np.linalg.norm(point_xy - projected))
        if distance < best_distance:
            weights = np.zeros(3, dtype=np.float64)
            weights[start] = 1.0 - alpha
            weights[end] = alpha
            best_weights = weights
            best_distance = distance
    if best_weights is None:
        raise ValueError("红色标记包含不可投影三角形")
    return best_weights, best_distance


def main() -> None:
    args = parse_args()
    if args.controlled_node_count < 1:
        raise ValueError("--controlled-node-count 必须为正整数")
    root = args.dataset.expanduser().resolve()
    tissue_path = root / "gui_assets" / "tissue_fixedsuperbest.npz"
    state_path = root / "ground_truth" / "tissue_state.npz"
    phase_path = root / "task_inputs" / "phases.json"
    for path in (tissue_path, state_path, phase_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(tissue_path, allow_pickle=False) as loaded:
        reconstruction_rest = np.asarray(
            loaded["rest_positions_table"], dtype=np.float64
        )
        reconstruction_top = np.asarray(loaded["top_node_mask"], dtype=bool)
        reconstruction_fixed = np.asarray(loaded["fixed_mask"], dtype=bool)
        gaussian_means = np.asarray(
            loaded["gaussian_rest_means_table"], dtype=np.float64
        )
        gaussian_colors = np.asarray(
            loaded["gaussian_colors_rgb"], dtype=np.float64
        )
    with np.load(state_path, allow_pickle=False) as loaded:
        timestamps = np.asarray(loaded["timestamps"], dtype=np.float64)
        truth_positions = np.asarray(
            loaded["simulation_positions"], dtype=np.float64
        )
        truth_rest = np.asarray(
            loaded["simulation_rest_local"], dtype=np.float64
        )
        truth_faces = np.asarray(loaded["visual_faces"], dtype=np.int64)
        truth_top_faces = np.asarray(
            loaded["top_face_indices"], dtype=np.int64
        )

    red_score = gaussian_colors[:, 0] - np.maximum(
        gaussian_colors[:, 1], gaussian_colors[:, 2]
    )
    red_gaussians = (
        (red_score >= float(args.red_score_threshold))
        & (gaussian_colors[:, 0] >= 0.70)
    )
    if int(np.count_nonzero(red_gaussians)) < 3:
        raise RuntimeError("重建资产中没有稳定识别到红色抓取标记")
    marker_center = gaussian_means[red_gaussians].mean(axis=0)

    truth_face_centers = truth_rest[truth_faces[truth_top_faces]].mean(axis=1)
    marker_face_mask = (
        np.linalg.norm(
            truth_face_centers[:, :2] - marker_center[None, :2], axis=1
        )
        <= float(args.marker_face_radius_m)
    )
    marker_face_ids = truth_top_faces[marker_face_mask]
    if len(marker_face_ids) == 0:
        raise RuntimeError("红色标记没有匹配到真值表面三角形")

    eligible_reconstruction = np.flatnonzero(
        reconstruction_top & ~reconstruction_fixed
    )
    nearby: list[tuple[int, int, np.ndarray, float, float]] = []
    for reconstruction_id in eligible_reconstruction:
        point = reconstruction_rest[reconstruction_id]
        candidates = []
        for face_id in marker_face_ids:
            weights, surface_distance = closest_barycentric_xy(
                point, truth_rest[truth_faces[face_id]]
            )
            candidates.append((int(face_id), weights, surface_distance))
        face_id, weights, surface_distance = min(
            candidates, key=lambda item: item[2]
        )
        center_distance = float(
            np.linalg.norm(point[:2] - marker_center[:2])
        )
        if center_distance <= float(args.maximum_control_radius_m):
            nearby.append(
                (
                    int(reconstruction_id),
                    face_id,
                    weights,
                    center_distance,
                    surface_distance,
                )
            )
    nearby.sort(key=lambda item: item[3])
    if len(nearby) < args.controlled_node_count:
        raise RuntimeError(
            "红色标记小邻域内的重建动态节点不足："
            f"需要 {args.controlled_node_count}，实际 {len(nearby)}"
        )
    selected = nearby[: args.controlled_node_count]
    reconstruction_ids = np.asarray(
        [item[0] for item in selected], dtype=np.int32
    )
    source_face_ids = np.asarray(
        [item[1] for item in selected], dtype=np.int32
    )
    source_node_ids = truth_faces[source_face_ids].astype(np.int32)
    barycentric_weights = np.stack(
        [item[2] for item in selected], axis=0
    ).astype(np.float32)
    trajectory = np.einsum(
        "fnik,ni->fnk",
        truth_positions[:, source_node_ids, :],
        barycentric_weights,
        optimize=True,
    ).astype(np.float32)
    velocities = np.gradient(trajectory, timestamps, axis=0).astype(np.float32)

    phases = json.loads(phase_path.read_text(encoding="utf-8"))
    grasped = np.asarray([bool(record["grasped"]) for record in phases])
    if len(grasped) != len(timestamps):
        raise ValueError("grasp phase 与红色标记轨迹帧数不一致")

    # Exclude only the prescribed marker patch and a small local margin.
    # from scoring. This prevents a known boundary from becoming prediction
    # credit while retaining nearly all of the fixed evaluation set.
    truth_top_ids = np.unique(
        truth_faces[truth_top_faces].reshape(-1)
    ).astype(np.int32)
    evaluation_exclusion_ids = truth_top_ids[
        np.linalg.norm(
            truth_rest[truth_top_ids, :2] - marker_center[None, :2], axis=1
        )
        <= float(args.evaluation_exclusion_radius_m)
    ]
    output = root / "task_inputs" / "red_marker_boundary.npz"
    report_path = root / "task_inputs" / "red_marker_boundary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema=np.asarray("fixedsuperbest.red_marker_boundary.v2"),
        timestamps=timestamps,
        grasped=grasped,
        reconstruction_particle_ids=reconstruction_ids,
        trajectory_positions_world=trajectory,
        trajectory_velocities_world=velocities,
        source_triangle_face_ids=source_face_ids,
        source_triangle_node_ids=source_node_ids,
        source_barycentric_weights=barycentric_weights,
        red_marker_center_world=marker_center.astype(np.float32),
        red_marker_face_ids=marker_face_ids.astype(np.int32),
        red_marker_radius_m=np.asarray(
            args.marker_face_radius_m, dtype=np.float32
        ),
        maximum_control_radius_m=np.asarray(
            args.maximum_control_radius_m, dtype=np.float32
        ),
        evaluation_exclusion_radius_m=np.asarray(
            args.evaluation_exclusion_radius_m, dtype=np.float32
        ),
        evaluation_exclusion_tissue_node_ids=evaluation_exclusion_ids,
    )
    first_grasp = int(np.flatnonzero(grasped)[0])
    final_motion = np.linalg.norm(
        trajectory[-1] - trajectory[first_grasp], axis=1
    )
    report = {
        "schema": "fixedsuperbest.red_marker_boundary.v2",
        "runtime_input": str(output.relative_to(root)),
        "construction_source": str(state_path.relative_to(root)),
        "controlled_reconstruction_particle_ids": reconstruction_ids.tolist(),
        "controlled_particle_count": int(len(reconstruction_ids)),
        "red_gaussian_count": int(np.count_nonzero(red_gaussians)),
        "red_marker_center_world_m": marker_center.tolist(),
        "source_triangle_face_ids": source_face_ids.tolist(),
        "source_triangle_node_ids": source_node_ids.tolist(),
        "source_barycentric_weights": barycentric_weights.tolist(),
        "controlled_center_distance_mm": [
            item[3] * 1.0e3 for item in selected
        ],
        "source_surface_projection_distance_mm": [
            item[4] * 1.0e3 for item in selected
        ],
        "first_grasp_frame": first_grasp,
        "final_prescribed_motion_mm": (final_motion * 1.0e3).tolist(),
        "evaluation_exclusion_tissue_node_ids": (
            evaluation_exclusion_ids.tolist()
        ),
        "semantics": (
            "闭合期已知的红色标记五点小片区位移边界；运行时只读取 task_inputs，"
            "周围节点不读取 GT 轨迹且不接受运动学覆盖，"
            "全部由 PBD、RGB residual 与在线刚度决定"
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
