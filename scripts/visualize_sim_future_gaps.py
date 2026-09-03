#!/usr/bin/env python3
"""生成仿照 Liang et al. Fig. 8 的因果 future-gap 空间图。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay


TASKS = {
    "front_pull": "tissue_retraction_free_support_front_v2",
    "side_pull": "tissue_retraction_free_support_side_v2",
}
METHODS = (
    ("pbd", "PBD"),
    ("pbd_visual_residual", "PBD + RGB residual"),
    ("pbd_visual_residual_stiffness", "PBD + RGB residual + stiffness"),
)
PRIMARY_ROWS = tuple(
    (task, a, 10) for task in TASKS for a in (170, 220)
)
HORIZON_ROWS = tuple(
    (task, 239, horizon) for task in TASKS for horizon in (10, 30, 60)
)


@dataclass(frozen=True)
class GapResult:
    task: str
    a: int
    b: int
    target: int
    method: str
    label: str
    node_ids: np.ndarray
    rest_positions_m: np.ndarray
    predicted_positions_m: np.ndarray
    reference_positions_m: np.ndarray
    gaps_mm: np.ndarray
    source: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/sim"))
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("outputs/two_free_support_ablation_v2"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/two_free_support_ablation_v2/visualizations"),
    )
    return parser.parse_args()


def prediction_path(
    evaluation_root: Path, output: Path, task: str, a: int, method: str
) -> Path:
    if method == "pbd" or a == 239:
        return evaluation_root / task / method / "artifacts/predicted_trajectories.npz"
    return (
        output
        / "future_gap_rollouts"
        / task
        / f"a_{a:03d}"
        / method
        / "artifacts/predicted_trajectories.npz"
    )


def load_gap_result(
    data_root: Path,
    evaluation_root: Path,
    output: Path,
    task: str,
    a: int,
    b: int,
    method: str,
    label: str,
) -> GapResult:
    target = a + b
    dataset = data_root / TASKS[task]
    source = prediction_path(evaluation_root, output, task, a, method)
    if not source.is_file():
        raise FileNotFoundError(f"缺少 future-gap rollout：{source}")
    with np.load(dataset / "ground_truth/trajectories_3d.npz", allow_pickle=False) as gt:
        gt_ids = np.asarray(gt["tissue_node_ids"], dtype=np.int32)
        gt_positions = np.asarray(gt["tissue_positions_world"], dtype=np.float64)
    with np.load(dataset / "task_inputs/red_marker_boundary.npz", allow_pickle=False) as boundary:
        excluded = set(
            int(value) for value in boundary["evaluation_exclusion_tissue_node_ids"]
        )
    with np.load(source, allow_pickle=False) as prediction:
        frames = np.asarray(prediction["frame_indices"], dtype=np.int32)
        slots = np.flatnonzero(frames == target)
        if len(slots) != 1:
            raise ValueError(f"{source} 中找不到唯一目标帧 {target}")
        node_ids_all = np.asarray(prediction["tissue_node_ids"], dtype=np.int32)
        keep = np.asarray([int(node) not in excluded for node in node_ids_all])
        node_ids = node_ids_all[keep]
        predicted = np.asarray(
            prediction["tissue_positions_world"][int(slots[0]), keep],
            dtype=np.float64,
        )
        rest = np.asarray(prediction["reference_first_positions_world"][keep], dtype=np.float64)
    lookup = {int(node): index for index, node in enumerate(gt_ids)}
    columns = np.asarray([lookup[int(node)] for node in node_ids], dtype=np.int64)
    reference = gt_positions[target, columns]
    if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        raise ValueError(f"{source} 在目标帧含非有限坐标")
    gaps = np.linalg.norm(predicted - reference, axis=1) * 1.0e3
    return GapResult(
        task=task,
        a=a,
        b=b,
        target=target,
        method=method,
        label=label,
        node_ids=node_ids,
        rest_positions_m=rest,
        predicted_positions_m=predicted,
        reference_positions_m=reference,
        gaps_mm=gaps,
        source=source,
    )


def triangulate(rest_positions_m: np.ndarray) -> np.ndarray:
    centered = rest_positions_m - rest_positions_m.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    coordinates_2d = centered @ vh[:2].T
    triangles = np.asarray(Delaunay(coordinates_2d).simplices, dtype=np.int32)
    edges = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]), axis=0
    )
    edge_lengths = np.linalg.norm(
        rest_positions_m[edges[:, 0]] - rest_positions_m[edges[:, 1]], axis=1
    )
    typical = float(np.median(edge_lengths))
    triangle_maximum = np.maximum.reduce(
        (
            np.linalg.norm(rest_positions_m[triangles[:, 0]] - rest_positions_m[triangles[:, 1]], axis=1),
            np.linalg.norm(rest_positions_m[triangles[:, 1]] - rest_positions_m[triangles[:, 2]], axis=1),
            np.linalg.norm(rest_positions_m[triangles[:, 2]] - rest_positions_m[triangles[:, 0]], axis=1),
        )
    )
    filtered = triangles[triangle_maximum <= max(typical * 3.5, 1.0e-6)]
    if len(filtered) < 20:
        raise ValueError("future-gap 稀疏表面三角化失败")
    return filtered


def crop_tissue_rgb(dataset: Path, target: int) -> np.ndarray:
    image = np.asarray(
        Image.open(dataset / f"rgb/stereo_left/{target:06d}.png").convert("RGB")
    )
    mask = np.asarray(
        Image.open(
            dataset / f"ground_truth/masks/tissue/stereo_left/{target:06d}.png"
        ).convert("L")
    ) > 0
    y, x = np.nonzero(mask)
    if len(x) == 0:
        return image
    padding = 50
    x0 = max(0, int(x.min()) - padding)
    x1 = min(image.shape[1], int(x.max()) + padding + 1)
    y0 = max(0, int(y.min()) - padding)
    y1 = min(image.shape[0], int(y.max()) + padding + 1)
    return image[y0:y1, x0:x1]


def common_limits(results: list[GapResult]) -> tuple[np.ndarray, float]:
    points = np.concatenate(
        [result.predicted_positions_m for result in results]
        + [results[0].reference_positions_m],
        axis=0,
    ) * 1.0e3
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    radius = max(float((points.max(axis=0) - points.min(axis=0)).max()) * 0.56, 1.0)
    return center, radius


def add_gap_mesh(
    ax: plt.Axes,
    result: GapResult,
    triangles: np.ndarray,
    center_mm: np.ndarray,
    radius_mm: float,
    norm: Normalize,
) -> None:
    xyz = result.predicted_positions_m * 1.0e3
    vertices = xyz[triangles]
    face_gap = result.gaps_mm[triangles].mean(axis=1)
    collection = Poly3DCollection(
        vertices,
        facecolors=plt.get_cmap("turbo")(norm(face_gap)),
        edgecolors=(0.05, 0.05, 0.05, 0.16),
        linewidths=0.18,
    )
    ax.add_collection3d(collection)
    ax.set_xlim(center_mm[0] - radius_mm, center_mm[0] + radius_mm)
    ax.set_ylim(center_mm[1] - radius_mm, center_mm[1] + radius_mm)
    ax.set_zlim(center_mm[2] - radius_mm, center_mm[2] + radius_mm)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=34, azim=-68)
    ax.set_axis_off()
    mean = float(result.gaps_mm.mean())
    rmse = float(np.sqrt(np.mean(np.square(result.gaps_mm))))
    p95 = float(np.percentile(result.gaps_mm, 95.0))
    ax.text2D(
        0.02, 0.02, f"Mean {mean:.2f} | RMSE {rmse:.2f} | P95 {p95:.2f} mm",
        transform=ax.transAxes, fontsize=7.5,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 2},
    )


def plot_rows(
    rows: tuple[tuple[str, int, int], ...],
    data_root: Path,
    evaluation_root: Path,
    output: Path,
    stem: str,
    title: str,
) -> tuple[list[GapResult], list[dict[str, object]]]:
    all_results: list[GapResult] = []
    summaries: list[dict[str, object]] = []
    fig = plt.figure(figsize=(16.5, 3.35 * len(rows)), constrained_layout=True)
    grid = fig.add_gridspec(len(rows), 5, width_ratios=(1.05, 1, 1, 1, 0.045))
    for row_index, (task, a, b) in enumerate(rows):
        target = a + b
        dataset = data_root / TASKS[task]
        results = [
            load_gap_result(
                data_root, evaluation_root, output, task, a, b, method, label
            )
            for method, label in METHODS
        ]
        all_results.extend(results)
        node_ids = results[0].node_ids
        if any(not np.array_equal(result.node_ids, node_ids) for result in results[1:]):
            raise ValueError(f"{task} a={a} 三种方法的评估点不一致")
        triangles = triangulate(results[0].rest_positions_m)
        row_maximum = max(float(result.gaps_mm.max()) for result in results)
        norm = Normalize(vmin=0.0, vmax=max(row_maximum, 1.0e-6))
        center, radius = common_limits(results)

        rgb_ax = fig.add_subplot(grid[row_index, 0])
        rgb_ax.imshow(crop_tissue_rgb(dataset, target))
        rgb_ax.axis("off")
        rgb_ax.set_title(f"GT RGB at t={target}", fontsize=10)
        rgb_ax.text(
            0.01, 0.02, f"{task}\na={a}, b={b}", transform=rgb_ax.transAxes,
            color="white", fontsize=9, weight="bold",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3},
        )
        for column, result in enumerate(results, start=1):
            ax = fig.add_subplot(grid[row_index, column], projection="3d")
            add_gap_mesh(ax, result, triangles, center, radius, norm)
            if row_index == 0:
                ax.set_title(result.label, fontsize=10)
            summary = {
                "task": task,
                "a": a,
                "b": b,
                "target": target,
                "method": result.method,
                "method_label": result.label,
                "evaluated_nodes": int(len(result.node_ids)),
                "mean_gap_mm": float(result.gaps_mm.mean()),
                "rmse_gap_mm": float(np.sqrt(np.mean(np.square(result.gaps_mm)))),
                "median_gap_mm": float(np.median(result.gaps_mm)),
                "p95_gap_mm": float(np.percentile(result.gaps_mm, 95.0)),
                "maximum_gap_mm": float(result.gaps_mm.max()),
                "prediction_source": str(result.source.resolve()),
            }
            summaries.append(summary)
        color_ax = fig.add_subplot(grid[row_index, 4])
        colorbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap="turbo"), cax=color_ax
        )
        colorbar.set_label("3D correspondence gap (mm)", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=15)
    fig.savefig(output / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    return all_results, summaries


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    primary_results, primary_summary = plot_rows(
        PRIMARY_ROWS,
        args.data_root,
        args.evaluation_root,
        args.output,
        "future_gap_fig8_b10",
        "Causal future-gap comparison (state/material frozen after frame a)",
    )
    horizon_results, horizon_summary = plot_rows(
        HORIZON_ROWS,
        args.data_root,
        args.evaluation_root,
        args.output,
        "future_gap_open_loop_horizons",
        "Open-loop future gaps from the last observed frame a=239",
    )
    all_results = primary_results + horizon_results
    summaries = primary_summary + horizon_summary
    csv_path = args.output / "future_gap_values.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("task", "a", "b", "target", "method", "node_id", "gap_mm"))
        for result in all_results:
            for node_id, gap in zip(result.node_ids, result.gaps_mm):
                writer.writerow(
                    (result.task, result.a, result.b, result.target, result.method, int(node_id), float(gap))
                )
    payload = {
        "schema": "fixedsuperbest.sim_future_gap_visualization.v1",
        "reference": "https://arxiv.org/abs/2309.11656 Figure 8",
        "definition": (
            "state and stiffness after frame a are retained; feedback and material updates "
            "are disabled from frame a+1; identical known controls are applied through a+b"
        ),
        "gap": "direct 3D correspondence error in millimeters; no target-frame registration",
        "controlled_region_excluded": True,
        "future_observations_used_by_rollout": False,
        "offline_rollout_used_for_parameter_selection": False,
        "primary_rows": [list(row) for row in PRIMARY_ROWS],
        "horizon_rows": [list(row) for row in HORIZON_ROWS],
        "summaries": summaries,
    }
    (args.output / "future_gap_values.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[future-gap 可视化] 完成：{args.output.resolve()}")


if __name__ == "__main__":
    main()
