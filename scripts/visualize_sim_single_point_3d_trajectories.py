#!/usr/bin/env python3
"""为每个固定关键点单独绘制 GT 与三种方法的 3D 运动轨迹。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TASKS = ("front_pull", "side_pull")
DATASETS = {
    "front_pull": "tissue_retraction_free_support_front_v2",
    "side_pull": "tissue_retraction_free_support_side_v2",
}
METHODS = (
    ("pbd", "PBD", "#d62728"),
    ("pbd_visual_residual", "PBD + RGB residual", "#1f77b4"),
    (
        "pbd_visual_residual_stiffness",
        "PBD + RGB residual + stiffness",
        "#8a2be2",
    ),
)
SPLIT_FRAME = 240


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/sim"))
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("outputs/two_free_support_ablation_v2"),
    )
    parser.add_argument(
        "--keypoint-manifest",
        type=Path,
        default=Path(
            "outputs/two_free_support_ablation_v2/visualizations/"
            "simple_3points/selected_keypoints.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/two_free_support_ablation_v2/visualizations/"
            "single_point_3d"
        ),
    )
    return parser.parse_args()


def load_task(
    task: str, data_root: Path, evaluation_root: Path, node_ids: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    dataset = data_root / DATASETS[task]
    with np.load(dataset / "ground_truth/trajectories_3d.npz", allow_pickle=False) as gt:
        gt_ids = np.asarray(gt["tissue_node_ids"], dtype=np.int32)
        gt_positions = np.asarray(gt["tissue_positions_world"], dtype=np.float64)
    gt_lookup = {int(node): index for index, node in enumerate(gt_ids)}
    gt_columns = np.asarray([gt_lookup[int(node)] for node in node_ids])
    reference = gt_positions[:, gt_columns]

    predictions: dict[str, np.ndarray] = {}
    for method, _label, _color in METHODS:
        path = (
            evaluation_root
            / task
            / method
            / "artifacts/predicted_trajectories.npz"
        )
        with np.load(path, allow_pickle=False) as prediction:
            frames = np.asarray(prediction["frame_indices"], dtype=np.int32)
            if not np.array_equal(frames, np.arange(len(reference), dtype=np.int32)):
                raise ValueError(f"{path} 帧索引不是完整 0..{len(reference) - 1}")
            pred_ids = np.asarray(prediction["tissue_node_ids"], dtype=np.int32)
            lookup = {int(node): index for index, node in enumerate(pred_ids)}
            try:
                columns = np.asarray([lookup[int(node)] for node in node_ids])
            except KeyError as error:
                raise ValueError(f"{path} 不包含关键点 {int(error.args[0])}") from error
            predictions[method] = np.asarray(
                prediction["tissue_positions_world"][:, columns], dtype=np.float64
            )
    return reference, predictions


def set_equal_limits(ax: plt.Axes, paths_mm: list[np.ndarray]) -> None:
    points = np.concatenate(paths_mm, axis=0)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    radius = max(float((maximum - minimum).max()) * 0.62, 0.5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def plot_one(
    ax: plt.Axes,
    task: str,
    node_id: int,
    reference_m: np.ndarray,
    predictions_m: dict[str, np.ndarray],
    *,
    show_legend: bool,
) -> list[dict[str, object]]:
    origin = reference_m[0]
    reference = (reference_m - origin) * 1.0e3
    paths = [reference]
    rows: list[dict[str, object]] = []

    # GT 也把未来段画成虚线，以直接显示 80/20 的时间分界。
    ax.plot(
        reference[:SPLIT_FRAME, 0],
        reference[:SPLIT_FRAME, 1],
        reference[:SPLIT_FRAME, 2],
        color="black",
        lw=3.0,
        label="Ground truth",
    )
    ax.plot(
        reference[SPLIT_FRAME - 1 :, 0],
        reference[SPLIT_FRAME - 1 :, 1],
        reference[SPLIT_FRAME - 1 :, 2],
        color="black",
        lw=3.0,
        linestyle="--",
    )
    ax.scatter(*reference[0], c="black", s=42, marker="o")
    ax.scatter(*reference[SPLIT_FRAME - 1], c="black", s=48, marker="^")
    ax.scatter(*reference[-1], c="black", s=55, marker="X")

    for method, label, color in METHODS:
        predicted = (predictions_m[method] - origin) * 1.0e3
        paths.append(predicted)
        ax.plot(
            predicted[:SPLIT_FRAME, 0],
            predicted[:SPLIT_FRAME, 1],
            predicted[:SPLIT_FRAME, 2],
            color=color,
            lw=2.2,
            label=label,
        )
        ax.plot(
            predicted[SPLIT_FRAME - 1 :, 0],
            predicted[SPLIT_FRAME - 1 :, 1],
            predicted[SPLIT_FRAME - 1 :, 2],
            color=color,
            lw=2.2,
            linestyle="--",
        )
        ax.scatter(*predicted[0], c=color, s=26, marker="o")
        ax.scatter(*predicted[SPLIT_FRAME - 1], c=color, s=34, marker="^")
        ax.scatter(*predicted[-1], c=color, s=42, marker="X")
        error_mm = np.linalg.norm(predictions_m[method] - reference_m, axis=1) * 1.0e3
        future = error_mm[SPLIT_FRAME:]
        rows.append(
            {
                "task": task,
                "node_id": int(node_id),
                "method": method,
                "future_mean_mm": float(future.mean()),
                "future_rmse_mm": float(np.sqrt(np.mean(np.square(future)))),
                "future_p95_mm": float(np.percentile(future, 95.0)),
                "final_frame_error_mm": float(error_mm[-1]),
            }
        )
    set_equal_limits(ax, paths)
    ax.view_init(elev=27, azim=-58)
    ax.set_xlabel("ΔX (mm)")
    ax.set_ylabel("ΔY (mm)")
    ax.set_zlabel("ΔZ (mm)")
    ax.set_title(f"{task} — node {node_id}")
    ax.grid(True, alpha=0.35)
    if show_legend:
        ax.legend(loc="upper left", fontsize=8)
    ax.text2D(
        0.02,
        0.02,
        "○ start   ▲ last observed (239)   × end\nsolid: observed 0–239   dashed: open-loop 240–299",
        transform=ax.transAxes,
        fontsize=7.5,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "0.8"},
    )
    return rows


def save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.keypoint_manifest.read_text(encoding="utf-8"))
    node_ids = np.asarray(manifest["tissue_node_ids"], dtype=np.int32)
    if not 2 <= len(node_ids) <= 3:
        raise ValueError("单点3D图要求关键点清单包含 2–3 个点")
    args.output.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    loaded: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for task in TASKS:
        reference, predictions = load_task(
            task, args.data_root, args.evaluation_root, node_ids
        )
        loaded[task] = (reference, predictions)
        for index, node_id in enumerate(node_ids):
            fig = plt.figure(figsize=(8.4, 7.2), constrained_layout=True)
            ax = fig.add_subplot(111, projection="3d")
            rows = plot_one(
                ax,
                task,
                int(node_id),
                reference[:, index],
                {method: values[:, index] for method, values in predictions.items()},
                show_legend=True,
            )
            all_rows.extend(rows)
            save(fig, args.output / f"{task}_node_{int(node_id)}_trajectory_3d")

        fig = plt.figure(figsize=(17, 5.8), constrained_layout=True)
        for index, node_id in enumerate(node_ids):
            ax = fig.add_subplot(1, len(node_ids), index + 1, projection="3d")
            plot_one(
                ax,
                task,
                int(node_id),
                reference[:, index],
                {method: values[:, index] for method, values in predictions.items()},
                show_legend=index == 0,
            )
        fig.suptitle(
            f"{task}: single-keypoint 3D trajectories — GT and three methods",
            fontsize=15,
        )
        save(fig, args.output / f"{task}_three_single_point_trajectories_3d")

    fig = plt.figure(figsize=(17, 11), constrained_layout=True)
    for task_index, task in enumerate(TASKS):
        reference, predictions = loaded[task]
        for point_index, node_id in enumerate(node_ids):
            ax = fig.add_subplot(2, len(node_ids), task_index * len(node_ids) + point_index + 1, projection="3d")
            plot_one(
                ax,
                task,
                int(node_id),
                reference[:, point_index],
                {method: values[:, point_index] for method, values in predictions.items()},
                show_legend=task_index == 0 and point_index == 0,
            )
    fig.suptitle("Single-keypoint 3D trajectory comparison", fontsize=16)
    save(fig, args.output / "two_tasks_single_point_trajectories_3d")

    fields = (
        "task",
        "node_id",
        "method",
        "future_mean_mm",
        "future_rmse_mm",
        "future_p95_mm",
        "final_frame_error_mm",
    )
    with (args.output / "single_point_future_errors.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    payload = {
        "schema": "fixedsuperbest.single_point_3d_trajectories.v1",
        "node_ids": node_ids.tolist(),
        "coordinate": "millimeter displacement relative to each GT point at frame 0",
        "solid_frames": [0, 239],
        "dashed_open_loop_frames": [239, 299],
        "rows": all_rows,
    }
    (args.output / "single_point_future_errors.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[单点3D轨迹] nodes={node_ids.tolist()}")
    print(f"[单点3D轨迹] 完成：{args.output.resolve()}")


if __name__ == "__main__":
    main()
