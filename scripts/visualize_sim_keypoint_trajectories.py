#!/usr/bin/env python3
"""绘制两套仿真数据的固定关键点 2D/3D 真值运动轨迹。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


TASKS = {
    "front_pull": "tissue_retraction_free_support_front_v2",
    "side_pull": "tissue_retraction_free_support_side_v2",
}
CAMERAS = ("stereo_left", "stereo_right")
TIME_MARKERS = (0, 60, 120, 180, 240, 299)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/sim"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/two_free_support_ablation_v2/visualizations"),
    )
    parser.add_argument("--keypoints", type=int, default=12)
    parser.add_argument("--minimum-visible-fraction", type=float, default=0.75)
    return parser.parse_args()


def load_task(root: Path) -> dict[str, object]:
    gt3 = np.load(root / "ground_truth/trajectories_3d.npz")
    gt2 = {
        camera: np.load(root / f"ground_truth/trajectories_2d/{camera}.npz")
        for camera in CAMERAS
    }
    tissue = np.load(root / "ground_truth/tissue_state.npz")
    boundary = np.load(root / "task_inputs/red_marker_boundary.npz")
    return {"root": root, "gt3": gt3, "gt2": gt2, "tissue": tissue, "boundary": boundary}


def farthest_point_sample(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) < count:
        raise ValueError(f"候选关键点只有 {len(points)} 个，少于请求的 {count} 个")
    center = points.mean(axis=0)
    selected = [int(np.argmin(np.linalg.norm(points - center, axis=1)))]
    minimum_distance = np.linalg.norm(points - points[selected[0]], axis=1)
    while len(selected) < count:
        index = int(np.argmax(minimum_distance))
        selected.append(index)
        minimum_distance = np.minimum(
            minimum_distance, np.linalg.norm(points - points[index], axis=1)
        )
    return np.asarray(selected, dtype=np.int64)


def select_shared_keypoints(
    tasks: dict[str, dict[str, object]], count: int, minimum_visible_fraction: float
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    first = tasks["front_pull"]
    gt3 = first["gt3"]
    assert isinstance(gt3, np.lib.npyio.NpzFile)
    all_ids = np.asarray(gt3["tissue_node_ids"], dtype=np.int32)
    evaluation_ids = np.asarray(gt3["tissue_evaluation_node_ids"], dtype=np.int32)
    candidate_ids = set(int(value) for value in evaluation_ids)
    excluded: set[int] = set()
    visibility: dict[str, dict[str, float]] = {}
    for task_name, task in tasks.items():
        boundary = task["boundary"]
        assert isinstance(boundary, np.lib.npyio.NpzFile)
        excluded.update(
            int(value) for value in boundary["evaluation_exclusion_tissue_node_ids"]
        )
        visibility[task_name] = {}
    candidate_ids.difference_update(excluded)

    id_to_column = {int(node_id): index for index, node_id in enumerate(all_ids)}
    accepted: list[int] = []
    for node_id in sorted(candidate_ids):
        fractions = []
        column = id_to_column[node_id]
        for task_name, task in tasks.items():
            gt2 = task["gt2"]
            assert isinstance(gt2, dict)
            for camera in CAMERAS:
                fraction = float(np.asarray(gt2[camera]["tissue_visible"][:, column]).mean())
                visibility[task_name][f"{camera}:{node_id}"] = fraction
                fractions.append(fraction)
        if min(fractions) >= minimum_visible_fraction:
            accepted.append(node_id)
    if len(accepted) < count:
        raise ValueError(
            f"共享可见候选只有 {len(accepted)} 个；可降低 --minimum-visible-fraction"
        )
    accepted_array = np.asarray(accepted, dtype=np.int32)
    columns = np.asarray([id_to_column[int(node)] for node in accepted_array])
    rest = np.asarray(gt3["tissue_positions_world"][0, columns], dtype=np.float64)
    selected_local = farthest_point_sample(rest, count)
    return accepted_array[selected_local], visibility


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def add_direction_arrow(ax: plt.Axes, points: np.ndarray, color: object) -> None:
    finite = np.isfinite(points).all(axis=1)
    valid = np.flatnonzero(finite)
    if len(valid) < 3:
        return
    anchor = valid[len(valid) * 2 // 3]
    before = valid[max(0, len(valid) * 2 // 3 - 1)]
    delta = points[anchor] - points[before]
    if float(np.linalg.norm(delta)) > 1.0e-3:
        ax.annotate(
            "",
            xy=points[anchor],
            xytext=points[before],
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.5},
        )


def plot_2d(
    task_name: str, task: dict[str, object], selected_ids: np.ndarray, output: Path
) -> None:
    gt3 = task["gt3"]
    gt2 = task["gt2"]
    root = task["root"]
    assert isinstance(gt3, np.lib.npyio.NpzFile)
    assert isinstance(gt2, dict)
    assert isinstance(root, Path)
    all_ids = np.asarray(gt3["tissue_node_ids"], dtype=np.int32)
    lookup = {int(node): index for index, node in enumerate(all_ids)}
    columns = np.asarray([lookup[int(node)] for node in selected_ids])
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, len(selected_ids)))
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    for ax, camera in zip(axes, CAMERAS):
        background = np.asarray(Image.open(root / f"rgb/{camera}/000000.png").convert("RGB"))
        ax.imshow(background)
        uv = np.asarray(gt2[camera]["tissue_uv_pixels"][:, columns], dtype=np.float64)
        visible = np.asarray(gt2[camera]["tissue_visible"][:, columns], dtype=bool)
        for keypoint_index, (node_id, color) in enumerate(zip(selected_ids, colors)):
            trajectory = uv[:, keypoint_index].copy()
            trajectory[~visible[:, keypoint_index]] = np.nan
            ax.plot(
                trajectory[:, 0], trajectory[:, 1], color=color, lw=2.0,
                label=f"K{keypoint_index + 1} (node {int(node_id)})",
            )
            finite = np.isfinite(trajectory).all(axis=1)
            valid = np.flatnonzero(finite)
            if len(valid):
                ax.scatter(*trajectory[valid[0]], s=35, c=[color], marker="o", edgecolors="white", linewidths=0.6)
                ax.scatter(*trajectory[valid[-1]], s=45, c=[color], marker="X", edgecolors="black", linewidths=0.6)
            marker_frames = [frame for frame in TIME_MARKERS if finite[frame]]
            if marker_frames:
                ax.scatter(
                    trajectory[marker_frames, 0], trajectory[marker_frames, 1],
                    s=16, c=[color], marker="o", edgecolors="black", linewidths=0.25,
                )
            add_direction_arrow(ax, trajectory, color)
        ax.set_title(camera.replace("_", " ").title())
        ax.set_xlim(0, background.shape[1])
        ax.set_ylim(background.shape[0], 0)
        ax.set_xlabel("u (pixel)")
        ax.set_ylabel("v (pixel)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
    fig.suptitle(f"{task_name}: ground-truth 2D keypoint trajectories", fontsize=16)
    save_figure(fig, output / f"{task_name}_gt_keypoints_2d")


def equal_3d_axes(ax: plt.Axes, points: np.ndarray) -> None:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    radius = max(float((maximum - minimum).max()) * 0.55, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def plot_3d(
    task_name: str, task: dict[str, object], selected_ids: np.ndarray, output: Path
) -> None:
    gt3 = task["gt3"]
    tissue = task["tissue"]
    assert isinstance(gt3, np.lib.npyio.NpzFile)
    assert isinstance(tissue, np.lib.npyio.NpzFile)
    all_ids = np.asarray(gt3["tissue_node_ids"], dtype=np.int32)
    lookup = {int(node): index for index, node in enumerate(all_ids)}
    columns = np.asarray([lookup[int(node)] for node in selected_ids])
    positions = np.asarray(gt3["tissue_positions_world"], dtype=np.float64) * 1.0e3
    trajectories = positions[:, columns]
    rest = positions[0]
    faces = np.asarray(tissue["visual_faces"][tissue["top_face_indices"]], dtype=np.int32)
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, len(selected_ids)))
    fig = plt.figure(figsize=(11, 9), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        rest[:, 0], rest[:, 1], rest[:, 2], triangles=faces,
        color=(0.78, 0.78, 0.78, 0.20), edgecolor=(0.35, 0.35, 0.35, 0.12),
        linewidth=0.15, shade=False,
    )
    for index, (node_id, color) in enumerate(zip(selected_ids, colors)):
        path = trajectories[:, index]
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, lw=2.4, label=f"K{index + 1} ({int(node_id)})")
        ax.scatter(*path[0], s=30, c=[color], marker="o", edgecolors="white", linewidths=0.5)
        ax.scatter(*path[-1], s=45, c=[color], marker="X", edgecolors="black", linewidths=0.5)
        ax.scatter(
            path[list(TIME_MARKERS), 0], path[list(TIME_MARKERS), 1], path[list(TIME_MARKERS), 2],
            s=13, c=[color], edgecolors="black", linewidths=0.2,
        )
    equal_3d_axes(ax, np.concatenate((rest, trajectories.reshape(-1, 3)), axis=0))
    ax.view_init(elev=32, azim=-68)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"{task_name}: ground-truth 3D keypoint trajectories")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    save_figure(fig, output / f"{task_name}_gt_keypoints_3d")


def plot_overview(tasks: dict[str, dict[str, object]], selected_ids: np.ndarray, output: Path) -> None:
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, len(selected_ids)))
    fig = plt.figure(figsize=(17, 14), constrained_layout=True)
    for row, (task_name, task) in enumerate(tasks.items()):
        gt3 = task["gt3"]
        gt2 = task["gt2"]
        root = task["root"]
        tissue = task["tissue"]
        assert isinstance(gt3, np.lib.npyio.NpzFile)
        assert isinstance(gt2, dict)
        assert isinstance(root, Path)
        assert isinstance(tissue, np.lib.npyio.NpzFile)
        ids = np.asarray(gt3["tissue_node_ids"])
        lookup = {int(node): index for index, node in enumerate(ids)}
        columns = np.asarray([lookup[int(node)] for node in selected_ids])
        ax2 = fig.add_subplot(2, 2, row * 2 + 1)
        image = np.asarray(Image.open(root / "rgb/stereo_left/000000.png").convert("RGB"))
        ax2.imshow(image)
        uv = np.asarray(gt2["stereo_left"]["tissue_uv_pixels"][:, columns])
        visible = np.asarray(gt2["stereo_left"]["tissue_visible"][:, columns])
        for k, color in enumerate(colors):
            path = uv[:, k].copy()
            path[~visible[:, k]] = np.nan
            ax2.plot(path[:, 0], path[:, 1], color=color, lw=2)
        ax2.set_title(f"{task_name}: left-view 2D GT")
        ax2.axis("off")

        ax3 = fig.add_subplot(2, 2, row * 2 + 2, projection="3d")
        xyz = np.asarray(gt3["tissue_positions_world"], dtype=np.float64) * 1.0e3
        rest = xyz[0]
        faces = np.asarray(tissue["visual_faces"][tissue["top_face_indices"]], dtype=np.int32)
        ax3.plot_trisurf(rest[:, 0], rest[:, 1], rest[:, 2], triangles=faces, color=(0.78, 0.78, 0.78, 0.20), linewidth=0)
        for k, color in enumerate(colors):
            path = xyz[:, columns[k]]
            ax3.plot(path[:, 0], path[:, 1], path[:, 2], color=color, lw=2)
        equal_3d_axes(ax3, xyz[:, columns].reshape(-1, 3))
        ax3.view_init(elev=32, azim=-68)
        ax3.set_title(f"{task_name}: 3D GT")
        ax3.set_xlabel("X (mm)")
        ax3.set_ylabel("Y (mm)")
        ax3.set_zlabel("Z (mm)")
    fig.suptitle("Ground-truth 2D and 3D keypoint motion", fontsize=17)
    save_figure(fig, output / "gt_keypoints_2d3d_overview")


def main() -> None:
    args = parse_args()
    if args.keypoints < 3:
        raise ValueError("--keypoints 至少为 3")
    if not 0.0 < args.minimum_visible_fraction <= 1.0:
        raise ValueError("--minimum-visible-fraction 必须位于 (0,1]")
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = {
        task: load_task(args.data_root / dataset)
        for task, dataset in TASKS.items()
    }
    selected_ids, visibility = select_shared_keypoints(
        tasks, args.keypoints, args.minimum_visible_fraction
    )
    manifest = {
        "schema": "fixedsuperbest.sim_visualization_keypoints.v1",
        "selection": (
            "shared non-controlled evaluation nodes; visible in both cameras and both tasks; "
            "deterministic farthest-point sampling using frame-0 3D positions only"
        ),
        "minimum_visible_fraction": args.minimum_visible_fraction,
        "keypoint_count": int(len(selected_ids)),
        "tissue_node_ids": selected_ids.tolist(),
        "time_markers": list(TIME_MARKERS),
        "uses_future_method_error_for_selection": False,
        "uses_material_ground_truth_for_selection": False,
        "visibility_fraction": {
            task: {
                camera: {
                    str(int(node)): visibility[task][f"{camera}:{int(node)}"]
                    for node in selected_ids
                }
                for camera in CAMERAS
            }
            for task in TASKS
        },
    }
    (args.output / "selected_keypoints.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for task_name, task in tasks.items():
        plot_2d(task_name, task, selected_ids, args.output)
        plot_3d(task_name, task, selected_ids, args.output)
    plot_overview(tasks, selected_ids, args.output)
    print(f"[关键点可视化] IDs={selected_ids.tolist()}")
    print(f"[关键点可视化] 完成：{args.output.resolve()}")


if __name__ == "__main__":
    main()
