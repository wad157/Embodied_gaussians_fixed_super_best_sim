#!/usr/bin/env python3
"""可视化固定十点的 2D/3D 轨迹与未来 20% 逐点预测误差。"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image, ImageDraw


METHODS = (
    ("pbd", "PBD", "#d62728"),
    ("pbd_visual_residual", "PBD + RGB residual", "#1f77b4"),
    (
        "pbd_visual_residual_stiffness",
        "PBD + RGB residual + stiffness",
        "#8a2be2",
    ),
)
CAMERAS = ("stereo_left", "stereo_right")


@dataclass(frozen=True)
class Prediction:
    positions_m: np.ndarray
    uv: dict[str, np.ndarray]
    valid: dict[str, np.ndarray]


@dataclass(frozen=True)
class EvaluationData:
    dataset: Path
    node_ids: np.ndarray
    frames: np.ndarray
    timestamps: np.ndarray
    split_frame: int
    gt_positions_m: np.ndarray
    gt_uv: dict[str, np.ndarray]
    gt_visible: dict[str, np.ndarray]
    predictions: dict[str, Prediction]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("outputs/sim_known_grasp_causal_full_pbd30hz_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/sim_known_grasp_causal_full_pbd30hz_v1/visualizations/ten_points"
        ),
    )
    return parser.parse_args()


def _columns(all_ids: np.ndarray, selected_ids: np.ndarray, source: Path) -> np.ndarray:
    lookup = {int(node): index for index, node in enumerate(all_ids)}
    missing = [int(node) for node in selected_ids if int(node) not in lookup]
    if missing:
        raise ValueError(f"{source} 缺少固定评估点：{missing}")
    return np.asarray([lookup[int(node)] for node in selected_ids], dtype=np.int64)


def load_evaluation(root: Path) -> EvaluationData:
    metric_payloads: dict[str, dict[str, object]] = {}
    for method, _label, _color in METHODS:
        path = root / "future_80to20" / method / "trajectory_metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"缺少未来预测指标：{path}")
        metric_payloads[method] = json.loads(path.read_text(encoding="utf-8"))

    first = metric_payloads[METHODS[0][0]]
    node_ids = np.asarray(first["evaluation_node_ids"], dtype=np.int32)
    if len(node_ids) != 10 or int(first["evaluated_nodes"]) != 10:
        raise ValueError(f"评估清单不是固定十点：{node_ids.tolist()}")
    for method, payload in metric_payloads.items():
        candidate = np.asarray(payload["evaluation_node_ids"], dtype=np.int32)
        if int(payload["evaluated_nodes"]) != 10 or not np.array_equal(
            candidate, node_ids
        ):
            raise ValueError(f"三种方法的固定十点评估清单不一致：{method}")

    dataset = Path(str(first["reference_dataset"])).resolve()
    split_frame = int(first["frame_start"])
    gt3_path = dataset / "ground_truth/trajectories_3d.npz"
    with np.load(gt3_path, allow_pickle=False) as gt3:
        gt_frames = np.arange(len(gt3["timestamps"]), dtype=np.int32)
        timestamps = np.asarray(gt3["timestamps"], dtype=np.float64)
        columns = _columns(
            np.asarray(gt3["tissue_node_ids"], dtype=np.int32), node_ids, gt3_path
        )
        gt_positions = np.asarray(
            gt3["tissue_positions_world"][:, columns], dtype=np.float64
        )

    gt_uv: dict[str, np.ndarray] = {}
    gt_visible: dict[str, np.ndarray] = {}
    for camera in CAMERAS:
        path = dataset / f"ground_truth/trajectories_2d/{camera}.npz"
        with np.load(path, allow_pickle=False) as gt2:
            columns = _columns(
                np.asarray(gt2["tissue_node_ids"], dtype=np.int32), node_ids, path
            )
            gt_uv[camera] = np.asarray(
                gt2["tissue_uv_pixels"][:, columns], dtype=np.float64
            )
            gt_visible[camera] = np.asarray(
                gt2["tissue_visible"][:, columns], dtype=bool
            )

    predictions: dict[str, Prediction] = {}
    for method, _label, _color in METHODS:
        path = root / "future_80to20" / method / "artifacts/predicted_trajectories.npz"
        with np.load(path, allow_pickle=False) as prediction:
            frames = np.asarray(prediction["frame_indices"], dtype=np.int32)
            if not np.array_equal(frames, gt_frames):
                raise ValueError(f"{path} 不是完整的 0..{len(gt_frames) - 1} 帧")
            columns = _columns(
                np.asarray(prediction["tissue_node_ids"], dtype=np.int32),
                node_ids,
                path,
            )
            positions = np.asarray(
                prediction["tissue_positions_world"][:, columns], dtype=np.float64
            )
            uv = {
                camera: np.asarray(
                    prediction[f"{camera}_tissue_uv_pixels"][:, columns],
                    dtype=np.float64,
                )
                for camera in CAMERAS
            }
            valid = {
                camera: np.asarray(
                    prediction[f"{camera}_tissue_valid"][:, columns], dtype=bool
                )
                for camera in CAMERAS
            }
            predictions[method] = Prediction(positions, uv, valid)

    boundary_path = dataset / "task_inputs/known_grasp_region_boundary.npz"
    with np.load(boundary_path, allow_pickle=False) as boundary:
        excluded = set(
            int(value)
            for value in boundary["evaluation_exclusion_tissue_node_ids"]
        )
    overlap = sorted(excluded.intersection(int(node) for node in node_ids))
    if overlap:
        raise ValueError(f"固定十点仍包含夹持排除点：{overlap}")

    if not 0 < split_frame < len(gt_frames):
        raise ValueError(f"非法的未来预测分界帧：{split_frame}")
    return EvaluationData(
        dataset=dataset,
        node_ids=node_ids,
        frames=gt_frames,
        timestamps=timestamps,
        split_frame=split_frame,
        gt_positions_m=gt_positions,
        gt_uv=gt_uv,
        gt_visible=gt_visible,
        predictions=predictions,
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_segmented_path_2d(
    ax: plt.Axes,
    path: np.ndarray,
    valid: np.ndarray,
    split: int,
    *,
    color: str,
    label: str,
    linewidth: float,
) -> None:
    values = path.copy()
    values[~valid] = np.nan
    ax.plot(
        values[:split, 0],
        values[:split, 1],
        color=color,
        lw=linewidth,
        alpha=0.48,
        label=label,
    )
    ax.plot(
        values[split - 1 :, 0],
        values[split - 1 :, 1],
        color=color,
        lw=linewidth,
        linestyle="--",
        alpha=0.98,
    )
    for frame, marker, size in ((split - 1, "^", 28), (len(values) - 1, "X", 34)):
        if valid[frame] and np.isfinite(values[frame]).all():
            ax.scatter(
                values[frame, 0],
                values[frame, 1],
                c=color,
                s=size,
                marker=marker,
                edgecolors="white",
                linewidths=0.45,
                zorder=5,
            )


def plot_2d_trajectories(data: EvaluationData, output: Path) -> None:
    for camera in CAMERAS:
        background_path = (
            data.dataset / f"rgb/{camera}/{data.split_frame:06d}.png"
        )
        background = np.asarray(Image.open(background_path).convert("RGB"))
        fig, axes = plt.subplots(5, 2, figsize=(15, 26), constrained_layout=True)
        for index, (ax, node_id) in enumerate(zip(axes.flat, data.node_ids)):
            ax.imshow(background, alpha=0.82)
            plot_segmented_path_2d(
                ax,
                data.gt_uv[camera][:, index],
                data.gt_visible[camera][:, index],
                data.split_frame,
                color="black",
                label="Ground truth",
                linewidth=3.0,
            )
            for method, label, color in METHODS:
                prediction = data.predictions[method]
                plot_segmented_path_2d(
                    ax,
                    prediction.uv[camera][:, index],
                    prediction.valid[camera][:, index],
                    data.split_frame,
                    color=color,
                    label=label,
                    linewidth=2.0,
                )
            ax.set_title(f"K{index + 1} — tissue node {int(node_id)}")
            ax.set_xlim(0, background.shape[1])
            ax.set_ylim(background.shape[0], 0)
            ax.set_xlabel("u (pixel)")
            ax.set_ylabel("v (pixel)")
            ax.grid(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=10)
        fig.suptitle(
            f"Fixed 10-point 2D trajectories — {camera}\n"
            f"solid: observed frames 0–{data.split_frame - 1}; "
            f"dashed: future frames {data.split_frame}–{len(data.frames) - 1}",
            fontsize=17,
        )
        save_figure(fig, output / f"ten_point_2d_trajectories_{camera}")


def _set_equal_3d_limits(ax: plt.Axes, paths: list[np.ndarray]) -> None:
    points = np.concatenate(paths, axis=0)
    minimum = np.nanmin(points, axis=0)
    maximum = np.nanmax(points, axis=0)
    center = 0.5 * (minimum + maximum)
    radius = max(float((maximum - minimum).max()) * 0.60, 0.5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def plot_segmented_path_3d(
    ax: plt.Axes,
    path: np.ndarray,
    split: int,
    *,
    color: str,
    label: str,
    linewidth: float,
) -> None:
    ax.plot(
        path[:split, 0],
        path[:split, 1],
        path[:split, 2],
        color=color,
        lw=linewidth,
        alpha=0.48,
        label=label,
    )
    ax.plot(
        path[split - 1 :, 0],
        path[split - 1 :, 1],
        path[split - 1 :, 2],
        color=color,
        lw=linewidth,
        linestyle="--",
        alpha=0.98,
    )
    ax.scatter(*path[split - 1], c=color, s=30, marker="^")
    ax.scatter(*path[-1], c=color, s=38, marker="X")


def plot_3d_trajectories(data: EvaluationData, output: Path) -> None:
    fig = plt.figure(figsize=(16, 28), constrained_layout=True)
    for index, node_id in enumerate(data.node_ids):
        ax = fig.add_subplot(5, 2, index + 1, projection="3d")
        origin = data.gt_positions_m[0, index]
        gt = (data.gt_positions_m[:, index] - origin) * 1.0e3
        paths = [gt]
        plot_segmented_path_3d(
            ax,
            gt,
            data.split_frame,
            color="black",
            label="Ground truth",
            linewidth=3.0,
        )
        for method, label, color in METHODS:
            predicted = (
                data.predictions[method].positions_m[:, index] - origin
            ) * 1.0e3
            paths.append(predicted)
            plot_segmented_path_3d(
                ax,
                predicted,
                data.split_frame,
                color=color,
                label=label,
                linewidth=2.0,
            )
        _set_equal_3d_limits(ax, paths)
        ax.view_init(elev=29, azim=-60)
        ax.set_xlabel("ΔX (mm)")
        ax.set_ylabel("ΔY (mm)")
        ax.set_zlabel("ΔZ (mm)")
        ax.set_title(f"K{index + 1} — tissue node {int(node_id)}")
        ax.grid(True, alpha=0.28)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=10)
    fig.suptitle(
        "Fixed 10-point 3D trajectories\n"
        f"solid: observed frames 0–{data.split_frame - 1}; "
        f"dashed: future frames {data.split_frame}–{len(data.frames) - 1}",
        fontsize=17,
    )
    save_figure(fig, output / "ten_point_3d_trajectories")


def compute_errors(
    data: EvaluationData,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    error_3d: dict[str, np.ndarray] = {}
    error_2d_camera: dict[str, dict[str, np.ndarray]] = {}
    error_2d_stereo: dict[str, np.ndarray] = {}
    for method, _label, _color in METHODS:
        prediction = data.predictions[method]
        error_3d[method] = (
            np.linalg.norm(prediction.positions_m - data.gt_positions_m, axis=2)
            * 1.0e3
        )
        camera_errors: dict[str, np.ndarray] = {}
        for camera in CAMERAS:
            valid = data.gt_visible[camera] & prediction.valid[camera]
            error = np.linalg.norm(prediction.uv[camera] - data.gt_uv[camera], axis=2)
            error[~valid] = np.nan
            camera_errors[camera] = error
        error_2d_camera[method] = camera_errors
        stack = np.stack([camera_errors[camera] for camera in CAMERAS], axis=0)
        count = np.isfinite(stack).sum(axis=0)
        total = np.nansum(stack, axis=0)
        error_2d_stereo[method] = np.divide(
            total,
            count,
            out=np.full_like(total, np.nan),
            where=count > 0,
        )
    return error_3d, error_2d_camera, error_2d_stereo


def _heatmap_norm(arrays: list[np.ndarray]) -> Normalize:
    finite = np.concatenate([array[np.isfinite(array)] for array in arrays])
    maximum = float(np.percentile(finite, 99.0)) if finite.size else 1.0
    return Normalize(vmin=0.0, vmax=max(maximum, 1.0e-6))


def plot_temporal_heatmaps(
    data: EvaluationData,
    errors: dict[str, np.ndarray],
    output: Path,
    *,
    stem: str,
    unit: str,
    title: str,
) -> None:
    future = slice(data.split_frame, None)
    arrays = [errors[method][future].T for method, _label, _color in METHODS]
    norm = _heatmap_norm(arrays)
    fig, axes = plt.subplots(3, 1, figsize=(16.5, 10.5), sharex=True)
    for ax, array, (_method, label, _color) in zip(axes, arrays, METHODS):
        image = ax.imshow(
            array,
            aspect="auto",
            interpolation="nearest",
            cmap="turbo",
            norm=norm,
            extent=(data.split_frame - 0.5, len(data.frames) - 0.5, 9.5, -0.5),
        )
        ax.set_yticks(np.arange(10))
        ax.set_yticklabels(
            [f"K{i + 1}: {int(node)}" for i, node in enumerate(data.node_ids)]
        )
        ax.set_ylabel("Fixed keypoint")
        ax.set_title(label, fontsize=11)
    axes[-1].set_xlabel("Future frame index")
    color_axis = fig.add_axes((0.905, 0.15, 0.018, 0.70))
    colorbar = fig.colorbar(image, cax=color_axis)
    colorbar.set_label(f"GT correspondence gap ({unit}); color clipped at joint P99")
    fig.suptitle(title, fontsize=16)
    # Explicit margins avoid backend-dependent clipping of the long ``K: node``
    # tick labels when the same figure also owns a shared colorbar.
    fig.subplots_adjust(
        left=0.145, right=0.88, top=0.91, bottom=0.08, hspace=0.31
    )
    save_figure(fig, output / stem)


def _rmse(values: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.sqrt(np.nanmean(np.square(values), axis=axis))


def plot_by_point_summary(
    data: EvaluationData,
    error_3d: dict[str, np.ndarray],
    error_2d: dict[str, np.ndarray],
    output: Path,
) -> None:
    future = slice(data.split_frame, None)
    values_3d = np.stack(
        [_rmse(error_3d[m][future], axis=0) for m, _l, _c in METHODS], axis=0
    )
    values_2d = np.stack(
        [_rmse(error_2d[m][future], axis=0) for m, _l, _c in METHODS], axis=0
    )
    fig, axes = plt.subplots(2, 1, figsize=(16, 7.2), constrained_layout=True)
    for ax, values, unit, title in (
        (axes[0], values_3d, "mm", "Future 3D RMSE by fixed keypoint"),
        (axes[1], values_2d, "pixel", "Future stereo-mean 2D RMSE by fixed keypoint"),
    ):
        image = ax.imshow(values, aspect="auto", cmap="turbo")
        ax.set_xticks(np.arange(10))
        ax.set_xticklabels(
            [f"K{i + 1}\nnode {int(node)}" for i, node in enumerate(data.node_ids)]
        )
        ax.set_yticks(np.arange(3))
        ax.set_yticklabels([label for _m, label, _c in METHODS])
        for row in range(values.shape[0]):
            threshold = 0.56 * float(np.nanmax(values))
            for column in range(values.shape[1]):
                value = values[row, column]
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value > threshold else "black",
                )
        colorbar = fig.colorbar(image, ax=ax, shrink=0.90, pad=0.012)
        colorbar.set_label(unit)
        ax.set_title(title)
    fig.suptitle(
        f"Fixed 10-point future prediction gaps — frames {data.split_frame}–{len(data.frames) - 1}",
        fontsize=16,
    )
    save_figure(fig, output / "future_error_by_point_heatmaps")


def plot_spatial_error_map(
    data: EvaluationData,
    error_3d: dict[str, np.ndarray],
    output: Path,
) -> None:
    future = slice(data.split_frame, None)
    values = {
        method: _rmse(error_3d[method][future], axis=0)
        for method, _label, _color in METHODS
    }
    norm = Normalize(
        vmin=0.0,
        vmax=max(float(np.max(np.concatenate(list(values.values())))), 1.0e-6),
    )
    background = np.asarray(
        Image.open(
            data.dataset / f"rgb/stereo_left/{data.split_frame:06d}.png"
        ).convert("RGB")
    )
    uv = data.gt_uv["stereo_left"][data.split_frame]
    visible = data.gt_visible["stereo_left"][data.split_frame]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
    for ax, (method, label, _color) in zip(axes, METHODS):
        ax.imshow(background)
        shown = visible & np.isfinite(uv).all(axis=1)
        ax.scatter(
            uv[shown, 0],
            uv[shown, 1],
            c=values[method][shown],
            cmap="turbo",
            norm=norm,
            s=135,
            edgecolors="white",
            linewidths=1.2,
        )
        for index in np.flatnonzero(shown):
            ax.annotate(
                f"K{index + 1}\n{values[method][index]:.2f}",
                xy=uv[index],
                xytext=(5, 5),
                textcoords="offset points",
                color="white",
                fontsize=8,
                weight="bold",
                bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 1.5},
            )
        ax.set_xlim(0, background.shape[1])
        ax.set_ylim(background.shape[0], 0)
        ax.set_title(label)
        ax.axis("off")
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap="turbo"), ax=axes, shrink=0.84, pad=0.015
    )
    colorbar.set_label("Future 3D RMSE to ground truth (mm)")
    fig.suptitle(
        f"Spatial distribution of fixed 10-point future gaps (GT frame {data.split_frame})",
        fontsize=16,
    )
    save_figure(fig, output / "future_3d_error_spatial_map")


def load_true_tissue_surface(
    data: EvaluationData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取真实组织三角面及材料坐标；顶面节点与 trajectory node 一一对应。"""
    path = data.dataset / "ground_truth/tissue_state.npz"
    with np.load(path, allow_pickle=False) as tissue:
        positions = np.asarray(tissue["simulation_positions"], dtype=np.float64)
        faces = np.asarray(
            tissue["visual_faces"][tissue["top_face_indices"]], dtype=np.int32
        )
        material_uv = np.asarray(tissue["material_uv"], dtype=np.float64)
    top_count = int(faces.max()) + 1
    positions = positions[:, :top_count]
    material_uv = material_uv[:top_count]
    if top_count != data.gt_positions_m.shape[1] and top_count != 899:
        raise ValueError(f"无法确认真实组织顶面节点数量：{top_count}")
    # 当前数据的 trajectory node ID 正好等于真实顶面 vertex ID。
    if int(data.node_ids.max()) >= top_count:
        raise ValueError("固定十点无法映射到真实组织三角面")
    return positions, faces, material_uv


def interpolate_ten_point_values(
    material_uv: np.ndarray,
    node_ids: np.ndarray,
    values: np.ndarray,
    *,
    neighbours: int = 4,
) -> np.ndarray:
    """在静止材料坐标中用局部 IDW 将十点误差映射到所有表面顶点。"""
    sample_uv = material_uv[node_ids]
    distances = np.linalg.norm(
        material_uv[:, None, :] - sample_uv[None, :, :], axis=2
    )
    nearest = np.argsort(distances, axis=1)[:, :neighbours]
    selected_distance = np.take_along_axis(distances, nearest, axis=1)
    selected_value = values[nearest]
    weights = 1.0 / np.maximum(selected_distance, 1.0e-8) ** 2
    interpolated = np.sum(weights * selected_value, axis=1) / np.sum(weights, axis=1)
    exact_vertex = np.min(distances, axis=1) < 1.0e-10
    if np.any(exact_vertex):
        exact_sample = np.argmin(distances[exact_vertex], axis=1)
        interpolated[exact_vertex] = values[exact_sample]
    return interpolated


def add_filled_tissue_surface(
    ax: plt.Axes,
    positions_m: np.ndarray,
    faces: np.ndarray,
    vertex_values: np.ndarray,
    norm: Normalize,
    *,
    sample_node_ids: np.ndarray,
    show_labels: bool,
) -> None:
    xyz = positions_m * 1.0e3
    face_values = vertex_values[faces].mean(axis=1)
    collection = Poly3DCollection(
        xyz[faces],
        facecolors=plt.get_cmap("turbo")(norm(face_values)),
        edgecolors=(0.04, 0.04, 0.04, 0.10),
        linewidths=0.13,
        antialiased=True,
    )
    ax.add_collection3d(collection)
    samples = xyz[sample_node_ids]
    ax.scatter(
        samples[:, 0],
        samples[:, 1],
        samples[:, 2] + 0.12,
        c=vertex_values[sample_node_ids],
        cmap="turbo",
        norm=norm,
        s=18 if not show_labels else 30,
        edgecolors="white",
        linewidths=0.55,
        depthshade=False,
    )
    if show_labels:
        for index, point in enumerate(samples):
            ax.text(
                point[0], point[1], point[2] + 0.35, f"K{index + 1}",
                fontsize=7, color="black", weight="bold",
            )
    minimum = xyz.min(axis=0)
    maximum = xyz.max(axis=0)
    extent = np.maximum(maximum - minimum, 1.0)
    padding = np.asarray((0.05, 0.05, 0.12)) * extent
    ax.set_xlim(minimum[0] - padding[0], maximum[0] + padding[0])
    ax.set_ylim(minimum[1] - padding[1], maximum[1] + padding[1])
    ax.set_zlim(minimum[2] - padding[2], maximum[2] + padding[2])
    ax.set_box_aspect(extent)
    ax.view_init(elev=57, azim=-72)
    ax.set_axis_off()


def plot_filled_surface_summary(
    data: EvaluationData,
    errors: dict[str, np.ndarray],
    output: Path,
    *,
    stem: str,
    unit: str,
    title: str,
) -> dict[str, object]:
    surface_positions, faces, material_uv = load_true_tissue_surface(data)
    future = slice(data.split_frame, None)
    point_rmse = {
        method: _rmse(errors[method][future], axis=0)
        for method, _label, _color in METHODS
    }
    maximum = max(float(np.nanmax(values)) for values in point_rmse.values())
    norm = Normalize(vmin=0.0, vmax=max(maximum, 1.0e-6))
    fig = plt.figure(figsize=(18, 6.4), constrained_layout=True)
    for column, (method, label, _color) in enumerate(METHODS):
        ax = fig.add_subplot(1, 3, column + 1, projection="3d")
        vertex_values = interpolate_ten_point_values(
            material_uv, data.node_ids, point_rmse[method]
        )
        add_filled_tissue_surface(
            ax,
            surface_positions[data.split_frame],
            faces,
            vertex_values,
            norm,
            sample_node_ids=data.node_ids,
            show_labels=True,
        )
        ax.set_title(label, fontsize=11)
        ax.text2D(
            0.02,
            0.02,
            f"10-point RMSE: {float(np.sqrt(np.nanmean(np.square(errors[method][future])))):.2f} {unit}",
            transform=ax.transAxes,
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "none", "pad": 2},
        )
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap="turbo"), ax=fig.axes, shrink=0.76, pad=0.015
    )
    colorbar.set_label(f"Future correspondence RMSE ({unit})")
    fig.suptitle(
        title + "\nactual GT tissue triangles; local IDW interpolation from fixed K1–K10",
        fontsize=15,
    )
    save_figure(fig, output / stem)
    return {
        "stem": stem,
        "unit": unit,
        "geometry_frame": data.split_frame,
        "point_rmse": {
            method: values.tolist() for method, values in point_rmse.items()
        },
    }


def plot_filled_surface_selected_times(
    data: EvaluationData,
    error_3d: dict[str, np.ndarray],
    output: Path,
) -> dict[str, object]:
    surface_positions, faces, material_uv = load_true_tissue_surface(data)
    selected_frames = (288, 311, 335, 359)
    if any(frame >= len(data.frames) for frame in selected_frames):
        raise ValueError("组织表面热度图选择帧超出数据范围")
    point_values = [
        error_3d[method][frame]
        for frame in selected_frames
        for method, _label, _color in METHODS
    ]
    maximum = max(float(np.max(values)) for values in point_values)
    norm = Normalize(vmin=0.0, vmax=max(maximum, 1.0e-6))
    fig = plt.figure(figsize=(16.5, 18), constrained_layout=True)
    for row, frame in enumerate(selected_frames):
        for column, (method, label, _color) in enumerate(METHODS):
            ax = fig.add_subplot(
                len(selected_frames), 3, row * 3 + column + 1, projection="3d"
            )
            values = error_3d[method][frame]
            vertex_values = interpolate_ten_point_values(
                material_uv, data.node_ids, values
            )
            add_filled_tissue_surface(
                ax,
                surface_positions[frame],
                faces,
                vertex_values,
                norm,
                sample_node_ids=data.node_ids,
                show_labels=False,
            )
            if row == 0:
                ax.set_title(label, fontsize=11)
            if column == 0:
                ax.text2D(
                    -0.03,
                    0.50,
                    f"t = 287 + {frame - 287}\nframe {frame}",
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10,
                    weight="bold",
                )
            ax.text2D(
                0.02,
                0.02,
                f"Mean {float(np.mean(values)):.2f} | RMSE {float(np.sqrt(np.mean(np.square(values)))):.2f} mm",
                transform=ax.transAxes,
                fontsize=7.5,
                bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "none", "pad": 2},
            )
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap="turbo"), ax=fig.axes, shrink=0.80, pad=0.012
    )
    colorbar.set_label("3D correspondence gap (mm)")
    fig.suptitle(
        "Future 3D gap on the actual tissue surface at selected times\n"
        "actual GT triangles; local IDW interpolation from the fixed 10 evaluation points",
        fontsize=15,
    )
    save_figure(fig, output / "future_3d_gap_filled_tissue_selected_times")
    return {
        "stem": "future_3d_gap_filled_tissue_selected_times",
        "unit": "mm",
        "selected_frames": list(selected_frames),
        "last_observed_frame": data.split_frame - 1,
    }


def write_surface_heatmap_manifest(
    data: EvaluationData,
    output: Path,
    figures: list[dict[str, object]],
) -> None:
    payload = {
        "schema": "fixedsuperbest.ten_point_filled_tissue_heatmap.v1",
        "reference_dataset": str(data.dataset),
        "evaluation_node_ids": data.node_ids.tolist(),
        "error_samples": "fixed 10 non-grasp evaluation points only",
        "surface_geometry": (
            "ground_truth/tissue_state.npz actual top visual triangles at the displayed frame"
        ),
        "interpolation": (
            "four-nearest inverse-distance weighting in frame-0 material_uv coordinates; "
            "sample-node values remain exact"
        ),
        "dense_error_ground_truth_claimed": False,
        "alignment": "direct tissue-node correspondence; no registration",
        "figures": figures,
    }
    (output / "filled_tissue_heatmap_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _surface_boundary_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.sort(
        np.concatenate(
            (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
        ),
        axis=1,
    )
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique[counts == 1]


def load_true_camera_surface(
    data: EvaluationData, camera: str = "stereo_left"
) -> dict[str, np.ndarray]:
    surface_positions, faces, material_uv = load_true_tissue_surface(data)
    path = data.dataset / f"ground_truth/trajectories_2d/{camera}.npz"
    with np.load(path, allow_pickle=False) as gt2:
        node_ids = np.asarray(gt2["tissue_node_ids"], dtype=np.int32)
        expected = np.arange(surface_positions.shape[1], dtype=np.int32)
        if not np.array_equal(node_ids, expected):
            raise ValueError("相机轨迹节点与真实组织顶面顶点不是一一对应")
        uv = np.asarray(gt2["tissue_uv_pixels"], dtype=np.float64)
        in_frame = np.asarray(gt2["tissue_in_frame"], dtype=bool)
        camera_z = np.asarray(gt2["tissue_camera_xyz_m"], dtype=np.float64)[..., 2]
    return {
        "positions_m": surface_positions,
        "faces": faces,
        "material_uv": material_uv,
        "uv": uv,
        "in_frame": in_frame,
        "camera_z": camera_z,
        "boundary_edges": _surface_boundary_edges(faces),
    }


def _tissue_mask(data: EvaluationData, camera: str, frame: int) -> np.ndarray:
    return np.asarray(
        Image.open(
            data.dataset
            / f"ground_truth/masks/tissue/{camera}/{frame:06d}.png"
        ).convert("L")
    )


def _crop_from_masks(*masks: np.ndarray, padding: int = 60) -> tuple[int, int, int, int]:
    union = np.logical_or.reduce([mask > 0 for mask in masks])
    y, x = np.nonzero(union)
    height, width = union.shape
    if len(x) == 0:
        return 0, width, 0, height
    return (
        max(0, int(x.min()) - padding),
        min(width, int(x.max()) + padding + 1),
        max(0, int(y.min()) - padding),
        min(height, int(y.max()) + padding + 1),
    )


def add_camera_aligned_filled_surface(
    ax: plt.Axes,
    data: EvaluationData,
    camera_surface: dict[str, np.ndarray],
    frame: int,
    vertex_values: np.ndarray,
    norm: Normalize,
    *,
    reference_frame: int,
    show_keypoint_labels: bool,
) -> None:
    """在真实数据相机像素平面中渲染实际形变三角面。"""
    camera = "stereo_left"
    rgb = Image.open(
        data.dataset / f"rgb/{camera}/{frame:06d}.png"
    ).convert("RGBA")
    width, height = rgb.size
    overlay = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    faces = camera_surface["faces"]
    uv = camera_surface["uv"][frame]
    in_frame = camera_surface["in_frame"][frame]
    camera_z = camera_surface["camera_z"][frame]
    face_values = vertex_values[faces].mean(axis=1)
    face_valid = (
        np.isfinite(uv[faces]).all(axis=(1, 2))
        & np.isfinite(camera_z[faces]).all(axis=1)
        & np.any(in_frame[faces], axis=1)
    )
    # 远处三角面先画、近处后画，投影重叠时仍保持真实可见顺序。
    order = np.argsort(camera_z[faces].mean(axis=1))[::-1]
    colors = (plt.get_cmap("turbo")(norm(face_values)) * 255.0).astype(np.uint8)
    for face_index in order:
        if not face_valid[face_index]:
            continue
        polygon = [tuple(point) for point in uv[faces[face_index]]]
        color = colors[face_index]
        draw.polygon(
            polygon,
            fill=(int(color[0]), int(color[1]), int(color[2]), 210),
        )
        draw.line(
            polygon + [polygon[0]], fill=(12, 12, 12, 30), width=1
        )
    mask = _tissue_mask(data, camera, frame)
    overlay_array = np.asarray(overlay).copy()
    overlay_array[..., 3] = np.minimum(
        overlay_array[..., 3], (mask.astype(np.float64) * (215.0 / 255.0)).astype(np.uint8)
    )
    composite = Image.alpha_composite(rgb, Image.fromarray(overlay_array, mode="RGBA"))
    ax.imshow(np.asarray(composite.convert("RGB")))

    boundary_edges = camera_surface["boundary_edges"]
    current_segments = uv[boundary_edges]
    reference_uv = camera_surface["uv"][reference_frame]
    reference_segments = reference_uv[boundary_edges]
    ax.add_collection(
        LineCollection(
            current_segments,
            colors=(1.0, 1.0, 1.0, 0.92),
            linewidths=1.15,
            zorder=6,
        )
    )
    ax.add_collection(
        LineCollection(
            reference_segments,
            colors=(0.02, 0.02, 0.02, 0.78),
            linewidths=1.0,
            linestyles="dashed",
            zorder=7,
        )
    )

    sample_uv = uv[data.node_ids]
    reference_sample_uv = reference_uv[data.node_ids]
    sample_values = vertex_values[data.node_ids]
    ax.scatter(
        sample_uv[:, 0],
        sample_uv[:, 1],
        c=sample_values,
        cmap="turbo",
        norm=norm,
        s=28 if show_keypoint_labels else 16,
        edgecolors="white",
        linewidths=0.75,
        zorder=9,
    )
    if show_keypoint_labels:
        for index, (start, end) in enumerate(zip(reference_sample_uv, sample_uv)):
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "white",
                    "lw": 0.8,
                    "alpha": 0.90,
                },
                zorder=8,
            )
            ax.annotate(
                f"K{index + 1}",
                xy=end,
                xytext=(4, 4),
                textcoords="offset points",
                color="white",
                fontsize=7,
                weight="bold",
                bbox={"facecolor": "black", "alpha": 0.52, "edgecolor": "none", "pad": 1.0},
                zorder=10,
            )
    reference_mask = _tissue_mask(data, camera, reference_frame)
    x0, x1, y0, y1 = _crop_from_masks(mask, reference_mask)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_camera_aligned_surface_summary(
    data: EvaluationData,
    errors: dict[str, np.ndarray],
    output: Path,
    *,
    stem: str,
    unit: str,
    title: str,
) -> dict[str, object]:
    camera_surface = load_true_camera_surface(data)
    future = slice(data.split_frame, None)
    display_frame = data.split_frame
    point_rmse = {
        method: _rmse(errors[method][future], axis=0)
        for method, _label, _color in METHODS
    }
    maximum = max(float(np.nanmax(values)) for values in point_rmse.values())
    norm = Normalize(vmin=0.0, vmax=max(maximum, 1.0e-6))
    fig, axes = plt.subplots(1, 3, figsize=(20, 7.0))
    for ax, (method, label, _color) in zip(axes, METHODS):
        vertex_values = interpolate_ten_point_values(
            camera_surface["material_uv"], data.node_ids, point_rmse[method]
        )
        add_camera_aligned_filled_surface(
            ax,
            data,
            camera_surface,
            display_frame,
            vertex_values,
            norm,
            reference_frame=0,
            show_keypoint_labels=True,
        )
        ax.set_title(label, fontsize=11)
        ax.text(
            0.015,
            0.02,
            f"10-point RMSE {float(np.sqrt(np.nanmean(np.square(errors[method][future])))):.2f} {unit}",
            transform=ax.transAxes,
            color="white",
            fontsize=8,
            weight="bold",
            bbox={"facecolor": "black", "alpha": 0.56, "edgecolor": "none", "pad": 2},
        )
    fig.subplots_adjust(left=0.02, right=0.90, top=0.84, bottom=0.06, wspace=0.045)
    color_axis = fig.add_axes((0.92, 0.16, 0.018, 0.66))
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap="turbo"), cax=color_axis
    )
    colorbar.set_label(f"Future correspondence RMSE ({unit})")
    fig.suptitle(
        title
        + f"\nexact stereo_left view at frame {display_frame}; solid white=current boundary, dashed black=frame-0 boundary",
        fontsize=15,
    )
    save_figure(fig, output / stem)
    return {
        "stem": stem,
        "unit": unit,
        "camera": "stereo_left",
        "display_frame": display_frame,
        "reference_outline_frame": 0,
    }


def plot_camera_aligned_selected_times(
    data: EvaluationData,
    error_3d: dict[str, np.ndarray],
    output: Path,
) -> dict[str, object]:
    camera_surface = load_true_camera_surface(data)
    selected_frames = (288, 311, 335, 359)
    maximum = max(
        float(np.max(error_3d[method][frame]))
        for frame in selected_frames
        for method, _label, _color in METHODS
    )
    norm = Normalize(vmin=0.0, vmax=max(maximum, 1.0e-6))
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    for row, frame in enumerate(selected_frames):
        for column, (method, label, _color) in enumerate(METHODS):
            ax = axes[row, column]
            values = error_3d[method][frame]
            vertex_values = interpolate_ten_point_values(
                camera_surface["material_uv"], data.node_ids, values
            )
            add_camera_aligned_filled_surface(
                ax,
                data,
                camera_surface,
                frame,
                vertex_values,
                norm,
                reference_frame=0,
                show_keypoint_labels=False,
            )
            if row == 0:
                ax.set_title(label, fontsize=11)
            if column == 0:
                ax.text(
                    -0.02,
                    0.5,
                    f"t=287+{frame - 287}\nframe {frame}",
                    transform=ax.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    fontsize=9,
                    weight="bold",
                )
            ax.text(
                0.015,
                0.02,
                f"Mean {float(np.mean(values)):.2f} | RMSE {float(np.sqrt(np.mean(np.square(values)))):.2f} mm",
                transform=ax.transAxes,
                color="white",
                fontsize=7.5,
                weight="bold",
                bbox={"facecolor": "black", "alpha": 0.56, "edgecolor": "none", "pad": 2},
            )
    fig.subplots_adjust(left=0.045, right=0.90, top=0.92, bottom=0.03, wspace=0.04, hspace=0.10)
    color_axis = fig.add_axes((0.92, 0.14, 0.017, 0.72))
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap="turbo"), cax=color_axis
    )
    colorbar.set_label("3D correspondence gap (mm)")
    fig.suptitle(
        "Future 3D gap filled on the deforming tissue in the exact dataset camera view\n"
        "actual projected triangles; solid white=current boundary, dashed black=frame-0 boundary",
        fontsize=15,
    )
    stem = "future_3d_gap_filled_tissue_selected_times_camera_view"
    save_figure(fig, output / stem)
    return {
        "stem": stem,
        "camera": "stereo_left",
        "selected_frames": list(selected_frames),
        "reference_outline_frame": 0,
    }


def plot_ground_truth_deformation_camera_view(
    data: EvaluationData, output: Path
) -> dict[str, object]:
    camera_surface = load_true_camera_surface(data)
    positions = camera_surface["positions_m"]
    displacement_mm = np.linalg.norm(positions - positions[0], axis=2) * 1.0e3
    maximum_frame = int(np.argmax(displacement_mm.mean(axis=1)))
    selected_frames = (0, maximum_frame, data.split_frame - 1, len(data.frames) - 1)
    norm = Normalize(vmin=0.0, vmax=max(float(displacement_mm.max()), 1.0e-6))
    fig, axes = plt.subplots(1, 4, figsize=(22, 6.2))
    labels = ("Rest", "Maximum deformation", "Last observed", "Future end")
    for ax, frame, label in zip(axes, selected_frames, labels):
        add_camera_aligned_filled_surface(
            ax,
            data,
            camera_surface,
            frame,
            displacement_mm[frame],
            norm,
            reference_frame=0,
            show_keypoint_labels=False,
        )
        ax.set_title(
            f"{label}\nframe {frame} | mean {float(displacement_mm[frame].mean()):.2f} mm | max {float(displacement_mm[frame].max()):.2f} mm",
            fontsize=9,
        )
    fig.subplots_adjust(left=0.015, right=0.91, top=0.82, bottom=0.05, wspace=0.04)
    color_axis = fig.add_axes((0.93, 0.16, 0.016, 0.64))
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap="turbo"), cax=color_axis
    )
    colorbar.set_label("Ground-truth displacement from frame 0 (mm)")
    fig.suptitle(
        "Actual tissue deformation in the exact stereo_left dataset camera view\n"
        "colors and projected triangle positions are both frame-specific ground truth",
        fontsize=15,
    )
    stem = "ground_truth_tissue_deformation_camera_view"
    save_figure(fig, output / stem)
    return {
        "stem": stem,
        "camera": "stereo_left",
        "selected_frames": list(selected_frames),
        "maximum_deformation_frame": maximum_frame,
        "future_mean_deformation_mm": {
            str(frame): float(displacement_mm[frame].mean())
            for frame in (data.split_frame, 311, 335, 359)
        },
    }


def write_numeric_outputs(
    data: EvaluationData,
    error_3d: dict[str, np.ndarray],
    error_2d_camera: dict[str, dict[str, np.ndarray]],
    error_2d_stereo: dict[str, np.ndarray],
    output: Path,
) -> None:
    fields = (
        "frame",
        "timestamp_s",
        "future_step",
        "keypoint",
        "node_id",
        "method",
        "error_3d_mm",
        "error_2d_left_px",
        "error_2d_right_px",
        "error_2d_stereo_mean_px",
    )
    csv_path = output / "future_point_errors.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for frame in range(data.split_frame, len(data.frames)):
            for point_index, node_id in enumerate(data.node_ids):
                for method, _label, _color in METHODS:
                    left = error_2d_camera[method]["stereo_left"][frame, point_index]
                    right = error_2d_camera[method]["stereo_right"][frame, point_index]
                    stereo = error_2d_stereo[method][frame, point_index]
                    writer.writerow(
                        {
                            "frame": frame,
                            "timestamp_s": float(data.timestamps[frame]),
                            "future_step": frame - data.split_frame + 1,
                            "keypoint": f"K{point_index + 1}",
                            "node_id": int(node_id),
                            "method": method,
                            "error_3d_mm": float(error_3d[method][frame, point_index]),
                            "error_2d_left_px": float(left) if np.isfinite(left) else "",
                            "error_2d_right_px": float(right) if np.isfinite(right) else "",
                            "error_2d_stereo_mean_px": float(stereo) if np.isfinite(stereo) else "",
                        }
                    )

    summaries: list[dict[str, object]] = []
    future = slice(data.split_frame, None)
    for point_index, node_id in enumerate(data.node_ids):
        for method, label, _color in METHODS:
            e3 = error_3d[method][future, point_index]
            e2 = error_2d_stereo[method][future, point_index]
            summaries.append(
                {
                    "keypoint": f"K{point_index + 1}",
                    "node_id": int(node_id),
                    "method": method,
                    "method_label": label,
                    "future_3d_mean_mm": float(np.mean(e3)),
                    "future_3d_rmse_mm": float(np.sqrt(np.mean(np.square(e3)))),
                    "future_3d_p95_mm": float(np.percentile(e3, 95.0)),
                    "future_2d_stereo_mean_px": float(np.nanmean(e2)),
                    "future_2d_stereo_rmse_px": float(np.sqrt(np.nanmean(np.square(e2)))),
                }
            )
    payload = {
        "schema": "fixedsuperbest.ten_point_evaluation_visualization.v1",
        "evaluation_root": str(output.parents[1].resolve()),
        "reference_dataset": str(data.dataset),
        "evaluated_nodes": 10,
        "evaluation_node_ids": data.node_ids.tolist(),
        "keypoint_labels": {
            f"K{i + 1}": int(node) for i, node in enumerate(data.node_ids)
        },
        "controlled_boundary_excluded": True,
        "selection_uses_future_errors": False,
        "observed_frames": [0, data.split_frame - 1],
        "future_frames": [data.split_frame, len(data.frames) - 1],
        "future_frame_count": len(data.frames) - data.split_frame,
        "coordinate_alignment": "direct node correspondence; no SE(3), scale, or temporal alignment",
        "summaries": summaries,
        "per_frame_csv": csv_path.name,
    }
    (output / "visualization_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_evaluation(args.evaluation_root)
    error_3d, error_2d_camera, error_2d_stereo = compute_errors(data)
    plot_2d_trajectories(data, args.output)
    plot_3d_trajectories(data, args.output)
    plot_temporal_heatmaps(
        data,
        error_3d,
        args.output,
        stem="future_3d_error_temporal_heatmaps",
        unit="mm",
        title="Future 3D correspondence gaps for the fixed 10 points",
    )
    plot_temporal_heatmaps(
        data,
        error_2d_stereo,
        args.output,
        stem="future_2d_error_temporal_heatmaps",
        unit="pixel",
        title="Future stereo-mean 2D correspondence gaps for the fixed 10 points",
    )
    plot_by_point_summary(data, error_3d, error_2d_stereo, args.output)
    plot_spatial_error_map(data, error_3d, args.output)
    surface_figures = [
        plot_filled_surface_summary(
            data,
            error_3d,
            args.output,
            stem="future_3d_rmse_filled_tissue_surface",
            unit="mm",
            title="Future 3D RMSE filled over the actual tissue surface",
        ),
        plot_filled_surface_summary(
            data,
            error_2d_stereo,
            args.output,
            stem="future_2d_rmse_filled_tissue_surface",
            unit="pixel",
            title="Future stereo-mean 2D RMSE filled over the actual tissue surface",
        ),
        plot_filled_surface_selected_times(data, error_3d, args.output),
        plot_camera_aligned_surface_summary(
            data,
            error_3d,
            args.output,
            stem="future_3d_rmse_filled_tissue_camera_view",
            unit="mm",
            title="Future 3D RMSE filled over the deforming tissue",
        ),
        plot_camera_aligned_surface_summary(
            data,
            error_2d_stereo,
            args.output,
            stem="future_2d_rmse_filled_tissue_camera_view",
            unit="pixel",
            title="Future stereo-mean 2D RMSE filled over the deforming tissue",
        ),
        plot_camera_aligned_selected_times(data, error_3d, args.output),
        plot_ground_truth_deformation_camera_view(data, args.output),
    ]
    write_surface_heatmap_manifest(data, args.output, surface_figures)
    write_numeric_outputs(
        data, error_3d, error_2d_camera, error_2d_stereo, args.output
    )
    print(f"[十点可视化] IDs={data.node_ids.tolist()}")
    print(
        f"[十点可视化] 未来区间={data.split_frame}..{len(data.frames) - 1} "
        f"({len(data.frames) - data.split_frame} frames)"
    )
    print(f"[十点可视化] 完成：{args.output.resolve()}")


if __name__ == "__main__":
    main()
