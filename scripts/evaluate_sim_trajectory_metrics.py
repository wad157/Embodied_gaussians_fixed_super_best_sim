#!/usr/bin/env python3
"""使用仿真特权真值评估重建导出的组织 3D 与双目 2D 轨迹。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


CAMERAS = ("stereo_left", "stereo_right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dataset", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        type=Path,
        required=True,
        help=(
            "预测 NPZ，必须包含 timestamps、tissue_node_ids、tissue_positions_world，"
            "以及 stereo_left_tissue_uv_pixels/stereo_right_tissue_uv_pixels。"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evaluation-node-manifest",
        type=Path,
        default=None,
        help=(
            "可选固定评估点 JSON；必须含 tissue_node_ids。所列节点仍须属于数据集"
            "评估集合且不能位于夹持控制排除区。"
        ),
    )
    parser.add_argument(
        "--controlled-boundary",
        type=Path,
        default=None,
        help=(
            "可选的本次运行已知运动边界NPZ；用其中的"
            "evaluation_exclusion_tissue_node_ids排除控制区。"
        ),
    )
    parser.add_argument(
        "--split-frame-index",
        type=int,
        default=-1,
        help="可选训练/未来预测分界帧；正式 80/20 协议使用 240。",
    )
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument(
        "--frame-end-exclusive",
        type=int,
        default=-1,
        help="-1 表示不限制末帧。",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="只统计满足 frame %% stride == offset 的帧。",
    )
    parser.add_argument("--frame-offset", type=int, default=0)
    return parser.parse_args()


def distribution_metrics(errors: np.ndarray, thresholds: tuple[float, ...], scale: float):
    values = np.asarray(errors, dtype=np.float64) * scale
    if not len(values):
        return {"count": 0}
    result = {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(values.max()),
    }
    for threshold in thresholds:
        result[f"fraction_le_{threshold:g}"] = float(np.mean(values <= threshold))
    return result


def align_prediction_nodes(reference_ids: np.ndarray, prediction_ids: np.ndarray) -> np.ndarray:
    lookup = {int(node_id): index for index, node_id in enumerate(prediction_ids)}
    missing = [int(node_id) for node_id in reference_ids if int(node_id) not in lookup]
    if missing:
        raise ValueError(
            f"预测缺少 {len(missing)} 个固定评估节点，前十个为：{missing[:10]}"
        )
    return np.asarray([lookup[int(node_id)] for node_id in reference_ids], dtype=np.int64)


def main() -> None:
    args = parse_args()
    root = args.reference_dataset.expanduser().resolve()
    prediction_path = args.prediction.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() or output.with_suffix(".csv").exists():
        raise FileExistsError(f"拒绝覆盖已有轨迹评估：{output}")
    prediction = np.load(prediction_path)
    reference_3d = np.load(root / "ground_truth/trajectories_3d.npz")
    required = {
        "timestamps",
        "tissue_node_ids",
        "tissue_positions_world",
        "stereo_left_tissue_uv_pixels",
        "stereo_right_tissue_uv_pixels",
    }
    missing = sorted(required.difference(prediction.files))
    if missing:
        raise ValueError(f"预测 NPZ 缺少字段：{missing}")
    all_timestamps = reference_3d["timestamps"].astype(np.float64)
    prediction_frame_indices = (
        prediction["frame_indices"].astype(np.int64)
        if "frame_indices" in prediction.files
        else np.arange(len(prediction["timestamps"]), dtype=np.int64)
    )
    if (
        prediction_frame_indices.ndim != 1
        or len(prediction_frame_indices) != len(prediction["timestamps"])
        or len(np.unique(prediction_frame_indices)) != len(prediction_frame_indices)
        or np.any(prediction_frame_indices < 0)
        or np.any(prediction_frame_indices >= len(all_timestamps))
    ):
        raise ValueError("预测 frame_indices 非法或与时间戳长度不一致")
    if np.any(np.diff(prediction_frame_indices) <= 0):
        raise ValueError("预测 frame_indices 必须严格递增")
    if args.frame_stride < 1:
        raise ValueError("frame_stride 必须为正整数")
    if not 0 <= args.frame_offset < args.frame_stride:
        raise ValueError("frame_offset 必须位于 [0, frame_stride)")
    if args.frame_start < 0:
        raise ValueError("frame_start 不能为负")
    if (
        args.frame_end_exclusive >= 0
        and args.frame_end_exclusive <= args.frame_start
    ):
        raise ValueError("frame_end_exclusive 必须大于 frame_start")
    selection = prediction_frame_indices >= args.frame_start
    if args.frame_end_exclusive >= 0:
        selection &= prediction_frame_indices < args.frame_end_exclusive
    selection &= (
        prediction_frame_indices % args.frame_stride == args.frame_offset
    )
    if not bool(selection.any()):
        raise ValueError("帧选择规则没有留下任何轨迹评估帧")
    frame_indices = prediction_frame_indices[selection]
    timestamps = all_timestamps[frame_indices]
    if not np.allclose(
        prediction["timestamps"][selection], timestamps, atol=1.0e-9
    ):
        raise ValueError("预测时间戳与真值不一致；评估不进行时间插值或挑帧")

    all_evaluation_ids = reference_3d["tissue_evaluation_node_ids"].astype(
        np.int64
    )
    boundary_path = (
        args.controlled_boundary.expanduser().resolve()
        if args.controlled_boundary is not None
        else root / "task_inputs" / "red_marker_boundary.npz"
    )
    controlled_exclusion_ids = np.empty(0, dtype=np.int64)
    if boundary_path.is_file():
        with np.load(boundary_path, allow_pickle=False) as boundary:
            if "evaluation_exclusion_tissue_node_ids" in boundary.files:
                controlled_exclusion_ids = np.asarray(
                    boundary["evaluation_exclusion_tissue_node_ids"],
                    dtype=np.int64,
                )
    allowed_evaluation_ids = all_evaluation_ids[
        ~np.isin(all_evaluation_ids, controlled_exclusion_ids)
    ]
    evaluation_manifest_path = None
    if args.evaluation_node_manifest is not None:
        evaluation_manifest_path = args.evaluation_node_manifest.expanduser().resolve()
        manifest = json.loads(evaluation_manifest_path.read_text(encoding="utf-8"))
        evaluation_ids = np.asarray(manifest["tissue_node_ids"], dtype=np.int64)
        if evaluation_ids.ndim != 1 or len(evaluation_ids) == 0:
            raise ValueError("固定评估点清单必须是一维非空 tissue_node_ids")
        if len(np.unique(evaluation_ids)) != len(evaluation_ids):
            raise ValueError("固定评估点清单含有重复节点")
        disallowed = evaluation_ids[
            ~np.isin(evaluation_ids, allowed_evaluation_ids)
        ]
        if len(disallowed):
            raise ValueError(
                "固定评估点不属于允许集合或落在夹持排除区："
                f"{disallowed.tolist()}"
            )
    else:
        evaluation_ids = allowed_evaluation_ids
    if len(evaluation_ids) == 0:
        raise ValueError("排除已知位移边界后没有剩余的轨迹评估节点")
    reference_all_ids = reference_3d["tissue_node_ids"].astype(np.int64)
    reference_lookup = {int(node_id): index for index, node_id in enumerate(reference_all_ids)}
    reference_columns = np.asarray(
        [reference_lookup[int(node_id)] for node_id in evaluation_ids], dtype=np.int64
    )
    prediction_columns = align_prediction_nodes(
        evaluation_ids, prediction["tissue_node_ids"].astype(np.int64)
    )

    reference_positions = reference_3d["tissue_positions_world"][frame_indices][
        :, reference_columns
    ]
    prediction_positions = prediction["tissue_positions_world"][selection][
        :, prediction_columns
    ]
    if prediction_positions.shape != reference_positions.shape:
        raise ValueError(
            f"预测 3D 形状不一致：{prediction_positions.shape} != {reference_positions.shape}"
        )
    valid_3d = np.isfinite(reference_positions).all(axis=2) & np.isfinite(
        prediction_positions
    ).all(axis=2)
    error_3d_m = np.linalg.norm(prediction_positions - reference_positions, axis=2)
    camera_errors: dict[str, np.ndarray] = {}
    camera_valid: dict[str, np.ndarray] = {}
    camera_reference_visible: dict[str, np.ndarray] = {}
    for camera_name in CAMERAS:
        reference_2d = np.load(
            root / "ground_truth/trajectories_2d" / f"{camera_name}.npz"
        )
        reference_uv = reference_2d["tissue_uv_pixels"][frame_indices][
            :, reference_columns
        ]
        reference_visible = reference_2d["tissue_visible"][frame_indices][
            :, reference_columns
        ]
        prediction_uv = prediction[f"{camera_name}_tissue_uv_pixels"][selection][
            :, prediction_columns
        ]
        if prediction_uv.shape != reference_uv.shape:
            raise ValueError(
                f"{camera_name} 预测 2D 形状不一致："
                f"{prediction_uv.shape} != {reference_uv.shape}"
            )
        predicted_valid_key = f"{camera_name}_tissue_valid"
        predicted_valid = (
            prediction[predicted_valid_key][selection][
                :, prediction_columns
            ].astype(bool)
            if predicted_valid_key in prediction.files
            else np.isfinite(prediction_uv).all(axis=2)
        )
        valid = (
            reference_visible
            & predicted_valid
            & np.isfinite(reference_uv).all(axis=2)
            & np.isfinite(prediction_uv).all(axis=2)
        )
        error_px = np.linalg.norm(prediction_uv - reference_uv, axis=2)
        camera_errors[camera_name] = error_px
        camera_valid[camera_name] = valid
        camera_reference_visible[camera_name] = reference_visible

    def segment_metrics(frame_mask: np.ndarray) -> dict[str, object]:
        selected_3d = valid_3d & frame_mask[:, None]
        metrics_3d = distribution_metrics(
            error_3d_m[selected_3d],
            thresholds=(1.0, 2.0, 5.0, 10.0),
            scale=1000.0,
        )
        metrics_3d["unit"] = "mm"
        metrics_3d["valid_fraction"] = float(
            selected_3d.sum()
            / max(int(frame_mask.sum()) * valid_3d.shape[1], 1)
        )
        metrics_2d: dict[str, object] = {}
        for camera_name in CAMERAS:
            visible = camera_reference_visible[camera_name] & frame_mask[:, None]
            valid = camera_valid[camera_name] & frame_mask[:, None]
            camera_metrics = distribution_metrics(
                camera_errors[camera_name][valid],
                thresholds=(1.0, 3.0, 5.0, 10.0),
                scale=1.0,
            )
            camera_metrics["unit"] = "pixel"
            camera_metrics["reference_visible_count"] = int(visible.sum())
            camera_metrics["evaluated_fraction_of_visible"] = float(
                valid.sum() / max(int(visible.sum()), 1)
            )
            metrics_2d[camera_name] = camera_metrics
        return {
            "frame_count": int(frame_mask.sum()),
            "3d": metrics_3d,
            "2d": metrics_2d,
        }

    all_mask = np.ones(len(frame_indices), dtype=bool)
    segments: dict[str, dict[str, object]] = {
        "all": segment_metrics(all_mask)
    }
    if args.split_frame_index >= 0:
        assimilation_mask = frame_indices < args.split_frame_index
        future_mask = frame_indices >= args.split_frame_index
        if not bool(assimilation_mask.any()) or not bool(future_mask.any()):
            raise ValueError("训练/未来分界必须把当前预测划分成两个非空区间")
        segments["assimilation_0_79_percent"] = segment_metrics(
            assimilation_mask
        )
        segments["future_open_loop_20_percent"] = segment_metrics(future_mask)

    rows: list[dict[str, object]] = []
    for row_index, frame_index in enumerate(frame_indices):
        selected = error_3d_m[row_index][valid_3d[row_index]] * 1.0e3
        row: dict[str, object] = {
            "frame": int(frame_index),
            "timestamp_s": float(timestamps[row_index]),
            "segment": (
                "future_open_loop_20_percent"
                if args.split_frame_index >= 0
                and frame_index >= args.split_frame_index
                else "assimilation_0_79_percent"
            ),
            "track_3d_mean_mm": float(selected.mean()) if len(selected) else math.nan,
            "track_3d_rmse_mm": (
                float(np.sqrt(np.mean(selected * selected)))
                if len(selected)
                else math.nan
            ),
        }
        for camera_name in CAMERAS:
            selected_2d = camera_errors[camera_name][row_index][
                camera_valid[camera_name][row_index]
            ]
            row[f"{camera_name}_track_2d_mean_px"] = (
                float(selected_2d.mean()) if len(selected_2d) else math.nan
            )
            row[f"{camera_name}_track_2d_rmse_px"] = (
                float(np.sqrt(np.mean(selected_2d * selected_2d)))
                if len(selected_2d)
                else math.nan
            )
        rows.append(row)

    report = {
        "schema": "fixedsuperbest.trajectory_metrics.v2",
        "reference_dataset": str(root),
        "prediction": str(prediction_path),
        "frames": int(len(timestamps)),
        "frame_start": int(frame_indices[0]),
        "frame_end_inclusive": int(frame_indices[-1]),
        "split_frame_index": (
            None if args.split_frame_index < 0 else args.split_frame_index
        ),
        "frame_selection": {
            "start": args.frame_start,
            "end_exclusive": (
                None
                if args.frame_end_exclusive < 0
                else args.frame_end_exclusive
            ),
            "stride": args.frame_stride,
            "offset": args.frame_offset,
            "selected_frame_indices": frame_indices.tolist(),
        },
        "fixed_evaluation_nodes_before_control_exclusion": int(
            len(all_evaluation_ids)
        ),
        "evaluated_nodes": int(len(evaluation_ids)),
        "evaluation_node_ids": evaluation_ids.tolist(),
        "evaluation_node_manifest": (
            str(evaluation_manifest_path)
            if evaluation_manifest_path is not None
            else None
        ),
        "controlled_boundary_evaluation_nodes_excluded": int(
            len(all_evaluation_ids) - len(allowed_evaluation_ids)
        ),
        "allowed_nodes_not_selected_by_fixed_manifest": int(
            len(allowed_evaluation_ids) - len(evaluation_ids)
        ),
        "controlled_boundary_source": (
            str(boundary_path) if boundary_path.is_file() else None
        ),
        "alignment": "按 tissue_node_ids 精确匹配；不做 SE(3)、尺度或时间对齐",
        "3d": segments["all"]["3d"],
        "2d": segments["all"]["2d"],
        "segments": segments,
        "per_frame_csv": output.with_suffix(".csv").name,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
