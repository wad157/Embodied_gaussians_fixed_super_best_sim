#!/usr/bin/env python3
"""Select a uniform material initialization using training RGB only.

Every candidate is produced by the unchanged visual-residual plus online-
stiffness estimator. Selection uses four evenly spaced RGB diagnostics from
the 20 frames immediately before the train/future split. Trajectory ground
truth is read only after selection for an explicitly labelled audit table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--split-frame-index", type=int, default=240)
    parser.add_argument("--history-frames", type=int, default=20)
    parser.add_argument("--snapshot-count", type=int, default=4)
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=("soft", "nominal", "firm"),
        help="Candidate subdirectory names under --evaluation-root.",
    )
    return parser.parse_args()


def fmt(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    root = args.evaluation_root.expanduser().resolve()
    if args.history_frames < 1 or args.snapshot_count < 1:
        raise ValueError("history-frames and snapshot-count must be positive")
    history_start = args.split_frame_index - args.history_frames
    if history_start < 0:
        raise ValueError("RGB history extends before frame zero")
    requested = np.rint(
        np.linspace(
            history_start,
            args.split_frame_index - 1,
            args.snapshot_count,
        )
    ).astype(np.int64)
    if len(np.unique(requested)) != len(requested):
        raise ValueError("RGB snapshot schedule contains duplicates")

    rows: list[dict[str, object]] = []
    for name in args.candidates:
        candidate_root = root / name
        diagnostic_path = (
            candidate_root / "artifacts/rgb_alignment_diagnostics.npz"
        )
        material_path = candidate_root / "artifacts/material_diagnostics.npz"
        metadata_path = candidate_root / "artifacts/artifact_metadata.json"
        if not diagnostic_path.is_file():
            raise FileNotFoundError(diagnostic_path)
        if not material_path.is_file():
            raise FileNotFoundError(material_path)
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata["open_loop_start_frame"]) != history_start:
            raise ValueError(
                f"{name} 没有在内部验证起点 {history_start} 冻结反馈"
            )
        if int(metadata["frame_end_inclusive"]) != args.split_frame_index - 1:
            raise ValueError(
                f"{name} 必须只运行正式前 80%（结束于 "
                f"{args.split_frame_index - 1}）"
            )
        with np.load(diagnostic_path, allow_pickle=False) as diagnostic:
            frame_indices = np.asarray(diagnostic["frame_indices"], dtype=np.int64)
            rgb_losses = np.asarray(
                diagnostic["equal_camera_rgb_loss"], dtype=np.float64
            )
            reused = np.asarray(
                diagnostic["reused_residual_exact_validation"], dtype=bool
            )
        frame_to_slot = {
            int(frame): slot for slot, frame in enumerate(frame_indices)
        }
        missing = [int(frame) for frame in requested if int(frame) not in frame_to_slot]
        if missing:
            raise ValueError(f"{name} is missing RGB snapshots: {missing}")
        snapshot_losses = np.asarray(
            [rgb_losses[frame_to_slot[int(frame)]] for frame in requested],
            dtype=np.float64,
        )
        if not np.isfinite(snapshot_losses).all():
            raise ValueError(f"{name} contains non-finite RGB snapshot losses")
        if bool(
            np.any(
                [reused[frame_to_slot[int(frame)]] for frame in requested]
            )
        ):
            raise ValueError(
                f"{name} 的内部验证 RGB 仍复用了视觉残差更新结果"
            )
        with np.load(material_path, allow_pickle=False) as material:
            distance_initial = np.asarray(
                material["distance_stiffness"], dtype=np.float64
            )[0]
            shape_initial = np.asarray(
                material["shape_stiffness"], dtype=np.float64
            )[0]
        row: dict[str, object] = {
            "candidate": name,
            "initial_distance_min_median_max": distance_initial.tolist(),
            "initial_shape_min_median_max": shape_initial.tolist(),
            "rgb_snapshot_frames": requested.tolist(),
            "rgb_snapshot_losses": snapshot_losses.tolist(),
            "rgb_selection_score": float(snapshot_losses.mean()),
        }
        rows.append(row)

    selected = min(rows, key=lambda row: float(row["rgb_selection_score"]))
    for row in rows:
        row["selected_by_rgb"] = row is selected
    report = {
        "schema": "fixedsuperbest.sim_rgb_material_multistart.v2",
        "selection_input": "RGB only; no depth, trajectory GT, or material GT",
        "estimator_per_candidate": (
            "unchanged PBD + visual residual + online stiffness update"
        ),
        "split_frame_index": args.split_frame_index,
        "internal_training_frames": [0, history_start - 1],
        "internal_validation_frames": [history_start, args.split_frame_index - 1],
        "history_frames": args.history_frames,
        "snapshot_count": args.snapshot_count,
        "snapshot_frames": requested.tolist(),
        "selection_metric": "mean equal-camera masked robust RGB loss",
        "selection_protocol": (
            "frames 0..219 online fit; frames 220..239 feedback frozen and "
            "scored; formal frames 240..299 never loaded"
        ),
        "selected_candidate": selected["candidate"],
        "candidates": rows,
    }
    json_path = root / "rgb_material_selection.json"
    markdown_path = root / "rgb_material_selection.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("Refusing to overwrite RGB material selection report")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# RGB-only 多初值材料分支选择",
        "",
        "选择阶段只使用正式前 80% 内的 RGB：0–219 帧拟合，220–239 帧冻结后验证；不使用深度、轨迹真值或材料真值。",
        "",
        "| 分支 | 初始 distance 中位数 | 初始 shape 中位数 | 冻结验证 RGB 分数 ↓ | 选择 |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {distance} | {shape} | {rgb} | {selected} |".format(
                name=row["candidate"],
                distance=fmt(row["initial_distance_min_median_max"][1], 4),
                shape=fmt(row["initial_shape_min_median_max"][1], 5),
                rgb=fmt(row["rgb_selection_score"], 8),
                selected="是" if row["selected_by_rgb"] else "",
            )
        )
    lines.extend(
        (
            "",
            f"RGB-only 选中：`{selected['candidate']}`。",
            "",
        )
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
