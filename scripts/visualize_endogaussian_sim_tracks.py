#!/usr/bin/env python3
"""Render fixed-point EndoGaussian predictions and GT as a side-by-side video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


CAMERAS = ("stereo_left", "stereo_right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--trail", type=int, default=30)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def colors(count: int) -> list[tuple[int, int, int]]:
    hsv = np.zeros((count, 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = np.linspace(0, 179, count, endpoint=False).astype(np.uint8)
    hsv[:, 0, 1] = 235
    hsv[:, 0, 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0]
    return [tuple(int(channel) for channel in value) for value in bgr]


def load_reference(dataset: Path, node_ids: np.ndarray):
    with np.load(dataset / "ground_truth" / "trajectories_3d.npz") as data:
        lookup = {int(node): i for i, node in enumerate(data["tissue_node_ids"])}
        columns = np.asarray([lookup[int(node)] for node in node_ids])
        xyz = np.asarray(data["tissue_positions_world"][:, columns], dtype=np.float32)
    camera_data = {}
    for camera in CAMERAS:
        with np.load(
            dataset / "ground_truth" / "trajectories_2d" / (camera + ".npz")
        ) as data:
            lookup = {int(node): i for i, node in enumerate(data["tissue_node_ids"])}
            columns = np.asarray([lookup[int(node)] for node in node_ids])
            camera_data[camera] = {
                "uv": np.asarray(data["tissue_uv_pixels"][:, columns], dtype=np.float32),
                "visible": np.asarray(data["tissue_visible"][:, columns], dtype=bool),
            }
    return xyz, camera_data


def draw_label(image: np.ndarray, text: str, origin: tuple[int, int], color) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(
        image,
        (x - 5, y - height - 7),
        (x + width + 5, y + baseline + 5),
        (0, 0, 0),
        -1,
    )
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def point(uv: np.ndarray, scale: float) -> tuple[int, int]:
    return int(round(float(uv[0]) * scale)), int(round(float(uv[1]) * scale))


def draw_tracks(
    image: np.ndarray,
    frame: int,
    pred_uv: np.ndarray,
    pred_valid: np.ndarray,
    gt_uv: np.ndarray,
    gt_visible: np.ndarray,
    palette: list[tuple[int, int, int]],
    scale: float,
    trail: int,
) -> None:
    start = max(0, frame - trail + 1)
    for node, color in enumerate(palette):
        for time in range(start + 1, frame + 1):
            if pred_valid[time - 1, node] and pred_valid[time, node]:
                cv2.line(
                    image,
                    point(pred_uv[time - 1, node], scale),
                    point(pred_uv[time, node], scale),
                    color,
                    2,
                    cv2.LINE_AA,
                )
            if gt_visible[time - 1, node] and gt_visible[time, node]:
                cv2.line(
                    image,
                    point(gt_uv[time - 1, node], scale),
                    point(gt_uv[time, node], scale),
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
        if gt_visible[frame, node]:
            gt = point(gt_uv[frame, node], scale)
            cv2.drawMarker(
                image, gt, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 9, 2, cv2.LINE_AA
            )
        if pred_valid[frame, node]:
            predicted = point(pred_uv[frame, node], scale)
            cv2.circle(image, predicted, 4, color, -1, cv2.LINE_AA)
            cv2.circle(image, predicted, 5, (0, 0, 0), 1, cv2.LINE_AA)
            if gt_visible[frame, node]:
                cv2.line(image, predicted, gt, (0, 230, 255), 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    prediction = args.prediction.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("拒绝覆盖已有轨迹视频：{}".format(output))
    if not 0.1 <= args.scale <= 1.0 or args.trail < 1 or args.fps <= 0:
        raise ValueError("scale、trail 或 fps 非法")

    with np.load(prediction, allow_pickle=False) as data:
        node_ids = np.asarray(data["tissue_node_ids"], dtype=np.int64)
        pred_xyz = np.asarray(data["tissue_positions_world"], dtype=np.float32)
        pred_uv = {
            camera: np.asarray(data[camera + "_tissue_uv_pixels"], dtype=np.float32)
            for camera in CAMERAS
        }
        pred_valid = {
            camera: np.asarray(data[camera + "_tissue_valid"], dtype=bool)
            for camera in CAMERAS
        }
    if node_ids.shape != (30,) or pred_xyz.ndim != 3:
        raise ValueError("预测文件不是固定 30 点轨迹格式")
    frames = pred_xyz.shape[0]
    gt_xyz, reference = load_reference(dataset, node_ids)
    if gt_xyz.shape != pred_xyz.shape:
        raise ValueError("预测与真值轨迹形状不一致")

    metadata = json.loads((dataset / "videos" / "stereo_left.json").read_text())
    width, height = (int(value) for value in metadata["resolution"])
    panel_width = int(round(width * args.scale))
    panel_height = int(round(height * args.scale))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (panel_width * 2, panel_height),
    )
    if not writer.isOpened():
        raise RuntimeError("无法创建视频：{}".format(output))

    palette = colors(len(node_ids))
    future_start = frames * 4 // 5
    try:
        for frame in range(frames):
            panels = []
            two_d_errors = []
            for camera in CAMERAS:
                image_path = dataset / "rgb" / camera / "{:06d}.png".format(frame)
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(image_path)
                image = cv2.resize(
                    image, (panel_width, panel_height), interpolation=cv2.INTER_AREA
                )
                draw_tracks(
                    image,
                    frame,
                    pred_uv[camera],
                    pred_valid[camera],
                    reference[camera]["uv"],
                    reference[camera]["visible"],
                    palette,
                    args.scale,
                    args.trail,
                )
                valid = pred_valid[camera][frame] & reference[camera]["visible"][frame]
                if np.any(valid):
                    two_d_errors.extend(
                        np.linalg.norm(
                            pred_uv[camera][frame, valid]
                            - reference[camera]["uv"][frame, valid],
                            axis=1,
                        ).tolist()
                    )
                draw_label(image, camera, (12, 25), (255, 255, 255))
                panels.append(image)

            valid_xyz = np.isfinite(pred_xyz[frame]).all(axis=1)
            error_3d = float(
                np.linalg.norm(
                    pred_xyz[frame, valid_xyz] - gt_xyz[frame, valid_xyz], axis=1
                ).mean()
                * 1000.0
            )
            error_2d = float(np.mean(two_d_errors)) if two_d_errors else float("nan")
            if frame >= future_start:
                phase = "FUTURE EXTRAPOLATION"
                phase_color = (80, 80, 255)
            elif frame % 8 == 7:
                phase = "RECON HOLDOUT"
                phase_color = (0, 230, 255)
            else:
                phase = "PREFIX TRAIN FRAME"
                phase_color = (80, 255, 80)
            canvas = np.concatenate(panels, axis=1)
            draw_label(
                canvas,
                "frame {:03d}/{:03d}  {}  mean 3D={:.2f} mm  2D={:.2f} px".format(
                    frame, frames - 1, phase, error_3d, error_2d
                ),
                (12, panel_height - 18),
                phase_color,
            )
            draw_label(
                canvas,
                "prediction: colored dot/trail | GT: white x/trail | error: yellow line",
                (panel_width + 12, panel_height - 18),
                (255, 255, 255),
            )
            writer.write(canvas)
            if (frame + 1) % 30 == 0 or frame + 1 == frames:
                print("video {}/{}".format(frame + 1, frames), flush=True)
    finally:
        writer.release()
    print(output, flush=True)


if __name__ == "__main__":
    main()
