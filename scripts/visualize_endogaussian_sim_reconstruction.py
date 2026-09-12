#!/usr/bin/env python3
"""Create a GT-versus-baseline tissue reconstruction video."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


CAMERAS = ("stereo_left", "stereo_right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.35)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--method-name", default="EndoGaussian")
    return parser.parse_args()


def label(image: np.ndarray, text: str, color=(255, 255, 255)) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(image, (5, 5), (width + 17, height + baseline + 17), (0, 0, 0), -1)
    cv2.putText(image, text, (11, height + 10), font, scale, color, thickness, cv2.LINE_AA)


def union_crop(mask: np.ndarray, alpha: np.ndarray, padding: int = 16):
    union = mask | (alpha > 0)
    y, x = np.nonzero(union)
    if not len(x):
        raise ValueError("真值和预测组织的并集为空")
    height, width = union.shape
    return (
        slice(max(int(y.min()) - padding, 0), min(int(y.max()) + padding + 1, height)),
        slice(max(int(x.min()) - padding, 0), min(int(x.max()) + padding + 1, width)),
    )


def psnr(reference: np.ndarray, prediction: np.ndarray) -> float:
    mse = float(np.mean((reference.astype(np.float32) - prediction.astype(np.float32)) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * math.log10((255.0**2) / mse)


def main() -> None:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    artifacts = args.artifacts.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("拒绝覆盖已有重建视频：{}".format(output))
    if not 0.1 <= args.scale <= 1.0 or args.fps <= 0:
        raise ValueError("scale 或 fps 非法")

    frame_sets = []
    for camera in CAMERAS:
        names = sorted(path.name for path in (artifacts / "rgb" / camera).glob("*.png"))
        if not names:
            raise ValueError("{} 没有 EndoGaussian 渲染".format(camera))
        frame_sets.append(names)
    if frame_sets[0] != frame_sets[1]:
        raise ValueError("左右相机渲染帧不一致")
    frames = [int(Path(name).stem) for name in frame_sets[0]]
    total_frames = int(
        __import__("json").loads((dataset / "episode.json").read_text())["frames"]
    )
    future_start = total_frames * 4 // 5
    wanted = [frame for frame in range(future_start) if frame % 8 == 7] + list(
        range(future_start, total_frames)
    )
    if frames != wanted:
        raise ValueError("渲染帧不符合固定 7:1 重建 + 后 20% 外推协议")

    first = cv2.imread(str(dataset / "rgb" / CAMERAS[0] / frame_sets[0][0]))
    if first is None:
        raise FileNotFoundError("无法读取第一帧")
    source_height, source_width = first.shape[:2]
    panel_width = int(round(source_width * args.scale))
    panel_height = int(round(source_height * args.scale))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (panel_width * 2, panel_height * 2),
    )
    if not writer.isOpened():
        raise RuntimeError("无法创建视频：{}".format(output))

    try:
        for local_index, (frame, name) in enumerate(zip(frames, frame_sets[0])):
            rows = []
            for camera in CAMERAS:
                reference = cv2.imread(str(dataset / "rgb" / camera / name))
                prediction = cv2.imread(str(artifacts / "rgb" / camera / name))
                mask = cv2.imread(
                    str(dataset / "ground_truth" / "masks" / "tissue" / camera / name),
                    cv2.IMREAD_GRAYSCALE,
                )
                alpha = cv2.imread(
                    str(artifacts / "alpha" / camera / name), cv2.IMREAD_GRAYSCALE
                )
                if any(value is None for value in (reference, prediction, mask, alpha)):
                    raise FileNotFoundError("frame={} camera={} 输入不完整".format(frame, camera))
                tissue_reference = reference.copy()
                tissue_reference[mask == 0] = 0
                crop_y, crop_x = union_crop(mask > 0, alpha)
                frame_psnr = psnr(
                    tissue_reference[crop_y, crop_x], prediction[crop_y, crop_x]
                )
                gt_panel = cv2.resize(
                    tissue_reference,
                    (panel_width, panel_height),
                    interpolation=cv2.INTER_AREA,
                )
                pred_panel = cv2.resize(
                    prediction,
                    (panel_width, panel_height),
                    interpolation=cv2.INTER_AREA,
                )
                phase = "FUTURE" if frame >= future_start else "RECON HOLDOUT"
                phase_color = (80, 80, 255) if frame >= future_start else (0, 230, 255)
                label(gt_panel, "{} | GT tissue | frame {}".format(camera, frame))
                label(
                    pred_panel,
                    "{} | {} | {} | PSNR {:.2f} dB".format(
                        camera, args.method_name, phase, frame_psnr
                    ),
                    phase_color,
                )
                rows.append(np.concatenate((gt_panel, pred_panel), axis=1))
            writer.write(np.concatenate(rows, axis=0))
            if (local_index + 1) % 15 == 0 or local_index + 1 == len(frames):
                print("video {}/{}".format(local_index + 1, len(frames)), flush=True)
    finally:
        writer.release()
    print(output, flush=True)


if __name__ == "__main__":
    main()
