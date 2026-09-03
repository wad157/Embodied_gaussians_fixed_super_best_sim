#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pick exactly two PSM jaw-tip keypoints on the first video frame."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=REPO
        / "data/super/grasp5_offline_demo/videos/stereo_left.mp4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO
        / "data/super/psm_tracking/online_videos/grasp5/PSM1_keypoints.txt",
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=2,
        help="Scale full-resolution clicks to the paper tracker's image resolution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.downsample_factor < 1:
        raise ValueError("--downsample-factor must be positive")
    cap = cv2.VideoCapture(str(args.video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read the first frame from {args.video}")

    base = frame.copy()
    view = frame.copy()
    points: list[tuple[int, int]] = []
    window = "Pick exactly two jaw tips"

    def redraw(message: str = "") -> None:
        view[:] = base
        for index, point in enumerate(points):
            cv2.circle(view, point, 8, (255, 0, 0), -1)
            cv2.putText(
                view,
                f"tip {index + 1}",
                (point[0] + 12, point[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
        lines = [
            "Left click: one point at each jaw's sharp endpoint (exactly 2)",
            "ENTER: save | r: reset | q/ESC: quit",
        ]
        if message:
            lines.append(message)
        for index, text in enumerate(lines):
            cv2.putText(
                view,
                text,
                (20, 40 + index * 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(points) < 2:
            points.append((x, y))
        redraw()

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    redraw()
    while True:
        cv2.imshow(window, view)
        key = cv2.waitKey(10) & 0xFF
        if key == 13:
            if len(points) == 2:
                break
            redraw(f"Need 2 points; currently have {len(points)}")
        elif key == ord("r"):
            points.clear()
            redraw()
        elif key in (ord("q"), 27):
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()
    keypoints = np.asarray(points, dtype=np.float64) / float(args.downsample_factor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, np.rint(keypoints).astype(np.int32), fmt="%d")
    print(f"Saved two downsampled jaw-tip keypoints to {args.output}")


if __name__ == "__main__":
    main()
