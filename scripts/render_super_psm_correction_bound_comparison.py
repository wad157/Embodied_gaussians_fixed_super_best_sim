#!/usr/bin/env python3

"""Compose honest A/B projections for two PSM correction-bound runs.

The source images are the actual NvDiffRast diagnostics produced by
track_super_psm_part_corrected.py.  This script only crops the shared
observation and each run's corrected panel, adds explicit labels, and writes
per-frame comparisons plus a contact sheet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[1]
TRACK_ROOT = REPO / "data/super/psm_tracking"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare original and expanded PSM correction bounds."
    )
    parser.add_argument(
        "--original-dir",
        type=Path,
        default=TRACK_ROOT / "part_pose_correction_full",
    )
    parser.add_argument(
        "--expanded-dir",
        type=Path,
        default=TRACK_ROOT
        / "part_pose_correction_expanded_3mm_8deg_12deg_25deg",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TRACK_ROOT / "correction_bound_comparison",
    )
    parser.add_argument(
        "--original-gui-dir",
        type=Path,
        default=TRACK_ROOT / "gui_projection_original_bounds",
    )
    parser.add_argument(
        "--expanded-gui-dir",
        type=Path,
        default=TRACK_ROOT / "gui_projection_expanded_bounds",
    )
    parser.add_argument(
        "--frames", default="0,480,640,800,1120,1280,1440"
    )
    return parser.parse_args()


def labelled_panel(image: np.ndarray, label: str) -> np.ndarray:
    band_height = 52
    result = np.zeros(
        (image.shape[0] + band_height, image.shape[1], 3), dtype=np.uint8
    )
    result[band_height:] = image
    cv2.putText(
        result,
        label,
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def corrected_panel(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    if image.shape[1] % 4 != 0:
        raise ValueError(f"Expected a four-panel diagnostic: {path}")
    width = image.shape[1] // 4
    return image[:, 3 * width : 4 * width]


def observation_panel(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    if image.shape[1] % 4 != 0:
        raise ValueError(f"Expected a four-panel diagnostic: {path}")
    return image[:, : image.shape[1] // 4]


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def tip_errors(report: dict, frame: int) -> list[float]:
    tips = report["diagnostic_metrics"][str(frame)]["part_corrected"]["tips"]
    return [
        float(tips["jaw_left"]["error_px"]),
        float(tips["jaw_right"]["error_px"]),
    ]


def main() -> None:
    args = parse_args()
    frames = [int(value) for value in args.frames.split(",") if value]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_report = json.loads(
        (args.original_dir / "report.json").read_text(encoding="utf-8")
    )
    expanded_report = json.loads(
        (args.expanded_dir / "report.json").read_text(encoding="utf-8")
    )

    rows = []
    frame_metrics = {}
    for frame in frames:
        name = f"frame{frame:06d}_comparison.png"
        original_path = args.original_dir / name
        expanded_path = args.expanded_dir / name
        original_error = tip_errors(original_report, frame)
        expanded_error = tip_errors(expanded_report, frame)
        panels = [
            labelled_panel(
                observation_panel(original_path),
                f"FRAME {frame}: THREE-PART OBSERVATION",
            ),
            labelled_panel(
                corrected_panel(original_path),
                "ORIGINAL LIMITS 2mm / 5deg / wrist8deg / jaw20deg  "
                f"tips={original_error[0]:.1f}/{original_error[1]:.1f}px",
            ),
            labelled_panel(
                corrected_panel(expanded_path),
                "EXPANDED LIMITS 3mm / 8deg / wrist12deg / jaw25deg  "
                f"tips={expanded_error[0]:.1f}/{expanded_error[1]:.1f}px",
            ),
        ]
        comparison = np.hstack(panels)
        output_path = args.output_dir / f"frame{frame:06d}_bounds.png"
        if not cv2.imwrite(str(output_path), comparison):
            raise RuntimeError(f"Failed to write {output_path}")
        rows.append(
            cv2.resize(comparison, (1440, 296), interpolation=cv2.INTER_AREA)
        )
        frame_metrics[str(frame)] = {
            "original_tip_error_left_right_px": original_error,
            "expanded_tip_error_left_right_px": expanded_error,
            "image": str(output_path),
        }

    contact_sheet = np.vstack(rows)
    contact_sheet_path = args.output_dir / "contact_sheet.png"
    if not cv2.imwrite(str(contact_sheet_path), contact_sheet):
        raise RuntimeError(f"Failed to write {contact_sheet_path}")
    gui_rows = []
    gui_images = {}
    for frame in frames:
        name = f"frame{frame:06d}_lnd_dvrk.png"
        panels = [
            labelled_panel(
                read_image(args.original_gui_dir / name),
                "ORIGINAL LIMITS: ACTUAL GUI TIP GAUSSIANS (CYAN)",
            ),
            labelled_panel(
                read_image(args.expanded_gui_dir / name),
                "EXPANDED LIMITS: ACTUAL GUI TIP GAUSSIANS (CYAN)",
            ),
        ]
        comparison = np.hstack(panels)
        output_path = args.output_dir / f"frame{frame:06d}_gui_bounds.png"
        if not cv2.imwrite(str(output_path), comparison):
            raise RuntimeError(f"Failed to write {output_path}")
        gui_rows.append(
            cv2.resize(comparison, (1440, 424), interpolation=cv2.INTER_AREA)
        )
        gui_images[str(frame)] = str(output_path)
    gui_contact_sheet = np.vstack(gui_rows)
    gui_contact_sheet_path = args.output_dir / "gui_contact_sheet.png"
    if not cv2.imwrite(str(gui_contact_sheet_path), gui_contact_sheet):
        raise RuntimeError(f"Failed to write {gui_contact_sheet_path}")

    summary = {
        "source": "actual NvDiffRast diagnostic panels; no synthetic image edit",
        "original_limits": {
            "translation_mm": 2.0,
            "rotation_deg": 5.0,
            "wrist_deg": 8.0,
            "jaw_deg": 20.0,
            "optimized_loss_p50_p95": original_report[
                "optimized_loss_min_p05_p50_p95_max"
            ][2:4],
        },
        "expanded_limits": {
            "translation_mm": 3.0,
            "rotation_deg": 8.0,
            "wrist_deg": 12.0,
            "jaw_deg": 25.0,
            "optimized_loss_p50_p95": expanded_report[
                "optimized_loss_min_p05_p50_p95_max"
            ][2:4],
        },
        "frames": frame_metrics,
        "contact_sheet": str(contact_sheet_path),
        "gui_frames": gui_images,
        "gui_contact_sheet": str(gui_contact_sheet_path),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
