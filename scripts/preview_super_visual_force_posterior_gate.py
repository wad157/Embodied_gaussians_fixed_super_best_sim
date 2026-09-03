#!/usr/bin/env python3
"""Render stereo before/after diagnostics for the posterior visual-force gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_gaussians.embodied_simulator.visual_force_masks import (  # noqa: E402
    PackedStereoInstrumentMasks,
    PackedTissueVisualForceWeights,
    tool_conditioned_tissue_weights,
)


DATASET_ROOT = REPO_ROOT / "data/super/grasp5_offline_demo"
NATIVE_ROOT = REPO_ROOT / "data/super/grasp5_native"
INSTRUMENT_MASKS = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_paper_lnd_stereo_dense_contact_v4/"
    "surgicalsam2_multianchor_parts_dense_contact_v6/"
    "stereo_multianchor_part_masks.npz"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs/super_visual_force_posterior_gate/"
    "frame_0420_stereo_before_after.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=int, default=420)
    parser.add_argument("--posterior-full-reach-px", type=float, default=140.0)
    parser.add_argument("--posterior-zero-reach-px", type=float, default=280.0)
    parser.add_argument("--tool-falloff-power", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_video_frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not decode frame {frame_index}: {path}")
    return frame


def add_label(image: np.ndarray, label: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 58), (0, 0, 0), -1)
    cv2.putText(
        image,
        label,
        (20, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def outline_mask(
    image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int
) -> None:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, color, thickness, cv2.LINE_AA)


def weight_overlay(
    frame: np.ndarray,
    weights: np.ndarray,
    tissue: np.ndarray,
    tool: np.ndarray,
    interaction: np.ndarray,
    label: str,
    posterior_reference_y: float,
    posterior_full_reach_px: float,
    posterior_zero_reach_px: float,
) -> np.ndarray:
    heat = cv2.applyColorMap(
        np.rint(np.clip(weights, 0.0, 1.0) * 255.0).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    output = np.rint(frame.astype(np.float32) * 0.28).astype(np.uint8)
    support = weights > 0.0
    output[support] = np.rint(
        0.32 * frame[support].astype(np.float32)
        + 0.68 * heat[support].astype(np.float32)
    ).astype(np.uint8)
    outline_mask(output, tissue, (255, 255, 255), 2)
    outline_mask(output, tool, (0, 0, 0), 4)
    outline_mask(output, interaction, (255, 0, 255), 3)
    full_y = int(round(posterior_reference_y - posterior_full_reach_px))
    zero_y = int(round(posterior_reference_y - posterior_zero_reach_px))
    if 0 <= full_y < output.shape[0]:
        cv2.line(output, (0, full_y), (output.shape[1] - 1, full_y), (0, 255, 255), 3)
    if 0 <= zero_y < output.shape[0]:
        cv2.line(output, (0, zero_y), (output.shape[1] - 1, zero_y), (0, 0, 255), 3)
    add_label(output, label)
    return output


def removed_overlay(
    frame: np.ndarray,
    removed: np.ndarray,
    tissue: np.ndarray,
    label: str,
) -> np.ndarray:
    output = np.rint(frame.astype(np.float32) * 0.28).astype(np.uint8)
    heat = cv2.applyColorMap(
        np.rint(np.clip(removed, 0.0, 1.0) * 255.0).astype(np.uint8),
        cv2.COLORMAP_MAGMA,
    )
    support = removed > 1.0e-6
    output[support] = np.rint(
        0.20 * frame[support].astype(np.float32)
        + 0.80 * heat[support].astype(np.float32)
    ).astype(np.uint8)
    outline_mask(output, tissue, (255, 255, 255), 2)
    add_label(output, label)
    return output


def main() -> None:
    args = parse_args()
    if args.frame < 0:
        raise ValueError("frame must be non-negative")
    tissue_assets = {
        "stereo_left": NATIVE_ROOT / "visual_force_masks_v1",
        "stereo_right": NATIVE_ROOT / "visual_force_masks_right_v1",
    }
    videos = {
        "stereo_left": DATASET_ROOT / "videos/stereo_left.mp4",
        "stereo_right": DATASET_ROOT / "videos/stereo_right.mp4",
    }
    instrument_provider = PackedStereoInstrumentMasks(
        INSTRUMENT_MASKS, maximum_frame_gap=1
    )
    rows: list[np.ndarray] = []
    statistics: dict[str, dict[str, int | float]] = {}
    for camera_name in ("stereo_left", "stereo_right"):
        tissue_provider = PackedTissueVisualForceWeights(
            tissue_assets[camera_name], erosion_radius_px=7
        )
        frame = read_video_frame(videos[camera_name], args.frame)
        tissue = tissue_provider.raw_mask_at_index(args.frame)
        pair = instrument_provider.masks_for_camera_frame(camera_name, args.frame)
        if pair is None:
            raise RuntimeError(f"No instrument mask for {camera_name}:{args.frame}")
        tool_low, interaction_low = pair
        tool = cv2.resize(
            tool_low.astype(np.uint8),
            (tissue.shape[1], tissue.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        interaction = cv2.resize(
            interaction_low.astype(np.uint8),
            (tissue.shape[1], tissue.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        old_weights = tool_conditioned_tissue_weights(
            tissue,
            tool_low,
            interaction_mask=interaction_low,
            tool_falloff_power=1.0,
            posterior_full_reach_px=None,
            posterior_zero_reach_px=None,
        )
        new_weights = tool_conditioned_tissue_weights(
            tissue,
            tool_low,
            interaction_mask=interaction_low,
            tool_falloff_power=args.tool_falloff_power,
            posterior_full_reach_px=args.posterior_full_reach_px,
            posterior_zero_reach_px=args.posterior_zero_reach_px,
        )
        removed = np.clip(old_weights - new_weights, 0.0, 1.0)
        scale_y = interaction_low.shape[0] / float(tissue.shape[0])
        reference_y = float(np.median(np.nonzero(interaction_low)[0]) / scale_y)

        original = frame.copy()
        outline_mask(original, tissue, (255, 255, 255), 2)
        outline_mask(original, interaction, (255, 0, 255), 3)
        add_label(original, f"{camera_name}: RGB + masks")
        before = weight_overlay(
            frame,
            old_weights,
            tissue,
            tool,
            interaction,
            "Before: power-1 isotropic tool SDF",
            reference_y,
            args.posterior_full_reach_px,
            args.posterior_zero_reach_px,
        )
        after = weight_overlay(
            frame,
            new_weights,
            tissue,
            tool,
            interaction,
            "After: power-2 far decay + posterior fade",
            reference_y,
            args.posterior_full_reach_px,
            args.posterior_zero_reach_px,
        )
        removed_panel = removed_overlay(
            frame, removed, tissue, "Removed visual-force weight"
        )
        panels = [original, before, after, removed_panel]
        panels = [cv2.resize(panel, (960, 540), interpolation=cv2.INTER_AREA) for panel in panels]
        rows.append(np.concatenate(panels, axis=1))
        posterior_rows = (
            reference_y - np.arange(tissue.shape[0])[:, None]
            >= args.posterior_zero_reach_px
        )
        statistics[camera_name] = {
            "posterior_reference_y_px": reference_y,
            "old_nonzero_pixels": int(np.count_nonzero(old_weights > 0.0)),
            "new_nonzero_pixels": int(np.count_nonzero(new_weights > 0.0)),
            "old_weight_sum": float(old_weights.sum()),
            "new_weight_sum": float(new_weights.sum()),
            "old_mean_nonzero_weight": float(
                old_weights[old_weights > 0.0].mean()
            ),
            "new_mean_nonzero_weight": float(
                new_weights[new_weights > 0.0].mean()
            ),
            "removed_nonzero_pixels": int(np.count_nonzero(removed > 1.0e-6)),
            "old_nonzero_beyond_posterior_zero": int(
                np.count_nonzero((old_weights > 0.0) & posterior_rows)
            ),
            "new_nonzero_beyond_posterior_zero": int(
                np.count_nonzero((new_weights > 0.0) & posterior_rows)
            ),
        }

    montage = np.concatenate(rows, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), montage):
        raise RuntimeError(f"Could not write preview: {args.output}")
    report = {
        "frame": args.frame,
        "posterior_full_reach_px": args.posterior_full_reach_px,
        "posterior_zero_reach_px": args.posterior_zero_reach_px,
        "tool_falloff_power": args.tool_falloff_power,
        "statistics": statistics,
        "output": str(args.output.resolve()),
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
