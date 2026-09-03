#!/usr/bin/env python3
"""Export representative paper-LND SurgicalSAM2 diagnostic overlays.

The current two-object sequence stores the black shaft and the silver distal
tool separately.  This exporter keeps those two masks visually distinct so
segmentation failures can be separated from CAD/kinematic fitting errors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from build_super_raw_psm_kinematics import load_stereo_calibration
from track_super_p420006_stereo_surgicalsam2 import strict_pair_stream


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data/super/psm_raw_kinematics_v1（纯机器人学版本）"
VISUAL_ROOT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_paper_lnd_stereo_multianchor_v2"
)
SAM2_ROOT = (
    VISUAL_ROOT
    / "surgicalsam2_multianchor_parts_v4（paper_LND+视觉矫正）"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export raw/SAM2/overlay panels at representative pairs."
    )
    parser.add_argument(
        "--bag",
        type=Path,
        default=REPO_ROOT / "data/grasp5/grasp5.bag",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO_ROOT / "data/camera_calibration.yaml",
    )
    parser.add_argument(
        "--kinematics",
        type=Path,
        default=RAW_ROOT / "kinematics.npz",
    )
    parser.add_argument(
        "--masks",
        type=Path,
        default=SAM2_ROOT / "stereo_multianchor_part_masks.npz",
    )
    parser.add_argument(
        "--slots",
        default="0,191,397,540,747,1271,1530,1600",
    )
    parser.add_argument(
        "--corrections",
        type=Path,
        default=(
            SAM2_ROOT
            / "online_stereo_cma_closedjaw_v1/online_stereo_corrections.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SAM2_ROOT / "previews/curated_error_diagnostics_v1",
    )
    parser.add_argument(
        "--tip-overlay",
        choices=("none", "auto", "manual", "both"),
        default="auto",
        help=(
            "auto draws finite non-anchor ContourTipNet detections; manual "
            "draws only the eight human anchor tip annotations"
        ),
    )
    return parser.parse_args()


def unpack_mask(packed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.unpackbits(
        packed,
        bitorder="little",
        count=shape[0] * shape[1],
    ).reshape(shape).astype(bool)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max() + 1),
        int(ys.max() + 1),
    )


def resize_mask(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    return cv2.resize(
        mask.astype(np.uint8),
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def overlay_parts(
    image: np.ndarray,
    shaft: np.ndarray,
    distal: np.ndarray,
) -> np.ndarray:
    full_shaft = resize_mask(shaft, image)
    full_distal = resize_mask(distal, image)
    shaft_only = full_shaft & ~full_distal
    distal_only = full_distal & ~full_shaft
    overlap = full_shaft & full_distal
    active = full_shaft | full_distal
    result = image.copy()
    tint = image.copy()
    tint[shaft_only] = (255, 120, 20)
    tint[distal_only] = (30, 190, 255)
    tint[overlap] = (220, 40, 220)
    result[active] = cv2.addWeighted(
        image[active],
        0.45,
        tint[active],
        0.55,
        0.0,
    )
    for mask, color in (
        (full_shaft, (255, 120, 20)),
        (full_distal, (30, 190, 255)),
    ):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(result, contours, -1, color, 2, cv2.LINE_AA)
    return result


def draw_tip_points(
    image: np.ndarray,
    tips_xy_at_mask_resolution: np.ndarray,
    mask_shape: tuple[int, int],
    *,
    labels: tuple[str, str],
    colors: tuple[tuple[int, int, int], tuple[int, int, int]],
) -> np.ndarray:
    result = image.copy()
    scale_x = image.shape[1] / mask_shape[1]
    scale_y = image.shape[0] / mask_shape[0]
    points = [
        (
            int(round(float(xy[0]) * scale_x)),
            int(round(float(xy[1]) * scale_y)),
        )
        for xy in tips_xy_at_mask_resolution
    ]
    if len(points) == 2:
        cv2.line(result, points[0], points[1], (255, 255, 255), 2, cv2.LINE_AA)
    for label, point, color in zip(
        labels,
        points,
        colors,
        strict=True,
    ):
        cv2.circle(result, point, 10, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(result, point, 6, color, -1, cv2.LINE_AA)
        cv2.putText(
            result,
            label,
            (point[0] + 12, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            result,
            label,
            (point[0] + 12, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            1,
            cv2.LINE_AA,
        )
    return result


def crop_around_mask(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    output_aspect: float = 16.0 / 9.0,
) -> np.ndarray:
    full_mask = resize_mask(mask, image)
    y_values, x_values = np.where(full_mask)
    if len(x_values) == 0:
        return image
    center_x = 0.5 * (float(x_values.min()) + float(x_values.max()))
    center_y = 0.5 * (float(y_values.min()) + float(y_values.max()))
    box_width = max(360.0, 1.45 * (float(x_values.max() - x_values.min()) + 1.0))
    box_height = max(220.0, 1.65 * (float(y_values.max() - y_values.min()) + 1.0))
    if box_width / box_height < output_aspect:
        box_width = box_height * output_aspect
    else:
        box_height = box_width / output_aspect
    box_width = min(box_width, float(image.shape[1]))
    box_height = min(box_height, float(image.shape[0]))
    x0 = int(round(center_x - 0.5 * box_width))
    y0 = int(round(center_y - 0.5 * box_height))
    x0 = min(max(x0, 0), image.shape[1] - int(round(box_width)))
    y0 = min(max(y0, 0), image.shape[0] - int(round(box_height)))
    x1 = x0 + int(round(box_width))
    y1 = y0 + int(round(box_height))
    return image[y0:y1, x0:x1]


def panel(
    image: np.ndarray,
    title: str,
    *,
    width: int = 480,
    height: int = 270,
) -> np.ndarray:
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    output = np.zeros((height + 38, width, 3), dtype=np.uint8)
    output[38:] = resized
    cv2.putText(
        output,
        title,
        (8, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def main() -> None:
    args = parse_args()
    selected = np.asarray(
        sorted({int(item) for item in args.slots.split(",")}),
        dtype=np.int64,
    )
    for path in (
        args.bag,
        args.calibration,
        args.kinematics,
        args.masks,
        args.corrections,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(args.kinematics, allow_pickle=False) as raw:
        all_left = raw["stereo_left_index"].astype(np.int64)
        all_right = raw["stereo_right_index"].astype(np.int64)
    if len(selected) == 0 or selected[0] < 0 or selected[-1] >= len(all_left):
        raise ValueError(
            f"Slots must lie in [0, {len(all_left) - 1}], got {selected}"
        )
    with np.load(args.masks, allow_pickle=False) as masks:
        schema = str(masks["schema"].item())
        expected_schema = (
            "super_paper_lnd_stereo_surgicalsam2_multianchor_parts_v4"
        )
        if schema != expected_schema:
            raise RuntimeError(
                f"Expected {expected_schema}, found {schema}"
            )
        mask_shape = tuple(masks["mask_shape"].astype(int).tolist())
        packed = {
            side: {
                part: masks[
                    f"{side}_{part}_masks_packbits"
                ][selected].copy()
                for part in ("shaft", "distal")
            }
            for side in ("left", "right")
        }
        partition_quality_valid = masks[
            "partition_quality_valid"
        ][selected].astype(bool)
        shaft_component_fraction = {
            side: masks[
                f"{side}_shaft_largest_component_fraction"
            ][selected].astype(np.float32)
            for side in ("left", "right")
        }
        anchor_slots = masks["anchor_slots"].astype(np.int64)
        anchor_tips = masks[
            "anchor_tip_points_xy_at_mask_resolution"
        ].astype(np.float32)
        manual_tip_lookup = {
            int(slot): {
                side: anchor_tips[anchor_index, side_index].copy()
                for side_index, side in enumerate(("left", "right"))
            }
            for anchor_index, slot in enumerate(anchor_slots)
        }

    with np.load(args.corrections, allow_pickle=False) as corrections:
        correction_slots = corrections["strict_pair_slot"].astype(np.int64)
        if not np.array_equal(
            correction_slots,
            np.arange(len(correction_slots), dtype=np.int64),
        ):
            raise RuntimeError("Correction slots are not contiguous")
        filtered_loss = corrections["filtered_loss"][selected].astype(
            np.float32
        )
        automatic_tips = corrections[
            "tipnet_detections_xy_at_mask_resolution"
        ][selected].astype(np.float32)

    calibration = load_stereo_calibration(args.calibration)
    anomalies: list[dict] = []
    stream = strict_pair_stream(
        bag_path=args.bag,
        calibration=calibration,
        left_indices=all_left[selected],
        right_indices=all_right[selected],
        decoding_anomalies=anomalies,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[np.ndarray] = []
    tip_closeup_rows: list[np.ndarray] = []
    metrics: list[dict] = []
    for local_index, (_local_slot, left, right) in enumerate(stream):
        source_slot = int(selected[local_index])
        row_panels: list[np.ndarray] = []
        tip_closeup_panels: list[np.ndarray] = []
        metric = {
            "strict_pair_slot": source_slot,
            "filtered_optimization_loss": float(filtered_loss[local_index]),
            "part_partition_valid": bool(
                partition_quality_valid[local_index]
            ),
            "views": {},
        }
        for side_index, (side, image) in enumerate(
            (("left", left), ("right", right))
        ):
            shaft = unpack_mask(
                packed[side]["shaft"][local_index],
                mask_shape,
            )
            distal = unpack_mask(
                packed[side]["distal"][local_index],
                mask_shape,
            )
            union = shaft | distal
            overlay = overlay_parts(image, shaft, distal)
            manual_tips = manual_tip_lookup.get(source_slot, {}).get(side)
            auto_tips = automatic_tips[local_index, side_index]
            auto_tips_valid = bool(np.isfinite(auto_tips).all())
            tip_kind = "none"
            displayed_tips = None
            if (
                args.tip_overlay in ("manual", "both")
                and manual_tips is not None
            ):
                overlay = draw_tip_points(
                    overlay,
                    manual_tips,
                    mask_shape,
                    labels=("M1", "M2"),
                    colors=((40, 255, 40), (40, 40, 255)),
                )
                tip_kind = "manual"
                displayed_tips = manual_tips
            if (
                args.tip_overlay in ("auto", "both")
                and source_slot not in manual_tip_lookup
                and auto_tips_valid
            ):
                overlay = draw_tip_points(
                    overlay,
                    auto_tips,
                    mask_shape,
                    labels=("A1", "A2"),
                    colors=((255, 255, 0), (255, 40, 255)),
                )
                tip_kind = "automatic"
                displayed_tips = auto_tips
            bbox = mask_bbox(union)
            area = int(union.sum())
            metric["views"][side] = {
                "union_area_px_at_tracking_resolution": area,
                "shaft_area_px_at_tracking_resolution": int(shaft.sum()),
                "distal_area_px_at_tracking_resolution": int(distal.sum()),
                "shaft_distal_overlap_px_at_tracking_resolution": int(
                    (shaft & distal).sum()
                ),
                "union_bbox_xyxy_at_tracking_resolution": list(bbox),
                "bbox_width_px": bbox[2] - bbox[0],
                "bbox_height_px": bbox[3] - bbox[1],
                "shaft_largest_component_fraction": float(
                    shaft_component_fraction[side][local_index]
                ),
                "manual_jaw_tip_points_xy_at_mask_resolution": (
                    manual_tips.tolist()
                    if manual_tips is not None
                    else None
                ),
                "automatic_tipnet_points_xy_at_mask_resolution": (
                    auto_tips.tolist() if auto_tips_valid else None
                ),
                "displayed_tip_kind": tip_kind,
            }
            raw_path = (
                args.output_dir
                / f"slot_{source_slot:04d}_{side}_raw_rectified.png"
            )
            overlay_path = (
                args.output_dir
                / f"slot_{source_slot:04d}_{side}_sam2_overlay.png"
            )
            if not (
                cv2.imwrite(str(raw_path), image)
                and cv2.imwrite(str(overlay_path), overlay)
            ):
                raise RuntimeError(f"Failed to write slot {source_slot}")
            for part, mask in (
                ("shaft", shaft),
                ("distal", distal),
                ("union", union),
            ):
                mask_path = (
                    args.output_dir
                    / f"slot_{source_slot:04d}_{side}_{part}_mask.png"
                )
                if not cv2.imwrite(
                    str(mask_path),
                    mask.astype(np.uint8) * 255,
                ):
                    raise RuntimeError(f"Failed to write {mask_path}")
            row_panels.extend(
                [
                    panel(image, f"{source_slot:04d} {side} RAW"),
                    panel(
                        overlay,
                        (
                            f"{source_slot:04d} {side} AUTO MASK "
                            f"loss={filtered_loss[local_index]:.3f} "
                            f"{'AUTO-TIPS ' if tip_kind == 'automatic' else ''}"
                            f"{'MANUAL-TIPS ' if tip_kind == 'manual' else ''}"
                            f"{'NO-AUTO-TIPS ' if args.tip_overlay == 'auto' and tip_kind == 'none' else ''}"
                            f"{'OK' if partition_quality_valid[local_index] else 'WARN'}"
                        ),
                    ),
                ]
            )
            if args.tip_overlay != "none":
                if tip_kind == "automatic":
                    tip_status = "AUTOMATIC JAW TIPS"
                elif tip_kind == "manual":
                    tip_status = "MANUAL JAW TIPS"
                else:
                    tip_status = "NO VALID AUTOMATIC TIPS"
                tip_closeup_panels.append(
                    panel(
                        crop_around_mask(overlay, distal),
                        (
                            f"{source_slot:04d} {side} "
                            f"AUTO MASK + {tip_status}"
                        ),
                        width=640,
                        height=360,
                    )
                )
        rows.append(np.concatenate(row_panels, axis=1))
        if len(tip_closeup_panels) == 2:
            tip_closeup_rows.append(
                np.concatenate(tip_closeup_panels, axis=1)
            )
        metrics.append(metric)

    if len(rows) != len(selected):
        raise RuntimeError(f"Extracted {len(rows)}/{len(selected)} selected pairs")
    sheet_path = args.output_dir / "representative_sam2_contact_sheet.png"
    if not cv2.imwrite(str(sheet_path), np.concatenate(rows, axis=0)):
        raise RuntimeError(f"Failed to write {sheet_path}")
    tip_closeup_sheet_path = args.output_dir / "tip_detection_closeups.png"
    if tip_closeup_rows and not cv2.imwrite(
        str(tip_closeup_sheet_path),
        np.concatenate(tip_closeup_rows, axis=0),
    ):
        raise RuntimeError(f"Failed to write {tip_closeup_sheet_path}")
    report = {
        "schema": "super_paper_lnd_sam2_part_diagnostics_v2",
        "selected_strict_pair_slots": selected.tolist(),
        "selection_policy": (
            "baseline, spaced optimization-loss peaks, and late shaft "
            "partition warnings"
        ),
        "mask_colors_bgr": {
            "shaft": [255, 120, 20],
            "distal": [30, 190, 255],
            "shaft_distal_overlap": [220, 40, 220],
        },
        "manual_jaw_tip_colors_bgr": {
            "T1": [40, 255, 40],
            "T2": [40, 40, 255],
        },
        "automatic_tipnet_colors_bgr": {
            "A1": [255, 255, 0],
            "A2": [255, 40, 255],
        },
        "tip_overlay_mode": args.tip_overlay,
        "metrics": metrics,
        "raw_message_decoding_anomalies": anomalies,
        "contact_sheet": str(sheet_path),
        "tip_detection_closeup_sheet": (
            str(tip_closeup_sheet_path) if tip_closeup_rows else None
        ),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(sheet_path)
    if tip_closeup_rows:
        print(tip_closeup_sheet_path)
    print(report_path)


if __name__ == "__main__":
    main()
