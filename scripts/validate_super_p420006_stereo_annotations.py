#!/usr/bin/env python3
"""Validate manual P420006 stereo masks and jaw-tip labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from annotate_super_p420006_stereo_keyframes import (
    CLASS_COLORS,
    CLASS_NAMES,
    DEFAULT_ROOT,
    TIP_NAMES,
    annotation_path,
    rasterize_view,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check completeness and stereo consistency of annotations."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--maximum-tip-mask-distance-px", type=float, default=35.0)
    parser.add_argument("--maximum-tip-epipolar-error-px", type=float, default=50.0)
    return parser.parse_args()


def add_error(
    errors: list[dict[str, Any]],
    keyframe: int,
    side: str,
    message: str,
) -> None:
    errors.append(
        {"keyframe_index": keyframe, "side": side, "message": message}
    )


def tip_mask_distance(
    labels: np.ndarray,
    class_id: int,
    point_xy: list[float],
) -> float:
    target = labels == class_id
    if not np.any(target):
        return float("inf")
    distance = cv2.distanceTransform(
        (~target).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    x = int(np.clip(round(point_xy[0]), 0, labels.shape[1] - 1))
    y = int(np.clip(round(point_xy[1]), 0, labels.shape[0] - 1))
    return float(distance[y, x])


def overlay_annotation(
    image: np.ndarray,
    labels: np.ndarray,
    view: dict[str, Any],
) -> np.ndarray:
    result = image.copy()
    tint = np.zeros_like(result)
    active = labels > 0
    for class_id, class_name in CLASS_NAMES.items():
        tint[labels == class_id] = CLASS_COLORS[class_name]
    result[active] = cv2.addWeighted(
        result[active],
        0.55,
        tint[active],
        0.45,
        0.0,
    )
    for tip_name, tip in view["tips"].items():
        if tip["visible"] and tip["point_xy"] is not None:
            point = tuple(np.rint(tip["point_xy"]).astype(int))
            cv2.drawMarker(
                result,
                point,
                CLASS_COLORS[tip_name],
                cv2.MARKER_CROSS,
                30,
                4,
                cv2.LINE_AA,
            )
    return result


def make_preview(
    root: Path,
    rows: list[dict[str, Any]],
    annotations: dict[int, dict[str, Any]],
) -> Path | None:
    if not annotations:
        return None
    panels: list[np.ndarray] = []
    panel_width = 800
    for row in rows:
        keyframe = int(row["keyframe_index"])
        if keyframe not in annotations:
            continue
        annotation = annotations[keyframe]
        width, height = annotation["image_size_wh"]
        side_panels: list[np.ndarray] = []
        for side in ("left", "right"):
            image = cv2.imread(
                str(root / row[f"{side}_image"]),
                cv2.IMREAD_COLOR,
            )
            labels = rasterize_view(
                annotation["views"][side],
                int(width),
                int(height),
            )
            overlay = overlay_annotation(
                image,
                labels,
                annotation["views"][side],
            )
            scale = panel_width / (2.0 * overlay.shape[1])
            side_panels.append(
                cv2.resize(
                    overlay,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
            )
        panel = np.concatenate(side_panels, axis=1)
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(
            panel,
            f"KF {keyframe:02d}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)
    if not panels:
        return None
    output = root / "previews/manual_annotation_contact_sheet.png"
    sheet = np.concatenate(panels, axis=0)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"Failed to write {output}")
    return output


def main() -> None:
    args = parse_args()
    manifest_path = args.root / "pair_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["keyframes"]
    confidence_path = args.root / "annotation_confidence.json"
    if confidence_path.is_file():
        confidence = json.loads(confidence_path.read_text(encoding="utf-8"))
        if (
            confidence.get("schema")
            != "super_p420006_stereo_annotation_confidence_v1"
        ):
            raise RuntimeError("Unexpected annotation confidence schema")
    else:
        confidence = {
            "default": {"body": 1.0, "jaw_masks": 1.0, "jaw_tips": 1.0},
            "keyframes": {},
        }
    for key, weights in {
        "default": confidence["default"],
        **confidence.get("keyframes", {}),
    }.items():
        for name in ("body", "jaw_masks", "jaw_tips"):
            value = float(weights[name])
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Invalid confidence {name}={value} for {key}"
                )
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    annotations: dict[int, dict[str, Any]] = {}
    measurements: list[dict[str, Any]] = []

    for row in rows:
        keyframe = int(row["keyframe_index"])
        path = annotation_path(args.root, keyframe)
        if not path.is_file():
            add_error(errors, keyframe, "stereo", "annotation JSON missing")
            continue
        annotation = json.loads(path.read_text(encoding="utf-8"))
        annotations[keyframe] = annotation
        if annotation.get("schema") != "super_p420006_stereo_manual_annotation_v1":
            add_error(errors, keyframe, "stereo", "unexpected annotation schema")
            continue
        width, height = annotation.get("image_size_wh", [0, 0])
        if [width, height] != manifest["image_size_wh"]:
            add_error(errors, keyframe, "stereo", "image size mismatch")
            continue

        labels_by_side: dict[str, np.ndarray] = {}
        areas_by_side: dict[str, dict[str, int]] = {}
        for side in ("left", "right"):
            source = args.root / row[f"{side}_image"]
            expected_hash = row[f"{side}_image_sha256"]
            if sha256(source) != expected_hash:
                add_error(errors, keyframe, side, "source image hash mismatch")
            view = annotation["views"][side]
            labels = rasterize_view(view, width, height)
            labels_by_side[side] = labels
            saved_labels_path = (
                args.root
                / "annotations"
                / f"keyframe_{keyframe:02d}_{side}_labels.png"
            )
            saved_labels = cv2.imread(
                str(saved_labels_path),
                cv2.IMREAD_UNCHANGED,
            )
            if saved_labels is None:
                add_error(errors, keyframe, side, "raster label PNG missing")
            elif not np.array_equal(labels, saved_labels):
                add_error(
                    errors,
                    keyframe,
                    side,
                    "raster label PNG differs from annotation JSON",
                )

            areas: dict[str, int] = {}
            for class_id, class_name in CLASS_NAMES.items():
                area = int(np.count_nonzero(labels == class_id))
                areas[class_name] = area
                if area == 0:
                    add_error(
                        errors,
                        keyframe,
                        side,
                        f"{class_name} polygon has zero area",
                    )
            areas_by_side[side] = areas

            for class_id, tip_name in ((2, "jaw_1"), (3, "jaw_2")):
                tip = view["tips"][tip_name]
                if tip["visible"] is None:
                    add_error(
                        errors,
                        keyframe,
                        side,
                        f"{tip_name} tip has not been labelled",
                    )
                    continue
                if not tip["visible"]:
                    warnings.append(
                        {
                            "keyframe_index": keyframe,
                            "side": side,
                            "message": f"{tip_name} tip marked invisible",
                        }
                    )
                    continue
                point = tip["point_xy"]
                if point is None:
                    add_error(
                        errors,
                        keyframe,
                        side,
                        f"{tip_name} visible but point is missing",
                    )
                    continue
                if not (0 <= point[0] < width and 0 <= point[1] < height):
                    add_error(
                        errors,
                        keyframe,
                        side,
                        f"{tip_name} point lies outside the image",
                    )
                    continue
                distance = tip_mask_distance(labels, class_id, point)
                if distance > args.maximum_tip_mask_distance_px:
                    add_error(
                        errors,
                        keyframe,
                        side,
                        f"{tip_name} tip is {distance:.1f}px from its jaw mask",
                    )

        for class_name in CLASS_NAMES.values():
            left_area = areas_by_side["left"][class_name]
            right_area = areas_by_side["right"][class_name]
            if min(left_area, right_area) > 0:
                ratio = left_area / right_area
                if not 0.45 <= ratio <= 2.20:
                    warnings.append(
                        {
                            "keyframe_index": keyframe,
                            "side": "stereo",
                            "message": (
                                f"{class_name} left/right area ratio "
                                f"is {ratio:.2f}"
                            ),
                        }
                    )

        epipolar_errors: dict[str, float | None] = {}
        for tip_name in TIP_NAMES.values():
            left_tip = annotation["views"]["left"]["tips"][tip_name]
            right_tip = annotation["views"]["right"]["tips"][tip_name]
            if left_tip["visible"] and right_tip["visible"]:
                error = abs(
                    float(left_tip["point_xy"][1])
                    - float(right_tip["point_xy"][1])
                )
                epipolar_errors[tip_name] = error
                if error > args.maximum_tip_epipolar_error_px:
                    add_error(
                        errors,
                        keyframe,
                        "stereo",
                        f"{tip_name} epipolar y error is {error:.1f}px",
                    )
            else:
                epipolar_errors[tip_name] = None
        measurements.append(
            {
                "keyframe_index": keyframe,
                "areas_px": areas_by_side,
                "tip_epipolar_error_px": epipolar_errors,
            }
        )

    preview_path = make_preview(args.root, rows, annotations)
    passed = len(errors) == 0 and len(annotations) == len(rows)
    report = {
        "schema": "super_p420006_stereo_annotation_validation_v1",
        "passed": passed,
        "expected_keyframes": len(rows),
        "annotated_keyframes": len(annotations),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "annotation_confidence": {
            "path": str(confidence_path) if confidence_path.is_file() else None,
            "default": confidence["default"],
            "overrides": confidence.get("keyframes", {}),
        },
        "thresholds": {
            "maximum_tip_mask_distance_px": args.maximum_tip_mask_distance_px,
            "maximum_tip_epipolar_error_px": (
                args.maximum_tip_epipolar_error_px
            ),
        },
        "errors": errors,
        "warnings": warnings,
        "measurements": measurements,
        "preview": str(preview_path) if preview_path else None,
    }
    output_path = args.root / "previews/annotation_validation.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
