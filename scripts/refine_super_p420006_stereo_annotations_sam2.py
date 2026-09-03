#!/usr/bin/env python3
"""Refine P420006 stereo part masks with the SurgicalSAM2 checkpoint.

The existing manual polygons are treated as prompts, not copied as the output
boundary.  Left and right rectified images are inferred independently with the
same checkpoint.  Manual jaw-tip points are preserved with explicit
provenance, because SAM2 predicts regions rather than anatomical keypoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL.Image import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_p420006_stereo_v1"
)
DEFAULT_PAPER_REPO = Path("/Media_HDD/jwshan/wad/online_dvrk_tracking")
DEFAULT_CHECKPOINT = (
    DEFAULT_PAPER_REPO
    / "SurgicalSAM2/checkpoints/sam2.1_hiera_s_endo18.pth"
)
DEFAULT_OUTPUT_NAME = "annotations_sam2_manual_prompted_v1"
CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_s.yaml"
CLASS_NAMES = {1: "body", 2: "jaw_1", 3: "jaw_2"}
CLASS_COLORS = {
    "body": (60, 220, 60),
    "jaw_1": (40, 90, 255),
    "jaw_2": (255, 170, 30),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the endoscopic SurgicalSAM2 checkpoint on the ten rectified "
            "stereo pairs, using the manual part polygons only as prompts."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--manual-annotations-dir",
        type=Path,
        default=None,
        help="Defaults to ROOT/annotations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Defaults to ROOT/{DEFAULT_OUTPUT_NAME}.",
    )
    parser.add_argument("--paper-repo", type=Path, default=DEFAULT_PAPER_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=CONFIG_NAME)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--keyframes",
        default="",
        help="Comma-separated keyframe indices; empty means all ten.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_sam2(
    paper_repo: Path,
    checkpoint: Path,
    config: str,
    device: str,
) -> Any:
    sys.path[:0] = [str(paper_repo / "SurgicalSAM2"), str(paper_repo)]
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    class Torch26SAM2ImagePredictor(SAM2ImagePredictor):
        """Fix the 512px SurgicalSAM2 single-image path for PyTorch 2.6."""

        @torch.no_grad()
        def set_image(self, image: np.ndarray | Image) -> None:
            self.reset_predictor()
            if isinstance(image, np.ndarray):
                self._orig_hw = [image.shape[:2]]
            elif isinstance(image, Image):
                width, height = image.size
                self._orig_hw = [(height, width)]
            else:
                raise NotImplementedError("Unsupported image type")

            input_image = self._transforms(image)[None, ...].to(self.device)
            backbone_out = self.model.forward_image(input_image)
            _, vision_feats, _, _ = self.model._prepare_backbone_features(
                backbone_out
            )
            if self.model.directly_add_no_mem_embed:
                vision_feats[-1] = (
                    vision_feats[-1] + self.model.no_mem_embed
                )
            feature_sizes: list[tuple[int, int]] = []
            for feature in vision_feats:
                side = int(round(math.sqrt(feature.shape[0])))
                if side * side != feature.shape[0]:
                    raise ValueError(
                        f"Non-square SAM feature map: {feature.shape}"
                    )
                feature_sizes.append((side, side))
            feats = [
                feature.permute(1, 2, 0).reshape(
                    1, -1, *feature_size
                )
                for feature, feature_size in zip(
                    vision_feats[::-1],
                    feature_sizes[::-1],
                    strict=True,
                )
            ][::-1]
            self._features = {
                "image_embed": feats[-1],
                "high_res_feats": feats[:-1],
            }
            self._is_image_set = True

    model = build_sam2(
        config,
        str(checkpoint),
        device=device,
        mode="eval",
    )
    return Torch26SAM2ImagePredictor(model)


def rasterize_view(
    view: dict[str, Any],
    width: int,
    height: int,
) -> np.ndarray:
    labels = np.zeros((height, width), dtype=np.uint8)
    for class_id, class_name in CLASS_NAMES.items():
        for polygon in view["polygons"][class_name]:
            points = np.rint(np.asarray(polygon, dtype=np.float64)).astype(
                np.int32
            )
            if len(points) >= 3:
                cv2.fillPoly(labels, [points], class_id)
    return labels


def spread_interior_points(mask: np.ndarray, count: int) -> np.ndarray:
    """Choose well-separated high-clearance positive points."""

    if not np.any(mask):
        raise ValueError("Cannot prompt an empty manual mask")
    distance = cv2.distanceTransform(
        mask.astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    work = distance.copy()
    radius = max(5, int(round(math.sqrt(float(mask.sum())) / (count + 2))))
    points: list[list[float]] = []
    for _ in range(count):
        _, maximum, _, location = cv2.minMaxLoc(work)
        if maximum <= 0:
            break
        x, y = location
        points.append([float(x), float(y)])
        cv2.circle(work, (x, y), radius, 0.0, -1)
    if not points:
        y, x = np.argwhere(mask)[len(np.argwhere(mask)) // 2]
        points.append([float(x), float(y)])
    return np.asarray(points, dtype=np.float32)


def padded_box(mask: np.ndarray, class_name: str) -> np.ndarray:
    y_values, x_values = np.where(mask)
    x0, x1 = int(x_values.min()), int(x_values.max())
    y0, y1 = int(y_values.min()), int(y_values.max())
    extent = max(x1 - x0 + 1, y1 - y0 + 1)
    fraction = 0.06 if class_name == "body" else 0.14
    padding = max(14, int(round(fraction * extent)))
    height, width = mask.shape
    return np.asarray(
        [
            max(0, x0 - padding),
            max(0, y0 - padding),
            min(width - 1, x1 + padding),
            min(height - 1, y1 + padding),
        ],
        dtype=np.float32,
    )


def background_points(
    all_parts: np.ndarray,
    box: np.ndarray,
    count: int,
) -> np.ndarray:
    """Pick high-clearance background negatives inside the prompt box."""

    x0, y0, x1, y1 = np.rint(box).astype(int)
    allowed = np.zeros_like(all_parts, dtype=np.uint8)
    allowed[y0 : y1 + 1, x0 : x1 + 1] = 1
    allowed[all_parts] = 0
    distance = cv2.distanceTransform(
        allowed,
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    points: list[list[float]] = []
    radius = max(10, int(round(max(x1 - x0, y1 - y0) / 8)))
    for _ in range(count):
        _, maximum, _, location = cv2.minMaxLoc(distance)
        if maximum <= 0:
            break
        x, y = location
        points.append([float(x), float(y)])
        cv2.circle(distance, (x, y), radius, 0.0, -1)
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def make_prompts(
    manual_labels: np.ndarray,
    class_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    class_name = CLASS_NAMES[class_id]
    target = manual_labels == class_id
    all_parts = manual_labels > 0
    positive_count = 7 if class_name == "body" else 4
    positive = spread_interior_points(target, positive_count)
    box = padded_box(target, class_name)

    negative_groups: list[np.ndarray] = []
    for other_id in CLASS_NAMES:
        if other_id == class_id:
            continue
        other = manual_labels == other_id
        if np.any(other):
            negative_groups.append(
                spread_interior_points(other, 2 if class_id == 1 else 1)
            )
    background = background_points(
        all_parts,
        box,
        4 if class_name == "body" else 3,
    )
    if len(background):
        negative_groups.append(background)
    negative = (
        np.concatenate(negative_groups, axis=0)
        if negative_groups
        else np.empty((0, 2), dtype=np.float32)
    )
    points = np.concatenate([positive, negative], axis=0).astype(np.float32)
    labels = np.concatenate(
        [
            np.ones(len(positive), dtype=np.int64),
            np.zeros(len(negative), dtype=np.int64),
        ]
    )
    return points, labels, box


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(a & b) / union)


def keep_prompted_components(
    mask: np.ndarray,
    positive_points: np.ndarray,
) -> np.ndarray:
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    if count <= 2:
        return mask
    keep: set[int] = set()
    height, width = mask.shape
    for x_value, y_value in positive_points:
        x = int(np.clip(round(float(x_value)), 0, width - 1))
        y = int(np.clip(round(float(y_value)), 0, height - 1))
        label = int(labels[y, x])
        if label:
            keep.add(label)
    if not keep:
        keep.add(
            int(
                1
                + np.argmax(
                    statistics[1:, cv2.CC_STAT_AREA]
                )
            )
        )
    return np.isin(labels, list(keep))


def predict_part(
    predictor: Any,
    manual_labels: np.ndarray,
    class_id: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    manual_mask = manual_labels == class_id
    points, point_labels, box = make_prompts(manual_labels, class_id)
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=point_labels,
        box=box,
        multimask_output=True,
    )
    candidates = np.asarray(masks) > 0
    positive_xy = np.rint(points[point_labels == 1]).astype(int)
    negative_xy = np.rint(points[point_labels == 0]).astype(int)
    manual_area = max(1, int(manual_mask.sum()))
    candidate_metrics: list[dict[str, Any]] = []
    ranking: list[tuple[float, float, int]] = []
    for index, candidate in enumerate(candidates):
        positive_misses = sum(
            not candidate[y, x] for x, y in positive_xy
        )
        negative_hits = sum(candidate[y, x] for x, y in negative_xy)
        violations = int(positive_misses + negative_hits)
        area = int(candidate.sum())
        area_ratio = area / manual_area
        manual_iou = iou(candidate, manual_mask)
        score = float(scores[index])
        selection_cost = (
            0.35 * (1.0 - manual_iou)
            + 0.20 * abs(math.log(max(area_ratio, 1.0e-6)))
            - score
        )
        ranking.append((float(violations), selection_cost, index))
        candidate_metrics.append(
            {
                "candidate_index": index,
                "predicted_iou_score": score,
                "positive_prompt_misses": int(positive_misses),
                "negative_prompt_hits": int(negative_hits),
                "area_px": area,
                "area_ratio_to_manual": float(area_ratio),
                "iou_with_manual_prompt_mask": manual_iou,
                "selection_cost_after_prompt_violations": selection_cost,
            }
        )
    selected_index = min(ranking)[-1]
    selected = keep_prompted_components(
        candidates[selected_index],
        points[point_labels == 1],
    )
    metadata = {
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "box_xyxy": box.tolist(),
        "positive_points_xy": points[point_labels == 1].tolist(),
        "negative_points_xy": points[point_labels == 0].tolist(),
        "selected_candidate_index": int(selected_index),
        "selected_area_after_component_filter_px": int(selected.sum()),
        "selected_iou_with_manual_prompt_mask": iou(
            selected,
            manual_mask,
        ),
        "candidates": candidate_metrics,
    }
    return selected, metadata


def distance_to_mask(mask: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform(
        (~mask).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )


def resolve_part_overlaps(
    raw_masks: dict[str, np.ndarray],
    manual_labels: np.ndarray,
) -> np.ndarray:
    """Make disjoint semantic labels while retaining SAM2 boundaries."""

    jaw_1 = raw_masks["jaw_1"].copy()
    jaw_2 = raw_masks["jaw_2"].copy()
    overlap = jaw_1 & jaw_2
    if np.any(overlap):
        distance_1 = distance_to_mask(manual_labels == 2)
        distance_2 = distance_to_mask(manual_labels == 3)
        give_to_1 = overlap & (distance_1 <= distance_2)
        give_to_2 = overlap & ~give_to_1
        jaw_1[give_to_2] = False
        jaw_2[give_to_1] = False
    body = raw_masks["body"] & ~(jaw_1 | jaw_2)
    labels = np.zeros(manual_labels.shape, dtype=np.uint8)
    labels[body] = 1
    labels[jaw_1] = 2
    labels[jaw_2] = 3
    return labels


def mask_to_polygons(mask: np.ndarray) -> list[list[list[float]]]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    polygons: list[list[list[float]]] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < 2.0:
            continue
        perimeter = cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(
            contour,
            max(0.75, 0.001 * perimeter),
            True,
        )
        if len(simplified) >= 3:
            polygons.append(
                simplified.reshape(-1, 2).astype(float).tolist()
            )
    return polygons


def nearest_mask_boundary(
    mask: np.ndarray,
    point_xy: list[float] | None,
) -> list[float] | None:
    if point_xy is None or not np.any(mask):
        return None
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    points = np.concatenate([contour.reshape(-1, 2) for contour in contours])
    point = np.asarray(point_xy, dtype=np.float32)
    nearest = points[np.argmin(np.linalg.norm(points - point, axis=1))]
    return nearest.astype(float).tolist()


def tint_labels(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
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
    for class_id, class_name in CLASS_NAMES.items():
        contours, _ = cv2.findContours(
            (labels == class_id).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            result,
            contours,
            -1,
            CLASS_COLORS[class_name],
            2,
        )
    return result


def make_comparison_panel(
    image: np.ndarray,
    manual_labels: np.ndarray,
    sam2_labels: np.ndarray,
    title: str,
) -> np.ndarray:
    manual = tint_labels(image, manual_labels)
    sam2 = tint_labels(image, sam2_labels)
    scale = 480.0 / image.shape[1]
    manual = cv2.resize(
        manual,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )
    sam2 = cv2.resize(
        sam2,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )
    panel = np.concatenate([manual, sam2], axis=1)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        panel,
        f"{title}: manual prompts | SurgicalSAM2",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def main() -> None:
    args = parse_args()
    manual_dir = (
        args.manual_annotations_dir
        if args.manual_annotations_dir is not None
        else args.root / "annotations"
    )
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.root / DEFAULT_OUTPUT_NAME
    )
    for path in (
        args.root / "pair_manifest.json",
        args.checkpoint,
        manual_dir,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run in the online_dvrk GPU environment."
        )
    manifest = json.loads(
        (args.root / "pair_manifest.json").read_text(encoding="utf-8")
    )
    selected_keyframes = (
        {int(value) for value in args.keyframes.split(",") if value.strip()}
        if args.keyframes
        else None
    )
    rows = [
        row
        for row in manifest["keyframes"]
        if selected_keyframes is None
        or int(row["keyframe_index"]) in selected_keyframes
    ]
    if not rows:
        raise ValueError("No keyframes selected")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} is not empty; pass --overwrite to regenerate it"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw_masks").mkdir(exist_ok=True)
    (output_dir / "previews").mkdir(exist_ok=True)

    predictor = load_sam2(
        args.paper_repo,
        args.checkpoint,
        args.config,
        args.device,
    )
    checkpoint_hash = sha256(args.checkpoint)
    report_rows: list[dict[str, Any]] = []
    comparison_rows: list[np.ndarray] = []

    for row_number, row in enumerate(rows, 1):
        keyframe = int(row["keyframe_index"])
        manual_path = manual_dir / f"keyframe_{keyframe:02d}.json"
        manual = json.loads(manual_path.read_text(encoding="utf-8"))
        width, height = manual["image_size_wh"]
        refined = {
            **manual,
            "schema": "super_p420006_stereo_sam2_manual_prompted_v1",
            "segmentation_provenance": {
                "method": (
                    "SurgicalSAM2 independent single-image inference; manual "
                    "polygons used only for semantic boxes and point prompts"
                ),
                "manual_annotation": str(manual_path),
                "manual_annotation_sha256": sha256(manual_path),
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "config": args.config,
                "device": args.device,
                "torch_version": torch.__version__,
                "left_right_policy": (
                    "independent inference with the same model and rules"
                ),
                "tip_policy": (
                    "manual tip retained; nearest SAM2 boundary is diagnostic "
                    "only because SAM2 is not a keypoint detector"
                ),
            },
            "views": {},
        }
        keyframe_metrics: dict[str, Any] = {
            "keyframe_index": keyframe,
            "views": {},
        }
        raw_to_save: dict[str, np.ndarray] = {}
        side_panels: list[np.ndarray] = []
        for side in ("left", "right"):
            image_path = args.root / row[f"{side}_image"]
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"Failed to load {image_path}")
            if image_bgr.shape[:2] != (height, width):
                raise RuntimeError(f"Image size mismatch for {image_path}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            manual_labels = rasterize_view(
                manual["views"][side],
                width,
                height,
            )
            predictor.set_image(image_rgb)
            raw_masks: dict[str, np.ndarray] = {}
            part_metrics: dict[str, Any] = {}
            for class_id, class_name in CLASS_NAMES.items():
                raw_masks[class_name], part_metrics[class_name] = predict_part(
                    predictor,
                    manual_labels,
                    class_id,
                )
                raw_to_save[f"{side}_{class_name}"] = raw_masks[class_name]
            refined_labels = resolve_part_overlaps(raw_masks, manual_labels)
            polygons = {
                class_name: mask_to_polygons(
                    refined_labels == class_id
                )
                for class_id, class_name in CLASS_NAMES.items()
            }
            tips: dict[str, Any] = {}
            for class_id, tip_name in ((2, "jaw_1"), (3, "jaw_2")):
                manual_tip = manual["views"][side]["tips"][tip_name]
                tips[tip_name] = {
                    **manual_tip,
                    "source": "manual_estimate_unmodified",
                    "sam2_nearest_boundary_xy": nearest_mask_boundary(
                        refined_labels == class_id,
                        manual_tip["point_xy"],
                    ),
                }
            refined["views"][side] = {
                "polygons": polygons,
                "tips": tips,
            }
            label_path = (
                output_dir
                / f"keyframe_{keyframe:02d}_{side}_labels.png"
            )
            if not cv2.imwrite(
                str(label_path),
                refined_labels,
                [cv2.IMWRITE_PNG_COMPRESSION, 3],
            ):
                raise RuntimeError(f"Failed to write {label_path}")
            keyframe_metrics["views"][side] = {
                "parts": part_metrics,
                "resolved_areas_px": {
                    class_name: int(
                        np.count_nonzero(refined_labels == class_id)
                    )
                    for class_id, class_name in CLASS_NAMES.items()
                },
                "manual_areas_px": {
                    class_name: int(
                        np.count_nonzero(manual_labels == class_id)
                    )
                    for class_id, class_name in CLASS_NAMES.items()
                },
            }
            side_panels.append(
                make_comparison_panel(
                    image_bgr,
                    manual_labels,
                    refined_labels,
                    f"KF {keyframe:02d} {side}",
                )
            )
        annotation_path = output_dir / f"keyframe_{keyframe:02d}.json"
        annotation_path.write_text(
            json.dumps(refined, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(
            output_dir / "raw_masks" / f"keyframe_{keyframe:02d}.npz",
            **raw_to_save,
        )
        report_rows.append(keyframe_metrics)
        comparison_rows.append(np.concatenate(side_panels, axis=0))
        print(
            f"[{row_number:02d}/{len(rows):02d}] "
            f"KF {keyframe:02d}: stereo SAM2 masks written",
            flush=True,
        )

    contact_sheet = np.concatenate(comparison_rows, axis=0)
    contact_sheet_path = (
        output_dir / "previews/manual_vs_surgicalsam2_contact_sheet.png"
    )
    if not cv2.imwrite(str(contact_sheet_path), contact_sheet):
        raise RuntimeError(f"Failed to write {contact_sheet_path}")
    report = {
        "schema": "super_p420006_stereo_sam2_refinement_report_v1",
        "complete_dataset": len(rows) == int(manifest["count"]),
        "processed_keyframes": [int(row["keyframe_index"]) for row in rows],
        "source_manual_annotations_dir": str(manual_dir),
        "output_dir": str(output_dir),
        "model": {
            "repository": str(args.paper_repo / "SurgicalSAM2"),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_size_bytes": args.checkpoint.stat().st_size,
            "config": args.config,
            "device": args.device,
            "torch_version": torch.__version__,
        },
        "policy": {
            "stereo": (
                "left and right images are inferred independently with the "
                "same checkpoint and prompt-generation algorithm"
            ),
            "manual_masks": (
                "used only to generate semantic boxes and positive/negative "
                "points and to rank the three model candidates"
            ),
            "output_masks": (
                "selected SurgicalSAM2 boundaries with jaw-overlap resolution; "
                "not the union or intersection of the manual masks"
            ),
            "occlusion_limit": (
                "SAM2 cannot observe fully tissue-occluded pixels; retained "
                "manual jaw tips remain explicitly marked manual estimates"
            ),
            "confidence": "no keyframe downweighting is applied here",
        },
        "contact_sheet": str(contact_sheet_path),
        "keyframes": report_rows,
    }
    report_path = output_dir / "sam2_refinement_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Report: {report_path}")
    print(f"Contact sheet: {contact_sheet_path}")


if __name__ == "__main__":
    main()
