#!/usr/bin/env python3
"""Build stage-B stereo tissue/tool masks and final proxy set cover.

This stage deliberately remains in image space.  It combines independent
left/right SAM2 tissue propagation with image-derived and corrected-CAD tool
masks, applies a conservative tool dilation and highlight rejection, and
selects stereo pairs by equal-weight left/right marginal coverage.

Depth validity and stereo confidence are stage-C products.  Consequently the
saved ``stage_b_visibility_proxy`` masks are not mislabeled as final
reconstruction masks; stage C must intersect them with per-view valid depth
and confidence before geometry fusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOT = REPO_ROOT / "data/super/grasp5_native"
TRACK_ROOT = REPO_ROOT / "data/super/psm_tracking"
MULTIVIEW_ROOT = NATIVE_ROOT / "tissue_multiview_v1"
SURGICAL_SAM_ROOT = Path(
    "/Media_HDD/jwshan/wad/online_dvrk_tracking/SurgicalSAM2"
)
sys.path.insert(0, str(SURGICAL_SAM_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sam2.build_sam import build_sam2  # noqa: E402

from build_super_tissue_right_seed import (  # noqa: E402
    CompatibleSAM2ImagePredictor,
    bounding_box,
    corrected_cad_mask,
    farthest_points,
    spread_points,
)
from select_super_tissue_multiview_frames import (  # noqa: E402
    array_sha256,
    verify_frozen_assets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build SUPER stage-B stereo candidate masks and set cover."
        )
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=MULTIVIEW_ROOT / "observations.json",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=NATIVE_ROOT / "calib_rectified.json",
    )
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=NATIVE_ROOT / "rgb",
    )
    parser.add_argument(
        "--left-tissue-packbits",
        type=Path,
        default=NATIVE_ROOT
        / "visual_force_masks_v1/tissue_masks_packbits.npy",
    )
    parser.add_argument(
        "--right-tissue-packbits",
        type=Path,
        default=MULTIVIEW_ROOT
        / "masks_right/propagated/tissue_masks_packbits.npy",
    )
    parser.add_argument(
        "--right-propagation-report",
        type=Path,
        default=MULTIVIEW_ROOT / "masks_right/propagated/report.json",
    )
    parser.add_argument(
        "--right-seed-report",
        type=Path,
        default=MULTIVIEW_ROOT / "masks_right/seed/report.json",
    )
    parser.add_argument(
        "--left-fixed-support",
        type=Path,
        default=NATIVE_ROOT / "masks/000000-tissue.png",
    )
    parser.add_argument(
        "--right-fixed-support",
        type=Path,
        default=MULTIVIEW_ROOT
        / "masks_right/seed/000000-tissue-refined.png",
    )
    parser.add_argument(
        "--left-part-masks",
        type=Path,
        default=TRACK_ROOT
        / "part_masks_full_sequence/part_masks_full.npz",
    )
    parser.add_argument(
        "--pose-driver",
        type=Path,
        default=TRACK_ROOT / "psm_part_corrected_pose_driver.npz",
    )
    parser.add_argument(
        "--surface-gaussians",
        type=Path,
        default=REPO_ROOT
        / "data/super/psm_robot/psm_surface_gaussians.npz",
    )
    parser.add_argument(
        "--left-timestamps",
        type=Path,
        default=MULTIVIEW_ROOT / "timestamps_left_native.npy",
    )
    parser.add_argument(
        "--right-timestamps",
        type=Path,
        default=MULTIVIEW_ROOT / "timestamps_right_native.npy",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=SURGICAL_SAM_ROOT
        / "checkpoints/sam2.1_hiera_s_endo18.pth",
    )
    parser.add_argument(
        "--model-config",
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=MULTIVIEW_ROOT,
    )
    parser.add_argument("--tool-dilation-fullres-px", type=int, default=30)
    parser.add_argument("--coverage-target", type=float, default=0.999)
    parser.add_argument("--keyframe-min", type=int, default=6)
    parser.add_argument("--keyframe-max", type=int, default=12)
    parser.add_argument("--minimum-coverage-gain", type=float, default=1.0e-4)
    parser.add_argument("--positive-points", type=int, default=12)
    parser.add_argument("--negative-points", type=int, default=12)
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=0,
        help="Diagnostic prefix only. Zero is required for a passing final run.",
    )
    parser.add_argument(
        "--reuse-right-image-masks",
        action="store_true",
        help="Reuse this script's existing right image-tool PNGs.",
    )
    parser.add_argument(
        "--accept-visual-review",
        action="store_true",
        help="Record manual acceptance of the generated candidate preview.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image > 0


def unpack_row(packed: np.ndarray, width: int) -> np.ndarray:
    return np.unpackbits(
        np.asarray(packed), axis=1, count=width
    ).astype(bool)


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return (
        cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        > 0
    )


def highlight_mask(image_bgr: np.ndarray) -> np.ndarray:
    normalized = image_bgr.astype(np.float32) / 255.0
    channel_max = normalized.max(axis=2)
    channel_min = normalized.min(axis=2)
    return (channel_max >= 0.90) & (
        channel_max - channel_min <= 0.18
    )


def keep_components_touching(
    mask: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    kept = np.zeros_like(mask, dtype=bool)
    for label in range(1, count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= 32 and np.any(component & reference):
            kept |= component
    return kept


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        raise RuntimeError("Corrected-CAD tool prompt mask is empty")
    selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == selected


def rank_tool_predictions(
    masks: np.ndarray,
    scores: np.ndarray,
    cad_mask: np.ndarray,
    positive_points: np.ndarray,
    negative_points: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    positive_xy = np.rint(positive_points).astype(np.int32)
    negative_xy = np.rint(negative_points).astype(np.int32)
    cad_area = int(cad_mask.sum(dtype=np.int64))
    diagnostics: list[dict[str, Any]] = []
    ranking: list[tuple[float, int]] = []
    for index, raw_mask in enumerate(masks):
        mask = np.asarray(raw_mask).astype(bool)
        positive_misses = int(
            sum(not mask[y, x] for x, y in positive_xy)
        )
        negative_hits = int(sum(mask[y, x] for x, y in negative_xy))
        intersection = int((mask & cad_mask).sum(dtype=np.int64))
        union = int((mask | cad_mask).sum(dtype=np.int64))
        area = int(mask.sum(dtype=np.int64))
        cad_recall = float(intersection / max(cad_area, 1))
        cad_iou = float(intersection / max(union, 1))
        area_ratio = float(area / max(cad_area, 1))
        objective = (
            0.20 * positive_misses
            + 2.0 * negative_hits
            + 1.0 * abs(math.log(max(area_ratio, 1.0e-6)))
            - 1.25 * cad_recall
            - 0.75 * cad_iou
            - 0.25 * float(scores[index])
        )
        diagnostics.append(
            {
                "candidate": index,
                "sam_score": float(scores[index]),
                "area_px_halfres": area,
                "area_ratio_to_cad": area_ratio,
                "cad_recall": cad_recall,
                "cad_iou": cad_iou,
                "positive_misses": positive_misses,
                "negative_hits": negative_hits,
                "ranking_objective": objective,
            }
        )
        ranking.append((objective, index))
    selected_index = min(ranking)[1]
    selected = np.asarray(masks[selected_index]).astype(bool)
    selected = cv2.morphologyEx(
        selected.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ).astype(bool)
    selected = keep_components_touching(selected, cad_mask)
    return selected, {
        "selected_candidate": selected_index,
        "candidates": diagnostics,
    }


def tool_prediction_quality_gate(
    metrics: dict[str, Any], positive_count: int
) -> bool:
    selected = metrics["candidates"][metrics["selected_candidate"]]
    return bool(
        selected["positive_misses"]
        <= math.ceil(positive_count * 2.0 / 3.0)
        and selected["negative_hits"] <= 2
        and selected["sam_score"] >= 0.50
        and selected["cad_recall"] >= 0.25
        and 0.35 <= selected["area_ratio_to_cad"] <= 3.0
    )


def tool_prompts(
    cad_mask: np.ndarray,
    *,
    positive_count: int,
    negative_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # The corrected CAD projection can contain small disconnected jaw/link
    # fragments.  A single SAM image object cannot be forced to bridge those
    # fragments without swallowing background, so prompt the dominant
    # visible component and retain every CAD fragment in the final union.
    prompt_reference = largest_component(cad_mask)
    positive_region = cv2.erode(
        prompt_reference.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ).astype(bool)
    if positive_region.sum() < 100:
        positive_region = prompt_reference
    positive = farthest_points(positive_region, positive_count)
    box = bounding_box(prompt_reference, margin=32)
    x0, y0, x1, y1 = np.rint(box).astype(int)
    box_region = np.zeros_like(cad_mask, dtype=bool)
    box_region[y0 : y1 + 1, x0 : x1 + 1] = True
    negative_region = box_region & ~cv2.dilate(
        prompt_reference.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)),
    ).astype(bool)
    if negative_region.sum() < 100:
        negative_region = ~cv2.dilate(
            prompt_reference.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        ).astype(bool)
    negative = spread_points(
        negative_region,
        negative_count,
        suppression_radius=48,
    )
    return positive, negative, box, prompt_reference


def predict_right_image_tool(
    predictor: CompatibleSAM2ImagePredictor,
    image_bgr_half: np.ndarray,
    cad_half: np.ndarray,
    *,
    positive_count: int,
    negative_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    positive, negative, box, prompt_reference = tool_prompts(
        cad_half,
        positive_count=positive_count,
        negative_count=negative_count,
    )
    points = np.concatenate((positive, negative), axis=0)
    labels = np.asarray(
        [1] * len(positive) + [0] * len(negative), dtype=np.int32
    )
    predictor.set_image(cv2.cvtColor(image_bgr_half, cv2.COLOR_BGR2RGB))
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        box=box,
        multimask_output=True,
    )
    selected, metrics = rank_tool_predictions(
        masks,
        scores,
        prompt_reference,
        positive,
        negative,
    )
    metrics.update(
        {
            "positive_xy_halfres": positive.tolist(),
            "negative_xy_halfres": negative.tolist(),
            "box_xyxy_halfres": box.tolist(),
            "full_cad_area_px_halfres": int(
                cad_half.sum(dtype=np.int64)
            ),
            "prompt_reference_area_px_halfres": int(
                prompt_reference.sum(dtype=np.int64)
            ),
            "selected_postprocessed_area_px_halfres": int(
                selected.sum(dtype=np.int64)
            ),
        }
    )
    metrics["selected_quality_gate"] = tool_prediction_quality_gate(
        metrics, positive_count
    )
    return selected, metrics


def greedy_stereo_set_cover(
    left_visibility: np.ndarray,
    right_visibility: np.ndarray,
    quality: np.ndarray,
    *,
    minimum_count: int,
    maximum_count: int,
    coverage_target: float,
    minimum_gain: float,
) -> tuple[list[int], list[dict[str, Any]], dict[str, int]]:
    left_domain = np.any(left_visibility, axis=0)
    right_domain = np.any(right_visibility, axis=0)
    left_denominator = int(left_domain.sum(dtype=np.int64))
    right_denominator = int(right_domain.sum(dtype=np.int64))
    if not left_denominator or not right_denominator:
        raise RuntimeError("Stereo coverage domain is empty")

    covered_left = np.zeros_like(left_domain)
    covered_right = np.zeros_like(right_domain)
    remaining = set(range(len(left_visibility)))
    selected: list[int] = []
    steps: list[dict[str, Any]] = []
    while remaining and len(selected) < maximum_count:
        ranked = []
        for index in remaining:
            new_left = int(
                (
                    left_visibility[index]
                    & left_domain
                    & ~covered_left
                ).sum(dtype=np.int64)
            )
            new_right = int(
                (
                    right_visibility[index]
                    & right_domain
                    & ~covered_right
                ).sum(dtype=np.int64)
            )
            gain_left = new_left / left_denominator
            gain_right = new_right / right_denominator
            joint_gain = 0.5 * (gain_left + gain_right)
            ranked.append(
                (joint_gain, float(quality[index]), -index, index)
            )
        joint_gain, _, _, best = max(ranked)
        if len(selected) >= minimum_count and joint_gain < minimum_gain:
            break
        new_left_mask = left_visibility[best] & left_domain & ~covered_left
        new_right_mask = (
            right_visibility[best] & right_domain & ~covered_right
        )
        new_left = int(new_left_mask.sum(dtype=np.int64))
        new_right = int(new_right_mask.sum(dtype=np.int64))
        covered_left |= left_visibility[best] & left_domain
        covered_right |= right_visibility[best] & right_domain
        selected.append(best)
        remaining.remove(best)
        left_fraction = float(
            covered_left.sum(dtype=np.int64) / left_denominator
        )
        right_fraction = float(
            covered_right.sum(dtype=np.int64) / right_denominator
        )
        joint_fraction = 0.5 * (left_fraction + right_fraction)
        steps.append(
            {
                "selection_order": len(selected) - 1,
                "candidate_index": best,
                "left_gain_px_halfres": new_left,
                "right_gain_px_halfres": new_right,
                "left_gain_fraction": float(new_left / left_denominator),
                "right_gain_fraction": float(new_right / right_denominator),
                "equal_weight_joint_gain_fraction": float(joint_gain),
                "cumulative_left_coverage": left_fraction,
                "cumulative_right_coverage": right_fraction,
                "cumulative_equal_weight_stereo_coverage": joint_fraction,
            }
        )
        if len(selected) >= minimum_count and joint_fraction >= coverage_target:
            break
    return selected, steps, {
        "left_domain_px_halfres": left_denominator,
        "right_domain_px_halfres": right_denominator,
    }


def tint(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    layer = np.zeros_like(image)
    layer[mask] = color
    cv2.addWeighted(image, 1.0, layer, alpha, 0.0, dst=image)


def draw_contour(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(image, contours, -1, color, thickness)


def make_panel(
    image_bgr: np.ndarray,
    tissue: np.ndarray,
    cad_tool: np.ndarray,
    image_tool: np.ndarray,
    dilated_tool: np.ndarray,
    visibility: np.ndarray,
    *,
    title: str,
) -> np.ndarray:
    panel = cv2.resize(image_bgr, (480, 270), interpolation=cv2.INTER_AREA)
    panel_tissue = resize_mask(tissue, panel.shape[:2])
    panel_cad = resize_mask(cad_tool, panel.shape[:2])
    panel_image_tool = resize_mask(image_tool, panel.shape[:2])
    panel_dilated = resize_mask(dilated_tool, panel.shape[:2])
    panel_visibility = resize_mask(visibility, panel.shape[:2])
    tint(panel, panel_visibility, (0, 180, 0), 0.24)
    draw_contour(panel, panel_tissue, (0, 255, 0), 1)
    draw_contour(panel, panel_cad, (0, 0, 255), 1)
    draw_contour(panel, panel_image_tool, (255, 255, 0), 1)
    draw_contour(panel, panel_dilated, (255, 0, 255), 1)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(
        panel,
        title,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def contact_sheet(
    pair_panels: list[np.ndarray],
    *,
    columns: int,
) -> np.ndarray:
    if not pair_panels:
        raise RuntimeError("Cannot build an empty contact sheet")
    height, width = pair_panels[0].shape[:2]
    columns = min(columns, len(pair_panels))
    rows = math.ceil(len(pair_panels) / columns)
    sheet = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, panel in enumerate(pair_panels):
        row, column = divmod(index, columns)
        sheet[
            row * height : (row + 1) * height,
            column * width : (column + 1) * width,
        ] = panel
    return sheet


def main() -> None:
    args = parse_args()
    if args.tool_dilation_fullres_px < 0:
        raise ValueError("Tool dilation must be non-negative")
    if not 0.0 < args.coverage_target <= 1.0:
        raise ValueError("Coverage target must be in (0, 1]")
    if not 1 <= args.keyframe_min <= args.keyframe_max:
        raise ValueError("Invalid keyframe bounds")

    frozen = verify_frozen_assets()
    observations = read_json(args.observations)
    if not observations.get("passed_for_stage_b", False):
        raise RuntimeError("Stage-A observations did not pass for stage B")
    seed_report = read_json(args.right_seed_report)
    propagation_report = read_json(args.right_propagation_report)
    if not seed_report.get("passed_for_right_propagation", False):
        raise RuntimeError("Right tissue seed was not accepted")
    if not propagation_report.get("passed", False):
        raise RuntimeError("Right tissue propagation did not pass")

    candidates = list(observations["candidate_frames"])
    requested_count = len(candidates)
    if args.candidate_limit > 0:
        candidates = candidates[: args.candidate_limit]
    complete_candidate_run = len(candidates) == requested_count
    if not candidates:
        raise RuntimeError("No candidate frames")

    calibration = read_json(args.calibration)
    K_left = np.asarray(calibration["K_left_rect"], dtype=np.float64)
    K_right = np.asarray(calibration["K_right_rect"], dtype=np.float64)
    baseline_m = float(calibration["baseline_m"])
    left_timestamps = np.load(args.left_timestamps).astype(np.float64)
    right_timestamps = np.load(args.right_timestamps).astype(np.float64)
    left_tissue_packed = np.load(
        args.left_tissue_packbits, mmap_mode="r"
    )
    right_tissue_packed = np.load(
        args.right_tissue_packbits, mmap_mode="r"
    )
    with np.load(args.left_part_masks, allow_pickle=False) as part_archive:
        part_names = part_archive["part_names"].tolist()
        left_part_packed = part_archive["masks_packbits"]
        part_mask_shape = tuple(
            int(value) for value in part_archive["mask_shape"]
        )
    if part_names != ["body", "jaw_left", "jaw_right"]:
        raise RuntimeError(f"Unexpected PSM mask parts: {part_names}")

    first_image = cv2.imread(
        str(args.rgb_dir / "000000-left.png"), cv2.IMREAD_COLOR
    )
    if first_image is None:
        raise FileNotFoundError(args.rgb_dir / "000000-left.png")
    full_shape = first_image.shape[:2]
    full_height, full_width = full_shape
    half_shape = part_mask_shape
    if half_shape != (full_height // 2, full_width // 2):
        raise RuntimeError(
            f"Part mask shape {half_shape} is not half of {full_shape}"
        )
    expected_packed_width = (full_width + 7) // 8
    if left_tissue_packed.shape[1:] != (
        full_height,
        expected_packed_width,
    ):
        raise RuntimeError("Left tissue packbits resolution mismatch")
    if right_tissue_packed.shape[1:] != (
        full_height,
        expected_packed_width,
    ):
        raise RuntimeError("Right tissue packbits resolution mismatch")

    output_dirs = {
        "left_tissue": args.output_root / "masks_left/tissue",
        "left_image_tool": args.output_root / "masks_left/tool_image",
        "left_cad_tool": args.output_root / "masks_left/tool_cad",
        "left_dilated_tool": args.output_root / "masks_left/tool_dilated",
        "left_proxy": args.output_root
        / "masks_left/stage_b_visibility_proxy",
        "right_tissue": args.output_root / "masks_right/tissue",
        "right_image_tool": args.output_root / "masks_right/tool_image",
        "right_cad_tool": args.output_root / "masks_right/tool_cad",
        "right_dilated_tool": args.output_root / "masks_right/tool_dilated",
        "right_proxy": args.output_root
        / "masks_right/stage_b_visibility_proxy",
        "coverage": args.output_root / "coverage",
        "previews": args.output_root / "previews",
    }
    report_path = args.output_root / "stage_b_report.json"
    coverage_path = (
        output_dirs["coverage"] / "stereo_candidate_visibility.npz"
    )
    candidate_preview_path = (
        output_dirs["previews"] / "stage_b_candidate_masks.png"
    )
    selected_preview_path = (
        output_dirs["previews"] / "stage_b_selected_keyframes.png"
    )
    previous_report: dict[str, Any] | None = None
    if args.reuse_right_image_masks:
        if not report_path.is_file():
            raise FileNotFoundError(
                "Reusing right image masks requires the prior "
                f"{report_path}"
            )
        previous_report = read_json(report_path)
        if previous_report.get("stage") != (
            "v10_stage_b_stereo_tissue_tool_masks"
        ):
            raise RuntimeError("Existing Stage-B report has wrong stage")
    previous_tool_metrics = (
        {
            int(item["right_frame"]): item["right"][
                "image_tool_prediction"
            ]
            for item in previous_report["candidate_reports"]
        }
        if previous_report is not None
        else {}
    )
    managed_summary_outputs = [
        report_path,
        coverage_path,
        candidate_preview_path,
        selected_preview_path,
    ]
    if not args.overwrite:
        collisions = [
            path for path in managed_summary_outputs if path.exists()
        ]
        if collisions:
            raise FileExistsError(
                "Stage-B outputs already exist:\n- "
                + "\n- ".join(str(path) for path in collisions)
            )
    for output_dir in output_dirs.values():
        output_dir.mkdir(parents=True, exist_ok=True)

    predictor: CompatibleSAM2ImagePredictor | None = None
    if not args.reuse_right_image_masks:
        if not args.device.startswith("cuda"):
            raise ValueError("SurgicalSAM2 image segmentation requires CUDA")
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        os.environ.setdefault("HYDRA_FULL_ERROR", "1")
        print(
            "[stage-b] loading SurgicalSAM2 small model for right-view "
            "image tool masks",
            flush=True,
        )
        model = build_sam2(
            args.model_config,
            str(args.checkpoint),
            device=args.device,
        )
        predictor = CompatibleSAM2ImagePredictor(model)

    dilation_size = 2 * args.tool_dilation_fullres_px + 1
    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_size, dilation_size)
    )
    left_fixed = load_mask(args.left_fixed_support)
    right_fixed = load_mask(args.right_fixed_support)
    if left_fixed.shape != full_shape or right_fixed.shape != full_shape:
        raise RuntimeError("Fixed tissue support resolution mismatch")

    left_visibility_half: list[np.ndarray] = []
    right_visibility_half: list[np.ndarray] = []
    candidate_reports: list[dict[str, Any]] = []
    pair_panels: list[np.ndarray] = []
    paired_dt_ms: list[float] = []
    started = time.perf_counter()
    for candidate_index, candidate in enumerate(candidates):
        left_frame = int(candidate["left_frame"])
        right_frame = int(candidate["right_frame"])
        left_timestamp = float(left_timestamps[left_frame])
        right_timestamp = float(right_timestamps[right_frame])
        paired_dt = (right_timestamp - left_timestamp) * 1000.0
        paired_dt_ms.append(paired_dt)

        left_path = args.rgb_dir / f"{left_frame:06d}-left.png"
        right_path = args.rgb_dir / f"{right_frame:06d}-right.png"
        left_image = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right_image = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left_image is None:
            raise FileNotFoundError(left_path)
        if right_image is None:
            raise FileNotFoundError(right_path)

        left_tissue = unpack_row(
            left_tissue_packed[left_frame], full_width
        )
        right_tissue = unpack_row(
            right_tissue_packed[right_frame], full_width
        )
        left_parts = np.unpackbits(
            left_part_packed[left_frame],
            axis=-1,
            count=half_shape[0] * half_shape[1],
        ).reshape(len(part_names), *half_shape).astype(bool)
        left_image_tool_half = np.any(left_parts, axis=0)
        left_image_tool = resize_mask(left_image_tool_half, full_shape)

        left_cad, left_cad_metrics = corrected_cad_mask(
            left_timestamp,
            K=K_left,
            baseline_m=baseline_m,
            camera_side="left",
            pose_driver=args.pose_driver,
            surface_gaussians=args.surface_gaussians,
            shape=full_shape,
        )
        right_cad, right_cad_metrics = corrected_cad_mask(
            right_timestamp,
            K=K_right,
            baseline_m=baseline_m,
            camera_side="right",
            pose_driver=args.pose_driver,
            surface_gaussians=args.surface_gaussians,
            shape=full_shape,
        )
        right_cad_half = resize_mask(right_cad, half_shape)
        right_image_tool_path = (
            output_dirs["right_image_tool"]
            / f"{right_frame:06d}-tool-image.png"
        )
        if args.reuse_right_image_masks:
            right_image_tool = load_mask(right_image_tool_path)
            right_image_tool_half = resize_mask(
                right_image_tool, half_shape
            )
            if right_frame not in previous_tool_metrics:
                raise RuntimeError(
                    f"No prior tool metrics for right frame {right_frame}"
                )
            right_tool_metrics = dict(
                previous_tool_metrics[right_frame]
            )
            right_tool_metrics.update(
                {
                    "reused_existing_script_output": True,
                    "path": str(right_image_tool_path.resolve()),
                    "sha256": sha256(right_image_tool_path),
                    "selected_postprocessed_area_px_halfres": int(
                        right_image_tool_half.sum(dtype=np.int64)
                    ),
                }
            )
            right_tool_metrics["selected_quality_gate"] = (
                tool_prediction_quality_gate(
                    right_tool_metrics, args.positive_points
                )
            )
        else:
            assert predictor is not None
            right_half = cv2.resize(
                right_image,
                (half_shape[1], half_shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16
            ):
                (
                    right_image_tool_half,
                    right_tool_metrics,
                ) = predict_right_image_tool(
                    predictor,
                    right_half,
                    right_cad_half,
                    positive_count=args.positive_points,
                    negative_count=args.negative_points,
                )
            right_tool_metrics["source"] = (
                "SurgicalSAM2 image prediction prompted by corrected CAD"
            )
            right_image_tool = resize_mask(
                right_image_tool_half, full_shape
            )

        left_tool_union = left_image_tool | left_cad
        right_tool_union = right_image_tool | right_cad
        left_tool_dilated = cv2.dilate(
            left_tool_union.astype(np.uint8), dilation_kernel
        ).astype(bool)
        right_tool_dilated = cv2.dilate(
            right_tool_union.astype(np.uint8), dilation_kernel
        ).astype(bool)
        left_highlight = highlight_mask(left_image)
        right_highlight = highlight_mask(right_image)
        left_semantic_proxy = (
            left_tissue & ~left_tool_dilated & ~left_highlight
        )
        right_semantic_proxy = (
            right_tissue & ~right_tool_dilated & ~right_highlight
        )

        # Fixed supports prevent slow SAM boundary drift from being counted as
        # new geometric coverage.  Per-frame tissue masks remain saved and are
        # audited independently.
        left_cover_proxy = (
            left_fixed & ~left_tool_dilated & ~left_highlight
        )
        right_cover_proxy = (
            right_fixed & ~right_tool_dilated & ~right_highlight
        )
        left_visibility_half.append(
            resize_mask(left_cover_proxy, half_shape)
        )
        right_visibility_half.append(
            resize_mask(right_cover_proxy, half_shape)
        )

        output_masks = {
            output_dirs["left_tissue"]
            / f"{left_frame:06d}-tissue.png": left_tissue,
            output_dirs["left_image_tool"]
            / f"{left_frame:06d}-tool-image.png": left_image_tool,
            output_dirs["left_cad_tool"]
            / f"{left_frame:06d}-tool-cad.png": left_cad,
            output_dirs["left_dilated_tool"]
            / f"{left_frame:06d}-tool-dilated.png": left_tool_dilated,
            output_dirs["left_proxy"]
            / f"{left_frame:06d}-visibility-proxy.png": (
                left_semantic_proxy
            ),
            output_dirs["right_tissue"]
            / f"{right_frame:06d}-tissue.png": right_tissue,
            right_image_tool_path: right_image_tool,
            output_dirs["right_cad_tool"]
            / f"{right_frame:06d}-tool-cad.png": right_cad,
            output_dirs["right_dilated_tool"]
            / f"{right_frame:06d}-tool-dilated.png": right_tool_dilated,
            output_dirs["right_proxy"]
            / f"{right_frame:06d}-visibility-proxy.png": (
                right_semantic_proxy
            ),
        }
        for path, mask in output_masks.items():
            if (
                path == right_image_tool_path
                and args.reuse_right_image_masks
            ):
                continue
            if path.exists() and not args.overwrite:
                raise FileExistsError(path)
            if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
                raise RuntimeError(f"Could not write {path}")

        left_panel = make_panel(
            left_image,
            left_tissue,
            left_cad,
            left_image_tool,
            left_tool_dilated,
            left_semantic_proxy,
            title=(
                f"L{left_frame} tissue {left_tissue.sum()/1e6:.3f}M "
                f"proxy {left_semantic_proxy.sum()/1e6:.3f}M"
            ),
        )
        right_panel = make_panel(
            right_image,
            right_tissue,
            right_cad,
            right_image_tool,
            right_tool_dilated,
            right_semantic_proxy,
            title=(
                f"R{right_frame} tissue {right_tissue.sum()/1e6:.3f}M "
                f"proxy {right_semantic_proxy.sum()/1e6:.3f}M"
            ),
        )
        pair_panels.append(np.concatenate((left_panel, right_panel), axis=1))
        candidate_reports.append(
            {
                "candidate_index": candidate_index,
                "left_frame": left_frame,
                "right_frame": right_frame,
                "left_timestamp": left_timestamp,
                "right_timestamp": right_timestamp,
                "right_minus_left_ms": paired_dt,
                "quality_score": float(candidate["quality_score"]),
                "left": {
                    "tissue_area_px": int(
                        left_tissue.sum(dtype=np.int64)
                    ),
                    "image_tool_area_px": int(
                        left_image_tool.sum(dtype=np.int64)
                    ),
                    "corrected_cad_tool_area_px": int(
                        left_cad.sum(dtype=np.int64)
                    ),
                    "dilated_tool_union_area_px": int(
                        left_tool_dilated.sum(dtype=np.int64)
                    ),
                    "highlight_inside_tissue_px": int(
                        (left_highlight & left_tissue).sum(
                            dtype=np.int64
                        )
                    ),
                    "semantic_visibility_proxy_area_px": int(
                        left_semantic_proxy.sum(dtype=np.int64)
                    ),
                    "fixed_support_coverage_proxy_area_px": int(
                        left_cover_proxy.sum(dtype=np.int64)
                    ),
                    "corrected_cad": left_cad_metrics,
                },
                "right": {
                    "tissue_area_px": int(
                        right_tissue.sum(dtype=np.int64)
                    ),
                    "image_tool_area_px": int(
                        right_image_tool.sum(dtype=np.int64)
                    ),
                    "corrected_cad_tool_area_px": int(
                        right_cad.sum(dtype=np.int64)
                    ),
                    "dilated_tool_union_area_px": int(
                        right_tool_dilated.sum(dtype=np.int64)
                    ),
                    "highlight_inside_tissue_px": int(
                        (right_highlight & right_tissue).sum(
                            dtype=np.int64
                        )
                    ),
                    "semantic_visibility_proxy_area_px": int(
                        right_semantic_proxy.sum(dtype=np.int64)
                    ),
                    "fixed_support_coverage_proxy_area_px": int(
                        right_cover_proxy.sum(dtype=np.int64)
                    ),
                    "corrected_cad": right_cad_metrics,
                    "image_tool_prediction": right_tool_metrics,
                },
            }
        )
        print(
            f"[stage-b] candidate {candidate_index + 1}/{len(candidates)} "
            f"L{left_frame}/R{right_frame}",
            flush=True,
        )

    left_visibility = np.stack(left_visibility_half)
    right_visibility = np.stack(right_visibility_half)
    quality = np.asarray(
        [candidate["quality_score"] for candidate in candidates],
        dtype=np.float64,
    )
    selected, coverage_steps, coverage_domains = greedy_stereo_set_cover(
        left_visibility,
        right_visibility,
        quality,
        minimum_count=min(args.keyframe_min, len(candidates)),
        maximum_count=min(args.keyframe_max, len(candidates)),
        coverage_target=args.coverage_target,
        minimum_gain=args.minimum_coverage_gain,
    )
    for step in coverage_steps:
        candidate = candidate_reports[step["candidate_index"]]
        step.update(
            {
                "left_frame": candidate["left_frame"],
                "right_frame": candidate["right_frame"],
                "left_timestamp": candidate["left_timestamp"],
                "right_timestamp": candidate["right_timestamp"],
                "right_minus_left_ms": candidate["right_minus_left_ms"],
                "quality_score": candidate["quality_score"],
            }
        )
    final_coverage = coverage_steps[-1]

    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        coverage_path,
        candidate_left_frames=np.asarray(
            [item["left_frame"] for item in candidate_reports],
            dtype=np.int32,
        ),
        candidate_right_frames=np.asarray(
            [item["right_frame"] for item in candidate_reports],
            dtype=np.int32,
        ),
        left_fixed_support_visibility=left_visibility,
        right_fixed_support_visibility=right_visibility,
        selected_candidate_indices=np.asarray(selected, dtype=np.int32),
        selection_order_left_frames=np.asarray(
            [candidate_reports[index]["left_frame"] for index in selected],
            dtype=np.int32,
        ),
        selection_order_right_frames=np.asarray(
            [candidate_reports[index]["right_frame"] for index in selected],
            dtype=np.int32,
        ),
    )
    cv2.imwrite(
        str(candidate_preview_path),
        contact_sheet(pair_panels, columns=5),
    )
    cv2.imwrite(
        str(selected_preview_path),
        contact_sheet(
            [pair_panels[index] for index in selected],
            columns=2,
        ),
    )

    right_prediction_gates = [
        item["right"]["image_tool_prediction"]["selected_quality_gate"]
        for item in candidate_reports
    ]
    gates = {
        "frozen_assets_match": bool(
            all(item["matches"] for item in frozen.values())
        ),
        "stage_a_passed": bool(
            observations.get("passed_for_stage_b", False)
        ),
        "right_seed_accepted": bool(
            seed_report.get("passed_for_right_propagation", False)
        ),
        "right_propagation_passed": bool(
            propagation_report.get("passed", False)
        ),
        "all_25_candidates_processed": bool(
            complete_candidate_run and requested_count == 25
        ),
        "all_pairs_within_20ms": bool(
            max(abs(value) for value in paired_dt_ms) <= 20.0
        ),
        "right_image_tool_quality_gates": bool(
            right_prediction_gates and all(right_prediction_gates)
        ),
        "minimum_six_keyframes_selected": len(selected) >= 6,
        "equal_weight_stereo_proxy_coverage_at_least_target": bool(
            final_coverage[
                "cumulative_equal_weight_stereo_coverage"
            ]
            >= args.coverage_target
        ),
        "visual_review_accepted": bool(args.accept_visual_review),
    }
    automatic_gate_names = [
        name for name in gates if name != "visual_review_accepted"
    ]
    passed_automatic = bool(
        all(gates[name] for name in automatic_gate_names)
    )
    passed = bool(passed_automatic and gates["visual_review_accepted"])
    report = {
        "stage": "v10_stage_b_stereo_tissue_tool_masks",
        "status": (
            "passed_for_stage_c"
            if passed
            else (
                "automatic_gates_passed_visual_review_pending"
                if passed_automatic
                else "automatic_gate_failed"
            )
        ),
        "runtime_switch_performed": False,
        "candidate_asset_created": False,
        "active_runtime": (
            "data/super/grasp5_native/"
            "bodies_v9_dense_0p5mm_rigid_tissue"
        ),
        "method": {
            "left_tissue": (
                "existing independent left-view SAM2 video propagation"
            ),
            "right_tissue": (
                "accepted independent right-view seed and right-view SAM2 "
                "video propagation"
            ),
            "left_tool": (
                "three-part left image masks union corrected-CAD projection"
            ),
            "right_tool": (
                "SurgicalSAM2 right-image prediction prompted by corrected "
                "CAD, union corrected-CAD projection"
            ),
            "tool_dilation_fullres_px": args.tool_dilation_fullres_px,
            "highlight_rejection": (
                "max RGB >= 0.90 and channel spread <= 0.18"
            ),
            "stage_b_visibility_proxy": (
                "per-frame SAM2 tissue & ~dilated tool union & ~highlight"
            ),
            "coverage_support": (
                "fixed manually reviewed frame-0 support per view; "
                "per-frame SAM boundary drift is not counted as new coverage"
            ),
            "stereo_set_cover": (
                "greedy marginal gain with left and right normalized "
                "coverage weighted 50%/50%"
            ),
        },
        "important_boundary": {
            "valid_dense_depth_applied": False,
            "stereo_confidence_applied": False,
            "reason": (
                "right-reference depth and per-view confidence are Stage-C "
                "products and do not yet exist"
            ),
            "next_intersection": (
                "stage_b_visibility_proxy & valid_dense_depth & "
                "acceptable_stereo_confidence"
            ),
            "masks_are_final_reconstruction_masks": False,
        },
        "inputs": {
            "observations": {
                "path": str(args.observations.resolve()),
                "sha256": sha256(args.observations),
            },
            "calibration": {
                "path": str(args.calibration.resolve()),
                "sha256": sha256(args.calibration),
            },
            "left_tissue_packbits": {
                "path": str(args.left_tissue_packbits.resolve()),
                "sha256": sha256(args.left_tissue_packbits),
            },
            "right_tissue_packbits": {
                "path": str(args.right_tissue_packbits.resolve()),
                "sha256": sha256(args.right_tissue_packbits),
            },
            "right_seed_report": {
                "path": str(args.right_seed_report.resolve()),
                "sha256": sha256(args.right_seed_report),
            },
            "right_propagation_report": {
                "path": str(args.right_propagation_report.resolve()),
                "sha256": sha256(args.right_propagation_report),
            },
            "left_part_masks": {
                "path": str(args.left_part_masks.resolve()),
                "sha256": sha256(args.left_part_masks),
            },
            "corrected_driver": {
                "path": str(args.pose_driver.resolve()),
                "sha256": sha256(args.pose_driver),
            },
            "surface_gaussians": {
                "path": str(args.surface_gaussians.resolve()),
                "sha256": sha256(args.surface_gaussians),
            },
            "left_timestamp_array_sha256": array_sha256(
                left_timestamps
            ),
            "right_timestamp_array_sha256": array_sha256(
                right_timestamps
            ),
            "surgical_sam_checkpoint": {
                "path": str(args.checkpoint.resolve()),
                "sha256": sha256(args.checkpoint),
                "config": args.model_config,
            },
        },
        "frozen_assets": frozen,
        "candidate_count": len(candidates),
        "candidate_reports": candidate_reports,
        "coverage": {
            "target": args.coverage_target,
            "minimum_gain": args.minimum_coverage_gain,
            "keyframe_min": args.keyframe_min,
            "keyframe_max": args.keyframe_max,
            **coverage_domains,
            "selection_order": coverage_steps,
            "selected_temporal_order": sorted(
                [
                    {
                        "selection_order": order,
                        "candidate_index": index,
                        "left_frame": candidate_reports[index][
                            "left_frame"
                        ],
                        "right_frame": candidate_reports[index][
                            "right_frame"
                        ],
                    }
                    for order, index in enumerate(selected)
                ],
                key=lambda item: item["left_frame"],
            ),
            "final_left_proxy_coverage": final_coverage[
                "cumulative_left_coverage"
            ],
            "final_right_proxy_coverage": final_coverage[
                "cumulative_right_coverage"
            ],
            "final_equal_weight_stereo_proxy_coverage": final_coverage[
                "cumulative_equal_weight_stereo_coverage"
            ],
        },
        "outputs": {
            "root": str(args.output_root.resolve()),
            "coverage": {
                "path": str(coverage_path.resolve()),
                "sha256": sha256(coverage_path),
            },
            "candidate_preview": {
                "path": str(candidate_preview_path.resolve()),
                "sha256": sha256(candidate_preview_path),
                "legend": (
                    "green fill/contour=semantic visibility/tissue; "
                    "red=corrected CAD; cyan=image tool; "
                    "magenta=dilated tool union"
                ),
            },
            "selected_preview": {
                "path": str(selected_preview_path.resolve()),
                "sha256": sha256(selected_preview_path),
            },
            "mask_directories": {
                name: str(path.resolve())
                for name, path in output_dirs.items()
                if name not in {"coverage", "previews"}
            },
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "gates": gates,
        "passed_automatic_gates": passed_automatic,
        "passed_for_stage_c": passed,
        "next_required_step": (
            "Build timestamp-synchronized left- and right-reference depth "
            "plus confidence, then intersect it with these Stage-B proxies."
            if passed
            else (
                "Inspect stage_b_candidate_masks.png and rerun with "
                "--reuse-right-image-masks --overwrite "
                "--accept-visual-review if the per-view tissue/tool "
                "boundaries are accepted."
            )
        ),
    }
    write_json(report_path, report)
    print(
        "[stage-b] selected "
        + ", ".join(
            f"L{candidate_reports[index]['left_frame']}/"
            f"R{candidate_reports[index]['right_frame']}"
            for index in selected
        ),
        flush=True,
    )
    print(
        "[stage-b] final proxy coverage "
        f"left={report['coverage']['final_left_proxy_coverage']:.6f}, "
        f"right={report['coverage']['final_right_proxy_coverage']:.6f}, "
        "equal-weight="
        f"{report['coverage']['final_equal_weight_stereo_proxy_coverage']:.6f}",
        flush=True,
    )
    if not passed_automatic:
        raise SystemExit("Stage-B automatic gate failed")


if __name__ == "__main__":
    main()
