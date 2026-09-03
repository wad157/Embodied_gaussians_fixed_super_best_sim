#!/usr/bin/env python3
"""Build and refine the independent right-view tissue seed for SUPER stage B.

The left frame-0 tissue/depth pair is used only to initialize a rectified
right-view mask.  A SAM2 image prediction is then run on the actual first
right image with positive tissue points, negative corrected-CAD tool points,
and exterior negative points.  The result is written under the isolated
``tissue_multiview_v1`` directory; frozen calibration/runtime assets are never
modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL.Image import Image
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOT = REPO_ROOT / "data/super/grasp5_native"
TRACK_ROOT = REPO_ROOT / "data/super/psm_tracking"
MULTIVIEW_ROOT = NATIVE_ROOT / "tissue_multiview_v1"
SURGICAL_SAM_ROOT = Path(
    "/Media_HDD/jwshan/wad/online_dvrk_tracking/SurgicalSAM2"
)
sys.path.insert(0, str(SURGICAL_SAM_ROOT))

from sam2.build_sam import build_sam2_hf  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402

from select_super_tissue_multiview_frames import (  # noqa: E402
    verify_frozen_assets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an image-refined right-view SUPER tissue seed."
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=MULTIVIEW_ROOT / "observations.json",
    )
    parser.add_argument(
        "--left-mask",
        type=Path,
        default=NATIVE_ROOT / "masks/000000-tissue.png",
    )
    parser.add_argument(
        "--left-depth",
        type=Path,
        default=NATIVE_ROOT
        / "depth_v4_foundation_dense_timestamped/000000-depth.npy",
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
        "--pose-driver",
        type=Path,
        default=TRACK_ROOT / "psm_part_corrected_pose_driver.npz",
    )
    parser.add_argument(
        "--surface-gaussians",
        type=Path,
        default=REPO_ROOT / "data/super/psm_robot/psm_surface_gaussians.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MULTIVIEW_ROOT / "masks_right/seed",
    )
    parser.add_argument("--model-id", default="facebook/sam2.1-hiera-large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--positive-points", type=int, default=24)
    parser.add_argument("--tool-negative-points", type=int, default=8)
    parser.add_argument("--exterior-negative-points", type=int, default=10)
    parser.add_argument(
        "--accept-visual-review",
        action="store_true",
        help="Record that the generated overlay was inspected and accepted.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pose_to_matrix(pose_xyz_xyzw: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(pose_xyz_xyzw[3:]).as_matrix()
    matrix[:3, 3] = pose_xyz_xyzw[:3]
    return matrix


def nearest_index(timestamps: np.ndarray, timestamp: float) -> int:
    after = int(np.searchsorted(timestamps, timestamp, side="left"))
    after = int(np.clip(after, 0, len(timestamps) - 1))
    before = max(0, after - 1)
    if abs(timestamps[after] - timestamp) < abs(
        timestamps[before] - timestamp
    ):
        return after
    return before


def project_left_mask_to_right(
    mask: np.ndarray,
    depth: np.ndarray,
    *,
    fx: float,
    baseline_m: float,
) -> tuple[np.ndarray, dict[str, float]]:
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    rows, columns = np.nonzero(valid)
    disparity = fx * baseline_m / depth[rows, columns]
    right_columns = np.rint(columns.astype(np.float64) - disparity).astype(
        np.int32
    )
    inside = (right_columns >= 0) & (right_columns < mask.shape[1])
    projected = np.zeros_like(mask, dtype=np.uint8)
    projected[rows[inside], right_columns[inside]] = 1
    projected = cv2.morphologyEx(
        projected,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    projected = cv2.dilate(
        projected,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        projected, connectivity=8
    )
    if component_count <= 1:
        raise RuntimeError("Projected right tissue seed is empty")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    projected = labels == largest
    return projected, {
        "valid_left_tissue_fraction": float(
            valid.sum(dtype=np.int64) / max(mask.sum(dtype=np.int64), 1)
        ),
        "disparity_px_min": float(disparity.min()),
        "disparity_px_p05": float(np.quantile(disparity, 0.05)),
        "disparity_px_median": float(np.median(disparity)),
        "disparity_px_p95": float(np.quantile(disparity, 0.95)),
        "disparity_px_max": float(disparity.max()),
        "projected_area_px": int(projected.sum(dtype=np.int64)),
    }


def corrected_cad_mask(
    timestamp: float,
    *,
    K: np.ndarray,
    baseline_m: float,
    camera_side: str,
    pose_driver: Path,
    surface_gaussians: Path,
    shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(pose_driver, allow_pickle=False) as driver:
        driver_timestamps = driver["timestamps"].astype(np.float64)
        driver_links = driver["link_names"].tolist()
        poses = driver["poses_rect_camera_xyz_xyzw"].astype(np.float64)
    with np.load(surface_gaussians, allow_pickle=False) as asset:
        local_means = asset["means"].astype(np.float64)
        local_scales = asset["scales"].astype(np.float64)
        link_ids = asset["link_ids"].astype(np.int64)
        asset_links = asset["link_names"].tolist()
    if driver_links != asset_links:
        raise RuntimeError("Corrected driver and surface asset link order differ")

    state_index = nearest_index(driver_timestamps, timestamp)
    points_left = np.empty_like(local_means)
    scales = np.empty(len(local_means), dtype=np.float64)
    for link_index in range(len(driver_links)):
        selected = link_ids == link_index
        transform = pose_to_matrix(poses[state_index, link_index])
        points_left[selected] = (
            local_means[selected] @ transform[:3, :3].T
            + transform[:3, 3]
        )
        scales[selected] = np.max(local_scales[selected], axis=1)

    if camera_side not in {"left", "right"}:
        raise ValueError(f"Unsupported camera side: {camera_side}")
    points_camera = points_left.copy()
    if camera_side == "right":
        points_camera[:, 0] -= baseline_m
    positive = points_camera[:, 2] > 1.0e-5
    points_camera = points_camera[positive]
    scales = scales[positive]
    pixels = np.empty((len(points_camera), 2), dtype=np.float64)
    pixels[:, 0] = (
        K[0, 0] * points_camera[:, 0] / points_camera[:, 2] + K[0, 2]
    )
    pixels[:, 1] = (
        K[1, 1] * points_camera[:, 1] / points_camera[:, 2] + K[1, 2]
    )
    radii = np.clip(
        np.ceil(0.45 * K[0, 0] * scales / points_camera[:, 2]),
        2,
        55,
    ).astype(np.int32)
    height, width = shape
    inside = (
        (pixels[:, 0] >= -80)
        & (pixels[:, 0] < width + 80)
        & (pixels[:, 1] >= -80)
        & (pixels[:, 1] < height + 80)
    )
    mask = np.zeros(shape, dtype=np.uint8)
    for pixel, radius in zip(pixels[inside], radii[inside], strict=True):
        center = tuple(np.rint(pixel).astype(np.int32))
        cv2.circle(mask, center, int(radius), 1, -1, cv2.LINE_AA)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ).astype(bool)
    return mask, {
        "driver_state_index": state_index,
        "driver_timestamp": float(driver_timestamps[state_index]),
        "driver_minus_image_ms": float(
            (driver_timestamps[state_index] - timestamp) * 1000.0
        ),
        "projected_gaussian_count": int(inside.sum(dtype=np.int64)),
        "mask_area_px": int(mask.sum(dtype=np.int64)),
    }


def spread_points(
    mask: np.ndarray,
    count: int,
    *,
    suppression_radius: int,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 2), dtype=np.float32)
    distance = cv2.distanceTransform(
        mask.astype(np.uint8), cv2.DIST_L2, 5
    )
    points: list[tuple[int, int]] = []
    working = distance.copy()
    for _ in range(count):
        _, maximum, _, location = cv2.minMaxLoc(working)
        if maximum <= 0.0:
            break
        points.append(location)
        cv2.circle(working, location, suppression_radius, 0.0, -1)
    if not points:
        raise RuntimeError("Could not select prompt points from mask")
    return np.asarray(points, dtype=np.float32)


def farthest_points(mask: np.ndarray, count: int) -> np.ndarray:
    """Select spatially distributed interior prompts across a large object."""

    if count <= 0:
        return np.empty((0, 2), dtype=np.float32)
    rows, columns = np.nonzero(mask)
    if not len(columns):
        raise RuntimeError("Could not select farthest points from empty mask")
    stride = max(1, len(columns) // 30000)
    candidates = np.stack((columns[::stride], rows[::stride]), axis=1).astype(
        np.float32
    )
    distance_to_boundary = cv2.distanceTransform(
        mask.astype(np.uint8), cv2.DIST_L2, 5
    )
    first_values = distance_to_boundary[
        candidates[:, 1].astype(np.int32),
        candidates[:, 0].astype(np.int32),
    ]
    selected = [candidates[int(np.argmax(first_values))]]
    minimum_squared = np.sum(
        (candidates - selected[0][None]) ** 2, axis=1
    )
    for _ in range(1, count):
        index = int(np.argmax(minimum_squared))
        selected.append(candidates[index])
        squared = np.sum(
            (candidates - candidates[index][None]) ** 2, axis=1
        )
        minimum_squared = np.minimum(minimum_squared, squared)
    return np.asarray(selected, dtype=np.float32)


def bounding_box(mask: np.ndarray, margin: int) -> np.ndarray:
    rows, columns = np.nonzero(mask)
    if not len(columns):
        raise RuntimeError("Cannot make a box from an empty mask")
    height, width = mask.shape
    return np.asarray(
        [
            max(0, int(columns.min()) - margin),
            max(0, int(rows.min()) - margin),
            min(width - 1, int(columns.max()) + margin),
            min(height - 1, int(rows.max()) + margin),
        ],
        dtype=np.float32,
    )


class CompatibleSAM2ImagePredictor(SAM2ImagePredictor):
    """Avoid non-contiguous view assumptions across local PyTorch versions."""

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
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        feature_sizes = []
        for feature in vision_feats:
            side = int(round(math.sqrt(feature.shape[0])))
            if side * side != feature.shape[0]:
                raise ValueError(
                    f"Non-square SAM feature map: {feature.shape}"
                )
            feature_sizes.append((side, side))
        features = [
            feature.permute(1, 2, 0).reshape(1, -1, *feature_size)
            for feature, feature_size in zip(
                vision_feats[::-1], feature_sizes[::-1], strict=True
            )
        ][::-1]
        self._features = {
            "image_embed": features[-1],
            "high_res_feats": features[:-1],
        }
        self._is_image_set = True


def select_prediction(
    masks: np.ndarray,
    scores: np.ndarray,
    projected_seed: np.ndarray,
    positive_points: np.ndarray,
    negative_points: np.ndarray,
) -> tuple[int, list[dict[str, Any]]]:
    masks = np.asarray(masks).astype(bool)
    positive_xy = np.rint(positive_points).astype(np.int32)
    negative_xy = np.rint(negative_points).astype(np.int32)
    seed_area = int(projected_seed.sum(dtype=np.int64))
    diagnostics = []
    ranking = []
    for index, mask in enumerate(masks):
        positive_misses = int(
            sum(not mask[y, x] for x, y in positive_xy)
        )
        negative_hits = int(sum(mask[y, x] for x, y in negative_xy))
        intersection = int((mask & projected_seed).sum(dtype=np.int64))
        union = int((mask | projected_seed).sum(dtype=np.int64))
        area = int(mask.sum(dtype=np.int64))
        iou = float(intersection / max(union, 1))
        seed_recall = float(intersection / max(seed_area, 1))
        area_ratio = float(area / max(seed_area, 1))
        prompt_violations = positive_misses + negative_hits
        area_penalty = abs(math.log(max(area_ratio, 1.0e-6)))
        objective = (
            2.0 * prompt_violations
            + 0.35 * area_penalty
            - iou
            - 0.15 * float(scores[index])
        )
        diagnostics.append(
            {
                "candidate": index,
                "sam_score": float(scores[index]),
                "area_px": area,
                "area_ratio_to_projection": area_ratio,
                "iou_with_projection": iou,
                "projection_recall": seed_recall,
                "positive_misses": positive_misses,
                "negative_hits": negative_hits,
                "ranking_objective": objective,
            }
        )
        ranking.append((objective, index))
    return min(ranking)[1], diagnostics


def overlay(
    image: np.ndarray,
    projected: np.ndarray,
    refined: np.ndarray,
    cad_tool: np.ndarray,
    positive_points: np.ndarray,
    negative_points: np.ndarray,
) -> np.ndarray:
    result = image.copy()
    layer = np.zeros_like(result)
    layer[projected] = (0, 180, 255)
    layer[refined] = (0, 210, 0)
    layer[cad_tool] = (30, 30, 230)
    result = cv2.addWeighted(result, 1.0, layer, 0.35, 0.0)
    for mask, color in (
        (projected, (0, 180, 255)),
        (refined, (0, 255, 0)),
        (cad_tool, (0, 0, 255)),
    ):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(result, contours, -1, color, 2)
    for point in positive_points:
        cv2.circle(
            result, tuple(np.rint(point).astype(int)), 7, (0, 255, 255), -1
        )
    for point in negative_points:
        cv2.circle(
            result, tuple(np.rint(point).astype(int)), 7, (255, 0, 255), -1
        )
    cv2.rectangle(result, (0, 0), (1120, 76), (0, 0, 0), -1)
    cv2.putText(
        result,
        "right frame 0: orange=left-depth projection, "
        "green=right-image SAM2, red=corrected CAD tool",
        (14, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        "yellow=positive tissue prompts, magenta=negative prompts",
        (14, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    return result


def main() -> None:
    args = parse_args()
    frozen = verify_frozen_assets()
    observations = read_json(args.observations)
    if not observations.get("passed_for_stage_b", False):
        raise RuntimeError("Stage-A observations did not pass for stage B")
    if observations.get("candidate_asset_created", True):
        raise RuntimeError("Stage A unexpectedly created a scene candidate")

    outputs = {
        "projected_mask": args.output_dir / "000000-tissue-projected.png",
        "refined_mask": args.output_dir / "000000-tissue-refined.png",
        "cad_tool_mask": args.output_dir / "000000-tool-cad.png",
        "overlay": args.output_dir / "000000-right-seed-overlay.png",
        "report": args.output_dir / "report.json",
    }
    if not args.overwrite:
        collisions = [path for path in outputs.values() if path.exists()]
        if collisions:
            raise FileExistsError(
                "Right-seed outputs already exist:\n- "
                + "\n- ".join(str(path) for path in collisions)
            )

    calibration = read_json(args.calibration)
    K = np.asarray(calibration["K_right_rect"], dtype=np.float64)
    baseline_m = float(calibration["baseline_m"])
    left_mask = (
        cv2.imread(str(args.left_mask), cv2.IMREAD_GRAYSCALE) > 0
    )
    depth = np.load(args.left_depth).astype(np.float32)
    if left_mask.shape != depth.shape:
        raise RuntimeError(
            f"Left mask/depth shape mismatch: {left_mask.shape}/{depth.shape}"
        )
    projected, projection_metrics = project_left_mask_to_right(
        left_mask,
        depth,
        fx=float(K[0, 0]),
        baseline_m=baseline_m,
    )

    right_timestamps = np.load(
        MULTIVIEW_ROOT / "timestamps_right_native.npy"
    ).astype(np.float64)
    right_frame = 0
    right_timestamp = float(right_timestamps[right_frame])
    right_path = args.rgb_dir / f"{right_frame:06d}-right.png"
    image_bgr = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(right_path)
    if image_bgr.shape[:2] != projected.shape:
        raise RuntimeError(
            f"Right image/projection shape mismatch: "
            f"{image_bgr.shape[:2]}/{projected.shape}"
        )
    cad_tool, cad_metrics = corrected_cad_mask(
        right_timestamp,
        K=K,
        baseline_m=baseline_m,
        camera_side="right",
        pose_driver=args.pose_driver,
        surface_gaussians=args.surface_gaussians,
        shape=projected.shape,
    )

    positive_region = cv2.erode(
        projected.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    ).astype(bool)
    positive_region &= ~cv2.dilate(
        cad_tool.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    ).astype(bool)
    positive_points = farthest_points(
        positive_region,
        args.positive_points,
    )
    tool_points = spread_points(
        cv2.erode(
            cad_tool.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        ).astype(bool),
        args.tool_negative_points,
        suppression_radius=45,
    )
    expanded_box = bounding_box(projected, margin=35)
    box_region = np.zeros_like(projected, dtype=np.uint8)
    x0, y0, x1, y1 = np.rint(expanded_box).astype(int)
    box_region[y0 : y1 + 1, x0 : x1 + 1] = 1
    exterior_region = (
        box_region.astype(bool)
        & ~cv2.dilate(
            projected.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        ).astype(bool)
        & ~cad_tool
    )
    exterior_points = spread_points(
        exterior_region,
        args.exterior_negative_points,
        suppression_radius=80,
    )
    negative_points = np.concatenate((tool_points, exterior_points), axis=0)
    points = np.concatenate((positive_points, negative_points), axis=0)
    labels = np.asarray(
        [1] * len(positive_points) + [0] * len(negative_points),
        dtype=np.int32,
    )

    if not args.device.startswith("cuda"):
        raise ValueError("SAM2.1-large seed refinement requires CUDA")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    print(
        f"[right-seed] loading {args.model_id} and refining on right frame 0",
        flush=True,
    )
    model = build_sam2_hf(args.model_id, device=args.device)
    predictor = CompatibleSAM2ImagePredictor(model)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16
    ):
        predictor.set_image(image_rgb)
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=expanded_box,
            multimask_output=True,
        )
    selected_index, candidates = select_prediction(
        masks,
        scores,
        projected,
        positive_points,
        negative_points,
    )
    refined = np.asarray(masks[selected_index]).astype(bool)
    selected = candidates[selected_index]

    projection_area = int(projected.sum(dtype=np.int64))
    refined_area = int(refined.sum(dtype=np.int64))
    cad_tool_core = cv2.erode(
        cad_tool.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    ).astype(bool)
    tool_overlap = int((refined & cad_tool_core).sum(dtype=np.int64))
    gates = {
        "frozen_assets_match": bool(
            all(result["matches"] for result in frozen.values())
        ),
        "projection_dense_enough": bool(
            projection_metrics["valid_left_tissue_fraction"] >= 0.99
        ),
        "all_positive_prompts_inside": selected["positive_misses"] == 0,
        "all_negative_prompts_outside": selected["negative_hits"] == 0,
        "projection_recall_at_least_95pct": bool(
            selected["projection_recall"] >= 0.95
        ),
        "refined_area_within_20pct_of_projection": bool(
            0.80 <= refined_area / projection_area <= 1.20
        ),
        "cad_tool_core_overlap_below_2pct": bool(
            tool_overlap / max(cad_tool_core.sum(dtype=np.int64), 1) <= 0.02
        ),
        "visual_review_accepted": bool(args.accept_visual_review),
    }
    automatic_gate_names = [
        name for name in gates if name != "visual_review_accepted"
    ]
    passed_automatic = bool(
        all(gates[name] for name in automatic_gate_names)
    )
    passed_for_propagation = bool(
        passed_automatic and gates["visual_review_accepted"]
    )
    report = {
        "stage": "v10_stage_b_right_tissue_seed",
        "status": (
            "accepted_for_right_propagation"
            if passed_for_propagation
            else "automatic_gates_passed_visual_review_pending"
        ),
        "model_id": args.model_id,
        "device": args.device,
        "right_frame": right_frame,
        "right_timestamp": right_timestamp,
        "method": (
            "left frame-0 dense depth projection is initialization only; "
            "the saved seed is a separate SAM2 prediction on the actual "
            "right frame with corrected-CAD tool negatives"
        ),
        "inputs": {
            "observations": {
                "path": str(args.observations.resolve()),
                "sha256": sha256(args.observations),
            },
            "left_mask": {
                "path": str(args.left_mask.resolve()),
                "sha256": sha256(args.left_mask),
            },
            "left_depth": {
                "path": str(args.left_depth.resolve()),
                "sha256": sha256(args.left_depth),
            },
            "calibration": {
                "path": str(args.calibration.resolve()),
                "sha256": sha256(args.calibration),
            },
            "right_image": {
                "path": str(right_path.resolve()),
                "sha256": sha256(right_path),
            },
            "corrected_driver": {
                "path": str(args.pose_driver.resolve()),
                "sha256": sha256(args.pose_driver),
            },
            "surface_gaussians": {
                "path": str(args.surface_gaussians.resolve()),
                "sha256": sha256(args.surface_gaussians),
            },
        },
        "frozen_assets": frozen,
        "projection": projection_metrics,
        "corrected_cad_tool": cad_metrics,
        "prompts": {
            "positive_xy": positive_points.tolist(),
            "negative_tool_xy": tool_points.tolist(),
            "negative_exterior_xy": exterior_points.tolist(),
            "box_xyxy": expanded_box.tolist(),
        },
        "sam_candidates": candidates,
        "selected_candidate": selected_index,
        "selected_metrics": {
            **selected,
            "tool_overlap_px": tool_overlap,
            "tool_core_overlap_fraction": float(
                tool_overlap / max(cad_tool_core.sum(dtype=np.int64), 1)
            ),
        },
        "outputs": {
            name: str(path.resolve()) for name, path in outputs.items()
        },
        "gates": gates,
        "passed_automatic_gates": passed_automatic,
        "passed_for_right_propagation": passed_for_propagation,
        "next_required_step": (
            "Use the accepted seed for independent right video propagation."
            if passed_for_propagation
            else (
                "Inspect 000000-right-seed-overlay.png.  Only after the "
                "right-only boundary and tool exclusion are accepted may "
                "this seed be used for independent right video propagation."
            )
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(outputs["projected_mask"]),
        projected.astype(np.uint8) * 255,
    )
    cv2.imwrite(
        str(outputs["refined_mask"]),
        refined.astype(np.uint8) * 255,
    )
    cv2.imwrite(
        str(outputs["cad_tool_mask"]),
        cad_tool.astype(np.uint8) * 255,
    )
    cv2.imwrite(
        str(outputs["overlay"]),
        overlay(
            image_bgr,
            projected,
            refined,
            cad_tool,
            positive_points,
            negative_points,
        ),
    )
    outputs["report"].write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "[right-seed] selected SAM candidate "
        f"{selected_index}; area={refined_area}; "
        f"projection_iou={selected['iou_with_projection']:.5f}; "
        f"projection_recall={selected['projection_recall']:.5f}",
        flush=True,
    )
    if not passed_automatic:
        raise SystemExit("Right tissue seed automatic gate failed")


if __name__ == "__main__":
    main()
