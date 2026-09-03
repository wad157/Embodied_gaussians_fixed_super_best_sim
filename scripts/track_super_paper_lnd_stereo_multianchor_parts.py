#!/usr/bin/env python3
"""Causally track confirmed shaft/distal masks in strict stereo pairs.

Eight manually confirmed, live-SurgicalSAM2 anchor masks restart the upstream
camera predictor every 200 strict stereo pairs.  Shaft and distal tool are
tracked as separate objects in each eye; their union is the pose-optimization
observation.  The original bag, frozen stereo mapping, and frozen rectification
are read directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from build_super_raw_psm_kinematics import load_stereo_calibration
from track_super_p420006_stereo_surgicalsam2 import (
    RAW_ROOT,
    build_predictor,
    pack_mask,
    strict_pair_stream,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
VISUAL_ROOT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_paper_lnd_stereo_multianchor_v2"
)
PAPER_REPO = Path("/Media_HDD/jwshan/wad/online_dvrk_tracking")
CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_s.yaml"
CHECKPOINT_NAME = "sam2.1_hiera_s_endo18.pth"
PART_NAMES = ("shaft", "distal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track separate shaft/distal SurgicalSAM2 objects from confirmed "
            "stereo masks at several causal segment starts."
        )
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
    parser.add_argument("--visual-root", type=Path, default=VISUAL_ROOT)
    parser.add_argument("--paper-repo", type=Path, default=PAPER_REPO)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PAPER_REPO / f"SurgicalSAM2/checkpoints/{CHECKPOINT_NAME}",
    )
    parser.add_argument("--config", default=CONFIG_NAME)
    parser.add_argument(
        "--anchor-dir",
        type=Path,
        default=(
            VISUAL_ROOT / "annotations/live_sam2_multianchor_v4"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VISUAL_ROOT / "surgicalsam2_multianchor_parts_v4（paper_LND+视觉矫正）",
    )
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--cuda-device", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--preview-interval", type=int, default=100)
    parser.add_argument("--max-part-area-fraction", type=float, default=0.20)
    parser.add_argument("--max-stereo-area-ratio", type=float, default=5.0)
    parser.add_argument(
        "--min-shaft-largest-component-fraction",
        type=float,
        default=0.75,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_logits(logits: torch.Tensor) -> np.ndarray:
    values = logits.detach().to(torch.float32).cpu().numpy()
    while values.ndim > 3 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim == 2:
        values = values[None]
    if values.ndim != 3 or len(values) != len(PART_NAMES):
        raise RuntimeError(f"Unexpected two-object logits {tuple(logits.shape)}")
    return values > 0


def mask_shape_metrics(mask: np.ndarray) -> dict[str, Any]:
    area = int(mask.sum())
    if area == 0:
        return {
            "area_px": 0,
            "bbox_xyxy": [-1, -1, -1, -1],
            "largest_component_fraction": 0.0,
            "principal_aspect_ratio": 0.0,
        }
    y_values, x_values = np.where(mask)
    components, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_fraction = (
        1.0 if components <= 1 else float(component_areas.max() / area)
    )
    xy = np.column_stack([x_values, y_values]).astype(np.float64)
    covariance = np.cov(xy, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    aspect = float(
        math.sqrt(
            max(float(eigenvalues[-1]), 1.0e-6)
            / max(float(eigenvalues[0]), 1.0e-6)
        )
    )
    return {
        "area_px": area,
        "bbox_xyxy": [
            int(x_values.min()),
            int(y_values.min()),
            int(x_values.max()),
            int(y_values.max()),
        ],
        "largest_component_fraction": largest_fraction,
        "principal_aspect_ratio": aspect,
    }


def tint_parts(
    image: np.ndarray,
    shaft: np.ndarray,
    distal: np.ndarray,
) -> np.ndarray:
    output = image.copy()
    overlay = image.copy()
    overlay[shaft & ~distal] = (255, 120, 20)
    overlay[distal & ~shaft] = (30, 190, 255)
    overlay[shaft & distal] = (220, 40, 220)
    active = shaft | distal
    output[active] = cv2.addWeighted(
        image[active],
        0.52,
        overlay[active],
        0.48,
        0.0,
    )
    for mask, color in (
        (shaft, (255, 120, 20)),
        (distal, (30, 190, 255)),
    ):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(output, contours, -1, color, 2, cv2.LINE_AA)
    return output


def load_anchor_mask(
    *,
    anchor_dir: Path,
    slot: int,
    side: str,
    part: str,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    path = anchor_dir / f"slot_{slot:04d}_{side}_{part}_mask.png"
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.shape != expected_shape:
        raise RuntimeError(
            f"Anchor mask {path} has shape {mask.shape}, "
            f"expected {expected_shape}"
        )
    return mask > 0


def main() -> None:
    args = parse_args()
    manifest_path = args.visual_root / "prompt_manifest.json"
    state_path = args.anchor_dir / "annotation_state.json"
    for path in (
        args.bag,
        args.calibration,
        args.kinematics,
        manifest_path,
        state_path,
        args.checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("completed") is not True:
        raise RuntimeError("Live SAM2 anchor masks are incomplete")
    if state.get("schema") != "super_paper_lnd_live_sam2_anchor_masks_v4":
        raise RuntimeError("Unexpected live SAM2 anchor schema")
    factor = args.downsample_factor
    source_width, source_height = map(int, manifest["image_size_wh"])
    if source_width % factor or source_height % factor:
        raise ValueError("Source dimensions are not divisible by factor")
    width, height = source_width // factor, source_height // factor
    mask_shape = (height, width)

    with np.load(args.kinematics, allow_pickle=False) as raw:
        left_indices = raw["stereo_left_index"].astype(np.int64)
        right_indices = raw["stereo_right_index"].astype(np.int64)
        left_timestamps = raw["left_timestamps_ros_ns"][
            left_indices
        ].astype(np.int64)
        right_timestamps = raw["right_timestamps_ros_ns"][
            right_indices
        ].astype(np.int64)
    source_pair_count = len(left_indices)
    pair_count = source_pair_count
    if args.max_pairs > 0:
        pair_count = min(pair_count, args.max_pairs)
        left_indices = left_indices[:pair_count]
        right_indices = right_indices[:pair_count]
        left_timestamps = left_timestamps[:pair_count]
        right_timestamps = right_timestamps[:pair_count]

    anchor_by_slot = {
        int(row["strict_pair_slot"]): row
        for row in manifest["anchors"]
        if int(row["strict_pair_slot"]) < pair_count
    }
    if not anchor_by_slot or min(anchor_by_slot) != 0:
        raise RuntimeError("The first processed pair must be an anchor")
    anchor_masks: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    anchor_tips = np.full(
        (len(anchor_by_slot), 2, 2, 2),
        np.nan,
        dtype=np.float32,
    )
    anchor_slots = np.asarray(sorted(anchor_by_slot), dtype=np.int64)
    for anchor_output_index, slot in enumerate(anchor_slots):
        anchor_masks[int(slot)] = {}
        for side_index, side in enumerate(("left", "right")):
            anchor_masks[int(slot)][side] = {
                part: load_anchor_mask(
                    anchor_dir=args.anchor_dir,
                    slot=int(slot),
                    side=side,
                    part=part,
                    expected_shape=mask_shape,
                )
                for part in PART_NAMES
            }
            key = f"slot_{int(slot):04d}_{side}_tips"
            anchor_tips[anchor_output_index, side_index] = np.asarray(
                state["accepted"][key]["tip_points_xy"],
                dtype=np.float32,
            )

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} is not empty; pass --overwrite"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output_dir / "previews"
    preview_dir.mkdir(exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.cuda_device)
    print(
        f"Building four SurgicalSAM2 object streams on cuda:{args.cuda_device}",
        flush=True,
    )
    predictors = {
        side: build_predictor(args.paper_repo, args.config, args.checkpoint)
        for side in ("left", "right")
    }

    packed_width = math.ceil(width * height / 8)
    packed = {
        side: {
            part: np.empty((pair_count, packed_width), dtype=np.uint8)
            for part in (*PART_NAMES, "union")
        }
        for side in ("left", "right")
    }
    metrics = {
        side: {
            part: {
                "area": np.empty(pair_count, dtype=np.int32),
                "largest_component_fraction": np.empty(
                    pair_count, dtype=np.float32
                ),
                "aspect": np.empty(pair_count, dtype=np.float32),
                "bbox": np.empty((pair_count, 4), dtype=np.int32),
            }
            for part in (*PART_NAMES, "union")
        }
        for side in ("left", "right")
    }
    overlap_area = {
        side: np.empty(pair_count, dtype=np.int32)
        for side in ("left", "right")
    }
    quality_valid = np.ones(pair_count, dtype=bool)
    quality_reasons: list[list[str]] = [[] for _ in range(pair_count)]
    partition_quality_valid = np.ones(pair_count, dtype=bool)
    partition_quality_reasons: list[list[str]] = [
        [] for _ in range(pair_count)
    ]
    preview_slots = set(
        [0, pair_count - 1]
        + list(range(0, pair_count, max(1, args.preview_interval)))
        + anchor_slots.tolist()
    )
    preview_panels: list[np.ndarray] = []
    timings: list[float] = []
    decoding_anomalies: list[dict[str, Any]] = []
    calibration = load_stereo_calibration(args.calibration)

    stream = strict_pair_stream(
        bag_path=args.bag,
        calibration=calibration,
        left_indices=left_indices,
        right_indices=right_indices,
        decoding_anomalies=decoding_anomalies,
    )
    for slot, left_full, right_full in stream:
        images_full = {"left": left_full, "right": right_full}
        images_small = {
            side: cv2.resize(
                image,
                (width, height),
                interpolation=cv2.INTER_AREA,
            )
            for side, image in images_full.items()
        }
        start = time.perf_counter()
        output_masks: dict[str, dict[str, np.ndarray]] = {}
        with torch.inference_mode(), torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):
            for side in ("left", "right"):
                predictor = predictors[side]
                if slot in anchor_masks:
                    predictor.frame_idx = 0
                    predictor.load_first_frame(images_small[side])
                    for obj_id, part in enumerate(PART_NAMES):
                        predictor.add_new_mask(
                            frame_idx=0,
                            obj_id=obj_id,
                            mask=anchor_masks[slot][side][part],
                        )
                    masks = np.stack(
                        [
                            anchor_masks[slot][side][part]
                            for part in PART_NAMES
                        ]
                    )
                else:
                    _obj_ids, logits = predictor.track(images_small[side])
                    masks = unpack_logits(logits)
                output_masks[side] = dict(
                    zip(PART_NAMES, masks, strict=True)
                )
        if slot > 0:
            torch.cuda.synchronize(args.cuda_device)
            timings.append(time.perf_counter() - start)

        per_side_area: dict[str, dict[str, int]] = {}
        for side in ("left", "right"):
            shaft = output_masks[side]["shaft"]
            distal = output_masks[side]["distal"]
            union = shaft | distal
            per_side_area[side] = {}
            for part, mask in (
                ("shaft", shaft),
                ("distal", distal),
                ("union", union),
            ):
                packed[side][part][slot] = pack_mask(mask)
                if part in (*PART_NAMES, "union"):
                    shape = mask_shape_metrics(mask)
                    metrics[side][part]["area"][slot] = shape["area_px"]
                    metrics[side][part][
                        "largest_component_fraction"
                    ][slot] = shape["largest_component_fraction"]
                    metrics[side][part]["aspect"][slot] = shape[
                        "principal_aspect_ratio"
                    ]
                    metrics[side][part]["bbox"][slot] = shape["bbox_xyxy"]
                    per_side_area[side][part] = shape["area_px"]
                    if shape["area_px"] > (
                        args.max_part_area_fraction * width * height
                    ):
                        quality_reasons[slot].append(
                            f"{side}_{part}_background_takeover"
                        )
                    if (
                        part == "shaft"
                        and shape["largest_component_fraction"]
                        < args.min_shaft_largest_component_fraction
                    ):
                        partition_quality_reasons[slot].append(
                            f"{side}_shaft_fragmented"
                        )
            overlap_area[side][slot] = int((shaft & distal).sum())
        for part in PART_NAMES:
            left_area = max(per_side_area["left"][part], 1)
            right_area = max(per_side_area["right"][part], 1)
            ratio = max(left_area, right_area) / min(left_area, right_area)
            if ratio > args.max_stereo_area_ratio:
                partition_quality_reasons[slot].append(
                    f"{part}_stereo_area_ratio_{ratio:.3f}"
                )
        left_union_area = max(per_side_area["left"]["union"], 1)
        right_union_area = max(per_side_area["right"]["union"], 1)
        union_ratio = max(left_union_area, right_union_area) / min(
            left_union_area,
            right_union_area,
        )
        if union_ratio > args.max_stereo_area_ratio:
            quality_reasons[slot].append(
                f"union_stereo_area_ratio_{union_ratio:.3f}"
            )
        quality_valid[slot] = len(quality_reasons[slot]) == 0
        partition_quality_valid[slot] = (
            len(partition_quality_reasons[slot]) == 0
        )

        if slot in preview_slots:
            panels = []
            for side in ("left", "right"):
                overlay = tint_parts(
                    images_full[side],
                    cv2.resize(
                        output_masks[side]["shaft"].astype(np.uint8),
                        (source_width, source_height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool),
                    cv2.resize(
                        output_masks[side]["distal"].astype(np.uint8),
                        (source_width, source_height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool),
                )
                cv2.rectangle(
                    overlay,
                    (0, 0),
                    (overlay.shape[1], 38),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    overlay,
                    (
                        f"slot {slot:04d} {side} "
                        f"{'VALID' if quality_valid[slot] else 'INVALID'}"
                    ),
                    (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                scale = 480.0 / source_width
                panels.append(
                    cv2.resize(
                        overlay,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_AREA,
                    )
                )
            preview_panels.append(np.concatenate(panels, axis=1))
        if (
            slot == 0
            or (slot + 1) % 100 == 0
            or slot + 1 == pair_count
        ):
            recent = float(np.mean(timings[-20:])) if timings else float("nan")
            print(
                f"[{slot + 1:04d}/{pair_count:04d}] "
                f"two-part stereo; valid={int(quality_valid[slot])}; "
                f"{recent:.3f}s/pair",
                flush=True,
            )

    output_path = args.output_dir / "stereo_multianchor_part_masks.npz"
    np.savez_compressed(
        output_path,
        schema=np.asarray(
            "super_paper_lnd_stereo_surgicalsam2_multianchor_parts_v4"
        ),
        mask_shape=np.asarray(mask_shape, dtype=np.int64),
        bitorder=np.asarray("little"),
        left_shaft_masks_packbits=packed["left"]["shaft"],
        right_shaft_masks_packbits=packed["right"]["shaft"],
        left_distal_masks_packbits=packed["left"]["distal"],
        right_distal_masks_packbits=packed["right"]["distal"],
        left_masks_packbits=packed["left"]["union"],
        right_masks_packbits=packed["right"]["union"],
        left_shaft_area_px=metrics["left"]["shaft"]["area"],
        right_shaft_area_px=metrics["right"]["shaft"]["area"],
        left_distal_area_px=metrics["left"]["distal"]["area"],
        right_distal_area_px=metrics["right"]["distal"]["area"],
        left_shaft_bbox_xyxy=metrics["left"]["shaft"]["bbox"],
        right_shaft_bbox_xyxy=metrics["right"]["shaft"]["bbox"],
        left_distal_bbox_xyxy=metrics["left"]["distal"]["bbox"],
        right_distal_bbox_xyxy=metrics["right"]["distal"]["bbox"],
        left_shaft_largest_component_fraction=metrics["left"]["shaft"][
            "largest_component_fraction"
        ],
        right_shaft_largest_component_fraction=metrics["right"]["shaft"][
            "largest_component_fraction"
        ],
        left_shaft_principal_aspect_ratio=metrics["left"]["shaft"]["aspect"],
        right_shaft_principal_aspect_ratio=metrics["right"]["shaft"]["aspect"],
        left_overlap_area_px=overlap_area["left"],
        right_overlap_area_px=overlap_area["right"],
        quality_valid=quality_valid,
        partition_quality_valid=partition_quality_valid,
        anchor_slots=anchor_slots,
        anchor_tip_points_xy_at_mask_resolution=anchor_tips,
        stereo_left_index=left_indices,
        stereo_right_index=right_indices,
        left_timestamp_ros_ns=left_timestamps,
        right_timestamp_ros_ns=right_timestamps,
    )
    overview_path = preview_dir / "representative_part_propagation.png"
    if preview_panels and not cv2.imwrite(
        str(overview_path),
        np.concatenate(preview_panels, axis=0),
    ):
        raise RuntimeError(f"Failed to write {overview_path}")
    invalid_rows = [
        {
            "strict_pair_slot": slot,
            "reasons": quality_reasons[slot],
        }
        for slot in range(pair_count)
        if not quality_valid[slot]
    ]
    partition_warning_rows = [
        {
            "strict_pair_slot": slot,
            "reasons": partition_quality_reasons[slot],
        }
        for slot in range(pair_count)
        if not partition_quality_valid[slot]
    ]
    report = {
        "schema": (
            "super_paper_lnd_stereo_surgicalsam2_multianchor_parts_report_v4"
        ),
        "passed": bool(
            pair_count == source_pair_count
            and len(invalid_rows) == 0
            and np.all(quality_valid)
        ),
        "method": {
            "model": "upstream SurgicalSAM2 endoscopic checkpoint",
            "objects_per_eye": list(PART_NAMES),
            "pose_observation": "union(shaft, distal)",
            "causal_segment_start_slots": anchor_slots.tolist(),
            "anchor_input": (
                "human-confirmed live SurgicalSAM2 masks; no dense manual "
                "pixels drawn"
            ),
            "stereo_extension": (
                "independent left/right predictors on strict timestamp pairs"
            ),
        },
        "sequence": {
            "pair_count": pair_count,
            "source_pair_count": source_pair_count,
            "complete_strict_pair_sequence": pair_count == source_pair_count,
            "tracking_image_size_wh": [width, height],
            "mean_seconds_per_pair_excluding_first": (
                float(np.mean(timings)) if timings else None
            ),
            "decoding_anomalies": decoding_anomalies,
        },
        "quality_gate": {
            "policy": (
                "hard validation only; no confidence downweighting and no "
                "invalid mask may enter pose optimization"
            ),
            "max_part_area_fraction": args.max_part_area_fraction,
            "max_stereo_area_ratio": args.max_stereo_area_ratio,
            "min_shaft_largest_component_fraction": (
                args.min_shaft_largest_component_fraction
            ),
            "invalid_count": len(invalid_rows),
            "invalid_rows": invalid_rows,
            "part_partition_warning_count": len(partition_warning_rows),
            "part_partition_warning_rows": partition_warning_rows,
            "part_partition_policy": (
                "record separately from pose validity because one component "
                "can leave one camera while the union silhouette remains "
                "fully usable"
            ),
        },
        "closed_jaw_evidence": {
            "anchor_slots": anchor_slots.tolist(),
            "manual_stereo_tip_points_xy_at_mask_resolution": (
                anchor_tips.tolist()
            ),
            "policy_for_next_stage": (
                "freeze visual correction of raw q7 jaw angle; manual tips "
                "are validation anchors, not permission to reopen the jaw"
            ),
        },
        "inputs": {
            "bag": str(args.bag),
            "bag_sha256": sha256(args.bag),
            "calibration": str(args.calibration),
            "calibration_sha256": sha256(args.calibration),
            "kinematics": str(args.kinematics),
            "kinematics_sha256": sha256(args.kinematics),
            "prompt_manifest": str(manifest_path),
            "prompt_manifest_sha256": sha256(manifest_path),
            "live_anchor_state": str(state_path),
            "live_anchor_state_sha256": sha256(state_path),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
        },
        "outputs": {
            "masks": str(output_path),
            "representative_overview": str(overview_path),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Masks: {output_path}")
    print(f"Overview: {overview_path}")
    print(
        f"Quality gate: {len(invalid_rows)} invalid / {pair_count}; "
        f"passed={report['passed']}"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
