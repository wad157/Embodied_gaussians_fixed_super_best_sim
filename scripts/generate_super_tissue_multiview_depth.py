#!/usr/bin/env python3
"""Generate left- and right-reference depth for selected SUPER tissue frames.

This is stage C of the multiview tissue rebuild.  FoundationStereo remains the
primary estimator; RAFT-Stereo and flipped-pair consistency only provide
confidence metadata.  No runtime scene asset is modified.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOT = REPO_ROOT / "data/super/grasp5_native"
MULTIVIEW_ROOT = NATIVE_ROOT / "tissue_multiview_v1"
OFFLINE_ROOT = REPO_ROOT / "data/super/grasp5_offline_demo"

if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_super_depth_foundation_timestamped import (  # noqa: E402
    colorized,
    depth_from_disparity,
    disparity_with_lr_consistency,
    five_number,
    image_tensor,
    infer_foundation,
    infer_raft,
    load_foundation_model,
    load_raft_model,
    read_json,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate selected SUPER left/right-reference tissue depths."
    )
    parser.add_argument(
        "--stage-b-report",
        type=Path,
        default=MULTIVIEW_ROOT / "stage_b_report.json",
    )
    parser.add_argument("--rgb-dir", type=Path, default=NATIVE_ROOT / "rgb")
    parser.add_argument(
        "--left-metadata",
        type=Path,
        default=OFFLINE_ROOT / "videos/stereo_left.json",
    )
    parser.add_argument(
        "--right-metadata",
        type=Path,
        default=OFFLINE_ROOT / "videos/stereo_right.json",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=NATIVE_ROOT / "calib_rectified.json",
    )
    parser.add_argument(
        "--foundation-checkpoint",
        type=Path,
        default=REPO_ROOT
        / "third_party/FoundationStereo/pretrained_models/23-51-11/"
        "model_best_bp2.pth",
    )
    parser.add_argument(
        "--raft-checkpoint",
        type=Path,
        default=REPO_ROOT
        / "third_party/Python-SuPer/depth/raft_core/weights/"
        "raft-pretrained.pth",
    )
    parser.add_argument("--output-root", type=Path, default=MULTIVIEW_ROOT)
    parser.add_argument("--sides", default="left,right")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--foundation-iters", type=int, default=32)
    parser.add_argument("--raft-iters", type=int, default=32)
    parser.add_argument("--lr-threshold-px", type=float, default=1.5)
    parser.add_argument("--model-agreement-mm", type=float, default=3.0)
    parser.add_argument("--min-depth-mm", type=float, default=35.0)
    parser.add_argument("--max-depth-mm", type=float, default=250.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def selected_frames(stage_b: dict, side: str) -> list[int]:
    selections = stage_b["coverage"]["selected_temporal_order"]
    frames = [int(item[f"{side}_frame"]) for item in selections]
    if len(frames) != len(set(frames)):
        raise ValueError(f"Selected {side} frames are not unique: {frames}")
    return frames


def bracket_indices(timestamps: np.ndarray, timestamp: float) -> tuple[int, int, float]:
    if timestamp <= float(timestamps[0]):
        return 0, 0, 0.0
    if timestamp >= float(timestamps[-1]):
        last = len(timestamps) - 1
        return last, last, 0.0
    after = int(np.searchsorted(timestamps, timestamp, side="left"))
    before = after - 1
    interval = float(timestamps[after] - timestamps[before])
    if interval <= 0.0:
        raise ValueError("Partner-camera timestamps must be strictly increasing")
    alpha = float((timestamp - timestamps[before]) / interval)
    return before, after, alpha


def infer_oriented(
    *,
    reference_side: str,
    infer,
    model: torch.nn.Module,
    reference: torch.Tensor,
    partner: torch.Tensor,
    iterations: int,
    hierarchical: bool,
    device: torch.device,
    lr_threshold_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if reference_side == "left":
        first = reference
        second = partner
    elif reference_side == "right":
        # Stereo networks expect positive left-reference disparity.  Horizontal
        # flipping and swapping converts a right-reference query into that
        # convention; flip the outputs back to native right-image coordinates.
        first = torch.flip(reference, dims=(3,))
        second = torch.flip(partner, dims=(3,))
    else:
        raise ValueError(reference_side)
    disparity, valid, error = disparity_with_lr_consistency(
        infer,
        model,
        first,
        second,
        iterations,
        hierarchical,
        device,
        lr_threshold_px,
    )
    if reference_side == "right":
        disparity = np.flip(disparity, axis=1).copy()
        valid = np.flip(valid, axis=1).copy()
        error = np.flip(error, axis=1).copy()
    return disparity, valid, error


def interpolate_pair(
    before: np.ndarray,
    after: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if before is after or alpha == 0.0:
        return before.astype(np.float32, copy=True)
    return ((1.0 - alpha) * before + alpha * after).astype(np.float32)


def make_preview(
    reference_bgr: np.ndarray,
    side: str,
    foundation_depth: np.ndarray,
    foundation_valid: np.ndarray,
    raft_depth: np.ndarray,
    raft_valid: np.ndarray,
    model_difference_mm: np.ndarray,
    confidence_level: np.ndarray,
    visibility: np.ndarray,
) -> np.ndarray:
    size = (640, 360)
    panels = [
        cv2.resize(reference_bgr, size, interpolation=cv2.INTER_AREA),
        cv2.resize(
            colorized(foundation_depth, foundation_valid),
            size,
            interpolation=cv2.INTER_AREA,
        ),
        cv2.resize(
            colorized(raft_depth, raft_valid),
            size,
            interpolation=cv2.INTER_AREA,
        ),
        cv2.resize(
            colorized(
                model_difference_mm,
                foundation_valid & raft_valid,
            ),
            size,
            interpolation=cv2.INTER_AREA,
        ),
        cv2.cvtColor(
            cv2.resize(
                confidence_level * np.uint8(85),
                size,
                interpolation=cv2.INTER_NEAREST,
            ),
            cv2.COLOR_GRAY2BGR,
        ),
        cv2.cvtColor(
            cv2.resize(
                visibility.astype(np.uint8) * 255,
                size,
                interpolation=cv2.INTER_NEAREST,
            ),
            cv2.COLOR_GRAY2BGR,
        ),
    ]
    labels = (
        f"{side} reference RGB",
        "Foundation depth",
        "RAFT depth",
        "absolute model difference",
        "confidence 1/2/3",
        "Stage-B visibility",
    )
    for panel, label in zip(panels, labels, strict=True):
        cv2.putText(
            panel,
            label,
            (12, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return np.concatenate(panels, axis=1)


def main() -> None:
    args = parse_args()
    sides = [item.strip() for item in args.sides.split(",") if item.strip()]
    if not sides or any(side not in {"left", "right"} for side in sides):
        raise ValueError(f"Unsupported sides: {sides}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    stage_b = read_json(args.stage_b_report)
    if stage_b.get("status") != "passed_for_stage_c":
        raise RuntimeError("Stage-B masks did not pass for Stage C")
    calibration = read_json(args.calibration)
    K_left = np.asarray(calibration["K_left_rect"], dtype=np.float64)
    K_right = np.asarray(calibration["K_right_rect"], dtype=np.float64)
    if not np.allclose(K_left, K_right, atol=1.0e-9):
        raise RuntimeError(
            "Right-reference flip currently requires identical rectified intrinsics"
        )
    fx = float(K_left[0, 0])
    baseline = float(calibration["baseline_m"])
    cx_delta = float(K_right[0, 2] - K_left[0, 2])
    min_depth = args.min_depth_mm / 1000.0
    max_depth = args.max_depth_mm / 1000.0

    metadata = {
        "left": read_json(args.left_metadata),
        "right": read_json(args.right_metadata),
    }
    timestamps = {
        side: np.asarray(metadata[side]["timestamps"], dtype=np.float64)
        for side in ("left", "right")
    }

    planned_outputs: list[Path] = []
    for side in sides:
        output_dir = args.output_root / f"depth_{side}"
        for frame in selected_frames(stage_b, side):
            planned_outputs.append(output_dir / f"{frame:06d}-depth.npy")
        planned_outputs.append(output_dir / "depth_generation_summary.json")
    collisions = [path for path in planned_outputs if path.exists()]
    if collisions and not args.overwrite:
        raise FileExistsError(
            "Depth outputs already exist; pass --overwrite:\n- "
            + "\n- ".join(str(path) for path in collisions)
        )

    print(f"loading FoundationStereo on {device}")
    foundation, foundation_metadata = load_foundation_model(
        args.foundation_checkpoint, device
    )
    print(f"loading RAFT-Stereo on {device}")
    raft, raft_metadata = load_raft_model(args.raft_checkpoint, device)

    summaries: dict[str, dict] = {}
    for side in sides:
        partner_side = "right" if side == "left" else "left"
        frames = selected_frames(stage_b, side)
        output_dir = args.output_root / f"depth_{side}"
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_reports = []
        for frame in frames:
            started = time.perf_counter()
            timestamp = float(timestamps[side][frame])
            before, after, alpha = bracket_indices(
                timestamps[partner_side], timestamp
            )
            reference_path = args.rgb_dir / f"{frame:06d}-{side}.png"
            before_path = args.rgb_dir / f"{before:06d}-{partner_side}.png"
            after_path = args.rgb_dir / f"{after:06d}-{partner_side}.png"
            reference_bgr = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
            partner_before_bgr = cv2.imread(str(before_path), cv2.IMREAD_COLOR)
            partner_after_bgr = cv2.imread(str(after_path), cv2.IMREAD_COLOR)
            if (
                reference_bgr is None
                or partner_before_bgr is None
                or partner_after_bgr is None
            ):
                raise FileNotFoundError(
                    f"Missing images for {side} frame {frame}: "
                    f"{reference_path}, {before_path}, {after_path}"
                )
            reference_tensor = image_tensor(reference_bgr, device)
            before_tensor = image_tensor(partner_before_bgr, device)
            after_tensor = (
                before_tensor
                if before == after
                else image_tensor(partner_after_bgr, device)
            )

            def pair(infer, model, tensor, iterations, hierarchical):
                return infer_oriented(
                    reference_side=side,
                    infer=infer,
                    model=model,
                    reference=reference_tensor,
                    partner=tensor,
                    iterations=iterations,
                    hierarchical=hierarchical,
                    device=device,
                    lr_threshold_px=args.lr_threshold_px,
                )

            foundation_before, foundation_before_valid, foundation_before_error = (
                pair(
                    infer_foundation,
                    foundation,
                    before_tensor,
                    args.foundation_iters,
                    True,
                )
            )
            if before == after:
                foundation_after = foundation_before
                foundation_after_valid = foundation_before_valid
                foundation_after_error = foundation_before_error
            else:
                (
                    foundation_after,
                    foundation_after_valid,
                    foundation_after_error,
                ) = pair(
                    infer_foundation,
                    foundation,
                    after_tensor,
                    args.foundation_iters,
                    True,
                )
            foundation_disparity = interpolate_pair(
                foundation_before, foundation_after, alpha
            )
            foundation_pair_valid = (
                foundation_before_valid & foundation_after_valid
            )
            foundation_depth, foundation_dense_valid = depth_from_disparity(
                foundation_disparity,
                np.ones(foundation_disparity.shape, dtype=bool),
                fx,
                baseline,
                cx_delta,
                min_depth,
                max_depth,
            )
            foundation_lr_depth, foundation_lr_valid = depth_from_disparity(
                foundation_disparity,
                foundation_pair_valid,
                fx,
                baseline,
                cx_delta,
                min_depth,
                max_depth,
            )

            raft_before, raft_before_valid, raft_before_error = pair(
                infer_raft,
                raft,
                before_tensor,
                args.raft_iters,
                False,
            )
            if before == after:
                raft_after = raft_before
                raft_after_valid = raft_before_valid
                raft_after_error = raft_before_error
            else:
                raft_after, raft_after_valid, raft_after_error = pair(
                    infer_raft,
                    raft,
                    after_tensor,
                    args.raft_iters,
                    False,
                )
            raft_disparity = interpolate_pair(raft_before, raft_after, alpha)
            raft_pair_valid = raft_before_valid & raft_after_valid
            raft_depth, raft_valid = depth_from_disparity(
                raft_disparity,
                raft_pair_valid,
                fx,
                baseline,
                cx_delta,
                min_depth,
                max_depth,
            )

            model_difference_mm = (
                np.abs(foundation_depth - raft_depth) * 1000.0
            )
            high_confidence = (
                foundation_lr_valid
                & raft_valid
                & np.isfinite(model_difference_mm)
                & (model_difference_mm <= args.model_agreement_mm)
            )
            raft_contradiction = (
                foundation_dense_valid
                & raft_valid
                & np.isfinite(model_difference_mm)
                & (model_difference_mm > args.model_agreement_mm)
            )
            confidence_level = np.zeros(
                foundation_disparity.shape, dtype=np.uint8
            )
            confidence_level[foundation_dense_valid] = 1
            confidence_level[foundation_lr_valid] = 2
            confidence_level[high_confidence] = 3

            visibility_path = (
                args.output_root
                / f"masks_{side}/stage_b_visibility_proxy/"
                f"{frame:06d}-visibility-proxy.png"
            )
            visibility_image = cv2.imread(
                str(visibility_path), cv2.IMREAD_GRAYSCALE
            )
            if visibility_image is None:
                raise FileNotFoundError(visibility_path)
            visibility = visibility_image > 0
            if visibility.shape != foundation_depth.shape:
                raise ValueError(
                    f"Visibility/depth shape mismatch for {side} {frame}: "
                    f"{visibility.shape}/{foundation_depth.shape}"
                )
            acceptable = visibility & foundation_dense_valid & (
                (confidence_level >= 2) | ~raft_contradiction
            )

            prefix = output_dir / f"{frame:06d}"
            np.save(f"{prefix}-depth.npy", foundation_depth)
            np.save(f"{prefix}-depth_lr_consistent.npy", foundation_lr_depth)
            np.save(f"{prefix}-raft_depth.npy", raft_depth)
            np.save(f"{prefix}-disparity.npy", foundation_disparity)
            np.savez_compressed(
                f"{prefix}-confidence.npz",
                foundation_dense_valid=foundation_dense_valid,
                foundation_lr_valid=foundation_lr_valid,
                foundation_lr_error_px=np.maximum(
                    foundation_before_error, foundation_after_error
                ),
                foundation_temporal_span_px=np.abs(
                    foundation_after - foundation_before
                ),
                raft_valid=raft_valid,
                raft_lr_error_px=np.maximum(
                    raft_before_error, raft_after_error
                ),
                raft_temporal_span_px=np.abs(raft_after - raft_before),
                absolute_model_difference_mm=model_difference_mm,
                raft_contradiction=raft_contradiction,
                high_confidence=high_confidence,
                confidence_level=confidence_level,
                stage_c_acceptable=acceptable,
            )
            cv2.imwrite(
                f"{prefix}-acceptable-mask.png",
                acceptable.astype(np.uint8) * 255,
            )
            cv2.imwrite(
                f"{prefix}-comparison.png",
                make_preview(
                    reference_bgr,
                    side,
                    foundation_depth,
                    foundation_dense_valid,
                    raft_depth,
                    raft_valid,
                    model_difference_mm,
                    confidence_level,
                    visibility,
                ),
            )

            selected = visibility & foundation_dense_valid
            frame_report = {
                "reference_side": side,
                "reference_frame": frame,
                "reference_timestamp": timestamp,
                "partner_before_frame": before,
                "partner_after_frame": after,
                "partner_before_minus_reference_ms": float(
                    (timestamps[partner_side][before] - timestamp) * 1000.0
                ),
                "partner_after_minus_reference_ms": float(
                    (timestamps[partner_side][after] - timestamp) * 1000.0
                ),
                "temporal_interpolation_alpha": alpha,
                "stage_b_visible_pixels": int(visibility.sum()),
                "foundation_dense_visible_fraction": float(
                    np.mean(foundation_dense_valid[visibility])
                ),
                "foundation_lr_visible_fraction": float(
                    np.mean(foundation_lr_valid[visibility])
                ),
                "high_confidence_visible_fraction": float(
                    np.mean(high_confidence[visibility])
                ),
                "acceptable_visible_fraction": float(
                    np.mean(acceptable[visibility])
                ),
                "foundation_visible_depth_mm_min_p05_p50_p95_max": five_number(
                    foundation_depth[selected] * 1000.0
                ),
                "elapsed_seconds": time.perf_counter() - started,
            }
            frame_reports.append(frame_report)
            print(json.dumps(frame_report, indent=2))

        summary = {
            "stage": "v10_stage_c_timestamped_stereo_depth",
            "reference_side": side,
            "runtime_asset_modified": False,
            "frames": frame_reports,
            "selected_frames": frames,
            "method": {
                "primary": "FoundationStereo dense disparity",
                "confidence": (
                    "flipped-pair consistency plus RAFT-Stereo audit; "
                    "RAFT contradiction excludes a pixel only when RAFT is valid"
                ),
                "timestamp": (
                    f"{partner_side} frames bracket each {side} timestamp; "
                    "disparities are linearly interpolated"
                ),
                "right_reference": (
                    "swap stereo order, flip both inputs, infer positive disparity, "
                    "then flip output back"
                    if side == "right"
                    else None
                ),
            },
            "parameters": {
                "foundation_iters": args.foundation_iters,
                "raft_iters": args.raft_iters,
                "lr_threshold_px": args.lr_threshold_px,
                "model_agreement_mm": args.model_agreement_mm,
                "depth_range_mm": [args.min_depth_mm, args.max_depth_mm],
            },
            "calibration": {
                "path": str(args.calibration.resolve()),
                "sha256": sha256(args.calibration),
                "fx_px": fx,
                "baseline_m": baseline,
                "rectified_intrinsics_identical": True,
            },
            "foundation": foundation_metadata,
            "raft": raft_metadata,
            "inputs": {
                "stage_b_report": str(args.stage_b_report.resolve()),
                "stage_b_report_sha256": sha256(args.stage_b_report),
                "left_metadata_sha256": sha256(args.left_metadata),
                "right_metadata_sha256": sha256(args.right_metadata),
            },
            "passed": all(
                item["acceptable_visible_fraction"] > 0.0
                for item in frame_reports
            ),
        }
        summary_path = output_dir / "depth_generation_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        summaries[side] = summary
        print(f"wrote {summary_path}")

    combined = {
        "stage": "v10_stage_c_stereo_depth_combined",
        "sides": sides,
        "passed": all(summary["passed"] for summary in summaries.values()),
        "summaries": {
            side: str(
                (
                    args.output_root
                    / f"depth_{side}/depth_generation_summary.json"
                ).resolve()
            )
            for side in sides
        },
    }
    combined_path = args.output_root / "stage_c_depth_report.json"
    combined_path.write_text(
        json.dumps(combined, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(combined, indent=2))
    if not combined["passed"]:
        raise SystemExit("Stage-C depth generation failed")


if __name__ == "__main__":
    main()
