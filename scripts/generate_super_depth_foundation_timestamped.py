#!/usr/bin/env python3

"""Generate timestamp-synchronized SUPER depth with two stereo models.

FoundationStereo is the primary estimator. RAFT-Stereo is evaluated on the
same timestamp-bracketed image pairs and is used only for comparison and a
conservative cross-model confidence mask. Existing depth and scene assets are
never modified by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[1]
FOUNDATION_ROOT = REPO / "third_party/FoundationStereo"
RAFT_ROOT = REPO / "third_party/Python-SuPer"
NATIVE_ROOT = REPO / "data/super/grasp5_native"
OFFLINE_ROOT = REPO / "data/super/grasp5_offline_demo"

# The official DINOv2 fallback uses PyTorch SDPA when xFormers is disabled.
os.environ.setdefault("XFORMERS_DISABLED", "1")
sys.path[:0] = [str(FOUNDATION_ROOT), str(RAFT_ROOT)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Timestamp-synchronized FoundationStereo depth with RAFT-Stereo "
            "comparison and confidence checks."
        )
    )
    parser.add_argument("--frames", default="0,1,2,3,4")
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
        "--tissue-mask",
        type=Path,
        default=NATIVE_ROOT / "masks/000000-tissue.png",
    )
    parser.add_argument(
        "--ground-mask",
        type=Path,
        default=NATIVE_ROOT / "masks/000000-ground.png",
    )
    parser.add_argument(
        "--foundation-checkpoint",
        type=Path,
        default=FOUNDATION_ROOT
        / "pretrained_models/23-51-11/model_best_bp2.pth",
    )
    parser.add_argument(
        "--raft-checkpoint",
        type=Path,
        default=RAFT_ROOT / "depth/raft_core/weights/raft-pretrained.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=NATIVE_ROOT / "depth_v4_foundation_dense_timestamped",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--foundation-iters", type=int, default=32)
    parser.add_argument("--raft-iters", type=int, default=32)
    parser.add_argument(
        "--foundation-hierarchical",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lr-threshold-px", type=float, default=1.5)
    parser.add_argument("--model-agreement-mm", type=float, default=3.0)
    parser.add_argument("--min-depth-mm", type=float, default=35.0)
    parser.add_argument("--max-depth-mm", type=float, default=250.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def five_number(values: np.ndarray) -> list[float] | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return np.percentile(values, [0, 5, 50, 95, 100]).tolist()


def image_tensor(image_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return (
        torch.from_numpy(image_rgb)
        .permute(2, 0, 1)
        .float()[None]
        .contiguous()
        .to(device)
    )


def load_foundation_model(
    checkpoint: Path, device: torch.device
) -> tuple[torch.nn.Module, dict]:
    from core.foundation_stereo import FoundationStereo

    config_path = checkpoint.parent / "cfg.yaml"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    cfg = OmegaConf.load(config_path)
    if "vit_size" not in cfg:
        cfg.vit_size = "vitl"

    # All EdgeNeXt weights are present in the FoundationStereo checkpoint.
    # Avoid an unrelated timm network download while constructing the model.
    import timm

    create_model = timm.create_model

    def create_model_without_pretraining(name: str, *args: object, **kwargs: object):
        kwargs["pretrained"] = False
        return create_model(name, *args, **kwargs)

    torch.hub._validate_not_a_forked_repo = lambda *args, **kwargs: True
    timm.create_model = create_model_without_pretraining
    try:
        model = FoundationStereo(cfg)
    finally:
        timm.create_model = create_model

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "global_step": int(state.get("global_step", -1)),
        "epoch": int(state.get("epoch", -1)),
    }
    del state
    return model, metadata


def raft_args() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_dims=[128, 128, 128],
        corr_levels=4,
        corr_radius=4,
        shared_backbone=False,
        n_downsample=2,
        context_norm="batch",
        slow_fast_gru=False,
        n_gru_layers=3,
        corr_implementation="reg",
        mixed_precision=True,
    )


def load_raft_model(
    checkpoint: Path, device: torch.device
) -> tuple[torch.nn.Module, dict]:
    from depth.raft_core.raft_stereo import RAFTStereo

    model = torch.nn.DataParallel(
        RAFTStereo(raft_args()),
        device_ids=[device.index or 0],
        output_device=device.index or 0,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
    }


@torch.inference_mode()
def infer_foundation(
    model: torch.nn.Module,
    first: torch.Tensor,
    second: torch.Tensor,
    iterations: int,
    hierarchical: bool,
    device: torch.device,
) -> np.ndarray:
    from core.utils.utils import InputPadder

    height, width = first.shape[-2:]
    padder = InputPadder(first.shape, divis_by=32, force_square=False)
    first_pad, second_pad = padder.pad(first, second)
    with torch.amp.autocast(
        "cuda", enabled=device.type == "cuda", dtype=torch.float16
    ):
        if hierarchical:
            disparity = model.run_hierachical(
                first_pad,
                second_pad,
                iters=iterations,
                test_mode=True,
                small_ratio=0.5,
            )
        else:
            disparity = model(
                first_pad, second_pad, iters=iterations, test_mode=True
            )
    disparity = padder.unpad(disparity.float())
    return disparity.detach().cpu().numpy().reshape(height, width).astype(np.float32)


@torch.inference_mode()
def infer_raft(
    model: torch.nn.Module,
    first: torch.Tensor,
    second: torch.Tensor,
    iterations: int,
    _hierarchical: bool,
    device: torch.device,
) -> np.ndarray:
    from depth.raft_core.utils.utils import InputPadder

    padder = InputPadder(first.shape)
    first_pad, second_pad = padder.pad(first, second)
    with torch.amp.autocast(
        "cuda", enabled=device.type == "cuda", dtype=torch.float16
    ):
        _, flow = model(first_pad, second_pad, iters=iterations, test_mode=True)
    flow = padder.unpad(flow)
    return (-flow[0, 0]).detach().float().cpu().numpy().astype(np.float32)


def disparity_with_lr_consistency(
    infer: Callable[..., np.ndarray],
    model: torch.nn.Module,
    left: torch.Tensor,
    right: torch.Tensor,
    iterations: int,
    hierarchical: bool,
    device: torch.device,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    disparity = infer(
        model, left, right, iterations, hierarchical, device
    )
    reverse_flipped = infer(
        model,
        torch.flip(right, dims=(3,)),
        torch.flip(left, dims=(3,)),
        iterations,
        hierarchical,
        device,
    )
    reverse = np.flip(reverse_flipped, axis=1).copy()
    height, width = disparity.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    right_x = grid_x - disparity
    sampled_reverse = cv2.remap(
        reverse,
        right_x,
        grid_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )
    error = np.abs(disparity - sampled_reverse)
    valid = (
        np.isfinite(disparity)
        & np.isfinite(sampled_reverse)
        & (disparity > 0.0)
        & (right_x >= 0.0)
        & (right_x < width - 1)
        & (error <= threshold)
    )
    return disparity, valid, error.astype(np.float32)


def depth_from_disparity(
    disparity: np.ndarray,
    valid: np.ndarray,
    fx: float,
    baseline: float,
    cx_delta: float,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    denominator = disparity + cx_delta
    output_valid = valid & np.isfinite(denominator) & (denominator > 0.0)
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    depth[output_valid] = fx * baseline / denominator[output_valid]
    output_valid &= (
        np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    )
    depth[~output_valid] = np.nan
    return depth, output_valid


def colorized(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    preview = np.zeros(values.shape, dtype=np.uint8)
    selected = values[valid & np.isfinite(values)]
    if len(selected):
        lo, hi = np.percentile(selected, [2, 98])
        finite_values = np.where(np.isfinite(values), values, lo)
        preview = np.clip(
            (finite_values - lo) / max(float(hi - lo), 1e-9) * 255.0,
            0,
            255,
        ).astype(np.uint8)
        preview[~valid] = 0
    return cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)


def make_preview(
    left: np.ndarray,
    foundation_depth: np.ndarray,
    foundation_valid: np.ndarray,
    raft_depth: np.ndarray,
    raft_valid: np.ndarray,
    agreement_mm: np.ndarray,
    confidence_level: np.ndarray,
) -> np.ndarray:
    size = (960, 540)
    rgb = cv2.resize(left, size, interpolation=cv2.INTER_AREA)
    foundation = cv2.resize(
        colorized(foundation_depth, foundation_valid),
        size,
        interpolation=cv2.INTER_AREA,
    )
    raft = cv2.resize(
        colorized(raft_depth, raft_valid), size, interpolation=cv2.INTER_AREA
    )
    agreement_valid = foundation_valid & raft_valid
    difference = cv2.resize(
        colorized(agreement_mm, agreement_valid),
        size,
        interpolation=cv2.INTER_AREA,
    )
    confidence = cv2.resize(
        (confidence_level * np.uint8(85)),
        size,
        interpolation=cv2.INTER_NEAREST,
    )
    confidence = cv2.cvtColor(confidence, cv2.COLOR_GRAY2BGR)
    panels = [rgb, foundation, raft, difference, confidence]
    labels = (
        "left RGB",
        "Foundation depth",
        "RAFT depth",
        "absolute difference",
        "confidence level 1/2/3",
    )
    for panel, label in zip(panels, labels, strict=True):
        cv2.putText(
            panel,
            label,
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return np.concatenate(panels, axis=1)


def region_report(
    mask: np.ndarray,
    foundation_depth: np.ndarray,
    foundation_dense_valid: np.ndarray,
    foundation_lr_valid: np.ndarray,
    raft_depth: np.ndarray,
    raft_valid: np.ndarray,
    high_confidence: np.ndarray,
) -> dict:
    count = int(np.count_nonzero(mask))
    overlap = mask & foundation_dense_valid & raft_valid
    difference_mm = np.abs(foundation_depth - raft_depth) * 1000.0
    return {
        "pixel_count": count,
        "foundation_dense_valid_fraction": float(
            np.mean(foundation_dense_valid[mask])
        ),
        "foundation_lr_consistent_fraction": float(
            np.mean(foundation_lr_valid[mask])
        ),
        "raft_valid_fraction": float(np.mean(raft_valid[mask])),
        "high_confidence_fraction": float(np.mean(high_confidence[mask])),
        "foundation_depth_mm_min_p05_p50_p95_max": five_number(
            foundation_depth[mask & foundation_dense_valid] * 1000.0
        ),
        "raft_depth_mm_min_p05_p50_p95_max": five_number(
            raft_depth[mask & raft_valid] * 1000.0
        ),
        "foundation_minus_raft_mm_min_p05_p50_p95_max": five_number(
            (foundation_depth - raft_depth)[overlap] * 1000.0
        ),
        "absolute_model_difference_mm_min_p05_p50_p95_max": five_number(
            difference_mm[overlap]
        ),
    }


def main() -> None:
    args = parse_args()
    frames = [int(value) for value in args.frames.split(",") if value.strip()]
    if not frames:
        raise ValueError("At least one frame is required")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    left_metadata = read_json(args.left_metadata)
    right_metadata = read_json(args.right_metadata)
    calibration = read_json(args.calibration)
    left_timestamps = np.asarray(left_metadata["timestamps"], dtype=np.float64)
    right_timestamps = np.asarray(right_metadata["timestamps"], dtype=np.float64)
    K_left = np.asarray(calibration["K_left_rect"], dtype=np.float64)
    K_right = np.asarray(calibration["K_right_rect"], dtype=np.float64)
    fx = float(K_left[0, 0])
    baseline = float(calibration["baseline_m"])
    cx_delta = float(K_right[0, 2] - K_left[0, 2])
    min_depth = args.min_depth_mm / 1000.0
    max_depth = args.max_depth_mm / 1000.0

    tissue = cv2.imread(str(args.tissue_mask), cv2.IMREAD_GRAYSCALE)
    ground = cv2.imread(str(args.ground_mask), cv2.IMREAD_GRAYSCALE)
    if tissue is None or ground is None:
        raise FileNotFoundError("The existing frame-0 SAM2 masks are missing")
    tissue_mask = tissue > 0
    ground_mask = ground > 0

    print(f"loading FoundationStereo on {device}")
    foundation, foundation_metadata = load_foundation_model(
        args.foundation_checkpoint, device
    )
    print(f"loading RAFT-Stereo on {device}")
    raft, raft_metadata = load_raft_model(args.raft_checkpoint, device)

    frame_reports = []
    for frame in frames:
        if not 0 <= frame < len(left_timestamps):
            raise IndexError(frame)
        started = time.perf_counter()
        timestamp = float(left_timestamps[frame])
        after = int(np.searchsorted(right_timestamps, timestamp, side="left"))
        after = int(np.clip(after, 1, len(right_timestamps) - 1))
        before = after - 1
        interval = float(right_timestamps[after] - right_timestamps[before])
        if interval <= 0.0:
            raise ValueError("Right-camera timestamps are not strictly increasing")
        alpha = float((timestamp - right_timestamps[before]) / interval)

        left = cv2.imread(
            str(args.rgb_dir / f"{frame:06d}-left.png"), cv2.IMREAD_COLOR
        )
        right_before = cv2.imread(
            str(args.rgb_dir / f"{before:06d}-right.png"), cv2.IMREAD_COLOR
        )
        right_after = cv2.imread(
            str(args.rgb_dir / f"{after:06d}-right.png"), cv2.IMREAD_COLOR
        )
        if left is None or right_before is None or right_after is None:
            raise FileNotFoundError(f"Missing stereo images around left frame {frame}")
        if left.shape[:2] != tissue_mask.shape:
            raise ValueError(
                f"SAM2 mask {tissue_mask.shape} does not match left image {left.shape[:2]}"
            )

        left_tensor = image_tensor(left, device)
        before_tensor = image_tensor(right_before, device)
        after_tensor = image_tensor(right_after, device)

        foundation_before, foundation_before_valid, foundation_before_error = (
            disparity_with_lr_consistency(
                infer_foundation,
                foundation,
                left_tensor,
                before_tensor,
                args.foundation_iters,
                args.foundation_hierarchical,
                device,
                args.lr_threshold_px,
            )
        )
        foundation_after, foundation_after_valid, foundation_after_error = (
            disparity_with_lr_consistency(
                infer_foundation,
                foundation,
                left_tensor,
                after_tensor,
                args.foundation_iters,
                args.foundation_hierarchical,
                device,
                args.lr_threshold_px,
            )
        )
        foundation_disparity = (
            (1.0 - alpha) * foundation_before + alpha * foundation_after
        ).astype(np.float32)
        foundation_pair_valid = foundation_before_valid & foundation_after_valid
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

        raft_before, raft_before_valid, raft_before_error = (
            disparity_with_lr_consistency(
                infer_raft,
                raft,
                left_tensor,
                before_tensor,
                args.raft_iters,
                False,
                device,
                args.lr_threshold_px,
            )
        )
        raft_after, raft_after_valid, raft_after_error = (
            disparity_with_lr_consistency(
                infer_raft,
                raft,
                left_tensor,
                after_tensor,
                args.raft_iters,
                False,
                device,
                args.lr_threshold_px,
            )
        )
        raft_disparity = (
            (1.0 - alpha) * raft_before + alpha * raft_after
        ).astype(np.float32)
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

        agreement_mm = np.abs(foundation_depth - raft_depth) * 1000.0
        high_confidence = (
            foundation_lr_valid
            & raft_valid
            & np.isfinite(agreement_mm)
            & (agreement_mm <= args.model_agreement_mm)
        )
        raft_contradiction = (
            foundation_dense_valid
            & raft_valid
            & np.isfinite(agreement_mm)
            & (agreement_mm > args.model_agreement_mm)
        )
        high_confidence_depth = foundation_depth.copy()
        high_confidence_depth[~high_confidence] = np.nan
        confidence_level = np.zeros(foundation_disparity.shape, dtype=np.uint8)
        confidence_level[foundation_dense_valid] = 1
        confidence_level[foundation_lr_valid] = 2
        confidence_level[high_confidence] = 3

        prefix = args.output_dir / f"{frame:06d}"
        np.save(f"{prefix}-depth.npy", foundation_depth)
        np.save(f"{prefix}-depth_lr_consistent.npy", foundation_lr_depth)
        np.save(f"{prefix}-depth_high_confidence.npy", high_confidence_depth)
        np.save(f"{prefix}-disparity.npy", foundation_disparity)
        np.save(f"{prefix}-raft_depth.npy", raft_depth)
        np.save(f"{prefix}-raft_disparity.npy", raft_disparity)
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
            raft_lr_error_px=np.maximum(raft_before_error, raft_after_error),
            raft_temporal_span_px=np.abs(raft_after - raft_before),
            absolute_model_difference_mm=agreement_mm,
            raft_contradiction=raft_contradiction,
            high_confidence=high_confidence,
            confidence_level=confidence_level,
        )
        cv2.imwrite(
            f"{prefix}-dense_valid_mask.png",
            foundation_dense_valid.astype(np.uint8) * 255,
        )
        cv2.imwrite(
            f"{prefix}-lr_consistent_mask.png",
            foundation_lr_valid.astype(np.uint8) * 255,
        )
        cv2.imwrite(
            f"{prefix}-high_confidence_mask.png",
            high_confidence.astype(np.uint8) * 255,
        )
        cv2.imwrite(
            f"{prefix}-confidence_level.png",
            confidence_level * np.uint8(85),
        )
        cv2.imwrite(
            f"{prefix}-comparison.png",
            make_preview(
                left,
                foundation_depth,
                foundation_dense_valid,
                raft_depth,
                raft_valid,
                agreement_mm,
                confidence_level,
            ),
        )

        overlap = foundation_dense_valid & raft_valid
        report = {
            "left_frame": frame,
            "region_mask_source_frame": 0,
            "region_mask_usage": (
                "exact SAM2 semantics for frame 0; fixed-pixel ROI diagnostic "
                "only for adjacent frames"
            ),
            "left_timestamp": timestamp,
            "right_before_frame": before,
            "right_after_frame": after,
            "right_before_minus_left_ms": float(
                (right_timestamps[before] - timestamp) * 1000.0
            ),
            "right_after_minus_left_ms": float(
                (right_timestamps[after] - timestamp) * 1000.0
            ),
            "temporal_interpolation_alpha": alpha,
            "foundation_dense_valid_fraction": float(
                np.mean(foundation_dense_valid)
            ),
            "foundation_lr_consistent_fraction": float(
                np.mean(foundation_lr_valid)
            ),
            "raft_valid_fraction": float(np.mean(raft_valid)),
            "both_models_valid_fraction": float(np.mean(overlap)),
            "raft_contradiction_fraction": float(np.mean(raft_contradiction)),
            "high_confidence_fraction": float(np.mean(high_confidence)),
            "foundation_depth_mm_min_p05_p50_p95_max": five_number(
                foundation_depth[foundation_dense_valid] * 1000.0
            ),
            "raft_depth_mm_min_p05_p50_p95_max": five_number(
                raft_depth[raft_valid] * 1000.0
            ),
            "foundation_minus_raft_mm_min_p05_p50_p95_max": five_number(
                (foundation_depth - raft_depth)[overlap] * 1000.0
            ),
            "absolute_model_difference_mm_min_p05_p50_p95_max": five_number(
                agreement_mm[overlap]
            ),
            "foundation_temporal_span_px_min_p05_p50_p95_max": five_number(
                np.abs(foundation_after - foundation_before)[
                    foundation_pair_valid
                ]
            ),
            "raft_temporal_span_px_min_p05_p50_p95_max": five_number(
                np.abs(raft_after - raft_before)[raft_pair_valid]
            ),
            "regions": {
                "tissue": region_report(
                    tissue_mask,
                    foundation_depth,
                    foundation_dense_valid,
                    foundation_lr_valid,
                    raft_depth,
                    raft_valid,
                    high_confidence,
                ),
                "ground": region_report(
                    ground_mask,
                    foundation_depth,
                    foundation_dense_valid,
                    foundation_lr_valid,
                    raft_depth,
                    raft_valid,
                    high_confidence,
                ),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        frame_reports.append(report)
        print(json.dumps(report, indent=2))

    summary = {
        "purpose": (
            "FoundationStereo primary depth on timestamp-bracketed right frames; "
            "RAFT-Stereo is comparison/confidence only"
        ),
        "foundation_repository": {
            "path": str(FOUNDATION_ROOT),
            "commit": "6e8806816b533e4d13ddbb95ffa907b797060a62",
        },
        "foundation": foundation_metadata,
        "raft": raft_metadata,
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "device": str(device),
        },
        "calibration": {
            "path": str(args.calibration),
            "sha256": sha256(args.calibration),
            "fx_px": fx,
            "baseline_m": baseline,
            "cx_right_minus_left_px": cx_delta,
        },
        "timestamp_method": (
            "right frames bracketing each left timestamp; linear disparity "
            "interpolation to the left timestamp"
        ),
        "confidence_method": (
            "Dense Foundation disparity is always retained inside the configured "
            "positive depth range. Confidence never deletes geometry: level 1 "
            "is dense Foundation depth, level 2 additionally passes flipped-pair "
            "LR consistency, and level 3 additionally has a valid RAFT estimate "
            f"with absolute depth agreement <= {args.model_agreement_mm} mm"
        ),
        "parameters": {
            "frames": frames,
            "foundation_iters": args.foundation_iters,
            "foundation_hierarchical": args.foundation_hierarchical,
            "raft_iters": args.raft_iters,
            "lr_threshold_px": args.lr_threshold_px,
            "model_agreement_mm": args.model_agreement_mm,
            "depth_range_mm": [args.min_depth_mm, args.max_depth_mm],
        },
        "sam2_masks": {
            "tissue": str(args.tissue_mask),
            "tissue_sha256": sha256(args.tissue_mask),
            "ground": str(args.ground_mask),
            "ground_sha256": sha256(args.ground_mask),
            "shape": list(tissue_mask.shape),
            "compatible_with_frame_0": True,
            "adjacent_frame_usage": (
                "fixed-pixel ROI diagnostics only; not treated as propagated "
                "semantic segmentation"
            ),
        },
        "outputs": {
            "directory": str(args.output_dir),
            "primary_dense_depth_pattern": "NNNNNN-depth.npy",
            "recommended_initial_modeling_pattern": "NNNNNN-depth.npy",
            "lr_consistent_audit_depth_pattern": (
                "NNNNNN-depth_lr_consistent.npy"
            ),
            "strict_audit_depth_pattern": (
                "NNNNNN-depth_high_confidence.npy"
            ),
        },
        "frames": frame_reports,
    }
    summary_path = args.output_dir / "depth_generation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
