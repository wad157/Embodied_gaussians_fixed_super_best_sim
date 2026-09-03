#!/usr/bin/env python3
"""Propagate a SUPER tissue seed with camera-parameterized SAM2.1 video mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_DIR = REPO_ROOT / "data/super/grasp5_native"
DATASET_DIR = REPO_ROOT / "data/super/grasp5_offline_demo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Propagate SUPER tissue masks.")
    parser.add_argument(
        "--camera-side",
        choices=("left", "right"),
        default="left",
    )
    parser.add_argument(
        "--rgb-dir", type=Path, default=NATIVE_DIR / "rgb"
    )
    parser.add_argument(
        "--manual-mask",
        type=Path,
        default=NATIVE_DIR / "masks/000000-tissue.png",
    )
    parser.add_argument(
        "--camera-metadata",
        type=Path,
        default=DATASET_DIR / "videos/stereo_left.json",
    )
    parser.add_argument(
        "--timestamps-npy",
        type=Path,
        default=None,
        help="Optional native timestamp array overriding camera metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=NATIVE_DIR / "visual_force_masks_v1",
    )
    parser.add_argument(
        "--frames-dir", type=Path, default=Path("/tmp/super_sam2_tissue_jpeg")
    )
    parser.add_argument("--model-id", default="facebook/sam2.1-hiera-large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--vos-optimized",
        action="store_true",
        help="Enable optional torch.compile VOS optimization (very slow first compile).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace this output directory's propagation arrays/report.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_jpeg_frames(
    rgb_dir: Path,
    frames_dir: Path,
    frame_count: int,
    quality: int,
    camera_side: str,
) -> list[Path]:
    source_frames = sorted(rgb_dir.glob(f"*-{camera_side}.png"))[:frame_count]
    if len(source_frames) != frame_count:
        raise RuntimeError(
            f"Expected {frame_count} {camera_side} RGB frames, "
            f"found {len(source_frames)}"
        )
    frames_dir.mkdir(parents=True, exist_ok=True)
    expected = [frames_dir / f"{index:06d}.jpg" for index in range(frame_count)]
    for index, (source, target) in enumerate(zip(source_frames, expected)):
        if target.exists():
            continue
        with Image.open(source) as image:
            image.convert("RGB").save(target, quality=quality, subsampling=0)
        if index % 100 == 0:
            print(f"[sam2-mask] prepared JPEG {index + 1}/{frame_count}", flush=True)
    return expected


def binary_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum(dtype=np.int64)
    union = np.logical_or(left, right).sum(dtype=np.int64)
    return float(intersection / union) if union > 0 else 1.0


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda"):
        raise ValueError("SAM2.1-large propagation requires a CUDA device")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    with args.camera_metadata.open("r") as file:
        camera_metadata = json.load(file)
    if args.timestamps_npy is None:
        timestamps = np.asarray(
            camera_metadata["timestamps"], dtype=np.float64
        )
    else:
        timestamps = np.load(args.timestamps_npy).astype(np.float64)
    frame_count = len(timestamps)
    if args.max_frames > 0:
        frame_count = min(frame_count, args.max_frames)
        timestamps = timestamps[:frame_count]
    width, height = map(int, camera_metadata["resolution"])

    with Image.open(args.manual_mask) as image:
        manual_mask = np.asarray(image.convert("L")) > 0
    if manual_mask.shape != (height, width):
        raise ValueError(
            f"Manual mask shape {manual_mask.shape} != video {(height, width)}"
        )

    run_frames_dir = (
        args.frames_dir / f"{args.camera_side}_n{frame_count}"
    )
    prepare_jpeg_frames(
        args.rgb_dir,
        run_frames_dir,
        frame_count,
        args.jpeg_quality,
        args.camera_side,
    )
    managed_outputs = [
        args.output_dir / "tissue_masks_packbits.npy",
        args.output_dir / "timestamps.npy",
        args.output_dir / "areas.npy",
        args.output_dir / "temporal_iou.npy",
        args.output_dir / "report.json",
    ]
    if not args.overwrite:
        collisions = [path for path in managed_outputs if path.exists()]
        if collisions:
            raise FileExistsError(
                "Propagation outputs already exist:\n- "
                + "\n- ".join(str(path) for path in collisions)
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from sam2.build_sam import build_sam2_video_predictor_hf

    if torch.cuda.get_device_capability(0)[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    predictor = build_sam2_video_predictor_hf(
        args.model_id,
        device=args.device,
        vos_optimized=args.vos_optimized,
    )
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        state = predictor.init_state(
            str(run_frames_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
            async_loading_frames=True,
        )
        predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=manual_mask)

        packed_width = (width + 7) // 8
        packed_path = args.output_dir / "tissue_masks_packbits.npy"
        packed = np.lib.format.open_memmap(
            packed_path,
            mode="w+",
            dtype=np.uint8,
            shape=(frame_count, height, packed_width),
        )
        areas = np.zeros(frame_count, dtype=np.int64)
        temporal_iou = np.ones(frame_count, dtype=np.float32)
        seen = np.zeros(frame_count, dtype=bool)
        previous = None
        first_prediction_iou = None
        for frame_index, object_ids, mask_logits in predictor.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=frame_count,
        ):
            if frame_index >= frame_count:
                continue
            object_ids_list = list(object_ids)
            if 1 not in object_ids_list:
                raise RuntimeError(f"Tissue object missing at frame {frame_index}")
            object_index = object_ids_list.index(1)
            prediction = (
                mask_logits[object_index, 0] > 0.0
            ).detach().cpu().numpy()
            if frame_index == 0:
                first_prediction_iou = binary_iou(prediction, manual_mask)
                prediction = manual_mask
            packed[frame_index] = np.packbits(prediction, axis=1)
            areas[frame_index] = prediction.sum(dtype=np.int64)
            if previous is not None:
                temporal_iou[frame_index] = binary_iou(prediction, previous)
            previous = prediction
            seen[frame_index] = True
            if frame_index % 50 == 0 or frame_index == frame_count - 1:
                print(
                    f"[sam2-mask] propagated {frame_index + 1}/{frame_count}; "
                    f"area={areas[frame_index]}",
                    flush=True,
                )
        packed.flush()

    if not seen.all():
        missing = np.flatnonzero(~seen)[:20].tolist()
        raise RuntimeError(f"SAM2 did not return every requested frame: {missing}")
    np.save(args.output_dir / "timestamps.npy", timestamps)
    np.save(args.output_dir / "areas.npy", areas)
    np.save(args.output_dir / "temporal_iou.npy", temporal_iou)
    area_ratio = areas.astype(np.float64) / max(float(areas[0]), 1.0)
    nonempty = areas > 0
    gates = {
        "all_frames_returned": bool(seen.all()),
        "first_prediction_matches_manual": bool(
            first_prediction_iou is not None and first_prediction_iou > 0.98
        ),
        "no_empty_masks": bool(nonempty.all()),
        "finite_statistics": bool(
            np.isfinite(area_ratio).all() and np.isfinite(temporal_iou).all()
        ),
    }
    report = {
        "stage": "visual_force_tissue_mask_propagation",
        "camera_side": args.camera_side,
        "model": args.model_id,
        "device": args.device,
        "frame_count": frame_count,
        "resolution_wh": [width, height],
        "manual_mask": {
            "path": str(args.manual_mask.resolve()),
            "sha256": sha256(args.manual_mask),
            "area": int(areas[0]),
        },
        "timestamp_source": {
            "camera_metadata": str(args.camera_metadata.resolve()),
            "camera_metadata_sha256": sha256(args.camera_metadata),
            "timestamps_npy": (
                str(args.timestamps_npy.resolve())
                if args.timestamps_npy is not None
                else None
            ),
            "timestamps_npy_sha256": (
                sha256(args.timestamps_npy)
                if args.timestamps_npy is not None
                else None
            ),
        },
        "first_prediction_iou_with_manual": first_prediction_iou,
        "area_ratio": {
            "min": float(area_ratio.min()),
            "p05": float(np.quantile(area_ratio, 0.05)),
            "median": float(np.median(area_ratio)),
            "p95": float(np.quantile(area_ratio, 0.95)),
            "max": float(area_ratio.max()),
        },
        "temporal_iou": {
            "min": float(temporal_iou[1:].min()) if frame_count > 1 else 1.0,
            "p05": float(np.quantile(temporal_iou[1:], 0.05))
            if frame_count > 1
            else 1.0,
            "median": float(np.median(temporal_iou[1:]))
            if frame_count > 1
            else 1.0,
        },
        "outputs": {
            "packed_masks": str(packed_path.resolve()),
            "timestamps": str((args.output_dir / "timestamps.npy").resolve()),
            "areas": str((args.output_dir / "areas.npy").resolve()),
            "temporal_iou": str(
                (args.output_dir / "temporal_iou.npy").resolve()
            ),
        },
        "gates": gates,
        "passed": bool(all(gates.values())),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Tissue mask propagation gate failed")


if __name__ == "__main__":
    main()
