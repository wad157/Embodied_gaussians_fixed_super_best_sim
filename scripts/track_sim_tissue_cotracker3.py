#!/usr/bin/env python3
"""Visualize CoTracker3 tissue motion on a tissue-retraction sequence.

This is a diagnostic-only tool.  It tracks a regular grid restricted to the
frame-0 tissue mask and never writes to the simulator or evaluation state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/embodied_gaussians_matplotlib")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = REPO / "data/super/grasp5_offline_demo/videos/stereo_left.mp4"
DEFAULT_MASK = REPO / "data/super/grasp5_native/masks/000000-tissue.png"
DEFAULT_DYNAMIC_MASKS = (
    REPO
    / "data/super/grasp5_native/visual_force_masks_v1/tissue_masks_packbits.npy"
)
DEFAULT_COTRACKER_REPO = Path(
    "/home/jwshan/.cache/torch/hub/facebookresearch_co-tracker_main"
)
DEFAULT_CHECKPOINT = Path(
    "/home/jwshan/.cache/torch/hub/checkpoints/scaled_offline.pth"
)
DEFAULT_OUTPUT = REPO / "outputs/grasp5_cotracker3_motion_left_20260828_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track grasp5 tissue points with CoTracker3 and draw their motion."
        )
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--tissue-mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument(
        "--dynamic-tissue-masks", type=Path, default=DEFAULT_DYNAMIC_MASKS
    )
    parser.add_argument("--cotracker-repo", type=Path, default=DEFAULT_COTRACKER_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--resize-width", type=int, default=640)
    parser.add_argument("--trail-length", type=int, default=30)
    parser.add_argument("--output-fps", type=float, default=0.0)
    return parser.parse_args()


def read_sampled_video(
    path: Path,
    *,
    frame_stride: int,
    max_frames: int,
    resize_width: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_width = resize_width if resize_width > 0 else source_width
    output_height = int(round(source_height * output_width / source_width))
    output_height += output_height % 2

    frames: list[np.ndarray] = []
    source_indices: list[int] = []
    source_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if source_index % frame_stride == 0:
            if (output_width, output_height) != (source_width, source_height):
                frame = cv2.resize(
                    frame,
                    (output_width, output_height),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            source_indices.append(source_index)
            if max_frames > 0 and len(frames) >= max_frames:
                break
        source_index += 1
    capture.release()
    if len(frames) < 2:
        raise RuntimeError("At least two sampled frames are required")
    metadata: dict[str, float | int] = {
        "source_width": source_width,
        "source_height": source_height,
        "source_fps": source_fps,
        "source_frame_count": source_frames,
        "sampled_width": output_width,
        "sampled_height": output_height,
        "sampled_fps": source_fps / frame_stride,
        "sampled_frame_count": len(frames),
        "frame_stride": frame_stride,
    }
    return (
        np.stack(frames, axis=0),
        np.asarray(source_indices, dtype=np.int32),
        metadata,
    )


def point_colors(count: int) -> np.ndarray:
    hues = np.linspace(0, 179, count, endpoint=False, dtype=np.uint8)
    hsv = np.stack(
        (hues, np.full(count, 220, np.uint8), np.full(count, 255, np.uint8)),
        axis=1,
    )[:, None, :]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0, :]


def sample_dynamic_mask_membership(
    packed_path: Path,
    source_frame_indices: np.ndarray,
    tracks_original: np.ndarray,
    visible: np.ndarray,
) -> np.ndarray:
    if not packed_path.exists():
        return visible.copy()
    packed = np.load(packed_path, mmap_mode="r")
    height = packed.shape[1]
    width = packed.shape[2] * 8
    inside = np.zeros_like(visible, dtype=bool)
    for sampled_index, source_index in enumerate(source_frame_indices):
        xy = np.rint(tracks_original[sampled_index]).astype(np.int64)
        x = np.clip(xy[:, 0], 0, width - 1)
        y = np.clip(xy[:, 1], 0, height - 1)
        byte_values = packed[source_index, y, x // 8]
        bit_offsets = 7 - (x % 8)
        inside[sampled_index] = (
            np.right_shift(byte_values, bit_offsets) & 1
        ).astype(bool)
    return inside & visible


def last_visible_displacements(
    tracks: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    point_count = tracks.shape[1]
    displacement = np.full(point_count, np.nan, dtype=np.float32)
    last_indices = np.full(point_count, -1, dtype=np.int32)
    for point_index in range(point_count):
        indices = np.flatnonzero(valid[:, point_index])
        if indices.size == 0:
            continue
        last_index = int(indices[-1])
        last_indices[point_index] = last_index
        displacement[point_index] = float(
            np.linalg.norm(
                tracks[last_index, point_index] - tracks[0, point_index]
            )
        )
    return displacement, last_indices


def compute_statistics(
    tracks: np.ndarray,
    visible: np.ndarray,
    tissue_valid: np.ndarray,
) -> dict[str, np.ndarray | float | int]:
    displacement = np.linalg.norm(tracks - tracks[0:1], axis=-1)
    step = np.zeros_like(displacement)
    step[1:] = np.linalg.norm(tracks[1:] - tracks[:-1], axis=-1)
    both_step_valid = tissue_valid.copy()
    both_step_valid[1:] &= tissue_valid[:-1]

    def per_frame(values: np.ndarray, mask: np.ndarray, quantile: float) -> np.ndarray:
        result = np.full(values.shape[0], np.nan, dtype=np.float32)
        for frame_index in range(values.shape[0]):
            active = values[frame_index, mask[frame_index]]
            if active.size:
                result[frame_index] = float(np.quantile(active, quantile))
        return result

    median_displacement = per_frame(displacement, tissue_valid, 0.5)
    p90_displacement = per_frame(displacement, tissue_valid, 0.9)
    median_step = per_frame(step, both_step_valid, 0.5)
    p90_step = per_frame(step, both_step_valid, 0.9)
    finite_peak = np.where(np.isfinite(median_displacement), median_displacement, -1)
    peak_index = int(np.argmax(finite_peak))
    return {
        "displacement": displacement,
        "step": step,
        "median_displacement": median_displacement,
        "p90_displacement": p90_displacement,
        "median_step": median_step,
        "p90_step": p90_step,
        "visible_fraction": visible.mean(axis=1).astype(np.float32),
        "tissue_valid_fraction": tissue_valid.mean(axis=1).astype(np.float32),
        "peak_index": peak_index,
    }


def draw_track_video(
    frames_rgb: np.ndarray,
    tracks: np.ndarray,
    visible: np.ndarray,
    tissue_valid: np.ndarray,
    source_indices: np.ndarray,
    median_displacement: np.ndarray,
    output_path: Path,
    fps: float,
    trail_length: int,
) -> None:
    height, width = frames_rgb.shape[1:3]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")
    colors = point_colors(tracks.shape[1])
    for frame_index, frame_rgb in enumerate(frames_rgb):
        canvas = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        start = max(0, frame_index - trail_length + 1)
        for point_index in range(tracks.shape[1]):
            color = tuple(int(v) for v in colors[point_index])
            previous: tuple[int, int] | None = None
            for trail_index in range(start, frame_index + 1):
                if not visible[trail_index, point_index]:
                    previous = None
                    continue
                point = tracks[trail_index, point_index]
                current = (int(round(point[0])), int(round(point[1])))
                if previous is not None:
                    cv2.line(canvas, previous, current, color, 1, cv2.LINE_AA)
                previous = current
            if visible[frame_index, point_index]:
                point = tracks[frame_index, point_index]
                center = (int(round(point[0])), int(round(point[1])))
                if tissue_valid[frame_index, point_index]:
                    cv2.circle(canvas, center, 2, color, -1, cv2.LINE_AA)
                else:
                    cv2.circle(canvas, center, 2, (140, 140, 140), 1, cv2.LINE_AA)
        active = int(tissue_valid[frame_index].sum())
        displacement = float(median_displacement[frame_index])
        label = (
            f"CoTracker3 | source frame {int(source_indices[frame_index])} | "
            f"tissue-valid {active}/{tracks.shape[1]} | "
            f"median displacement {displacement:.1f}px"
        )
        cv2.rectangle(canvas, (0, 0), (width, 30), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            label,
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(canvas)
    writer.release()


def draw_peak_vectors(
    frames_rgb: np.ndarray,
    tracks: np.ndarray,
    tissue_valid: np.ndarray,
    peak_index: int,
    source_indices: np.ndarray,
    output_path: Path,
) -> None:
    first = cv2.cvtColor(frames_rgb[0], cv2.COLOR_RGB2BGR)
    peak = cv2.cvtColor(frames_rgb[peak_index], cv2.COLOR_RGB2BGR)
    overlay = peak.copy()
    valid = tissue_valid[0] & tissue_valid[peak_index]
    magnitudes = np.linalg.norm(tracks[peak_index] - tracks[0], axis=1)
    scale_max = max(float(np.quantile(magnitudes[valid], 0.95)), 1.0)
    normalized = np.clip(magnitudes / scale_max, 0.0, 1.0)
    colormap = cv2.applyColorMap(
        np.rint(normalized * 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO
    )[:, 0]
    for point_index in np.flatnonzero(valid):
        start = tuple(np.rint(tracks[0, point_index]).astype(int))
        end = tuple(np.rint(tracks[peak_index, point_index]).astype(int))
        color = tuple(int(v) for v in colormap[point_index])
        cv2.arrowedLine(overlay, start, end, color, 1, cv2.LINE_AA, tipLength=0.15)
        cv2.circle(overlay, end, 2, color, -1, cv2.LINE_AA)
    combined = np.concatenate((first, overlay), axis=1)
    cv2.putText(
        combined,
        "frame 0",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        combined,
        f"peak median displacement: source frame {int(source_indices[peak_index])}",
        (first.shape[1] + 12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), combined)


def draw_motion_timeline(
    source_indices: np.ndarray,
    source_fps: float,
    statistics: dict[str, np.ndarray | float | int],
    output_path: Path,
) -> None:
    time_s = source_indices.astype(np.float64) / source_fps
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(time_s, statistics["median_displacement"], label="median")
    axes[0].plot(time_s, statistics["p90_displacement"], label="p90")
    axes[0].set_ylabel("from frame 0 (px)")
    axes[0].set_title("CoTracker3 grasp5 tissue motion")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(time_s, statistics["median_step"], label="median")
    axes[1].plot(time_s, statistics["p90_step"], label="p90")
    axes[1].set_ylabel("step motion (px)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    axes[2].plot(time_s, statistics["visible_fraction"], label="CoTracker visible")
    axes[2].plot(
        time_s,
        statistics["tissue_valid_fraction"],
        label="visible and in dynamic tissue mask",
    )
    axes[2].set_ylabel("point fraction")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("frame-stride must be positive")
    if args.grid_size < 2:
        raise ValueError("grid-size must be at least 2")
    for required in (
        args.video,
        args.tissue_mask,
        args.cotracker_repo,
        args.checkpoint,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames_rgb, source_indices, metadata = read_sampled_video(
        args.video,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        resize_width=args.resize_width,
    )
    source_mask = cv2.imread(str(args.tissue_mask), cv2.IMREAD_GRAYSCALE)
    if source_mask is None:
        raise RuntimeError(f"Could not read tissue mask: {args.tissue_mask}")
    sampled_mask = cv2.resize(
        source_mask,
        (frames_rgb.shape[2], frames_rgb.shape[1]),
        interpolation=cv2.INTER_NEAREST,
    )

    sys.path.insert(0, str(args.cotracker_repo))
    from cotracker.predictor import CoTrackerPredictor

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = CoTrackerPredictor(
        checkpoint=str(args.checkpoint),
        offline=True,
        v2=False,
        window_len=60,
    ).to(device)
    video_tensor = (
        torch.from_numpy(frames_rgb)
        .permute(0, 3, 1, 2)[None]
        .float()
        .to(device)
    )
    mask_tensor = torch.from_numpy(sampled_mask)[None, None].float().to(device)
    with torch.inference_mode():
        predicted_tracks, predicted_visibility = model(
            video_tensor,
            grid_size=args.grid_size,
            grid_query_frame=0,
            backward_tracking=False,
            segm_mask=mask_tensor,
        )
    tracks_sampled = predicted_tracks[0].float().cpu().numpy()
    visible = predicted_visibility[0].bool().cpu().numpy()
    del video_tensor, model, predicted_tracks, predicted_visibility
    if device.type == "cuda":
        torch.cuda.empty_cache()

    tracks_original = tracks_sampled.copy()
    tracks_original[..., 0] *= float(metadata["source_width"]) / float(
        metadata["sampled_width"]
    )
    tracks_original[..., 1] *= float(metadata["source_height"]) / float(
        metadata["sampled_height"]
    )
    tissue_valid = sample_dynamic_mask_membership(
        args.dynamic_tissue_masks,
        source_indices,
        tracks_original,
        visible,
    )
    sampled_statistics = compute_statistics(
        tracks_sampled, visible, tissue_valid
    )
    original_statistics = compute_statistics(
        tracks_original, visible, tissue_valid
    )
    peak_index = int(original_statistics["peak_index"])
    output_fps = (
        args.output_fps
        if args.output_fps > 0.0
        else float(metadata["sampled_fps"])
    )

    np.savez_compressed(
        args.output_dir / "tracks.npz",
        source_frame_indices=source_indices,
        tracks_sampled_px=tracks_sampled.astype(np.float32),
        tracks_original_px=tracks_original.astype(np.float32),
        visibility=visible,
        dynamic_tissue_valid=tissue_valid,
    )
    draw_track_video(
        frames_rgb,
        tracks_sampled,
        visible,
        tissue_valid,
        source_indices,
        sampled_statistics["median_displacement"],
        args.output_dir / "cotracker3_tissue_tracks.mp4",
        output_fps,
        args.trail_length,
    )
    draw_peak_vectors(
        frames_rgb,
        tracks_sampled,
        tissue_valid,
        peak_index,
        source_indices,
        args.output_dir / "peak_motion_vectors.png",
    )
    draw_motion_timeline(
        source_indices,
        float(metadata["source_fps"]),
        original_statistics,
        args.output_dir / "motion_timeline.png",
    )

    last_displacement, _ = last_visible_displacements(
        tracks_original, tissue_valid
    )
    valid_last = np.isfinite(last_displacement)
    summary = {
        "purpose": "sim RGB motion observation for flow-depth state estimation",
        "model": "CoTracker3 scaled offline",
        "model_source": "https://github.com/facebookresearch/co-tracker",
        "video": str(args.video.resolve()),
        "frame0_tissue_mask": str(args.tissue_mask.resolve()),
        "dynamic_tissue_masks": str(args.dynamic_tissue_masks.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "grid_size": args.grid_size,
        "tracked_point_count": int(tracks_sampled.shape[1]),
        "metadata": metadata,
        "peak_sampled_frame": peak_index,
        "peak_source_frame": int(source_indices[peak_index]),
        "peak_time_s": float(source_indices[peak_index] / metadata["source_fps"]),
        "peak_median_displacement_original_px": float(
            original_statistics["median_displacement"][peak_index]
        ),
        "peak_p90_displacement_original_px": float(
            original_statistics["p90_displacement"][peak_index]
        ),
        "mean_cotracker_visibility_fraction": float(
            np.mean(original_statistics["visible_fraction"])
        ),
        "mean_dynamic_tissue_valid_fraction": float(
            np.mean(original_statistics["tissue_valid_fraction"])
        ),
        "points_with_a_tissue_valid_observation": int(valid_last.sum()),
        "median_last_valid_displacement_original_px": float(
            np.nanmedian(last_displacement)
        ),
        "p90_last_valid_displacement_original_px": float(
            np.nanquantile(last_displacement, 0.9)
        ),
        "outputs": {
            "tracks": "tracks.npz",
            "video": "cotracker3_tissue_tracks.mp4",
            "peak_vectors": "peak_motion_vectors.png",
            "timeline": "motion_timeline.png",
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
