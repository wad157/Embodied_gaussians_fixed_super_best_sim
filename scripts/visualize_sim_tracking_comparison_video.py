#!/usr/bin/env python3
"""导出 SIM 固定30点评测的 GT/A/B/C 左相机轨迹对比视频。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm"
METHODS = (
    ("A", "Pure PBD", (255, 72, 72, 255)),
    ("B", "PBD + CoTracker/GT depth", (40, 220, 255, 255)),
    ("C", "B + stiffness update", (105, 255, 90, 255)),
)
GT_COLOR = (255, 255, 255, 255)
PANEL_COLOR = (8, 12, 20, 205)
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--protocol",
        choices=("future_80to20", "reconstruction_7to1"),
        default="future_80to20",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trail-frames", type=int, default=18)
    parser.add_argument("--split-frame", type=int, default=288)
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def columns(all_ids: np.ndarray, selected_ids: np.ndarray, source: Path) -> np.ndarray:
    lookup = {int(node_id): index for index, node_id in enumerate(all_ids)}
    missing = [int(node_id) for node_id in selected_ids if int(node_id) not in lookup]
    if missing:
        raise ValueError(f"{source} 缺少评估点：{missing}")
    return np.asarray([lookup[int(node_id)] for node_id in selected_ids], dtype=np.int64)


def default_output(protocol: str) -> Path:
    suffix = "future80to20" if protocol == "future_80to20" else "reconstruction7to1"
    return (
        ROOT
        / "outputs/sim_global_trackmean_warpfd_cos095_complete_v6/visualizations"
        / f"tracking_comparison_30points_{suffix}.mp4"
    )


def prediction_sources(protocol: str) -> dict[str, Path]:
    return {
        "A": (
            ROOT
            / "outputs/sim_known_grasp_causal_full_pbd30hz_v1"
            / protocol
            / "pbd/artifacts/predicted_trajectories.npz"
        ),
        "B": (
            ROOT
            / "outputs/sim_cotracker_gt_depth_eval30_v2"
            / protocol
            / "pbd_cotracker_gt_depth/artifacts/predicted_trajectories.npz"
        ),
        "C": (
            ROOT
            / "outputs/sim_global_trackmean_warpfd_cos095_complete_v6"
            / protocol
            / "pbd_cotracker_gt_depth_global_distribution"
            / "artifacts/predicted_trajectories.npz"
        ),
    }


def load_inputs(dataset: Path, protocol: str) -> dict[str, object]:
    manifest_path = dataset / "evaluation/evaluation_points_30_non_grasp.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    node_ids = np.asarray(manifest["tissue_node_ids"], dtype=np.int32)
    if len(node_ids) != 30 or int(manifest["count"]) != 30:
        raise ValueError("评估清单必须严格包含30个固定点")

    gt_path = dataset / "ground_truth/trajectories_2d/stereo_left.npz"
    with np.load(gt_path, allow_pickle=False) as source:
        gt_columns = columns(
            np.asarray(source["tissue_node_ids"], dtype=np.int32), node_ids, gt_path
        )
        gt_uv = np.asarray(source["tissue_uv_pixels"][:, gt_columns], dtype=np.float32)
        gt_valid = np.asarray(source["tissue_visible"][:, gt_columns], dtype=bool)
        timestamps = np.asarray(source["timestamps"], dtype=np.float64)

    predictions: dict[str, dict[str, object]] = {}
    frame_indices = np.arange(len(timestamps), dtype=np.int32)
    sources = prediction_sources(protocol)
    for key, label, color in METHODS:
        source_path = sources[key]
        if not source_path.is_file():
            raise FileNotFoundError(f"缺少{key}组轨迹：{source_path}")
        with np.load(source_path, allow_pickle=False) as source:
            if not np.array_equal(source["frame_indices"], frame_indices):
                raise ValueError(f"{source_path} 不是完整连续帧")
            prediction_columns = columns(
                np.asarray(source["tissue_node_ids"], dtype=np.int32),
                node_ids,
                source_path,
            )
            predictions[key] = {
                "label": label,
                "color": color,
                "source": source_path,
                "uv": np.asarray(
                    source["stereo_left_tissue_uv_pixels"][:, prediction_columns],
                    dtype=np.float32,
                ),
                "valid": np.asarray(
                    source["stereo_left_tissue_valid"][:, prediction_columns],
                    dtype=bool,
                ),
            }
    return {
        "manifest": manifest_path,
        "node_ids": node_ids,
        "gt_source": gt_path,
        "gt_uv": gt_uv,
        "gt_valid": gt_valid,
        "timestamps": timestamps,
        "predictions": predictions,
    }


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if FONT_PATH.is_file():
        return ImageFont.truetype(str(FONT_PATH), size=size)
    return ImageFont.load_default()


def finite_point(point: np.ndarray) -> bool:
    return bool(np.isfinite(point).all())


def as_xy(point: np.ndarray) -> tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def draw_trails(
    draw: ImageDraw.ImageDraw,
    uv: np.ndarray,
    valid: np.ndarray,
    frame: int,
    trail_frames: int,
    color: tuple[int, int, int, int],
    width: int,
) -> None:
    first = max(1, frame - trail_frames + 1)
    for node in range(uv.shape[1]):
        for current in range(first, frame + 1):
            previous = current - 1
            if not (valid[previous, node] and valid[current, node]):
                continue
            if not (
                finite_point(uv[previous, node]) and finite_point(uv[current, node])
            ):
                continue
            age = frame - current
            fade = 1.0 - age / max(trail_frames, 1)
            alpha = int(max(18, min(170, color[3] * fade * 0.65)))
            draw.line(
                (as_xy(uv[previous, node]), as_xy(uv[current, node])),
                fill=(*color[:3], alpha),
                width=width,
            )


def draw_marker(
    draw: ImageDraw.ImageDraw,
    key: str,
    point: tuple[int, int],
    color: tuple[int, int, int, int],
    radius: int = 7,
) -> None:
    x, y = point
    if key == "GT":
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=(0, 0, 0, 255),
            width=5,
        )
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=color,
            width=2,
        )
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    elif key == "A":
        draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=4)
        draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=4)
    elif key == "B":
        draw.polygon(
            ((x, y - radius - 1), (x - radius, y + radius), (x + radius, y + radius)),
            outline=color,
            width=4,
        )
    elif key == "C":
        draw.polygon(
            ((x, y - radius - 1), (x - radius, y), (x, y + radius + 1), (x + radius, y)),
            outline=color,
            width=4,
        )


def instantaneous_errors(
    gt_uv: np.ndarray,
    gt_valid: np.ndarray,
    prediction_uv: np.ndarray,
    prediction_valid: np.ndarray,
    frame: int,
) -> float:
    valid = (
        gt_valid[frame]
        & prediction_valid[frame]
        & np.isfinite(gt_uv[frame]).all(axis=1)
        & np.isfinite(prediction_uv[frame]).all(axis=1)
    )
    if not np.any(valid):
        return float("nan")
    return float(
        np.linalg.norm(prediction_uv[frame, valid] - gt_uv[frame, valid], axis=1).mean()
    )


def draw_frame(
    background: Image.Image,
    data: dict[str, object],
    frame: int,
    trail_frames: int,
    split_frame: int,
    protocol: str,
) -> Image.Image:
    image = background.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    gt_uv = data["gt_uv"]
    gt_valid = data["gt_valid"]
    predictions = data["predictions"]
    node_ids = data["node_ids"]
    timestamps = data["timestamps"]
    assert isinstance(gt_uv, np.ndarray) and isinstance(gt_valid, np.ndarray)
    assert isinstance(predictions, dict) and isinstance(node_ids, np.ndarray)
    assert isinstance(timestamps, np.ndarray)

    draw_trails(draw, gt_uv, gt_valid, frame, trail_frames, GT_COLOR, 3)
    for key, _label, color in METHODS:
        prediction = predictions[key]
        draw_trails(
            draw,
            prediction["uv"],
            prediction["valid"],
            frame,
            trail_frames,
            color,
            2,
        )

    # Faint GT-to-prediction connectors expose the instantaneous tracking gap.
    for node in range(len(node_ids)):
        if not gt_valid[frame, node] or not finite_point(gt_uv[frame, node]):
            continue
        gt_point = as_xy(gt_uv[frame, node])
        for key, _label, color in METHODS:
            prediction = predictions[key]
            if prediction["valid"][frame, node] and finite_point(
                prediction["uv"][frame, node]
            ):
                draw.line(
                    (gt_point, as_xy(prediction["uv"][frame, node])),
                    fill=(*color[:3], 68),
                    width=1,
                )

    # Draw method markers first and the hollow GT ring last so coincident points
    # remain distinguishable without shifting any measured coordinate.
    for node in range(len(node_ids)):
        for key, _label, color in METHODS:
            prediction = predictions[key]
            if prediction["valid"][frame, node] and finite_point(
                prediction["uv"][frame, node]
            ):
                draw_marker(draw, key, as_xy(prediction["uv"][frame, node]), color)
        if gt_valid[frame, node] and finite_point(gt_uv[frame, node]):
            point = as_xy(gt_uv[frame, node])
            draw_marker(draw, "GT", point, GT_COLOR)
            draw.text(
                (point[0] + 9, point[1] - 18),
                f"K{node + 1}",
                font=font(16),
                fill=(255, 255, 255, 245),
                stroke_width=2,
                stroke_fill=(0, 0, 0, 220),
            )

    panel = (18, 18, 685, 230)
    draw.rounded_rectangle(panel, radius=16, fill=PANEL_COLOR, outline=(255, 255, 255, 80), width=1)
    if protocol == "future_80to20":
        phase = (
            "参数估计/状态修正（前80%）"
            if frame < split_frame
            else "开环未来预测（后20%，视觉与刚度更新关闭）"
        )
    else:
        phase = (
            "7:1重建 · 留出测评帧（计分）"
            if frame % 8 == 7
            else "7:1重建 · 训练/修正帧（不计分）"
        )
    draw.text(
        (36, 30),
        f"固定30点评测 · 左相机 · {phase}",
        font=font(27),
        fill=(255, 255, 255, 255),
    )
    draw.text(
        (36, 70),
        f"Frame {frame:03d}/359   t={timestamps[frame]:.2f}s   尾迹={trail_frames / 30:.1f}s",
        font=font(22),
        fill=(220, 225, 235, 255),
    )
    legend_y = 113
    draw_marker(draw, "GT", (52, legend_y + 9), GT_COLOR, radius=7)
    draw.text((72, legend_y - 5), "GT", font=font(22), fill=GT_COLOR)
    x = 132
    for key, label, color in METHODS:
        prediction = predictions[key]
        error = instantaneous_errors(
            gt_uv, gt_valid, prediction["uv"], prediction["valid"], frame
        )
        draw_marker(draw, key, (x + 8, legend_y + 9), color, radius=7)
        error_text = "n/a" if not np.isfinite(error) else f"{error:.2f}px"
        draw.text(
            (x + 25, legend_y - 5),
            f"{key}: {label}  ({error_text})",
            font=font(20),
            fill=color,
        )
        x = 132 if key == "A" else x
        if key == "A":
            legend_y += 37
        elif key == "B":
            legend_y += 37
    # A/B/C are deliberately on separate rows; GT remains at the first row.

    # Timeline: exact 80/20 divider or all 45 reconstruction holdouts.
    width, height = image.size
    x0, x1, y = 35, width - 35, height - 31
    current_x = int(round(x0 + (x1 - x0) * frame / (len(timestamps) - 1)))
    draw.line((x0, y, x1, y), fill=(210, 215, 225, 170), width=5)
    if protocol == "future_80to20":
        split_x = int(round(x0 + (x1 - x0) * split_frame / len(timestamps)))
        draw.line((split_x, y - 9, split_x, y + 9), fill=(255, 213, 79, 255), width=4)
        draw.text(
            (split_x - 65, y - 42),
            "80/20分界",
            font=font(18),
            fill=(255, 225, 105, 255),
            stroke_width=1,
            stroke_fill=(0, 0, 0, 220),
        )
    else:
        for holdout in range(7, len(timestamps), 8):
            holdout_x = int(
                round(x0 + (x1 - x0) * holdout / (len(timestamps) - 1))
            )
            draw.line(
                (holdout_x, y - 7, holdout_x, y + 7),
                fill=(255, 213, 79, 210),
                width=2,
            )
        draw.text(
            (x1 - 245, y - 42),
            "橙色刻度 = 7:1留出帧",
            font=font(18),
            fill=(255, 225, 105, 255),
            stroke_width=1,
            stroke_fill=(0, 0, 0, 220),
        )
    current_color = (
        (255, 213, 79, 255)
        if protocol == "reconstruction_7to1" and frame % 8 == 7
        else (255, 255, 255, 255)
    )
    draw.ellipse(
        (current_x - 7, y - 7, current_x + 7, y + 7), fill=current_color
    )
    return Image.alpha_composite(image, overlay).convert("RGB")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = (
        default_output(args.protocol)
        if args.output is None
        else args.output
    ).resolve()
    if args.trail_frames < 1:
        raise ValueError("--trail-frames 必须为正数")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"拒绝覆盖已有视频：{output}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("找不到 ffmpeg")
    data = load_inputs(dataset, args.protocol)
    timestamps = data["timestamps"]
    assert isinstance(timestamps, np.ndarray)
    if not 0 < args.split_frame < len(timestamps):
        raise ValueError("非法的80/20分界帧")
    first_path = dataset / "rgb/stereo_left/000000.png"
    with Image.open(first_path) as first_image:
        width, height = first_image.size
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.encoding.mp4")
    if temporary.exists():
        temporary.unlink()
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        "30",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None and process.stderr is not None
    try:
        for frame in range(len(timestamps)):
            rgb_path = dataset / f"rgb/stereo_left/{frame:06d}.png"
            with Image.open(rgb_path) as background:
                rendered = draw_frame(
                    background,
                    data,
                    frame,
                    args.trail_frames,
                    args.split_frame,
                    args.protocol,
                )
            process.stdin.write(np.asarray(rendered, dtype=np.uint8).tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        if temporary.exists():
            temporary.unlink()
        raise
    if return_code != 0:
        if temporary.exists():
            temporary.unlink()
        raise RuntimeError(f"ffmpeg编码失败：{stderr.strip()}")
    temporary.replace(output)

    predictions = data["predictions"]
    assert isinstance(predictions, dict)
    metadata = {
        "schema": "fixedsuperbest.sim_tracking_comparison_video.v2",
        "video": str(output),
        "sha256": sha256(output),
        "dataset": str(dataset),
        "camera": "stereo_left",
        "frames": len(timestamps),
        "fps": 30,
        "resolution": [width, height],
        "duration_s": len(timestamps) / 30.0,
        "protocol": args.protocol,
        "split_frame": (
            args.split_frame if args.protocol == "future_80to20" else None
        ),
        "reconstruction_holdout": (
            {
                "stride": 8,
                "offset": 7,
                "scored_frames": 45,
                "official_metrics_only_on_holdout_frames": True,
            }
            if args.protocol == "reconstruction_7to1"
            else None
        ),
        "visualization_note": (
            "RGB background and white GT markers are visualization-only; "
            "they are not additional optimizer inputs."
        ),
        "evaluation_points": 30,
        "evaluation_node_ids": data["node_ids"].tolist(),
        "trail_frames": args.trail_frames,
        "marker_encoding": {
            "GT": "white hollow circle",
            "A": "red cross; Pure PBD",
            "B": "cyan triangle; PBD + CoTracker/GT depth",
            "C": "green diamond; B + stiffness update",
        },
        "sources": {
            "manifest": str(data["manifest"]),
            "ground_truth": str(data["gt_source"]),
            **{key: str(value["source"]) for key, value in predictions.items()},
        },
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
