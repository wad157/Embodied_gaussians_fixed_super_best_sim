#!/usr/bin/env python3

"""Diagnose PSM camera-depth error from timestamp-interpolated stereo pairs.

This is deliberately a read-only pose diagnostic: it never rewrites a pose
driver.  For each requested left frame it brackets the left timestamp with two
right frames, estimates left-to-right disparity for both pairs, interpolates
the disparity to the left timestamp, and compares the resulting stereo depth
against the currently selected GUI surface.

Metal and specular highlights make surgical-tool stereo unusually fragile.
Therefore every disparity used here must pass a flipped-pair left/right
consistency check, lie inside the propagated instrument part mask, and land on
a front-facing projected GUI Gaussian.  The final report decides whether the
observed depth shift is stable enough to justify rebuilding projections; it
does not apply that shift itself.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation


REPO = Path(__file__).resolve().parents[1]
TRACK_ROOT = REPO / "data/super/psm_tracking"
RAFT_ROOT = REPO / "third_party/Python-SuPer"
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(RAFT_ROOT)]

# online_dvrk intentionally does not carry this small pure-Python dependency,
# while the existing eg_codex environment does.  Import only this package from
# there rather than mixing the two environments wholesale.
try:
    import opt_einsum  # noqa: F401
except ModuleNotFoundError:
    sys.path.append(
        "/Media_HDD/jwshan/conda_envs/eg_codex/lib/python3.11/site-packages"
    )
    import opt_einsum  # noqa: F401

from depth.raft_core.raft_stereo import RAFTStereo  # noqa: E402
from depth.raft_core.utils.utils import InputPadder  # noqa: E402
from super_psm_tracking_common import (  # noqa: E402
    pose_to_matrix,
    resample_pose_sequence,
)


PART_NAMES = ("body", "jaw_left", "jaw_right")
JAW_LINK_TO_PART = {
    "PSM1_tool_wrist_sca_ee_link_1": "jaw_left",
    "PSM1_tool_wrist_sca_ee_link_2": "jaw_right",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Timestamp-aware SUPER PSM stereo-depth diagnostic."
    )
    parser.add_argument("--frames", default="0,480,800,1120,1280")
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=REPO / "data/super/grasp5_native/rgb",
    )
    parser.add_argument(
        "--left-metadata",
        type=Path,
        default=REPO
        / "data/super/grasp5_offline_demo/videos/stereo_left.json",
    )
    parser.add_argument(
        "--right-metadata",
        type=Path,
        default=REPO
        / "data/super/grasp5_offline_demo/videos/stereo_right.json",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO / "data/super/grasp5_native/calib_rectified.json",
    )
    parser.add_argument(
        "--part-masks",
        type=Path,
        default=TRACK_ROOT / "part_masks_full_sequence/part_masks_full.npz",
    )
    parser.add_argument(
        "--pose-driver",
        type=Path,
        default=TRACK_ROOT / "psm_part_corrected_pose_driver.npz",
    )
    parser.add_argument(
        "--gaussians",
        type=Path,
        default=REPO / "data/super/psm_robot/psm_surface_gaussians.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=RAFT_ROOT
        / "depth/raft_core/weights/raft-pretrained.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TRACK_ROOT / "stereo_depth_diagnostic",
    )
    parser.add_argument("--iters", type=int, default=32)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--roll-offset-deg", type=float, default=-27.0)
    parser.add_argument("--lr-threshold-px", type=float, default=1.5)
    parser.add_argument("--min-depth-mm", type=float, default=45.0)
    parser.add_argument("--max-depth-mm", type=float, default=130.0)
    parser.add_argument("--patch-radius", type=int, default=2)
    parser.add_argument("--save-maps", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    gpu_id = 0 if device.index is None else device.index
    if device.type == "cuda":
        model = torch.nn.DataParallel(
            RAFTStereo(raft_args()), device_ids=[gpu_id], output_device=gpu_id
        )
    else:
        model = torch.nn.DataParallel(RAFTStereo(raft_args()))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return (
        torch.from_numpy(rgb)
        .permute(2, 0, 1)
        .float()[None]
        .to(device)
    )


@torch.inference_mode()
def infer_disparity(
    model: torch.nn.Module,
    first: torch.Tensor,
    second: torch.Tensor,
    iters: int,
    device: torch.device,
) -> np.ndarray:
    padder = InputPadder(first.shape)
    first_pad, second_pad = padder.pad(first, second)
    with torch.amp.autocast(
        "cuda", enabled=device.type == "cuda", dtype=torch.float16
    ):
        _, flow = model(first_pad, second_pad, iters=iters, test_mode=True)
    flow = padder.unpad(flow)
    return (-flow[0, 0]).detach().float().cpu().numpy()


def disparity_with_lr_consistency(
    model: torch.nn.Module,
    left: torch.Tensor,
    right: torch.Tensor,
    iters: int,
    device: torch.device,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    disparity = infer_disparity(model, left, right, iters, device)
    reverse_flipped = infer_disparity(
        model,
        torch.flip(right, dims=(3,)),
        torch.flip(left, dims=(3,)),
        iters,
        device,
    )
    reverse = np.flip(reverse_flipped, axis=1).copy()
    height, width = disparity.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    right_x = grid_x - disparity.astype(np.float32)
    sampled_reverse = cv2.remap(
        reverse.astype(np.float32),
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
    return disparity.astype(np.float32), valid, error.astype(np.float32)


def unpack_frame_masks(asset: np.lib.npyio.NpzFile, frame: int) -> np.ndarray:
    shape = tuple(int(value) for value in asset["mask_shape"])
    pixel_count = int(np.prod(shape))
    unpacked = np.unpackbits(
        asset["masks_packbits"][frame], axis=-1
    )[..., :pixel_count]
    return unpacked.reshape(len(PART_NAMES), *shape).astype(bool)


def apply_roll_offset(poses: np.ndarray, names: list[str], angle_deg: float) -> np.ndarray:
    if math.isclose(angle_deg, 0.0):
        return poses.copy()
    matrices = np.stack([pose_to_matrix(pose) for pose in poses])
    main_index = names.index("PSM1_tool_main_link")
    local_delta = np.eye(4, dtype=np.float64)
    local_delta[:3, :3] = Rotation.from_euler(
        "z", math.radians(angle_deg)
    ).as_matrix()
    world_delta = matrices[main_index] @ local_delta @ np.linalg.inv(
        matrices[main_index]
    )
    for index, name in enumerate(names):
        if name != "PSM1_tool_main_link":
            matrices[index] = world_delta @ matrices[index]
    output = np.empty_like(poses, dtype=np.float64)
    output[:, :3] = matrices[:, :3, 3]
    output[:, 3:] = Rotation.from_matrix(matrices[:, :3, :3]).as_quat()
    return output


def projected_surface(
    local_means: np.ndarray,
    link_ids: np.ndarray,
    asset_names: list[str],
    driver_names: list[str],
    poses: np.ndarray,
    intrinsics: np.ndarray,
) -> dict[str, np.ndarray]:
    points = []
    pixels = []
    predicted_z = []
    parts = []
    for asset_index, name in enumerate(asset_names):
        if name == "PSM1_tool_main_link" or name not in driver_names:
            continue
        selected = link_ids == asset_index
        if not np.any(selected):
            continue
        pose = poses[driver_names.index(name)]
        rotation = Rotation.from_quat(pose[3:]).as_matrix()
        world = local_means[selected] @ rotation.T + pose[:3]
        positive = world[:, 2] > 1.0e-6
        world = world[positive]
        projected = (intrinsics @ (world / world[:, 2:3]).T).T[:, :2]
        points.append(world)
        pixels.append(projected)
        predicted_z.append(world[:, 2])
        parts.extend(
            [JAW_LINK_TO_PART.get(name, "body")] * len(world)
        )
    return {
        "points": np.concatenate(points),
        "pixels": np.concatenate(pixels),
        "predicted_z": np.concatenate(predicted_z),
        "parts": np.asarray(parts),
    }


def front_surface_mask(
    pixels: np.ndarray,
    depth: np.ndarray,
    width: int,
    height: int,
    cell_size: int = 4,
    tolerance_m: float = 0.001,
) -> np.ndarray:
    cell_depth: dict[tuple[int, int], float] = {}
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    for index in np.flatnonzero(inside):
        key = (
            int(pixels[index, 0]) // cell_size,
            int(pixels[index, 1]) // cell_size,
        )
        cell_depth[key] = min(cell_depth.get(key, np.inf), float(depth[index]))
    visible = np.zeros(len(pixels), dtype=bool)
    for index in np.flatnonzero(inside):
        key = (
            int(pixels[index, 0]) // cell_size,
            int(pixels[index, 1]) // cell_size,
        )
        visible[index] = depth[index] <= cell_depth[key] + tolerance_m
    return visible


def five_number(values: np.ndarray) -> list[float]:
    if len(values) == 0:
        return []
    return np.percentile(values, [0, 5, 50, 95, 100]).tolist()


def summarize_samples(samples: list[dict[str, float]]) -> dict[str, object]:
    if not samples:
        return {
            "sample_count": 0,
            "reliable": False,
            "reason": "no samples passed stereo and surface gates",
        }
    stereo_mm = np.asarray([sample["stereo_z_m"] for sample in samples]) * 1000.0
    driver_mm = np.asarray([sample["driver_z_m"] for sample in samples]) * 1000.0
    residual_mm = stereo_mm - driver_mm
    lr_error = np.asarray([sample["lr_error_px"] for sample in samples])
    temporal_span = np.asarray(
        [sample["temporal_disparity_span_px"] for sample in samples]
    )
    median = float(np.median(residual_mm))
    mad = float(np.median(np.abs(residual_mm - median)))
    robust_scale = max(1.4826 * mad, 0.5)
    inlier = np.abs(residual_mm - median) <= 3.0 * robust_scale
    inlier_residual = residual_mm[inlier]
    reliable = (
        int(np.count_nonzero(inlier)) >= 20
        and mad <= 2.0
        and float(np.percentile(lr_error, 95)) <= 1.5
        and float(np.percentile(temporal_span, 95)) <= 35.0
    )
    return {
        "sample_count": len(samples),
        "inlier_count": int(np.count_nonzero(inlier)),
        "driver_depth_mm_min_p05_p50_p95_max": five_number(driver_mm),
        "stereo_depth_mm_min_p05_p50_p95_max": five_number(stereo_mm),
        "stereo_minus_driver_mm_min_p05_p50_p95_max": five_number(residual_mm),
        "stereo_minus_driver_inlier_mm_min_p05_p50_p95_max": five_number(
            inlier_residual
        ),
        "residual_mad_mm": mad,
        "lr_error_px_min_p05_p50_p95_max": five_number(lr_error),
        "temporal_disparity_span_px_min_p05_p50_p95_max": five_number(
            temporal_span
        ),
        "reliable": bool(reliable),
    }


def sample_surface_depths(
    surface: dict[str, np.ndarray],
    masks: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    lr_error: np.ndarray,
    disparity_before: np.ndarray,
    disparity_after: np.ndarray,
    patch_radius: int,
) -> tuple[dict[str, list[dict[str, float]]], np.ndarray]:
    height, width = depth.shape
    visible = front_surface_mask(
        surface["pixels"], surface["predicted_z"], width, height
    )
    dilated_masks = []
    kernel = np.ones((7, 7), dtype=np.uint8)
    for mask in masks:
        resized = cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        dilated_masks.append(cv2.dilate(resized, kernel) > 0)
    mask_by_name = dict(zip(PART_NAMES, dilated_masks, strict=True))
    samples = {name: [] for name in PART_NAMES}
    accepted = np.zeros(len(surface["pixels"]), dtype=bool)
    for index, (pixel, predicted_z, part) in enumerate(
        zip(
            surface["pixels"],
            surface["predicted_z"],
            surface["parts"],
            strict=True,
        )
    ):
        if not visible[index]:
            continue
        u, v = np.rint(pixel).astype(int)
        if not (0 <= u < width and 0 <= v < height):
            continue
        if not mask_by_name[str(part)][v, u]:
            continue
        x0, x1 = max(0, u - patch_radius), min(width, u + patch_radius + 1)
        y0, y1 = max(0, v - patch_radius), min(height, v + patch_radius + 1)
        patch_valid = valid[y0:y1, x0:x1] & mask_by_name[str(part)][
            y0:y1, x0:x1
        ]
        if np.count_nonzero(patch_valid) < 3:
            continue
        stereo_z = float(np.median(depth[y0:y1, x0:x1][patch_valid]))
        sample = {
            "u": float(pixel[0]),
            "v": float(pixel[1]),
            "driver_z_m": float(predicted_z),
            "stereo_z_m": stereo_z,
            "lr_error_px": float(
                np.median(lr_error[y0:y1, x0:x1][patch_valid])
            ),
            "temporal_disparity_span_px": float(
                np.median(
                    np.abs(
                        disparity_after[y0:y1, x0:x1]
                        - disparity_before[y0:y1, x0:x1]
                    )[patch_valid]
                )
            ),
        }
        samples[str(part)].append(sample)
        accepted[index] = True
    return samples, accepted


def residual_color(residual_mm: float) -> tuple[int, int, int]:
    if abs(residual_mm) <= 0.5:
        return (70, 255, 70)
    if residual_mm > 0.0:
        return (40, 40, 255)  # stereo says farther: red
    return (255, 120, 20)  # stereo says closer: blue


def crop_bounds(pixels: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    center = np.median(pixels, axis=0)
    crop_width, crop_height = min(960, width), min(640, height)
    x0 = int(np.clip(round(center[0] - crop_width / 2), 0, width - crop_width))
    y0 = int(np.clip(round(center[1] - crop_height / 2), 0, height - crop_height))
    return x0, y0, x0 + crop_width, y0 + crop_height


def diagnostic_image(
    left: np.ndarray,
    surface: dict[str, np.ndarray],
    samples: dict[str, list[dict[str, float]]],
    depth: np.ndarray,
    valid: np.ndarray,
    frame: int,
    before_index: int,
    after_index: int,
    before_ms: float,
    after_ms: float,
    body_summary: dict[str, object],
) -> np.ndarray:
    overlay = left.copy()
    for pixel in surface["pixels"]:
        u, v = np.rint(pixel).astype(int)
        if 0 <= u < overlay.shape[1] and 0 <= v < overlay.shape[0]:
            cv2.circle(overlay, (u, v), 2, (255, 255, 0), -1, cv2.LINE_AA)
    for part_samples in samples.values():
        for sample in part_samples:
            residual = (sample["stereo_z_m"] - sample["driver_z_m"]) * 1000.0
            cv2.circle(
                overlay,
                (int(round(sample["u"])), int(round(sample["v"]))),
                5,
                residual_color(residual),
                2,
                cv2.LINE_AA,
            )

    clipped = np.clip((depth * 1000.0 - 45.0) / (130.0 - 45.0), 0.0, 1.0)
    clipped = np.nan_to_num(clipped, nan=0.0, posinf=1.0, neginf=0.0)
    colored = cv2.applyColorMap(
        np.rint(clipped * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    depth_view = cv2.addWeighted(left, 0.35, colored, 0.65, 0.0)
    depth_view[~valid] = (left[~valid] * 0.35).astype(np.uint8)

    x0, y0, x1, y1 = crop_bounds(
        surface["pixels"], left.shape[1], left.shape[0]
    )
    crops = [overlay[y0:y1, x0:x1], depth_view[y0:y1, x0:x1]]
    panels = []
    shift = body_summary.get("stereo_minus_driver_inlier_mm_min_p05_p50_p95_max", [])
    shift_text = "n/a" if not shift else f"{float(shift[2]):+.2f} mm"
    labels = (
        (
            f"frame {frame}: cyan=GUI, rings=stereo samples | "
            f"body depth shift {shift_text}"
        ),
        (
            f"stereo depth 45..130mm | right {before_index}/{after_index} "
            f"at {before_ms:+.1f}/{after_ms:+.1f}ms"
        ),
    )
    for crop, label in zip(crops, labels, strict=True):
        panel = np.zeros((crop.shape[0] + 46, crop.shape[1], 3), dtype=np.uint8)
        panel[46:] = crop
        cv2.putText(
            panel,
            label,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)
    return np.hstack(panels)


def main() -> None:
    args = parse_args()
    frames = sorted({int(value) for value in args.frames.split(",") if value})
    if not frames:
        raise ValueError("At least one frame is required")
    if args.iters <= 0 or args.patch_radius < 0:
        raise ValueError("iters must be positive and patch-radius non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    left_meta = read_json(args.left_metadata)
    right_meta = read_json(args.right_metadata)
    calibration = read_json(args.calibration)
    left_timestamps = np.asarray(left_meta["timestamps"], dtype=np.float64)
    right_timestamps = np.asarray(right_meta["timestamps"], dtype=np.float64)
    K_left = np.asarray(calibration["K_left_rect"], dtype=np.float64)
    K_right = np.asarray(calibration["K_right_rect"], dtype=np.float64)
    baseline = float(calibration["baseline_m"])
    fx = float(K_left[0, 0])
    cx_delta = float(K_right[0, 2] - K_left[0, 2])

    with np.load(args.pose_driver, allow_pickle=False) as driver:
        driver_timestamps = driver["timestamps"].astype(np.float64)
        driver_names = driver["link_names"].tolist()
        driver_poses = driver["poses_rect_camera_xyz_xyzw"].astype(np.float64)
    with np.load(args.gaussians, allow_pickle=False) as gaussians:
        local_means = gaussians["means"].astype(np.float64)
        link_ids = gaussians["link_ids"].astype(np.int64)
        asset_names = gaussians["link_names"].tolist()
    mask_asset = np.load(args.part_masks, allow_pickle=False)

    print(f"loading RAFT-Stereo on {device}")
    model = load_model(args.checkpoint, device)
    frame_reports = []
    diagnostic_images = []
    for frame in frames:
        if not (0 <= frame < len(left_timestamps)):
            raise IndexError(frame)
        timestamp = float(left_timestamps[frame])
        after = int(np.searchsorted(right_timestamps, timestamp, side="left"))
        after = int(np.clip(after, 1, len(right_timestamps) - 1))
        before = after - 1
        interval = float(right_timestamps[after] - right_timestamps[before])
        alpha = float((timestamp - right_timestamps[before]) / interval)
        before_ms = float((right_timestamps[before] - timestamp) * 1000.0)
        after_ms = float((right_timestamps[after] - timestamp) * 1000.0)
        print(
            f"frame {frame}: right={before}/{after} "
            f"dt={before_ms:+.2f}/{after_ms:+.2f}ms alpha={alpha:.3f}"
        )

        left_image = cv2.imread(
            str(args.rgb_dir / f"{frame:06d}-left.png"), cv2.IMREAD_COLOR
        )
        right_before = cv2.imread(
            str(args.rgb_dir / f"{before:06d}-right.png"), cv2.IMREAD_COLOR
        )
        right_after = cv2.imread(
            str(args.rgb_dir / f"{after:06d}-right.png"), cv2.IMREAD_COLOR
        )
        if left_image is None or right_before is None or right_after is None:
            raise FileNotFoundError(f"Missing stereo image around frame {frame}")
        left_tensor = image_to_tensor(left_image, device)
        before_tensor = image_to_tensor(right_before, device)
        after_tensor = image_to_tensor(right_after, device)
        disparity_before, valid_before, error_before = (
            disparity_with_lr_consistency(
                model,
                left_tensor,
                before_tensor,
                args.iters,
                device,
                args.lr_threshold_px,
            )
        )
        disparity_after, valid_after, error_after = disparity_with_lr_consistency(
            model,
            left_tensor,
            after_tensor,
            args.iters,
            device,
            args.lr_threshold_px,
        )
        disparity = (
            (1.0 - alpha) * disparity_before + alpha * disparity_after
        ).astype(np.float32)
        valid = valid_before & valid_after
        denominator = disparity + cx_delta
        depth = np.full_like(disparity, np.nan, dtype=np.float32)
        depth[valid] = fx * baseline / denominator[valid]
        valid &= (
            np.isfinite(depth)
            & (depth >= args.min_depth_mm / 1000.0)
            & (depth <= args.max_depth_mm / 1000.0)
        )
        depth[~valid] = np.nan
        lr_error = np.maximum(error_before, error_after)

        poses = resample_pose_sequence(
            driver_timestamps, driver_poses, np.asarray([timestamp])
        )[0].astype(np.float64)
        poses = apply_roll_offset(poses, driver_names, args.roll_offset_deg)
        surface = projected_surface(
            local_means,
            link_ids,
            asset_names,
            driver_names,
            poses,
            K_left,
        )
        masks = unpack_frame_masks(mask_asset, frame)
        samples, accepted = sample_surface_depths(
            surface,
            masks,
            depth,
            valid,
            lr_error,
            disparity_before,
            disparity_after,
            args.patch_radius,
        )
        summaries = {
            part: summarize_samples(part_samples)
            for part, part_samples in samples.items()
        }
        body_summary = summaries["body"]
        print(
            f"  body samples={body_summary['sample_count']} "
            f"reliable={body_summary['reliable']} "
            f"shift={body_summary.get('stereo_minus_driver_inlier_mm_min_p05_p50_p95_max', [])}"
        )
        image = diagnostic_image(
            left_image,
            surface,
            samples,
            depth,
            valid,
            frame,
            before,
            after,
            before_ms,
            after_ms,
            body_summary,
        )
        image_path = args.output_dir / f"frame{frame:06d}_stereo_depth.png"
        cv2.imwrite(str(image_path), image)
        diagnostic_images.append(
            cv2.resize(image, (1440, 515), interpolation=cv2.INTER_AREA)
        )
        if args.save_maps:
            np.savez_compressed(
                args.output_dir / f"frame{frame:06d}_stereo_maps.npz",
                disparity=disparity,
                depth_m=depth,
                valid=valid,
                lr_error_px=lr_error,
            )
        frame_reports.append(
            {
                "left_frame": frame,
                "left_timestamp": timestamp,
                "right_before_frame": before,
                "right_after_frame": after,
                "right_before_minus_left_ms": before_ms,
                "right_after_minus_left_ms": after_ms,
                "temporal_interpolation_alpha": alpha,
                "part_confidence": dict(
                    zip(
                        PART_NAMES,
                        mask_asset["confidence"][frame].astype(float).tolist(),
                        strict=True,
                    )
                ),
                "dense_lr_valid_fraction": float(np.mean(valid)),
                "projected_surface_point_count": len(surface["pixels"]),
                "accepted_surface_point_count": int(np.count_nonzero(accepted)),
                "parts": summaries,
                "diagnostic_image": str(image_path.relative_to(REPO)),
            }
        )

    contact_sheet = np.vstack(diagnostic_images)
    contact_sheet_path = args.output_dir / "contact_sheet.png"
    cv2.imwrite(str(contact_sheet_path), contact_sheet)

    reliable_frames = [
        frame_report
        for frame_report in frame_reports
        if frame_report["parts"]["body"]["reliable"]
    ]
    shifts = np.asarray(
        [
            frame_report["parts"]["body"][
                "stereo_minus_driver_inlier_mm_min_p05_p50_p95_max"
            ][2]
            for frame_report in reliable_frames
        ],
        dtype=np.float64,
    )
    if len(shifts):
        shift_median = float(np.median(shifts))
        shift_mad = float(np.median(np.abs(shifts - shift_median)))
        sign_agreement = float(
            max(np.mean(shifts >= 0.0), np.mean(shifts <= 0.0))
        )
    else:
        shift_median = float("nan")
        shift_mad = float("nan")
        sign_agreement = 0.0
    stable_systematic_shift = bool(
        len(shifts) >= 3
        and abs(shift_median) >= 0.5
        and shift_mad <= 1.5
        and sign_agreement >= 0.8
    )
    if stable_systematic_shift:
        decision = (
            "Stereo finds a stable camera-depth shift. Build a separate stereo-"
            "corrected candidate and regenerate left/right projections before use."
        )
    elif len(shifts) >= 3:
        decision = (
            "Stereo samples are usable but do not show one stable systematic shift. "
            "Do not change the current driver or redo projections yet."
        )
    else:
        decision = (
            "Too few frames passed the metal-surface stereo gates. The diagnosis is "
            "inconclusive; do not change the current driver or projections."
        )
    report = {
        "purpose": "read-only stereo depth diagnosis; no pose driver modified",
        "frames": frames,
        "pose_driver": str(args.pose_driver.relative_to(REPO)),
        "roll_offset_deg": args.roll_offset_deg,
        "stereo_method": (
            "RAFT-Stereo on right frames bracketing each left timestamp; linear "
            "right-correspondence interpolation plus flipped-pair LR consistency"
        ),
        "calibration": {
            "fx_px": fx,
            "baseline_m": baseline,
            "cx_right_minus_left_px": cx_delta,
        },
        "gates": {
            "lr_threshold_px": args.lr_threshold_px,
            "depth_range_mm": [args.min_depth_mm, args.max_depth_mm],
            "minimum_body_inliers_per_frame": 20,
            "maximum_body_residual_mad_mm": 2.0,
            "systematic_shift_min_abs_mm": 0.5,
            "cross_frame_shift_mad_max_mm": 1.5,
            "minimum_reliable_frames": 3,
            "minimum_sign_agreement": 0.8,
        },
        "frame_reports": frame_reports,
        "reliable_body_frame_count": len(reliable_frames),
        "reliable_body_frames": [
            report["left_frame"] for report in reliable_frames
        ],
        "body_shift_mm_per_reliable_frame": shifts.tolist(),
        "body_shift_median_mm": shift_median if np.isfinite(shift_median) else None,
        "body_shift_cross_frame_mad_mm": shift_mad if np.isfinite(shift_mad) else None,
        "body_shift_sign_agreement": sign_agreement,
        "stable_systematic_depth_shift_detected": stable_systematic_shift,
        "projection_regeneration_required": stable_systematic_shift,
        "decision": decision,
        "contact_sheet": str(contact_sheet_path.relative_to(REPO)),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
