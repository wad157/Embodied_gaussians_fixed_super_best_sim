#!/usr/bin/env python3
"""Export TRACE full-scene renders and query-anchored SIM trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_functional
from PIL import Image

from arguments import PipelineParams
from gaussian_renderer import render

from common import (
    RenderCamera,
    classify_motion,
    deformation_at_time,
    internal_world_from_camera,
    load_calibration,
    load_trained_model,
)
from protocol import (
    ADAPTER_VERSION,
    CAMERAS,
    DATASETS,
    INTERNAL_UNITS_PER_METER,
    QUERY_FRAME,
    TRACK_DECODER_VERSION,
    UPSTREAM_COMMIT,
    dataset_spec,
    max_observed_time,
    normalized_time,
    project_world_points,
    rendering_frames,
    resolve_dataset,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    pipeline = PipelineParams(parser)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--physics-code", type=int, default=16)
    parser.add_argument("--light", action="store_true")
    parser.add_argument("--freegave", action="store_true")
    parser.add_argument("--alpha-threshold", type=float, default=1.0 / 255.0)
    args = parser.parse_args()
    args.source_path = resolve_dataset(REPO_ROOT, args.dataset_key, args.source_path)
    args.model_path = args.model_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists():
        parser.error("Refusing to overwrite export directory: {}".format(args.output_dir))
    if args.freegave:
        parser.error("Formal TRACE baseline excludes the optional FreeGave extension")
    return args, pipeline.extract(args)


def make_camera(calibration: dict, frame: int, frames: int) -> RenderCamera:
    return RenderCamera(
        calibration["K"],
        calibration["X_WC_ros_optical"],
        calibration["resolution_wh"],
        normalized_time(frame, frames),
    )


def save_rgb(path: Path, tensor: torch.Tensor) -> None:
    array = (
        tensor.detach().clamp(0.0, 1.0).mul(255.0).add(0.5).byte()
        .permute(1, 2, 0).cpu().numpy()
    )
    Image.fromarray(array, mode="RGB").save(str(path))


def save_alpha(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().clamp(0.0, 1.0).mul(255.0).add(0.5).byte().cpu().numpy()
    Image.fromarray(array, mode="L").save(str(path))


def sample_chw(image: torch.Tensor, pixels_uv: np.ndarray) -> torch.Tensor:
    height, width = image.shape[-2:]
    pixels = torch.as_tensor(pixels_uv, dtype=image.dtype, device=image.device)
    grid = pixels.clone()
    grid[:, 0] = 2.0 * grid[:, 0] / float(width - 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / float(height - 1) - 1.0
    sampled = torch_functional.grid_sample(
        image.unsqueeze(0),
        grid.view(1, 1, -1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[0, :, 0, :].transpose(0, 1)


def evaluation_queries(dataset: Path):
    manifest_path = dataset / "evaluation" / "evaluation_points_30_non_grasp.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    node_ids = np.asarray(manifest["tissue_node_ids"], dtype=np.int64)
    reference_path = dataset / "ground_truth" / "trajectories_2d" / "stereo_left.npz"
    with np.load(str(reference_path), allow_pickle=False) as reference:
        all_ids = np.asarray(reference["tissue_node_ids"], dtype=np.int64)
        lookup = {int(node): index for index, node in enumerate(all_ids)}
        columns = np.asarray([lookup[int(node)] for node in node_ids], dtype=np.int64)
        visible = np.asarray(reference["tissue_visible"])[QUERY_FRAME, columns]
        pixels = np.asarray(reference["tissue_uv_pixels"])[QUERY_FRAME, columns]
    if node_ids.shape != (30,) or not bool(np.all(visible)) or not np.isfinite(pixels).all():
        raise ValueError("The frozen frame-0 query must contain 30 visible finite points")
    return node_ids.astype(np.int32), pixels.astype(np.float32), manifest_path


def render_state(camera, gaussians, pipe, background, state, override_color=None):
    d_xyz, d_rotation, d_scaling = state
    return render(
        camera,
        gaussians,
        pipe,
        background,
        d_xyz,
        d_rotation,
        d_scaling,
        False,
        override_color=override_color,
    )


def export_tracks(
    dataset,
    frames,
    calibration,
    gaussians,
    deform,
    moving,
    fps,
    pipe,
    background,
    alpha_threshold,
    output,
):
    node_ids, query_pixels, manifest_path = evaluation_queries(dataset)
    query_camera = make_camera(calibration["stereo_left"], QUERY_FRAME, frames)
    query_state = deformation_at_time(gaussians, deform, moving, 0.0, fps)
    ones = torch.ones_like(gaussians.get_xyz)
    query_render = render_state(
        query_camera, gaussians, pipe, background, query_state, override_color=ones
    )
    query_alpha = sample_chw(query_render["render"][0:1], query_pixels)[:, 0]
    query_depth = sample_chw(query_render["depth"], query_pixels)[:, 0]
    query_depth = query_depth / query_alpha.clamp_min(1.0e-8)
    query_valid = (
        (query_alpha > float(alpha_threshold))
        & torch.isfinite(query_depth)
        & (query_depth > 0.0)
    )
    if int(query_valid.sum().item()) != len(node_ids):
        raise RuntimeError(
            "TRACE query alpha/depth coverage is {}/30".format(int(query_valid.sum().item()))
        )
    intrinsic = torch.as_tensor(
        calibration["stereo_left"]["K"], dtype=query_depth.dtype, device=query_depth.device
    )
    pixels = torch.as_tensor(query_pixels, dtype=query_depth.dtype, device=query_depth.device)
    camera_xyz = torch.stack(
        (
            (pixels[:, 0] - intrinsic[0, 2]) / intrinsic[0, 0] * query_depth,
            (pixels[:, 1] - intrinsic[1, 2]) / intrinsic[1, 1] * query_depth,
            query_depth,
        ),
        dim=1,
    )
    world_from_camera = torch.as_tensor(
        internal_world_from_camera(calibration["stereo_left"]["X_WC_ros_optical"]),
        dtype=query_depth.dtype,
        device=query_depth.device,
    )
    query_world = torch.einsum(
        "ij,nj->ni",
        world_from_camera[:3],
        torch_functional.pad(camera_xyz, (0, 1), value=1.0),
    )
    query_centers = gaussians.get_xyz + query_state[0]
    feature_scale = (
        query_centers.max(dim=0).values - query_centers.min(dim=0).values
    ).max().clamp_min(1.0e-6)
    positions = np.full((frames, len(node_ids), 3), np.nan, dtype=np.float32)
    for frame in range(frames):
        state = deformation_at_time(
            gaussians, deform, moving, normalized_time(frame, frames), fps
        )
        target_centers = gaussians.get_xyz + state[0]
        encoded = (target_centers - query_centers) / feature_scale
        displacement_image = render_state(
            query_camera,
            gaussians,
            pipe,
            background,
            query_state,
            override_color=encoded,
        )["render"]
        displacement = (
            sample_chw(displacement_image, query_pixels)
            / query_alpha[:, None].clamp_min(1.0e-8)
            * feature_scale
        )
        decoded = query_world + displacement
        decoded[~query_valid] = torch.nan
        positions[frame] = (
            decoded.detach().cpu().numpy().astype(np.float32)
            / INTERNAL_UNITS_PER_METER
        )
        if (frame + 1) % 20 == 0 or frame + 1 == frames:
            print("[track] {:04d}/{:04d}".format(frame + 1, frames), flush=True)
    uv, valid, camera_xyz_m = {}, {}, {}
    for name in CAMERAS:
        current_uv = np.empty((frames, len(node_ids), 2), dtype=np.float32)
        current_valid = np.empty((frames, len(node_ids)), dtype=bool)
        current_xyz = np.empty((frames, len(node_ids), 3), dtype=np.float32)
        for frame in range(frames):
            current_uv[frame], current_valid[frame], current_xyz[frame] = project_world_points(
                positions[frame],
                calibration[name]["K"],
                calibration[name]["X_WC_ros_optical"],
                calibration[name]["resolution_wh"],
            )
        uv[name], valid[name], camera_xyz_m[name] = current_uv, current_valid, current_xyz
    initial_error = np.linalg.norm(uv["stereo_left"][0] - query_pixels, axis=1)
    if not np.isfinite(initial_error).all() or float(initial_error.max()) > 1.0e-3:
        raise RuntimeError("TRACE query reprojection failed: {} px".format(float(initial_error.max())))
    trajectory_path = output / "predicted_trajectories.npz"
    np.savez_compressed(
        str(trajectory_path),
        frame_indices=np.arange(frames, dtype=np.int64),
        timestamps=calibration["stereo_left"]["timestamps"].astype(np.float64),
        tissue_node_ids=node_ids,
        tissue_positions_world=positions,
        stereo_left_tissue_uv_pixels=uv["stereo_left"],
        stereo_right_tissue_uv_pixels=uv["stereo_right"],
        stereo_left_tissue_valid=valid["stereo_left"],
        stereo_right_tissue_valid=valid["stereo_right"],
        stereo_left_tissue_camera_xyz_m=camera_xyz_m["stereo_left"],
        stereo_right_tissue_camera_xyz_m=camera_xyz_m["stereo_right"],
        query_frame=np.asarray(QUERY_FRAME, dtype=np.int64),
        query_pixels_stereo_left=query_pixels,
        query_alpha=query_alpha.detach().cpu().numpy().astype(np.float32),
        query_depth_trace_m=(
            query_depth.detach().cpu().numpy().astype(np.float32)
            / INTERNAL_UNITS_PER_METER
        ),
    )
    return {
        "trajectory": str(trajectory_path),
        "evaluation_manifest": str(manifest_path),
        "evaluation_manifest_sha256": sha256_file(manifest_path),
        "query_point_count": int(len(node_ids)),
        "query_valid_count": int(query_valid.sum().item()),
        "initial_query_reprojection_max_px": float(initial_error.max()),
        "decoder_version": TRACK_DECODER_VERSION,
        "query_anchor": "frozen frame-0 query pixels plus TRACE-rendered depth",
        "decoder": "TRACE target-minus-query persistent Gaussian displacement rendered at query geometry",
        "alignment": "none",
    }


def export_renders(
    frames, calibration, gaussians, deform, moving, fps, pipe, background, output
):
    selected = rendering_frames(frames)
    ones = torch.ones_like(gaussians.get_xyz)
    for camera_name in CAMERAS:
        rgb_dir = output / "rgb" / camera_name
        alpha_dir = output / "alpha" / camera_name
        rgb_dir.mkdir(parents=True, exist_ok=False)
        alpha_dir.mkdir(parents=True, exist_ok=False)
        for local_index, frame in enumerate(selected):
            camera = make_camera(calibration[camera_name], frame, frames)
            state = deformation_at_time(
                gaussians, deform, moving, normalized_time(frame, frames), fps
            )
            rgb = render_state(camera, gaussians, pipe, background, state)["render"]
            alpha = render_state(
                camera, gaussians, pipe, background, state, override_color=ones
            )["render"][0]
            save_rgb(rgb_dir / "{:06d}.png".format(frame), rgb)
            save_alpha(alpha_dir / "{:06d}.png".format(frame), alpha)
            if (local_index + 1) % 12 == 0 or local_index + 1 == len(selected):
                print(
                    "[render] {} {:04d}/{:04d}".format(
                        camera_name, local_index + 1, len(selected)
                    ),
                    flush=True,
                )
    return selected


def main() -> None:
    args, pipe = parse_args()
    frames = int(dataset_spec(args.dataset_key)["frames"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    gaussians, deform, iteration = load_trained_model(
        args.model_path,
        args.iteration,
        args.sh_degree,
        max_observed_time(frames),
        args.fps,
        args.light,
        args.physics_code,
        args.freegave,
    )
    calibration = load_calibration(args.source_path)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    with torch.inference_mode():
        moving = classify_motion(gaussians, deform, args.fps)
        tracks = export_tracks(
            args.source_path,
            frames,
            calibration,
            gaussians,
            deform,
            moving,
            args.fps,
            pipe,
            background,
            args.alpha_threshold,
            args.output_dir,
        )
        selected = export_renders(
            frames,
            calibration,
            gaussians,
            deform,
            moving,
            args.fps,
            pipe,
            background,
            args.output_dir,
        )
    provenance = {
        "schema": "fixedsuperbest.trace_export.v1",
        "adapter_version": ADAPTER_VERSION,
        "dataset_key": args.dataset_key,
        "trace_commit": UPSTREAM_COMMIT,
        "iteration": iteration,
        "freegave": False,
        "external_psm_control": False,
        "full_k_camera": True,
        "render_representation": "full scene; alpha is full TRACE opacity, not a GT-derived tissue mask",
        "rendered_frame_indices": selected,
        "time_normalization": "frame_index / (full_sequence_frame_count - 1)",
        "max_observed_time": max_observed_time(frames),
        "tracks": tracks,
        "ground_truth_usage": "only query IDs/frame-0 query pixels after checkpoint freeze; scorer reads remaining GT",
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
