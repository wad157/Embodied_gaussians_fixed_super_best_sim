#!/usr/bin/env python3
"""Export EH-SurGS renders and Shape-of-Motion-style SIM trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_functional
from PIL import Image

from arguments import FDMHiddenParams, ModelParams, PipelineParams
from gaussian_renderer import render_flow
from utils.params_utils import merge_hparams

from common import (
    RenderCamera,
    deformed_centers,
    internal_world_from_camera,
    load_calibration,
    load_trained_gaussians,
)
from protocol import (
    CAMERAS,
    DATASETS,
    INTERNAL_UNITS_PER_METER,
    QUERY_FRAME,
    SHAPE_OF_MOTION_COMMIT,
    TRACK_DECODER_VERSION,
    UPSTREAM_COMMIT,
    dataset_spec,
    project_world_points,
    rendering_frames,
    resolve_dataset,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "sim_official.py"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    hidden = FDMHiddenParams(parser)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--configs", default=str(DEFAULT_CONFIG))
    parser.add_argument("--alpha-threshold", type=float, default=1.0 / 255.0)
    args = parser.parse_args()
    import mmcv

    args = merge_hparams(args, mmcv.Config.fromfile(args.configs))
    if not args.model_path or not args.source_path:
        parser.error("--model_path 与 --source_path 必须显式指定")
    args.source_path = str(
        resolve_dataset(REPO_ROOT, args.dataset_key, Path(args.source_path))
    )
    args.model_path = str(Path(args.model_path).expanduser().resolve())
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists():
        parser.error("拒绝覆盖已有导出目录：{}".format(args.output_dir))
    return args, model, pipeline, hidden


def make_render_camera(calibration: dict, frame: int, frames: int) -> RenderCamera:
    return RenderCamera(
        calibration["K"],
        calibration["X_WC_ros_optical"],
        calibration["resolution_wh"],
        float(frame) / float(frames - 1),
    )


def render(camera, gaussians, pipe, background, learned_mask, override_color=None):
    return render_flow(
        camera,
        gaussians,
        pipe,
        background,
        True,
        override_color=override_color,
        mask=learned_mask,
    )


def save_rgb(path: Path, tensor: torch.Tensor) -> None:
    array = (
        tensor.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .add(0.5)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray(array, mode="RGB").save(str(path))


def save_alpha(path: Path, tensor: torch.Tensor) -> None:
    array = (
        tensor.detach().clamp(0.0, 1.0).mul(255.0).add(0.5).byte().cpu().numpy()
    )
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
    if node_ids.shape != (30,):
        raise ValueError("正式清单必须含 30 个测评点")
    reference_path = dataset / "ground_truth" / "trajectories_2d" / "stereo_left.npz"
    with np.load(str(reference_path), allow_pickle=False) as reference:
        all_ids = np.asarray(reference["tissue_node_ids"], dtype=np.int64)
        lookup = {int(node): index for index, node in enumerate(all_ids)}
        columns = np.asarray([lookup[int(node)] for node in node_ids], dtype=np.int64)
        visible = np.asarray(reference["tissue_visible"])[QUERY_FRAME, columns]
        pixels = np.asarray(reference["tissue_uv_pixels"])[QUERY_FRAME, columns]
    if not bool(np.all(visible)) or not np.all(np.isfinite(pixels)):
        raise ValueError("frame-0 的 30 个固定查询点必须全部可见且有限")
    return node_ids.astype(np.int32), pixels.astype(np.float32), manifest_path


def export_tracks(
    dataset,
    frames,
    calibration,
    gaussians,
    learned_mask,
    pipe,
    background,
    alpha_threshold,
    output,
):
    node_ids, query_pixels, manifest_path = evaluation_queries(dataset)
    query_camera = make_render_camera(calibration["stereo_left"], QUERY_FRAME, frames)
    ones = torch.ones(
        (gaussians.get_xyz.shape[0], 3),
        dtype=gaussians.get_xyz.dtype,
        device=gaussians.get_xyz.device,
    )
    query_render = render(
        query_camera, gaussians, pipe, background, learned_mask, override_color=ones
    )
    query_alpha_image = query_render["render"][0:1]
    query_alpha = sample_chw(query_alpha_image, query_pixels)[:, 0]
    query_depth_premultiplied = sample_chw(
        query_render["depth"], query_pixels
    )[:, 0]
    query_depth = query_depth_premultiplied / query_alpha.clamp_min(1.0e-8)
    query_valid = (
        (query_alpha > float(alpha_threshold))
        & torch.isfinite(query_depth)
        & (query_depth > 0.0)
    )
    if int(query_valid.sum().item()) != len(node_ids):
        raise RuntimeError(
            "固定 30 点的 query alpha/depth 覆盖不足：{}/{}；拒绝生成不完整正式结果".format(
                int(query_valid.sum().item()), len(node_ids)
            )
        )

    intrinsic = torch.as_tensor(
        calibration["stereo_left"]["K"],
        dtype=query_depth.dtype,
        device=query_depth.device,
    )
    query_pixels_tensor = torch.as_tensor(
        query_pixels, dtype=query_depth.dtype, device=query_depth.device
    )
    query_camera_xyz = torch.stack(
        (
            (query_pixels_tensor[:, 0] - intrinsic[0, 2])
            / intrinsic[0, 0]
            * query_depth,
            (query_pixels_tensor[:, 1] - intrinsic[1, 2])
            / intrinsic[1, 1]
            * query_depth,
            query_depth,
        ),
        dim=1,
    )
    world_from_query_camera = torch.as_tensor(
        internal_world_from_camera(calibration["stereo_left"]["X_WC_ros_optical"]),
        dtype=query_depth.dtype,
        device=query_depth.device,
    )
    query_world = torch.einsum(
        "ij,nj->ni",
        world_from_query_camera[:3],
        torch_functional.pad(query_camera_xyz, (0, 1), value=1.0),
    )

    query_centers = deformed_centers(gaussians, learned_mask, 0.0)
    xyz_span = gaussians.get_xyz.max(dim=0).values - gaussians.get_xyz.min(dim=0).values
    feature_scale = xyz_span.max().clamp_min(1.0e-6)
    positions = np.full((frames, len(node_ids), 3), np.nan, dtype=np.float32)
    for frame in range(frames):
        normalized_time = float(frame) / float(frames - 1)
        target_centers = deformed_centers(gaussians, learned_mask, normalized_time)
        encoded_displacement = (target_centers - query_centers) / feature_scale
        displacement_image = render(
            query_camera,
            gaussians,
            pipe,
            background,
            learned_mask,
            override_color=encoded_displacement,
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

    uv, valid, camera_xyz = {}, {}, {}
    for camera_name in CAMERAS:
        camera_uv = np.empty((frames, len(node_ids), 2), dtype=np.float32)
        camera_valid = np.empty((frames, len(node_ids)), dtype=bool)
        camera_points = np.empty((frames, len(node_ids), 3), dtype=np.float32)
        current = calibration[camera_name]
        for frame in range(frames):
            projected, is_valid, points_camera = project_world_points(
                positions[frame],
                current["K"],
                current["X_WC_ros_optical"],
                current["resolution_wh"],
            )
            camera_uv[frame] = projected
            camera_valid[frame] = is_valid
            camera_points[frame] = points_camera
        uv[camera_name], valid[camera_name] = camera_uv, camera_valid
        camera_xyz[camera_name] = camera_points

    initial_query_error = np.linalg.norm(
        uv["stereo_left"][QUERY_FRAME] - query_pixels, axis=1
    )
    initial_query_error_max = float(np.max(initial_query_error))
    if not np.all(np.isfinite(initial_query_error)) or initial_query_error_max > 1.0e-3:
        raise RuntimeError(
            "查询锚定失败：frame-0 最大重投影误差 {:.6g}px".format(
                initial_query_error_max
            )
        )

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
        stereo_left_tissue_camera_xyz_m=camera_xyz["stereo_left"],
        stereo_right_tissue_camera_xyz_m=camera_xyz["stereo_right"],
        query_frame=np.asarray(QUERY_FRAME, dtype=np.int64),
        query_pixels_stereo_left=query_pixels,
        query_alpha=query_alpha.detach().cpu().numpy().astype(np.float32),
        query_depth_eh_surgs_m=(
            query_depth.detach().cpu().numpy().astype(np.float32)
            / INTERNAL_UNITS_PER_METER
        ),
    )
    return {
        "trajectory": str(trajectory_path),
        "evaluation_manifest": str(manifest_path),
        "evaluation_manifest_sha256": sha256_file(manifest_path),
        "query_frame": QUERY_FRAME,
        "query_camera": "stereo_left",
        "query_point_count": int(len(node_ids)),
        "query_valid_count": int(query_valid.sum().item()),
        "decoder_version": TRACK_DECODER_VERSION,
        "decoder": "target-minus-query deformed Gaussian displacement rasterized through fixed query-time geometry, alpha-normalized and added to the model-depth query anchor",
        "query_anchor": "fixed frame-0 2D query plus EH-SurGS rendered depth; no GT depth or GT 3D",
        "initial_query_reprojection_max_px": initial_query_error_max,
        "feature_scale_internal_units": float(feature_scale.item()),
        "alignment": "none",
        "internal_units_per_meter": INTERNAL_UNITS_PER_METER,
    }


def export_renders(
    frames, calibration, gaussians, learned_mask, pipe, background, output
):
    selected_frames = rendering_frames(frames)
    ones = torch.ones(
        (gaussians.get_xyz.shape[0], 3),
        dtype=gaussians.get_xyz.dtype,
        device=gaussians.get_xyz.device,
    )
    for camera_name in CAMERAS:
        rgb_dir = output / "rgb" / camera_name
        alpha_dir = output / "alpha" / camera_name
        rgb_dir.mkdir(parents=True, exist_ok=False)
        alpha_dir.mkdir(parents=True, exist_ok=False)
        for local_index, frame in enumerate(selected_frames):
            camera = make_render_camera(calibration[camera_name], frame, frames)
            rgb = render(
                camera, gaussians, pipe, background, learned_mask
            )["render"]
            alpha = render(
                camera,
                gaussians,
                pipe,
                background,
                learned_mask,
                override_color=ones,
            )["render"][0]
            save_rgb(rgb_dir / "{:06d}.png".format(frame), rgb)
            save_alpha(alpha_dir / "{:06d}.png".format(frame), alpha)
            if (local_index + 1) % 12 == 0 or local_index + 1 == len(selected_frames):
                print(
                    "[render] {} {:04d}/{:04d}".format(
                        camera_name, local_index + 1, len(selected_frames)
                    ),
                    flush=True,
                )
    return selected_frames


def main():
    args, model, pipeline, hidden = parse_args()
    dataset = Path(args.source_path)
    frames = int(dataset_spec(args.dataset_key)["frames"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    gaussians, learned_mask, iteration = load_trained_gaussians(
        Path(args.model_path),
        args.iteration,
        int(args.sh_degree),
        hidden.extract(args),
    )
    pipe = pipeline.extract(args)
    calibration = load_calibration(dataset)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    with torch.inference_mode():
        tracks = export_tracks(
            dataset,
            frames,
            calibration,
            gaussians,
            learned_mask,
            pipe,
            background,
            args.alpha_threshold,
            args.output_dir,
        )
        selected_frames = export_renders(
            frames,
            calibration,
            gaussians,
            learned_mask,
            pipe,
            background,
            args.output_dir,
        )
    provenance = {
        "schema": "fixedsuperbest.eh_surgs_export.v1",
        "dataset_key": args.dataset_key,
        "dataset": str(dataset),
        "model_path": args.model_path,
        "iteration": iteration,
        "eh_surgs_commit": UPSTREAM_COMMIT,
        "shape_of_motion_commit": SHAPE_OF_MOTION_COMMIT,
        "shape_of_motion_usage": "trajectory decoder design only",
        "trajectory_decoder_version": TRACK_DECODER_VERSION,
        "time_normalization": "frame_index / (full_sequence_frame_count - 1)",
        "coordinate_units": "1000 internal units per meter; output divided by 1000; no fitted alignment",
        "rendered_frame_indices": selected_frames,
        "render_background": "black",
        "tracks": tracks,
        "ground_truth_usage": "only immutable node IDs and frame-0 left-camera 2D queries after checkpoint freeze; evaluators consume remaining truth",
        "future_observations_used": False,
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
