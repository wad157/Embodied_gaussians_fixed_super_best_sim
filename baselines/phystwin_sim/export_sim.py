#!/usr/bin/env python3
"""Export PhysTwin native particle-LBS trajectories and tissue renders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.rendering import rasterization
from PIL import Image

from common import load_calibration
from protocol import (
    CAMERAS,
    DATASETS,
    QUERY_FRAME,
    UPSTREAM_COMMIT,
    dataset_spec,
    project_world_points,
    read_json,
    rendering_frames,
    resolve_dataset,
    sha256_file,
)
from gaussian_splatting.dynamic_utils import (
    get_topk_indices,
    interpolate_motions,
    knn_weights,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--physics-dir", type=Path, required=True)
    parser.add_argument("--appearance-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha-threshold", type=float, default=1.0 / 255.0)
    parser.add_argument("--lbs-neighbours", type=int, default=16)
    parser.add_argument("--lbs-chunk", type=int, default=10000)
    parser.add_argument("--allow-smoke", action="store_true")
    return parser.parse_args()


def evaluation_queries(dataset: Path) -> tuple[np.ndarray, np.ndarray, Path]:
    manifest_path = dataset / "evaluation" / "evaluation_points_30_non_grasp.json"
    manifest = read_json(manifest_path)
    node_ids = np.asarray(manifest["tissue_node_ids"], dtype=np.int32)
    if node_ids.shape != (30,) or len(np.unique(node_ids)) != 30:
        raise ValueError("formal SIM manifest must contain 30 unique nodes")
    truth_path = dataset / "ground_truth" / "trajectories_2d" / "stereo_left.npz"
    with np.load(truth_path, allow_pickle=False) as archive:
        all_ids = np.asarray(archive["tissue_node_ids"], dtype=np.int64)
        lookup = {int(node): index for index, node in enumerate(all_ids)}
        columns = np.asarray([lookup[int(node)] for node in node_ids], dtype=np.int64)
        pixels = np.asarray(
            archive["tissue_uv_pixels"][QUERY_FRAME, columns], dtype=np.float32
        )
        visible = np.asarray(
            archive["tissue_visible"][QUERY_FRAME, columns], dtype=bool
        )
    if not visible.all() or not np.isfinite(pixels).all():
        raise ValueError("formal query points are not all visible at frame 0")
    return node_ids, pixels, manifest_path


def sample_hwc(image: torch.Tensor, uv: np.ndarray) -> torch.Tensor:
    height, width = image.shape[:2]
    grid = torch.as_tensor(uv, dtype=image.dtype, device=image.device).clone()
    grid[:, 0] = 2.0 * grid[:, 0] / float(width - 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / float(height - 1) - 1.0
    sampled = F.grid_sample(
        image.permute(2, 0, 1)[None],
        grid.view(1, 1, -1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[0, :, 0].T


def render_gaussians(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    calibration: dict,
    camera: str,
    render_depth: bool,
):
    current = calibration[camera]
    width, height = current["resolution_wh"]
    mode = "RGB+D" if render_depth else "RGB"
    rendered, alpha, _ = rasterization(
        means=means,
        quats=F.normalize(quats, dim=-1),
        scales=scales,
        colors=colors,
        opacities=opacities,
        viewmats=torch.as_tensor(
            np.linalg.inv(current["X_WC_ros_optical"]).astype(np.float32),
            device=means.device,
        )[None],
        Ks=torch.as_tensor(current["K"].astype(np.float32), device=means.device)[None],
        width=width,
        height=height,
        # A single fixed SIM camera is rendered per call. The unpacked path is
        # compatible with both RGB and RGB+D in gsplat 1.5.
        packed=False,
        render_mode=mode,
        backgrounds=torch.zeros((1, 3), dtype=torch.float32, device=means.device),
        near_plane=0.01,
        far_plane=2.0,
    )
    return rendered[0], alpha[0]


def query_anchors(
    pixels: np.ndarray,
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    calibration: dict,
    alpha_threshold: float,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    rendered, alpha = render_gaussians(
        means, quats, scales, colors, opacities, calibration, "stereo_left", True
    )
    query_alpha = sample_hwc(alpha, pixels)[:, 0]
    depth = sample_hwc(rendered[..., 3:4], pixels)[:, 0] / query_alpha.clamp_min(1.0e-8)
    valid = (query_alpha > alpha_threshold) & torch.isfinite(depth) & (depth > 0)
    current = calibration["stereo_left"]
    camera_from_world = torch.as_tensor(
        np.linalg.inv(current["X_WC_ros_optical"]).astype(np.float32), device=means.device
    )
    if bool((~valid).any()):
        homogeneous = torch.cat((means, torch.ones_like(means[:, :1])), dim=1)
        camera_points = (camera_from_world @ homogeneous.T).T[:, :3]
        z = camera_points[:, 2]
        K = torch.as_tensor(current["K"].astype(np.float32), device=means.device)
        projected_h = (K @ camera_points.T).T
        projected = projected_h[:, :2] / z[:, None]
        uv_t = torch.as_tensor(pixels, device=means.device)
        distance = torch.cdist(uv_t, projected)
        distance[:, z <= 0] = float("inf")
        nearest = distance.argmin(dim=1)
        depth[~valid] = z[nearest[~valid]]
    K_inv = torch.linalg.inv(
        torch.as_tensor(current["K"].astype(np.float32), device=means.device)
    )
    uv_h = torch.cat(
        (torch.as_tensor(pixels, device=means.device), torch.ones((len(pixels), 1), device=means.device)),
        dim=1,
    )
    camera_points = (K_inv @ uv_h.T).T * depth[:, None]
    world_from_camera = torch.as_tensor(
        current["X_WC_ros_optical"].astype(np.float32), device=means.device
    )
    anchors = (
        world_from_camera
        @ torch.cat((camera_points, torch.ones_like(camera_points[:, :1])), dim=1).T
    ).T[:, :3]
    return (
        anchors,
        query_alpha.detach().cpu().numpy().astype(np.float32),
        depth.detach().cpu().numpy().astype(np.float32),
    )


def save_render(path: Path, image: torch.Tensor) -> None:
    array = (
        image.detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
    )
    Image.fromarray(array).save(path)


def save_alpha(path: Path, alpha: torch.Tensor) -> None:
    array = (
        alpha[..., 0].detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
    )
    Image.fromarray(array, mode="L").save(path)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"拒绝覆盖 {args.output_dir}")
    dataset = resolve_dataset(REPO_ROOT, args.dataset_key, args.dataset)
    spec = dataset_spec(args.dataset_key)
    frames = int(spec["frames"])
    physics_path = args.physics_dir / "raw_particle_rollout.npz"
    physics_meta_path = args.physics_dir / "metadata.json"
    gaussian_path = args.appearance_dir / "gaussians.npz"
    appearance_meta_path = args.appearance_dir / "metadata.json"
    for path in (physics_path, physics_meta_path, gaussian_path, appearance_meta_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    physics_meta = read_json(physics_meta_path)
    if not physics_meta.get("full_protocol", False) and not args.allow_smoke:
        raise ValueError("refusing to export smoke-test physics as formal output")
    with np.load(physics_path, allow_pickle=False) as archive:
        particles_np = np.asarray(archive["particle_positions_world"], dtype=np.float32)
    with np.load(gaussian_path, allow_pickle=False) as archive:
        means_np = np.asarray(archive["means_world"], dtype=np.float32)
        colors_np = np.asarray(archive["colors_rgb"], dtype=np.float32)
        opacities_np = np.asarray(archive["opacities"], dtype=np.float32)
        scales_np = np.asarray(archive["scales"], dtype=np.float32)
        quats_np = np.asarray(archive["quats"], dtype=np.float32)
    if particles_np.shape[0] != frames:
        raise ValueError("physics rollout frame count mismatch")
    node_ids, query_pixels, manifest_path = evaluation_queries(dataset)
    calibration = load_calibration(dataset)
    device = torch.device(args.device)
    means = torch.from_numpy(means_np).to(device)
    colors = torch.from_numpy(colors_np).to(device)
    opacities = torch.from_numpy(opacities_np).to(device)
    scales = torch.from_numpy(scales_np).to(device)
    quats = torch.from_numpy(quats_np).to(device)
    anchors, query_alpha, query_depth = query_anchors(
        query_pixels, means, quats, scales, colors, opacities,
        calibration, args.alpha_threshold
    )
    query_quats = torch.zeros((len(anchors), 4), dtype=torch.float32, device=device)
    query_quats[:, 0] = 1.0
    all_positions = torch.cat((means, anchors), dim=0)
    all_quats = torch.cat((quats, query_quats), dim=0)
    particle_positions = torch.from_numpy(particles_np).to(device)
    relations = get_topk_indices(particle_positions[0], K=args.lbs_neighbours)
    selected_frames = set(rendering_frames(frames))
    for camera in CAMERAS:
        (args.output_dir / "rgb" / camera).mkdir(parents=True, exist_ok=False)
        (args.output_dir / "alpha" / camera).mkdir(parents=True, exist_ok=False)
    query_positions = np.empty((frames, len(anchors), 3), dtype=np.float32)
    query_positions[0] = anchors.detach().cpu().numpy()

    with torch.inference_mode():
        for frame in range(frames):
            if frame > 0:
                previous = particle_positions[frame - 1]
                motion = particle_positions[frame] - previous
                for start in range(0, len(all_positions), args.lbs_chunk):
                    end = min(start + args.lbs_chunk, len(all_positions))
                    weights = knn_weights(
                        previous, all_positions[start:end], K=args.lbs_neighbours
                    )
                    position_chunk, quat_chunk, _ = interpolate_motions(
                        bones=previous,
                        motions=motion,
                        relations=relations,
                        xyz=all_positions[start:end],
                        quat=all_quats[start:end],
                        weights=weights,
                    )
                    all_positions[start:end] = position_chunk
                    all_quats[start:end] = quat_chunk
                query_positions[frame] = all_positions[len(means) :].cpu().numpy()
            if frame in selected_frames:
                for camera in CAMERAS:
                    rendered, alpha = render_gaussians(
                        all_positions[: len(means)],
                        all_quats[: len(means)],
                        scales,
                        colors,
                        opacities,
                        calibration,
                        camera,
                        False,
                    )
                    save_render(
                        args.output_dir / "rgb" / camera / f"{frame:06d}.png",
                        rendered[..., :3],
                    )
                    save_alpha(
                        args.output_dir / "alpha" / camera / f"{frame:06d}.png",
                        alpha,
                    )
            if frame % 20 == 0 or frame == frames - 1:
                print(f"[LBS/render] {frame}/{frames - 1}", flush=True)

    uv, valid, camera_xyz = {}, {}, {}
    for camera in CAMERAS:
        uv_values = np.empty((frames, len(node_ids), 2), dtype=np.float32)
        valid_values = np.empty((frames, len(node_ids)), dtype=bool)
        xyz_values = np.empty((frames, len(node_ids), 3), dtype=np.float32)
        current = calibration[camera]
        for frame in range(frames):
            uv_values[frame], valid_values[frame], xyz_values[frame] = project_world_points(
                query_positions[frame], current["K"], current["X_WC_ros_optical"],
                current["resolution_wh"]
            )
        uv[camera], valid[camera], camera_xyz[camera] = uv_values, valid_values, xyz_values
    initial_error = np.linalg.norm(uv["stereo_left"][0] - query_pixels, axis=1)
    if not np.isfinite(initial_error).all() or float(initial_error.max()) > 1.0e-3:
        raise RuntimeError(f"query anchor reprojection failed: {initial_error.max()} px")
    trajectory_path = args.output_dir / "predicted_trajectories.npz"
    np.savez_compressed(
        trajectory_path,
        frame_indices=np.arange(frames, dtype=np.int64),
        timestamps=calibration["stereo_left"]["timestamps"].astype(np.float64),
        tissue_node_ids=node_ids,
        tissue_positions_world=query_positions,
        stereo_left_tissue_uv_pixels=uv["stereo_left"],
        stereo_right_tissue_uv_pixels=uv["stereo_right"],
        stereo_left_tissue_valid=valid["stereo_left"],
        stereo_right_tissue_valid=valid["stereo_right"],
        stereo_left_tissue_camera_xyz_m=camera_xyz["stereo_left"],
        stereo_right_tissue_camera_xyz_m=camera_xyz["stereo_right"],
        query_frame=np.asarray(QUERY_FRAME, dtype=np.int64),
        query_pixels_stereo_left=query_pixels,
        query_alpha=query_alpha,
        query_depth_phystwin_m=query_depth,
    )
    metadata = {
        "schema": "fixedsuperbest.phystwin_sim_export.v1",
        "dataset_key": args.dataset_key,
        "phystwin_commit": UPSTREAM_COMMIT,
        "physics_sha256": sha256_file(physics_path),
        "gaussians_sha256": sha256_file(gaussian_path),
        "evaluation_manifest": str(manifest_path),
        "evaluation_manifest_sha256": sha256_file(manifest_path),
        "query_frame": QUERY_FRAME,
        "query_camera": "stereo_left",
        "query_count": int(len(node_ids)),
        "query_alpha_valid_count": int((query_alpha > args.alpha_threshold).sum()),
        "initial_query_reprojection_max_px": float(initial_error.max()),
        "trajectory_source": "PhysTwin native spring-mass particles plus upstream Gaussian LBS",
        "shape_of_motion_used": False,
        "lbs_neighbours": args.lbs_neighbours,
        "rendered_frame_indices": sorted(selected_frames),
        "render_background": "black",
        "future_observations_used": False,
        "ground_truth_usage": "node IDs and frame-0 left-camera 2D queries only after physics/appearance freeze",
        "alignment": "none",
        "formal_physics_protocol": bool(physics_meta.get("full_protocol", False)),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
