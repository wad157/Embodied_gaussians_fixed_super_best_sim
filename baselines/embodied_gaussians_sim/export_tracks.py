#!/usr/bin/env python3
"""Export native PBD trajectories from frozen EG Gaussian-particle bonds.

The EG paper evaluates a query by finding its closest Gaussian at the first
timestep and tracking that Gaussian's bonded frame. This evaluator-side step
uses the model's own rendered query depth, never GT depth or GT 3D, to turn the
fixed 2D query into a point in that frame. Shape of Motion is unnecessary
because EG already exposes persistent physical frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_functional
from gsplat.rendering import rasterization
from scipy.spatial import cKDTree


BASELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BASELINE_ROOT.parents[1]
sys.path.insert(0, str(BASELINE_ROOT))
from protocol import (  # noqa: E402
    CAMERAS,
    QUERY_FRAME,
    project_world_points,
    read_json,
    resolve_dataset,
    sha256_file,
)
from run_soft import TissueGaussianBody, matrix_to_quaternion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-key", choices=("sim01", "sim02", "sim03"), required=True
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alpha-threshold", type=float, default=1.0 / 255.0)
    return parser.parse_args()


def evaluation_queries(dataset: Path) -> tuple[np.ndarray, np.ndarray, Path]:
    manifest_path = dataset / "evaluation" / "evaluation_points_30_non_grasp.json"
    manifest = read_json(manifest_path)
    node_ids = np.asarray(manifest["tissue_node_ids"], dtype=np.int32)
    if node_ids.shape != (30,) or len(np.unique(node_ids)) != 30:
        raise ValueError("正式 manifest 必须恰好包含30个互异节点")
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
    if not bool(visible.all()) or not np.isfinite(pixels).all():
        raise ValueError("固定评估点必须在 query frame 的左相机全部可见且有限")
    return node_ids, pixels, manifest_path


def sample_hwc(image: torch.Tensor, pixels_uv: np.ndarray) -> torch.Tensor:
    height, width = image.shape[:2]
    pixels = torch.as_tensor(pixels_uv, dtype=image.dtype, device=image.device)
    grid = pixels.clone()
    grid[:, 0] = 2.0 * grid[:, 0] / float(width - 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / float(height - 1) - 1.0
    sampled = torch_functional.grid_sample(
        image.permute(2, 0, 1).unsqueeze(0),
        grid.view(1, 1, -1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[0, :, 0, :].transpose(0, 1)


def gaussian_positions_from_particles(
    particle_positions: np.ndarray,
    particle_rotations: np.ndarray,
    parents: np.ndarray,
    local_offsets: np.ndarray,
) -> np.ndarray:
    return particle_positions[:, parents] + np.einsum(
        "tgij,gj->tgi", particle_rotations[:, parents], local_offsets
    )


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("EG 自身 splatting depth 查询需要 CUDA")
    if not 0.0 < args.alpha_threshold < 1.0:
        raise ValueError("alpha threshold 必须位于 (0,1)")
    device = torch.device(args.device)
    dataset = resolve_dataset(REPO_ROOT, args.dataset_key, args.dataset)
    rollout = args.rollout_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    metadata_path = output.with_suffix(".metadata.json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"拒绝覆盖已有 EG 轨迹导出：{output}")
    run_metadata = read_json(rollout / "rollout_metadata.json")
    if bool(run_metadata.get("evaluation_truth_opened", True)):
        raise ValueError("rollout metadata 未证明算法阶段与评测真值隔离")
    body_path = args.body.expanduser().resolve()
    if sha256_file(body_path) != run_metadata["body_sha256"]:
        raise ValueError("轨迹导出的 body 与 rollout 使用的 body 不一致")

    node_ids, query_pixels, manifest_path = evaluation_queries(dataset)
    body = TissueGaussianBody(body_path, device)
    with np.load(rollout / "raw_particle_rollout.npz", allow_pickle=False) as archive:
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int32)
        timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
        particle_positions = np.asarray(
            archive["particle_positions_world"], dtype=np.float32
        )
        particle_rotations = np.asarray(
            archive["particle_rotations_world"], dtype=np.float32
        )
    if QUERY_FRAME not in frame_indices:
        raise ValueError("rollout 不包含 query frame")
    query_slot = int(np.flatnonzero(frame_indices == QUERY_FRAME)[0])
    parents_all = body.parents.detach().cpu().numpy()
    local_offsets_all = body.local_offsets.detach().cpu().numpy()
    gaussian_positions = gaussian_positions_from_particles(
        particle_positions, particle_rotations, parents_all, local_offsets_all
    )

    cameras = read_json(dataset / "cameras.json")
    left_video = read_json(dataset / "videos" / "stereo_left.json")
    intrinsic_np = np.asarray(left_video["K"], dtype=np.float32)
    world_from_camera_np = np.asarray(
        cameras["stereo_left"]["X_WC_ros_optical"], dtype=np.float32
    )
    camera_from_world_np = np.linalg.inv(world_from_camera_np).astype(np.float32)
    width, height = (int(value) for value in left_video["resolution"])
    query_parent_rotations = torch.as_tensor(
        particle_rotations[query_slot, parents_all], device=device
    )
    query_gaussian_rotations = query_parent_rotations @ body.local_rotations
    query_quaternions = matrix_to_quaternion(query_gaussian_rotations)
    query_means = torch.as_tensor(gaussian_positions[query_slot], device=device)
    rendered, alpha, _ = rasterization(
        means=query_means,
        quats=query_quaternions,
        scales=body.scales,
        colors=torch.ones_like(query_means),
        opacities=body.opacity_logits.sigmoid(),
        viewmats=torch.as_tensor(camera_from_world_np, device=device).unsqueeze(0),
        Ks=torch.as_tensor(intrinsic_np, device=device).unsqueeze(0),
        width=width,
        height=height,
        camera_model="pinhole",
        render_mode="RGB+D",
        backgrounds=torch.zeros((1, 3), dtype=torch.float32, device=device),
        near_plane=0.01,
        far_plane=2.0,
        packed=False,
    )
    query_alpha = sample_hwc(alpha[0], query_pixels)[:, 0]
    query_depth_premultiplied = sample_hwc(
        rendered[0, ..., 3:4], query_pixels
    )[:, 0]
    query_depth = query_depth_premultiplied / query_alpha.clamp_min(1.0e-8)
    query_valid = (
        (query_alpha > float(args.alpha_threshold))
        & torch.isfinite(query_depth)
        & (query_depth > 0.0)
    )
    fallback_mask = ~query_valid
    fallback_gaussian_ids = torch.full(
        (len(node_ids),), -1, dtype=torch.long, device=device
    )
    fallback_pixel_distances = torch.zeros(
        (len(node_ids),), dtype=query_depth.dtype, device=device
    )
    if bool(fallback_mask.any()):
        # The paper defines evaluation by selecting the query-time closest
        # Gaussian.  When the shared 1/255 alpha coverage test has no raster
        # depth, select the closest visible projected Gaussian centre and use
        # that Gaussian's own z as the model-only depth for the original query
        # ray.  This is deterministic and never reads GT depth or GT 3D.
        camera_from_world = torch.as_tensor(camera_from_world_np, device=device)
        gaussian_camera = (
            query_means @ camera_from_world[:3, :3].T
            + camera_from_world[:3, 3]
        )
        gaussian_pixels_h = gaussian_camera @ torch.as_tensor(
            intrinsic_np, device=device
        ).T
        gaussian_pixels = gaussian_pixels_h[:, :2] / gaussian_pixels_h[:, 2:].clamp_min(
            1.0e-8
        )
        visible_gaussians = (
            torch.isfinite(gaussian_pixels).all(dim=1)
            & torch.isfinite(gaussian_camera).all(dim=1)
            & (gaussian_camera[:, 2] > 0.01)
            & (gaussian_pixels[:, 0] >= 0.0)
            & (gaussian_pixels[:, 0] <= width - 1.0)
            & (gaussian_pixels[:, 1] >= 0.0)
            & (gaussian_pixels[:, 1] <= height - 1.0)
        )
        visible_ids = torch.nonzero(visible_gaussians, as_tuple=False).flatten()
        if len(visible_ids) == 0:
            raise RuntimeError("query-time 没有可见 EG Gaussian，无法导出轨迹")
        missing_pixels = torch.as_tensor(
            query_pixels, dtype=query_depth.dtype, device=device
        )[fallback_mask]
        distances_2d = torch.cdist(missing_pixels, gaussian_pixels[visible_ids])
        minimum, local_ids = distances_2d.min(dim=1)
        selected_visible_ids = visible_ids[local_ids]
        fallback_gaussian_ids[fallback_mask] = selected_visible_ids
        fallback_pixel_distances[fallback_mask] = minimum
        query_depth[fallback_mask] = gaussian_camera[selected_visible_ids, 2]

    pixels_tensor = torch.as_tensor(query_pixels, device=device)
    intrinsic = torch.as_tensor(intrinsic_np, device=device)
    query_camera = torch.stack(
        (
            (pixels_tensor[:, 0] - intrinsic[0, 2]) * query_depth / intrinsic[0, 0],
            (pixels_tensor[:, 1] - intrinsic[1, 2]) * query_depth / intrinsic[1, 1],
            query_depth,
        ),
        dim=1,
    )
    world_from_camera = torch.as_tensor(world_from_camera_np, device=device)
    query_world = (
        query_camera @ world_from_camera[:3, :3].T + world_from_camera[:3, 3]
    )
    query_world_np = query_world.detach().cpu().numpy().astype(np.float32)
    distances, gaussian_ids = cKDTree(gaussian_positions[query_slot]).query(
        query_world_np, k=1, workers=-1
    )
    gaussian_ids = np.asarray(gaussian_ids, dtype=np.int64)
    fallback_mask_np = fallback_mask.detach().cpu().numpy()
    fallback_ids_np = fallback_gaussian_ids.detach().cpu().numpy()
    gaussian_ids[fallback_mask_np] = fallback_ids_np[fallback_mask_np]
    distances = np.asarray(distances, dtype=np.float64)
    distances[fallback_mask_np] = np.linalg.norm(
        query_world_np[fallback_mask_np]
        - gaussian_positions[query_slot, gaussian_ids[fallback_mask_np]],
        axis=1,
    )
    selected_parents = parents_all[gaussian_ids]

    # Track the exact model-depth query point in the selected Gaussian's bonded
    # parent frame. This is the native PBD frame trajectory described by EG.
    query_particle_positions = particle_positions[query_slot, selected_parents]
    query_particle_rotations = particle_rotations[query_slot, selected_parents]
    query_local_offsets = np.einsum(
        "nij,nj->ni",
        np.swapaxes(query_particle_rotations, 1, 2),
        query_world_np - query_particle_positions,
    )
    selected_positions = particle_positions[:, selected_parents] + np.einsum(
        "tnij,nj->tni",
        particle_rotations[:, selected_parents],
        query_local_offsets,
    )

    projected: dict[str, np.ndarray] = {}
    projected_valid: dict[str, np.ndarray] = {}
    camera_xyz: dict[str, np.ndarray] = {}
    for camera in CAMERAS:
        video = read_json(dataset / "videos" / f"{camera}.json")
        camera_intrinsic = np.asarray(video["K"], dtype=np.float64)
        camera_world = np.asarray(
            cameras[camera]["X_WC_ros_optical"], dtype=np.float64
        )
        camera_size = tuple(int(value) for value in video["resolution"])
        uv = np.empty((len(frame_indices), len(node_ids), 2), dtype=np.float32)
        valid = np.empty((len(frame_indices), len(node_ids)), dtype=bool)
        xyz = np.empty((len(frame_indices), len(node_ids), 3), dtype=np.float32)
        for slot, positions in enumerate(selected_positions):
            uv[slot], valid[slot], xyz[slot] = project_world_points(
                positions, camera_intrinsic, camera_world, camera_size
            )
        projected[camera] = uv
        projected_valid[camera] = valid
        camera_xyz[camera] = xyz

    initial_error = np.linalg.norm(
        projected["stereo_left"][query_slot] - query_pixels, axis=1
    )
    initial_error_max = float(np.max(initial_error))
    if not np.all(np.isfinite(initial_error)) or initial_error_max > 1.0e-3:
        raise RuntimeError(f"PBD 查询锚定失败：最大重投影误差 {initial_error_max:.6g}px")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frame_indices=frame_indices,
        timestamps=timestamps,
        tissue_node_ids=node_ids,
        tissue_positions_world=selected_positions.astype(np.float32),
        stereo_left_tissue_uv_pixels=projected["stereo_left"],
        stereo_right_tissue_uv_pixels=projected["stereo_right"],
        stereo_left_tissue_valid=projected_valid["stereo_left"],
        stereo_right_tissue_valid=projected_valid["stereo_right"],
        stereo_left_tissue_camera_xyz_m=camera_xyz["stereo_left"],
        stereo_right_tissue_camera_xyz_m=camera_xyz["stereo_right"],
        query_frame=np.asarray(QUERY_FRAME, dtype=np.int64),
        query_pixels_stereo_left=query_pixels,
        query_alpha=query_alpha.detach().cpu().numpy().astype(np.float32),
        query_depth_embodied_gaussians_m=(
            query_depth.detach().cpu().numpy().astype(np.float32)
        ),
    )
    export_metadata = {
        "schema": "fixedsuperbest.embodied_gaussians_native_pbd_query_export.v1",
        "evaluation_stage_only": True,
        "dataset_key": args.dataset_key,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "query_camera": "stereo_left",
        "query_frame": QUERY_FRAME,
        "query_input": "evaluation 2D pixel plus EG-rendered alpha/depth",
        "query_gt_depth_used": False,
        "query_gt_3d_used": False,
        "trajectory_source": (
            "native PBD particle poses; exact query point tracked in the parent frame "
            "of the nearest query-time Gaussian"
        ),
        "shape_of_motion_used": False,
        "shape_of_motion_reason": (
            "unnecessary: EG natively provides persistent Gaussian-particle physical frames"
        ),
        "selected_gaussian_ids": gaussian_ids.tolist(),
        "selected_parent_particle_ids": selected_parents.tolist(),
        "query_to_selected_gaussian_distance_mm": (
            1000.0 * np.asarray(distances)
        ).tolist(),
        "query_alpha_threshold": float(args.alpha_threshold),
        "query_raster_coverage_count": int(query_valid.sum().item()),
        "query_projected_nearest_gaussian_fallback_count": int(
            fallback_mask.sum().item()
        ),
        "query_projected_nearest_gaussian_fallback_indices": np.flatnonzero(
            fallback_mask_np
        ).astype(int).tolist(),
        "query_projected_nearest_gaussian_fallback_pixel_distances": (
            fallback_pixel_distances[fallback_mask]
            .detach()
            .cpu()
            .numpy()
            .astype(float)
            .tolist()
        ),
        "query_alpha_min": float(query_alpha.min().item()),
        "query_depth_model_m_min": float(query_depth.min().item()),
        "query_depth_model_m_max": float(query_depth.max().item()),
        "initial_query_reprojection_max_px": initial_error_max,
        "alignment": "none",
        "prediction_sha256": sha256_file(output),
    }
    metadata_path.write_text(
        json.dumps(export_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(export_metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
