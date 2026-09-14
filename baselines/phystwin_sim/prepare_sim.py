#!/usr/bin/env python3
"""Prepare leakage-free PhysTwin tracks, controller points and appearance.

Only legal prefix RGB, estimated depth and tissue masks are opened.  Evaluation
truth is deliberately absent from this module.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

from common import (
    PackedTissueMasks,
    deproject_pixels,
    farthest_point_indices,
    load_calibration,
    load_rgb,
    resize_rgb,
    sample_image,
)
from protocol import (
    CAMERAS,
    COTRACKER_CHECKPOINT,
    COTRACKER_VARIANT,
    DATASETS,
    UPSTREAM_COMMIT,
    dataset_spec,
    future_start,
    resolve_dataset,
    sha256_file,
    training_frames,
)


BASELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BASELINE_ROOT.parents[1]
DEFAULT_COTRACKER_ROOT = Path(
    "/home/jwshan/.cache/torch/hub/facebookresearch_co-tracker_main"
)
DEFAULT_COTRACKER_WEIGHT = Path(
    "/home/jwshan/.cache/torch/hub/checkpoints/scaled_offline.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--track-size", type=int, default=384)
    parser.add_argument("--queries-per-camera", type=int, default=768)
    parser.add_argument("--max-physics-points", type=int, default=768)
    parser.add_argument("--appearance-points", type=int, default=50000)
    parser.add_argument("--controller-points", type=int, default=30)
    parser.add_argument("--cotracker-root", type=Path, default=DEFAULT_COTRACKER_ROOT)
    parser.add_argument("--cotracker-weight", type=Path, default=DEFAULT_COTRACKER_WEIGHT)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initial_queries(mask: np.ndarray, count: int, size: int, seed: int) -> np.ndarray:
    small = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    # Avoid uncertain tissue boundaries while retaining the official random
    # dense-query semantics.
    eroded = cv2.erode(small, np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    y, x = np.nonzero(eroded)
    if len(x) < count:
        raise ValueError(f"tissue mask only contains {len(x)} eligible query pixels")
    rng = np.random.RandomState(seed)
    candidates = rng.choice(len(x), size=min(len(x), count * 20), replace=False)
    points = np.stack((x[candidates], y[candidates]), axis=1).astype(np.float32)
    chosen = farthest_point_indices(points, count, seed)
    return points[chosen]


def track_camera(
    dataset: Path,
    camera: str,
    allowed: list[int],
    masks: PackedTissueMasks,
    predictor,
    args: argparse.Namespace,
    camera_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    frames = []
    for local, frame in enumerate(allowed):
        frames.append(resize_rgb(load_rgb(dataset, camera, frame), args.track_size))
        if (local + 1) % 40 == 0 or local + 1 == len(allowed):
            print(f"[load] {camera} {local + 1}/{len(allowed)}", flush=True)
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].float()
    query_xy = initial_queries(
        masks.get(camera, allowed[0]),
        args.queries_per_camera,
        args.track_size,
        args.seed * 101 + camera_index,
    )
    queries = np.concatenate(
        (np.zeros((len(query_xy), 1), dtype=np.float32), query_xy), axis=1
    )
    with torch.inference_mode():
        tracks, visible = predictor(
            video.to(args.device),
            queries=torch.from_numpy(queries)[None].to(args.device),
            backward_tracking=False,
        )
    del video
    scale = np.asarray(
        [masks.get(camera, 0).shape[1] / args.track_size,
         masks.get(camera, 0).shape[0] / args.track_size],
        dtype=np.float32,
    )
    tracks_np = tracks[0].cpu().numpy().astype(np.float32) * scale[None, None]
    visible_np = visible[0].cpu().numpy().astype(bool)
    torch.cuda.empty_cache()
    return tracks_np, visible_np


def controller_trajectory(
    dataset: Path, initial_tissue_points: np.ndarray, count: int, seed: int
) -> tuple[np.ndarray, dict]:
    poses_path = dataset / "task_inputs" / "psm_link_poses.npz"
    meshes_path = dataset / "gui_assets" / "official_psm_tip_meshes_v2.npz"
    with np.load(poses_path, allow_pickle=False) as poses, np.load(
        meshes_path, allow_pickle=False
    ) as meshes:
        pose_names = [str(x) for x in poses["link_names"]]
        mesh_names = [str(x) for x in meshes["link_names"]]
        x_wl = np.asarray(poses["X_WL"], dtype=np.float64)
        local_points, local_links = [], []
        for mesh_index, name in enumerate(mesh_names):
            vertices = np.asarray(meshes[f"{name}__vertices"], dtype=np.float64)
            local_points.append(vertices)
            local_links.extend([pose_names.index(name)] * len(vertices))
        local = np.concatenate(local_points, axis=0)
        links = np.asarray(local_links, dtype=np.int64)
    homogeneous = np.concatenate((local, np.ones((len(local), 1))), axis=1)
    world_zero = np.einsum("nij,nj->ni", x_wl[0, links], homogeneous)[:, :3]
    distances, _ = cKDTree(np.asarray(initial_tissue_points)).query(world_zero, k=1)
    # PhysTwin extracts controller points from observed controller surfaces. In
    # SIM the exact equivalent is the official articulated mesh; retain the
    # closest surface pool and spatially cover it without consulting GT masks.
    pool_count = min(len(world_zero), max(count * 200, count))
    pool = np.argpartition(distances, pool_count - 1)[:pool_count]
    selected_in_pool = farthest_point_indices(world_zero[pool], count, seed)
    selected = pool[selected_in_pool]
    local = local[selected]
    links = links[selected]
    homogeneous = homogeneous[selected]
    trajectory = np.empty((x_wl.shape[0], len(local), 3), dtype=np.float32)
    for frame in range(x_wl.shape[0]):
        trajectory[frame] = np.einsum(
            "nij,nj->ni", x_wl[frame, links], homogeneous
        )[:, :3]
    return trajectory, {
        "pose_sha256": sha256_file(poses_path),
        "mesh_sha256": sha256_file(meshes_path),
        "link_names": mesh_names,
        "points": int(len(local)),
        "selection": "nearest official FK mesh surface pool to observed frame-0 tissue, then FPS",
        "initial_distance_to_observed_tissue_m": {
            "min": float(distances[selected].min()),
            "median": float(np.median(distances[selected])),
            "max": float(distances[selected].max()),
        },
    }


def appearance_cloud(
    dataset: Path,
    depth_root: Path,
    masks: PackedTissueMasks,
    calibration: dict,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_points, all_colors = [], []
    rng = np.random.RandomState(seed)
    for camera in CAMERAS:
        image = load_rgb(dataset, camera, 0)
        depth = np.load(depth_root / camera / "000000-depth.npy").astype(np.float32)
        valid = masks.get(camera, 0) & np.isfinite(depth) & (depth > 0)
        y, x = np.nonzero(valid)
        take = min(len(x), max(count, count // len(CAMERAS)))
        selected = rng.choice(len(x), size=take, replace=False)
        uv = np.stack((x[selected], y[selected]), axis=1).astype(np.float32)
        world, valid_world = deproject_pixels(
            uv,
            depth,
            calibration[camera]["K"],
            calibration[camera]["X_WC_ros_optical"],
        )
        all_points.append(world[valid_world])
        all_colors.append(image[y[selected][valid_world], x[selected][valid_world]])
    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    # Appearance Gaussians only need dense image coverage; random sampling is
    # the upstream 3DGS initialization convention and avoids an unnecessary
    # O(NK) farthest-point pass over tens of thousands of pixels.
    selected = rng.choice(len(points), size=min(count, len(points)), replace=False)
    return points[selected].astype(np.float32), (colors[selected] / 255.0).astype(np.float32)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("CoTracker/PhysTwin preprocessing requires CUDA")
    if args.output_dir.exists():
        raise FileExistsError(f"拒绝覆盖 {args.output_dir}")
    if not args.cotracker_root.is_dir() or not args.cotracker_weight.is_file():
        raise FileNotFoundError("缺少离线 CoTracker3 源码或 checkpoint")
    set_seed(args.seed)
    dataset = resolve_dataset(REPO_ROOT, args.dataset_key, args.dataset)
    spec = dataset_spec(args.dataset_key)
    frame_count = int(spec["frames"])
    allowed = training_frames(frame_count)
    calibration = load_calibration(dataset)
    masks = PackedTissueMasks(dataset)
    depth_root = dataset / "estimated_depth" / str(spec["depth"])

    sys.path.insert(0, str(args.cotracker_root.resolve()))
    from cotracker.predictor import CoTrackerPredictor

    predictor = CoTrackerPredictor(
        checkpoint=str(args.cotracker_weight.resolve()), offline=True, v2=False, window_len=60
    ).to(args.device).eval()
    tracked = {}
    for camera_index, camera in enumerate(CAMERAS):
        tracked[camera] = track_camera(
            dataset, camera, allowed, masks, predictor, args, camera_index
        )
    del predictor
    torch.cuda.empty_cache()

    candidates_points, candidates_colors, candidates_visible = [], [], []
    for camera in CAMERAS:
        tracks, cotrack_visible = tracked[camera]
        n = tracks.shape[1]
        points = np.repeat(np.zeros((1, n, 3), dtype=np.float32), frame_count, axis=0)
        colors = np.repeat(np.zeros((1, n, 3), dtype=np.float32), frame_count, axis=0)
        visibility = np.zeros((frame_count, n), dtype=bool)
        points[:] = np.nan
        colors[:] = 0.0
        for local, frame in enumerate(allowed):
            depth = np.load(depth_root / camera / f"{frame:06d}-depth.npy").astype(np.float32)
            image = load_rgb(dataset, camera, frame)
            mask_values = sample_image(masks.get(camera, frame), tracks[local]).astype(bool)
            world, valid_depth = deproject_pixels(
                tracks[local], depth, calibration[camera]["K"],
                calibration[camera]["X_WC_ros_optical"]
            )
            valid = cotrack_visible[local] & mask_values & valid_depth
            points[frame] = world
            colors[frame] = sample_image(image, tracks[local]).astype(np.float32) / 255.0
            visibility[frame] = valid
        # Query-time points must be fully defined for persistent particle IDs.
        keep = visibility[0] & np.isfinite(points[0]).all(axis=1)
        candidates_points.append(points[:, keep])
        candidates_colors.append(colors[:, keep])
        candidates_visible.append(visibility[:, keep])

    object_points = np.concatenate(candidates_points, axis=1)
    object_colors = np.concatenate(candidates_colors, axis=1)
    object_visibilities = np.concatenate(candidates_visible, axis=1)
    selected = farthest_point_indices(
        object_points[0], min(args.max_physics_points, object_points.shape[1]), args.seed
    )
    object_points = object_points[:, selected]
    object_colors = object_colors[:, selected]
    object_visibilities = object_visibilities[:, selected]
    # Values behind false masks are inert, but finite placeholders make the
    # serialized input robust to generic readers.
    for point_index in range(object_points.shape[1]):
        last = object_points[0, point_index].copy()
        last_color = object_colors[0, point_index].copy()
        for frame in range(frame_count):
            if object_visibilities[frame, point_index]:
                last = object_points[frame, point_index].copy()
                last_color = object_colors[frame, point_index].copy()
            else:
                object_points[frame, point_index] = last
                object_colors[frame, point_index] = last_color
    object_motions_valid = object_visibilities[:-1] & object_visibilities[1:]
    controller_points, controller_meta = controller_trajectory(
        dataset, object_points[0], args.controller_points, args.seed
    )
    appearance_points, appearance_colors = appearance_cloud(
        dataset, depth_root, masks, calibration, args.appearance_points, args.seed
    )

    args.output_dir.mkdir(parents=True)
    final_data = {
        "object_points": object_points.astype(np.float32),
        "object_colors": object_colors.astype(np.float32),
        "object_visibilities": object_visibilities,
        "object_motions_valid": object_motions_valid,
        "controller_points": controller_points,
        "surface_points": np.zeros((0, 3), dtype=np.float32),
        "interior_points": np.zeros((0, 3), dtype=np.float32),
    }
    with (args.output_dir / "final_data.pkl").open("wb") as stream:
        pickle.dump(final_data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez_compressed(
        args.output_dir / "appearance.npz",
        means_world=appearance_points,
        colors_rgb=appearance_colors,
    )
    metadata = {
        "schema": "fixedsuperbest.phystwin_sim_preprocess.v1",
        "dataset_key": args.dataset_key,
        "dataset": str(dataset),
        "seed": args.seed,
        "phystwin_commit": UPSTREAM_COMMIT,
        "tracker": COTRACKER_VARIANT,
        "tracker_checkpoint": COTRACKER_CHECKPOINT,
        "tracker_checkpoint_sha256": sha256_file(args.cotracker_weight),
        "frame_count": frame_count,
        "future_start": future_start(frame_count),
        "allowed_observation_frames": allowed,
        "opened_observation_frame_count_per_camera": len(allowed),
        "reconstruction_holdouts_opened": False,
        "future_observations_opened": False,
        "evaluation_truth_opened": False,
        "physics_points": int(object_points.shape[1]),
        "appearance_points": int(len(appearance_points)),
        "visibility_fraction_legal_frames": float(object_visibilities[allowed].mean()),
        "controller": controller_meta,
        "shape_of_motion_used": False,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
