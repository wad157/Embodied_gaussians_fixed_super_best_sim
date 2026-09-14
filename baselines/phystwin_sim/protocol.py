#!/usr/bin/env python3
"""Immutable PhysTwin SIM evaluation protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional


UPSTREAM_COMMIT = "81c718790a37e5e0102eb77af2c6edd34a9db25f"
COTRACKER_VARIANT = "cotracker3_scaled_offline"
COTRACKER_CHECKPOINT = "scaled_offline.pth"
QUERY_FRAME = 0
HOLDOUT_STRIDE = 8
HOLDOUT_OFFSET = 7
CAMERAS = ("stereo_left", "stereo_right")

DATASETS: Mapping[str, Mapping[str, object]] = {
    "sim01": {
        "name": "tissue_retraction_free_support_front_v2",
        "frames": 300,
        "depth": "foundation_stereo_rgb_rig_aware_v1",
    },
    "sim02": {
        "name": "tissue_retraction_free_support_side_v2",
        "frames": 300,
        "depth": "foundation_stereo_rgb_rig_aware_v1",
    },
    "sim03": {
        "name": "tissue_long_edge_lift_return_sufia_v2_lift30mm",
        "frames": 360,
        "depth": "foundation_stereo_rgb_v1",
    },
}


def dataset_spec(key: str) -> Mapping[str, object]:
    try:
        return DATASETS[key.lower()]
    except KeyError as error:
        raise ValueError(f"未知数据集键 {key}; 应为 {sorted(DATASETS)}") from error


def future_start(frame_count: int) -> int:
    return int(frame_count) * 4 // 5


def training_frames(frame_count: int) -> list[int]:
    split = future_start(frame_count)
    return [i for i in range(split) if i % HOLDOUT_STRIDE != HOLDOUT_OFFSET]


def reconstruction_holdouts(frame_count: int) -> list[int]:
    split = future_start(frame_count)
    return [i for i in range(split) if i % HOLDOUT_STRIDE == HOLDOUT_OFFSET]


def future_frames(frame_count: int) -> list[int]:
    return list(range(future_start(frame_count), frame_count))


def rendering_frames(frame_count: int) -> list[int]:
    return reconstruction_holdouts(frame_count) + future_frames(frame_count)


def resolve_dataset(repo_root: Path, key: str, supplied: Optional[Path] = None) -> Path:
    expected = (repo_root / "data" / "sim" / str(dataset_spec(key)["name"])).resolve()
    actual = expected if supplied is None else supplied.expanduser().resolve()
    if actual != expected:
        raise ValueError(f"正式协议只接受 {expected}; 收到 {actual}")
    if not actual.is_dir():
        raise FileNotFoundError(actual)
    return actual


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def project_world_points(points_world, intrinsic, world_from_camera, resolution_wh):
    import numpy as np

    points = np.asarray(points_world, dtype=np.float64)
    camera_from_world = np.linalg.inv(np.asarray(world_from_camera, dtype=np.float64))
    homogeneous = np.concatenate(
        (points, np.ones((points.shape[0], 1), dtype=np.float64)), axis=1
    )
    camera = (camera_from_world @ homogeneous.T).T[:, :3]
    z = camera[:, 2]
    pixels_h = (np.asarray(intrinsic, dtype=np.float64) @ camera.T).T
    uv = pixels_h[:, :2] / z[:, None]
    width, height = resolution_wh
    valid = (
        np.isfinite(camera).all(axis=1)
        & np.isfinite(uv).all(axis=1)
        & (z > 0.0)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] <= width - 1)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] <= height - 1)
    )
    return uv.astype(np.float32), valid.astype(bool), camera.astype(np.float32)
