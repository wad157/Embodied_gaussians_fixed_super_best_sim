#!/usr/bin/env python3
"""Immutable definitions for the TRACE-on-SIM baseline protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple


UPSTREAM_COMMIT = "a4597585bc0e56c56922abe75be9198eb119c95a"
ADAPTER_VERSION = "trace_sim_v1_rgb_only_full_k"
TRACK_DECODER_VERSION = "query_anchored_gaussian_displacement_v1"
PROTOCOL = "joint_reconstruction_7to1_future_80to20"
QUERY_FRAME = 0
HOLDOUT_STRIDE = 8
HOLDOUT_OFFSET = 7
CAMERAS = ("stereo_left", "stereo_right")

# TRACE's positional encodings and public scenes operate at roughly unit scale.
# SIM is recorded in SI meters, so this fixed, dataset-independent conversion
# expresses the reconstruction in decimeters.  It is not fitted to evaluation GT.
INTERNAL_UNITS_PER_METER = 10.0
DEFAULT_TRAIN_DOWNSAMPLE = 2
DEFAULT_INIT_POINTS = 50000
DEFAULT_ITERATIONS = 40000

DATASETS: Mapping[str, Mapping[str, object]] = {
    "sim01": {
        "name": "tissue_retraction_free_support_front_v2",
        "frames": 300,
    },
    "sim02": {
        "name": "tissue_retraction_free_support_side_v2",
        "frames": 300,
    },
    "sim03": {
        "name": "tissue_long_edge_lift_return_sufia_v2_lift30mm",
        "frames": 360,
    },
}


def dataset_spec(key: str) -> Mapping[str, object]:
    try:
        return DATASETS[key.lower()]
    except KeyError as error:
        raise ValueError("Unknown SIM dataset {}; choose from {}".format(key, sorted(DATASETS))) from error


def future_start(frame_count: int) -> int:
    return int(frame_count) * 4 // 5


def max_observed_time(frame_count: int) -> float:
    """Last timestamp in the observed interval, including held-out interpolation."""
    return float(future_start(frame_count) - 1) / float(frame_count - 1)


def normalized_time(frame: int, frame_count: int) -> float:
    return float(frame) / float(frame_count - 1)


def reconstruction_holdouts(frame_count: int) -> List[int]:
    split = future_start(frame_count)
    return [frame for frame in range(split) if frame % HOLDOUT_STRIDE == HOLDOUT_OFFSET]


def training_frames(frame_count: int) -> List[int]:
    split = future_start(frame_count)
    return [frame for frame in range(split) if frame % HOLDOUT_STRIDE != HOLDOUT_OFFSET]


def future_frames(frame_count: int) -> List[int]:
    return list(range(future_start(frame_count), frame_count))


def rendering_frames(frame_count: int) -> List[int]:
    return reconstruction_holdouts(frame_count) + future_frames(frame_count)


def resolve_dataset(repo_root: Path, key: str, supplied: Optional[Path] = None) -> Path:
    expected = (repo_root / "data" / "sim" / str(dataset_spec(key)["name"])).resolve()
    actual = expected if supplied is None else supplied.expanduser().resolve()
    if actual != expected:
        raise ValueError("Formal TRACE baseline requires {}; received {}".format(expected, actual))
    if not actual.is_dir():
        raise FileNotFoundError(actual)
    return actual


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_world_points(
    points_world,
    intrinsic,
    world_from_camera,
    image_size_wh: Tuple[int, int],
):
    import numpy as np

    points = np.asarray(points_world, dtype=np.float64)
    camera_from_world = np.linalg.inv(np.asarray(world_from_camera, dtype=np.float64))
    homogeneous = np.concatenate(
        (points, np.ones((points.shape[0], 1), dtype=np.float64)), axis=1
    )
    camera = (camera_from_world @ homogeneous.T).T[:, :3]
    z = camera[:, 2]
    pixels_h = (np.asarray(intrinsic, dtype=np.float64) @ camera.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = pixels_h[:, :2] / z[:, None]
    width, height = image_size_wh
    valid = (
        np.isfinite(camera).all(axis=1)
        & np.isfinite(uv).all(axis=1)
        & (z > 0.0)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] <= width - 1.0)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] <= height - 1.0)
    )
    return uv.astype(np.float32), valid.astype(bool), camera.astype(np.float32)
