#!/usr/bin/env python3
"""Immutable protocol and disclosed parameters for the Embodied Gaussians baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


# This is the last upstream RAI reference commit mirrored by the local checkout.
# The public repository explicitly omits paper shape matching; this adapter adds
# only equations (4)--(5) and Algorithm 2 from the paper in shape_matching.py.
UPSTREAM_COMMIT = "c97ec671f97af25985e0af8844c0aac8d8119b97"
UPSTREAM_URL = "https://github.com/rai-opensource/embodied_gaussians"
PAPER_URL = "https://arxiv.org/html/2406.10788"
UPSTREAM_FILE_SHA256 = {
    "README.md": "160130d72d3b1b098cc48fc97aea7cf553f452e91c66895db6d367be6adba521",
    "src/embodied_gaussians/scene_builders/simple_body_builder.py": "055a380050df20ebd1b1ce9bad04b9bbf969206554db41e6c45a65c9719b2aa1",
    "src/embodied_gaussians/scene_builders/domain.py": "2902c5062f6306be82176607fa2e34f0e3847ddec0cb00b55cbb8ed07ae81966",
}

CAMERAS = ("stereo_left", "stereo_right")
INITIALIZATION_FRAME = 0
HOLDOUT_STRIDE = 8
HOLDOUT_OFFSET = 7
QUERY_FRAME = 0

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

# Values stated in the paper/supplement or in the pinned public defaults.  The
# public SimpleBodyBuilder default is 6 mm, inside the paper's reported 4--7 mm
# interval.  The paper does not publish a deformable k_S value, so the baseline
# introduces no fitted stiffness: k_S=1 means apply Eq. (5)'s full constraint
# projection.  Neighbours use Delaunay geometric adjacency, not a tuned k-NN.
PAPER_PARAMETERS = {
    "fps": 30,
    "dt_s": 1.0 / 30.0,
    "substeps": 20,
    "jacobi_iterations": 4,
    "velocity_damping_per_frame": 0.9,
    # Paper default for every non-rope object.  The reported 0.2/0.3 kg
    # exceptions apply only to the real/simulated rope experiments.
    "particle_mass_kg": 0.1,
    "gravity_m_s2": -9.80665,
    "initial_particle_iterations": 80,
    "initial_gaussian_iterations": 250,
    "initial_opacity_threshold": 0.3,
    "visual_iterations": 5,
    "visual_lr_position": 1.0e-3,
    "visual_lr_rotation": 1.0e-4,
    "visual_lr_color": 5.0e-4,
    "visual_lr_opacity": 5.0e-4,
    "visual_kp": 60.0,
    "visual_displacement_deadzone_m": 0.002,
    "online_width": 640,
    "shape_constraint_projection": 1.0,
    "particle_radius_m": 0.006,
}

def dataset_spec(key: str) -> Mapping[str, object]:
    try:
        return DATASETS[key.lower()]
    except KeyError as error:
        raise ValueError(f"未知数据集键：{key}；应为 {sorted(DATASETS)}") from error


def resolve_dataset(repo_root: Path, key: str, supplied: Path | None = None) -> Path:
    expected = (repo_root / "data" / "sim" / str(dataset_spec(key)["name"])).resolve()
    actual = expected if supplied is None else supplied.expanduser().resolve()
    if actual != expected:
        raise ValueError(f"正式 baseline 只接受固定目录：{expected}；收到 {actual}")
    if not actual.is_dir():
        raise FileNotFoundError(f"找不到正式数据集：{actual}")
    return actual


def future_start(frame_count: int) -> int:
    return int(frame_count) * 4 // 5


def observation_allowed(frame_index: int, frame_count: int) -> bool:
    return (
        frame_index < future_start(frame_count)
        and frame_index % HOLDOUT_STRIDE != HOLDOUT_OFFSET
    )


def reconstruction_holdouts(frame_count: int) -> list[int]:
    return [
        index
        for index in range(future_start(frame_count))
        if index % HOLDOUT_STRIDE == HOLDOUT_OFFSET
    ]


def future_frames(frame_count: int) -> list[int]:
    return list(range(future_start(frame_count), frame_count))


def rendering_frames(frame_count: int) -> list[int]:
    return reconstruction_holdouts(frame_count) + future_frames(frame_count)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def project_world_points(points_world, intrinsic, world_from_camera, image_size_wh):
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
    width, height = (int(value) for value in image_size_wh)
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
