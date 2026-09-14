#!/usr/bin/env python3
"""Shared, observation-only SIM input helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from protocol import CAMERAS, read_json


class PackedTissueMasks:
    def __init__(self, dataset: Path):
        self.arrays = {}
        self.widths = {}
        for camera in CAMERAS:
            root = dataset / "gui_assets" / "visual_force_masks" / camera
            report = read_json(root / "report.json")
            self.widths[camera] = int(report["resolution_wh"][0])
            self.arrays[camera] = np.load(
                root / "tissue_masks_packbits.npy", mmap_mode="r"
            )

    def get(self, camera: str, frame: int) -> np.ndarray:
        packed = np.asarray(self.arrays[camera][frame])
        return np.unpackbits(packed, axis=1, bitorder="big")[
            :, : self.widths[camera]
        ].astype(bool)


def load_calibration(dataset: Path) -> dict[str, dict]:
    cameras = read_json(dataset / "cameras.json")
    result = {}
    for name in CAMERAS:
        video = read_json(dataset / "videos" / f"{name}.json")
        result[name] = {
            "K": np.asarray(video["K"], dtype=np.float64),
            "timestamps": np.asarray(video["timestamps"], dtype=np.float64),
            "resolution_wh": tuple(int(x) for x in video["resolution"]),
            "X_WC_ros_optical": np.asarray(
                cameras[name]["X_WC_ros_optical"], dtype=np.float64
            ),
        }
    return result


def load_rgb(dataset: Path, camera: str, frame: int) -> np.ndarray:
    return np.asarray(
        Image.open(dataset / "rgb" / camera / f"{frame:06d}.png").convert("RGB")
    )


def sample_image(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x = np.rint(uv[:, 0]).astype(np.int64)
    y = np.rint(uv[:, 1]).astype(np.int64)
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    values = np.zeros((len(uv),) + image.shape[2:], dtype=image.dtype)
    values[valid] = image[y[valid], x[valid]]
    return values


def deproject_pixels(
    uv: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    world_from_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    x = np.rint(uv[:, 0]).astype(np.int64)
    y = np.rint(uv[:, 1]).astype(np.int64)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    z = np.zeros(len(uv), dtype=np.float64)
    z[inside] = depth[y[inside], x[inside]]
    valid = inside & np.isfinite(z) & (z > 0.0)
    camera = np.stack(
        (
            (uv[:, 0] - intrinsic[0, 2]) / intrinsic[0, 0] * z,
            (uv[:, 1] - intrinsic[1, 2]) / intrinsic[1, 1] * z,
            z,
            np.ones(len(uv), dtype=np.float64),
        ),
        axis=1,
    )
    world = (world_from_camera @ camera.T).T[:, :3]
    valid &= np.isfinite(world).all(axis=1)
    return world.astype(np.float32), valid


def resize_rgb(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def farthest_point_indices(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Deterministic O(NK) farthest-point sampling."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= count:
        return np.arange(len(points), dtype=np.int64)
    rng = np.random.RandomState(seed)
    chosen = np.empty(count, dtype=np.int64)
    chosen[0] = int(rng.randint(len(points)))
    distances = np.full(len(points), np.inf, dtype=np.float64)
    for i in range(1, count):
        delta = points - points[chosen[i - 1]]
        distances = np.minimum(distances, np.einsum("ij,ij->i", delta, delta))
        chosen[i] = int(np.argmax(distances))
    return chosen
