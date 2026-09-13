#!/usr/bin/env python3
"""Shared camera, RGB-layer, and PNG I/O for the two EG audit variants."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from protocol import CAMERAS, read_json


class StereoInputs:
    def __init__(self, dataset: Path, online_width: int):
        cameras = read_json(dataset / "cameras.json")
        self.dataset = dataset
        self.online_width = int(online_width)
        self.calibration = {}
        self.packed_masks = {}
        self.mask_width = {}
        for name in CAMERAS:
            metadata = read_json(dataset / "videos" / f"{name}.json")
            width, height = (int(value) for value in metadata["resolution"])
            intrinsic = np.asarray(metadata["K"], dtype=np.float32)
            world_from_camera = np.asarray(
                cameras[name]["X_WC_ros_optical"], dtype=np.float32
            )
            self.calibration[name] = {
                "K": intrinsic,
                "X_WC": world_from_camera,
                "X_CW": np.linalg.inv(world_from_camera).astype(np.float32),
                "resolution": (width, height),
                "timestamps": np.asarray(metadata["timestamps"], dtype=np.float64),
            }
            mask_root = dataset / "gui_assets" / "visual_force_masks" / name
            report = read_json(mask_root / "report.json")
            self.mask_width[name] = int(report["resolution_wh"][0])
            self.packed_masks[name] = np.load(
                mask_root / "tissue_masks_packbits.npy", mmap_mode="r"
            )

    def load_online(self, camera: str, frame: int, device: torch.device):
        image = np.asarray(
            Image.open(
                self.dataset / "rgb" / camera / f"{frame:06d}.png"
            ).convert("RGB"),
            dtype=np.float32,
        ) / 255.0
        packed = np.asarray(self.packed_masks[camera][frame])
        mask = (
            np.unpackbits(packed, axis=1, bitorder="big")[:, : self.mask_width[camera]]
            > 0
        )
        source_height, source_width = image.shape[:2]
        target_height = int(round(source_height * self.online_width / source_width))
        size = (self.online_width, target_height)
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        mask = (
            cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
            > 0
        )
        image = image * mask[..., None].astype(np.float32)
        intrinsic = self.calibration[camera]["K"].copy()
        intrinsic[0, :] *= self.online_width / source_width
        intrinsic[1, :] *= target_height / source_height
        intrinsic[2, :] = (0.0, 0.0, 1.0)
        return (
            torch.as_tensor(np.ascontiguousarray(image), device=device),
            torch.as_tensor(intrinsic, device=device).unsqueeze(0),
            torch.as_tensor(
                self.calibration[camera]["X_CW"], device=device
            ).unsqueeze(0),
            size,
        )

    def full_camera_tensors(self, camera: str, device: torch.device):
        calibration = self.calibration[camera]
        return (
            torch.as_tensor(calibration["K"], device=device).unsqueeze(0),
            torch.as_tensor(calibration["X_CW"], device=device).unsqueeze(0),
            calibration["resolution"],
        )


def save_render(path: Path, tensor: torch.Tensor, *, alpha: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = tensor.detach().clamp(0.0, 1.0)
    array = value.mul(255.0).add(0.5).byte().cpu().numpy()
    if alpha:
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        Image.fromarray(array, mode="L").save(path)
    else:
        Image.fromarray(array, mode="RGB").save(path)
