#!/usr/bin/env python3
"""Test stereo mask coverage and equal-per-camera visual loss."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_gaussians.embodied_simulator.frames import Frames  # noqa: E402
from embodied_gaussians.embodied_simulator.simulator import (  # noqa: E402
    equal_camera_weighted_loss,
)
from embodied_gaussians.embodied_simulator.visual_force_masks import (  # noqa: E402
    MultiCameraPackedTissueVisualForceWeights,
)


def write_asset(path: Path, timestamps: list[float], masks: np.ndarray) -> None:
    path.mkdir(parents=True)
    height, width = masks.shape[1:]
    np.save(path / "timestamps.npy", np.asarray(timestamps, dtype=np.float64))
    np.save(path / "tissue_masks_packbits.npy", np.packbits(masks, axis=2))
    (path / "report.json").write_text(
        json.dumps(
            {
                "passed": True,
                "resolution_wh": [width, height],
                "frame_count": len(timestamps),
            }
        )
        + "\n"
    )


def make_frames() -> Frames:
    height, width = 3, 4
    return Frames(
        width=width,
        height=height,
        names=["stereo_left", "stereo_right"],
        timestamps=[0.0, 0.0],
        Ks_cpu=torch.eye(3).repeat(2, 1, 1),
        Ks_gpu=torch.eye(3).repeat(2, 1, 1),
        X_WCs_cpu=torch.eye(4).repeat(2, 1, 1),
        X_CWs_opencv_gpu=torch.eye(4).repeat(2, 1, 1),
        colors_gpu=torch.zeros((2, height, width, 3)),
        device="cpu",
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="super-stereo-mask-") as directory:
        root = Path(directory)
        full_masks = np.ones((2, 3, 4), dtype=bool)
        partial_masks = np.ones((1, 3, 4), dtype=bool)
        write_asset(root / "left", [0.0, 1.0], full_masks)
        write_asset(root / "right", [0.0], partial_masks)
        provider = MultiCameraPackedTissueVisualForceWeights(
            {
                "stereo_left": root / "left",
                "stereo_right": root / "right",
            },
            erosion_radius_px=0,
            highlight_weight=1.0,
        )
        frames = make_frames()

        frames.timestamps = [0.0, 0.0]
        both_weights = provider.update_frames(frames, 0.0).clone()
        both_active = provider.last_active_cameras.copy()

        frames.timestamps = [1.0, 1.0]
        partial_weights = provider.update_frames(frames, 1.0).clone()
        partial_active = provider.last_active_cameras.copy()

    pixel_loss = torch.stack(
        (
            torch.ones((3, 4)),
            torch.full((3, 4), 3.0),
        )
    )
    unequal_area_weights = torch.zeros_like(pixel_loss)
    unequal_area_weights[0] = 1.0
    unequal_area_weights[1, 0, 0] = 1.0
    total_loss, camera_losses, active = equal_camera_weighted_loss(
        pixel_loss,
        unequal_area_weights,
    )
    left_only_loss, _, left_only_active = equal_camera_weighted_loss(
        pixel_loss,
        torch.stack((torch.ones((3, 4)), torch.zeros((3, 4)))),
    )

    gates = {
        "both_cameras_active_in_shared_range": (
            both_active == ["stereo_left", "stereo_right"]
            and bool((both_weights.sum(dim=(1, 2)) > 0.0).all().item())
        ),
        "partial_right_asset_not_clamped_after_end": (
            partial_active == ["stereo_left"]
            and bool(partial_weights[0].sum().item() > 0.0)
            and bool(partial_weights[1].sum().item() == 0.0)
        ),
        "camera_losses_independent_of_mask_area": bool(
            torch.allclose(camera_losses, torch.tensor([1.0, 3.0]))
        ),
        "active_cameras_are_equal_weighted": bool(
            active.all().item()
            and torch.isclose(total_loss, torch.tensor(2.0)).item()
        ),
        "single_active_camera_uses_its_own_loss": bool(
            left_only_active.tolist() == [True, False]
            and torch.isclose(left_only_loss, torch.tensor(1.0)).item()
        ),
    }
    report = {
        "stage": "stereo_visual_force_weighting_unit_gate",
        "camera_losses": camera_losses.tolist(),
        "equal_camera_loss": float(total_loss.item()),
        "both_active": both_active,
        "partial_active": partial_active,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Stereo visual-force weighting gate failed")


if __name__ == "__main__":
    main()
