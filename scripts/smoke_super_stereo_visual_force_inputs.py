#!/usr/bin/env python3
"""Validate real left/right video timestamps and packed visual-force masks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_gaussians import DatasetManager  # noqa: E402
from embodied_gaussians.embodied_simulator.visual_force_masks import (  # noqa: E402
    MultiCameraPackedTissueVisualForceWeights,
)


LEFT_MASK_DIR = REPO_ROOT / "data/super/grasp5_native/visual_force_masks_v1"
RIGHT_MASK_DIR = (
    REPO_ROOT / "data/super/grasp5_native/visual_force_masks_right_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke real SUPER stereo visual-force inputs."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "data/super/grasp5_offline_demo",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RIGHT_MASK_DIR / "stereo_input_gate_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = DatasetManager(args.dataset)
    dataset.keep_only_cameras(["stereo_left", "stereo_right"])
    provider = MultiCameraPackedTissueVisualForceWeights(
        {
            "stereo_left": LEFT_MASK_DIR,
            "stereo_right": RIGHT_MASK_DIR,
        },
        erosion_radius_px=7,
        highlight_weight=0.10,
    )
    dataset.set_visual_force_weight_provider(provider)
    left_timestamps = dataset.offline_cameras.cameras["stereo_left"].timestamps
    sample_times = [
        float(left_timestamps[0]),
        float(left_timestamps[len(left_timestamps) // 2]),
        float(left_timestamps[-1]),
    ]
    samples = []
    all_active = True
    all_native_timestamps = True
    all_nonempty = True
    all_finite = True
    for requested_timestamp in sample_times:
        dataset.update_frames(requested_timestamp)
        weights = dataset.frames.loss_weights_gpu
        if weights is None:
            raise RuntimeError("Stereo provider did not install loss weights")
        active = provider.last_active_cameras.copy()
        all_active &= active == ["stereo_left", "stereo_right"]
        weight_sums = weights.sum(dim=(1, 2))
        all_nonempty &= bool((weight_sums > 0.0).all().item())
        all_finite &= bool(
            torch.isfinite(weights).all().item()
            and torch.isfinite(dataset.frames.colors_gpu).all().item()
        )
        camera_timestamps = {}
        expected_timestamps = {}
        for camera_name in dataset.frames.names:
            camera_index = dataset.frames.names.index(camera_name)
            camera = dataset.offline_cameras.cameras[camera_name]
            expected_index = int(
                np.searchsorted(
                    camera.timestamps,
                    requested_timestamp,
                    side="right",
                )
                - 1
            )
            expected_index = max(0, min(expected_index, len(camera.timestamps) - 1))
            expected_timestamp = float(camera.timestamps[expected_index])
            actual_timestamp = float(dataset.frames.timestamps[camera_index])
            all_native_timestamps &= actual_timestamp == expected_timestamp
            camera_timestamps[camera_name] = actual_timestamp
            expected_timestamps[camera_name] = expected_timestamp
        samples.append(
            {
                "requested_timestamp": requested_timestamp,
                "decoded_timestamps": camera_timestamps,
                "expected_decoded_timestamps": expected_timestamps,
                "active_cameras": active,
                "weight_sums": {
                    name: float(weight_sums[index].item())
                    for index, name in enumerate(dataset.frames.names)
                },
                "provider": provider.last_camera_statistics,
            }
        )

    gates = {
        "both_cameras_active_first_middle_last": all_active,
        "decoded_native_timestamps_used": all_native_timestamps,
        "both_camera_weights_nonempty": all_nonempty,
        "frames_and_weights_finite": all_finite,
        "both_assets_cover_1441_frames": all(
            len(item.timestamps) == 1441
            for item in provider.providers.values()
        ),
    }
    report = {
        "stage": "stereo_visual_force_real_input_gate",
        "camera_names": dataset.frames.names,
        "samples": samples,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Stereo visual-force input gate failed")


if __name__ == "__main__":
    main()
