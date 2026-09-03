#!/usr/bin/env python3
"""Run one real stereo visual-force backward pass on the SUPER tissue Gaussians."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_gaussians import Body, DatasetManager  # noqa: E402
from embodied_gaussians.embodied_simulator.builder import (  # noqa: E402
    EmbodiedGaussiansBuilder,
)
from embodied_gaussians.embodied_simulator.simulator import (  # noqa: E402
    EmbodiedGaussiansSimulator,
)
from embodied_gaussians.embodied_simulator.visual_force_masks import (  # noqa: E402
    MultiCameraPackedTissueVisualForceWeights,
)
from embodied_gaussians.embodied_simulator.visual_forces import (  # noqa: E402
    VisualForcesSettings,
)


TISSUE_PATH = (
    REPO_ROOT
    / "data/super/grasp5_native/bodies_v9_dense_0p5mm_rigid_tissue/tissue.json"
)
LEFT_MASK_DIR = REPO_ROOT / "data/super/grasp5_native/visual_force_masks_v1"
RIGHT_MASK_DIR = (
    REPO_ROOT / "data/super/grasp5_native/visual_force_masks_right_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke one real SUPER stereo visual-force backward pass."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "data/super/grasp5_offline_demo",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RIGHT_MASK_DIR / "stereo_backward_gate_report.json",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def build_visual_only_tissue_simulator(device: str) -> EmbodiedGaussiansSimulator:
    """Build the real tissue appearance without its 79,567 collision spheres."""
    tissue = Body.model_validate_json(TISSUE_PATH.read_text())
    if tissue.gaussians is None:
        raise ValueError(f"Tissue body has no Gaussians: {TISSUE_PATH}")

    builder = EmbodiedGaussiansBuilder()
    X_WB = np.asarray(tissue.X_WB, dtype=np.float32)
    quaternion_xyzw = wp.quat_from_matrix(X_WB[:3, :3])
    body_id = builder.add_body(
        origin=wp.transformf(*X_WB[:3, 3], *quaternion_xyzw)
    )
    gaussians = tissue.gaussians
    builder.gaussian_means.extend(gaussians.means)
    builder.gaussian_quats.extend(gaussians.quats)
    builder.gaussian_scales.extend(gaussians.scales)
    builder.gaussian_opacities.extend(gaussians.opacities)
    builder.gaussian_colors.extend(gaussians.colors)
    builder.gaussian_body_ids.extend([body_id] * len(gaussians))
    builder.bodies_affected_by_visual_forces.append(body_id)

    simulator = EmbodiedGaussiansSimulator(builder, device=device)
    simulator.update_gaussian_transforms()
    return simulator


def main() -> None:
    args = parse_args()
    wp.init()
    simulator = build_visual_only_tissue_simulator(args.device)

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
    requested_timestamp = float(left_timestamps[len(left_timestamps) // 2])
    dataset.update_frames(requested_timestamp)

    settings = VisualForcesSettings(
        iterations=1,
        lr_means=0.0001,
        lr_quats=0.0001,
        kp=1.0,
        observations_are_bgr=True,
        reset_optimizer_each_step=True,
        max_force=0.005,
        max_moment=0.00005,
        robust_loss_beta=0.05,
    )
    simulator.state_0.body_f.zero_()
    simulator.compute_visual_forces(settings, dataset.frames, dt=1.0 / 1200.0)
    wp.synchronize_device(simulator.model.device)

    body_force = wp.to_torch(simulator.state_0.body_f).detach().clone()
    spatial_force_norm = torch.linalg.vector_norm(body_force, dim=1)
    camera_losses = simulator.last_visual_force_camera_losses
    weight_sums = dataset.frames.loss_weights_gpu.sum(dim=(1, 2))
    gates = {
        "both_cameras_active": provider.last_active_cameras
        == ["stereo_left", "stereo_right"],
        "both_camera_losses_reported": all(
            camera_losses.get(name) is not None
            and np.isfinite(float(camera_losses[name]))
            for name in dataset.frames.names
        ),
        "combined_loss_finite": bool(
            torch.isfinite(simulator.last_visual_force_loss).item()
        ),
        "both_weight_maps_nonempty": bool((weight_sums > 0.0).all().item()),
        "body_force_finite": bool(torch.isfinite(body_force).all().item()),
        "body_force_nonzero": bool(spatial_force_norm.max().item() > 0.0),
        "linear_force_clamped": bool(
            torch.linalg.vector_norm(body_force[:, 3:], dim=1).max().item()
            <= settings.max_force + 1.0e-7
        ),
        "moment_clamped": bool(
            torch.linalg.vector_norm(body_force[:, :3], dim=1).max().item()
            <= settings.max_moment + 1.0e-7
        ),
    }
    report = {
        "stage": "stereo_visual_force_real_backward_gate",
        "requested_timestamp": requested_timestamp,
        "decoded_timestamps": {
            name: float(dataset.frames.timestamps[index])
            for index, name in enumerate(dataset.frames.names)
        },
        "active_cameras": provider.last_active_cameras,
        "camera_losses": camera_losses,
        "equal_camera_mean_loss": float(
            simulator.last_visual_force_loss.detach().item()
        ),
        "weight_sums": {
            name: float(weight_sums[index].item())
            for index, name in enumerate(dataset.frames.names)
        },
        "spatial_body_force": body_force.cpu().tolist(),
        "gates": gates,
        "passed": bool(all(gates.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Stereo visual-force backward smoke failed")


if __name__ == "__main__":
    main()
