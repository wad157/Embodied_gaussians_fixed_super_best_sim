#!/usr/bin/env python3
"""Run one real masked RGB optimizer step through soft particle physics."""

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
EXAMPLES_DIR = REPO_ROOT / "examples"
for path in (SRC_DIR, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from embodied_environments.super_embodied.super_embodied import (  # noqa: E402
    build_environment,
)
from embodied_gaussians import DatasetManager  # noqa: E402
from embodied_gaussians.embodied_simulator.visual_force_masks import (  # noqa: E402
    PackedTissueVisualForceWeights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke masked SUPER visual force.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "data/super/grasp5_offline_demo",
    )
    parser.add_argument(
        "--mask-asset",
        type=Path,
        default=REPO_ROOT / "data/super/grasp5_native/visual_force_masks_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "data/super/grasp5_native/soft_tissue_v1/masked_visual_force_smoke.json",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wp.init()
    environment = build_environment(num_envs=1, add_gaussians=True, device=args.device)
    dataset = DatasetManager(args.dataset)
    dataset.keep_only_cameras(["stereo_left"])
    provider = PackedTissueVisualForceWeights(
        args.mask_asset,
        camera_name="stereo_left",
        erosion_radius_px=7,
        highlight_weight=0.10,
    )
    dataset.set_visual_force_weight_provider(provider)
    timestamp = float(dataset.offline_cameras.cameras["stereo_left"].timestamps[0])
    dataset.update_frames(timestamp)
    environment.frames = dataset.frames

    settings = environment.visual_forces_settings
    settings.iterations = 1
    settings.enable_soft_particle_forces = True
    settings.require_loss_weights_for_soft = True
    environment.sim.state_0.particle_f.zero_()
    environment.sim.state_0.body_f.zero_()
    environment.sim.compute_visual_forces(
        settings,
        environment.frames,
        environment.physics_settings.dt / environment.physics_settings.substeps,
    )
    wp.synchronize_device(environment.sim.model.device)

    particle_force = wp.to_torch(environment.sim.state_0.particle_f).clone()
    body_force = wp.to_torch(environment.sim.state_0.body_f).clone()
    particle_force_norm = torch.linalg.vector_norm(particle_force, dim=1)
    body_force_norm = torch.linalg.vector_norm(body_force, dim=1)
    weights = environment.frames.loss_weights_gpu
    if weights is None:
        raise RuntimeError("Visual-force weights disappeared before optimization")
    soft_ids = environment.sim.gaussian_model.soft_gaussian_ids.long()
    gradient_enabled = (
        ~environment.sim.visual_forces._gaussians_not_involved_in_visual_forces
    )

    environment.sim.physics_step(environment.physics_settings)
    environment.sim.update_gaussian_transforms()
    wp.synchronize_device(environment.sim.model.device)
    positions = wp.to_torch(environment.sim.state_0.particle_q)
    tet_indices = wp.to_torch(environment.sim.model.tet_indices).long()
    tet_positions = positions[tet_indices]
    ds = torch.stack(
        (
            tet_positions[:, 1] - tet_positions[:, 0],
            tet_positions[:, 2] - tet_positions[:, 0],
            tet_positions[:, 3] - tet_positions[:, 0],
        ),
        dim=-1,
    )
    volume = torch.linalg.det(ds) / 6.0
    inverse_rest = torch.as_tensor(
        np.asarray(environment.builder().tet_poses),
        device=positions.device,
        dtype=torch.float32,
    )
    rest_volume = torch.linalg.det(inverse_rest).reciprocal() / 6.0

    gates = {
        "pixel_weights_present": bool(weights.sum().item() > 0.0),
        "mask_timestamp_exact": provider.last_frame_index == 0,
        "only_soft_gaussian_gradients": bool(
            gradient_enabled[soft_ids].all().item()
            and int(gradient_enabled.sum().item())
            == int(environment.sim.gaussian_model.num_soft_gaussians)
        ),
        "particle_force_finite": bool(torch.isfinite(particle_force).all().item()),
        "particle_force_nonzero": bool(particle_force_norm.max().item() > 0.0),
        "particle_force_budget": bool(
            particle_force_norm.sum().item() <= settings.soft_max_total_force + 1.0e-8
        ),
        "psm_body_force_zero": bool(body_force_norm.max().item() == 0.0),
        "post_step_finite": bool(
            torch.isfinite(positions).all().item()
            and torch.isfinite(environment.sim.gaussian_state.means).all().item()
        ),
        "post_step_no_inverted_tetrahedra": bool((volume <= 0.0).sum().item() == 0),
        "post_step_volume_error_below_2_percent": bool(
            abs(float((volume.sum() / rest_volume.sum()).item()) - 1.0) < 0.02
        ),
    }
    report = {
        "stage": "masked_RGB_visual_force_one_step_smoke",
        "device": str(environment.sim.model.device),
        "timestamp": timestamp,
        "mask": {
            "frame_index": provider.last_frame_index,
            "valid_pixels_after_erosion": provider.last_valid_pixels,
            "highlight_pixels_downweighted": provider.last_highlight_pixels,
            "weight_sum": float(weights.sum().item()),
        },
        "force": {
            "max_particle_force_n": float(particle_force_norm.max().item()),
            "particle_force_budget_n": float(particle_force_norm.sum().item()),
            "max_psm_body_force": float(body_force_norm.max().item()),
        },
        "post_physics": {
            "inverted_tetrahedra": int((volume <= 0.0).sum().item()),
            "total_volume_ratio": float((volume.sum() / rest_volume.sum()).item()),
        },
        "gates": gates,
        "passed": bool(all(gates.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Masked visual-force smoke failed")


if __name__ == "__main__":
    main()
