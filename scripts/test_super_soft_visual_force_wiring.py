#!/usr/bin/env python3
"""Smoke the optimizer-target to soft-particle visual-force wiring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import warp as wp


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_gaussians.embodied_simulator.builder import (  # noqa: E402
    EmbodiedGaussiansBuilder,
)
from embodied_gaussians.embodied_simulator.frames import Frames  # noqa: E402
from embodied_gaussians.embodied_simulator.simulator import (  # noqa: E402
    EmbodiedGaussiansSimulator,
)
from embodied_gaussians.embodied_simulator.visual_forces import (  # noqa: E402
    VisualForcesSettings,
)
from embodied_gaussians.scene_builders.domain import SoftBody  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test soft visual-force wiring.")
    parser.add_argument(
        "--asset",
        type=Path,
        default=REPO_ROOT
        / "data/super/grasp5_native/soft_tissue_v1/tissue_soft.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "data/super/grasp5_native/soft_tissue_v1/visual_force_wiring_report.json",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--warp-cache-dir", type=Path, default=Path("/tmp/warp-super-visual-cache")
    )
    return parser.parse_args()


def make_simulator(asset_path: Path, device: str) -> EmbodiedGaussiansSimulator:
    soft_body = SoftBody.from_npz(asset_path, name="super_tissue")
    local = EmbodiedGaussiansBuilder(up_vector=(0.0, 0.0, 1.0))
    local.add_soft_body(
        soft_body,
        young_modulus_pa=15000.0,
        poisson_ratio=0.45,
        anchor_mode="support_candidate",
        add_gaussians=True,
    )
    builder = EmbodiedGaussiansBuilder(up_vector=(0.0, 0.0, 1.0))
    builder.gaussian_means.append([1.0, 1.0, 1.0])
    builder.gaussian_quats.append([1.0, 0.0, 0.0, 0.0])
    builder.gaussian_scales.append([0.001, 0.001, 0.001])
    builder.gaussian_opacities.append(0.5)
    builder.gaussian_colors.append([0.5, 0.5, 0.5])
    builder.gaussian_body_ids.append(-1)
    builder.add_builder(local)
    return EmbodiedGaussiansSimulator(builder, device=device)


def dummy_frames(device: str) -> Frames:
    return Frames(
        width=2,
        height=2,
        names=["dummy"],
        timestamps=[0.0],
        Ks_cpu=torch.eye(3).reshape(1, 3, 3),
        Ks_gpu=torch.eye(3, device=device).reshape(1, 3, 3),
        X_WCs_cpu=torch.eye(4).reshape(1, 4, 4),
        X_CWs_opencv_gpu=torch.eye(4, device=device).reshape(1, 4, 4),
        colors_gpu=torch.zeros((1, 2, 2, 3), device=device),
        device=device,
    )


def main() -> None:
    args = parse_args()
    wp.config.kernel_cache_dir = str(args.warp_cache_dir)
    wp.init()
    if args.device == "auto":
        args.device = "cuda" if wp.is_cuda_available() else "cpu"
    sim = make_simulator(args.asset, args.device)
    model = sim.gaussian_model
    ids = model.soft_gaussian_ids.long()
    sim.visual_forces.configure_body_participation([], [])
    sim.visual_forces.configure_gaussian_participation(ids)

    settings = VisualForcesSettings(
        iterations=1,
        lr_means=1.0e-4,
        lr_quats=0.0,
        kp=1.0,
        enable_soft_particle_forces=True,
        require_loss_weights_for_soft=True,
        soft_max_gaussian_force=2.0e-5,
        soft_max_particle_force=2.0e-5,
        soft_max_total_force=5.0e-3,
    )

    frames = dummy_frames(args.device)
    missing_weights_rejected = False
    try:
        sim.compute_visual_forces(settings, frames, dt=1.0 / 720.0)
    except ValueError as error:
        missing_weights_rejected = "per-pixel loss weights" in str(error)

    frames.set_loss_weights(torch.ones((1, 2, 2), device=args.device))
    invalid_weight_shape_rejected = False
    try:
        frames.set_loss_weights(torch.ones((1, 3, 2), device=args.device))
    except ValueError:
        invalid_weight_shape_rejected = True

    with torch.no_grad():
        sim.visual_forces.means.copy_(sim.gaussian_state.means)
        sim.visual_forces.quats.copy_(sim.gaussian_state.quats)
        # A huge non-soft target must be ignored by both the gradient mask and
        # sparse soft scatter. A local soft patch receives a bounded target.
        sim.visual_forces.means[0] += torch.tensor(
            [1.0, -2.0, 3.0], device=args.device
        )
        soft_means = sim.gaussian_state.means[ids]
        center = soft_means[:, :2].mean(dim=0)
        radius = torch.linalg.vector_norm(soft_means[:, :2] - center, dim=1)
        selected = radius < 0.008
        sim.visual_forces.means[ids[selected], 2] += 0.002

    sim.state_0.particle_f.zero_()
    result = sim.apply_soft_visual_forces(settings)
    if result is None:
        raise RuntimeError("Enabled soft visual-force wiring returned no result")
    wp.synchronize_device(sim.model.device)
    particle_f = wp.to_torch(sim.state_0.particle_f)
    force_norm = torch.linalg.vector_norm(particle_f, dim=1)
    inverse_mass = wp.to_torch(sim.model.particle_inv_mass)
    allowed_gradient = ~sim.visual_forces._gaussians_not_involved_in_visual_forces

    disabled_settings = VisualForcesSettings(enable_soft_particle_forces=False)
    before_disabled = particle_f.clone()
    disabled_result = sim.apply_soft_visual_forces(disabled_settings)
    wp.synchronize_device(sim.model.device)

    gates = {
        "missing_pixel_weights_rejected": missing_weights_rejected,
        "invalid_weight_shape_rejected": invalid_weight_shape_rejected,
        "only_soft_gaussians_have_pose_gradients": bool(
            not allowed_gradient[0].item()
            and allowed_gradient[ids].all().item()
            and int(allowed_gradient.sum().item()) == int(model.num_soft_gaussians)
        ),
        "soft_target_reaches_particles": bool(force_norm.max().item() > 0.0),
        "fixed_particles_receive_zero": bool(
            force_norm[inverse_mass == 0.0].max().item() == 0.0
        ),
        "total_budget_enforced": bool(force_norm.sum().item() <= 5.0e-3 + 1.0e-8),
        "disabled_switch_writes_nothing": bool(
            disabled_result is None and torch.equal(particle_f, before_disabled)
        ),
        "finite": bool(torch.isfinite(particle_f).all().item()),
    }
    report = {
        "stage": "D_soft_visual_force_wiring",
        "device": str(sim.model.device),
        "counts": {
            "gaussians": int(model.num_gaussians),
            "soft_gaussians": int(model.num_soft_gaussians),
            "selected_local_gaussians": int(selected.sum().item()),
            "gradient_enabled_gaussians": int(allowed_gradient.sum().item()),
        },
        "forces": {
            "max_particle_force_n": float(force_norm.max().item()),
            "particle_force_budget_n": float(force_norm.sum().item()),
            "global_scale": float(result.global_scale.item()),
        },
        "gates": gates,
        "passed": bool(all(gates.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Soft visual-force wiring gate failed")


if __name__ == "__main__":
    main()
