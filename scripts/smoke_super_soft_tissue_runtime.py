#!/usr/bin/env python3
"""Joint runtime smoke for the main Simulator and soft Gaussian skinning."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_gaussians.embodied_simulator.builder import (  # noqa: E402
    EmbodiedGaussiansBuilder,
)
from embodied_gaussians.embodied_simulator.simulator import (  # noqa: E402
    EmbodiedGaussiansSimulator,
)
from embodied_gaussians.physics_simulator.simulator import PhysicsSettings  # noqa: E402
from embodied_gaussians.scene_builders.domain import SoftBody  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run main-runtime soft tissue smoke.")
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
        / "data/super/grasp5_native/soft_tissue_v1/runtime_smoke_report.json",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--sample-interval", type=int, default=50)
    parser.add_argument(
        "--warp-cache-dir", type=Path, default=Path("/tmp/warp-super-runtime-cache")
    )
    return parser.parse_args()


def sample_metrics(
    sim: EmbodiedGaussiansSimulator,
    rest_particles: torch.Tensor,
    tet_indices: torch.Tensor,
    rest_tet_volume: torch.Tensor,
    radii: torch.Tensor,
    support_mask: torch.Tensor,
    rest_gaussian_means: torch.Tensor,
    nearest_gaussian: torch.Tensor,
    step: int,
) -> dict:
    positions = wp.to_torch(sim.state_0.particle_q)
    velocities = wp.to_torch(sim.state_0.particle_qd)
    tet_positions = positions[tet_indices]
    matrices = torch.stack(
        (
            tet_positions[:, 1] - tet_positions[:, 0],
            tet_positions[:, 2] - tet_positions[:, 0],
            tet_positions[:, 3] - tet_positions[:, 0],
        ),
        dim=-1,
    )
    signed_volume = torch.linalg.det(matrices) / 6.0
    displacement = torch.linalg.vector_norm(positions - rest_particles, dim=1)
    speed = torch.linalg.vector_norm(velocities, dim=1)
    penetration = torch.clamp(radii - positions[:, 2], min=0.0)
    anchor_drift = displacement[support_mask]
    gaussian_means = sim.gaussian_state.means
    gaussian_displacement = torch.linalg.vector_norm(
        gaussian_means - rest_gaussian_means, dim=1
    )
    neighbor_jump = torch.abs(
        gaussian_displacement - gaussian_displacement[nearest_gaussian]
    )
    quat_norm_error = torch.abs(
        torch.linalg.vector_norm(sim.gaussian_state.quats, dim=1) - 1.0
    )
    return {
        "step": int(step),
        "all_finite": bool(
            torch.isfinite(positions).all().item()
            and torch.isfinite(velocities).all().item()
            and torch.isfinite(signed_volume).all().item()
            and torch.isfinite(gaussian_means).all().item()
            and torch.isfinite(sim.gaussian_state.quats).all().item()
        ),
        "inverted_tetrahedra": int((signed_volume <= 0.0).sum().item()),
        "total_volume_ratio": float(
            (signed_volume.sum() / rest_tet_volume.sum()).item()
        ),
        "ground_penetration_max_m": float(penetration.max().item()),
        "anchor_drift_max_m": float(anchor_drift.max().item()),
        "particle_displacement_max_m": float(displacement.max().item()),
        "particle_speed_max_m_s": float(speed.max().item()),
        "particle_speed_p99_m_s": float(torch.quantile(speed, 0.99).item()),
        "gaussian_displacement_max_m": float(gaussian_displacement.max().item()),
        "gaussian_neighbor_jump_p95_m": float(
            torch.quantile(neighbor_jump, 0.95).item()
        ),
        "gaussian_quaternion_norm_error_max": float(quat_norm_error.max().item()),
    }


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.sample_interval <= 0:
        raise ValueError("steps and sample-interval must be positive")
    wp.config.kernel_cache_dir = str(args.warp_cache_dir)
    wp.init()
    if args.device == "auto":
        args.device = "cuda" if wp.is_cuda_available() else "cpu"

    soft_body = SoftBody.from_npz(args.asset, name="super_tissue")
    builder = EmbodiedGaussiansBuilder(up_vector=(0.0, 0.0, 1.0))
    builder.particle_max_velocity = 1.0
    builder.add_soft_body(
        soft_body,
        young_modulus_pa=15000.0,
        poisson_ratio=0.45,
        anchor_mode="support_candidate",
        add_gaussians=True,
    )
    builder.set_ground_plane(normal=(0.0, 0.0, 1.0), offset=0.0, mu=0.05)
    sim = EmbodiedGaussiansSimulator(builder, device=args.device)
    settings = PhysicsSettings(
        dt=1.0 / 60.0,
        substeps=12,
        xpbd_iterations=3,
        use_project_material_tetrahedra=True,
        material_iterations=20,
        material_relaxation=0.15,
        particle_velocity_damping_per_second=12.0,
        particle_ground_relaxation=0.9,
        enable_particle_shape_contacts=False,
        enable_particle_particle_contacts=False,
    )

    mesh = soft_body.tetra_mesh
    rest_particles = torch.as_tensor(
        mesh.rest_positions, device=args.device, dtype=torch.float32
    )
    tet_indices = torch.as_tensor(
        mesh.tet_indices, device=args.device, dtype=torch.long
    )
    rest_tet_volume = torch.as_tensor(
        mesh.rest_tet_volume, device=args.device, dtype=torch.float32
    )
    radii = torch.as_tensor(
        mesh.particle_radius, device=args.device, dtype=torch.float32
    )
    support_mask = torch.as_tensor(
        mesh.support_candidate_mask, device=args.device, dtype=torch.bool
    )
    rest_gaussian_means = sim.gaussian_model.means.clone()
    rest_gaussian_scales = sim.gaussian_state.scale_log.clone()
    rest_means_cpu = rest_gaussian_means.detach().cpu().numpy()
    _, nearest = cKDTree(rest_means_cpu).query(rest_means_cpu, k=2)
    nearest_gaussian = torch.as_tensor(
        nearest[:, 1], device=args.device, dtype=torch.long
    )

    # Compile/capture the full main-runtime graph once, then restore the exact
    # rest state so compilation time and warm-up deformation are not measured.
    initial_state = sim.clone_embodied_gaussian_state()
    sim.physics_step(settings)
    sim.update_gaussian_transforms()
    wp.synchronize_device(sim.model.device)
    sim.copy_embodied_gaussian_state(initial_state)
    sim.reset()
    sim.update_gaussian_transforms()
    samples = [
        sample_metrics(
            sim,
            rest_particles,
            tet_indices,
            rest_tet_volume,
            radii,
            support_mask,
            rest_gaussian_means,
            nearest_gaussian,
            0,
        )
    ]
    wp.synchronize_device(sim.model.device)
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        sim.physics_step(settings)
        sim.update_gaussian_transforms()
        if step % args.sample_interval == 0 or step == args.steps:
            samples.append(
                sample_metrics(
                    sim,
                    rest_particles,
                    tet_indices,
                    rest_tet_volume,
                    radii,
                    support_mask,
                    rest_gaussian_means,
                    nearest_gaussian,
                    step,
                )
            )
    wp.synchronize_device(sim.model.device)
    elapsed = time.perf_counter() - start

    max_volume_error = max(abs(s["total_volume_ratio"] - 1.0) for s in samples)
    gates = {
        "finite": all(s["all_finite"] for s in samples),
        "no_inverted_tetrahedra": max(s["inverted_tetrahedra"] for s in samples)
        == 0,
        "volume_error_below_2_percent": max_volume_error < 0.02,
        "ground_penetration_below_0_1_mm": max(
            s["ground_penetration_max_m"] for s in samples
        )
        < 1.0e-4,
        "anchor_drift_below_1_um": max(s["anchor_drift_max_m"] for s in samples)
        < 1.0e-6,
        "gaussian_neighbor_jump_below_0_25_mm": max(
            s["gaussian_neighbor_jump_p95_m"] for s in samples
        )
        < 2.5e-4,
        "gaussian_quaternion_normalized": max(
            s["gaussian_quaternion_norm_error_max"] for s in samples
        )
        < 2.0e-5,
        "gaussian_scale_unchanged": bool(
            torch.equal(sim.gaussian_state.scale_log, rest_gaussian_scales)
        ),
    }
    report = {
        "stage": "C_main_runtime_joint_smoke",
        "device": str(sim.model.device),
        "steps": args.steps,
        "settings": vars(settings),
        "counts": {
            "particles": int(sim.model.particle_count),
            "tetrahedra": int(sim.model.tet_count),
            "gaussians": int(sim.gaussian_model.num_gaussians),
        },
        "performance": {
            "elapsed_seconds": elapsed,
            "milliseconds_per_frame": 1000.0 * elapsed / args.steps,
        },
        "max_total_volume_error": max_volume_error,
        "samples": samples,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Main-runtime soft tissue smoke failed")


if __name__ == "__main__":
    main()
