#!/usr/bin/env python3
"""Run the SUPER stage-B 1000-step soft-tissue physics smoke test."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_gaussians.embodied_simulator.builder import (  # noqa: E402
    EmbodiedGaussiansBuilder,
)
from embodied_gaussians.physics_simulator.integrator import (  # noqa: E402
    MaterialTetrahedronXPBDProjector,
)
from embodied_gaussians.scene_builders.domain import SoftBody  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-B SUPER soft physics smoke.")
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
        / "data/super/grasp5_native/soft_tissue_v1/physics_smoke_report.json",
    )
    parser.add_argument(
        "--ground-plane",
        type=Path,
        default=REPO_ROOT
        / "data/super/grasp5_native/bodies_v5_table/ground_plane.json",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--warp-cache-dir", type=Path, default=Path("/tmp/warp-super-b-cache"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--substeps", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--young-modulus-pa", type=float, default=15000.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.45)
    parser.add_argument("--material-relaxation", type=float, default=0.15)
    parser.add_argument("--ground-relaxation", type=float, default=0.9)
    parser.add_argument("--velocity-damping-per-second", type=float, default=12.0)
    parser.add_argument("--sample-interval", type=int, default=50)
    return parser.parse_args()


def current_metrics(
    state: wp.sim.State,
    rest_positions: torch.Tensor,
    tet_indices: torch.Tensor,
    rest_tet_volume: torch.Tensor,
    particle_radius: torch.Tensor,
    support_mask: torch.Tensor,
    ground_normal: torch.Tensor,
    ground_d: float,
    step: int,
) -> dict[str, float | int | bool]:
    positions = wp.to_torch(state.particle_q)
    velocities = wp.to_torch(state.particle_qd)
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
    total_volume_ratio = signed_volume.sum() / rest_tet_volume.sum()
    particle_displacement = torch.linalg.vector_norm(positions - rest_positions, dim=1)
    speed = torch.linalg.vector_norm(velocities, dim=1)
    max_speed_index = int(torch.argmax(speed).item())
    ground_clearance = positions @ ground_normal + ground_d - particle_radius
    penetration = torch.clamp(-ground_clearance, min=0.0)
    anchor_drift = particle_displacement[support_mask]
    return {
        "step": int(step),
        "all_finite": bool(
            torch.isfinite(positions).all().item()
            and torch.isfinite(velocities).all().item()
            and torch.isfinite(signed_volume).all().item()
        ),
        "inverted_tetrahedra": int((signed_volume <= 0.0).sum().item()),
        "total_volume_ratio": float(total_volume_ratio.item()),
        "tet_volume_ratio_min": float(
            (signed_volume / rest_tet_volume).min().item()
        ),
        "tet_volume_ratio_max": float(
            (signed_volume / rest_tet_volume).max().item()
        ),
        "max_speed_m_s": float(speed.max().item()),
        "speed_p95_m_s": float(torch.quantile(speed, 0.95).item()),
        "speed_p99_m_s": float(torch.quantile(speed, 0.99).item()),
        "mean_speed_m_s": float(speed.mean().item()),
        "max_speed_particle_z_m": float(positions[max_speed_index, 2].item()),
        "max_speed_particle_ground_clearance_m": float(
            ground_clearance[max_speed_index].item()
        ),
        "max_displacement_m": float(particle_displacement.max().item()),
        "mean_displacement_m": float(particle_displacement.mean().item()),
        "ground_penetration_max_m": float(penetration.max().item()),
        "anchor_drift_max_m": float(anchor_drift.max().item()),
    }


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.substeps <= 0 or args.iterations <= 0:
        raise ValueError("steps, substeps and iterations must be positive")
    if args.substeps % 2 != 0:
        raise ValueError("substeps must be even so a captured frame returns to state_0")

    wp.config.kernel_cache_dir = str(args.warp_cache_dir)
    wp.init()
    if args.device == "auto":
        args.device = "cuda" if wp.is_cuda_available() else "cpu"

    soft_body = SoftBody.from_npz(args.asset, name="super_tissue")
    ground_plane = np.asarray(
        json.loads(args.ground_plane.read_text(encoding="utf-8"))["plane"],
        dtype=np.float64,
    )
    ground_plane /= np.linalg.norm(ground_plane[:3])
    if ground_plane[2] < 0.0:
        ground_plane = -ground_plane
    builder = EmbodiedGaussiansBuilder(up_vector=ground_plane[:3])
    builder.particle_max_velocity = 1.0
    handle = builder.add_soft_body(
        soft_body,
        young_modulus_pa=args.young_modulus_pa,
        poisson_ratio=args.poisson_ratio,
        anchor_mode="support_candidate",
        add_gaussians=False,
    )
    builder.set_ground_plane(
        normal=ground_plane[:3],
        offset=-float(ground_plane[3]),
        mu=0.05,
    )
    model = builder.finalize(device=args.device)
    state_0 = model.state()
    state_1 = model.state()
    projector = MaterialTetrahedronXPBDProjector(
        model,
        iterations=args.iterations,
        relaxation=args.material_relaxation,
    )

    asset = soft_body.tetra_mesh
    rest_positions = torch.as_tensor(
        asset.rest_positions, device=args.device, dtype=torch.float32
    )
    tet_indices = torch.as_tensor(
        asset.tet_indices, device=args.device, dtype=torch.long
    )
    rest_tet_volume = torch.as_tensor(
        asset.rest_tet_volume, device=args.device, dtype=torch.float32
    )
    particle_radius = torch.as_tensor(
        asset.particle_radius, device=args.device, dtype=torch.float32
    )
    support_mask = torch.as_tensor(
        asset.support_candidate_mask, device=args.device, dtype=torch.bool
    )
    ground_normal = torch.as_tensor(
        ground_plane[:3], device=args.device, dtype=torch.float32
    )

    substep_dt = args.dt / args.substeps
    velocity_damping = math.exp(-args.velocity_damping_per_second * substep_dt)

    def run_frame() -> None:
        nonlocal state_0, state_1
        for _ in range(args.substeps):
            projector.simulate_unconstrained_particles(
                model,
                state_0,
                state_1,
                substep_dt,
                velocity_damping=velocity_damping,
                solve_ground=True,
                ground_relaxation=args.ground_relaxation,
            )
            state_0, state_1 = state_1, state_0

    samples = [
        current_metrics(
            state_0,
            rest_positions,
            tet_indices,
            rest_tet_volume,
            particle_radius,
            support_mask,
            ground_normal,
            float(ground_plane[3]),
            0,
        )
    ]
    use_cuda_graph = args.device.startswith("cuda")
    graph = None
    if use_cuda_graph:
        with wp.ScopedCapture() as capture:
            run_frame()
        graph = capture.graph
        # Restore the exact rest state after capture executed one frame.
        wp.copy(state_0.particle_q, model.particle_q)
        state_0.particle_qd.zero_()
        state_0.particle_f.zero_()
        wp.copy(state_1.particle_q, model.particle_q)
        state_1.particle_qd.zero_()
        state_1.particle_f.zero_()

    wp.synchronize_device(args.device)
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        if graph is None:
            run_frame()
        else:
            wp.capture_launch(graph)
        if step % args.sample_interval == 0 or step == args.steps:
            wp.synchronize_device(args.device)
            samples.append(
                current_metrics(
                    state_0,
                    rest_positions,
                    tet_indices,
                    rest_tet_volume,
                    particle_radius,
                    support_mask,
                    ground_normal,
                    float(ground_plane[3]),
                    step,
                )
            )
    wp.synchronize_device(args.device)
    elapsed = time.perf_counter() - start

    finite = all(bool(sample["all_finite"]) for sample in samples)
    max_inverted = max(int(sample["inverted_tetrahedra"]) for sample in samples)
    max_volume_error = max(
        abs(float(sample["total_volume_ratio"]) - 1.0) for sample in samples
    )
    max_penetration = max(
        float(sample["ground_penetration_max_m"]) for sample in samples
    )
    max_anchor_drift = max(float(sample["anchor_drift_max_m"]) for sample in samples)
    max_sampled_speed = max(float(sample["max_speed_m_s"]) for sample in samples)
    stability_gates = {
        "all_samples_finite": finite,
        "no_inverted_tetrahedra": max_inverted == 0,
        "total_volume_error_below_2pct": max_volume_error < 0.02,
        "ground_penetration_below_0.1mm": max_penetration < 1.0e-4,
        "fixed_support_drift_below_1um": max_anchor_drift < 1.0e-6,
        "max_sampled_speed_below_0.12m_s": max_sampled_speed < 0.12,
        "final_max_speed_below_0.1m_s": float(samples[-1]["max_speed_m_s"]) < 0.1,
        "final_p95_speed_below_0.05m_s": float(samples[-1]["speed_p95_m_s"]) < 0.05,
        "final_mean_speed_below_0.01m_s": float(samples[-1]["mean_speed_m_s"]) < 0.01,
    }
    milliseconds_per_frame = elapsed * 1000.0 / args.steps
    performance_gates = {
        "a800_pure_physics_below_20ms_per_frame": (
            milliseconds_per_frame <= 20.0
            if args.device.startswith("cuda")
            else True
        )
    }
    gates = {**stability_gates, **performance_gates}
    device = wp.get_device(args.device)
    report = {
        "stage": "B_soft_particles_tets_ground_smoke",
        "device": args.device,
        "runtime_demo_switched": False,
        "asset": str(args.asset.resolve()),
        "soft_body_handle": {
            "name": handle.name,
            "particle_range": [handle.particle_start, handle.particle_end],
            "tet_range": [handle.tet_start, handle.tet_end],
        },
        "settings": {
            "steps": args.steps,
            "dt_s": args.dt,
            "substeps": args.substeps,
            "substep_dt_s": substep_dt,
            "iterations": args.iterations,
            "young_modulus_pa": args.young_modulus_pa,
            "poisson_ratio": args.poisson_ratio,
            "material_relaxation": args.material_relaxation,
            "ground_relaxation": args.ground_relaxation,
            "velocity_damping_per_second": args.velocity_damping_per_second,
            "ground_plane": ground_plane.tolist(),
            "ground_plane_path": str(args.ground_plane.resolve()),
            "anchors": "support_candidate_mask",
            "psm_collision": False,
            "visual_force": False,
            "rendering": False,
        },
        "performance": {
            "elapsed_s": elapsed,
            "milliseconds_per_frame": milliseconds_per_frame,
            "cuda_graph": use_cuda_graph,
            "warp_mempool_high_bytes": (
                int(wp.get_mempool_used_mem_high(device))
                if use_cuda_graph
                else None
            ),
            "device_total_memory_bytes": (
                int(device.total_memory) if use_cuda_graph else None
            ),
            "device_free_memory_after_bytes": (
                int(device.free_memory) if use_cuda_graph else None
            ),
        },
        "summary": {
            "max_inverted_tetrahedra": max_inverted,
            "max_total_volume_error": max_volume_error,
            "max_ground_penetration_m": max_penetration,
            "max_anchor_drift_m": max_anchor_drift,
            "max_sampled_speed_m_s": max_sampled_speed,
        },
        "samples": samples,
        "stability_passed": bool(all(stability_gates.values())),
        "performance_passed": bool(all(performance_gates.values())),
        "gates": gates,
        "passed": bool(all(gates.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
