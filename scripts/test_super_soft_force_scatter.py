#!/usr/bin/env python3
"""Stage-D gates for soft Gaussian displacement-to-particle force scatter."""

from __future__ import annotations

import argparse
import json
import math
import sys
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
from embodied_gaussians.embodied_simulator.simulator import (  # noqa: E402
    EmbodiedGaussiansSimulator,
)
from embodied_gaussians.physics_simulator.simulator import PhysicsSettings  # noqa: E402
from embodied_gaussians.scene_builders.domain import SoftBody  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-D soft force gates.")
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
        / "data/super/grasp5_native/soft_tissue_v1/soft_force_gate_report.json",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--physics-steps", type=int, default=1000)
    parser.add_argument("--sample-interval", type=int, default=50)
    parser.add_argument("--young-modulus-pa", type=float, default=15000.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.45)
    parser.add_argument("--gravity-m-s2", type=float, default=-9.80665)
    parser.add_argument(
        "--material-min-volume-ratio", type=float, default=1.0e-4
    )
    parser.add_argument("--velocity-damping-per-second", type=float, default=12.0)
    parser.add_argument("--local-patch-radius-m", type=float, default=0.008)
    parser.add_argument("--local-target-displacement-m", type=float, default=0.002)
    parser.add_argument("--local-max-gaussian-force-n", type=float, default=2.0e-5)
    parser.add_argument("--local-max-particle-force-n", type=float, default=2.0e-5)
    parser.add_argument(
        "--local-max-particle-acceleration-m-s2", type=float, default=0.1
    )
    parser.add_argument("--local-max-total-force-n", type=float, default=5.0e-3)
    parser.add_argument("--force-spread-layers", type=int, default=0)
    parser.add_argument(
        "--warp-cache-dir", type=Path, default=Path("/tmp/warp-super-d-cache")
    )
    return parser.parse_args()


def build_simulator(
    asset_path: Path,
    device: str,
    young_modulus_pa: float,
    poisson_ratio: float,
    gravity_m_s2: float,
):
    soft_body = SoftBody.from_npz(asset_path, name="super_tissue")
    local_builder = EmbodiedGaussiansBuilder(
        up_vector=(0.0, 0.0, 1.0),
        gravity=gravity_m_s2,
    )
    local_builder.particle_max_velocity = 1.0
    local_builder.add_soft_body(
        soft_body,
        young_modulus_pa=young_modulus_pa,
        poisson_ratio=poisson_ratio,
        anchor_mode="support_candidate",
        add_gaussians=True,
    )

    # A non-soft sentinel Gaussian/particle exercises scene offsets and proves
    # that large PSM/ground-like residuals cannot enter the soft scatter path.
    builder = EmbodiedGaussiansBuilder(
        up_vector=(0.0, 0.0, 1.0),
        gravity=gravity_m_s2,
    )
    builder.particle_max_velocity = 1.0
    builder.add_particle((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), 0.0, radius=0.001)
    builder.gaussian_means.append([1.0, 1.0, 1.0])
    builder.gaussian_quats.append([1.0, 0.0, 0.0, 0.0])
    builder.gaussian_scales.append([0.001, 0.001, 0.001])
    builder.gaussian_opacities.append(0.5)
    builder.gaussian_colors.append([0.5, 0.5, 0.5])
    builder.gaussian_body_ids.append(-1)
    builder.add_builder(local_builder)
    builder.set_ground_plane(normal=(0.0, 0.0, 1.0), offset=0.0, mu=0.05)
    return EmbodiedGaussiansSimulator(builder, device=device), soft_body


def clear_particle_forces(sim: EmbodiedGaussiansSimulator) -> torch.Tensor:
    sim.state_0.particle_f.zero_()
    return wp.to_torch(sim.state_0.particle_f)


def reference_scatter(
    sim: EmbodiedGaussiansSimulator,
    displacements: torch.Tensor,
    kp: float,
) -> torch.Tensor:
    model = sim.gaussian_model
    indices = model.soft_gaussian_particle_indices.long()
    weights = model.soft_gaussian_barycentric_weights
    ids = model.soft_gaussian_ids.long()
    gaussian_forces = model.opacities[ids, None] * kp * displacements[ids]
    expected = torch.zeros_like(sim._soft_particle_forces)
    for corner in range(4):
        particle_ids = indices[:, corner]
        support = sim._soft_particle_weight_sums[particle_ids]
        normalized = torch.where(
            support > 1.0e-8,
            weights[:, corner] / torch.clamp(support, min=1.0e-8),
            torch.zeros_like(support),
        )
        expected.index_add_(0, particle_ids, normalized[:, None] * gaussian_forces)
    inverse_mass = wp.to_torch(sim.model.particle_inv_mass)
    expected[inverse_mass == 0.0] = 0.0
    return expected


def deformation_metrics(
    sim: EmbodiedGaussiansSimulator,
    rest_volume: torch.Tensor,
    tet_indices: torch.Tensor,
    support_mask: torch.Tensor,
    rest_particles: torch.Tensor,
    step: int,
) -> dict:
    positions = wp.to_torch(sim.state_0.particle_q)
    velocities = wp.to_torch(sim.state_0.particle_qd)
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
    displacement = torch.linalg.vector_norm(positions - rest_particles, dim=1)
    return {
        "step": int(step),
        "finite": bool(
            torch.isfinite(positions).all().item()
            and torch.isfinite(velocities).all().item()
            and torch.isfinite(volume).all().item()
            and torch.isfinite(sim.gaussian_state.means).all().item()
        ),
        "inverted_tetrahedra": int((volume <= 0.0).sum().item()),
        "minimum_tetrahedron_volume_ratio": float(
            (volume / rest_volume).min().item()
        ),
        "total_volume_ratio": float((volume.sum() / rest_volume.sum()).item()),
        "anchor_drift_max_m": float(displacement[support_mask].max().item()),
        "particle_speed_max_m_s": float(
            torch.linalg.vector_norm(velocities, dim=1).max().item()
        ),
        "particle_displacement_max_m": float(displacement.max().item()),
    }


def advance_physics(
    sim: EmbodiedGaussiansSimulator,
    settings: PhysicsSettings,
    use_cuda_graph: bool,
) -> None:
    if use_cuda_graph:
        sim.physics_step(settings)
        return
    if sim.material_projector is None:
        raise RuntimeError("CPU fallback requires the material projector")
    substep_dt = settings.dt / settings.substeps
    velocity_damping = math.exp(
        -settings.particle_velocity_damping_per_second * substep_dt
    )
    sim.material_projector.iterations = settings.material_iterations
    sim.material_projector.relaxation = settings.material_relaxation
    sim.material_projector.compliance_scale = settings.material_compliance_scale
    for _ in range(settings.substeps):
        sim.material_projector.simulate_unconstrained_particles(
            sim.model,
            sim.state_0,
            sim.state_1,
            substep_dt,
            velocity_damping=velocity_damping,
            solve_ground=True,
            ground_relaxation=settings.particle_ground_relaxation,
            material_min_volume_ratio=settings.material_min_volume_ratio,
        )
        sim.state_0, sim.state_1 = sim.state_1, sim.state_0
    sim.state_0.clear_forces()


def main() -> None:
    args = parse_args()
    if args.physics_steps < 0 or args.sample_interval <= 0:
        raise ValueError("physics-steps must be non-negative and sample-interval positive")
    wp.config.kernel_cache_dir = str(args.warp_cache_dir)
    wp.init()
    if args.device == "auto":
        args.device = "cuda" if wp.is_cuda_available() else "cpu"
    sim, soft_body = build_simulator(
        args.asset,
        args.device,
        args.young_modulus_pa,
        args.poisson_ratio,
        args.gravity_m_s2,
    )
    model = sim.gaussian_model
    soft_ids = model.soft_gaussian_ids.long()
    handle = sim.builder.soft_body_handles[0]
    inverse_mass = wp.to_torch(sim.model.particle_inv_mass)

    # Gate 1: a huge residual on a non-soft sentinel must produce exactly zero.
    isolation_displacement = torch.zeros_like(model.means)
    isolation_displacement[0] = torch.tensor(
        [1.0, -2.0, 3.0], device=model.device, dtype=torch.float32
    )
    applied = clear_particle_forces(sim)
    isolation_result = sim.scatter_soft_gaussian_forces(
        isolation_displacement, kp=1000.0
    )
    wp.synchronize_device(sim.model.device)
    isolation_max_force = float(
        torch.linalg.vector_norm(applied, dim=1).max().item()
    )

    # Gate 2: compare the Warp atomic scatter against a Torch reference.
    uniform_displacement = torch.zeros_like(model.means)
    direction = torch.tensor(
        [0.0002, -0.0001, 0.0003], device=model.device, dtype=torch.float32
    )
    uniform_displacement[soft_ids] = direction
    applied = clear_particle_forces(sim)
    uniform_result = sim.scatter_soft_gaussian_forces(
        uniform_displacement,
        kp=2.5,
        gaussian_opacities=model.opacities,
    )
    wp.synchronize_device(sim.model.device)
    expected = reference_scatter(sim, uniform_displacement, kp=2.5)
    reference_error = torch.linalg.vector_norm(applied - expected, dim=1)
    fixed_force_max = torch.linalg.vector_norm(applied[inverse_mass == 0.0], dim=1).max()

    # Gate 3: only a central Gaussian patch is loaded; far particles stay zero.
    rest_gaussian_means = model.means[soft_ids]
    xy_center = rest_gaussian_means[:, :2].mean(dim=0)
    gaussian_radius = torch.linalg.vector_norm(
        rest_gaussian_means[:, :2] - xy_center, dim=1
    )
    selected_soft = gaussian_radius < args.local_patch_radius_m
    local_displacement = torch.zeros_like(model.means)
    local_displacement[
        soft_ids[selected_soft], 2
    ] = args.local_target_displacement_m
    applied = clear_particle_forces(sim)
    local_result = sim.scatter_soft_gaussian_forces(
        local_displacement,
        kp=1.0,
        max_gaussian_force=args.local_max_gaussian_force_n,
        max_particle_force=args.local_max_particle_force_n,
        max_total_force=args.local_max_total_force_n,
        max_particle_acceleration=(
            args.local_max_particle_acceleration_m_s2
        ),
    )
    wp.synchronize_device(sim.model.device)
    local_global_scale = float(local_result.global_scale.item())
    rest_particles = wp.to_torch(sim.model.particle_q).clone()
    soft_positions = rest_particles[handle.particle_start : handle.particle_end]
    particle_radius = torch.linalg.vector_norm(
        soft_positions[:, :2] - xy_center, dim=1
    )
    local_applied = applied[handle.particle_start : handle.particle_end]
    local_norm = torch.linalg.vector_norm(local_applied, dim=1)
    near_force_mean = local_norm[particle_radius < 0.012].mean()
    far_force_max = local_norm[particle_radius > 0.025].max()

    # Gate 4: deliberately excessive targets must obey both safety budgets.
    excessive = torch.zeros_like(model.means)
    excessive[soft_ids] = torch.tensor(
        [0.1, 0.1, 0.1], device=model.device, dtype=torch.float32
    )
    applied = clear_particle_forces(sim)
    clamp_result = sim.scatter_soft_gaussian_forces(
        excessive,
        kp=10.0,
        max_gaussian_force=1.0e-3,
        max_particle_force=2.0e-4,
        max_total_force=5.0e-3,
        max_particle_acceleration=0.1,
    )
    wp.synchronize_device(sim.model.device)
    applied_norm = torch.linalg.vector_norm(applied, dim=1)
    applied_acceleration = applied_norm * inverse_mass
    applied_budget = applied_norm.sum()
    clamp_global_scale = float(clamp_result.global_scale.item())
    clamp_force_budget_before = float(
        clamp_result.force_budget_before_total_clamp.item()
    )

    dynamics = {"skipped": args.physics_steps == 0, "samples": []}
    dynamics_passed = True
    if args.physics_steps > 0:
        settings = PhysicsSettings(
            dt=1.0 / 60.0,
            substeps=12,
            xpbd_iterations=3,
            use_project_material_tetrahedra=True,
            material_iterations=20,
            material_relaxation=0.15,
            particle_velocity_damping_per_second=(
                args.velocity_damping_per_second
            ),
            particle_ground_relaxation=0.9,
            material_min_volume_ratio=args.material_min_volume_ratio,
            enable_particle_shape_contacts=False,
            enable_particle_particle_contacts=False,
        )
        initial = sim.clone_embodied_gaussian_state()
        use_cuda_graph = str(sim.model.device).startswith("cuda")
        advance_physics(sim, settings, use_cuda_graph)
        sim.update_gaussian_transforms()
        wp.synchronize_device(sim.model.device)
        sim.copy_embodied_gaussian_state(initial)
        sim.reset()
        sim.update_gaussian_transforms()

        tet_indices = torch.as_tensor(
            np.asarray(sim.builder.tet_indices), device=model.device, dtype=torch.long
        )
        rest_volume = torch.as_tensor(
            soft_body.tetra_mesh.rest_tet_volume,
            device=model.device,
            dtype=torch.float32,
        )
        support_mask = inverse_mass == 0.0
        samples = []
        for step in range(1, args.physics_steps + 1):
            sim.scatter_soft_gaussian_forces(
                local_displacement,
                kp=1.0,
                max_gaussian_force=args.local_max_gaussian_force_n,
                max_particle_force=args.local_max_particle_force_n,
                max_total_force=args.local_max_total_force_n,
                max_particle_acceleration=(
                    args.local_max_particle_acceleration_m_s2
                ),
                spread_layers=args.force_spread_layers,
            )
            advance_physics(sim, settings, use_cuda_graph)
            sim.update_gaussian_transforms()
            if step % args.sample_interval == 0 or step == args.physics_steps:
                samples.append(
                    deformation_metrics(
                        sim,
                        rest_volume,
                        tet_indices,
                        support_mask,
                        rest_particles,
                        step,
                    )
                )
        dynamics = {"skipped": False, "samples": samples}
        dynamics_passed = bool(
            all(sample["finite"] for sample in samples)
            and max(sample["inverted_tetrahedra"] for sample in samples) == 0
            and max(abs(sample["total_volume_ratio"] - 1.0) for sample in samples)
            < 0.02
            and max(sample["anchor_drift_max_m"] for sample in samples) < 1.0e-6
        )

    gates = {
        "non_soft_gaussians_isolated": isolation_max_force == 0.0,
        "warp_matches_reference": bool(reference_error.max().item() < 1.0e-9),
        "fixed_particles_receive_zero": bool(fixed_force_max.item() == 0.0),
        "local_force_is_local": bool(
            near_force_mean.item() > 0.0 and far_force_max.item() < 1.0e-12
        ),
        "per_particle_clamp": bool(applied_norm.max().item() <= 2.0e-4 + 1.0e-9),
        "mass_aware_particle_acceleration_clamp": bool(
            applied_acceleration.max().item() <= 0.1 + 1.0e-5
        ),
        "total_force_budget": bool(applied_budget.item() <= 5.0e-3 + 1.0e-8),
        "finite": bool(
            torch.isfinite(clamp_result.particle_forces).all().item()
        ),
        "bounded_dynamics": dynamics_passed,
    }
    report = {
        "stage": "D_soft_gaussian_force_scatter",
        "qualification": (
            "full_1000_step_gate"
            if args.physics_steps >= 1000
            else "functional_or_short_dynamics_only"
        ),
        "device": str(sim.model.device),
        "counts": {
            "particles": int(sim.model.particle_count),
            "soft_gaussians": int(model.num_soft_gaussians),
            "selected_local_gaussians": int(selected_soft.sum().item()),
        },
        "dynamics_configuration": {
            "young_modulus_pa": args.young_modulus_pa,
            "poisson_ratio": args.poisson_ratio,
            "gravity_m_s2": args.gravity_m_s2,
            "material_min_volume_ratio": (
                args.material_min_volume_ratio
            ),
            "velocity_damping_per_second": (
                args.velocity_damping_per_second
            ),
            "local_patch_radius_m": args.local_patch_radius_m,
            "local_target_displacement_m": (
                args.local_target_displacement_m
            ),
            "local_max_gaussian_force_n": (
                args.local_max_gaussian_force_n
            ),
            "local_max_particle_force_n": (
                args.local_max_particle_force_n
            ),
            "local_max_particle_acceleration_m_s2": (
                args.local_max_particle_acceleration_m_s2
            ),
            "local_max_total_force_n": args.local_max_total_force_n,
            "force_spread_layers": args.force_spread_layers,
        },
        "isolation": {"max_particle_force_n": isolation_max_force},
        "reference": {
            "max_particle_force_error_n": float(reference_error.max().item()),
            "fixed_particle_force_max_n": float(fixed_force_max.item()),
        },
        "locality": {
            "near_force_mean_n": float(near_force_mean.item()),
            "far_force_max_n": float(far_force_max.item()),
            "global_scale": local_global_scale,
        },
        "clamps": {
            "max_applied_particle_force_n": float(applied_norm.max().item()),
            "max_applied_particle_acceleration_m_s2": float(
                applied_acceleration.max().item()
            ),
            "applied_force_budget_n": float(applied_budget.item()),
            "force_budget_before_total_clamp_n": clamp_force_budget_before,
            "global_scale": clamp_global_scale,
        },
        "dynamics": dynamics,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Stage-D soft force gate failed")


if __name__ == "__main__":
    main()
