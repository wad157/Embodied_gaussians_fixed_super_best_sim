#!/usr/bin/env python3
"""Run the SUPER stage-B0 material semantics gate.

The gate first demonstrates that the installed Warp 1.7 XPBD tetrahedron path
is insensitive to the stored Lamé parameters.  It then exercises the
project-local material-aware XPBD projector on a loaded single tetrahedron and
checks that the stage-A tissue asset can be consumed at rest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp
import warp.sim


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_gaussians.physics_simulator.integrator import (  # noqa: E402
    MaterialTetrahedronXPBDProjector,
)


REST_TETRAHEDRON = np.asarray(
    [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SUPER B0 solver gate.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--warp-cache-dir", type=Path, default=Path("/tmp/warp-super-b0-cache")
    )
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
        / "data/super/grasp5_native/soft_tissue_v1/solver_gate_report.json",
    )
    parser.add_argument("--skip-tissue-smoke", action="store_true")
    return parser.parse_args()


def lame_parameters(young_modulus_pa: float, poisson_ratio: float) -> tuple[float, float]:
    if not (-1.0 < poisson_ratio < 0.5):
        raise ValueError("Poisson ratio must be in (-1, 0.5)")
    mu = young_modulus_pa / (2.0 * (1.0 + poisson_ratio))
    lame_lambda = (
        young_modulus_pa
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    return float(mu), float(lame_lambda)


def build_single_tetrahedron(
    device: str, young_modulus_pa: float, poisson_ratio: float
) -> warp.sim.Model:
    builder = warp.sim.ModelBuilder(gravity=0.0)
    rest_volume = abs(
        np.linalg.det(
            np.stack(
                (
                    REST_TETRAHEDRON[1] - REST_TETRAHEDRON[0],
                    REST_TETRAHEDRON[2] - REST_TETRAHEDRON[0],
                    REST_TETRAHEDRON[3] - REST_TETRAHEDRON[0],
                ),
                axis=-1,
            )
        )
        / 6.0
    )
    free_mass = 1000.0 * rest_volume / 4.0
    for index, position in enumerate(REST_TETRAHEDRON):
        builder.add_particle(
            position,
            (0.0, 0.0, 0.0),
            free_mass if index == 3 else 0.0,
            radius=0.0,
        )
    mu, lame_lambda = lame_parameters(young_modulus_pa, poisson_ratio)
    volume = builder.add_tetrahedron(0, 1, 2, 3, mu, lame_lambda, 0.0)
    if volume <= 0.0:
        raise RuntimeError(f"Invalid single-tetrahedron rest volume: {volume}")
    return builder.finalize(device=device)


def set_tip_force(state: warp.sim.State, force_z_n: float) -> None:
    forces = wp.to_torch(state.particle_f)
    forces.zero_()
    forces[3, 2] = force_z_n


def tetrahedron_metrics(state: warp.sim.State) -> dict[str, float | list[float]]:
    positions = wp.to_torch(state.particle_q).detach().cpu().numpy().astype(np.float64)
    rest_matrix = np.stack(
        (
            REST_TETRAHEDRON[1] - REST_TETRAHEDRON[0],
            REST_TETRAHEDRON[2] - REST_TETRAHEDRON[0],
            REST_TETRAHEDRON[3] - REST_TETRAHEDRON[0],
        ),
        axis=-1,
    )
    current_matrix = np.stack(
        (
            positions[1] - positions[0],
            positions[2] - positions[0],
            positions[3] - positions[0],
        ),
        axis=-1,
    )
    deformation = current_matrix @ np.linalg.inv(rest_matrix)
    tip_displacement = positions[3] - REST_TETRAHEDRON[3]
    return {
        "tip_position_m": positions[3].tolist(),
        "tip_displacement_m": tip_displacement.tolist(),
        "tip_displacement_norm_m": float(np.linalg.norm(tip_displacement)),
        "volume_ratio": float(np.linalg.det(deformation)),
        "all_finite": bool(np.isfinite(positions).all()),
    }


def simulate_stock(
    device: str,
    young_modulus_pa: float,
    poisson_ratio: float,
    dt: float = 1.0 / 120.0,
    duration: float = 2.0,
    iterations: int = 10,
    force_z_n: float = -2.0e-4,
) -> dict[str, float | list[float]]:
    model = build_single_tetrahedron(device, young_modulus_pa, poisson_ratio)
    state_0 = model.state()
    state_1 = model.state()
    integrator = warp.sim.XPBDIntegrator(
        iterations=iterations, soft_body_relaxation=0.25
    )
    for _ in range(round(duration / dt)):
        set_tip_force(state_0, force_z_n)
        integrator.simulate(model, state_0, state_1, dt)
        # The gate is quasi-static: remove inertial oscillation after every step.
        wp.to_torch(state_1.particle_qd).zero_()
        state_0, state_1 = state_1, state_0
    wp.synchronize_device(device)
    return tetrahedron_metrics(state_0)


def simulate_material_projector(
    device: str,
    young_modulus_pa: float,
    poisson_ratio: float,
    dt: float = 1.0 / 120.0,
    duration: float = 2.0,
    iterations: int = 10,
    force_z_n: float = -2.0e-4,
    relaxation: float = 1.0,
) -> dict[str, float | list[float]]:
    model = build_single_tetrahedron(device, young_modulus_pa, poisson_ratio)
    state_0 = model.state()
    state_1 = model.state()
    projector = MaterialTetrahedronXPBDProjector(
        model, iterations=iterations, relaxation=relaxation
    )
    for _ in range(round(duration / dt)):
        set_tip_force(state_0, force_z_n)
        projector.simulate_unconstrained_particles(
            model, state_0, state_1, dt, velocity_damping=0.0
        )
        state_0, state_1 = state_1, state_0
    wp.synchronize_device(device)
    return tetrahedron_metrics(state_0)


def relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0e-12)


def build_tissue_model(
    asset: dict[str, np.ndarray], device: str, young_modulus_pa: float, poisson_ratio: float
) -> warp.sim.Model:
    builder = warp.sim.ModelBuilder(gravity=0.0)
    positions = asset["rest_positions_table"]
    masses = asset["particle_mass"].astype(np.float64)
    support = asset["support_candidate_mask"].astype(bool)
    radii = asset["particle_radius"]
    for index, position in enumerate(positions):
        builder.add_particle(
            position,
            (0.0, 0.0, 0.0),
            0.0 if support[index] else float(masses[index]),
            radius=float(radii[index]),
        )
    mu, lame_lambda = lame_parameters(young_modulus_pa, poisson_ratio)
    for tet in asset["tet_indices"]:
        volume = builder.add_tetrahedron(
            int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3]), mu, lame_lambda, 0.0
        )
        if volume <= 0.0:
            raise RuntimeError("Stage-A asset contains an invalid tetrahedron")
    return builder.finalize(device=device)


def tissue_rest_smoke(asset_path: Path, device: str) -> dict[str, float | int | bool]:
    with np.load(asset_path) as loaded:
        asset = {key: loaded[key] for key in loaded.files}
    model = build_tissue_model(asset, device, 15000.0, 0.45)
    state_0 = model.state()
    state_1 = model.state()
    rest = wp.to_torch(state_0.particle_q).detach().clone()
    projector = MaterialTetrahedronXPBDProjector(
        model, iterations=5, relaxation=0.15
    )
    for _ in range(3):
        wp.to_torch(state_0.particle_f).zero_()
        projector.simulate_unconstrained_particles(
            model, state_0, state_1, 1.0 / 120.0, velocity_damping=1.0
        )
        state_0, state_1 = state_1, state_0
    current = wp.to_torch(state_0.particle_q)
    drift = torch.linalg.vector_norm(current - rest, dim=1)
    result = {
        "particles": int(model.particle_count),
        "tetrahedra": int(model.tet_count),
        "all_finite": bool(torch.isfinite(current).all().item()),
        "rest_drift_max_m": float(drift.max().item()),
        "rest_drift_mean_m": float(drift.mean().item()),
    }
    del projector, state_0, state_1, model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    wp.config.kernel_cache_dir = str(args.warp_cache_dir)
    wp.init()
    if args.device == "auto":
        args.device = "cuda" if wp.is_cuda_available() else "cpu"

    stock_low = simulate_stock(args.device, 5000.0, 0.45)
    stock_high = simulate_stock(args.device, 30000.0, 0.45)
    custom_low = simulate_material_projector(args.device, 5000.0, 0.45)
    custom_high = simulate_material_projector(args.device, 30000.0, 0.45)
    custom_nu_low = simulate_material_projector(args.device, 15000.0, 0.35)
    custom_nu_high = simulate_material_projector(args.device, 15000.0, 0.45)
    custom_dt_60 = simulate_material_projector(
        args.device, 15000.0, 0.45, dt=1.0 / 60.0
    )
    custom_dt_120 = simulate_material_projector(
        args.device, 15000.0, 0.45, dt=1.0 / 120.0
    )
    custom_iter_5 = simulate_material_projector(
        args.device, 15000.0, 0.45, iterations=5
    )
    custom_iter_15 = simulate_material_projector(
        args.device, 15000.0, 0.45, iterations=15
    )

    stock_difference = relative_difference(
        float(stock_low["tip_displacement_norm_m"]),
        float(stock_high["tip_displacement_norm_m"]),
    )
    stiffness_ratio = float(custom_high["tip_displacement_norm_m"]) / max(
        float(custom_low["tip_displacement_norm_m"]), 1.0e-12
    )
    dt_difference = relative_difference(
        float(custom_dt_60["tip_displacement_norm_m"]),
        float(custom_dt_120["tip_displacement_norm_m"]),
    )
    iteration_difference = relative_difference(
        float(custom_iter_5["tip_displacement_norm_m"]),
        float(custom_iter_15["tip_displacement_norm_m"]),
    )
    nu_low_volume_error = abs(float(custom_nu_low["volume_ratio"]) - 1.0)
    nu_high_volume_error = abs(float(custom_nu_high["volume_ratio"]) - 1.0)

    tissue = None
    if not args.skip_tissue_smoke:
        tissue = tissue_rest_smoke(args.asset, args.device)

    gates = {
        "stock_warp_material_insensitivity_reproduced": stock_difference < 1.0e-6,
        "custom_stiffness_monotonic": stiffness_ratio < 0.75,
        "custom_poisson_volume_monotonic": nu_high_volume_error < nu_low_volume_error,
        "custom_timestep_relative_difference_below_10pct": dt_difference < 0.10,
        "custom_iteration_relative_difference_below_5pct": iteration_difference < 0.05,
        "stage_a_tissue_rest_finite": bool(
            tissue is None
            or (tissue["all_finite"] and tissue["rest_drift_max_m"] < 1.0e-7)
        ),
    }
    report = {
        "gate_version": 1,
        "device": args.device,
        "formulation": {
            "energy_density": "mu/2*(I1-3-2*log(J)) + lambda/2*log(J)^2",
            "xpbd_constraint": "sqrt(2 * rest_volume * energy_density)",
            "lambda_accumulation": "within each substep",
            "scope": "B0 unconstrained material projection only; contacts are not integrated yet",
        },
        "stock_warp": {
            "E_5kPa_nu_0.45": stock_low,
            "E_30kPa_nu_0.45": stock_high,
            "material_response_relative_difference": stock_difference,
        },
        "project_material_xpbd": {
            "E_5kPa_nu_0.45": custom_low,
            "E_30kPa_nu_0.45": custom_high,
            "high_to_low_E_displacement_ratio": stiffness_ratio,
            "E_15kPa_nu_0.35": custom_nu_low,
            "E_15kPa_nu_0.45": custom_nu_high,
            "nu_0.35_volume_error": nu_low_volume_error,
            "nu_0.45_volume_error": nu_high_volume_error,
            "dt_1_60": custom_dt_60,
            "dt_1_120": custom_dt_120,
            "timestep_response_relative_difference": dt_difference,
            "iterations_5": custom_iter_5,
            "iterations_15": custom_iter_15,
            "iteration_response_relative_difference": iteration_difference,
        },
        "stage_a_tissue_rest_smoke": tissue,
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
