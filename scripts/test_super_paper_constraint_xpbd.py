#!/usr/bin/env python3
"""Validate supported fixed XPBD profiles on the current tissue.

This test deliberately excludes the residual mapper, visual forces, tool
contact, and online stiffness optimization.  It checks the rebuilt physical
mesh and the three fixed constraint families in isolation, either at the GUI
reset baseline or at the lowest state reachable by online adaptation.
"""

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
sys.path.insert(0, str(REPO_ROOT / "src"))

from embodied_gaussians.embodied_simulator.builder import (  # noqa: E402
    EmbodiedGaussiansBuilder,
)
from embodied_gaussians.physics_simulator.integrator import (  # noqa: E402
    MaterialTetrahedronXPBDProjector,
)
from embodied_gaussians.scene_builders.domain import SoftBody  # noqa: E402


DEFAULT_ASSET = (
    REPO_ROOT
    / "data/super/grasp5_native/tissue_multiview_v1/"
    "paper_pbd_tissue_v15_denser_mild_paper_constraints_centroid_ellipsoids/"
    "tissue_paper_pbd_centroid_gaussians.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile and validate fixed distance/volume/shape-matching XPBD "
            "without online optimization."
        )
    )
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--warp-cache-dir",
        type=Path,
        default=Path("/tmp/warp-super-paper-constraint-cache"),
    )
    parser.add_argument("--rest-substeps", type=int, default=20)
    parser.add_argument("--recovery-substeps", type=int, default=80)
    parser.add_argument("--substep-dt", type=float, default=1.0 / 720.0)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--relaxation", type=float, default=1.0)
    parser.add_argument("--distance-stiffness", type=float, default=0.2)
    parser.add_argument("--volume-stiffness", type=float, default=1.0e10)
    parser.add_argument("--shape-stiffness", type=float, default=0.004)
    parser.add_argument(
        "--expected-profile",
        choices=("current_baseline", "online_floor", "online_ceiling"),
        default="current_baseline",
        help=(
            "Select the exact configuration gate: the GUI reset baseline or "
            "the lowest state reachable by verified online adaptation."
        ),
    )
    parser.add_argument("--velocity-damping-per-second", type=float, default=10.0)
    parser.add_argument("--perturbation-mm", type=float, default=0.75)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def signed_tet_volumes(
    positions: torch.Tensor, tetrahedra: torch.Tensor
) -> torch.Tensor:
    points = positions[tetrahedra]
    matrices = torch.stack(
        (
            points[:, 1] - points[:, 0],
            points[:, 2] - points[:, 0],
            points[:, 3] - points[:, 0],
        ),
        dim=-1,
    )
    return torch.linalg.det(matrices) / 6.0


def mesh_metrics(
    state: wp.sim.State,
    rest_positions: torch.Tensor,
    tetrahedra: torch.Tensor,
    rest_volumes: torch.Tensor,
    edges: torch.Tensor,
    rest_edge_lengths: torch.Tensor,
    fixed_mask: torch.Tensor,
) -> dict[str, float | int | bool]:
    positions = wp.to_torch(state.particle_q).detach()
    velocities = wp.to_torch(state.particle_qd).detach()
    volumes = signed_tet_volumes(positions, tetrahedra)
    edge_lengths = torch.linalg.vector_norm(
        positions[edges[:, 0]] - positions[edges[:, 1]], dim=1
    )
    displacements = torch.linalg.vector_norm(positions - rest_positions, dim=1)
    relative_edge_error = torch.abs(edge_lengths - rest_edge_lengths) / torch.clamp(
        rest_edge_lengths, min=1.0e-12
    )
    relative_volume_error = torch.abs(volumes - rest_volumes) / torch.clamp(
        rest_volumes, min=1.0e-18
    )
    edge_constraint = edge_lengths - rest_edge_lengths
    volume_constraint = volumes - rest_volumes
    return {
        "all_finite": bool(
            torch.isfinite(positions).all().item()
            and torch.isfinite(velocities).all().item()
            and torch.isfinite(volumes).all().item()
        ),
        "inverted_tetrahedra": int((volumes <= 0.0).sum().item()),
        "minimum_volume_ratio": float((volumes / rest_volumes).min().item()),
        "maximum_volume_ratio": float((volumes / rest_volumes).max().item()),
        "mean_relative_volume_error": float(relative_volume_error.mean().item()),
        "maximum_relative_volume_error": float(relative_volume_error.max().item()),
        "mean_relative_edge_error": float(relative_edge_error.mean().item()),
        "maximum_relative_edge_error": float(relative_edge_error.max().item()),
        "rms_edge_constraint_m": float(
            torch.sqrt(torch.mean(edge_constraint * edge_constraint)).item()
        ),
        "rms_volume_constraint_m3": float(
            torch.sqrt(torch.mean(volume_constraint * volume_constraint)).item()
        ),
        "maximum_displacement_m": float(displacements.max().item()),
        "maximum_anchor_drift_m": float(displacements[fixed_mask].max().item()),
        "maximum_speed_m_s": float(
            torch.linalg.vector_norm(velocities, dim=1).max().item()
        ),
    }


def main() -> None:
    args = parse_args()
    if args.rest_substeps < 1 or args.recovery_substeps < 1:
        raise ValueError("substep counts must be positive")
    if args.substep_dt <= 0.0 or args.iterations < 1:
        raise ValueError("substep_dt and iterations must be positive")
    args.warp_cache_dir.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(args.warp_cache_dir)
    wp.init()
    if args.device == "auto":
        args.device = "cuda" if wp.is_cuda_available() else "cpu"

    soft_body = SoftBody.from_npz(args.asset, name="super_paper_tissue_v15")
    mesh = soft_body.tetra_mesh
    builder = EmbodiedGaussiansBuilder(gravity=0.0)
    builder.particle_max_velocity = 1.0
    handle = builder.add_soft_body(
        soft_body,
        young_modulus_pa=15_000.0,
        poisson_ratio=0.45,
        anchor_mode="support_candidate",
        add_gaussians=False,
        add_collision_skin=False,
    )
    model = builder.finalize(device=args.device)
    state_0 = model.state()
    state_1 = model.state()
    projector = MaterialTetrahedronXPBDProjector(
        model,
        iterations=args.iterations,
        relaxation=args.relaxation,
        constraint_model="paper",
        paper_distance_stiffness=args.distance_stiffness,
        paper_volume_stiffness=args.volume_stiffness,
        paper_shape_stiffness=args.shape_stiffness,
    )

    rest_positions = torch.as_tensor(
        mesh.rest_positions, device=args.device, dtype=torch.float32
    )
    tetrahedra = torch.as_tensor(
        mesh.tet_indices, device=args.device, dtype=torch.long
    )
    rest_volumes = torch.as_tensor(
        mesh.rest_tet_volume, device=args.device, dtype=torch.float32
    )
    fixed_mask = torch.as_tensor(
        mesh.support_candidate_mask, device=args.device, dtype=torch.bool
    )
    top_mask = torch.as_tensor(
        mesh.top_node_mask, device=args.device, dtype=torch.bool
    )
    with np.load(args.asset, allow_pickle=False) as loaded:
        asset_edges_np = loaded["pbd_edge_indices"].astype(np.int64)
        asset_rest_edge_lengths_np = loaded["pbd_rest_edge_length"].astype(
            np.float32
        )
        gaussian_count = int(len(loaded["gaussian_rest_means_table"]))
        visual_vertex_count = int(len(loaded["visual_surface_rest_vertices_table"]))
        visual_face_count = int(len(loaded["visual_surface_faces"]))
        collision_face_count = int(len(loaded["collision_skin_faces"]))
    asset_edges = torch.as_tensor(
        asset_edges_np, device=args.device, dtype=torch.long
    )
    asset_rest_edge_lengths = torch.as_tensor(
        asset_rest_edge_lengths_np, device=args.device, dtype=torch.float32
    )

    solver_edges_np = projector.paper_edge_indices.numpy().astype(np.int64)
    solver_lengths_np = projector.paper_edge_rest_lengths.numpy().astype(np.float32)
    asset_order = np.lexsort((asset_edges_np[:, 1], asset_edges_np[:, 0]))
    solver_order = np.lexsort((solver_edges_np[:, 1], solver_edges_np[:, 0]))
    topology_matches = bool(
        np.array_equal(asset_edges_np[asset_order], solver_edges_np[solver_order])
        and np.allclose(
            asset_rest_edge_lengths_np[asset_order],
            solver_lengths_np[solver_order],
            rtol=2.0e-5,
            atol=1.0e-9,
        )
    )

    velocity_damping = math.exp(
        -args.velocity_damping_per_second * args.substep_dt
    )

    def substep() -> None:
        nonlocal state_0, state_1
        state_0.particle_f.zero_()
        projector.simulate_unconstrained_particles(
            model,
            state_0,
            state_1,
            args.substep_dt,
            velocity_damping=velocity_damping,
            solve_ground=False,
            material_min_volume_ratio=1.0e-4,
        )
        state_0, state_1 = state_1, state_0

    for _ in range(args.rest_substeps):
        substep()
    wp.synchronize_device(args.device)
    rest_metrics = mesh_metrics(
        state_0,
        rest_positions,
        tetrahedra,
        rest_volumes,
        asset_edges,
        asset_rest_edge_lengths,
        fixed_mask,
    )

    # Reset exactly, then perturb one unanchored top node near the tissue center.
    for state in (state_0, state_1):
        wp.copy(state.particle_q, model.particle_q)
        state.particle_qd.zero_()
        state.particle_f.zero_()
    center = rest_positions[:, :2].mean(dim=0)
    candidate_mask = top_mask & ~fixed_mask
    candidate_ids = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    distances = torch.linalg.vector_norm(
        rest_positions[candidate_ids, :2] - center[None, :], dim=1
    )
    perturb_particle = int(candidate_ids[torch.argmin(distances)].item())
    perturbation_m = args.perturbation_mm * 1.0e-3
    positions_0 = wp.to_torch(state_0.particle_q)
    positions_1 = wp.to_torch(state_1.particle_q)
    perturbation = torch.tensor(
        [0.45 * perturbation_m, 0.0, perturbation_m],
        device=args.device,
        dtype=torch.float32,
    )
    positions_0[perturb_particle] += perturbation
    positions_1[perturb_particle] += perturbation
    initial_perturbed_metrics = mesh_metrics(
        state_0,
        rest_positions,
        tetrahedra,
        rest_volumes,
        asset_edges,
        asset_rest_edge_lengths,
        fixed_mask,
    )
    for _ in range(args.recovery_substeps):
        substep()
    wp.synchronize_device(args.device)
    recovery_metrics = mesh_metrics(
        state_0,
        rest_positions,
        tetrahedra,
        rest_volumes,
        asset_edges,
        asset_rest_edge_lengths,
        fixed_mask,
    )

    expected_profiles = {
        "current_baseline": (0.20, 0.004),
        "online_floor": (0.10, 0.003),
        "online_ceiling": (2.00, 0.020),
    }
    expected_distance, expected_shape = expected_profiles[
        args.expected_profile
    ]
    gates = {
        "v15_mesh_counts_match": bool(
            model.particle_count == 1465
            and model.tet_count == 5502
            and collision_face_count == 2116
        ),
        "visual_detail_preserved": bool(
            gaussian_count == 26754
            and visual_vertex_count == 13583
            and visual_face_count == 26754
        ),
        "solver_uses_asset_tetrahedral_edges": topology_matches,
        "supported_constraint_profile_selected": bool(
            args.distance_stiffness == expected_distance
            and args.volume_stiffness == 1.0e10
            and args.shape_stiffness == expected_shape
            and args.iterations == 6
            and args.relaxation == 1.0
        ),
        "rest_state_is_finite": bool(rest_metrics["all_finite"]),
        "rest_state_has_no_inversion": rest_metrics["inverted_tetrahedra"] == 0,
        "rest_drift_below_1um": rest_metrics["maximum_displacement_m"] < 1.0e-6,
        "fixed_support_drift_below_0.1um": bool(
            recovery_metrics["maximum_anchor_drift_m"] < 1.0e-7
        ),
        "recovery_is_finite": bool(recovery_metrics["all_finite"]),
        "recovery_has_no_inversion": recovery_metrics["inverted_tetrahedra"] == 0,
        "initial_perturbation_has_no_inversion": bool(
            initial_perturbed_metrics["inverted_tetrahedra"] == 0
        ),
        "recovery_reduces_edge_constraint_energy": bool(
            recovery_metrics["rms_edge_constraint_m"]
            < initial_perturbed_metrics["rms_edge_constraint_m"]
        ),
        "recovery_reduces_volume_constraint_energy": bool(
            recovery_metrics["rms_volume_constraint_m3"]
            < initial_perturbed_metrics["rms_volume_constraint_m3"]
        ),
        "recovery_minimum_volume_ratio_above_0.05": bool(
            recovery_metrics["minimum_volume_ratio"] > 0.05
        ),
    }
    report = {
        "stage": "v15_fixed_paper_constraint_xpbd",
        "asset": str(args.asset.resolve()),
        "device": args.device,
        "scope": {
            "online_stiffness_optimization": False,
            "residual_mapping": False,
            "visual_force": False,
            "tool_contact": False,
            "gravity": False,
        },
        "mesh": {
            "particles": int(model.particle_count),
            "tetrahedra": int(model.tet_count),
            "tetrahedral_edges": int(len(asset_edges_np)),
            "collision_triangles": collision_face_count,
            "visual_vertices": visual_vertex_count,
            "visual_triangles": visual_face_count,
            "ellipsoid_gaussians": gaussian_count,
            "fixed_support_particles": int(fixed_mask.sum().item()),
        },
        "settings": {
            "expected_profile": args.expected_profile,
            "constraint_model": "paper",
            "distance_stiffness": args.distance_stiffness,
            "volume_stiffness": args.volume_stiffness,
            "shape_stiffness": args.shape_stiffness,
            "substep_dt_s": args.substep_dt,
            "iterations": args.iterations,
            "relaxation": args.relaxation,
            "velocity_damping_per_second": args.velocity_damping_per_second,
            "rest_substeps": args.rest_substeps,
            "recovery_substeps": args.recovery_substeps,
            "perturbation_mm": args.perturbation_mm,
            "perturbed_particle": perturb_particle,
        },
        "rest_metrics": rest_metrics,
        "initial_perturbed_metrics": initial_perturbed_metrics,
        "recovery_metrics": recovery_metrics,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }
    output = args.output
    if output is None:
        output = args.asset.parent / "paper_constraint_solver_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
