#!/usr/bin/env python3
"""Small deterministic gate for deformable triangle-skin tool contact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import warp as wp
import warp.sim

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embodied_gaussians.physics_simulator.triangle_skin_contact import (  # noqa: E402
    TriangleSkinContactProjector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test barycentric triangle-skin contact on one tetrahedron."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--warp-cache-dir",
        type=Path,
        default=Path("/tmp/warp-triangle-skin-contact"),
    )
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def oriented_tetra_surface(
    points: np.ndarray, tet: np.ndarray
) -> np.ndarray:
    center = points[tet].mean(axis=0)
    faces = np.asarray(
        [
            [tet[1], tet[2], tet[3]],
            [tet[0], tet[3], tet[2]],
            [tet[0], tet[1], tet[3]],
            [tet[0], tet[2], tet[1]],
        ],
        dtype=np.int32,
    )
    for face in faces:
        point_a, point_b, point_c = points[face]
        normal = np.cross(point_b - point_a, point_c - point_a)
        if np.dot(normal, (point_a + point_b + point_c) / 3.0 - center) < 0.0:
            face[[1, 2]] = face[[2, 1]]
    return faces


def main() -> None:
    args = parse_args()
    args.warp_cache_dir.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(args.warp_cache_dir)
    wp.init()

    contact_tet_points = np.asarray(
        [
            [0.000, 0.000, 0.000],
            [0.020, 0.000, 0.000],
            [0.000, 0.020, 0.000],
            [0.000, 0.000, 0.020],
        ],
        dtype=np.float32,
    )
    unrelated_tet_points = contact_tet_points + np.asarray(
        [0.080, 0.0, 0.0], dtype=np.float32
    )
    tissue_points = np.vstack(
        (contact_tet_points, unrelated_tet_points)
    )
    tissue_tet = np.asarray([0, 1, 2, 3], dtype=np.int32)
    unrelated_tet = np.asarray([4, 5, 6, 7], dtype=np.int32)
    skin_faces = np.vstack(
        (
            oriented_tetra_surface(tissue_points, tissue_tet),
            oriented_tetra_surface(tissue_points, unrelated_tet),
        )
    )

    tool_center = np.asarray([0.006, 0.006, 0.006], dtype=np.float32)
    tool_offsets = np.asarray(
        [
            [-0.00035, -0.00035, -0.00035],
            [+0.00035, -0.00035, -0.00035],
            [-0.00035, +0.00035, -0.00035],
            [-0.00035, -0.00035, +0.00035],
        ],
        dtype=np.float32,
    )
    tool_vertices = tool_center[None, :] + tool_offsets
    tool_faces = oriented_tetra_surface(
        tool_vertices, np.asarray([0, 1, 2, 3], dtype=np.int32)
    )

    builder = warp.sim.ModelBuilder(gravity=0.0)
    for particle_index, point in enumerate(tissue_points):
        # Keep the horizontal base fixed so contact must deform the skin.
        mass = 0.0 if particle_index < 3 else 1.0
        builder.add_particle(point, (0.0, 0.0, 0.0), mass, radius=0.0)
    rest_volume = builder.add_tetrahedron(0, 1, 2, 3, 1.0e3, 1.0e3)
    if rest_volume <= 0.0:
        raise RuntimeError("Synthetic tetrahedron is inverted")
    unrelated_rest_volume = builder.add_tetrahedron(
        4, 5, 6, 7, 1.0e3, 1.0e3
    )
    if unrelated_rest_volume <= 0.0:
        raise RuntimeError("Unrelated synthetic tetrahedron is inverted")
    body = builder.add_body(origin=wp.transform_identity())
    tool_mesh = warp.sim.Mesh(
        vertices=tool_vertices,
        indices=tool_faces.reshape(-1),
    )
    shape = builder.add_shape_mesh(
        body=body,
        mesh=tool_mesh,
        density=0.0,
        has_ground_collision=False,
        has_shape_collision=False,
    )
    model = builder.finalize(args.device)
    model.ground = False
    state = model.state()
    # Put an unrelated surface tet below the safety threshold before contact.
    # It has zero proposed delta and must not globally suppress the valid
    # contact correction on the first component.
    compressed_positions = state.particle_q.numpy()
    compressed_positions[7, 2] = 0.0004
    state.particle_q.assign(compressed_positions)
    projector = TriangleSkinContactProjector(
        model,
        skin_faces,
        [shape],
        sample_spacing_m=0.0002,
        spread_layers=1,
    )

    positions_before = state.particle_q.numpy().copy()
    projector.project(
        model,
        state,
        1.0 / 720.0,
        contact_margin_m=0.0004,
        query_distance_m=0.005,
        ccd_velocity_scale=1.0,
        friction_coefficient=0.35,
        relaxation=0.7,
        maximum_correction_m=0.0004,
        minimum_volume_ratio=0.05,
    )
    first_metrics = projector.metrics()
    positions_after = state.particle_q.numpy().copy()
    displacement = np.linalg.norm(
        positions_after - positions_before, axis=1
    )
    current_matrix = np.column_stack(
        (
            positions_after[1] - positions_after[0],
            positions_after[2] - positions_after[0],
            positions_after[3] - positions_after[0],
        )
    )
    current_volume = np.linalg.det(current_matrix) / 6.0
    volume_ratio = current_volume / rest_volume

    projector.project(
        model,
        state,
        1.0 / 720.0,
        contact_margin_m=0.0004,
        query_distance_m=0.005,
        ccd_velocity_scale=1.0,
        friction_coefficient=0.35,
        relaxation=0.7,
        maximum_correction_m=0.0004,
        minimum_volume_ratio=0.05,
    )
    second_metrics = projector.metrics()

    gates = {
        "contacts_generated": bool(first_metrics["contact_count"] > 0),
        "dynamic_skin_vertex_moved": bool(float(displacement[3]) > 0.0),
        "fixed_vertices_unchanged": bool(
            np.array_equal(positions_before[:3], positions_after[:3])
        ),
        "unrelated_compressed_tet_unchanged": bool(
            np.array_equal(positions_before[4:], positions_after[4:])
        ),
        "unrelated_tet_does_not_freeze_contact": bool(
            first_metrics["global_inversion_safe_scale"] > 0.0
        ),
        "tetrahedron_not_inverted": bool(volume_ratio >= 0.05),
        "finite": bool(np.isfinite(state.particle_q.numpy()).all()),
        "penetration_nonincreasing": bool(
            second_metrics["maximum_penetration_m"]
            <= first_metrics["maximum_penetration_m"] + 1.0e-7
        ),
    }
    report = {
        "device": args.device,
        "sample_count": projector.sample_count,
        "contact_spread_layers": projector.spread_layers,
        "rest_volume_m3": float(rest_volume),
        "unrelated_rest_volume_m3": float(unrelated_rest_volume),
        "volume_ratio_after_first_projection": float(volume_ratio),
        "maximum_particle_displacement_m": float(displacement.max()),
        "first_projection": first_metrics,
        "second_projection": second_metrics,
        "gates": gates,
        "passed": all(gates.values()),
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
