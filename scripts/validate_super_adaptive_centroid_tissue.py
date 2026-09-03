#!/usr/bin/env python3
"""Independently validate adaptive tetrahedra and centroid ellipsoid skinning."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from build_super_adaptive_tissue import (
    MULTIVIEW_ROOT,
    connected_component_count,
    oriented_boundary_faces,
    signed_tet_volumes,
    tet_condition_numbers,
    tet_edges,
)


DEFAULT_DIR = (
    MULTIVIEW_ROOT / "paper_pbd_tissue_v11_uniform_surface_centroid_ellipsoids"
)
DEFAULT_ASSET = DEFAULT_DIR / "tissue_paper_pbd_centroid_gaussians.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_DIR / "metadata.json")
    parser.add_argument(
        "--physics-metadata",
        type=Path,
        default=(
            MULTIVIEW_ROOT
            / "paper_pbd_tissue_v11_uniform_surface_physics/metadata.json"
        ),
    )
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_DIR / "asset_validation_report.json"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sorted_rows(values: np.ndarray) -> np.ndarray:
    values = np.sort(np.asarray(values), axis=1)
    return values[np.lexsort(values.T[::-1])]


def edge_incidence(faces: np.ndarray) -> Counter[tuple[int, int]]:
    return Counter(
        tuple(sorted(edge))
        for a, b, c in faces.tolist()
        for edge in ((a, b), (b, c), (c, a))
    )


def visual_face_components(faces: np.ndarray) -> int:
    parent = np.arange(len(faces), dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    owners: dict[tuple[int, int], int] = {}
    for face_id, (a, b, c) in enumerate(faces.tolist()):
        for edge in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(edge))
            if key in owners:
                union(face_id, owners[key])
            else:
                owners[key] = face_id
    return len({find(face_id) for face_id in range(len(faces))})


def main() -> None:
    args = parse_args()
    args.asset = args.asset.resolve()
    args.metadata = args.metadata.resolve()
    args.physics_metadata = args.physics_metadata.resolve()
    args.report = args.report.resolve()
    if args.report.exists() and not args.overwrite:
        raise FileExistsError(f"Report exists; pass --overwrite: {args.report}")
    visual_metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    physics_metadata = json.loads(args.physics_metadata.read_text(encoding="utf-8"))
    with np.load(args.asset, allow_pickle=False) as loaded:
        asset = {name: loaded[name].copy() for name in loaded.files}

    required = {
        "rest_positions_table",
        "tet_indices",
        "collision_skin_faces",
        "collision_skin_face_markers",
        "particle_mass",
        "rest_tet_volume",
        "particle_lateral_density_zone",
        "pbd_edge_indices",
        "pbd_rest_edge_length",
        "pbd_shape_cluster_indices",
        "gaussian_rest_means_table",
        "gaussian_rest_quats_table_wxyz",
        "gaussian_scales",
        "gaussian_binding_mode",
        "gaussian_visual_face_ids",
        "visual_surface_rest_vertices_table",
        "visual_surface_faces",
        "visual_surface_face_density_zone",
        "visual_surface_vertex_source_class",
        "visual_vertex_particle_indices",
        "visual_vertex_barycentric_weights",
        "visual_vertex_rest_offset_table",
        "gaussian_first_pair_observed",
        "gaussian_later_only_observed",
    }
    missing = sorted(required - set(asset))
    if missing:
        raise RuntimeError(f"Asset is missing arrays: {missing}")

    positions = asset["rest_positions_table"].astype(np.float64)
    tets = asset["tet_indices"].astype(np.int32)
    faces = asset["collision_skin_faces"].astype(np.int32)
    volumes = signed_tet_volumes(positions, tets)
    stored_volumes = asset["rest_tet_volume"].astype(np.float64)
    conditions = tet_condition_numbers(positions, tets)
    derived_faces = oriented_boundary_faces(tets)
    derived_edges = tet_edges(tets)
    stored_edges = asset["pbd_edge_indices"].astype(np.int32)
    stored_edge_lengths = asset["pbd_rest_edge_length"].astype(np.float64)
    recomputed_edge_lengths = np.linalg.norm(
        positions[stored_edges[:, 1]] - positions[stored_edges[:, 0]], axis=1
    )
    skin_incidence = edge_incidence(faces)

    density = float(physics_metadata["parameters"]["material_density_kg_m3"])
    integrated_mass = np.zeros(len(positions), dtype=np.float64)
    for corner in range(4):
        np.add.at(integrated_mass, tets[:, corner], density * volumes / 4.0)
    stored_mass = asset["particle_mass"].astype(np.float64)

    visual_vertices = asset["visual_surface_rest_vertices_table"].astype(np.float64)
    visual_faces = asset["visual_surface_faces"].astype(np.int32)
    face_ids = asset["gaussian_visual_face_ids"].astype(np.int32)
    means = asset["gaussian_rest_means_table"].astype(np.float64)
    expected_means = visual_vertices[visual_faces[face_ids]].mean(axis=1)
    centroid_error = np.linalg.norm(means - expected_means, axis=1)
    vertex_support = asset["visual_vertex_particle_indices"].astype(np.int32)
    vertex_weights = asset["visual_vertex_barycentric_weights"].astype(np.float64)
    vertex_offsets = asset["visual_vertex_rest_offset_table"].astype(np.float64)
    reconstructed_vertices = (
        np.einsum("vi,vij->vj", vertex_weights, positions[vertex_support])
        + vertex_offsets
    )
    vertex_reconstruction_error = np.linalg.norm(
        reconstructed_vertices - visual_vertices, axis=1
    )
    visual_incidence = edge_incidence(visual_faces)
    visual_boundary_degree: Counter[int] = Counter()
    for edge, count in visual_incidence.items():
        if count == 1:
            visual_boundary_degree.update(edge)

    scales = asset["gaussian_scales"].astype(np.float64)
    anisotropy = scales.max(axis=1) / scales.min(axis=1)
    qnorm = np.linalg.norm(
        asset["gaussian_rest_quats_table_wxyz"].astype(np.float64), axis=1
    )
    zones = asset["particle_lateral_density_zone"].astype(np.uint8)
    node_spacing = asset["particle_target_spacing"].astype(np.float64)
    zone_spacing_medians = [float(np.median(node_spacing[zones == zone])) for zone in range(3)]

    # A topology-preserving affine pull is a direct numerical demonstration
    # that fixed connectivity does not mean fixed edge lengths or fixed shape.
    stretched_positions = positions.copy()
    center_x = float(asset["grasp_roi_center_xy_table"][0])
    stretched_positions[:, 0] = center_x + 1.05 * (positions[:, 0] - center_x)
    stretched_volumes = signed_tet_volumes(stretched_positions, tets)
    stretched_edge_lengths = np.linalg.norm(
        stretched_positions[stored_edges[:, 1]]
        - stretched_positions[stored_edges[:, 0]],
        axis=1,
    )
    edge_stretch_ratio = stretched_edge_lengths / recomputed_edge_lengths

    gates = {
        "upstream_physics_and_visual_reports_passed": bool(
            physics_metadata["passed"] and visual_metadata["passed"]
        ),
        "single_connected_tetrahedral_body": connected_component_count(
            len(positions), tets
        ) == 1,
        "positive_tetrahedron_rest_volumes": bool(np.all(volumes > 0.0)),
        "stored_tetrahedron_volumes_match": bool(
            np.allclose(volumes, stored_volumes, rtol=2.0e-5, atol=1.0e-14)
        ),
        "minimum_tet_volume_gate_reproduced": bool(volumes.min() * 1.0e9 >= 1.0e-4),
        "maximum_tet_condition_gate_reproduced": bool(conditions.max() <= 250.0),
        "collision_skin_is_exact_tet_boundary": bool(
            np.array_equal(sorted_rows(faces), sorted_rows(derived_faces))
        ),
        "collision_skin_is_closed_two_manifold": bool(
            skin_incidence and set(skin_incidence.values()) == {2}
        ),
        "stored_pbd_edges_are_exact_tet_edges": bool(
            np.array_equal(sorted_rows(stored_edges), sorted_rows(derived_edges))
        ),
        "stored_pbd_edge_lengths_match": bool(
            np.allclose(stored_edge_lengths, recomputed_edge_lengths, rtol=2.0e-5, atol=1.0e-9)
        ),
        "pbd_shape_clusters_are_tetrahedra": bool(
            np.array_equal(asset["pbd_shape_cluster_indices"], tets)
        ),
        "tet_volume_integrated_particle_mass_matches": bool(
            np.allclose(stored_mass, integrated_mass, rtol=2.0e-5, atol=1.0e-10)
        ),
        "adaptive_spacing_increases_center_to_outer": bool(
            zone_spacing_medians[0] < zone_spacing_medians[1] < zone_spacing_medians[2]
        ),
        "visual_surface_is_single_edge_connected_component": visual_face_components(visual_faces) == 1,
        "visual_surface_is_manifold_with_loop_boundaries": bool(
            max(visual_incidence.values()) <= 2
            and visual_boundary_degree
            and set(visual_boundary_degree.values()) == {2}
        ),
        "one_gaussian_per_visual_triangle": bool(
            len(means) == len(visual_faces)
            and np.array_equal(face_ids, np.arange(len(visual_faces)))
        ),
        "gaussian_centers_are_float32_exact_visual_centroids": bool(
            centroid_error.max() <= 1.0e-8
        ),
        "visual_vertices_reconstruct_from_physical_surface": bool(
            vertex_reconstruction_error.max() <= 2.0e-8
            and np.allclose(vertex_weights.sum(axis=1), 1.0, atol=2.0e-6)
        ),
        "binding_mode_is_visual_surface_face_centroid": str(
            asset["gaussian_binding_mode"].item()
        ) == "visual_surface_face_centroid",
        "ellipsoids_and_unit_quaternions_retained": bool(
            np.percentile(anisotropy, 95) > 1.5
            and np.allclose(qnorm, 1.0, atol=2.0e-5)
        ),
        "left_right_and_later_frame_visual_evidence_retained": bool(
            np.any(asset["visual_surface_vertex_source_class"] == 1)
            and np.any(asset["visual_surface_vertex_source_class"] == 2)
            and np.any(asset["gaussian_later_only_observed"])
        ),
        "fixed_topology_supports_continuous_stretch_without_inversion": bool(
            np.all(stretched_volumes > 0.0)
            and edge_stretch_ratio.max() > 1.04
            and np.array_equal(tets, asset["tet_indices"])
        ),
    }
    counts = {
        "particles": int(len(positions)),
        "tetrahedra": int(len(tets)),
        "collision_triangles": int(len(faces)),
        "pbd_edges": int(len(stored_edges)),
        "visual_vertices": int(len(visual_vertices)),
        "visual_triangles": int(len(visual_faces)),
        "gaussians": int(len(means)),
        "physical_lateral_zone_particles": [
            int(np.count_nonzero(zones == zone)) for zone in range(3)
        ],
        "visual_zone_triangles": [
            int(np.count_nonzero(asset["visual_surface_face_density_zone"] == zone))
            for zone in range(3)
        ],
    }
    report = {
        "stage": "independent_adaptive_tet_and_centroid_ellipsoid_validation",
        "asset": str(args.asset),
        "counts": counts,
        "statistics": {
            "tet_volume_mm3_min": float(volumes.min() * 1.0e9),
            "tet_condition_max": float(conditions.max()),
            "mass_g": float(stored_mass.sum() * 1000.0),
            "zone_target_spacing_mm_medians": [value * 1000.0 for value in zone_spacing_medians],
            "visual_centroid_error_nm_max": float(centroid_error.max() * 1.0e9),
            "visual_vertex_reconstruction_error_nm_max": float(
                vertex_reconstruction_error.max() * 1.0e9
            ),
            "gaussian_anisotropy_p50_p95_max": np.percentile(
                anisotropy, [50, 95, 100]
            ).tolist(),
            "affine_pull_edge_stretch_ratio_min_p50_max": np.percentile(
                edge_stretch_ratio, [0, 50, 100]
            ).tolist(),
            "affine_pull_tet_volume_ratio_min_max": [
                float(np.min(stretched_volumes / volumes)),
                float(np.max(stretched_volumes / volumes)),
            ],
        },
        "gates": gates,
        "passed": bool(all(gates.values())),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        failed = [name for name, passed in gates.items() if not passed]
        raise SystemExit(f"Independent validation failed: {failed}")


if __name__ == "__main__":
    main()
