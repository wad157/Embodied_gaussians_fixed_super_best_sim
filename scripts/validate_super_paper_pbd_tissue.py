#!/usr/bin/env python3
"""Independently validate the offline paper-scale tissue asset."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from build_super_adaptive_tissue import (
    BOTTOM_MARKER,
    MULTIVIEW_ROOT,
    SIDE_MARKER,
    TOP_MARKER,
    connected_component_count,
    oriented_boundary_faces,
    signed_tet_volumes,
    tet_edges,
)


DEFAULT_DIR = (
    MULTIVIEW_ROOT / "paper_pbd_tissue_v5_gaustar_face_union"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset",
        type=Path,
        default=DEFAULT_DIR / "tissue_paper_pbd_dense_multiview.npz",
    )
    parser.add_argument(
        "--metadata", type=Path, default=DEFAULT_DIR / "metadata.json"
    )
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_DIR / "asset_validation_report.json"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def same_unoriented_faces(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape:
        return False
    left_sorted = np.sort(left, axis=1)
    right_sorted = np.sort(right, axis=1)
    order_left = np.lexsort(left_sorted.T[::-1])
    order_right = np.lexsort(right_sorted.T[::-1])
    return bool(np.array_equal(left_sorted[order_left], right_sorted[order_right]))


def skin_edge_histogram(faces: np.ndarray) -> dict[int, int]:
    incidence = Counter(
        tuple(sorted(edge))
        for a, b, c in faces.tolist()
        for edge in ((a, b), (b, c), (c, a))
    )
    return {
        int(key): int(value)
        for key, value in sorted(Counter(incidence.values()).items())
    }


def main() -> None:
    args = parse_args()
    if args.report.exists() and not args.overwrite:
        raise FileExistsError(f"Report exists; pass --overwrite: {args.report}")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    with np.load(args.asset) as loaded:
        asset = {key: loaded[key].copy() for key in loaded.files}

    required = {
        "rest_positions_table",
        "tet_indices",
        "surface_faces",
        "surface_face_markers",
        "collision_skin_faces",
        "collision_skin_face_markers",
        "particle_mass",
        "particle_radius",
        "fixed_mask",
        "support_candidate_mask",
        "rest_tet_volume",
        "surface_node_mask",
        "top_node_mask",
        "visible_surface_mask",
        "pbd_edge_indices",
        "pbd_rest_edge_length",
        "pbd_shape_cluster_indices",
        "gaussian_rest_means_table",
        "gaussian_rest_quats_table_wxyz",
        "gaussian_scales",
        "gaussian_opacities",
        "gaussian_colors_rgb",
        "gaussian_tet_ids",
        "gaussian_particle_indices",
        "gaussian_barycentric_weights",
        "gaussian_binding_distance",
        "gaussian_rest_offset_table",
        "gaussian_binding_mode",
        "gaussian_face_ids",
        "gaussian_face_particle_indices",
        "gaussian_face_barycentric_weights",
        "gaussian_face_projection_distance",
        "gaussian_preprojection_surface_distance",
        "gaussian_surface_source_class",
    }
    missing = sorted(required - set(asset))
    if missing:
        raise RuntimeError(f"Asset is missing required arrays: {missing}")

    positions = np.asarray(asset["rest_positions_table"], dtype=np.float64)
    tets = np.asarray(asset["tet_indices"], dtype=np.int32)
    faces = np.asarray(asset["surface_faces"], dtype=np.int32)
    markers = np.asarray(asset["surface_face_markers"], dtype=np.uint8)
    masses = np.asarray(asset["particle_mass"], dtype=np.float64)
    stored_volumes = np.asarray(asset["rest_tet_volume"], dtype=np.float64)
    recomputed_volumes = signed_tet_volumes(positions, tets)
    derived_faces = oriented_boundary_faces(tets)
    derived_edges = tet_edges(tets)
    stored_edges = np.asarray(asset["pbd_edge_indices"], dtype=np.int32)
    stored_lengths = np.asarray(asset["pbd_rest_edge_length"], dtype=np.float64)
    recomputed_lengths = np.linalg.norm(
        positions[stored_edges[:, 0]] - positions[stored_edges[:, 1]], axis=1
    )

    tet_ids = np.asarray(asset["gaussian_tet_ids"], dtype=np.int32)
    particle_indices = np.asarray(asset["gaussian_particle_indices"], dtype=np.int32)
    weights = np.asarray(asset["gaussian_barycentric_weights"], dtype=np.float64)
    offsets = np.asarray(asset["gaussian_rest_offset_table"], dtype=np.float64)
    means = np.asarray(asset["gaussian_rest_means_table"], dtype=np.float64)
    binding_mode = str(np.asarray(asset["gaussian_binding_mode"]).item())
    collision_faces = np.asarray(asset["collision_skin_faces"], dtype=np.int32)
    face_ids = np.asarray(asset["gaussian_face_ids"], dtype=np.int32)
    face_particle_indices = np.asarray(
        asset["gaussian_face_particle_indices"], dtype=np.int32
    )
    face_weights = np.asarray(
        asset["gaussian_face_barycentric_weights"], dtype=np.float64
    )
    source_class = np.asarray(asset["gaussian_surface_source_class"], dtype=np.uint8)
    reconstructed = (
        np.einsum("ni,nij->nj", weights, positions[particle_indices]) + offsets
    )
    reconstruction_error = np.linalg.norm(reconstructed - means, axis=1)
    face_reconstructed = np.einsum(
        "ni,nij->nj", face_weights, positions[face_particle_indices]
    )
    face_reconstruction_error = np.linalg.norm(face_reconstructed - means, axis=1)
    face_vertices_in_tet = np.any(
        face_particle_indices[:, :, None] == particle_indices[:, None, :], axis=2
    )
    tet_corners_on_face = np.any(
        particle_indices[:, :, None] == face_particle_indices[:, None, :], axis=2
    )
    quaternion_norm = np.linalg.norm(
        np.asarray(asset["gaussian_rest_quats_table_wxyz"], dtype=np.float64),
        axis=1,
    )
    edge_histogram = skin_edge_histogram(faces)

    density = float(metadata["parameters"]["material_density_kg_m3"])
    integrated_mass = np.zeros(len(positions), dtype=np.float64)
    for corner in range(4):
        np.add.at(
            integrated_mass,
            tets[:, corner],
            density * recomputed_volumes / 4.0,
        )

    surface_mask = np.asarray(asset["surface_node_mask"], dtype=bool)
    expected_surface_mask = np.zeros(len(positions), dtype=bool)
    expected_surface_mask[np.unique(faces)] = True
    top_mask = np.asarray(asset["top_node_mask"], dtype=bool)
    expected_top_mask = np.zeros(len(positions), dtype=bool)
    expected_top_mask[np.unique(faces[markers == TOP_MARKER])] = True
    visible_mask = np.asarray(asset["visible_surface_mask"], dtype=bool)
    fixed_mask = np.asarray(asset["fixed_mask"], dtype=bool)
    support_mask = np.asarray(asset["support_candidate_mask"], dtype=bool)

    gates = {
        "builder_metadata_passed": bool(metadata.get("passed", False)),
        "positions_are_n_by_3_and_finite": positions.ndim == 2
        and positions.shape[1] == 3
        and bool(np.isfinite(positions).all()),
        "tetrahedron_indices_are_valid": tets.ndim == 2
        and tets.shape[1] == 4
        and int(tets.min()) >= 0
        and int(tets.max()) < len(positions),
        "tetrahedra_have_four_distinct_vertices": bool(
            np.all(np.diff(np.sort(tets, axis=1), axis=1) != 0)
        ),
        "all_recomputed_tet_volumes_positive": bool(
            np.all(recomputed_volumes > 0.0)
        ),
        "stored_tet_volumes_match_geometry": bool(
            np.allclose(stored_volumes, recomputed_volumes, rtol=2.0e-4, atol=1.0e-14)
        ),
        "single_tetrahedral_component": connected_component_count(
            len(positions), tets
        )
        == 1,
        "surface_faces_equal_tet_boundary": same_unoriented_faces(
            faces, derived_faces
        ),
        "triangle_skin_is_closed_manifold": edge_histogram
        == {2: int(3 * len(faces) / 2)},
        "surface_markers_are_complete": markers.shape == (len(faces),)
        and bool(np.all(np.isin(markers, (TOP_MARKER, SIDE_MARKER, BOTTOM_MARKER)))),
        "collision_skin_matches_surface": bool(
            np.array_equal(asset["collision_skin_faces"], faces)
            and np.array_equal(asset["collision_skin_face_markers"], markers)
        ),
        "surface_node_mask_is_exact": bool(
            np.array_equal(surface_mask, expected_surface_mask)
        ),
        "top_node_mask_is_exact": bool(np.array_equal(top_mask, expected_top_mask)),
        "visible_surface_is_subset_of_top": bool(np.all(~visible_mask | top_mask)),
        "fixed_mask_equals_declared_support_mask": bool(
            np.array_equal(fixed_mask, support_mask) and np.any(fixed_mask)
        ),
        "stored_pbd_edges_equal_tet_edges": bool(
            np.array_equal(stored_edges, derived_edges)
        ),
        "pbd_rest_edge_lengths_match_geometry": bool(
            np.allclose(stored_lengths, recomputed_lengths, rtol=2.0e-5, atol=1.0e-9)
        ),
        "pbd_shape_clusters_equal_tetrahedra": bool(
            np.array_equal(asset["pbd_shape_cluster_indices"], tets)
        ),
        "particle_mass_matches_integrated_tet_mass": bool(
            np.allclose(masses, integrated_mass, rtol=2.0e-4, atol=1.0e-10)
        ),
        "particle_collision_radius_is_zero": bool(
            np.all(np.asarray(asset["particle_radius"]) == 0.0)
        ),
        "gaussian_tet_ids_are_valid": int(tet_ids.min()) >= 0
        and int(tet_ids.max()) < len(tets),
        "gaussian_particle_indices_match_tet_ids": bool(
            np.array_equal(particle_indices, tets[tet_ids])
        ),
        "gaussian_weights_are_convex_and_normalized": bool(
            np.all(weights >= -1.0e-7)
            and np.all(weights <= 1.0 + 1.0e-7)
            and np.max(np.abs(weights.sum(axis=1) - 1.0)) <= 2.0e-6
        ),
        "gaussian_binding_mode_is_surface_face_barycentric": (
            binding_mode == "surface_face_barycentric"
        ),
        "gaussian_face_ids_are_valid": face_ids.shape == (len(means),)
        and int(face_ids.min()) >= 0
        and int(face_ids.max()) < len(collision_faces),
        "gaussian_face_particles_match_face_ids": bool(
            face_particle_indices.shape == (len(means), 3)
            and np.array_equal(face_particle_indices, collision_faces[face_ids])
        ),
        "gaussian_face_vertices_belong_to_adjacent_tet": bool(
            np.all(face_vertices_in_tet)
        ),
        "gaussian_face_weights_are_convex_and_normalized": bool(
            face_weights.shape == (len(means), 3)
            and np.all(face_weights >= -1.0e-7)
            and np.all(face_weights <= 1.0 + 1.0e-7)
            and np.max(np.abs(face_weights.sum(axis=1) - 1.0)) <= 2.0e-6
        ),
        "gaussian_nonface_tet_corner_weights_are_zero": bool(
            np.max(np.abs(weights[~tet_corners_on_face])) <= 1.0e-7
            and np.all(np.count_nonzero(weights > 1.0e-8, axis=1) <= 3)
        ),
        "gaussian_face_binding_has_no_rest_offset": bool(
            np.max(np.abs(offsets)) <= 1.0e-12
        ),
        "gaussian_face_pose_reconstructs": bool(
            np.max(face_reconstruction_error) <= 2.0e-8
        ),
        "gaussian_source_classes_are_complete": bool(
            source_class.shape == (len(means),)
            and np.array_equal(np.unique(source_class), np.array([1, 2, 3, 4]))
        ),
        "gaussian_rest_pose_reconstructs": bool(
            np.max(reconstruction_error) <= 2.0e-8
        ),
        "gaussian_quaternions_are_unit_length": bool(
            np.max(np.abs(quaternion_norm - 1.0)) <= 2.0e-5
        ),
        "metadata_counts_match_asset": (
            int(metadata["counts"]["particles"]) == len(positions)
            and int(metadata["counts"]["tetrahedra"]) == len(tets)
            and int(metadata["counts"]["skin_triangles"]) == len(faces)
            and int(metadata["counts"]["gaussians"]) == len(means)
        ),
    }
    report = {
        "asset": str(args.asset.resolve()),
        "metadata": str(args.metadata.resolve()),
        "counts": {
            "particles": int(len(positions)),
            "tetrahedra": int(len(tets)),
            "skin_triangles": int(len(faces)),
            "pbd_edges": int(len(stored_edges)),
            "gaussians": int(len(means)),
        },
        "metrics": {
            "minimum_recomputed_tet_volume_mm3": float(
                recomputed_volumes.min() * 1.0e9
            ),
            "stored_volume_max_abs_error_mm3": float(
                np.max(np.abs(stored_volumes - recomputed_volumes)) * 1.0e9
            ),
            "mass_max_abs_error_g": float(
                np.max(np.abs(masses - integrated_mass)) * 1000.0
            ),
            "gaussian_weight_sum_max_error": float(
                np.max(np.abs(weights.sum(axis=1) - 1.0))
            ),
            "gaussian_rest_reconstruction_max_error_m": float(
                reconstruction_error.max()
            ),
            "gaussian_face_reconstruction_max_error_m": float(
                face_reconstruction_error.max()
            ),
            "gaussian_face_projection_distance_mm_percentiles": [
                float(value * 1000.0)
                for value in np.percentile(
                    np.asarray(asset["gaussian_face_projection_distance"]),
                    [0, 5, 50, 95, 100],
                )
            ],
            "gaussian_surface_source_class_counts": {
                str(int(key)): int(value)
                for key, value in zip(*np.unique(source_class, return_counts=True))
            },
            "skin_edge_incidence_histogram": edge_histogram,
        },
        "gates": gates,
        "passed": bool(all(gates.values())),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        failed = [name for name, passed in gates.items() if not passed]
        raise SystemExit(f"Asset validation failed: {failed}")


if __name__ == "__main__":
    main()
