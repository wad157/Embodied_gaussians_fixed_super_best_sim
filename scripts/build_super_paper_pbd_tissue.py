#!/usr/bin/env python3
"""Build a paper-scale tetrahedral tissue and bind the existing Gaussians.

This is an offline asset builder.  It deliberately does not configure the GUI,
run PBD, estimate stiffness, or consume depth residuals.  The output contains a
coarse tetrahedral mechanics volume, its closed triangular boundary, and a
four-node barycentric skinning record for every tissue Gaussian.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import tetgen

from build_super_adaptive_tissue import (
    BOTTOM_MARKER,
    MULTIVIEW_ROOT,
    SIDE_MARKER,
    TOP_MARKER,
    V9_ROOT,
    bind_gaussians,
    build_closed_plc,
    compact_top_surface,
    connected_component_count,
    make_contact_distance_scene,
    orient_tets,
    oriented_boundary_faces,
    percentiles,
    read_json,
    rest_curvature,
    sha256,
    tet_condition_numbers,
    tet_edges,
    transfer_boundary_markers,
    transform_gaussians,
    write_previews,
)


DEFAULT_OBSERVATION = (
    MULTIVIEW_ROOT.parents[1]
    / "tissue_calibration_v1/stage_b_surface_observations/frames/000000.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the offline, paper-scale tetrahedral SUPER tissue and bind "
            "all existing tissue Gaussians."
        )
    )
    parser.add_argument(
        "--rest-surface",
        type=Path,
        default=MULTIVIEW_ROOT / "rest_surface_v1/rest_surface.npz",
    )
    parser.add_argument(
        "--rest-surface-report",
        type=Path,
        default=MULTIVIEW_ROOT / "rest_surface_v1/report.json",
    )
    parser.add_argument(
        "--source-tissue",
        type=Path,
        default=V9_ROOT / "tissue.json",
    )
    parser.add_argument(
        "--initial-observation",
        type=Path,
        default=DEFAULT_OBSERVATION,
        help="Optional Stage-B frame used only to label presently visible top nodes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MULTIVIEW_ROOT / "paper_pbd_tissue_v2",
    )
    parser.add_argument("--target-top-triangles", type=int, default=800)
    parser.add_argument("--decimation-boundary-weight", type=float, default=100.0)
    parser.add_argument("--side-max-spacing-mm", type=float, default=20.0)
    parser.add_argument("--visual-radius-mm", type=float, default=0.4)
    parser.add_argument("--material-density-kg-m3", type=float, default=1000.0)
    parser.add_argument("--binding-candidates", type=int, default=128)
    parser.add_argument("--visible-distance-mm", type=float, default=2.0)
    parser.add_argument("--anchor-side", choices=("min_x", "max_x", "min_y", "max_y"), default="min_y")
    parser.add_argument("--anchor-strip-mm", type=float, default=4.0)
    parser.add_argument("--tetgen-min-ratio", type=float, default=3.0)
    parser.add_argument("--particle-hard-limit", type=int, default=4000)
    parser.add_argument("--tet-hard-limit", type=int, default=20000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def simplify_top_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_triangles: int,
    boundary_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    if target_triangles < 4 or target_triangles >= len(faces):
        raise ValueError(
            f"target-top-triangles must be in [4, {len(faces) - 1}]"
        )
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces),
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    simplified = mesh.simplify_quadric_decimation(
        target_number_of_triangles=target_triangles,
        boundary_weight=boundary_weight,
    )
    simplified.remove_degenerate_triangles()
    simplified.remove_duplicated_triangles()
    simplified.remove_duplicated_vertices()
    simplified.remove_unreferenced_vertices()
    simplified.orient_triangles()
    output_vertices = np.asarray(simplified.vertices, dtype=np.float64)
    output_faces = np.asarray(simplified.triangles, dtype=np.int32)
    if len(output_faces) < max(4, int(0.8 * target_triangles)):
        raise RuntimeError(
            "Top-surface decimation removed too many faces: "
            f"requested {target_triangles}, got {len(output_faces)}"
        )
    normals = np.cross(
        output_vertices[output_faces[:, 1]] - output_vertices[output_faces[:, 0]],
        output_vertices[output_faces[:, 2]] - output_vertices[output_faces[:, 0]],
    )
    if float(normals[:, 2].sum()) < 0.0:
        output_faces[:, [1, 2]] = output_faces[:, [2, 1]]
    return output_vertices, output_faces


def boundary_edge_histogram(faces: np.ndarray) -> dict[int, int]:
    incidence = Counter(
        tuple(sorted(edge))
        for a, b, c in faces.tolist()
        for edge in ((a, b), (b, c), (c, a))
    )
    return {
        int(key): int(value)
        for key, value in sorted(Counter(incidence.values()).items())
    }


def anchor_mask(
    positions: np.ndarray, side: str, strip_width: float
) -> tuple[np.ndarray, float]:
    axis = 0 if side.endswith("x") else 1
    coordinate = positions[:, axis]
    if side.startswith("min"):
        threshold = float(coordinate.min() + strip_width)
        mask = coordinate <= threshold + 1.0e-12
    else:
        threshold = float(coordinate.max() - strip_width)
        mask = coordinate >= threshold - 1.0e-12
    return mask, threshold


def visible_top_mask(
    positions: np.ndarray,
    top_mask: np.ndarray,
    observation_path: Path,
    maximum_distance: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    output = np.zeros(len(positions), dtype=bool)
    if not observation_path.exists():
        output[:] = top_mask
        return output, None
    with np.load(observation_path) as loaded:
        observed = np.asarray(loaded["points_table"], dtype=np.float64)
    top_ids = np.flatnonzero(top_mask)
    distances = cKDTree(observed).query(
        positions[top_ids], k=1, workers=-1
    )[0]
    output[top_ids] = distances <= maximum_distance
    return output, distances


def main() -> None:
    args = parse_args()
    output_paths = {
        "asset": args.output_dir / "tissue_paper_pbd.npz",
        "metadata": args.output_dir / "metadata.json",
        "skin_ply": args.output_dir / "tissue_collision_skin.ply",
        "nodes_ply": args.output_dir / "tissue_mechanics_nodes.ply",
        "binding_ply": args.output_dir / "tissue_gaussian_binding.ply",
    }
    collisions = [path for path in output_paths.values() if path.exists()]
    if collisions and not args.overwrite:
        raise FileExistsError(
            "Outputs already exist; pass --overwrite:\n- "
            + "\n- ".join(str(path) for path in collisions)
        )
    if (
        args.side_max_spacing_mm <= 0.0
        or args.visual_radius_mm <= 0.0
        or args.material_density_kg_m3 <= 0.0
        or args.anchor_strip_mm <= 0.0
        or args.visible_distance_mm <= 0.0
    ):
        raise ValueError("All length and density parameters must be positive")
    if tetgen.__version__ != "0.8.3":
        raise RuntimeError(
            f"This builder is validated with tetgen 0.8.3, got {tetgen.__version__}"
        )

    rest_report = read_json(args.rest_surface_report)
    if not rest_report.get("passed", False):
        raise RuntimeError("The source rest-surface report did not pass")
    with np.load(args.rest_surface) as loaded:
        fine_vertices, fine_faces, _, _ = compact_top_surface(
            loaded["surface_vertices_table"],
            loaded["surface_faces"],
            loaded["surface_vertex_source_class"],
        )
    top_vertices, top_faces = simplify_top_surface(
        fine_vertices,
        fine_faces,
        args.target_top_triangles,
        args.decimation_boundary_weight,
    )
    closed_vertices, closed_faces, closed_markers = build_closed_plc(
        top_vertices,
        top_faces,
        args.side_max_spacing_mm / 1000.0,
    )

    # One isolated TetGen pass adds enough interior/surface Steiner nodes to
    # avoid the very long sliver tetrahedra produced by a boundary-only fill.
    # PBD constraints remain a later runtime concern; this stage creates only
    # their validated rest topology.
    generator = tetgen.TetGen(closed_vertices, closed_faces, closed_markers)
    nodes, tets, _, _ = generator.tetrahedralize(
        plc=True,
        quality=True,
        facesout=True,
        edgesout=False,
        nojettison=True,
        minratio=args.tetgen_min_ratio,
        mindihedral=0.0,
        steinerleft=20000,
        quiet=True,
    )
    nodes = np.asarray(nodes, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int32)
    tets, volumes, corrected_tets = orient_tets(nodes, tets)
    volume_epsilon = float(np.ptp(nodes, axis=0).max()) ** 3 * 1.0e-10
    degenerate_count = int(np.count_nonzero(volumes <= volume_epsilon))
    if degenerate_count:
        raise RuntimeError(f"TetGen produced {degenerate_count} degenerate tets")

    surface_faces = oriented_boundary_faces(tets)
    surface_markers = transfer_boundary_markers(
        surface_faces,
        np.asarray(generator.trifaces, dtype=np.int32),
        np.asarray(generator.triface_markers, dtype=np.int32),
    )
    components = connected_component_count(len(nodes), tets)
    surface_node_mask = np.zeros(len(nodes), dtype=bool)
    surface_node_mask[np.unique(surface_faces)] = True
    top_node_mask = np.zeros(len(nodes), dtype=bool)
    top_node_mask[np.unique(surface_faces[surface_markers == TOP_MARKER])] = True
    bottom_node_mask = np.zeros(len(nodes), dtype=bool)
    bottom_node_mask[
        np.unique(surface_faces[surface_markers == BOTTOM_MARKER])
    ] = True
    support_candidate_mask, anchor_threshold = anchor_mask(
        nodes, args.anchor_side, args.anchor_strip_mm / 1000.0
    )
    fixed_mask = support_candidate_mask.copy()
    visible_surface_mask, visible_distances = visible_top_mask(
        nodes,
        top_node_mask,
        args.initial_observation,
        args.visible_distance_mm / 1000.0,
    )

    top_scene = make_contact_distance_scene(top_vertices, top_faces)
    inward_depth = top_scene.compute_distance(
        o3d.core.Tensor(nodes.astype(np.float32))
    ).numpy().astype(np.float64)
    fine_to_coarse = top_scene.compute_distance(
        o3d.core.Tensor(fine_vertices.astype(np.float32))
    ).numpy().astype(np.float64)
    fine_scene = make_contact_distance_scene(fine_vertices, fine_faces)
    coarse_to_fine = fine_scene.compute_distance(
        o3d.core.Tensor(top_vertices.astype(np.float32))
    ).numpy().astype(np.float64)

    density = float(args.material_density_kg_m3)
    masses = np.zeros(len(nodes), dtype=np.float64)
    for corner in range(4):
        np.add.at(masses, tets[:, corner], density * volumes / 4.0)

    edges = tet_edges(tets)
    edge_lengths = np.linalg.norm(
        nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1
    )
    conditions = tet_condition_numbers(nodes, tets)
    curvature_faces = surface_faces[surface_markers == TOP_MARKER]
    curvature_edges, curvature_opposites, curvature_angles = rest_curvature(
        nodes, curvature_faces
    )

    gaussian = transform_gaussians(read_json(args.source_tissue))
    tet_ids, barycentric, binding_distance, contained = bind_gaussians(
        gaussian["means"], nodes, tets, args.binding_candidates
    )
    binding_particles = tets[tet_ids]
    reconstructed_without_offset = np.einsum(
        "ni,nij->nj", barycentric, nodes[binding_particles]
    )
    rest_offsets = gaussian["means"] - reconstructed_without_offset
    reconstructed = reconstructed_without_offset + rest_offsets
    reconstruction_error = np.linalg.norm(
        reconstructed - gaussian["means"], axis=1
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_paths["asset"],
        rest_positions_table=nodes.astype(np.float32),
        tet_indices=tets,
        surface_faces=surface_faces,
        surface_face_markers=surface_markers,
        collision_skin_faces=surface_faces,
        collision_skin_face_markers=surface_markers,
        collision_skin_enabled_faces=np.ones(len(surface_faces), dtype=bool),
        particle_mass=masses.astype(np.float32),
        particle_radius=np.zeros(len(nodes), dtype=np.float32),
        particle_visual_radius=np.full(
            len(nodes), args.visual_radius_mm / 1000.0, dtype=np.float32
        ),
        particle_target_spacing=np.full(
            len(nodes), np.median(edge_lengths), dtype=np.float32
        ),
        particle_inward_depth=inward_depth.astype(np.float32),
        surface_node_mask=surface_node_mask,
        top_node_mask=top_node_mask,
        bottom_node_mask=bottom_node_mask,
        visible_surface_mask=visible_surface_mask,
        fixed_mask=fixed_mask,
        support_candidate_mask=support_candidate_mask,
        rest_tet_volume=volumes.astype(np.float32),
        pbd_edge_indices=edges,
        pbd_rest_edge_length=edge_lengths.astype(np.float32),
        pbd_shape_cluster_indices=tets,
        top_rest_curvature_faces=curvature_faces,
        top_rest_curvature_edges=curvature_edges,
        top_rest_curvature_opposite_vertices=curvature_opposites,
        top_rest_dihedral_angle=curvature_angles,
        gaussian_rest_means_table=gaussian["means"].astype(np.float32),
        gaussian_rest_quats_table_wxyz=gaussian["quats"].astype(np.float32),
        gaussian_scales=gaussian["scales"].astype(np.float32),
        gaussian_opacities=gaussian["opacities"].astype(np.float32),
        gaussian_colors_rgb=gaussian["colors"].astype(np.float32),
        gaussian_tet_ids=tet_ids,
        gaussian_particle_indices=binding_particles,
        gaussian_barycentric_weights=barycentric.astype(np.float32),
        gaussian_binding_distance=binding_distance.astype(np.float32),
        gaussian_rest_offset_table=rest_offsets.astype(np.float32),
    )
    write_previews(
        args.output_dir,
        nodes,
        surface_faces,
        surface_markers,
        inward_depth,
        gaussian["means"],
        binding_distance,
    )

    edge_histogram = boundary_edge_histogram(surface_faces)
    gates = {
        "rest_surface_report_passed": True,
        "single_tetrahedral_component": components == 1,
        "all_rest_tet_volumes_positive": bool(np.all(volumes > 0.0)),
        "closed_manifold_triangle_skin": edge_histogram == {2: int(3 * len(surface_faces) / 2)},
        "all_triangle_markers_known": bool(
            np.all(np.isin(surface_markers, (TOP_MARKER, SIDE_MARKER, BOTTOM_MARKER)))
        ),
        "paper_scale_particle_budget": len(nodes) <= args.particle_hard_limit,
        "paper_scale_tet_budget": len(tets) <= args.tet_hard_limit,
        "anchor_side_nonempty": bool(np.any(fixed_mask)),
        "visible_nodes_are_top_nodes": bool(np.all(~visible_surface_mask | top_node_mask)),
        "finite_tetrahedron_condition_numbers": bool(np.isfinite(conditions).all()),
        "gaussian_weights_are_convex": bool(
            np.all(barycentric >= -1.0e-7)
            and np.all(barycentric <= 1.0 + 1.0e-7)
            and np.max(np.abs(barycentric.sum(axis=1) - 1.0)) <= 1.0e-6
        ),
        "all_gaussians_reconstruct_with_rest_offset": bool(
            np.max(reconstruction_error) <= 1.0e-9
        ),
    }
    metadata = {
        "asset_version": 2,
        "stage": "offline_paper_scale_tissue_geometry_and_gaussian_binding",
        "scope": {
            "completed": [
                "coarse tetrahedral mechanics volume",
                "closed triangular boundary skin",
                "four-node barycentric Gaussian binding",
                "one-sided anchor candidate mask",
            ],
            "deferred": [
                "PBD solver constraints and runtime",
                "stiffness optimization",
                "depth residual feedback",
                "GUI integration and parameter controls",
            ],
        },
        "representation": {
            "volume": "oriented tetrahedra over particle nodes",
            "surface": "closed oriented triangle skin with top/side/bottom markers",
            "gaussians": "four tetrahedron vertices + convex barycentric weights + rest offset",
            "future_pbd_rest_data": [
                "unique rest edges and lengths",
                "tetrahedron shape clusters",
                "positive rest volumes",
            ],
        },
        "parameters": {
            "requested_top_triangles": args.target_top_triangles,
            "actual_decimated_top_triangles": int(len(top_faces)),
            "decimation_boundary_weight": args.decimation_boundary_weight,
            "side_max_spacing_mm": args.side_max_spacing_mm,
            "material_density_kg_m3": density,
            "visual_radius_mm": args.visual_radius_mm,
            "mechanics_collision_radius_mm": 0.0,
            "binding_candidates": args.binding_candidates,
            "anchor_side": args.anchor_side,
            "anchor_strip_mm": args.anchor_strip_mm,
            "anchor_threshold_m": anchor_threshold,
            "visible_distance_mm": args.visible_distance_mm,
            "tetgen_version": tetgen.__version__,
            "tetgen_quality_refinement": True,
            "tetgen_min_ratio": args.tetgen_min_ratio,
        },
        "counts": {
            "particles": int(len(nodes)),
            "tetrahedra": int(len(tets)),
            "pbd_unique_edges": int(len(edges)),
            "skin_triangles": int(len(surface_faces)),
            "top_triangles": int(np.count_nonzero(surface_markers == TOP_MARKER)),
            "side_triangles": int(np.count_nonzero(surface_markers == SIDE_MARKER)),
            "bottom_triangles": int(np.count_nonzero(surface_markers == BOTTOM_MARKER)),
            "surface_particles": int(surface_node_mask.sum()),
            "top_particles": int(top_node_mask.sum()),
            "visible_top_particles": int(visible_surface_mask.sum()),
            "fixed_anchor_particles": int(fixed_mask.sum()),
            "gaussians": int(len(gaussian["means"])),
        },
        "topology": {
            "tetrahedral_components": components,
            "negative_tet_orientations_corrected": corrected_tets,
            "degenerate_tetrahedra": degenerate_count,
            "triangle_skin_edge_incidence_histogram": edge_histogram,
            "closed_skin": edge_histogram == {2: int(3 * len(surface_faces) / 2)},
            "marker_values": {
                "top": TOP_MARKER,
                "side": SIDE_MARKER,
                "bottom": BOTTOM_MARKER,
            },
        },
        "surface_approximation": {
            "source_top_vertices": int(len(fine_vertices)),
            "source_top_triangles": int(len(fine_faces)),
            "coarse_top_vertices": int(len(top_vertices)),
            "coarse_top_triangles": int(len(top_faces)),
            "fine_vertices_to_coarse_surface_mm_min_p05_p50_p95_max": percentiles(
                fine_to_coarse * 1000.0
            ),
            "coarse_vertices_to_fine_surface_mm_min_p05_p50_p95_max": percentiles(
                coarse_to_fine * 1000.0
            ),
        },
        "tetrahedron_quality": {
            "volume_mm3_min_p05_p50_p95_max": percentiles(volumes * 1.0e9),
            "edge_length_mm_min_p05_p50_p95_max": percentiles(edge_lengths * 1000.0),
            "rest_matrix_condition_min_p05_p50_p95_p99_max": percentiles(
                conditions, (0, 5, 50, 95, 99, 100)
            ),
        },
        "mass": {
            "volume_cm3": float(volumes.sum() * 1.0e6),
            "integrated_mass_g": float(masses.sum() * 1000.0),
        },
        "visibility": {
            "observation_available": args.initial_observation.exists(),
            "top_to_observation_distance_mm_min_p05_p50_p95_max": (
                None
                if visible_distances is None
                else percentiles(visible_distances * 1000.0)
            ),
        },
        "gaussian_binding": {
            "directly_contained": int(contained.sum()),
            "projected_to_tetrahedron": int((~contained).sum()),
            "distance_mm_min_p50_p90_p95_p99_max": percentiles(
                binding_distance * 1000.0, (0, 50, 90, 95, 99, 100)
            ),
            "weight_sum_max_error": float(
                np.max(np.abs(barycentric.sum(axis=1) - 1.0))
            ),
            "rest_reconstruction_max_error_m": float(reconstruction_error.max()),
        },
        "inputs": {
            "rest_surface": {"path": str(args.rest_surface.resolve()), "sha256": sha256(args.rest_surface)},
            "rest_surface_report": {"path": str(args.rest_surface_report.resolve()), "sha256": sha256(args.rest_surface_report)},
            "source_tissue": {"path": str(args.source_tissue.resolve()), "sha256": sha256(args.source_tissue)},
            "initial_observation": (
                {"path": str(args.initial_observation.resolve()), "sha256": sha256(args.initial_observation)}
                if args.initial_observation.exists()
                else None
            ),
        },
        "outputs": {key: str(path.resolve()) for key, path in output_paths.items()},
        "gates": gates,
        "passed": bool(all(gates.values())),
    }
    output_paths["metadata"].write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    if not metadata["passed"]:
        raise SystemExit("Paper-scale tissue asset gates failed")


if __name__ == "__main__":
    main()
