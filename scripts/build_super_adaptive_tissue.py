#!/usr/bin/env python3
"""Build an adaptive soft-tissue volume from an unsmoothed rest skin.

The observed top-surface vertices are immutable.  TetGen may add conforming
Steiner points on their piecewise-linear triangles, but it must not move an
input vertex or change the represented surface.  Mechanics use tetrahedral
nodes; the closed boundary triangle skin is saved separately as the future
authoritative tool-contact geometry.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import tetgen


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOT = REPO_ROOT / "data" / "super" / "grasp5_native"
MULTIVIEW_ROOT = NATIVE_ROOT / "tissue_multiview_v1"
V9_ROOT = NATIVE_ROOT / "bodies_v9_dense_0p5mm_rigid_tissue"

TOP_MARKER = 1
SIDE_MARKER = 2
BOTTOM_MARKER = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an adaptive tetrahedral SUPER tissue without smoothing."
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
        "--source-metadata",
        type=Path,
        default=V9_ROOT / "build_metadata.json",
    )
    parser.add_argument(
        "--ground-plane",
        type=Path,
        default=V9_ROOT / "ground_plane.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MULTIVIEW_ROOT / "soft_tissue_adaptive_v1",
    )
    parser.add_argument("--surface-spacing-mm", type=float, default=0.82)
    parser.add_argument("--dense-depth-mm", type=float, default=1.5)
    parser.add_argument("--transition-end-mm", type=float, default=8.0)
    parser.add_argument("--deep-spacing-mm", type=float, default=1.80)
    parser.add_argument(
        "--lateral-adaptive", action="store_true",
        help="Use grasp-centered fine/transition/outer spacing in table x-y.",
    )
    parser.add_argument("--lateral-fine-radius-mm", type=float, default=20.0)
    parser.add_argument("--lateral-transition-radius-mm", type=float, default=35.0)
    parser.add_argument("--lateral-outer-spacing-mm", type=float, default=2.0)
    parser.add_argument("--grasp-center-x-mm", type=float, default=None)
    parser.add_argument("--grasp-center-y-mm", type=float, default=None)
    parser.add_argument("--visual-radius-mm", type=float, default=0.50)
    parser.add_argument("--material-density-kg-m3", type=float, default=1000.0)
    parser.add_argument("--background-spacing-mm", type=float, default=2.0)
    parser.add_argument("--tetgen-min-ratio", type=float, default=20.0)
    parser.add_argument("--tetgen-min-dihedral-deg", type=float, default=0.0)
    parser.add_argument("--tetgen-steiner-limit", type=int, default=200000)
    parser.add_argument(
        "--minimum-tet-volume-mm3",
        type=float,
        default=1.0e-4,
        help="Reject sliver tetrahedra below this signed rest-volume threshold.",
    )
    parser.add_argument(
        "--maximum-tet-condition",
        type=float,
        default=250.0,
        help="Reject tetrahedra whose rest edge matrix exceeds this condition number.",
    )
    parser.add_argument(
        "--dense-neighbor-p95-factor",
        type=float,
        default=1.65,
        help=(
            "Maximum dense-layer nearest-neighbor p95 divided by the "
            "configured surface spacing."
        ),
    )
    parser.add_argument("--binding-candidates", type=int, default=64)
    parser.add_argument("--particle-hard-limit", type=int, default=45000)
    parser.add_argument("--tet-hard-limit", type=int, default=200000)
    parser.add_argument("--skin-hard-limit", type=int, default=30000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentiles(values: np.ndarray, q: tuple[float, ...] = (0, 5, 50, 95, 100)):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return np.percentile(values, q).tolist()


def compact_top_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    source_class: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    used = np.unique(faces)
    old_to_new = np.full(len(vertices), -1, dtype=np.int32)
    old_to_new[used] = np.arange(len(used), dtype=np.int32)
    return (
        vertices[used].astype(np.float64),
        old_to_new[faces].astype(np.int32),
        source_class[used].astype(np.uint8),
        used.astype(np.int32),
    )


def directed_boundary_edges(faces: np.ndarray) -> np.ndarray:
    incidence: Counter[tuple[int, int]] = Counter()
    directed: dict[tuple[int, int], tuple[int, int]] = {}
    for a, b, c in faces.tolist():
        for edge in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(edge))
            incidence[key] += 1
            directed[key] = edge
    if max(incidence.values(), default=0) != 2:
        offenders = sum(value > 2 for value in incidence.values())
        if offenders:
            raise RuntimeError(
                f"Top surface contains {offenders} non-manifold edges"
            )
    boundary = [
        directed[key] for key, count in incidence.items() if count == 1
    ]
    degree: Counter[int] = Counter()
    for a, b in boundary:
        degree[a] += 1
        degree[b] += 1
    if not boundary or any(value != 2 for value in degree.values()):
        raise RuntimeError("Top-surface boundary is not a collection of loops")
    return np.asarray(boundary, dtype=np.int32)


def ordered_boundary_loop(boundary: np.ndarray) -> list[int]:
    next_vertex: dict[int, int] = {}
    incoming: Counter[int] = Counter()
    for start, end in boundary.tolist():
        if start in next_vertex:
            raise RuntimeError("Boundary has more than one outgoing edge")
        next_vertex[start] = end
        incoming[end] += 1
    if len(next_vertex) != len(boundary) or any(
        count != 1 for count in incoming.values()
    ):
        raise RuntimeError("Boundary edges do not form one oriented loop")
    start = int(boundary[0, 0])
    loop = [start]
    while True:
        current = next_vertex[loop[-1]]
        if current == start:
            break
        if current in loop or len(loop) > len(boundary):
            raise RuntimeError("Boundary loop traversal did not close cleanly")
        loop.append(current)
    if len(loop) != len(boundary):
        raise RuntimeError("Top surface has more than one boundary loop")
    return loop


def polygon_signed_area(vertices_xy: np.ndarray, loop: list[int]) -> float:
    points = vertices_xy[loop]
    following = np.roll(points, -1, axis=0)
    return float(
        0.5
        * np.sum(points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1])
    )


def simplify_collinear_loop(
    vertices_xy: np.ndarray, loop: list[int]
) -> list[int]:
    kept: list[int] = []
    extent = np.ptp(vertices_xy[loop], axis=0)
    tolerance = float(np.max(extent) ** 2 * 1.0e-7)
    for index, current in enumerate(loop):
        previous = loop[index - 1]
        following = loop[(index + 1) % len(loop)]
        vector_a = vertices_xy[current] - vertices_xy[previous]
        vector_b = vertices_xy[following] - vertices_xy[current]
        cross = vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]
        if abs(cross) > tolerance:
            kept.append(current)
    if len(kept) < 3:
        raise RuntimeError("Boundary polygon collapsed during collinear cleanup")
    return kept


def point_in_ccw_triangle(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    tolerance: float,
) -> bool:
    def cross(left: np.ndarray, right: np.ndarray) -> float:
        return float(left[0] * right[1] - left[1] * right[0])

    return (
        cross(b - a, point - a) >= -tolerance
        and cross(c - b, point - b) >= -tolerance
        and cross(a - c, point - c) >= -tolerance
    )


def triangulate_bottom_polygon(
    vertices_xy: np.ndarray, full_loop: list[int]
) -> np.ndarray:
    """Ear-clip a simplified outline, then restore every fine boundary edge."""
    if polygon_signed_area(vertices_xy, full_loop) < 0.0:
        full_loop = list(reversed(full_loop))
    simple_loop = simplify_collinear_loop(vertices_xy, full_loop)
    active = simple_loop.copy()
    triangles: list[tuple[int, int, int]] = []
    extent = np.ptp(vertices_xy[full_loop], axis=0)
    # Coarse adaptive outlines contain long edges next to retained short
    # boundary details.  The former 1e-9 scale tolerance classified nearby
    # but exterior vertices as lying inside every candidate ear and could
    # stall a valid, non-self-intersecting polygon.  Keep the test close to
    # float64 geometric precision; actual degenerate triangles are still
    # rejected explicitly below.
    tolerance = max(float(np.max(extent) ** 2 * 1.0e-12), 1.0e-16)
    while len(active) > 3:
        ear_found = False
        for index, current in enumerate(active):
            previous = active[index - 1]
            following = active[(index + 1) % len(active)]
            a = vertices_xy[previous]
            b = vertices_xy[current]
            c = vertices_xy[following]
            ab = b - a
            bc = c - b
            cross_value = ab[0] * bc[1] - ab[1] * bc[0]
            # A coarse outline can contain long, almost-collinear runs whose
            # float32 source coordinates leave a tiny positive cross product.
            # Accepting such a point as an ear creates a long diagonal that
            # overlaps the same boundary run after the removed collinear
            # vertices are restored.  Use an angle-relative test in addition
            # to the absolute predicate tolerance.
            turn_tolerance = max(
                tolerance,
                1.0e-6 * float(np.linalg.norm(ab) * np.linalg.norm(bc)),
            )
            if cross_value <= turn_tolerance:
                continue
            contains_other = any(
                point_in_ccw_triangle(
                    vertices_xy[candidate], a, b, c, tolerance
                )
                for candidate in active
                if candidate not in (previous, current, following)
            )
            if contains_other:
                continue
            triangles.append((previous, current, following))
            del active[index]
            ear_found = True
            break
        if not ear_found:
            raise RuntimeError(
                "Ear clipping stalled on the tissue bottom outline"
            )
    triangles.append(tuple(active))

    loop_position = {vertex: index for index, vertex in enumerate(full_loop)}
    for edge_index, start in enumerate(simple_loop):
        end = simple_loop[(edge_index + 1) % len(simple_loop)]
        chain = [start]
        position = loop_position[start]
        while True:
            position = (position + 1) % len(full_loop)
            chain.append(full_loop[position])
            if full_loop[position] == end:
                break
        if len(chain) == 2:
            continue
        triangle_index = next(
            (
                index
                for index, triangle in enumerate(triangles)
                if start in triangle and end in triangle
            ),
            None,
        )
        if triangle_index is None:
            raise RuntimeError("Simplified bottom boundary edge was not triangulated")
        triangle = triangles.pop(triangle_index)
        opposite = next(vertex for vertex in triangle if vertex not in (start, end))
        for chain_start, chain_end in zip(chain[:-1], chain[1:], strict=True):
            candidate = (chain_start, chain_end, opposite)
            if polygon_signed_area(vertices_xy, list(candidate)) < 0.0:
                candidate = (chain_end, chain_start, opposite)
            triangles.append(candidate)
    output = np.asarray(triangles, dtype=np.int32)
    edge_a = vertices_xy[output[:, 1]] - vertices_xy[output[:, 0]]
    edge_b = vertices_xy[output[:, 2]] - vertices_xy[output[:, 0]]
    double_area = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
    if np.any(np.abs(double_area) <= tolerance):
        raise RuntimeError("Bottom triangulation contains a degenerate triangle")
    return output


def build_closed_plc(
    top_vertices: np.ndarray,
    top_faces: np.ndarray,
    side_max_spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boundary = directed_boundary_edges(top_faces)
    loop = ordered_boundary_loop(boundary)
    if polygon_signed_area(top_vertices[:, :2], loop) < 0.0:
        loop = list(reversed(loop))
        boundary = np.asarray(
            list(zip(loop, np.roll(loop, -1), strict=True)),
            dtype=np.int32,
        )
    bottom_top_faces = triangulate_bottom_polygon(top_vertices[:, :2], loop)
    added_vertices: list[np.ndarray] = []
    column_by_top: dict[int, list[int]] = {}
    for top_id in loop:
        top_height = float(top_vertices[top_id, 2])
        interval_count = max(1, int(np.ceil(top_height / side_max_spacing)))
        z_values = np.linspace(0.0, top_height, interval_count + 1)
        column: list[int] = []
        for z_value in z_values[:-1]:
            point = top_vertices[top_id].copy()
            point[2] = z_value
            column.append(len(top_vertices) + len(added_vertices))
            added_vertices.append(point)
        column.append(top_id)
        column_by_top[top_id] = column
    vertices = np.concatenate(
        (top_vertices, np.asarray(added_vertices, dtype=np.float64)), axis=0
    )
    bottom_by_top = {
        top_id: column_by_top[top_id][0] for top_id in loop
    }
    bottom_faces = np.asarray(
        [
            (
                bottom_by_top[int(face[0])],
                bottom_by_top[int(face[2])],
                bottom_by_top[int(face[1])],
            )
            for face in bottom_top_faces
        ],
        dtype=np.int32,
    )
    side_faces: list[tuple[int, int, int]] = []
    for top_a, top_b in boundary.tolist():
        column_a = column_by_top[top_a]
        column_b = column_by_top[top_b]
        index_a = 0
        index_b = 0
        while index_a < len(column_a) - 1 or index_b < len(column_b) - 1:
            next_a_z = (
                vertices[column_a[index_a + 1], 2]
                if index_a < len(column_a) - 1
                else np.inf
            )
            next_b_z = (
                vertices[column_b[index_b + 1], 2]
                if index_b < len(column_b) - 1
                else np.inf
            )
            current_a = column_a[index_a]
            current_b = column_b[index_b]
            if abs(next_a_z - next_b_z) <= 1.0e-10:
                next_a = column_a[index_a + 1]
                next_b = column_b[index_b + 1]
                side_faces.extend(
                    (
                        (current_a, current_b, next_a),
                        (next_a, current_b, next_b),
                    )
                )
                index_a += 1
                index_b += 1
            elif next_a_z < next_b_z:
                next_a = column_a[index_a + 1]
                side_faces.append((current_a, current_b, next_a))
                index_a += 1
            else:
                next_b = column_b[index_b + 1]
                side_faces.append((current_a, current_b, next_b))
                index_b += 1
    side_faces_array = np.asarray(side_faces, dtype=np.int32)
    faces = np.concatenate((top_faces, side_faces_array, bottom_faces), axis=0)
    markers = np.concatenate(
        (
            np.full(len(top_faces), TOP_MARKER, dtype=np.int32),
            np.full(len(side_faces_array), SIDE_MARKER, dtype=np.int32),
            np.full(len(bottom_faces), BOTTOM_MARKER, dtype=np.int32),
        )
    )

    incidence = Counter(
        tuple(sorted(edge))
        for a, b, c in faces.tolist()
        for edge in ((a, b), (b, c), (c, a))
    )
    histogram = Counter(incidence.values())
    if histogram != {2: len(incidence)}:
        raise RuntimeError(f"Closed PLC edge incidence is invalid: {histogram}")
    return vertices, faces, markers


def regular_axis(low: float, high: float, spacing: float) -> np.ndarray:
    count = max(1, int(np.ceil((high - low) / spacing)))
    return np.linspace(low, high, count + 1, dtype=np.float64)


def make_contact_distance_scene(
    vertices: np.ndarray, faces: np.ndarray
) -> o3d.t.geometry.RaycastingScene:
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices.astype(np.float32)),
        o3d.core.Tensor(faces.astype(np.uint32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    return scene


def target_spacing(
    points: np.ndarray,
    contact_distance_scene: o3d.t.geometry.RaycastingScene,
    dense_spacing: float,
    dense_depth: float,
    transition_end: float,
    deep_spacing: float,
    grasp_center_xy: np.ndarray | None = None,
    lateral_fine_radius: float = 0.0,
    lateral_transition_radius: float = 0.0,
    lateral_outer_spacing: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    inward_depth = contact_distance_scene.compute_distance(
        o3d.core.Tensor(points.astype(np.float32))
    ).numpy().astype(np.float64)
    alpha = np.clip(
        (inward_depth - dense_depth) / (transition_end - dense_depth),
        0.0,
        1.0,
    )
    # Geometric interpolation bounds the ratio change along any mesh edge;
    # unlike a vertical height lookup, it remains continuous at steep folds
    # and side walls.
    surface_spacing = np.full(len(points), dense_spacing, dtype=np.float64)
    if grasp_center_xy is not None:
        radius = np.linalg.norm(points[:, :2] - grasp_center_xy[None], axis=1)
        radial_alpha = np.clip(
            (radius - lateral_fine_radius)
            / (lateral_transition_radius - lateral_fine_radius),
            0.0,
            1.0,
        )
        surface_spacing = dense_spacing * np.exp(
            radial_alpha * np.log(lateral_outer_spacing / dense_spacing)
        )
    spacing = surface_spacing * np.exp(
        alpha * np.log(deep_spacing / surface_spacing)
    )
    return spacing, inward_depth


def build_background_mesh(
    closed_vertices: np.ndarray,
    contact_distance_scene: o3d.t.geometry.RaycastingScene,
    background_spacing: float,
    dense_spacing: float,
    dense_depth: float,
    transition_end: float,
    deep_spacing: float,
    grasp_center_xy: np.ndarray | None = None,
    lateral_fine_radius: float = 0.0,
    lateral_transition_radius: float = 0.0,
    lateral_outer_spacing: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minimum = closed_vertices.min(axis=0)
    maximum = closed_vertices.max(axis=0)
    # Keep the PLC strictly inside the background domain. TetGen's metric
    # point-location can otherwise fail on a large set of coincident outer
    # boundary points.
    padding = background_spacing
    x_axis = regular_axis(
        minimum[0] - padding, maximum[0] + padding, background_spacing
    )
    y_axis = regular_axis(
        minimum[1] - padding, maximum[1] + padding, background_spacing
    )
    z_axis = regular_axis(
        minimum[2] - padding, maximum[2] + padding, background_spacing
    )
    grid = np.stack(
        np.meshgrid(x_axis, y_axis, z_axis, indexing="ij"), axis=-1
    )
    vertices = grid.reshape(-1, 3)
    shape = grid.shape[:3]
    ids = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    tets: list[tuple[int, int, int, int]] = []
    for ix in range(shape[0] - 1):
        for iy in range(shape[1] - 1):
            for iz in range(shape[2] - 1):
                v000 = int(ids[ix, iy, iz])
                v100 = int(ids[ix + 1, iy, iz])
                v110 = int(ids[ix + 1, iy + 1, iz])
                v010 = int(ids[ix, iy + 1, iz])
                v001 = int(ids[ix, iy, iz + 1])
                v101 = int(ids[ix + 1, iy, iz + 1])
                v111 = int(ids[ix + 1, iy + 1, iz + 1])
                v011 = int(ids[ix, iy + 1, iz + 1])
                tets.extend(
                    (
                        (v000, v100, v110, v111),
                        (v000, v110, v010, v111),
                        (v000, v010, v011, v111),
                        (v000, v011, v001, v111),
                        (v000, v001, v101, v111),
                        (v000, v101, v100, v111),
                    )
                )
    tets_array = np.asarray(tets, dtype=np.int32)
    sizing, _ = target_spacing(
        vertices,
        contact_distance_scene,
        dense_spacing,
        dense_depth,
        transition_end,
        deep_spacing,
        grasp_center_xy,
        lateral_fine_radius,
        lateral_transition_radius,
        lateral_outer_spacing,
    )
    return vertices, tets_array, sizing


def signed_tet_volumes(
    positions: np.ndarray, tets: np.ndarray
) -> np.ndarray:
    points = positions[tets]
    return np.linalg.det(
        np.stack(
            (
                points[:, 1] - points[:, 0],
                points[:, 2] - points[:, 0],
                points[:, 3] - points[:, 0],
            ),
            axis=-1,
        )
    ) / 6.0


def orient_tets(
    positions: np.ndarray, tets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    tets = tets.copy()
    volume = signed_tet_volumes(positions, tets)
    inverted = volume < 0.0
    inverted_count = int(inverted.sum())
    if inverted_count:
        swap = tets[inverted].copy()
        swap[:, [1, 2]] = swap[:, [2, 1]]
        tets[inverted] = swap
        volume[inverted] *= -1.0
    return tets, volume, inverted_count


def oriented_boundary_faces(tets: np.ndarray) -> np.ndarray:
    face_map: dict[
        tuple[int, int, int], tuple[int, tuple[int, int, int]]
    ] = {}
    for a, b, c, d in tets.tolist():
        for face in ((a, c, b), (a, b, d), (a, d, c), (b, c, d)):
            key = tuple(sorted(face))
            count, stored = face_map.get(key, (0, face))
            face_map[key] = (count + 1, stored)
    if max((count for count, _ in face_map.values()), default=0) > 2:
        raise RuntimeError("Tetrahedral mesh contains a non-manifold face")
    return np.asarray(
        [face for count, face in face_map.values() if count == 1],
        dtype=np.int32,
    )


def transfer_boundary_markers(
    boundary_faces: np.ndarray,
    tetgen_faces: np.ndarray,
    tetgen_markers: np.ndarray,
) -> np.ndarray:
    marker_by_face = {
        tuple(sorted(face.tolist())): int(marker)
        for face, marker in zip(
            tetgen_faces, tetgen_markers, strict=True
        )
        if int(marker) > 0
    }
    markers = np.asarray(
        [
            marker_by_face.get(tuple(sorted(face.tolist())), 0)
            for face in boundary_faces
        ],
        dtype=np.uint8,
    )
    if np.any(markers == 0):
        raise RuntimeError(
            f"{int(np.count_nonzero(markers == 0))} boundary faces lost PLC markers"
        )
    return markers


def connected_component_count(num_vertices: int, tets: np.ndarray) -> int:
    parent = np.arange(num_vertices, dtype=np.int32)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for tet in tets:
        union(int(tet[0]), int(tet[1]))
        union(int(tet[0]), int(tet[2]))
        union(int(tet[0]), int(tet[3]))
    return len({find(int(index)) for index in np.unique(tets)})


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def transform_gaussians(source_tissue: dict) -> dict[str, np.ndarray]:
    transform = np.asarray(source_tissue["X_WB"], dtype=np.float64)
    gaussians = source_tissue["gaussians"]
    means = transform_points(
        transform, np.asarray(gaussians["means"], dtype=np.float64)
    )
    body_rotation = Rotation.from_matrix(transform[:3, :3])
    local_rotation = Rotation.from_quat(
        np.asarray(gaussians["quats"], dtype=np.float64)[:, [1, 2, 3, 0]]
    )
    world_xyzw = (body_rotation * local_rotation).as_quat()
    return {
        "means": means,
        "quats": world_xyzw[:, [3, 0, 1, 2]],
        "scales": np.asarray(gaussians["scales"], dtype=np.float64),
        "opacities": np.asarray(gaussians["opacities"], dtype=np.float64),
        "colors": np.asarray(gaussians["colors"], dtype=np.float64),
    }


def bind_gaussians(
    gaussian_means: np.ndarray,
    positions: np.ndarray,
    tets: np.ndarray,
    candidate_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tet_positions = positions[tets]
    origins = tet_positions[:, 0]
    rest_matrices = np.stack(
        (
            tet_positions[:, 1] - origins,
            tet_positions[:, 2] - origins,
            tet_positions[:, 3] - origins,
        ),
        axis=-1,
    )
    inverse_rest = np.linalg.inv(rest_matrices)
    centroids = tet_positions.mean(axis=1)
    tree = cKDTree(centroids)
    k = min(max(int(candidate_count), 1), len(tets))
    _, candidates = tree.query(gaussian_means, k=k, workers=-1)
    if k == 1:
        candidates = candidates[:, None]
    relative = gaussian_means[:, None, :] - origins[candidates]
    coordinates = np.einsum(
        "nkij,nkj->nki", inverse_rest[candidates], relative
    )
    weights = np.concatenate(
        (1.0 - coordinates.sum(axis=-1, keepdims=True), coordinates),
        axis=-1,
    )
    contained = np.all(weights >= -1.0e-7, axis=-1) & np.all(
        weights <= 1.0 + 1.0e-7, axis=-1
    )
    projected = np.clip(weights, 0.0, 1.0)
    projected /= np.maximum(projected.sum(axis=-1, keepdims=True), 1.0e-12)
    reconstructed = np.einsum(
        "nki,nkij->nkj", projected, tet_positions[candidates]
    )
    distance = np.linalg.norm(
        reconstructed - gaussian_means[:, None, :], axis=-1
    )
    distance[contained] = 0.0
    selected_column = np.argmin(distance, axis=1)
    row = np.arange(len(gaussian_means))
    tet_ids = candidates[row, selected_column]
    return (
        tet_ids.astype(np.int32),
        projected[row, selected_column],
        distance[row, selected_column],
        contained[row, selected_column],
    )


def rest_curvature(
    positions: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_faces: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_id, (a, b, c) in enumerate(faces.tolist()):
        for edge_a, edge_b, opposite in ((a, b, c), (b, c, a), (c, a, b)):
            key = tuple(sorted((edge_a, edge_b)))
            edge_faces.setdefault(key, []).append((face_id, opposite))
    edges: list[tuple[int, int]] = []
    opposites: list[tuple[int, int]] = []
    angles: list[float] = []
    face_points = positions[faces]
    normals = np.cross(
        face_points[:, 1] - face_points[:, 0],
        face_points[:, 2] - face_points[:, 0],
    )
    normals /= np.maximum(
        np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-15
    )
    for edge, adjacent in edge_faces.items():
        if len(adjacent) != 2:
            continue
        (left_face, left_opposite), (right_face, right_opposite) = adjacent
        cosine = np.clip(
            np.dot(normals[left_face], normals[right_face]), -1.0, 1.0
        )
        edges.append(edge)
        opposites.append((left_opposite, right_opposite))
        angles.append(float(np.arccos(cosine)))
    return (
        np.asarray(edges, dtype=np.int32),
        np.asarray(opposites, dtype=np.int32),
        np.asarray(angles, dtype=np.float32),
    )


def tet_edges(tets: np.ndarray) -> np.ndarray:
    return np.unique(
        np.sort(
            np.concatenate(
                (
                    tets[:, [0, 1]],
                    tets[:, [0, 2]],
                    tets[:, [0, 3]],
                    tets[:, [1, 2]],
                    tets[:, [1, 3]],
                    tets[:, [2, 3]],
                ),
                axis=0,
            ),
            axis=1,
        ),
        axis=0,
    )


def tet_condition_numbers(positions: np.ndarray, tets: np.ndarray) -> np.ndarray:
    points = positions[tets]
    matrices = np.stack(
        (
            points[:, 1] - points[:, 0],
            points[:, 2] - points[:, 0],
            points[:, 3] - points[:, 0],
        ),
        axis=-1,
    )
    return np.linalg.cond(matrices)


def write_previews(
    output_dir: Path,
    positions: np.ndarray,
    surface_faces: np.ndarray,
    surface_markers: np.ndarray,
    inward_depth: np.ndarray,
    gaussian_means: np.ndarray,
    binding_distance: np.ndarray,
) -> None:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(positions)
    mesh.triangles = o3d.utility.Vector3iVector(surface_faces)
    vertex_colors = np.zeros((len(positions), 3), dtype=np.float64)
    vertex_marker = np.full(len(positions), BOTTOM_MARKER, dtype=np.uint8)
    vertex_marker[np.unique(surface_faces[surface_markers == SIDE_MARKER])] = (
        SIDE_MARKER
    )
    vertex_marker[np.unique(surface_faces[surface_markers == TOP_MARKER])] = (
        TOP_MARKER
    )
    vertex_colors[vertex_marker == TOP_MARKER] = (0.84, 0.32, 0.28)
    vertex_colors[vertex_marker == SIDE_MARKER] = (0.94, 0.62, 0.30)
    vertex_colors[vertex_marker == BOTTOM_MARKER] = (0.30, 0.48, 0.82)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(
        str(output_dir / "tissue_collision_skin.ply"),
        mesh,
        write_ascii=False,
    ):
        raise RuntimeError("Failed to write tissue_collision_skin.ply")

    node_colors = np.zeros((len(positions), 3), dtype=np.float64)
    node_colors[inward_depth <= 0.003] = (0.90, 0.20, 0.18)
    node_colors[(inward_depth > 0.003) & (inward_depth <= 0.005)] = (
        0.95,
        0.75,
        0.15,
    )
    node_colors[inward_depth > 0.005] = (0.20, 0.50, 0.90)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(positions)
    cloud.colors = o3d.utility.Vector3dVector(node_colors)
    if not o3d.io.write_point_cloud(
        str(output_dir / "tissue_mechanics_nodes.ply"),
        cloud,
        write_ascii=False,
    ):
        raise RuntimeError("Failed to write tissue_mechanics_nodes.ply")

    binding_colors = np.zeros((len(gaussian_means), 3), dtype=np.float64)
    binding_colors[binding_distance <= 0.0005] = (0.10, 0.80, 0.20)
    binding_colors[
        (binding_distance > 0.0005) & (binding_distance <= 0.0015)
    ] = (0.95, 0.75, 0.10)
    binding_colors[binding_distance > 0.0015] = (0.90, 0.10, 0.10)
    binding_cloud = o3d.geometry.PointCloud()
    binding_cloud.points = o3d.utility.Vector3dVector(gaussian_means)
    binding_cloud.colors = o3d.utility.Vector3dVector(binding_colors)
    if not o3d.io.write_point_cloud(
        str(output_dir / "tissue_gaussian_binding.ply"),
        binding_cloud,
        write_ascii=False,
    ):
        raise RuntimeError("Failed to write tissue_gaussian_binding.ply")


def main() -> None:
    args = parse_args()
    output_paths = {
        "asset": args.output_dir / "tissue_soft_adaptive.npz",
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
    if not (
        0.0 < args.surface_spacing_mm <= args.deep_spacing_mm
        and 0.0 < args.dense_depth_mm < args.transition_end_mm
        and args.material_density_kg_m3 > 0.0
        and args.dense_neighbor_p95_factor > 1.0
    ):
        raise ValueError("Invalid adaptive sizing or material parameters")
    if tetgen.__version__ != "0.8.3":
        raise RuntimeError(
            f"This builder is validated with tetgen 0.8.3, got {tetgen.__version__}"
        )

    rest_report = read_json(args.rest_surface_report)
    if not rest_report.get("passed", False):
        raise RuntimeError("Unsmoothed rest-surface report did not pass")
    with np.load(args.rest_surface) as loaded:
        surface = {key: loaded[key].copy() for key in loaded.files}
    grasp_center_xy: np.ndarray | None = None
    if args.lateral_adaptive:
        if not (
            0.0 < args.lateral_fine_radius_mm
            < args.lateral_transition_radius_mm
            and args.surface_spacing_mm
            <= args.lateral_outer_spacing_mm
            <= args.deep_spacing_mm
        ):
            raise ValueError("Invalid lateral adaptive sizing parameters")
        explicit_center = (
            args.grasp_center_x_mm is not None
            and args.grasp_center_y_mm is not None
        )
        if explicit_center:
            grasp_center_xy = np.asarray(
                [args.grasp_center_x_mm, args.grasp_center_y_mm],
                dtype=np.float64,
            ) / 1000.0
        elif "grasp_roi_center_xy_table" in surface:
            grasp_center_xy = np.asarray(
                surface["grasp_roi_center_xy_table"], dtype=np.float64
            )
        else:
            raise ValueError(
                "Lateral adaptation needs explicit grasp center or a surface ROI"
            )
    top_vertices, top_faces, top_classes, source_top_ids = compact_top_surface(
        surface["surface_vertices_table"],
        surface["surface_faces"],
        surface["surface_vertex_source_class"],
    )
    dense_spacing = args.surface_spacing_mm / 1000.0
    dense_depth = args.dense_depth_mm / 1000.0
    transition_end = args.transition_end_mm / 1000.0
    deep_spacing = args.deep_spacing_mm / 1000.0
    closed_vertices, closed_faces, closed_markers = build_closed_plc(
        top_vertices, top_faces, deep_spacing
    )
    contact_distance_scene = make_contact_distance_scene(
        top_vertices, top_faces
    )
    background_vertices, background_tets, background_target = (
        build_background_mesh(
            closed_vertices,
            contact_distance_scene,
            args.background_spacing_mm / 1000.0,
            dense_spacing,
            dense_depth,
            transition_end,
            deep_spacing,
            grasp_center_xy,
            args.lateral_fine_radius_mm / 1000.0,
            args.lateral_transition_radius_mm / 1000.0,
            args.lateral_outer_spacing_mm / 1000.0,
        )
    )

    generator = tetgen.TetGen(
        closed_vertices, closed_faces, closed_markers
    )
    # tetgen-python currently exposes background array loading through its
    # wrapped TetGen object, while its public helper requires PyVista/VTK.
    generator._tetgen.load_bgmesh_from_arrays(
        background_vertices.astype(np.float64, copy=False),
        background_tets.astype(np.int32, copy=False),
        background_target.astype(np.float64, copy=False),
    )
    nodes, tets, _, _ = generator.tetrahedralize(
        plc=True,
        quality=True,
        metric=True,
        facesout=True,
        edgesout=False,
        nojettison=True,
        minratio=args.tetgen_min_ratio,
        mindihedral=args.tetgen_min_dihedral_deg,
        steinerleft=args.tetgen_steiner_limit,
        quiet=True,
    )
    nodes = np.asarray(nodes, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int32)
    tets, volumes, tetgen_inverted_output = orient_tets(nodes, tets)
    volume_epsilon = dense_spacing**3 * 1.0e-10
    degenerate_count = int(np.count_nonzero(volumes <= volume_epsilon))
    if degenerate_count:
        raise RuntimeError(f"TetGen produced {degenerate_count} degenerate tets")
    boundary_faces = oriented_boundary_faces(tets)
    boundary_markers = transfer_boundary_markers(
        boundary_faces,
        np.asarray(generator.trifaces, dtype=np.int32),
        np.asarray(generator.triface_markers, dtype=np.int32),
    )
    component_count = connected_component_count(len(nodes), tets)

    source_tree = cKDTree(nodes)
    source_distance, source_node_ids = source_tree.query(
        top_vertices, k=1, workers=-1
    )
    source_vertex_max_error = float(source_distance.max())
    if source_vertex_max_error > 1.0e-10:
        raise RuntimeError(
            "TetGen moved or removed a frozen top vertex; "
            f"maximum error={source_vertex_max_error}"
        )
    if len(np.unique(source_node_ids)) != len(top_vertices):
        raise RuntimeError("Frozen top-vertex map is not one-to-one")

    node_target, inward_depth = target_spacing(
        nodes,
        contact_distance_scene,
        dense_spacing,
        dense_depth,
        transition_end,
        deep_spacing,
        grasp_center_xy,
        args.lateral_fine_radius_mm / 1000.0,
        args.lateral_transition_radius_mm / 1000.0,
        args.lateral_outer_spacing_mm / 1000.0,
    )
    nearest_neighbor = cKDTree(nodes).query(
        nodes, k=2, workers=-1
    )[0][:, 1]
    dense_mask = inward_depth <= dense_depth + 1.0e-9
    transition_mask = (
        (inward_depth > dense_depth + 1.0e-9)
        & (inward_depth <= transition_end + 1.0e-9)
    )
    deep_mask = inward_depth > transition_end + 1.0e-9
    if grasp_center_xy is None:
        lateral_radius = np.zeros(len(nodes), dtype=np.float64)
        lateral_fine_mask = np.zeros(len(nodes), dtype=bool)
        lateral_transition_mask = np.zeros(len(nodes), dtype=bool)
        lateral_outer_mask = np.zeros(len(nodes), dtype=bool)
    else:
        lateral_radius = np.linalg.norm(
            nodes[:, :2] - grasp_center_xy[None], axis=1
        )
        lateral_fine_mask = (
            lateral_radius <= args.lateral_fine_radius_mm / 1000.0 + 1.0e-9
        )
        lateral_transition_mask = (
            (lateral_radius > args.lateral_fine_radius_mm / 1000.0 + 1.0e-9)
            & (
                lateral_radius
                <= args.lateral_transition_radius_mm / 1000.0 + 1.0e-9
            )
        )
        lateral_outer_mask = (
            lateral_radius
            > args.lateral_transition_radius_mm / 1000.0 + 1.0e-9
        )

    density = float(args.material_density_kg_m3)
    masses = np.zeros(len(nodes), dtype=np.float64)
    corner_mass = density * volumes / 4.0
    for corner in range(4):
        np.add.at(masses, tets[:, corner], corner_mass)

    bottom_node_mask = np.isclose(nodes[:, 2], 0.0, atol=1.0e-9)
    side_node_mask = np.zeros(len(nodes), dtype=bool)
    side_node_mask[
        np.unique(boundary_faces[boundary_markers == SIDE_MARKER])
    ] = True
    support_candidate_mask = bottom_node_mask & side_node_mask
    fixed_mask = np.zeros(len(nodes), dtype=bool)
    surface_node_mask = np.zeros(len(nodes), dtype=bool)
    surface_node_mask[np.unique(boundary_faces)] = True
    top_node_mask = np.zeros(len(nodes), dtype=bool)
    top_node_mask[
        np.unique(boundary_faces[boundary_markers == TOP_MARKER])
    ] = True

    gaussian = transform_gaussians(read_json(args.source_tissue))
    tet_ids, barycentric, binding_distance, contained = bind_gaussians(
        gaussian["means"], nodes, tets, args.binding_candidates
    )
    binding_particles = tets[tet_ids]
    reconstructed = np.einsum(
        "ni,nij->nj", barycentric, nodes[binding_particles]
    )
    rest_offsets = gaussian["means"] - reconstructed
    if not np.allclose(
        np.linalg.norm(rest_offsets, axis=1),
        binding_distance,
        atol=1.0e-9,
    ):
        raise RuntimeError("Gaussian rest binding failed reconstruction check")

    curvature_faces = boundary_faces[boundary_markers == TOP_MARKER]
    curvature_edges, curvature_opposites, curvature_angles = rest_curvature(
        nodes, curvature_faces
    )
    edges = tet_edges(tets)
    edge_lengths = np.linalg.norm(
        nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1
    )
    edge_target_ratio = np.maximum(
        node_target[edges[:, 0]], node_target[edges[:, 1]]
    ) / np.minimum(node_target[edges[:, 0]], node_target[edges[:, 1]])
    edge_length_over_target = edge_lengths / (
        0.5 * (node_target[edges[:, 0]] + node_target[edges[:, 1]])
    )
    conditions = tet_condition_numbers(nodes, tets)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_paths["asset"],
        rest_positions_table=nodes.astype(np.float32),
        tet_indices=tets.astype(np.int32),
        surface_faces=boundary_faces.astype(np.int32),
        surface_face_markers=boundary_markers.astype(np.uint8),
        collision_skin_faces=boundary_faces.astype(np.int32),
        collision_skin_face_markers=boundary_markers.astype(np.uint8),
        collision_skin_enabled_faces=np.ones(
            len(boundary_faces), dtype=bool
        ),
        particle_mass=masses.astype(np.float32),
        particle_radius=np.zeros(len(nodes), dtype=np.float32),
        particle_visual_radius=np.full(
            len(nodes), args.visual_radius_mm / 1000.0, dtype=np.float32
        ),
        particle_target_spacing=node_target.astype(np.float32),
        particle_inward_depth=inward_depth.astype(np.float32),
        particle_lateral_density_zone=np.where(
            lateral_fine_mask,
            0,
            np.where(lateral_transition_mask, 1, 2),
        ).astype(np.uint8),
        grasp_roi_center_xy_table=(
            np.full(2, np.nan, dtype=np.float32)
            if grasp_center_xy is None
            else grasp_center_xy.astype(np.float32)
        ),
        surface_node_mask=surface_node_mask,
        top_node_mask=top_node_mask,
        visible_surface_mask=top_node_mask,
        fixed_mask=fixed_mask,
        support_candidate_mask=support_candidate_mask,
        rest_tet_volume=volumes.astype(np.float32),
        pbd_edge_indices=edges.astype(np.int32),
        pbd_rest_edge_length=edge_lengths.astype(np.float32),
        pbd_shape_cluster_indices=tets.astype(np.int32),
        frozen_source_surface_vertex_ids=source_top_ids,
        frozen_source_surface_particle_ids=source_node_ids.astype(np.int32),
        top_rest_curvature_faces=curvature_faces.astype(np.int32),
        top_rest_curvature_edges=curvature_edges,
        top_rest_curvature_opposite_vertices=curvature_opposites,
        top_rest_dihedral_angle=curvature_angles,
        gaussian_rest_means_table=gaussian["means"].astype(np.float32),
        gaussian_rest_quats_table_wxyz=gaussian["quats"].astype(np.float32),
        gaussian_scales=gaussian["scales"].astype(np.float32),
        gaussian_opacities=gaussian["opacities"].astype(np.float32),
        gaussian_colors_rgb=gaussian["colors"].astype(np.float32),
        gaussian_tet_ids=tet_ids,
        gaussian_particle_indices=binding_particles.astype(np.int32),
        gaussian_barycentric_weights=barycentric.astype(np.float32),
        gaussian_binding_distance=binding_distance.astype(np.float32),
        gaussian_rest_offset_table=rest_offsets.astype(np.float32),
    )
    write_previews(
        args.output_dir,
        nodes,
        boundary_faces,
        boundary_markers,
        inward_depth,
        gaussian["means"],
        binding_distance,
    )

    counts = {
        "particles": int(len(nodes)),
        "tetrahedra": int(len(tets)),
        "skin_triangles": int(len(boundary_faces)),
        "skin_top_triangles": int(np.count_nonzero(boundary_markers == TOP_MARKER)),
        "skin_side_triangles": int(np.count_nonzero(boundary_markers == SIDE_MARKER)),
        "skin_bottom_triangles": int(
            np.count_nonzero(boundary_markers == BOTTOM_MARKER)
        ),
        "dense_layer_particles": int(dense_mask.sum()),
        "transition_layer_particles": int(transition_mask.sum()),
        "deep_layer_particles": int(deep_mask.sum()),
        "lateral_fine_particles": int(lateral_fine_mask.sum()),
        "lateral_transition_particles": int(lateral_transition_mask.sum()),
        "lateral_outer_particles": int(lateral_outer_mask.sum()),
        "surface_particles": int(surface_node_mask.sum()),
        "top_particles": int(top_node_mask.sum()),
        "support_candidate_particles": int(support_candidate_mask.sum()),
        "gaussians": int(len(gaussian["means"])),
    }
    source_metadata = read_json(args.source_metadata)
    legacy_mass_g = float(
        source_metadata.get("mass_preservation", {}).get(
            "source_sphere_sum_mass_g", np.nan
        )
    )
    gates = {
        "rest_surface_report_passed": True,
        "no_spatial_smoothing": True,
        "all_frozen_top_vertices_preserved_below_1e-7mm": (
            source_vertex_max_error <= 1.0e-10
        ),
        "single_tet_component": component_count == 1,
        "no_degenerate_tetrahedra": degenerate_count == 0,
        "all_rest_volumes_positive": bool(np.all(volumes > 0.0)),
        "closed_skin_has_only_known_markers": bool(
            np.all(np.isin(boundary_markers, (TOP_MARKER, SIDE_MARKER, BOTTOM_MARKER)))
        ),
        "particle_budget": len(nodes) <= args.particle_hard_limit,
        "tetrahedron_budget": len(tets) <= args.tet_hard_limit,
        "skin_triangle_budget": len(boundary_faces) <= args.skin_hard_limit,
        "dense_layer_nearest_neighbor_p95_within_configured_factor": bool(
            np.percentile(nearest_neighbor[dense_mask], 95)
            <= args.dense_neighbor_p95_factor * dense_spacing
        ),
        "finite_condition_numbers": bool(np.isfinite(conditions).all()),
        "minimum_tetrahedron_volume_above_configured_limit": bool(
            volumes.min() * 1.0e9 >= args.minimum_tet_volume_mm3
        ),
        "maximum_rest_matrix_condition_below_configured_limit": bool(
            conditions.max() <= args.maximum_tet_condition
        ),
        "neighbor_target_spacing_ratio_below_1p35": bool(
            edge_target_ratio.max() <= 1.35
        ),
    }
    metadata = {
        "asset_version": 10,
        "stage": "A_adaptive_soft_asset_unsmoothed_multiview",
        "runtime_enabled": True,
        "contact_runtime_available": True,
        "contact_runtime_enabled": False,
        "representation": {
            "mechanics": "tetrahedral particle nodes",
            "tool_contact": "closed external triangle skin; particles have zero collision radius",
            "gaussians": "four-node tetrahedral barycentric binding",
            "spatial_smoothing": False,
            "topology_completion": rest_report["parameters"].get(
                "hole_completion"
            ),
        },
        "parameters": {
            "surface_spacing_mm": args.surface_spacing_mm,
            "dense_depth_mm": args.dense_depth_mm,
            "transition_end_mm": args.transition_end_mm,
            "deep_spacing_mm": args.deep_spacing_mm,
            "lateral_adaptive": args.lateral_adaptive,
            "lateral_fine_radius_mm": args.lateral_fine_radius_mm,
            "lateral_transition_radius_mm": args.lateral_transition_radius_mm,
            "lateral_outer_spacing_mm": args.lateral_outer_spacing_mm,
            "grasp_center_xy_mm": (
                None if grasp_center_xy is None else (grasp_center_xy * 1000.0).tolist()
            ),
            "visual_radius_mm": args.visual_radius_mm,
            "mechanics_collision_radius_mm": 0.0,
            "material_density_kg_m3": density,
            "background_spacing_mm": args.background_spacing_mm,
            "tetgen_version": tetgen.__version__,
            "tetgen_min_ratio": args.tetgen_min_ratio,
            "tetgen_min_dihedral_deg": args.tetgen_min_dihedral_deg,
            "minimum_tet_volume_mm3": args.minimum_tet_volume_mm3,
            "maximum_tet_condition": args.maximum_tet_condition,
            "dense_neighbor_p95_factor": args.dense_neighbor_p95_factor,
        },
        "counts": counts,
        "topology": {
            "connected_components": component_count,
            "tetgen_negative_orientation_tets_corrected": tetgen_inverted_output,
            "degenerate_tetrahedra": degenerate_count,
            "closed_skin": True,
            "boundary_marker_values": {
                "top": TOP_MARKER,
                "side": SIDE_MARKER,
                "bottom": BOTTOM_MARKER,
            },
        },
        "surface_fidelity": {
            "frozen_source_vertices": int(len(top_vertices)),
            "maximum_frozen_vertex_displacement_mm": (
                source_vertex_max_error * 1000.0
            ),
            "source_surface_vertices_modified": False,
            "tetgen_surface_rule": (
                "only conforming Steiner subdivision of frozen piecewise-linear "
                "PLC facets is allowed"
            ),
            "inferred_top_vertex_count": int(np.count_nonzero(top_classes == 4)),
        },
        "adaptive_density": {
            "lateral_rule": (
                "geometric spacing interpolation from center surface spacing "
                "to outer spacing across the grasp-centered transition annulus; "
                "then geometric interpolation toward deep spacing by inward depth"
                if grasp_center_xy is not None
                else "surface-distance-only spacing"
            ),
            "nearest_neighbor_mm": {
                "all_min_p05_p50_p95_max": percentiles(
                    nearest_neighbor * 1000.0
                ),
                (
                    f"dense_0_to_{args.dense_depth_mm:g}mm_"
                    "min_p05_p50_p95_max"
                ): percentiles(
                    nearest_neighbor[dense_mask] * 1000.0
                ),
                (
                    f"transition_{args.dense_depth_mm:g}_to_"
                    f"{args.transition_end_mm:g}mm_min_p05_p50_p95_max"
                ): percentiles(
                    nearest_neighbor[transition_mask] * 1000.0
                ),
                (
                    f"deep_over_{args.transition_end_mm:g}mm_"
                    "min_p05_p50_p95_max"
                ): percentiles(
                    nearest_neighbor[deep_mask] * 1000.0
                ),
            },
            "target_spacing_mm_min_p05_p50_p95_max": percentiles(
                node_target * 1000.0
            ),
        },
        "tetrahedron_quality": {
            "volume_mm3_min_p05_p50_p95_max": percentiles(
                volumes * 1.0e9
            ),
            "edge_length_mm_min_p05_p50_p95_max": percentiles(
                edge_lengths * 1000.0
            ),
            "neighbor_target_spacing_ratio_min_p05_p50_p95_max": percentiles(
                edge_target_ratio
            ),
            "edge_length_over_local_target_min_p05_p50_p95_max": percentiles(
                edge_length_over_target
            ),
            "rest_matrix_condition_min_p05_p50_p95_p99_max": percentiles(
                conditions, (0, 5, 50, 95, 99, 100)
            ),
        },
        "mass": {
            "material_density_kg_m3": density,
            "tet_mesh_volume_cm3": float(volumes.sum() * 1.0e6),
            "tet_mesh_mass_g": float(masses.sum() * 1000.0),
            "legacy_overlapping_sphere_sum_mass_g": legacy_mass_g,
            "rule": (
                "node mass is integrated from incident tetra rest volume; "
                "legacy overlapping display-sphere volume is not reused"
            ),
        },
        "gaussian_binding": {
            "directly_contained": int(contained.sum()),
            "projected_to_nearest_candidate_tet": int((~contained).sum()),
            "distance_mm_min_p50_p90_p95_p99_max": percentiles(
                binding_distance * 1000.0, (0, 50, 90, 95, 99, 100)
            ),
            "weight_sum_max_error": float(
                np.max(np.abs(barycentric.sum(axis=1) - 1.0))
            ),
        },
        "rest_curvature": {
            "top_internal_edges": int(len(curvature_edges)),
            "rest_dihedral_deg_min_p05_p50_p95_max": percentiles(
                np.rad2deg(curvature_angles)
            ),
            "runtime_constraint_enabled": False,
        },
        "inputs": {
            "rest_surface": {
                "path": str(args.rest_surface.resolve()),
                "sha256": sha256(args.rest_surface),
            },
            "rest_surface_report": {
                "path": str(args.rest_surface_report.resolve()),
                "sha256": sha256(args.rest_surface_report),
            },
            "source_tissue_gaussians": {
                "path": str(args.source_tissue.resolve()),
                "sha256": sha256(args.source_tissue),
            },
            "ground_plane": {
                "path": str(args.ground_plane.resolve()),
                "sha256": sha256(args.ground_plane),
            },
        },
        "outputs": {
            key: str(path.resolve()) for key, path in output_paths.items()
        },
        "gates": gates,
        "passed": bool(all(gates.values())),
    }
    output_paths["metadata"].write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    if not metadata["passed"]:
        raise SystemExit("Adaptive tissue asset gates failed")


if __name__ == "__main__":
    main()
