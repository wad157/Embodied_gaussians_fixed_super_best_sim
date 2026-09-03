#!/usr/bin/env python3
"""Build a grasp-centered adaptive visual surface from the dense stereo union."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay, cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
MULTIVIEW_ROOT = REPO_ROOT / "data/super/grasp5_native/tissue_multiview_v1"
DEFAULT_INPUT = MULTIVIEW_ROOT / "rest_surface_gaussian_view_union_v3/rest_surface.npz"
DEFAULT_INPUT_REPORT = MULTIVIEW_ROOT / "rest_surface_gaussian_view_union_v3/report.json"
DEFAULT_POSE_DRIVER = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_paper_lnd_stereo_dense_contact_v4/"
    "surgicalsam2_multianchor_parts_dense_contact_v6/"
    "online_stereo_cma_se3_unbounded_dense_contact_v4/gui_unbounded_xyz_v1/"
    "psm_paper_lnd_gui_pose_driver.npz"
)
DEFAULT_OUTPUT_DIR = MULTIVIEW_ROOT / "rest_surface_adaptive_grasp_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-surface", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--input-report", type=Path, default=DEFAULT_INPUT_REPORT)
    parser.add_argument("--pose-driver", type=Path, default=DEFAULT_POSE_DRIVER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fine-radius-mm", type=float, default=20.0)
    parser.add_argument("--transition-radius-mm", type=float, default=35.0)
    parser.add_argument("--fine-stride", type=int, default=1)
    parser.add_argument("--transition-stride", type=int, default=2)
    parser.add_argument("--outer-stride", type=int, default=3)
    parser.add_argument("--maximum-edge-mm", type=float, default=2.0)
    parser.add_argument(
        "--minimum-faces",
        type=int,
        default=1000,
        help="Reject an adaptive surface smaller than this explicit budget.",
    )
    parser.add_argument(
        "--maximum-hole-boundary-vertices",
        type=int,
        default=8,
        help=(
            "Close only decimation-created inner loops no larger than this; "
            "the largest outer boundary is always preserved."
        ),
    )
    parser.add_argument("--closed-jaw-angle-max-rad", type=float, default=0.10)
    parser.add_argument("--near-surface-jaw-height-mm", type=float, default=25.0)
    parser.add_argument(
        "--preserve-view-exclusive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force all left-only and right-only dense vertices into the mesh.",
    )
    parser.add_argument(
        "--preserve-every-boundary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep every dense outline vertex; disable for a coarser mechanics PLC.",
    )
    parser.add_argument(
        "--preserve-inferred-completion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Force every inferred completion vertex into the surface. Disable "
            "for a mechanics PLC so inferred pixels obey the configured stride."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def derive_grasp_center(
    path: Path, maximum_height_m: float, maximum_jaw_angle: float
) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as loaded:
        names = loaded["link_names"]
        poses = loaded["poses_gui_world_xyz_xyzw"].astype(np.float64)
        q7 = loaded["q7"].astype(np.float64)
        timestamps = loaded["timestamps"].astype(np.float64)
    jaw_names = (
        "PSM1_tool_wrist_sca_ee_link_1",
        "PSM1_tool_wrist_sca_ee_link_2",
    )
    jaw_ids = []
    for name in jaw_names:
        ids = np.flatnonzero(names == name)
        if len(ids) != 1:
            raise RuntimeError(f"Pose driver does not contain one {name}")
        jaw_ids.append(int(ids[0]))
    midpoint = poses[:, jaw_ids, :3].mean(axis=1)
    selected = (midpoint[:, 2] <= maximum_height_m) & (
        q7[:, 6] <= maximum_jaw_angle
    )
    if np.count_nonzero(selected) < 10:
        raise RuntimeError("Too few low, closed-jaw samples to derive grasp center")
    center = np.median(midpoint[selected, :2], axis=0)
    audit = {
        "selection_count": int(selected.sum()),
        "xy_median_mm": (center * 1000.0).tolist(),
        "xy_mean_mm": (midpoint[selected, :2].mean(axis=0) * 1000.0).tolist(),
        "xy_p05_p95_mm": np.percentile(
            midpoint[selected, :2] * 1000.0, [5, 95], axis=0
        ).tolist(),
        "time_range_s": [
            float(timestamps[selected].min()),
            float(timestamps[selected].max()),
        ],
        "maximum_height_mm": maximum_height_m * 1000.0,
        "maximum_jaw_angle_rad": maximum_jaw_angle,
    }
    return center, audit


def boundary_mask(occupied: np.ndarray) -> np.ndarray:
    padded = np.pad(occupied, 1, mode="constant", constant_values=False)
    interior = occupied.copy()
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        interior &= padded[
            1 + di : 1 + di + occupied.shape[0],
            1 + dj : 1 + dj + occupied.shape[1],
        ]
    return occupied & ~interior


def sample_occupied(
    xy: np.ndarray,
    x0: float,
    y0: float,
    spacing: float,
    occupied: np.ndarray,
) -> np.ndarray:
    i = np.rint((xy[:, 0] - x0) / spacing).astype(np.int32)
    j = np.rint((xy[:, 1] - y0) / spacing).astype(np.int32)
    inside = (
        (i >= 0)
        & (i < occupied.shape[0])
        & (j >= 0)
        & (j < occupied.shape[1])
    )
    result = np.zeros(len(xy), dtype=bool)
    result[inside] = occupied[i[inside], j[inside]]
    return result


def connected_face_components(face_count: int, faces: np.ndarray) -> np.ndarray:
    parent = np.arange(face_count, dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    edge_owner: dict[tuple[int, int], int] = {}
    for face_id, (a, b, c) in enumerate(faces.tolist()):
        for edge in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(edge))
            if key in edge_owner:
                union(face_id, edge_owner[key])
            else:
                edge_owner[key] = face_id
    return np.asarray([find(i) for i in range(face_count)], dtype=np.int32)


def split_boundary_pinches(
    vertices: np.ndarray,
    faces: np.ndarray,
    classes: np.ndarray,
    source_vertex_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Duplicate coincident boundary vertices until every boundary degree is two.

    A binary occupancy outline can contain a hole that touches the exterior at
    exactly one grid vertex.  Keeping that vertex shared produces a geometric
    point contact but a non-manifold visual topology.  Splitting its distinct
    incident face fans preserves every measured point and triangle while making
    each boundary loop topologically valid.
    """
    vertices_out = vertices.tolist()
    classes_out = classes.tolist()
    source_ids_out = source_vertex_ids.tolist()
    faces_out = faces.copy()
    split_count = 0
    while True:
        edge_faces: dict[tuple[int, int], list[int]] = {}
        vertex_faces: dict[int, list[int]] = {}
        for face_id, triangle in enumerate(faces_out.tolist()):
            for vertex in triangle:
                vertex_faces.setdefault(vertex, []).append(face_id)
            a, b, c = triangle
            for edge in ((a, b), (b, c), (c, a)):
                edge_faces.setdefault(tuple(sorted(edge)), []).append(face_id)
        boundary_degree: dict[int, int] = {}
        for edge, owners in edge_faces.items():
            if len(owners) == 1:
                for vertex in edge:
                    boundary_degree[vertex] = boundary_degree.get(vertex, 0) + 1
        pinches = [
            vertex for vertex, degree in boundary_degree.items() if degree != 2
        ]
        if not pinches:
            break
        progressed = False
        for vertex in pinches:
            incident = vertex_faces[vertex]
            incident_set = set(incident)
            adjacency = {face_id: set() for face_id in incident}
            for edge, owners in edge_faces.items():
                if vertex not in edge or len(owners) != 2:
                    continue
                left, right = owners
                if left in incident_set and right in incident_set:
                    adjacency[left].add(right)
                    adjacency[right].add(left)
            fans: list[list[int]] = []
            unseen = set(incident)
            while unseen:
                seed = unseen.pop()
                stack = [seed]
                fan = [seed]
                while stack:
                    current = stack.pop()
                    for neighbor in adjacency[current]:
                        if neighbor in unseen:
                            unseen.remove(neighbor)
                            stack.append(neighbor)
                            fan.append(neighbor)
                fans.append(fan)
            if len(fans) <= 1:
                raise RuntimeError(
                    f"Boundary vertex {vertex} has degree {boundary_degree[vertex]} "
                    "but only one incident face fan"
                )
            for fan in fans[1:]:
                duplicate = len(vertices_out)
                vertices_out.append(vertices_out[vertex])
                classes_out.append(classes_out[vertex])
                source_ids_out.append(source_ids_out[vertex])
                for face_id in fan:
                    faces_out[face_id, faces_out[face_id] == vertex] = duplicate
                split_count += 1
            progressed = True
        if not progressed:
            raise RuntimeError("Could not split non-manifold boundary pinches")
    return (
        np.asarray(vertices_out, dtype=vertices.dtype),
        faces_out.astype(np.int32),
        np.asarray(classes_out, dtype=classes.dtype),
        np.asarray(source_ids_out, dtype=source_vertex_ids.dtype),
        split_count,
    )


def boundary_degrees(faces: np.ndarray) -> dict[int, int]:
    edge_incidence: dict[tuple[int, int], int] = {}
    for a, b, c in faces.tolist():
        for edge in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(edge))
            edge_incidence[key] = edge_incidence.get(key, 0) + 1
    degrees: dict[int, int] = {}
    for edge, count in edge_incidence.items():
        if count != 1:
            continue
        for vertex in edge:
            degrees[vertex] = degrees.get(vertex, 0) + 1
    return degrees


def boundary_loops(faces: np.ndarray) -> list[list[int]]:
    incidence: dict[tuple[int, int], int] = {}
    for a, b, c in faces.tolist():
        for edge in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(edge))
            incidence[key] = incidence.get(key, 0) + 1
    adjacency: dict[int, list[int]] = {}
    for (left, right), count in incidence.items():
        if count != 1:
            continue
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)
    if not adjacency or any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise RuntimeError("Boundary is not a collection of simple loops")
    loops: list[list[int]] = []
    unseen = set(adjacency)
    while unseen:
        start = min(unseen)
        loop = [start]
        previous = -1
        current = start
        while True:
            candidates = [value for value in adjacency[current] if value != previous]
            following = candidates[0]
            if following == start:
                break
            if following in loop:
                raise RuntimeError("Boundary loop self-intersects topologically")
            loop.append(following)
            previous, current = current, following
        unseen.difference_update(loop)
        loops.append(loop)
    return loops


def close_small_inner_boundary_loops(
    vertices: np.ndarray,
    faces: np.ndarray,
    maximum_vertices: int,
) -> tuple[np.ndarray, list[int], int]:
    """Fan-fill only tiny artificial holes while preserving the outer loop."""
    loops = boundary_loops(faces)
    loop_sizes = sorted((len(loop) for loop in loops), reverse=True)
    if len(loops) <= 1:
        return faces, loop_sizes, 0
    outer_id = int(np.argmax([len(loop) for loop in loops]))
    additions: list[tuple[int, int, int]] = []
    filled = 0
    for loop_id, loop in enumerate(loops):
        if loop_id == outer_id:
            continue
        if len(loop) > maximum_vertices:
            raise RuntimeError(
                "Adaptive decimation created a nontrivial inner boundary "
                f"with {len(loop)} vertices"
            )
        for index in range(1, len(loop) - 1):
            triangle = [loop[0], loop[index], loop[index + 1]]
            points = vertices[triangle]
            if np.cross(points[1] - points[0], points[2] - points[0])[2] < 0.0:
                triangle[1], triangle[2] = triangle[2], triangle[1]
            additions.append(tuple(triangle))
        filled += 1
    output = np.concatenate(
        (faces, np.asarray(additions, dtype=np.int32)), axis=0
    )
    return output, loop_sizes, filled


def main() -> None:
    args = parse_args()
    for name in ("input_surface", "input_report", "pose_driver", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if not 0.0 < args.fine_radius_mm < args.transition_radius_mm:
        raise ValueError("Adaptive radii must satisfy 0 < fine < transition")
    if args.fine_stride < 1 or min(args.transition_stride, args.outer_stride) < 2:
        raise ValueError("Adaptive strides are invalid")
    if (
        args.maximum_edge_mm <= 0.0
        or args.minimum_faces < 4
        or args.maximum_hole_boundary_vertices < 3
    ):
        raise ValueError("Maximum edge and minimum face budget are invalid")

    output_surface = args.output_dir / "rest_surface.npz"
    output_report = args.output_dir / "report.json"
    if (output_surface.exists() or output_report.exists()) and not args.overwrite:
        raise FileExistsError("Adaptive surface exists; pass --overwrite")
    with np.load(args.input_surface, allow_pickle=False) as loaded:
        source = {name: loaded[name].copy() for name in loaded.files}
    source_report = json.loads(args.input_report.read_text(encoding="utf-8"))

    occupied = source["occupied_grid"].astype(bool)
    vertex_ids_grid = source["surface_vertex_ids_grid"].astype(np.int32)
    source_vertices = source["surface_vertices_table"].astype(np.float32)
    source_classes = source["surface_vertex_source_class"].astype(np.uint8)
    x_centers = source["x_centers"].astype(np.float64)
    y_centers = source["y_centers"].astype(np.float64)
    spacing = float(source["surface_spacing_m"])
    grasp_center, grasp_audit = derive_grasp_center(
        args.pose_driver,
        args.near_surface_jaw_height_mm / 1000.0,
        args.closed_jaw_angle_max_rad,
    )

    ii, jj = np.indices(occupied.shape)
    xx = x_centers[ii]
    yy = y_centers[jj]
    radius = np.sqrt((xx - grasp_center[0]) ** 2 + (yy - grasp_center[1]) ** 2)
    fine = occupied & (radius <= args.fine_radius_mm / 1000.0)
    transition = occupied & (radius > args.fine_radius_mm / 1000.0) & (
        radius <= args.transition_radius_mm / 1000.0
    )
    outer = occupied & (radius > args.transition_radius_mm / 1000.0)
    selected = fine & (ii % args.fine_stride == 0) & (
        jj % args.fine_stride == 0
    )
    selected |= transition & (ii % args.transition_stride == 0) & (
        jj % args.transition_stride == 0
    )
    selected |= outer & (ii % args.outer_stride == 0) & (
        jj % args.outer_stride == 0
    )
    dense_boundary = boundary_mask(occupied)
    if args.preserve_every_boundary:
        selected |= dense_boundary
    else:
        selected |= dense_boundary & (
            (ii + jj) % args.outer_stride == 0
        )

    valid_grid_ids = vertex_ids_grid >= 0
    grid_classes = np.zeros_like(vertex_ids_grid, dtype=np.uint8)
    grid_classes[valid_grid_ids] = source_classes[vertex_ids_grid[valid_grid_ids]]
    forced_classes_list: list[int] = []
    if args.preserve_view_exclusive:
        forced_classes_list.extend((1, 2))
    if args.preserve_inferred_completion:
        forced_classes_list.append(4)
    forced_classes = tuple(forced_classes_list)
    forced_source = occupied & np.isin(grid_classes, forced_classes)
    source_faces = source["surface_faces"].astype(np.int32)
    forced_source_ids = np.flatnonzero(
        np.isin(source_classes, forced_classes)
    )
    forced_source_id_mask = np.zeros(len(source_vertices), dtype=bool)
    forced_source_id_mask[forced_source_ids] = True
    forced_dense_faces = source_faces[
        np.any(forced_source_id_mask[source_faces], axis=1)
    ]
    # Preserve one dense one-ring around mandatory points so they cannot become
    # isolated after the surrounding structured decimation.
    forced_expanded_ids = np.unique(forced_dense_faces)
    forced_expanded_grid = occupied & np.isin(
        vertex_ids_grid, forced_expanded_ids
    )
    selected |= forced_source
    selected |= forced_expanded_grid
    selected &= occupied & valid_grid_ids
    selected_source_ids = vertex_ids_grid[selected]
    selected_vertices = source_vertices[selected_source_ids]
    selected_classes = source_classes[selected_source_ids]

    triangulation = Delaunay(selected_vertices[:, :2])
    candidate_faces = triangulation.simplices.astype(np.int32)
    candidate_points = selected_vertices[candidate_faces]
    centroid = candidate_points[:, :, :2].mean(axis=1)
    midpoints = np.stack(
        (
            0.5 * (candidate_points[:, 0, :2] + candidate_points[:, 1, :2]),
            0.5 * (candidate_points[:, 1, :2] + candidate_points[:, 2, :2]),
            0.5 * (candidate_points[:, 2, :2] + candidate_points[:, 0, :2]),
        ),
        axis=1,
    )
    edge_lengths = np.linalg.norm(
        candidate_points[:, [1, 2, 0], :2]
        - candidate_points[:, [0, 1, 2], :2],
        axis=2,
    )
    keep = edge_lengths.max(axis=1) <= args.maximum_edge_mm / 1000.0
    keep &= sample_occupied(
        centroid, x_centers[0], y_centers[0], spacing, occupied
    )
    for corner in range(3):
        keep &= sample_occupied(
            midpoints[:, corner], x_centers[0], y_centers[0], spacing, occupied
        )
    selected_id_to_local = np.full(len(source_vertices), -1, dtype=np.int32)
    selected_id_to_local[selected_source_ids] = np.arange(
        len(selected_source_ids), dtype=np.int32
    )
    forced_selected_local = selected_id_to_local[forced_source_ids]
    forced_selected_local = forced_selected_local[forced_selected_local >= 0]
    mandatory_candidate = np.any(
        np.isin(candidate_faces, forced_selected_local), axis=1
    )
    # Stay a strict subset of one Delaunay triangulation so no edge can gain
    # more than two incident faces.  Around mandatory source points only the
    # occupancy sampling is relaxed; the maximum-edge safety bound remains.
    mandatory_local = mandatory_candidate & (
        edge_lengths.max(axis=1) <= args.maximum_edge_mm / 1000.0
    )
    faces = candidate_faces[keep | mandatory_local]
    if not len(faces):
        raise RuntimeError("Adaptive triangulation rejected every face")
    components = connected_face_components(len(faces), faces)
    component_ids, component_counts = np.unique(components, return_counts=True)
    largest_component = component_ids[np.argmax(component_counts)]
    faces = faces[components == largest_component]

    used_vertices = np.unique(faces)
    remap = np.full(len(selected_vertices), -1, dtype=np.int32)
    remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int32)
    vertices = selected_vertices[used_vertices]
    classes = selected_classes[used_vertices]
    source_vertex_ids = selected_source_ids[used_vertices]
    faces = remap[faces]
    vertices, faces, classes, source_vertex_ids, boundary_split_count = (
        split_boundary_pinches(
            vertices, faces, classes, source_vertex_ids
        )
    )
    faces, prefill_boundary_loop_sizes, filled_small_holes = (
        close_small_inner_boundary_loops(
            vertices,
            faces,
            args.maximum_hole_boundary_vertices,
        )
    )

    face_points = vertices[faces]
    face_centers = face_points.mean(axis=1)
    face_radius = np.linalg.norm(face_centers[:, :2] - grasp_center[None], axis=1)
    face_zones = np.full(len(faces), 2, dtype=np.uint8)
    face_zones[face_radius <= args.transition_radius_mm / 1000.0] = 1
    face_zones[face_radius <= args.fine_radius_mm / 1000.0] = 0
    face_area = 0.5 * np.linalg.norm(
        np.cross(face_points[:, 1] - face_points[:, 0], face_points[:, 2] - face_points[:, 0]),
        axis=1,
    )

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices.astype(np.float64)),
        o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    closest = scene.compute_closest_points(
        o3d.core.Tensor(source_vertices.astype(np.float32))
    )["points"].numpy()
    source_to_adaptive_distance = np.linalg.norm(source_vertices - closest, axis=1)
    nearest_selected_distance = cKDTree(vertices[:, :2]).query(
        source_vertices[:, :2], k=1, workers=-1
    )[0]

    output = {
        "surface_vertices_table": vertices.astype(np.float32),
        "surface_faces": faces.astype(np.int32),
        "surface_vertex_source_class": classes,
        "surface_vertex_dense_source_ids": source_vertex_ids.astype(np.int32),
        "surface_face_density_zone": face_zones,
        "grasp_roi_center_xy_table": grasp_center.astype(np.float32),
        "source_surface_spacing_m": np.asarray(spacing, dtype=np.float32),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_surface, **output)

    retained_source_mask = np.zeros(len(source_vertices), dtype=bool)
    retained_source_mask[source_vertex_ids] = True
    relevant_source_gates = {
        name: passed
        for name, passed in source_report["gates"].items()
        if name not in {"surface_triangle_budget", "surface_vertex_budget"}
    }
    expected_fine = fine & (ii % args.fine_stride == 0) & (
        jj % args.fine_stride == 0
    )
    gates = {
        "source_surface_visual_gates_passed": bool(
            all(relevant_source_gates.values())
        ),
        "single_connected_adaptive_surface": len(np.unique(connected_face_components(len(faces), faces))) == 1,
        "fine_roi_uses_configured_stride": bool(
            np.all(retained_source_mask[vertex_ids_grid[expected_fine]])
        ),
        "left_only_vertex_policy_satisfied": bool(
            not args.preserve_view_exclusive
            or np.all(retained_source_mask[source_classes == 1])
        ),
        "right_only_vertex_policy_satisfied": bool(
            not args.preserve_view_exclusive
            or np.all(retained_source_mask[source_classes == 2])
        ),
        "inferred_completion_vertex_policy_satisfied": bool(
            not args.preserve_inferred_completion
            or np.all(retained_source_mask[source_classes == 4])
        ),
        "adaptive_top_surface_meets_configured_face_budget": bool(
            len(faces) >= args.minimum_faces
        ),
        "adaptive_surface_is_smaller_than_dense_surface": len(faces) < 0.75 * len(source["surface_faces"]),
        "dense_surface_coverage_p95_below_0p5mm": bool(
            np.percentile(source_to_adaptive_distance, 95) <= 0.0005
        ),
        "finite_geometry": bool(
            np.isfinite(vertices).all()
            and np.isfinite(face_area).all()
            and np.all(face_area > 0.0)
        ),
        "boundary_is_manifold_after_topological_pinch_splits": bool(
            boundary_degrees(faces)
            and all(degree == 2 for degree in boundary_degrees(faces).values())
        ),
        "single_outer_boundary_after_small_hole_fill": bool(
            len(boundary_loops(faces)) == 1
        ),
    }
    report = {
        "stage": "grasp_centered_adaptive_visual_surface",
        "method": {
            "fine_zone": f"structured stride {args.fine_stride}",
            "transition_zone": f"structured stride {args.transition_stride}",
            "outer_zone": f"structured stride {args.outer_stride}",
            "forced_vertices": (
                "configured boundary plus enabled source-class policies and "
                "their mandatory one-ring vertices"
            ),
            "triangulation": "2D Delaunay in table x-y; reject triangles crossing unoccupied grid samples or maximum edge",
            "mechanics_changed": False,
        },
        "parameters": {
            "fine_radius_mm": args.fine_radius_mm,
            "transition_radius_mm": args.transition_radius_mm,
            "fine_stride": args.fine_stride,
            "transition_stride": args.transition_stride,
            "outer_stride": args.outer_stride,
            "maximum_edge_mm": args.maximum_edge_mm,
            "minimum_faces": args.minimum_faces,
            "maximum_hole_boundary_vertices": (
                args.maximum_hole_boundary_vertices
            ),
            "preserve_view_exclusive": args.preserve_view_exclusive,
            "preserve_every_boundary": args.preserve_every_boundary,
            "preserve_inferred_completion": (
                args.preserve_inferred_completion
            ),
        },
        "grasp_roi": grasp_audit,
        "counts": {
            "dense_vertices": int(len(source_vertices)),
            "dense_faces": int(len(source["surface_faces"])),
            "adaptive_vertices": int(len(vertices)),
            "adaptive_faces": int(len(faces)),
            "fine_faces": int(np.count_nonzero(face_zones == 0)),
            "transition_faces": int(np.count_nonzero(face_zones == 1)),
            "outer_faces": int(np.count_nonzero(face_zones == 2)),
            "boundary_pinch_vertex_duplicates": int(boundary_split_count),
            "prefill_boundary_loop_sizes": prefill_boundary_loop_sizes,
            "filled_small_inner_boundary_loops": int(filled_small_holes),
            "left_only_vertices": int(np.count_nonzero(classes == 1)),
            "right_only_vertices": int(np.count_nonzero(classes == 2)),
            "dual_vertices": int(np.count_nonzero(classes == 3)),
            "inferred_vertices": int(np.count_nonzero(classes == 4)),
        },
        "statistics": {
            "face_area_mm2_percentiles": np.percentile(
                face_area * 1.0e6, [0, 5, 50, 95, 100]
            ).tolist(),
            "dense_vertex_to_adaptive_surface_mm_percentiles": np.percentile(
                source_to_adaptive_distance * 1000.0, [0, 5, 50, 95, 100]
            ).tolist(),
            "dense_vertex_to_adaptive_vertex_xy_mm_percentiles": np.percentile(
                nearest_selected_distance * 1000.0, [0, 5, 50, 95, 100]
            ).tolist(),
        },
        "inputs": {
            "dense_surface": str(args.input_surface.relative_to(REPO_ROOT)),
            "dense_surface_sha256": sha256(args.input_surface),
            "pose_driver": str(args.pose_driver.relative_to(REPO_ROOT)),
            "pose_driver_sha256": sha256(args.pose_driver),
        },
        "outputs": {"surface": str(output_surface.relative_to(REPO_ROOT))},
        "gates": gates,
        "passed": bool(all(gates.values())),
    }
    output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        failed = [name for name, passed in gates.items() if not passed]
        raise SystemExit(f"Adaptive visual surface gates failed: {failed}")


if __name__ == "__main__":
    main()
