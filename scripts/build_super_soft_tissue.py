#!/usr/bin/env python3
"""Build and validate the first SuPer volumetric soft-tissue asset.

This script deliberately stops at the asset boundary.  It does not register the
particles with Warp, change the runtime tissue body, or enable visual forces.
The output is a table-frame tetrahedral mesh plus a rest-pose binding from the
existing tissue Gaussians to that mesh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_DIR = REPO_ROOT / "data" / "super" / "grasp5_native"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the stage-A tetrahedral soft-tissue asset for SuPer grasp5."
    )
    parser.add_argument(
        "--depth", type=Path, default=NATIVE_DIR / "depth_v2/000000-depth.npy"
    )
    parser.add_argument(
        "--tissue-mask", type=Path, default=NATIVE_DIR / "masks/000000-tissue.png"
    )
    parser.add_argument(
        "--calib", type=Path, default=NATIVE_DIR / "calib_rectified.json"
    )
    parser.add_argument(
        "--table-frame", type=Path, default=REPO_ROOT / "data/super/table_frame.json"
    )
    parser.add_argument(
        "--ground-plane",
        type=Path,
        default=NATIVE_DIR / "bodies_v5_table/ground_plane.json",
    )
    parser.add_argument(
        "--source-tissue",
        type=Path,
        default=NATIVE_DIR / "bodies_v5_table/tissue.json",
    )
    parser.add_argument(
        "--source-metadata",
        type=Path,
        default=NATIVE_DIR / "bodies_v5_table/build_metadata.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=NATIVE_DIR / "soft_tissue_v1"
    )
    parser.add_argument("--spacing", type=float, default=0.0015)
    parser.add_argument("--contact-radius", type=float, default=0.0007)
    parser.add_argument("--density", type=float, default=1000.0)
    parser.add_argument("--minimum-height", type=float, default=0.003)
    parser.add_argument("--maximum-height", type=float, default=0.015)
    parser.add_argument("--height-smoothing-sigma", type=float, default=1.0)
    parser.add_argument("--surface-voxel-size", type=float, default=0.0015)
    parser.add_argument("--binding-candidates", type=int, default=64)
    parser.add_argument(
        "--anchor-mode",
        choices=("none", "bottom_perimeter"),
        default="none",
        help="Stage A defaults to no hard anchors; a bottom-perimeter candidate mask is always saved.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def backproject_mask(depth: np.ndarray, mask: np.ndarray, k: np.ndarray) -> np.ndarray:
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    rows, columns = np.nonzero(valid)
    z = depth[rows, columns]
    x = (columns.astype(np.float64) - k[0, 2]) * z / k[0, 0]
    y = (rows.astype(np.float64) - k[1, 2]) * z / k[1, 1]
    return np.stack((x, y, z), axis=1)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    downsampled = cloud.voxel_down_sample(voxel_size=float(voxel_size))
    return np.asarray(downsampled.points, dtype=np.float64)


def project_table_points(
    points_table: np.ndarray,
    x_camera_table: np.ndarray,
    k: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    points_camera = transform_points(x_camera_table, points_table)
    z = points_camera[:, 2]
    valid = z > 1.0e-8
    pixels = np.zeros((len(points_table), 2), dtype=np.int64)
    pixels[valid, 0] = np.rint(
        points_camera[valid, 0] * k[0, 0] / z[valid] + k[0, 2]
    ).astype(np.int64)
    pixels[valid, 1] = np.rint(
        points_camera[valid, 1] * k[1, 1] / z[valid] + k[1, 2]
    ).astype(np.int64)
    valid &= (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    return pixels, valid


def build_height_field(
    surface_table: np.ndarray,
    tissue_mask: np.ndarray,
    x_camera_table: np.ndarray,
    k: np.ndarray,
    origin: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    plane: np.ndarray,
    spacing: float,
    minimum_height: float,
    maximum_height: float,
    smoothing_sigma: float,
) -> dict[str, np.ndarray]:
    normal = plane[:3]
    heights = surface_table @ normal + plane[3]
    valid_surface = np.isfinite(heights) & (heights > -spacing)
    surface_table = surface_table[valid_surface]
    raw_heights = heights[valid_surface]
    clipped_heights = np.clip(raw_heights, minimum_height, maximum_height)
    centered = surface_table - origin
    surface_uv = np.stack((centered @ axis_x, centered @ axis_y), axis=1)

    min_uv = np.percentile(surface_uv, 1.0, axis=0)
    max_uv = np.percentile(surface_uv, 99.0, axis=0)
    x0 = np.floor(min_uv[0] / spacing) * spacing
    x1 = np.ceil(max_uv[0] / spacing) * spacing
    y0 = np.floor(min_uv[1] / spacing) * spacing
    y1 = np.ceil(max_uv[1] / spacing) * spacing
    x_edges = np.arange(x0, x1 + spacing * 0.5, spacing, dtype=np.float64)
    y_edges = np.arange(y0, y1 + spacing * 0.5, spacing, dtype=np.float64)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    centers_uv = np.stack(
        np.meshgrid(x_centers, y_centers, indexing="ij"), axis=-1
    ).reshape(-1, 2)

    tree = cKDTree(surface_uv)
    neighbor_count = min(8, len(surface_uv))
    distances, indices = tree.query(centers_uv, k=neighbor_count, workers=-1)
    if neighbor_count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    weights = 1.0 / np.maximum(distances, spacing * 0.25)
    cell_heights = np.sum(weights * clipped_heights[indices], axis=1) / np.sum(
        weights, axis=1
    )

    base_points = (
        origin[None]
        + centers_uv[:, :1] * axis_x[None]
        + centers_uv[:, 1:] * axis_y[None]
    )
    top_points = base_points.copy()
    top_points += cell_heights[:, None] * normal[None]
    image_height, image_width = tissue_mask.shape
    pixels, valid_projection = project_table_points(
        top_points, x_camera_table, k, image_width, image_height
    )
    in_mask = np.zeros(len(top_points), dtype=bool)
    ids = np.flatnonzero(valid_projection)
    in_mask[ids] = tissue_mask[pixels[ids, 1], pixels[ids, 0]]
    occupied = in_mask & (distances[:, 0] <= spacing * 2.5)
    if not np.any(occupied):
        raise RuntimeError("The tissue height field contains no occupied cells.")

    grid_shape = (len(x_centers), len(y_centers))
    height_grid = cell_heights.reshape(grid_shape)
    occupied_grid = occupied.reshape(grid_shape)
    if smoothing_sigma > 0.0:
        occupancy_float = occupied_grid.astype(np.float64)
        smooth_weight = gaussian_filter(
            occupancy_float, smoothing_sigma, mode="nearest"
        )
        smooth_height = gaussian_filter(
            height_grid * occupancy_float, smoothing_sigma, mode="nearest"
        ) / np.maximum(smooth_weight, 1.0e-12)
        height_grid[occupied_grid] = smooth_height[occupied_grid]
    height_grid = np.clip(height_grid, minimum_height, maximum_height)
    return {
        "x_edges": x_edges,
        "y_edges": y_edges,
        "height_grid": height_grid,
        "occupied_grid": occupied_grid,
        "surface_height_raw": raw_heights,
    }


def hexahedron_tets(vertices: tuple[int, ...], parity: int) -> list[tuple[int, ...]]:
    v0, v1, v2, v3, v4, v5, v6, v7 = vertices
    if parity:
        return [
            (v0, v1, v4, v3),
            (v2, v3, v6, v1),
            (v5, v4, v1, v6),
            (v7, v6, v3, v4),
            (v4, v1, v6, v3),
        ]
    return [
        (v1, v2, v5, v0),
        (v3, v0, v7, v2),
        (v4, v7, v0, v5),
        (v6, v5, v2, v7),
        (v5, v2, v7, v0),
    ]


def resolve_diagonal_voxel_contacts(
    vertical_cells: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Remove edge-only voxel contacts that create non-manifold surface edges."""
    vertical_cells = vertical_cells.copy()
    changes = 0
    for _ in range(vertical_cells.size * 2):
        changed = False
        for ix in range(vertical_cells.shape[0] - 1):
            for iy in range(vertical_cells.shape[1] - 1):
                block = vertical_cells[ix : ix + 2, iy : iy + 2]
                for level in range(int(block.max())):
                    active = block > level
                    diagonal = (
                        active[0, 0]
                        and active[1, 1]
                        and not active[0, 1]
                        and not active[1, 0]
                    ) or (
                        active[0, 1]
                        and active[1, 0]
                        and not active[0, 0]
                        and not active[1, 1]
                    )
                    if not diagonal:
                        continue

                    options: list[tuple[int, int, int, int]] = []
                    for local_x in range(2):
                        for local_y in range(2):
                            value = int(block[local_x, local_y])
                            if active[local_x, local_y]:
                                # Lower one column so that it is inactive at this level.
                                options.append((value - level, 1, local_x, local_y))
                            else:
                                # Prefer filling on ties to preserve the observed footprint.
                                options.append((level + 1 - value, 0, local_x, local_y))
                    _, _, local_x, local_y = min(options)
                    global_x = ix + local_x
                    global_y = iy + local_y
                    if active[local_x, local_y]:
                        vertical_cells[global_x, global_y] = level
                    else:
                        vertical_cells[global_x, global_y] = level + 1
                    changes += 1
                    changed = True
                    break
                if changed:
                    break
            if changed:
                break
        if not changed:
            return vertical_cells, changes
    raise RuntimeError("Diagonal voxel-contact cleanup did not converge")


def build_tetrahedral_mesh(
    height_field: dict[str, np.ndarray],
    origin: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    plane_normal: np.ndarray,
    spacing: float,
    contact_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    x_edges = height_field["x_edges"]
    y_edges = height_field["y_edges"]
    height_grid = height_field["height_grid"]
    occupied_grid = height_field["occupied_grid"]

    vertex_ids: dict[tuple[int, int, int], int] = {}
    positions: list[np.ndarray] = []
    tets: list[tuple[int, int, int, int]] = []
    represented_heights: list[float] = []
    meshed_observed_heights: list[float] = []

    vertical_cells_grid = np.zeros_like(height_grid, dtype=np.int32)
    vertical_cells_grid[occupied_grid] = np.maximum(
        1,
        np.rint(
            (height_grid[occupied_grid] - contact_radius) / spacing
        ).astype(np.int32),
    )
    vertical_cells_grid, diagonal_cleanup_changes = resolve_diagonal_voxel_contacts(
        vertical_cells_grid
    )

    def vertex(ix: int, iy: int, iz: int) -> int:
        key = (ix, iy, iz)
        if key not in vertex_ids:
            point = (
                origin
                + x_edges[ix] * axis_x
                + y_edges[iy] * axis_y
                + plane_normal * (contact_radius + iz * spacing)
            )
            vertex_ids[key] = len(positions)
            positions.append(point)
        return vertex_ids[key]

    for ix, iy in np.argwhere(vertical_cells_grid > 0):
        observed_height = float(height_grid[ix, iy])
        vertical_cells = int(vertical_cells_grid[ix, iy])
        meshed_observed_heights.append(observed_height)
        represented_heights.append(
            contact_radius + vertical_cells * spacing
        )
        for iz in range(vertical_cells):
            hexahedron = (
                vertex(ix, iy, iz),
                vertex(ix + 1, iy, iz),
                vertex(ix + 1, iy, iz + 1),
                vertex(ix, iy, iz + 1),
                vertex(ix, iy + 1, iz),
                vertex(ix + 1, iy + 1, iz),
                vertex(ix + 1, iy + 1, iz + 1),
                vertex(ix, iy + 1, iz + 1),
            )
            tets.extend(hexahedron_tets(hexahedron, (ix ^ iy ^ iz) & 1))

    positions_array = np.asarray(positions, dtype=np.float64)
    tets_array = np.asarray(tets, dtype=np.int32)
    a = positions_array[tets_array[:, 0]]
    b = positions_array[tets_array[:, 1]]
    c = positions_array[tets_array[:, 2]]
    d = positions_array[tets_array[:, 3]]
    signed_volume = np.linalg.det(
        np.stack((b - a, c - a, d - a), axis=-1)
    ) / 6.0
    inverted = signed_volume < 0.0
    if np.any(inverted):
        corrected = tets_array[inverted].copy()
        corrected[:, [1, 2]] = corrected[:, [2, 1]]
        tets_array[inverted] = corrected
        signed_volume[inverted] *= -1.0
    return (
        positions_array,
        tets_array,
        np.asarray(represented_heights),
        np.asarray(meshed_observed_heights),
        diagonal_cleanup_changes,
    )


def boundary_faces(
    tets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    face_map: dict[tuple[int, int, int], tuple[int, tuple[int, int, int]]] = {}
    for a, b, c, d in tets.tolist():
        oriented = ((a, c, b), (a, b, d), (a, d, c), (b, c, d))
        for face in oriented:
            key = tuple(sorted(face))
            count, original = face_map.get(key, (0, face))
            face_map[key] = (count + 1, original)
    surface = [face for count, face in face_map.values() if count == 1]
    nonmanifold = sum(count > 2 for count, _ in face_map.values())
    counts = np.asarray([count for count, _ in face_map.values()], dtype=np.int32)
    return np.asarray(surface, dtype=np.int32), counts, nonmanifold


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
    used = np.unique(tets)
    return len({find(int(vertex)) for vertex in used})


def transform_gaussians(source_tissue: dict) -> dict[str, np.ndarray]:
    x_world_body = np.asarray(source_tissue["X_WB"], dtype=np.float64)
    gaussians = source_tissue["gaussians"]
    means_local = np.asarray(gaussians["means"], dtype=np.float64)
    quats_local_wxyz = np.asarray(gaussians["quats"], dtype=np.float64)
    means_table = transform_points(x_world_body, means_local)

    body_rotation = Rotation.from_matrix(x_world_body[:3, :3])
    local_rotation = Rotation.from_quat(quats_local_wxyz[:, [1, 2, 3, 0]])
    world_xyzw = (body_rotation * local_rotation).as_quat()
    quats_table_wxyz = world_xyzw[:, [3, 0, 1, 2]]
    return {
        "means": means_table,
        "quats": quats_table_wxyz,
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

    candidate_origins = origins[candidates]
    relative = gaussian_means[:, None, :] - candidate_origins
    coordinates = np.einsum(
        "nkij,nkj->nki", inverse_rest[candidates], relative
    )
    weights = np.concatenate(
        (1.0 - coordinates.sum(axis=-1, keepdims=True), coordinates), axis=-1
    )
    contained = np.all(weights >= -1.0e-7, axis=-1) & np.all(
        weights <= 1.0 + 1.0e-7, axis=-1
    )

    projected_weights = np.clip(weights, 0.0, 1.0)
    projected_weights /= np.maximum(
        projected_weights.sum(axis=-1, keepdims=True), 1.0e-12
    )
    reconstructed = np.einsum(
        "nki,nkij->nkj", projected_weights, tet_positions[candidates]
    )
    distances = np.linalg.norm(reconstructed - gaussian_means[:, None, :], axis=-1)
    distances[contained] = 0.0
    selected_column = np.argmin(distances, axis=1)
    row = np.arange(len(gaussian_means))
    selected_tets = candidates[row, selected_column]
    selected_weights = projected_weights[row, selected_column]
    selected_distance = distances[row, selected_column]
    selected_contained = contained[row, selected_column]
    return selected_tets, selected_weights, selected_distance, selected_contained


def write_previews(
    output_dir: Path,
    positions: np.ndarray,
    surface_faces_array: np.ndarray,
    gaussian_means: np.ndarray,
    binding_distance: np.ndarray,
    contact_radius: float,
    spacing: float,
) -> None:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(positions)
    mesh.triangles = o3d.utility.Vector3iVector(surface_faces_array)
    mesh.paint_uniform_color((0.82, 0.35, 0.28))
    mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(
        str(output_dir / "tissue_soft_surface.ply"), mesh, write_ascii=False
    ):
        raise RuntimeError("Failed to write tissue_soft_surface.ply")

    colors = np.zeros((len(gaussian_means), 3), dtype=np.float64)
    colors[binding_distance <= contact_radius] = (0.1, 0.8, 0.2)
    colors[
        (binding_distance > contact_radius) & (binding_distance <= spacing)
    ] = (0.95, 0.75, 0.1)
    colors[binding_distance > spacing] = (0.9, 0.1, 0.1)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(gaussian_means)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if not o3d.io.write_point_cloud(
        str(output_dir / "tissue_soft_gaussian_binding.ply"), cloud, write_ascii=False
    ):
        raise RuntimeError("Failed to write tissue_soft_gaussian_binding.ply")


def percentile(values: np.ndarray, q: list[float]) -> list[float]:
    return [float(value) for value in np.percentile(values, q)]


def main() -> None:
    args = parse_args()
    if args.spacing <= 0.0 or args.contact_radius <= 0.0 or args.density <= 0.0:
        raise ValueError("spacing, contact-radius and density must be positive")
    if 2.0 * args.contact_radius >= args.minimum_height:
        raise ValueError("2*contact-radius must be smaller than minimum-height")

    source_tissue = read_json(args.source_tissue)
    source_metadata = read_json(args.source_metadata)
    calibration = read_json(args.calib)
    table_frame = read_json(args.table_frame)
    ground_plane_data = read_json(args.ground_plane)
    depth = np.load(args.depth).astype(np.float64)
    mask_image = cv2.imread(str(args.tissue_mask), cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        raise FileNotFoundError(args.tissue_mask)
    tissue_mask = mask_image > 0
    if depth.shape != tissue_mask.shape:
        raise ValueError(f"Depth/mask shape mismatch: {depth.shape} != {tissue_mask.shape}")

    k = np.asarray(calibration["K_left_rect"], dtype=np.float64)
    x_table_camera = np.asarray(table_frame["X_table_camera"], dtype=np.float64)
    x_camera_table = np.asarray(table_frame["X_camera_table"], dtype=np.float64)
    surface_camera = backproject_mask(depth, tissue_mask, k)
    surface_camera = voxel_downsample(surface_camera, args.surface_voxel_size)
    surface_table = transform_points(x_table_camera, surface_camera)

    x_world_body = np.asarray(source_tissue["X_WB"], dtype=np.float64)
    plane = np.asarray(ground_plane_data["plane"], dtype=np.float64)
    plane /= np.linalg.norm(plane[:3])
    if np.median(surface_table @ plane[:3] + plane[3]) < 0.0:
        plane = -plane
    normal = plane[:3]
    origin_seed = x_world_body[:3, 3]
    origin = origin_seed - normal * (origin_seed @ normal + plane[3])
    axis_x = x_world_body[:3, 0].copy()
    axis_x -= normal * np.dot(axis_x, normal)
    axis_x /= np.linalg.norm(axis_x)
    axis_y = np.cross(normal, axis_x)
    axis_y /= np.linalg.norm(axis_y)
    if np.dot(axis_y, x_world_body[:3, 1]) < 0.0:
        axis_y *= -1.0
        axis_x *= -1.0

    height_field = build_height_field(
        surface_table=surface_table,
        tissue_mask=tissue_mask,
        x_camera_table=x_camera_table,
        k=k,
        origin=origin,
        axis_x=axis_x,
        axis_y=axis_y,
        plane=plane,
        spacing=args.spacing,
        minimum_height=args.minimum_height,
        maximum_height=args.maximum_height,
        smoothing_sigma=args.height_smoothing_sigma,
    )
    (
        positions,
        tets,
        represented_heights,
        meshed_observed_heights,
        diagonal_cleanup_changes,
    ) = build_tetrahedral_mesh(
        height_field,
        origin,
        axis_x,
        axis_y,
        normal,
        args.spacing,
        args.contact_radius,
    )
    surface_faces_array, face_counts, nonmanifold_faces = boundary_faces(tets)

    tet_positions = positions[tets]
    signed_volumes = np.linalg.det(
        np.stack(
            (
                tet_positions[:, 1] - tet_positions[:, 0],
                tet_positions[:, 2] - tet_positions[:, 0],
                tet_positions[:, 3] - tet_positions[:, 0],
            ),
            axis=-1,
        )
    ) / 6.0
    volume_epsilon = args.spacing**3 * 1.0e-8
    inverted_count = int(np.count_nonzero(signed_volumes < 0.0))
    degenerate_count = int(np.count_nonzero(signed_volumes <= volume_epsilon))
    if inverted_count or degenerate_count:
        raise RuntimeError(
            f"Invalid tetrahedra: inverted={inverted_count}, degenerate={degenerate_count}"
        )
    if nonmanifold_faces:
        raise RuntimeError(f"Found {nonmanifold_faces} non-manifold faces")

    masses = np.zeros(len(positions), dtype=np.float64)
    per_corner_mass = args.density * signed_volumes / 4.0
    for corner in range(4):
        np.add.at(masses, tets[:, corner], per_corner_mass)

    signed_particle_height = positions @ normal + plane[3]
    bottom_vertices = np.isclose(
        signed_particle_height,
        args.contact_radius,
        atol=args.spacing * 1.0e-6,
    )
    face_height_range = np.ptp(
        signed_particle_height[surface_faces_array], axis=1
    )
    side_faces = surface_faces_array[face_height_range > args.spacing * 1.0e-6]
    side_vertices = np.zeros(len(positions), dtype=bool)
    side_vertices[np.unique(side_faces)] = True
    support_candidate_mask = side_vertices & bottom_vertices
    fixed_mask = np.zeros(len(positions), dtype=bool)
    if args.anchor_mode == "bottom_perimeter":
        fixed_mask = support_candidate_mask.copy()

    gaussians = transform_gaussians(source_tissue)
    bind_tet_ids, bind_weights, bind_distance, bind_contained = bind_gaussians(
        gaussians["means"], positions, tets, args.binding_candidates
    )
    bind_particle_indices = tets[bind_tet_ids]
    reconstructed_gaussians = np.einsum(
        "ni,nij->nj", bind_weights, positions[bind_particle_indices]
    )
    gaussian_rest_offsets = gaussians["means"] - reconstructed_gaussians
    reconstruction_error = np.linalg.norm(gaussian_rest_offsets, axis=1)
    if not np.allclose(reconstruction_error, bind_distance, atol=1.0e-9):
        raise RuntimeError("Gaussian binding distance validation failed")

    component_count = connected_component_count(len(positions), tets)
    if component_count != 1:
        raise RuntimeError(f"Expected one connected tissue component, got {component_count}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "tissue_soft.npz",
        rest_positions_table=positions.astype(np.float32),
        tet_indices=tets.astype(np.int32),
        surface_faces=surface_faces_array.astype(np.int32),
        particle_mass=masses.astype(np.float32),
        particle_radius=np.full(len(positions), args.contact_radius, dtype=np.float32),
        fixed_mask=fixed_mask,
        support_candidate_mask=support_candidate_mask,
        rest_tet_volume=signed_volumes.astype(np.float32),
        gaussian_rest_means_table=gaussians["means"].astype(np.float32),
        gaussian_rest_quats_table_wxyz=gaussians["quats"].astype(np.float32),
        gaussian_scales=gaussians["scales"].astype(np.float32),
        gaussian_opacities=gaussians["opacities"].astype(np.float32),
        gaussian_colors_rgb=gaussians["colors"].astype(np.float32),
        gaussian_tet_ids=bind_tet_ids.astype(np.int32),
        gaussian_particle_indices=bind_particle_indices.astype(np.int32),
        gaussian_barycentric_weights=bind_weights.astype(np.float32),
        gaussian_binding_distance=bind_distance.astype(np.float32),
        gaussian_rest_offset_table=gaussian_rest_offsets.astype(np.float32),
    )
    write_previews(
        args.output_dir,
        positions,
        surface_faces_array,
        gaussians["means"],
        bind_distance,
        args.contact_radius,
        args.spacing,
    )

    observed_cell_heights = height_field["height_grid"][height_field["occupied_grid"]]
    represented_height_error = represented_heights - meshed_observed_heights
    old_radius = np.asarray(source_tissue["particles"]["radii"], dtype=np.float64)
    old_sphere_volume = float(np.sum(4.0 / 3.0 * np.pi * old_radius**3))
    total_volume = float(signed_volumes.sum())
    metadata = {
        "asset_version": 1,
        "stage": "A_soft_asset_only",
        "runtime_enabled": False,
        "frame": (
            "frozen right-handed table frame; tissue layers follow the fitted "
            "ground-plane normal without changing X_table_camera"
        ),
        "ground_plane_table": plane.tolist(),
        "parameters": {
            "spacing_m": float(args.spacing),
            "particle_contact_radius_m": float(args.contact_radius),
            "density_kg_m3": float(args.density),
            "minimum_height_m": float(args.minimum_height),
            "maximum_height_m": float(args.maximum_height),
            "height_smoothing_sigma_cells": float(args.height_smoothing_sigma),
            "anchor_mode": args.anchor_mode,
            "binding_candidates": int(args.binding_candidates),
        },
        "counts": {
            "surface_points_downsampled": int(len(surface_table)),
            "occupied_xy_cells": int(height_field["occupied_grid"].sum()),
            "meshed_xy_cells": int(len(meshed_observed_heights)),
            "particles": int(len(positions)),
            "tetrahedra": int(len(tets)),
            "surface_faces": int(len(surface_faces_array)),
            "gaussians": int(len(gaussians["means"])),
            "fixed_particles": int(fixed_mask.sum()),
            "support_candidate_particles": int(support_candidate_mask.sum()),
        },
        "topology": {
            "connected_components": int(component_count),
            "inverted_tetrahedra": inverted_count,
            "degenerate_tetrahedra": degenerate_count,
            "nonmanifold_faces": int(nonmanifold_faces),
            "face_incidence_min_max": [int(face_counts.min()), int(face_counts.max())],
            "diagonal_voxel_cleanup_changes": int(diagonal_cleanup_changes),
        },
        "geometry": {
            "bbox_min_m": positions.min(axis=0).tolist(),
            "bbox_max_m": positions.max(axis=0).tolist(),
            "bbox_extent_m": np.ptp(positions, axis=0).tolist(),
            "observed_height_mm_percentiles": percentile(
                observed_cell_heights * 1000.0, [0, 5, 50, 95, 100]
            ),
            "represented_surface_height_error_mm_percentiles": percentile(
                represented_height_error * 1000.0, [0, 5, 50, 95, 100]
            ),
            "tet_volume_mm3_percentiles": percentile(
                signed_volumes * 1.0e9, [0, 5, 50, 95, 100]
            ),
        },
        "mass": {
            "tet_mesh_volume_cm3": total_volume * 1.0e6,
            "tet_mesh_mass_g": float(masses.sum() * 1000.0),
            "legacy_rigid_sphere_volume_cm3": old_sphere_volume * 1.0e6,
            "legacy_mass_at_same_density_g": old_sphere_volume
            * args.density
            * 1000.0,
            "note": "The legacy 14.8 g is the sum of collision-sphere volumes, not a measured tissue mass.",
        },
        "gaussian_binding": {
            "directly_contained": int(bind_contained.sum()),
            "projected_to_tet": int((~bind_contained).sum()),
            "distance_mm_percentiles": percentile(
                bind_distance * 1000.0, [0, 50, 90, 95, 99, 100]
            ),
            "over_contact_radius": int(np.count_nonzero(bind_distance > args.contact_radius)),
            "over_grid_spacing": int(np.count_nonzero(bind_distance > args.spacing)),
            "weight_sum_max_error": float(
                np.max(np.abs(bind_weights.sum(axis=1) - 1.0))
            ),
            "weight_min_max": [float(bind_weights.min()), float(bind_weights.max())],
            "rest_offset_rule": "At runtime rotate gaussian_rest_offset_table by the rest-to-current tetra polar rotation, then add it to the barycentric particle position.",
        },
        "material_solver_gate": {
            "passed": False,
            "reason": "Stage A contains no runtime solve. The installed Warp 1.7 XPBD tetra kernel ignores tet k_mu/k_lambda/k_damp and must be validated or replaced before E/nu scans.",
        },
        "sources": {
            "depth": str(args.depth.resolve()),
            "depth_sha256": sha256(args.depth),
            "tissue_mask": str(args.tissue_mask.resolve()),
            "tissue_mask_sha256": sha256(args.tissue_mask),
            "calibration": str(args.calib.resolve()),
            "table_frame": str(args.table_frame.resolve()),
            "ground_plane": str(args.ground_plane.resolve()),
            "ground_plane_sha256": sha256(args.ground_plane),
            "source_tissue": str(args.source_tissue.resolve()),
            "source_tissue_sha256": sha256(args.source_tissue),
            "source_world_frame": source_metadata.get("world_frame"),
        },
        "outputs": {
            "asset": str((args.output_dir / "tissue_soft.npz").resolve()),
            "surface_preview": str(
                (args.output_dir / "tissue_soft_surface.ply").resolve()
            ),
            "binding_preview": str(
                (args.output_dir / "tissue_soft_gaussian_binding.ply").resolve()
            ),
        },
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
