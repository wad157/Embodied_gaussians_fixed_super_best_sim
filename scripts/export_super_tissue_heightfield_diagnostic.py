#!/usr/bin/env python3
"""Export a side-by-side PLY diagnosing SUPER tissue surface smoothing.

The output is an isolated diagnostic and never modifies runtime assets.  It
contains four panels at the same metric scale:

1. raw frame-0 FoundationStereo tissue surface with RGB;
2. the 2 mm height field before Gaussian smoothing (after the 3--15 mm clip);
3. the sigma=1-cell smoothed height field used by the original rigid fill; and
4. the current v9 visual Gaussians represented as colored 2-sigma octahedra.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOT = REPO_ROOT / "data/super/grasp5_native"
V9_ROOT = NATIVE_ROOT / "bodies_v9_dense_0p5mm_rigid_tissue"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export raw/smoothed/current SUPER tissue surface PLY."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=NATIVE_ROOT / "rgb/000000-left.png",
    )
    parser.add_argument(
        "--depth",
        type=Path,
        default=NATIVE_ROOT
        / "depth_v4_foundation_dense_timestamped/000000-depth.npy",
    )
    parser.add_argument(
        "--tissue-mask",
        type=Path,
        default=NATIVE_ROOT / "masks/000000-tissue.png",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=NATIVE_ROOT / "calib_rectified.json",
    )
    parser.add_argument(
        "--table-frame",
        type=Path,
        default=REPO_ROOT / "data/super/table_frame.json",
    )
    parser.add_argument(
        "--tissue",
        type=Path,
        default=V9_ROOT / "tissue.json",
    )
    parser.add_argument(
        "--build-metadata",
        type=Path,
        default=V9_ROOT / "build_metadata.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=NATIVE_ROOT
        / "tissue_heightfield_diagnostic_v1/"
        "tissue_surface_side_by_side.ply",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=NATIVE_ROOT
        / "tissue_heightfield_diagnostic_v1/report.json",
    )
    parser.add_argument(
        "--raw-voxel-mm",
        type=float,
        default=0.35,
        help="Display-only voxel size for the raw RGB point-cloud panel.",
    )
    parser.add_argument(
        "--source-surface-voxel-mm",
        type=float,
        default=1.0,
        help="Voxel size used by the recorded v6/v9 source build.",
    )
    parser.add_argument(
        "--gaussian-sigma-multiplier",
        type=float,
        default=2.0,
        help="Octahedron radius as a multiple of each saved Gaussian scale.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def backproject(
    depth: np.ndarray, mask: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    rows, columns = np.nonzero(valid)
    z = depth[rows, columns].astype(np.float32)
    x = (columns.astype(np.float32) - K[0, 2]) * z / K[0, 0]
    y = (rows.astype(np.float32) - K[1, 2]) * z / K[1, 1]
    return (
        np.stack((x, y, z), axis=1).astype(np.float32),
        np.stack((columns, rows), axis=1).astype(np.int32),
    )


def voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray | None,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors)
    down = cloud.voxel_down_sample(voxel_size)
    down_points = np.asarray(down.points, dtype=np.float64)
    down_colors = (
        np.asarray(down.colors, dtype=np.float64)
        if colors is not None
        else None
    )
    return down_points, down_colors


def make_plane_frame(
    points: np.ndarray, plane: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normal = plane[:3].astype(np.float32)
    normal /= np.linalg.norm(normal)
    signed = points @ normal + plane[3]
    projected = points - signed[:, None] * normal[None]
    origin = projected.mean(axis=0)
    centered = projected - origin
    covariance = centered.T @ centered / max(len(centered), 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    axis_x = eigenvectors[:, -1].astype(np.float32)
    axis_x = axis_x - np.dot(axis_x, normal) * normal
    if np.linalg.norm(axis_x) < 1.0e-6:
        axis_x = np.cross(
            normal, np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        )
    if np.linalg.norm(axis_x) < 1.0e-6:
        axis_x = np.cross(
            normal, np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        )
    axis_x /= np.linalg.norm(axis_x)
    axis_y = np.cross(normal, axis_x)
    axis_y /= np.linalg.norm(axis_y)
    return origin.astype(np.float32), axis_x, axis_y, normal


def project_points(
    points: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    z = points[:, 2]
    positive = z > 1.0e-8
    columns = np.zeros(len(points), dtype=np.int32)
    rows = np.zeros(len(points), dtype=np.int32)
    columns[positive] = np.rint(
        points[positive, 0] * K[0, 0] / z[positive] + K[0, 2]
    ).astype(np.int32)
    rows[positive] = np.rint(
        points[positive, 1] * K[1, 1] / z[positive] + K[1, 2]
    ).astype(np.int32)
    valid = (
        positive
        & (columns >= 0)
        & (columns < width)
        & (rows >= 0)
        & (rows < height)
    )
    return np.stack((columns, rows), axis=1), valid


def reproduce_height_fields(
    surface_points: np.ndarray,
    mask: np.ndarray,
    plane: np.ndarray,
    K: np.ndarray,
    *,
    grid_step: float,
    minimum_height: float,
    maximum_height: float,
    smoothing_sigma: float,
) -> dict[str, Any]:
    origin, axis_x, axis_y, normal = make_plane_frame(
        surface_points, plane
    )
    surface_heights = surface_points @ normal + plane[3]
    valid_surface = np.isfinite(surface_heights) & (
        surface_heights > -0.5 * grid_step
    )
    surface_points = surface_points[valid_surface]
    surface_heights = np.clip(
        surface_heights[valid_surface],
        minimum_height,
        maximum_height,
    )
    projected = surface_points - (
        surface_points @ normal + plane[3]
    )[:, None] * normal[None]
    centered = projected - origin
    footprint = np.stack(
        (centered @ axis_x, centered @ axis_y), axis=1
    ).astype(np.float32)
    minimum_xy = np.percentile(footprint, 1.0, axis=0)
    maximum_xy = np.percentile(footprint, 99.0, axis=0)
    xs = np.arange(
        minimum_xy[0],
        maximum_xy[0] + grid_step * 0.5,
        grid_step,
        dtype=np.float32,
    )
    ys = np.arange(
        minimum_xy[1],
        maximum_xy[1] + grid_step * 0.5,
        grid_step,
        dtype=np.float32,
    )
    grid_xy = np.stack(
        np.meshgrid(xs, ys, indexing="ij"), axis=-1
    ).reshape(-1, 2)
    height_tree = cKDTree(footprint)
    neighbor_count = min(8, len(footprint))
    distances, indices = height_tree.query(grid_xy, k=neighbor_count)
    if neighbor_count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    weights = 1.0 / np.maximum(distances, grid_step * 0.25)
    estimated = np.sum(
        weights * surface_heights[indices], axis=1
    ) / np.sum(weights, axis=1)

    base_points = (
        origin[None]
        + grid_xy[:, :1] * axis_x[None]
        + grid_xy[:, 1:] * axis_y[None]
    )
    top_points = base_points + estimated[:, None] * normal[None]
    pixels, valid = project_points(
        top_points, K, mask.shape[1], mask.shape[0]
    )
    inside_mask = np.zeros(len(grid_xy), dtype=bool)
    valid_ids = np.flatnonzero(valid)
    inside_mask[valid_ids] = mask[
        pixels[valid_ids, 1], pixels[valid_ids, 0]
    ]
    occupied = inside_mask & (distances[:, 0] <= grid_step * 2.5)
    occupied_grid = occupied.reshape(len(xs), len(ys))

    unsmoothed_grid = np.clip(
        estimated.reshape(len(xs), len(ys)),
        minimum_height,
        maximum_height,
    )
    occupied_float = occupied_grid.astype(np.float64)
    if smoothing_sigma > 0.0:
        smoothed_weight = gaussian_filter(
            occupied_float, smoothing_sigma, mode="nearest"
        )
        smoothed_grid = gaussian_filter(
            unsmoothed_grid * occupied_float,
            smoothing_sigma,
            mode="nearest",
        ) / np.maximum(smoothed_weight, 1.0e-6)
    else:
        smoothed_grid = unsmoothed_grid.copy()
    smoothed_grid = np.clip(
        smoothed_grid, minimum_height, maximum_height
    )

    return {
        "origin": origin,
        "axis_x": axis_x,
        "axis_y": axis_y,
        "normal": normal,
        "xs": xs,
        "ys": ys,
        "base_points": base_points.reshape(len(xs), len(ys), 3),
        "occupied": occupied_grid,
        "estimated_before_clip": estimated.reshape(len(xs), len(ys)),
        "unsmoothed": unsmoothed_grid,
        "smoothed": smoothed_grid,
    }


def height_mesh(
    field: dict[str, Any],
    heights: np.ndarray,
    transform: np.ndarray,
    offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points_camera = (
        field["base_points"]
        + heights[..., None] * field["normal"][None, None]
    )
    occupied = field["occupied"]
    index_grid = np.full(occupied.shape, -1, dtype=np.int32)
    points = transform_points(points_camera[occupied], transform)
    points += offset[None]
    index_grid[occupied] = np.arange(len(points), dtype=np.int32)
    faces: list[list[int]] = []
    for x_index in range(occupied.shape[0] - 1):
        for y_index in range(occupied.shape[1] - 1):
            corners = [
                index_grid[x_index, y_index],
                index_grid[x_index + 1, y_index],
                index_grid[x_index + 1, y_index + 1],
                index_grid[x_index, y_index + 1],
            ]
            if min(corners) < 0:
                continue
            faces.append([corners[0], corners[1], corners[2]])
            faces.append([corners[0], corners[2], corners[3]])
    return points, np.asarray(faces, dtype=np.int32)


def height_colors(points: np.ndarray) -> np.ndarray:
    normalized = np.clip((points[:, 2] - 0.003) / 0.012, 0.0, 1.0)
    indices = np.rint(normalized * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(indices[:, None], cv2.COLORMAP_TURBO)
    return bgr[:, 0, ::-1]


def gaussian_octahedra(
    tissue: dict[str, Any],
    offset: np.ndarray,
    sigma_multiplier: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transform = np.asarray(tissue["X_WB"], dtype=np.float64)
    gaussians = tissue["gaussians"]
    centers_local = np.asarray(gaussians["means"], dtype=np.float64)
    scales = np.asarray(gaussians["scales"], dtype=np.float64)
    rotations_local = Rotation.from_quat(
        np.asarray(gaussians["quats"], dtype=np.float64),
        scalar_first=True,
    ).as_matrix()
    rotations_world = np.einsum(
        "ij,njk->nik", transform[:3, :3], rotations_local
    )
    centers_world = transform_points(centers_local, transform) + offset[None]
    colors_per_gaussian = np.clip(
        np.rint(
            np.asarray(gaussians["colors"], dtype=np.float64) * 255.0
        ),
        0,
        255,
    ).astype(np.uint8)
    signs = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    local_vertices = (
        signs[None] * scales[:, None] * sigma_multiplier
    )
    vertices = (
        np.einsum("nij,nkj->nki", rotations_world, local_vertices)
        + centers_world[:, None]
    ).reshape(-1, 3)
    colors = np.repeat(colors_per_gaussian, 6, axis=0)
    base_faces = np.asarray(
        [
            [4, 0, 2],
            [4, 2, 1],
            [4, 1, 3],
            [4, 3, 0],
            [5, 2, 0],
            [5, 1, 2],
            [5, 3, 1],
            [5, 0, 3],
        ],
        dtype=np.int32,
    )
    faces = (
        base_faces[None]
        + 6
        * np.arange(len(centers_world), dtype=np.int32)[:, None, None]
    ).reshape(-1, 3)
    return vertices, colors, faces


def neighbor_jump_statistics(
    heights: np.ndarray, occupied: np.ndarray
) -> dict[str, float]:
    jumps = []
    horizontal = occupied[:-1] & occupied[1:]
    vertical = occupied[:, :-1] & occupied[:, 1:]
    jumps.append(np.abs(heights[:-1][horizontal] - heights[1:][horizontal]))
    jumps.append(
        np.abs(heights[:, :-1][vertical] - heights[:, 1:][vertical])
    )
    values = np.concatenate(jumps) * 1000.0
    return {
        "p50_mm": float(np.quantile(values, 0.50)),
        "p95_mm": float(np.quantile(values, 0.95)),
        "max_mm": float(values.max()),
    }


def write_binary_ply(
    path: Path,
    vertices: np.ndarray,
    colors: np.ndarray,
    faces: np.ndarray,
    comments: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        *[f"comment {comment}" for comment in comments],
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
        "",
    ]
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    packed_vertices = np.empty(len(vertices), dtype=vertex_dtype)
    for axis, name in enumerate(("x", "y", "z")):
        packed_vertices[name] = vertices[:, axis].astype(np.float32)
    for channel, name in enumerate(("red", "green", "blue")):
        packed_vertices[name] = colors[:, channel]
    with path.open("wb") as file:
        file.write("\n".join(header_lines).encode("ascii"))
        file.write(packed_vertices.tobytes())
        for face in faces:
            file.write(struct.pack("<Biii", 3, *map(int, face)))


def main() -> None:
    args = parse_args()
    if not args.overwrite:
        collisions = [
            path for path in (args.output, args.report) if path.exists()
        ]
        if collisions:
            raise FileExistsError(
                "Diagnostic outputs already exist:\n- "
                + "\n- ".join(str(path) for path in collisions)
            )
    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    mask_image = cv2.imread(str(args.tissue_mask), cv2.IMREAD_GRAYSCALE)
    if image_bgr is None:
        raise FileNotFoundError(args.image)
    if mask_image is None:
        raise FileNotFoundError(args.tissue_mask)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask = mask_image > 0
    depth = np.load(args.depth).astype(np.float32)
    calibration = read_json(args.calibration)
    K = np.asarray(calibration["K_left_rect"], dtype=np.float32)
    table = read_json(args.table_frame)
    X_table_camera = np.asarray(
        table["X_table_camera"], dtype=np.float64
    )
    metadata = read_json(args.build_metadata)
    tissue = read_json(args.tissue)
    plane_camera = np.asarray(
        metadata["source_plane_camera"], dtype=np.float32
    )
    grid_step = float(metadata["tissue_fill"]["grid_step_m"])
    minimum_height = float(metadata["minimum_tissue_height"])
    maximum_height = float(metadata["maximum_tissue_height"])
    smoothing_sigma = float(metadata["height_smoothing_sigma"])

    raw_camera, raw_pixels = backproject(depth, mask, K)
    raw_colors = (
        image_rgb[raw_pixels[:, 1], raw_pixels[:, 0]].astype(np.float64)
        / 255.0
    )
    source_surface, _ = voxel_downsample(
        raw_camera,
        None,
        voxel_size=args.source_surface_voxel_mm / 1000.0,
    )
    source_surface = source_surface.astype(np.float32)
    field = reproduce_height_fields(
        source_surface,
        mask,
        plane_camera,
        K,
        grid_step=grid_step,
        minimum_height=minimum_height,
        maximum_height=maximum_height,
        smoothing_sigma=smoothing_sigma,
    )

    raw_table = transform_points(raw_camera, X_table_camera)
    display_raw, display_colors = voxel_downsample(
        raw_table,
        raw_colors,
        voxel_size=args.raw_voxel_mm / 1000.0,
    )
    assert display_colors is not None
    display_colors_u8 = np.clip(
        np.rint(display_colors * 255.0), 0, 255
    ).astype(np.uint8)
    raw_extent_x = float(np.ptp(raw_table[:, 0]))
    panel_step = raw_extent_x + 0.025
    offsets = {
        "raw_depth_rgb": np.asarray([0.0, 0.0, 0.0]),
        "unsmoothed_heightfield": np.asarray([panel_step, 0.0, 0.0]),
        "smoothed_heightfield": np.asarray(
            [2.0 * panel_step, 0.0, 0.0]
        ),
        "current_v9_visual_gaussians": np.asarray(
            [3.0 * panel_step, 0.0, 0.0]
        ),
    }

    unsmoothed_points, unsmoothed_faces = height_mesh(
        field,
        field["unsmoothed"],
        X_table_camera,
        offsets["unsmoothed_heightfield"],
    )
    smoothed_points, smoothed_faces = height_mesh(
        field,
        field["smoothed"],
        X_table_camera,
        offsets["smoothed_heightfield"],
    )
    gaussian_points, gaussian_colors, gaussian_faces = gaussian_octahedra(
        tissue,
        offsets["current_v9_visual_gaussians"],
        args.gaussian_sigma_multiplier,
    )
    point_groups = [
        display_raw,
        unsmoothed_points,
        smoothed_points,
        gaussian_points,
    ]
    color_groups = [
        display_colors_u8,
        height_colors(
            unsmoothed_points
            - offsets["unsmoothed_heightfield"][None]
        ),
        height_colors(
            smoothed_points
            - offsets["smoothed_heightfield"][None]
        ),
        gaussian_colors,
    ]
    face_groups = [
        np.empty((0, 3), dtype=np.int32),
        unsmoothed_faces,
        smoothed_faces,
        gaussian_faces,
    ]
    face_offsets = np.cumsum(
        [0] + [len(points) for points in point_groups[:-1]]
    )
    all_faces = np.concatenate(
        [
            faces + int(offset)
            for faces, offset in zip(
                face_groups, face_offsets, strict=True
            )
            if len(faces)
        ],
        axis=0,
    )
    all_points = np.concatenate(point_groups, axis=0)
    all_colors = np.concatenate(color_groups, axis=0)

    comments = [
        "world axes are frozen table-frame axes; panels differ by +X only",
        "panel_0 raw frame-0 FoundationStereo surface with observed RGB",
        "panel_1 clipped 2mm height field before Gaussian smoothing",
        "panel_2 clipped height field after sigma=1-cell smoothing",
        "panel_3 current v9 visual Gaussians as colored 2-sigma octahedra",
        f"panel_step_m {panel_step:.9f}",
        "height-field panels use identical Turbo colors for 3--15mm height",
        "this file is diagnostic only and is not a runtime asset",
    ]
    write_binary_ply(
        args.output,
        all_points,
        all_colors,
        all_faces,
        comments,
    )

    occupied = field["occupied"]
    smoothing_delta = np.abs(
        field["smoothed"][occupied] - field["unsmoothed"][occupied]
    ) * 1000.0
    before_clip = field["estimated_before_clip"][occupied]
    report = {
        "stage": "super_tissue_heightfield_smoothing_diagnostic",
        "runtime_asset_modified": False,
        "inputs": {
            "image": {"path": str(args.image.resolve()), "sha256": sha256(args.image)},
            "depth": {"path": str(args.depth.resolve()), "sha256": sha256(args.depth)},
            "tissue_mask": {
                "path": str(args.tissue_mask.resolve()),
                "sha256": sha256(args.tissue_mask),
            },
            "calibration": {
                "path": str(args.calibration.resolve()),
                "sha256": sha256(args.calibration),
            },
            "table_frame": {
                "path": str(args.table_frame.resolve()),
                "sha256": sha256(args.table_frame),
            },
            "current_v9_tissue": {
                "path": str(args.tissue.resolve()),
                "sha256": sha256(args.tissue),
            },
        },
        "parameters": {
            "source_surface_voxel_mm": args.source_surface_voxel_mm,
            "heightfield_grid_step_mm": grid_step * 1000.0,
            "minimum_height_mm": minimum_height * 1000.0,
            "maximum_height_mm": maximum_height * 1000.0,
            "smoothing_sigma_cells": smoothing_sigma,
            "smoothing_sigma_mm": smoothing_sigma * grid_step * 1000.0,
            "raw_display_voxel_mm": args.raw_voxel_mm,
            "gaussian_sigma_multiplier": args.gaussian_sigma_multiplier,
        },
        "counts": {
            "raw_depth_points": len(raw_camera),
            "raw_display_points": len(display_raw),
            "source_surface_points": len(source_surface),
            "heightfield_occupied_cells": int(occupied.sum()),
            "current_v9_visual_gaussians": len(
                tissue["gaussians"]["means"]
            ),
            "output_vertices": len(all_points),
            "output_faces": len(all_faces),
        },
        "height_diagnostics": {
            "raw_depth_table_z_mm_percentiles": np.percentile(
                raw_table[:, 2] * 1000.0,
                [0, 1, 5, 50, 95, 99, 100],
            ).tolist(),
            "height_estimate_before_clip_mm_percentiles": np.percentile(
                before_clip * 1000.0,
                [0, 5, 50, 95, 100],
            ).tolist(),
            "cells_clipped_low": int(
                np.count_nonzero(before_clip < minimum_height)
            ),
            "cells_clipped_high": int(
                np.count_nonzero(before_clip > maximum_height)
            ),
            "absolute_smoothing_delta_mm_percentiles": np.percentile(
                smoothing_delta, [0, 50, 90, 95, 99, 100]
            ).tolist(),
            "neighbor_height_jump_before_smoothing": (
                neighbor_jump_statistics(field["unsmoothed"], occupied)
            ),
            "neighbor_height_jump_after_smoothing": (
                neighbor_jump_statistics(field["smoothed"], occupied)
            ),
        },
        "panels": {
            name: {
                "x_offset_m": float(offset[0]),
                "x_offset_mm": float(offset[0] * 1000.0),
            }
            for name, offset in offsets.items()
        },
        "output": {
            "ply": str(args.output.resolve()),
            "ply_sha256": sha256(args.output),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
