#!/usr/bin/env python3

"""Repack the frozen SUPER rigid tissue with a tblock-like sphere layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree


REPO = Path(__file__).resolve().parents[1]
NATIVE_ROOT = REPO / "data/super/grasp5_native"
DEFAULT_SOURCE = NATIVE_ROOT / "bodies_v7_dense_ground_z0"
DEFAULT_OUTPUT = NATIVE_ROOT / "bodies_v8_packed_rigid_tissue"
EXPECTED_SHA256 = {
    "table_frame": "6dddc2178cdf816f5dada5febdd528f80e42d52e631076e1f5f4a952297adecf",
    "cameras": "e1e7b7e7e21ca8a9409c88a29409d2e9ad6b783a85b277e71cec0c84340ce4ef",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a close-packed rigid SUPER tissue without moving its frame."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--particle-radius-mm", type=float, default=1.0)
    parser.add_argument("--nearest-pitch-mm", type=float, default=1.85)
    parser.add_argument("--footprint-radius-mm", type=float, default=1.05)
    parser.add_argument("--surface-layer-allowance", type=float, default=0.75)
    parser.add_argument("--xy-jitter-mm", type=float, default=0.08)
    parser.add_argument("--z-jitter-mm", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def inverse_transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (points - transform[:3, 3]) @ transform[:3, :3]


def five_number(values: np.ndarray) -> list[float]:
    return np.percentile(values, [0, 5, 50, 95, 100]).tolist()


def plane_basis(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    axis_x = np.asarray(transform[:3, 0], dtype=np.float64)
    axis_x -= np.dot(axis_x, z_axis) * z_axis
    axis_x /= np.linalg.norm(axis_x)
    axis_y = np.cross(z_axis, axis_x)
    axis_y /= np.linalg.norm(axis_y)
    return axis_x, axis_y


def source_columns(
    world_points: np.ndarray,
    origin: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centered = world_points - origin
    uv = np.stack((centered @ axis_x, centered @ axis_y), axis=1)
    unique_uv, inverse = np.unique(
        np.round(uv, decimals=6), axis=0, return_inverse=True
    )
    top_height = np.full(len(unique_uv), -np.inf, dtype=np.float64)
    np.maximum.at(top_height, inverse, world_points[:, 2])
    return unique_uv, top_height


def hcp_candidates(
    *,
    columns_uv: np.ndarray,
    column_top: np.ndarray,
    origin: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    radius: float,
    pitch: float,
    footprint_radius: float,
    surface_allowance: float,
) -> np.ndarray:
    row_pitch = math.sqrt(3.0) / 2.0 * pitch
    layer_pitch = math.sqrt(2.0 / 3.0) * pitch
    tree = cKDTree(columns_uv)
    u_min = float(columns_uv[:, 0].min() - pitch)
    u_max = float(columns_uv[:, 0].max() + pitch)
    v_min = float(columns_uv[:, 1].min() - row_pitch)
    v_max = float(columns_uv[:, 1].max() + row_pitch)
    candidate_local = []
    layer_heights = np.arange(
        radius,
        float(column_top.max()) + layer_pitch,
        layer_pitch,
        dtype=np.float64,
    )
    for layer_index, height in enumerate(layer_heights):
        layer_u_offset = 0.5 * pitch if layer_index % 2 else 0.0
        layer_v_offset = row_pitch / 3.0 if layer_index % 2 else 0.0
        rows = np.arange(v_min, v_max + row_pitch, row_pitch, dtype=np.float64)
        for row_index, row_v in enumerate(rows):
            row_u_offset = 0.5 * pitch if row_index % 2 else 0.0
            values_u = np.arange(
                u_min + row_u_offset + layer_u_offset,
                u_max + pitch,
                pitch,
                dtype=np.float64,
            )
            uv = np.stack(
                (
                    values_u,
                    np.full_like(values_u, row_v + layer_v_offset),
                ),
                axis=1,
            )
            distances, indices = tree.query(uv, k=min(4, len(columns_uv)))
            if distances.ndim == 1:
                distances = distances[:, None]
                indices = indices[:, None]
            weights = 1.0 / np.maximum(distances, pitch * 0.25)
            local_top = np.sum(weights * column_top[indices], axis=1) / np.sum(
                weights, axis=1
            )
            keep = (distances[:, 0] <= footprint_radius) & (
                height
                <= np.maximum(
                    radius,
                    local_top - radius + layer_pitch * surface_allowance,
                )
            )
            if np.any(keep):
                candidate_local.append(
                    np.column_stack(
                        (uv[keep], np.full(np.count_nonzero(keep), height))
                    )
                )
    if not candidate_local:
        raise RuntimeError("Close-packed lattice contains no tissue particles")
    local = np.concatenate(candidate_local, axis=0)
    return (
        origin[None]
        + local[:, :1] * axis_x[None]
        + local[:, 1:2] * axis_y[None]
        + local[:, 2:3] * np.asarray([[0.0, 0.0, 1.0]])
    )


def jitter_points(
    points: np.ndarray,
    *,
    radius: float,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    xy_jitter: float,
    z_jitter: float,
    rng: np.random.Generator,
) -> np.ndarray:
    output = points.copy()
    output += rng.uniform(-xy_jitter, xy_jitter, (len(points), 1)) * axis_x
    output += rng.uniform(-xy_jitter, xy_jitter, (len(points), 1)) * axis_y
    above_ground = points[:, 2] > radius + 1.0e-8
    output[above_ground, 2] += rng.uniform(
        -z_jitter, z_jitter, np.count_nonzero(above_ground)
    )
    output[:, 2] = np.maximum(output[:, 2], radius)
    return output


def topdown_panel(
    points: np.ndarray,
    radius: float,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    title: str,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    width, height = 900, 720
    margin = 35
    output = np.full((height, width, 3), 245, dtype=np.uint8)
    u = points @ axis_x
    v = points @ axis_y
    u_min, u_max, v_min, v_max = bounds
    scale = min(
        (width - 2 * margin) / max(u_max - u_min, 1.0e-9),
        (height - 2 * margin - 45) / max(v_max - v_min, 1.0e-9),
    )
    order = np.argsort(points[:, 2])
    circle_radius = max(1, int(round(radius * scale)))
    z_min = float(points[:, 2].min())
    z_span = max(float(np.ptp(points[:, 2])), 1.0e-9)
    for index in order:
        x = int(round(margin + (u[index] - u_min) * scale))
        y = int(round(height - margin - (v[index] - v_min) * scale))
        value = (points[index, 2] - z_min) / z_span
        color = (
            int(65 + 130 * value),
            int(90 + 100 * (1.0 - value)),
            int(220 - 80 * value),
        )
        cv2.circle(output, (x, y), circle_radius, color, -1, cv2.LINE_AA)
        cv2.circle(output, (x, y), circle_radius, (45, 45, 45), 1, cv2.LINE_AA)
    cv2.rectangle(output, (0, 0), (width, 45), (20, 20, 20), -1)
    cv2.putText(
        output,
        title,
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    if (
        args.particle_radius_mm <= 0.0
        or args.nearest_pitch_mm <= 0.0
        or args.footprint_radius_mm <= 0.0
    ):
        raise ValueError("Packing distances must be positive")
    if args.nearest_pitch_mm >= 2.0 * args.particle_radius_mm:
        raise ValueError("nearest pitch must be smaller than the particle diameter")
    source_paths = {
        name: args.source_dir / name
        for name in (
            "tissue.json",
            "ground.json",
            "ground_plane.json",
            "build_metadata.json",
        )
    }
    for path in source_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    frozen_paths = {
        "table_frame": REPO / "data/super/table_frame.json",
        "cameras": REPO / "data/super/grasp5_offline_demo/cameras.json",
    }
    frozen_hashes = {name: sha256(path) for name, path in frozen_paths.items()}
    if frozen_hashes != EXPECTED_SHA256:
        raise RuntimeError(f"Frozen coordinate hash mismatch: {frozen_hashes}")

    source_tissue = read_json(source_paths["tissue.json"])
    source_ground = read_json(source_paths["ground.json"])
    source_plane = read_json(source_paths["ground_plane.json"])
    source_metadata = read_json(source_paths["build_metadata.json"])
    transform = np.asarray(source_tissue["X_WB"], dtype=np.float64)
    local_points = np.asarray(source_tissue["particles"]["means"], dtype=np.float64)
    old_radii = np.asarray(source_tissue["particles"]["radii"], dtype=np.float64)
    old_colors = np.asarray(source_tissue["particles"]["colors"], dtype=np.float64)
    if not np.allclose(old_radii, old_radii[0], atol=1.0e-9):
        raise ValueError("Source tissue must use one collision-sphere radius")
    source_radius = float(old_radii[0])
    radius = args.particle_radius_mm / 1000.0
    world_points = transform_points(local_points, transform)
    if abs(float(world_points[:, 2].min()) - source_radius) > 1.0e-5:
        raise RuntimeError("Source bottom layer does not touch the frozen z=0 plane")
    axis_x, axis_y = plane_basis(transform)
    origin = np.mean(world_points, axis=0)
    origin[2] = 0.0
    columns_uv, column_top_centers = source_columns(
        world_points, origin, axis_x, axis_y
    )
    column_top_surfaces = column_top_centers + source_radius
    pitch = args.nearest_pitch_mm / 1000.0
    candidate_world = hcp_candidates(
        columns_uv=columns_uv,
        column_top=column_top_surfaces,
        origin=origin,
        axis_x=axis_x,
        axis_y=axis_y,
        radius=radius,
        pitch=pitch,
        footprint_radius=args.footprint_radius_mm / 1000.0,
        surface_allowance=args.surface_layer_allowance,
    )
    rng = np.random.default_rng(args.seed)
    candidate_world = jitter_points(
        candidate_world,
        radius=radius,
        axis_x=axis_x,
        axis_y=axis_y,
        xy_jitter=args.xy_jitter_mm / 1000.0,
        z_jitter=args.z_jitter_mm / 1000.0,
        rng=rng,
    )
    candidate_local = inverse_transform_points(candidate_world, transform)
    nearest_source = cKDTree(world_points).query(candidate_world, k=1)[1]
    candidate_colors = old_colors[nearest_source]

    old_nearest = cKDTree(world_points).query(world_points, k=2)[0][:, 1]
    new_nearest = cKDTree(candidate_world).query(candidate_world, k=2)[0][:, 1]
    overlap = 2.0 * radius - new_nearest
    if np.median(overlap) <= 0.0:
        raise RuntimeError("Packed tissue particles do not overlap at the median")
    if np.mean(overlap > 0.0) < 0.95:
        raise RuntimeError("Fewer than 95% of packed particles overlap a neighbor")
    if float(candidate_world[:, 2].min() - radius) < -1.0e-8:
        raise RuntimeError("Packed tissue penetrates the frozen z=0 ground")

    output_tissue = json.loads(json.dumps(source_tissue))
    output_tissue["particles"] = {
        "means": candidate_local.tolist(),
        "quats": np.tile(
            np.asarray([[1.0, 0.0, 0.0, 0.0]]), (len(candidate_local), 1)
        ).tolist(),
        "radii": np.full(len(candidate_local), radius).tolist(),
        "colors": candidate_colors.tolist(),
    }
    default_density = 1000.0
    source_mass = (
        len(world_points)
        * 4.0
        / 3.0
        * math.pi
        * source_radius**3
        * default_density
    )
    shape_density = source_mass / (
        len(candidate_world) * 4.0 / 3.0 * math.pi * radius**3
    )
    packed_mass = len(candidate_world) * 4.0 / 3.0 * math.pi * radius**3 * shape_density
    if not np.isclose(source_mass, packed_mass, rtol=1.0e-12):
        raise RuntimeError("Packed tissue density does not preserve sphere-sum mass")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "tissue.json", output_tissue)
    write_json(args.output_dir / "ground.json", source_ground)
    write_json(args.output_dir / "ground_plane.json", source_plane)

    all_world = np.concatenate((world_points, candidate_world), axis=0)
    all_u = all_world @ axis_x
    all_v = all_world @ axis_y
    bounds = (
        float(all_u.min() - radius),
        float(all_u.max() + radius),
        float(all_v.min() - radius),
        float(all_v.max() + radius),
    )
    old_panel = topdown_panel(
        world_points,
        source_radius,
        axis_x,
        axis_y,
        f"v7 simple cubic: {len(world_points)} spheres, radius={source_radius * 1000.0:.2f} mm",
        bounds,
    )
    new_panel = topdown_panel(
        candidate_world,
        radius,
        axis_x,
        axis_y,
        f"packed HCP+jitter: {len(candidate_world)} spheres, radius={radius * 1000.0:.2f} mm",
        bounds,
    )
    cv2.imwrite(
        str(args.output_dir / "particle_packing_comparison.png"),
        np.hstack((old_panel, new_panel)),
    )

    center_bbox_delta = np.concatenate(
        (
            candidate_world.min(axis=0) - world_points.min(axis=0),
            candidate_world.max(axis=0) - world_points.max(axis=0),
        )
    )
    metadata = {
        **source_metadata,
        "output_dir": str(args.output_dir.resolve()),
        "source_v7_dir": str(args.source_dir.resolve()),
        "source_v7_tissue_sha256": sha256(source_paths["tissue.json"]),
        "tissue_particles": len(candidate_world),
        "particle_radius": radius,
        "tissue_shape_density_kg_m3": shape_density,
        "packing": {
            "method": "HCP AB layers with bounded deterministic jitter",
            "nearest_pitch_mm": args.nearest_pitch_mm,
            "source_particle_radius_mm": source_radius * 1000.0,
            "packed_particle_radius_mm": radius * 1000.0,
            "row_pitch_mm": math.sqrt(3.0) / 2.0 * args.nearest_pitch_mm,
            "layer_pitch_mm": math.sqrt(2.0 / 3.0) * args.nearest_pitch_mm,
            "xy_jitter_mm": args.xy_jitter_mm,
            "z_jitter_mm": args.z_jitter_mm,
            "source_particle_count": len(world_points),
            "packed_particle_count": len(candidate_world),
            "source_nearest_mm_min_p05_p50_p95_max": (
                np.asarray(five_number(old_nearest)) * 1000.0
            ).tolist(),
            "packed_nearest_mm_min_p05_p50_p95_max": (
                np.asarray(five_number(new_nearest)) * 1000.0
            ).tolist(),
            "packed_overlap_mm_min_p05_p50_p95_max": (
                np.asarray(five_number(overlap)) * 1000.0
            ).tolist(),
            "overlapping_nearest_neighbor_fraction": float(np.mean(overlap > 0.0)),
            "center_bbox_delta_mm_min_xyz_max_xyz": (center_bbox_delta * 1000.0).tolist(),
        },
        "mass_preservation": {
            "source_default_density_kg_m3": default_density,
            "packed_shape_density_kg_m3": shape_density,
            "source_sphere_sum_mass_g": source_mass * 1000.0,
            "packed_sphere_sum_mass_g": packed_mass * 1000.0,
        },
        "coordinate_guard": {
            **source_metadata["coordinate_guard"],
            "table_frame_sha256": frozen_hashes["table_frame"],
            "camera_manifest_sha256": frozen_hashes["cameras"],
            "X_WB_unchanged": bool(
                np.array_equal(
                    np.asarray(output_tissue["X_WB"]),
                    np.asarray(source_tissue["X_WB"]),
                )
            ),
            "tissue_gaussians_unchanged": bool(
                output_tissue["gaussians"] == source_tissue["gaussians"]
            ),
            "ground_assets_unchanged": True,
        },
    }
    write_json(args.output_dir / "build_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
