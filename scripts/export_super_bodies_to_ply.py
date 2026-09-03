#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SuPer bodies to a world-coordinate PLY point cloud.")
    parser.add_argument(
        "--bodies-dir",
        type=Path,
        default=REPO_ROOT / "data" / "super" / "grasp5_native" / "bodies",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "super" / "grasp5_native" / "bodies" / "super_bodies_world.ply",
    )
    parser.add_argument("--plane-resolution", type=int, default=35)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def transform_points(points: np.ndarray, X_WB: np.ndarray) -> np.ndarray:
    return points @ X_WB[:3, :3].T + X_WB[:3, 3]


def body_points(body_path: Path, key: str) -> np.ndarray:
    body = load_json(body_path)
    item = body.get(key)
    if item is None or item.get("means") is None:
        return np.zeros((0, 3), dtype=np.float32)
    points = np.asarray(item["means"], dtype=np.float32)
    X_WB = np.asarray(body["X_WB"], dtype=np.float32)
    return transform_points(points, X_WB).astype(np.float32)


def plane_points(plane: np.ndarray, bounds_points: np.ndarray, resolution: int) -> np.ndarray:
    normal = plane[:3].astype(np.float32)
    normal /= np.linalg.norm(normal)
    d = float(plane[3])

    center = bounds_points.mean(axis=0)
    center = center - normal * (float(np.dot(normal, center)) + d)

    axis_a = np.cross(normal, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    if np.linalg.norm(axis_a) < 1e-6:
        axis_a = np.cross(normal, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    axis_a /= np.linalg.norm(axis_a)
    axis_b = np.cross(normal, axis_a)
    axis_b /= np.linalg.norm(axis_b)

    projected = bounds_points - center
    extent_a = max(0.02, float(np.max(np.abs(projected @ axis_a))) * 1.1)
    extent_b = max(0.02, float(np.max(np.abs(projected @ axis_b))) * 1.1)

    aa = np.linspace(-extent_a, extent_a, resolution, dtype=np.float32)
    bb = np.linspace(-extent_b, extent_b, resolution, dtype=np.float32)
    grid_a, grid_b = np.meshgrid(aa, bb)
    points = center + grid_a[..., None] * axis_a + grid_b[..., None] * axis_b
    return points.reshape(-1, 3).astype(np.float32)


def add_colored(points: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    colors = np.repeat(np.asarray(color, dtype=np.uint8)[None, :], points.shape[0], axis=0)
    return np.column_stack([points, colors]).astype(np.float32)


def write_ascii_ply(path: Path, rows: np.ndarray, comments: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        for comment in comments:
            f.write(f"comment {comment}\n")
        f.write(f"element vertex {rows.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for x, y, z, r, g, b in rows:
            f.write(f"{x:.8f} {y:.8f} {z:.8f} {int(r)} {int(g)} {int(b)}\n")


def main() -> None:
    args = parse_args()
    tissue_path = args.bodies_dir / "tissue.json"
    ground_path = args.bodies_dir / "ground.json"
    plane_path = args.bodies_dir / "ground_plane.json"

    tissue_particles = body_points(tissue_path, "particles")
    tissue_gaussians = body_points(tissue_path, "gaussians")
    ground_gaussians = body_points(ground_path, "gaussians")

    all_body_points = np.concatenate([tissue_particles, tissue_gaussians, ground_gaussians], axis=0)
    plane = np.asarray(load_json(plane_path)["plane"], dtype=np.float32)
    plane_grid = plane_points(plane, all_body_points, args.plane_resolution)

    rows = np.concatenate(
        [
            add_colored(tissue_particles, (230, 45, 45)),
            add_colored(tissue_gaussians, (245, 150, 35)),
            add_colored(ground_gaussians, (35, 170, 70)),
            add_colored(plane_grid, (40, 110, 240)),
        ],
        axis=0,
    )

    comments = [
        "world frame: existing frozen right-handed table frame",
        "red=tissue particles",
        "orange=tissue gaussians",
        "green=ground visual gaussians",
        "blue=ground plane sample grid",
        f"tissue_particles={len(tissue_particles)}",
        f"tissue_gaussians={len(tissue_gaussians)}",
        f"ground_gaussians={len(ground_gaussians)}",
        f"ground_plane_points={len(plane_grid)}",
    ]
    write_ascii_ply(args.output, rows, comments)

    print(f"wrote {args.output}")
    print(f"vertices={rows.shape[0]}")
    print(f"bounds_min={rows[:, :3].min(axis=0).tolist()}")
    print(f"bounds_max={rows[:, :3].max(axis=0).tolist()}")


if __name__ == "__main__":
    main()
