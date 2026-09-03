#!/usr/bin/env python3

"""Transform rebuilt SUPER assets into the existing frozen table frame.

Unlike the original table-frame conversion, this script never estimates a new
frame and never writes camera manifests. The newly fitted ground remains a
general plane in the existing table coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOT = REPO_ROOT / "data/super/grasp5_native"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform rebuilt SUPER bodies with a frozen X_table_camera."
    )
    parser.add_argument(
        "--source-bodies",
        type=Path,
        default=NATIVE_ROOT / "bodies_v6_dense_camera",
    )
    parser.add_argument(
        "--output-bodies",
        type=Path,
        default=NATIVE_ROOT / "bodies_v6_dense_frozen_table",
    )
    parser.add_argument(
        "--table-frame",
        type=Path,
        default=REPO_ROOT / "data/super/table_frame.json",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform_plane(plane_camera: np.ndarray, x_table_camera: np.ndarray) -> np.ndarray:
    plane_table = np.linalg.inv(x_table_camera).T @ plane_camera
    plane_table /= np.linalg.norm(plane_table[:3])
    if plane_table[2] < 0.0:
        plane_table = -plane_table
    return plane_table


def transform_tissue(tissue: dict, x_table_camera: np.ndarray) -> dict:
    transformed = json.loads(json.dumps(tissue))
    x_camera_body = np.asarray(tissue["X_WB"], dtype=np.float64)
    transformed["X_WB"] = (x_table_camera @ x_camera_body).tolist()
    return transformed


def transform_ground(
    ground: dict,
    x_table_camera: np.ndarray,
    plane_table: np.ndarray,
) -> dict:
    transformed = json.loads(json.dumps(ground))
    x_camera_body = np.asarray(ground["X_WB"], dtype=np.float64)
    means_body = np.asarray(ground["gaussians"]["means"], dtype=np.float64)
    means_camera = (
        means_body @ x_camera_body[:3, :3].T + x_camera_body[:3, 3]
    )
    means_table = (
        means_camera @ x_table_camera[:3, :3].T + x_table_camera[:3, 3]
    )

    quats_wxyz = np.asarray(ground["gaussians"]["quats"], dtype=np.float64)
    rotations_body = Rotation.from_quat(
        quats_wxyz[:, [1, 2, 3, 0]]
    ).as_matrix()
    rotation_table_body = x_table_camera[:3, :3] @ x_camera_body[:3, :3]
    rotations_table = np.einsum(
        "ij,njk->nik", rotation_table_body, rotations_body
    )
    quats_xyzw = Rotation.from_matrix(rotations_table).as_quat()

    transformed["X_WB"] = np.eye(4, dtype=np.float64).tolist()
    transformed["gaussians"]["means"] = means_table.tolist()
    transformed["gaussians"]["quats"] = quats_xyzw[:, [3, 0, 1, 2]].tolist()
    transformed["ground_plane"] = {
        "a": float(plane_table[0]),
        "b": float(plane_table[1]),
        "c": float(plane_table[2]),
        "d": float(plane_table[3]),
    }
    return transformed


def main() -> None:
    args = parse_args()
    if args.source_bodies.resolve() == args.output_bodies.resolve():
        raise ValueError("Source and output body directories must differ")

    table_frame = read_json(args.table_frame)
    x_table_camera = np.asarray(table_frame["X_table_camera"], dtype=np.float64)
    if x_table_camera.shape != (4, 4):
        raise ValueError("X_table_camera must be 4x4")

    tissue_path = args.source_bodies / "tissue.json"
    ground_path = args.source_bodies / "ground.json"
    plane_path = args.source_bodies / "ground_plane.json"
    metadata_path = args.source_bodies / "build_metadata.json"
    tissue = read_json(tissue_path)
    ground = read_json(ground_path)
    plane_camera = np.asarray(read_json(plane_path)["plane"], dtype=np.float64)
    plane_camera /= np.linalg.norm(plane_camera[:3])
    plane_table = transform_plane(plane_camera, x_table_camera)

    transformed_tissue = transform_tissue(tissue, x_table_camera)
    transformed_ground = transform_ground(ground, x_table_camera, plane_table)

    write_json(args.output_bodies / "tissue.json", transformed_tissue)
    write_json(args.output_bodies / "ground.json", transformed_ground)
    write_json(
        args.output_bodies / "ground_plane.json",
        {"plane": plane_table.tolist()},
    )

    source_metadata = read_json(metadata_path) if metadata_path.exists() else {}
    old_target_plane = np.asarray(
        table_frame.get("target_ground_plane", [0.0, 0.0, 1.0, 0.0]),
        dtype=np.float64,
    )
    angle_deg = float(
        np.degrees(
            np.arccos(
                np.clip(
                    np.dot(plane_table[:3], old_target_plane[:3]), -1.0, 1.0
                )
            )
        )
    )
    metadata = {
        **source_metadata,
        "output_dir": str(args.output_bodies.resolve()),
        "world_frame": "existing frozen right-handed table frame",
        "coordinate_policy": (
            "X_table_camera is copied unchanged from table_frame.json; the new "
            "ground remains a general plane and does not redefine the frame"
        ),
        "source_plane_camera": plane_camera.tolist(),
        "plane": plane_table.tolist(),
        "new_plane_angle_from_old_table_z_deg": angle_deg,
        "X_table_camera": x_table_camera.tolist(),
        "coordinate_guard": {
            "table_frame": str(args.table_frame.resolve()),
            "table_frame_sha256": sha256(args.table_frame),
            "camera_manifest_modified": False,
            "table_frame_modified": False,
        },
        "sources_frozen_transform": {
            "tissue_sha256": sha256(tissue_path),
            "ground_sha256": sha256(ground_path),
            "ground_plane_sha256": sha256(plane_path),
        },
    }
    write_json(args.output_bodies / "build_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
