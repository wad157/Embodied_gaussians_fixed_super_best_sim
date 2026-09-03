#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOT = REPO_ROOT / "data/super/grasp5_native"
DEFAULT_SOURCE = NATIVE_ROOT / "bodies_v6_dense_frozen_table"
DEFAULT_OUTPUT = NATIVE_ROOT / "bodies_v7_dense_ground_z0"
DEFAULT_TABLE_FRAME = REPO_ROOT / "data/super/table_frame.json"
DEFAULT_CAMERA_MANIFEST = REPO_ROOT / "data/super/grasp5_offline_demo/cameras.json"
DEFAULT_BACKUP = REPO_ROOT / "data/super/coordinate_backups/pre_dense_v7_z0_20260719"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align the fitted dense ground to z=0 and migrate all canonical "
            "table-frame scene/camera assets with one rigid transform."
        )
    )
    parser.add_argument("--source-bodies", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-bodies", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--table-frame", type=Path, default=DEFAULT_TABLE_FRAME)
    parser.add_argument(
        "--camera-manifest", type=Path, default=DEFAULT_CAMERA_MANIFEST
    )
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write v7 assets and replace canonical table/camera transforms.",
    )
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


def normalize_plane(plane: np.ndarray, *, positive_z: bool = False) -> np.ndarray:
    result = np.asarray(plane, dtype=np.float64).copy()
    norm = np.linalg.norm(result[:3])
    if not np.isfinite(norm) or norm < 1.0e-12:
        raise ValueError(f"Invalid plane normal: {result.tolist()}")
    result /= norm
    if positive_z and result[2] < 0.0:
        result = -result
    return result


def transform_plane(plane_b: np.ndarray, x_a_b: np.ndarray) -> np.ndarray:
    plane_a = np.linalg.inv(x_a_b).T @ normalize_plane(plane_b)
    return normalize_plane(plane_a, positive_z=True)


def make_minimal_plane_alignment(plane_old: np.ndarray) -> np.ndarray:
    plane_old = normalize_plane(plane_old, positive_z=True)
    normal = plane_old[:3]
    target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cross = np.cross(normal, target)
    sine = np.linalg.norm(cross)
    cosine = float(np.dot(normal, target))
    if sine < 1.0e-12:
        if cosine < 0.0:
            raise ValueError("Ground normal points opposite +Z after normalization")
        rotation = np.eye(3, dtype=np.float64)
    else:
        axis = cross / sine
        angle = np.arctan2(sine, cosine)
        rotation = Rotation.from_rotvec(axis * angle).as_matrix()

    x_new_old = np.eye(4, dtype=np.float64)
    x_new_old[:3, :3] = rotation
    # z_new is exactly the old plane signed distance n_old*x_old+d_old.
    x_new_old[2, 3] = plane_old[3]
    aligned = transform_plane(plane_old, x_new_old)
    if not np.allclose(aligned, [0.0, 0.0, 1.0, 0.0], atol=1.0e-10):
        raise RuntimeError(f"Plane alignment failed: {aligned.tolist()}")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-10):
        raise RuntimeError("Alignment rotation is not right-handed")
    return x_new_old


def transform_tissue(tissue: dict, x_new_old: np.ndarray) -> dict:
    transformed = json.loads(json.dumps(tissue))
    x_old_body = np.asarray(tissue["X_WB"], dtype=np.float64)
    transformed["X_WB"] = (x_new_old @ x_old_body).tolist()
    return transformed


def transform_ground(ground: dict, x_new_old: np.ndarray) -> dict:
    transformed = json.loads(json.dumps(ground))
    x_old_body = np.asarray(ground["X_WB"], dtype=np.float64)
    means_body = np.asarray(ground["gaussians"]["means"], dtype=np.float64)
    means_old = means_body @ x_old_body[:3, :3].T + x_old_body[:3, 3]
    means_new = means_old @ x_new_old[:3, :3].T + x_new_old[:3, 3]

    quats_body = np.asarray(ground["gaussians"]["quats"], dtype=np.float64)
    rotations_body = Rotation.from_quat(
        quats_body, scalar_first=True
    ).as_matrix()
    rotations_new = np.einsum(
        "ij,njk->nik",
        x_new_old[:3, :3] @ x_old_body[:3, :3],
        rotations_body,
    )

    transformed["X_WB"] = np.eye(4, dtype=np.float64).tolist()
    transformed["gaussians"]["means"] = means_new.tolist()
    transformed["gaussians"]["quats"] = Rotation.from_matrix(
        rotations_new
    ).as_quat(scalar_first=True).tolist()
    transformed["ground_plane"] = {"a": 0.0, "b": 0.0, "c": 1.0, "d": 0.0}
    return transformed


def transform_camera_manifest(manifest: dict, x_new_old: np.ndarray) -> dict:
    transformed = json.loads(json.dumps(manifest))
    for camera_name, camera in transformed.items():
        if "X_WC" not in camera:
            continue
        x_old_camera = np.asarray(camera["X_WC"], dtype=np.float64)
        if x_old_camera.shape != (4, 4):
            raise ValueError(f"Invalid {camera_name}.X_WC shape: {x_old_camera.shape}")
        camera["X_WC"] = (x_new_old @ x_old_camera).tolist()
    return transformed


def active_driver_hashes() -> dict[str, str]:
    paths = [
        REPO_ROOT / "data/super/psm_robot/psm_lnd_pose_driver.npz",
        REPO_ROOT / "data/super/psm_tracking/psm_hybrid_pose_driver.npz",
        REPO_ROOT / "data/super/psm_tracking/psm_paper_exact_pose_driver.npz",
        REPO_ROOT / "data/super/psm_tracking/psm_paper_pose_driver.npz",
        REPO_ROOT / "data/super/psm_tracking/psm_part_corrected_pose_driver.npz",
        REPO_ROOT / "data/super/psm_tracking/psm_registered_lnd_pose_driver.npz",
    ]
    return {
        str(path.relative_to(REPO_ROOT)): sha256(path)
        for path in paths
        if path.exists()
    }


def main() -> None:
    args = parse_args()
    if args.source_bodies.resolve() == args.output_bodies.resolve():
        raise ValueError("Source and output body directories must differ")

    source_tissue_path = args.source_bodies / "tissue.json"
    source_ground_path = args.source_bodies / "ground.json"
    source_plane_path = args.source_bodies / "ground_plane.json"
    source_metadata_path = args.source_bodies / "build_metadata.json"
    source_tissue = read_json(source_tissue_path)
    source_ground = read_json(source_ground_path)
    source_plane_old = normalize_plane(
        read_json(source_plane_path)["plane"], positive_z=True
    )
    source_metadata = read_json(source_metadata_path)
    old_table_frame = read_json(args.table_frame)
    old_camera_manifest = read_json(args.camera_manifest)
    x_old_camera = np.asarray(old_table_frame["X_table_camera"], dtype=np.float64)

    metadata_x_old_camera = np.asarray(
        source_metadata["X_table_camera"], dtype=np.float64
    )
    if not np.allclose(x_old_camera, metadata_x_old_camera, atol=1.0e-10):
        raise RuntimeError(
            "Canonical table_frame no longer matches the v6 source frame; "
            "refusing a possible second migration."
        )

    x_new_old = make_minimal_plane_alignment(source_plane_old)
    x_old_new = np.linalg.inv(x_new_old)
    x_new_camera = x_new_old @ x_old_camera
    x_camera_new = np.linalg.inv(x_new_camera)

    # Keep the source orientation: it must match row 3 of X_table_camera so
    # positive plane distance is the new +Z direction.
    source_plane_camera = normalize_plane(source_metadata["source_plane_camera"])
    target_plane = transform_plane(source_plane_camera, x_new_camera)
    if not np.allclose(target_plane, [0.0, 0.0, 1.0, 0.0], atol=1.0e-9):
        raise RuntimeError(
            f"Camera-to-new-table transform does not align the plane: {target_plane}"
        )

    transformed_tissue = transform_tissue(source_tissue, x_new_old)
    transformed_ground = transform_ground(source_ground, x_new_old)
    transformed_cameras = transform_camera_manifest(old_camera_manifest, x_new_old)
    new_table_frame = {
        **old_table_frame,
        "target_frame": "dense-ground-aligned right-handed table frame",
        "X_table_camera": x_new_camera.tolist(),
        "X_camera_table": x_camera_new.tolist(),
        "table_origin_in_camera": x_camera_new[:3, 3].tolist(),
        "source_ground_plane": source_plane_camera.tolist(),
        "target_ground_plane": [0.0, 0.0, 1.0, 0.0],
        "gravity_table_m_s2": [0.0, 0.0, -9.80665],
        "alignment_policy": (
            "minimal rotation from the dense fitted normal to +Z; no extra "
            "yaw; origin shifted only along the fitted normal"
        ),
        "parent_table_frame_sha256": sha256(args.table_frame),
        "X_new_table_old_table": x_new_old.tolist(),
        "X_old_table_new_table": x_old_new.tolist(),
    }

    rotation = Rotation.from_matrix(x_new_old[:3, :3])
    driver_hashes_before = active_driver_hashes()
    preview = {
        "apply": bool(args.apply),
        "source_plane_old_table": source_plane_old.tolist(),
        "target_plane_new_table": target_plane.tolist(),
        "X_new_table_old_table": x_new_old.tolist(),
        "rotation_angle_deg": float(np.degrees(rotation.magnitude())),
        "rotation_axis": (
            rotation.as_rotvec() / max(rotation.magnitude(), 1.0e-12)
        ).tolist(),
        "translation_mm": (x_new_old[:3, 3] * 1000.0).tolist(),
        "old_X_table_camera": x_old_camera.tolist(),
        "new_X_table_camera": x_new_camera.tolist(),
        "old_table_frame_sha256": sha256(args.table_frame),
        "old_camera_manifest_sha256": sha256(args.camera_manifest),
        "active_psm_driver_hashes_before": driver_hashes_before,
    }
    if not args.apply:
        print(json.dumps(preview, indent=2, ensure_ascii=True))
        return

    if args.output_bodies.exists() and any(args.output_bodies.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_bodies}. Refusing overwrite."
        )
    args.backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.table_frame, args.backup_dir / "table_frame.json")
    shutil.copy2(args.camera_manifest, args.backup_dir / "cameras.json")
    write_json(args.backup_dir / "active_psm_driver_hashes.json", driver_hashes_before)

    args.output_bodies.mkdir(parents=True, exist_ok=False)
    write_json(args.output_bodies / "tissue.json", transformed_tissue)
    write_json(args.output_bodies / "ground.json", transformed_ground)
    write_json(
        args.output_bodies / "ground_plane.json", {"plane": [0.0, 0.0, 1.0, 0.0]}
    )
    write_json(args.table_frame, new_table_frame)
    write_json(args.camera_manifest, transformed_cameras)

    new_table_hash = sha256(args.table_frame)
    new_camera_hash = sha256(args.camera_manifest)
    driver_hashes_after = active_driver_hashes()
    if driver_hashes_after != driver_hashes_before:
        raise RuntimeError("A rectified-camera PSM pose driver changed unexpectedly")

    output_metadata = {
        **source_metadata,
        "output_dir": str(args.output_bodies.resolve()),
        "world_frame": "dense-ground-aligned right-handed table frame, z-up",
        "plane": [0.0, 0.0, 1.0, 0.0],
        "X_table_camera": x_new_camera.tolist(),
        "coordinate_policy": (
            "The dense fitted plane defines z=0. All canonical table-frame "
            "assets are migrated by the same X_new_table_old_table transform."
        ),
        "coordinate_migration": {
            "source_bodies": str(args.source_bodies.resolve()),
            "source_plane_old_table": source_plane_old.tolist(),
            "X_new_table_old_table": x_new_old.tolist(),
            "X_old_table_new_table": x_old_new.tolist(),
            "rotation_angle_deg": float(np.degrees(rotation.magnitude())),
            "translation_mm": (x_new_old[:3, 3] * 1000.0).tolist(),
            "backup_dir": str(args.backup_dir.resolve()),
        },
        "coordinate_guard": {
            "table_frame": str(args.table_frame.resolve()),
            "table_frame_sha256": new_table_hash,
            "camera_manifest": str(args.camera_manifest.resolve()),
            "camera_manifest_sha256": new_camera_hash,
            "active_psm_pose_drivers_modified": False,
        },
    }
    write_json(args.output_bodies / "build_metadata.json", output_metadata)

    report = {
        **preview,
        "new_table_frame_sha256": new_table_hash,
        "new_camera_manifest_sha256": new_camera_hash,
        "active_psm_driver_hashes_after": driver_hashes_after,
        "output_bodies": str(args.output_bodies.resolve()),
        "backup_dir": str(args.backup_dir.resolve()),
    }
    write_json(args.output_bodies / "coordinate_migration_report.json", report)
    write_json(args.backup_dir / "coordinate_migration_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
