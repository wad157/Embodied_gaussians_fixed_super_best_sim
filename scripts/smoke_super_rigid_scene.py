#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import warp
from scipy.spatial.transform import Rotation

from examples.embodied_environments.super_embodied.super_embodied import (
    GROUND_BODY_PATH,
    GROUND_PATH,
    PSM_POSE_DRIVER_PATHS,
    TABLE_FRAME_PATH,
    TISSUE_PARTICLE_RADIUS_SCALE,
    TISSUE_PATH,
    build_environment,
    set_psm_tissue_collisions,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMERA_MANIFEST_PATH = REPO_ROOT / "data/super/grasp5_offline_demo/cameras.json"
DEFAULT_OUTPUT = (
    GROUND_PATH.parent / "rigid_scene_smoke_report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the dense SuPer plane, ground, and rigid tissue scene."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--psm-pose-driver",
        choices=tuple(PSM_POSE_DRIVER_PATHS),
        default="corrected",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform_points(points: np.ndarray, pose_xyz_xyzw: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_quat(pose_xyz_xyzw[3:]).as_matrix()
    return points @ rotation.T + pose_xyz_xyzw[:3]


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    table_hash_before = sha256(TABLE_FRAME_PATH)
    camera_hash_before = sha256(CAMERA_MANIFEST_PATH)
    build_metadata = read_json(GROUND_PATH.parent / "build_metadata.json")
    coordinate_guard = build_metadata["coordinate_guard"]
    expected_table_hash = coordinate_guard["table_frame_sha256"]
    expected_camera_hash = coordinate_guard["camera_manifest_sha256"]
    tissue_data = read_json(TISSUE_PATH)
    ground_data = read_json(GROUND_BODY_PATH)
    plane = np.asarray(read_json(GROUND_PATH)["plane"], dtype=np.float64)
    plane /= np.linalg.norm(plane[:3])

    tissue_local_points = np.asarray(
        tissue_data["particles"]["means"], dtype=np.float64
    )
    tissue_radii = (
        np.asarray(tissue_data["particles"]["radii"], dtype=np.float64)
        * TISSUE_PARTICLE_RADIUS_SCALE
    )

    pose_driver_path = PSM_POSE_DRIVER_PATHS[args.psm_pose_driver]
    env = build_environment(
        num_envs=1,
        add_gaussians=True,
        device=args.device,
        psm_pose_driver_path=pose_driver_path,
    )
    tissue_body_ids = list(env.super_tissue_body_ids)  # type: ignore[attr-defined]
    tissue_body_id = int(tissue_body_ids[0])
    staged_contact_pair_count = int(
        env.super_psm_tissue_contact_pair_count  # type: ignore[attr-defined]
    )
    set_psm_tissue_collisions(env, True)
    collision_enable_count = int(env.sim.model.shape_contact_pair_count)
    set_psm_tissue_collisions(env, False)
    collision_disable_count = int(env.sim.model.shape_contact_pair_count)
    initial_pose = (
        warp.to_torch(env.sim.state_0.body_q)[tissue_body_id]
        .detach()
        .cpu()
        .numpy()
        .copy()
    )

    start = time.perf_counter()
    for _ in range(args.steps):
        env.step(compute_visual_forces=False)
    warp.synchronize()
    elapsed = time.perf_counter() - start

    final_pose = (
        warp.to_torch(env.sim.state_0.body_q)[tissue_body_id]
        .detach()
        .cpu()
        .numpy()
        .copy()
    )
    final_velocity = (
        warp.to_torch(env.sim.state_0.body_qd)[tissue_body_id]
        .detach()
        .cpu()
        .numpy()
        .copy()
    )
    tissue_mass = float(
        warp.to_torch(env.sim.model.body_mass)[tissue_body_id]
        .detach()
        .cpu()
        .item()
    )
    expected_tissue_mass = (
        float(build_metadata["mass_preservation"]["source_sphere_sum_mass_g"])
        / 1000.0
    )
    final_world_points = transform_points(tissue_local_points, final_pose)
    clearances = final_world_points @ plane[:3] + plane[3] - tissue_radii

    runtime_plane = (
        warp.to_torch(env.sim.model.ground_plane).detach().cpu().numpy().copy()
    )
    expected_runtime_plane = np.r_[plane[:3], -plane[3]]
    table_hash_after = sha256(TABLE_FRAME_PATH)
    camera_hash_after = sha256(CAMERA_MANIFEST_PATH)

    finite = bool(
        np.isfinite(final_pose).all()
        and np.isfinite(final_velocity).all()
        and np.isfinite(clearances).all()
    )
    speed = float(np.linalg.norm(final_velocity))
    translation = float(np.linalg.norm(final_pose[:3] - initial_pose[:3]))
    gates = {
        "finite": finite,
        "single_rigid_tissue_body": len(tissue_body_ids) == 1,
        "no_rigid_shape_contact_pairs": (
            int(env.sim.model.shape_contact_pair_count) == 0
        ),
        "staged_psm_tissue_contacts_preallocated": (
            staged_contact_pair_count > 0
            and collision_enable_count == staged_contact_pair_count
        ),
        "psm_tissue_contacts_start_disabled": collision_disable_count == 0,
        "no_soft_particles": int(env.sim.model.particle_count) == 0,
        "no_soft_gaussians": int(env.sim.gaussian_model.num_soft_gaussians) == 0,
        "ground_plane_matches_asset": bool(
            np.allclose(runtime_plane, expected_runtime_plane, atol=1.0e-7)
        ),
        "ground_penetration_below_0.1mm": float(clearances.min()) >= -1.0e-4,
        "translation_below_0.5mm": translation <= 5.0e-4,
        "final_speed_below_1mm_s": speed <= 1.0e-3,
        "tissue_mass_matches_v7": bool(
            np.isclose(tissue_mass, expected_tissue_mass, rtol=1.0e-5)
        ),
        "table_frame_hash_unchanged": (
            table_hash_before
            == table_hash_after
            == expected_table_hash
        ),
        "camera_manifest_hash_unchanged": (
            camera_hash_before
            == camera_hash_after
            == expected_camera_hash
        ),
    }
    report = {
        "stage": "dense_v4_ground_aligned_z0_packed_rigid_tissue_smoke",
        "device": args.device,
        "steps": args.steps,
        "runtime_demo_switched": True,
        "psm_pose_driver": {
            "name": args.psm_pose_driver,
            "path": str(pose_driver_path.resolve()),
            "storage_frame": "left rectified OpenCV camera frame",
            "converted_at_runtime_by": "table_frame.json:X_table_camera",
        },
        "assets": {
            "ground_plane": str(GROUND_PATH.resolve()),
            "ground": str(GROUND_BODY_PATH.resolve()),
            "tissue": str(TISSUE_PATH.resolve()),
        },
        "coordinate_guard": {
            "table_frame_sha256_before": table_hash_before,
            "table_frame_sha256_after": table_hash_after,
            "camera_manifest_sha256_before": camera_hash_before,
            "camera_manifest_sha256_after": camera_hash_after,
        },
        "counts": {
            "bodies": int(env.sim.model.body_count),
            "shapes": int(env.sim.model.shape_count),
            "rigid_shape_contact_pairs": int(
                env.sim.model.shape_contact_pair_count
            ),
            "staged_psm_tissue_contact_pairs": staged_contact_pair_count,
            "ground_contact_pairs": int(
                env.sim.model.shape_ground_contact_pair_count
            ),
            "all_gaussians": int(env.sim.gaussian_model.means.shape[0]),
            "tissue_collision_spheres": len(tissue_local_points),
            "tissue_gaussians": len(tissue_data["gaussians"]["means"]),
            "ground_gaussians": len(ground_data["gaussians"]["means"]),
            "soft_particles": int(env.sim.model.particle_count),
            "soft_gaussians": int(env.sim.gaussian_model.num_soft_gaussians),
        },
        "geometry": {
            "asset_plane_ax_by_cz_d": plane.tolist(),
            "runtime_plane_normal_offset": runtime_plane.tolist(),
            "particle_radius_mm": float(np.median(tissue_radii) * 1000.0),
            "final_ground_clearance_mm_percentiles": (
                np.percentile(clearances, [0, 1, 50, 99, 100]) * 1000.0
            ).tolist(),
            "tissue_mass_g": tissue_mass * 1000.0,
            "expected_v7_tissue_mass_g": expected_tissue_mass * 1000.0,
        },
        "stability": {
            "initial_pose_xyz_xyzw": initial_pose.tolist(),
            "final_pose_xyz_xyzw": final_pose.tolist(),
            "translation_m": translation,
            "final_velocity_angular_linear": final_velocity.tolist(),
            "final_speed_norm": speed,
        },
        "performance": {
            "elapsed_s": elapsed,
            "milliseconds_per_frame": elapsed * 1000.0 / args.steps,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
