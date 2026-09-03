#!/usr/bin/env python3
"""Stage-D offline calibration of global SUPER tissue material parameters.

Only Young's modulus and velocity damping are varied.  The stage-A pose,
contact configuration, Poisson ratio, fixed boundary, time base, and all
solver settings remain frozen.  Visual forces and residual mapping stay off.

The default controller performs:

1. a logarithmic coarse search on the continuous calibration interval;
2. a 3x3 logarithmic refinement around the best coarse candidate;
3. full validation/test replay for the three best calibration candidates;
4. two dense, independent replays of the selected material on one GPU.

Two worker processes are used by default so cuda:0 and cuda:1 can evaluate
independent candidates concurrently without sharing a Warp environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
import torch
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from embodied_environments.super_embodied.super_embodied import (  # noqa: E402
    build_environment,
    set_psm_tissue_collisions,
)
from replay_super_tissue_calibration_stage_c import (  # noqa: E402
    apply_pose,
    camera_to_world_translation,
    deformation_metrics,
    detect_contact,
    load_manifest,
    local_tissue_tensors,
    selected_frames,
    sha256_file,
)


DEFAULT_MANIFEST = (
    REPO_ROOT
    / "data/super/tissue_calibration_v1/stage_a_frozen_manifest.json"
)
DEFAULT_SURFACE_ROOT = (
    REPO_ROOT
    / "data/super/tissue_calibration_v1/stage_b_surface_observations"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "data/super/tissue_calibration_v1/stage_d_material_calibration"
)

CALIBRATION_START = 270
CALIBRATION_END = 906
VALIDATION_START = 907
VALIDATION_END = 1276
TEST_START = 1277
TEST_END = 1439

SPLIT_SPECS = {
    "calibration": (CALIBRATION_START, CALIBRATION_END, 15),
    "validation": (VALIDATION_START, VALIDATION_END, 15),
    "test": (TEST_START, TEST_END, 5),
}

KEYPOINT_LOSS_WEIGHT = 0.70
SURFACE_LOSS_WEIGHT = 0.30
KEYPOINT_ROBUST_CAP_M = 0.010
SURFACE_ROBUST_CAP_M = 0.005
SURFACE_MOTION_WEIGHT = 4.0
SURFACE_MOTION_SCALE_M = 0.004
SURFACE_STATIC_TOLERANCE_M = 0.0005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--surface-root", type=Path, default=DEFAULT_SURFACE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--devices",
        default="cuda:0,cuda:1",
        help="Comma-separated controller worker devices.",
    )
    parser.add_argument(
        "--worker-batch",
        type=Path,
        help="Internal worker mode: evaluate the candidates in this JSON file.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--warp-cache-dir",
        type=Path,
        default=Path("/tmp/warp-super-tissue-stage-d"),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse a candidate report only when its parameters and inputs match.",
    )
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def candidate_signature(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "young_modulus_pa": float(candidate["young_modulus_pa"]),
        "velocity_damping_per_second": float(
            candidate["velocity_damping_per_second"]
        ),
        "frame_end": int(candidate["frame_end"]),
        "dense_stability": bool(candidate.get("dense_stability", False)),
    }


def split_for_frame(frame: int) -> str | None:
    for name, (start, end, _stride) in SPLIT_SPECS.items():
        if start <= frame <= end:
            return name
    return None


def evaluation_frames(
    surface_root: Path,
    manifest: dict[str, Any],
    frame_end: int,
) -> list[int]:
    frames: set[int] = set()
    for start, end, stride in SPLIT_SPECS.values():
        actual_end = min(end, frame_end)
        if actual_end < start:
            continue
        frames.update(range(start, actual_end + 1, stride))
        frames.add(actual_end)
    for entry in manifest["sequence"]["landmarks"].values():
        frame = int(entry["left_frame"])
        if CALIBRATION_START <= frame <= frame_end:
            frames.add(frame)
    return sorted(
        frame
        for frame in frames
        if (surface_root / "frames" / f"{frame:06d}.npz").is_file()
    )


def load_observations(
    *,
    surface_root: Path,
    tissue_asset_path: Path,
    manifest: dict[str, Any],
    frame_end: int,
) -> dict[str, Any]:
    with np.load(tissue_asset_path, allow_pickle=False) as tissue:
        rest_positions = np.asarray(
            tissue["rest_positions_table"], dtype=np.float32
        )
        top_mask = np.asarray(tissue["top_node_mask"], dtype=bool)
    top_ids = np.flatnonzero(top_mask).astype(np.int64)
    rest_top = rest_positions[top_ids]
    rest_tree = cKDTree(rest_top)

    keypoint_path = surface_root / "keypoints" / "keypoint_tracks.npz"
    with np.load(keypoint_path, allow_pickle=False) as keypoints:
        kp = {key: np.asarray(keypoints[key]) for key in keypoints.files}
    valid = kp["has_3d"] & np.isfinite(kp["points_table"]).all(axis=1)

    # The twelve tracks born at first contact span the calibration interval.
    # Keeping this cohort fixed avoids changing correspondence when a candidate
    # changes the simulated deformation.
    track_ids: list[int] = []
    track_particle_ids: dict[int, int] = {}
    track_reference_points: dict[int, np.ndarray] = {}
    for track_id in kp["selected_track_ids"]:
        track_id = int(track_id)
        mask = (kp["track_ids"] == track_id) & valid
        track_frames = kp["frames"][mask]
        if len(track_frames) == 0 or int(track_frames.min()) != CALIBRATION_START:
            continue
        reference_mask = mask & (kp["frames"] == CALIBRATION_START)
        if not np.any(reference_mask):
            continue
        reference = np.asarray(
            kp["points_table"][np.flatnonzero(reference_mask)[0]],
            dtype=np.float64,
        )
        _distance, local_top_id = rest_tree.query(reference, k=1)
        track_ids.append(track_id)
        track_particle_ids[track_id] = int(top_ids[int(local_top_id)])
        track_reference_points[track_id] = reference
    if len(track_ids) < 8:
        raise RuntimeError(
            f"Only {len(track_ids)} first-contact 3D tracks are available"
        )

    observations_by_frame: dict[int, dict[str, Any]] = {}
    for frame in evaluation_frames(surface_root, manifest, frame_end):
        surface_path = surface_root / "frames" / f"{frame:06d}.npz"
        with np.load(surface_path, allow_pickle=False) as surface:
            points = np.asarray(surface["points_table"], dtype=np.float64)
        rest_distance = rest_tree.query(points, k=1, workers=1)[0]
        surface_weights = 1.0 + SURFACE_MOTION_WEIGHT * np.clip(
            (rest_distance - SURFACE_STATIC_TOLERANCE_M)
            / SURFACE_MOTION_SCALE_M,
            0.0,
            1.0,
        )

        kp_mask = (
            (kp["frames"] == frame)
            & valid
            & np.isin(kp["track_ids"], np.asarray(track_ids, dtype=np.int32))
        )
        frame_track_ids = kp["track_ids"][kp_mask].astype(np.int32)
        frame_track_points = kp["points_table"][kp_mask].astype(np.float64)
        frame_fb_error = kp["fb_error_px"][kp_mask].astype(np.float64)
        observations_by_frame[frame] = {
            "surface_points": points,
            "surface_weights": surface_weights,
            "track_ids": frame_track_ids,
            "track_displacements": np.asarray(
                [
                    point - track_reference_points[int(track_id)]
                    for track_id, point in zip(
                        frame_track_ids, frame_track_points, strict=True
                    )
                ],
                dtype=np.float64,
            ).reshape(-1, 3),
            "track_weights": 1.0 / (1.0 + np.square(frame_fb_error / 2.0)),
        }

    return {
        "top_ids": top_ids,
        "track_ids": track_ids,
        "track_particle_ids": track_particle_ids,
        "observations_by_frame": observations_by_frame,
        "keypoint_path": keypoint_path,
    }


def set_candidate_material(
    environment: Any,
    young_modulus_pa: float | np.ndarray,
    velocity_damping_per_second: float,
) -> None:
    young_array = np.asarray(young_modulus_pa, dtype=np.float64) if not np.isscalar(young_modulus_pa) else None
    if (young_array is not None and (young_array.ndim != 1 or np.any(young_array <= 0))) or (young_array is None and young_modulus_pa <= 0.0) or velocity_damping_per_second < 0.0:
        raise ValueError("Young's modulus must be >0 and damping must be >=0")
    poisson_ratio = float(environment.super_tissue_poisson_ratio)
    young_for_material = young_modulus_pa if young_array is None else young_array
    mu = young_for_material / (2.0 * (1.0 + poisson_ratio))
    lame_lambda = (
        young_for_material
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    handle = environment.super_tissue_soft_handle
    materials = wp.to_torch(environment.sim.model.tet_materials)
    materials[handle.tet_start : handle.tet_end, 0] = torch.as_tensor(mu, device=materials.device, dtype=materials.dtype)
    materials[handle.tet_start : handle.tet_end, 1] = torch.as_tensor(lame_lambda, device=materials.device, dtype=materials.dtype)
    materials[handle.tet_start : handle.tet_end, 2] = 0.0
    environment.physics_settings.particle_velocity_damping_per_second = (
        velocity_damping_per_second
    )
    environment.super_tissue_young_modulus_pa = float(np.mean(young_array)) if young_array is not None else float(young_modulus_pa)

    # Damping is captured as a scalar in the physics graph.
    if hasattr(environment.sim, "_physics_step_cache"):
        delattr(environment.sim, "_physics_step_cache")


def reset_candidate(
    *,
    environment: Any,
    initial_state: Any,
    initial_timestamp: float,
    joint_offsets: np.ndarray,
    translation_world: np.ndarray,
) -> None:
    environment.sim.copy_embodied_gaussian_state(initial_state)
    environment.sim.reset()
    environment.sim.clear_soft_visual_force_cache()
    environment.sim.state_0.clear_forces()
    apply_pose(
        environment,
        initial_timestamp,
        joint_offsets,
        translation_world,
    )
    environment.sim.sync_kinematic_body_interpolation()
    environment.sim.update_gaussian_transforms()


def summarize_split(accumulator: dict[str, list[np.ndarray]]) -> dict[str, Any]:
    key_errors = (
        np.concatenate(accumulator["key_errors"])
        if accumulator["key_errors"]
        else np.empty(0, dtype=np.float64)
    )
    key_weights = (
        np.concatenate(accumulator["key_weights"])
        if accumulator["key_weights"]
        else np.empty(0, dtype=np.float64)
    )
    surface_errors = (
        np.concatenate(accumulator["surface_errors"])
        if accumulator["surface_errors"]
        else np.empty(0, dtype=np.float64)
    )
    surface_weights = (
        np.concatenate(accumulator["surface_weights"])
        if accumulator["surface_weights"]
        else np.empty(0, dtype=np.float64)
    )

    def weighted_robust_rmse(
        errors: np.ndarray, weights: np.ndarray, cap: float
    ) -> float | None:
        if not len(errors):
            return None
        return float(
            np.sqrt(
                np.sum(weights * np.minimum(errors, cap) ** 2)
                / np.sum(weights)
            )
        )

    key_rmse_m = weighted_robust_rmse(
        key_errors, key_weights, KEYPOINT_ROBUST_CAP_M
    )
    surface_rmse_m = weighted_robust_rmse(
        surface_errors, surface_weights, SURFACE_ROBUST_CAP_M
    )
    available: list[tuple[float, float]] = []
    if key_rmse_m is not None:
        available.append((KEYPOINT_LOSS_WEIGHT, key_rmse_m * 1000.0))
    if surface_rmse_m is not None:
        available.append((SURFACE_LOSS_WEIGHT, surface_rmse_m * 1000.0))
    objective_mm = (
        sum(weight * value for weight, value in available)
        / sum(weight for weight, _value in available)
        if available
        else float("inf")
    )
    return {
        "objective_mm": objective_mm,
        "keypoint_robust_rmse_mm": (
            None if key_rmse_m is None else key_rmse_m * 1000.0
        ),
        "keypoint_error_p50_p95_max_mm": (
            None
            if not len(key_errors)
            else (
                np.quantile(key_errors, [0.5, 0.95, 1.0]) * 1000.0
            ).tolist()
        ),
        "keypoint_observation_count": int(len(key_errors)),
        "surface_robust_rmse_mm": (
            None if surface_rmse_m is None else surface_rmse_m * 1000.0
        ),
        "surface_error_p50_p95_max_mm": (
            None
            if not len(surface_errors)
            else (
                np.quantile(surface_errors, [0.5, 0.95, 1.0]) * 1000.0
            ).tolist()
        ),
        "surface_point_count": int(len(surface_errors)),
        "evaluated_frame_count": int(
            len(accumulator["evaluated_frames"])
        ),
        "evaluated_frames": [int(x) for x in accumulator["evaluated_frames"]],
    }


def evaluate_candidate(
    *,
    candidate: dict[str, Any],
    environment: Any,
    initial_state: Any,
    manifest: dict[str, Any],
    frame_indices: np.ndarray,
    frame_timestamps: np.ndarray,
    joint_offsets: np.ndarray,
    translation_world: np.ndarray,
    rest_positions: torch.Tensor,
    local_tetrahedra: torch.Tensor,
    rest_inverse: torch.Tensor,
    dynamic_mask: torch.Tensor,
    observation_bundle: dict[str, Any],
    manifest_path: Path,
    surface_report_path: Path,
    tet_young_modulus_pa: np.ndarray | None = None,
) -> dict[str, Any]:
    young_modulus_pa = float(candidate["young_modulus_pa"])
    damping = float(candidate["velocity_damping_per_second"])
    dense_stability = bool(candidate.get("dense_stability", False))
    set_candidate_material(environment, tet_young_modulus_pa if tet_young_modulus_pa is not None else young_modulus_pa, damping)
    reset_candidate(
        environment=environment,
        initial_state=initial_state,
        initial_timestamp=float(frame_timestamps[0]),
        joint_offsets=joint_offsets,
        translation_world=translation_world,
    )

    # Capture the candidate-specific damping scalar, then restore exactly.
    environment.step(compute_visual_forces=False)
    wp.synchronize()
    reset_candidate(
        environment=environment,
        initial_state=initial_state,
        initial_timestamp=float(frame_timestamps[0]),
        joint_offsets=joint_offsets,
        translation_world=translation_world,
    )

    accumulators = {
        name: {
            "key_errors": [],
            "key_weights": [],
            "surface_errors": [],
            "surface_weights": [],
            "evaluated_frames": [],
        }
        for name in SPLIT_SPECS
    }
    observations = observation_bundle["observations_by_frame"]
    top_ids = observation_bundle["top_ids"]
    track_particle_ids = observation_bundle["track_particle_ids"]
    simulated_reference: dict[int, np.ndarray] | None = None
    landmark_frames = {
        int(entry["left_frame"]): name
        for name, entry in manifest["sequence"]["landmarks"].items()
    }
    frame_records: dict[str, Any] = {}
    state_hashes: dict[str, str] = {}
    contact_records: dict[str, Any] = {}
    stability = {
        "finite": True,
        "inverted_tetrahedra_max": 0,
        "minimum_tetrahedron_volume_ratio": float("inf"),
        "maximum_dynamic_speed_m_s": 0.0,
        "maximum_dynamic_displacement_m": 0.0,
        "maximum_anchor_drift_m": 0.0,
        "checked_frame_count": 0,
    }

    dt = float(environment.physics_settings.dt)
    first_timestamp = float(frame_timestamps[0])
    completed_steps = 0
    next_step_timestamp = first_timestamp + dt
    started = time.perf_counter()

    for frame_index, frame_timestamp_value in zip(
        frame_indices, frame_timestamps, strict=True
    ):
        frame = int(frame_index)
        frame_timestamp = float(frame_timestamp_value)
        while next_step_timestamp <= frame_timestamp + 1.0e-12:
            apply_pose(
                environment,
                next_step_timestamp,
                joint_offsets,
                translation_world,
            )
            environment.step(compute_visual_forces=False)
            apply_pose(
                environment,
                next_step_timestamp,
                joint_offsets,
                translation_world,
            )
            completed_steps += 1
            next_step_timestamp = first_timestamp + (completed_steps + 1) * dt
        if completed_steps == 0 or (
            first_timestamp + completed_steps * dt
            < frame_timestamp - 1.0e-12
        ):
            apply_pose(
                environment,
                next_step_timestamp,
                joint_offsets,
                translation_world,
            )
            environment.step(compute_visual_forces=False)
            apply_pose(
                environment,
                next_step_timestamp,
                joint_offsets,
                translation_world,
            )
            completed_steps += 1
            next_step_timestamp = first_timestamp + (completed_steps + 1) * dt
        apply_pose(
            environment,
            frame_timestamp,
            joint_offsets,
            translation_world,
        )

        is_evaluation = frame in observations
        check_stability = dense_stability or is_evaluation
        if not check_stability:
            continue
        positions, velocities = local_tissue_tensors(environment)
        dynamics = deformation_metrics(
            positions,
            velocities,
            rest_positions,
            local_tetrahedra,
            rest_inverse,
            dynamic_mask,
        )
        stability["checked_frame_count"] += 1
        stability["finite"] = bool(stability["finite"] and dynamics["finite"])
        stability["inverted_tetrahedra_max"] = max(
            int(stability["inverted_tetrahedra_max"]),
            int(dynamics["inverted_tetrahedra"]),
        )
        stability["minimum_tetrahedron_volume_ratio"] = min(
            float(stability["minimum_tetrahedron_volume_ratio"]),
            float(dynamics["minimum_tetrahedron_volume_ratio"]),
        )
        for key in (
            "maximum_dynamic_speed_m_s",
            "maximum_dynamic_displacement_m",
            "maximum_anchor_drift_m",
        ):
            stability[key] = max(float(stability[key]), float(dynamics[key]))
        if not is_evaluation:
            continue

        position_cpu = positions.detach().cpu().numpy().copy()
        if frame == CALIBRATION_START:
            simulated_reference = {
                track_id: position_cpu[particle_id].astype(
                    np.float64, copy=True
                )
                for track_id, particle_id in track_particle_ids.items()
            }
        observation = observations[frame]
        split = split_for_frame(frame)
        if split is None:
            continue
        accumulator = accumulators[split]
        accumulator["evaluated_frames"].append(frame)

        simulated_top = np.asarray(position_cpu[top_ids], dtype=np.float64)
        surface_errors = cKDTree(simulated_top).query(
            observation["surface_points"], k=1, workers=1
        )[0]
        accumulator["surface_errors"].append(surface_errors)
        accumulator["surface_weights"].append(
            observation["surface_weights"]
        )

        key_errors = np.empty(0, dtype=np.float64)
        if simulated_reference is not None and len(observation["track_ids"]):
            simulated_displacements = np.asarray(
                [
                    position_cpu[track_particle_ids[int(track_id)]]
                    - simulated_reference[int(track_id)]
                    for track_id in observation["track_ids"]
                ],
                dtype=np.float64,
            )
            key_errors = np.linalg.norm(
                simulated_displacements
                - observation["track_displacements"],
                axis=1,
            )
            accumulator["key_errors"].append(key_errors)
            accumulator["key_weights"].append(observation["track_weights"])

        frame_records[str(frame)] = {
            "split": split,
            "surface_robust_rmse_mm": float(
                np.sqrt(
                    np.sum(
                        observation["surface_weights"]
                        * np.minimum(
                            surface_errors, SURFACE_ROBUST_CAP_M
                        )
                        ** 2
                    )
                    / np.sum(observation["surface_weights"])
                )
                * 1000.0
            ),
            "surface_p95_mm": float(
                np.quantile(surface_errors, 0.95) * 1000.0
            ),
            "keypoint_rmse_mm": (
                None
                if not len(key_errors)
                else float(np.sqrt(np.mean(key_errors**2)) * 1000.0)
            ),
            "maximum_dynamic_displacement_mm": float(
                dynamics["maximum_dynamic_displacement_m"] * 1000.0
            ),
            "minimum_tetrahedron_volume_ratio": float(
                dynamics["minimum_tetrahedron_volume_ratio"]
            ),
        }
        if frame in landmark_frames or frame == int(frame_indices[-1]):
            state_hashes[str(frame)] = array_sha256(position_cpu)
            contact_records[str(frame)] = detect_contact(environment)

    wp.synchronize()
    split_metrics = {
        name: summarize_split(accumulator)
        for name, accumulator in accumulators.items()
    }
    physics_penalty_mm = 0.0
    if (
        not stability["finite"]
        or int(stability["inverted_tetrahedra_max"]) > 0
        or float(stability["minimum_tetrahedron_volume_ratio"]) <= 0.0
    ):
        physics_penalty_mm = 1.0e6 + 1000.0 * int(
            stability["inverted_tetrahedra_max"]
        )
    calibration_objective = float(
        split_metrics["calibration"]["objective_mm"] + physics_penalty_mm
    )
    return {
        "schema": "super_tissue_stage_d_candidate_v1",
        "stage": "D_global_material_calibration_candidate",
        "candidate_id": str(candidate["candidate_id"]),
        "passed": bool(
            np.isfinite(calibration_objective)
            and physics_penalty_mm == 0.0
            and stability["maximum_anchor_drift_m"] == 0.0
        ),
        "candidate": candidate_signature(candidate),
        "fixed_parameters": {
            "poisson_ratio": float(environment.super_tissue_poisson_ratio),
            "gravity_m_s2": float(environment.super_tissue_gravity_m_s2),
            "visual_force_iterations": int(
                environment.visual_forces_settings.iterations
            ),
            "residual_mapping_enabled": bool(
                environment.super_tissue_residual_mapping_enabled
            ),
            "online_stiffness_optimization_enabled": bool(
                environment.super_tissue_stiffness_optimization_enabled
            ),
            "contact_and_boundary": "frozen stage-A paper_soft configuration",
            "psm_trajectory": "frozen stage-A absolute kinematic trajectory",
        },
        "objective": {
            "calibration_objective_mm": calibration_objective,
            "physics_penalty_mm": physics_penalty_mm,
            "keypoint_weight": KEYPOINT_LOSS_WEIGHT,
            "surface_weight": SURFACE_LOSS_WEIGHT,
            "keypoint_robust_cap_mm": KEYPOINT_ROBUST_CAP_M * 1000.0,
            "surface_robust_cap_mm": SURFACE_ROBUST_CAP_M * 1000.0,
        },
        "continuous_time_splits_inclusive": {
            name: [start, end]
            for name, (start, end, _stride) in SPLIT_SPECS.items()
        },
        "split_metrics": split_metrics,
        "stability": stability,
        "frame_records": frame_records,
        "state_hashes": state_hashes,
        "landmark_contact_metrics": contact_records,
        "runtime": {
            "device": str(environment.sim.model.device),
            "physics_steps": completed_steps,
            "wall_seconds": time.perf_counter() - started,
        },
        "inputs": {
            "stage_a_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": sha256_file(manifest_path),
            },
            "stage_b_surface_report": {
                "path": str(surface_report_path.resolve()),
                "sha256": sha256_file(surface_report_path),
            },
            "stage_b_keypoints": {
                "path": str(
                    observation_bundle["keypoint_path"].resolve()
                ),
                "sha256": sha256_file(observation_bundle["keypoint_path"]),
            },
        },
    }


def build_worker_context(
    *,
    manifest_path: Path,
    surface_root: Path,
    frame_end: int,
    device: str,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    frame_indices, frame_timestamps, _landmarks = selected_frames(
        manifest, 0, frame_end
    )
    driver_path = Path(manifest["psm_trajectory"]["driver"]["path"])
    if not driver_path.is_absolute():
        driver_path = REPO_ROOT / driver_path
    correction = manifest["psm_trajectory"]["manual_corrections"]
    joint_offsets = np.zeros(7, dtype=np.float64)
    joint_offsets[3] = math.radians(
        correction["psm_self_spin_offset_deg"]
    )
    joint_offsets[6] = math.radians(
        correction["psm_jaw_opening_offset_deg_open_positive"]
    )
    translation_world = camera_to_world_translation(manifest)

    environment = build_environment(
        add_gaussians=True,
        device=device,
        tissue_mode="paper_soft",
        psm_pose_driver_path=driver_path,
        psm_visual_tip_only=False,
    )
    environment.visual_forces_settings.iterations = 0
    environment.frames = None
    if not set_psm_tissue_collisions(environment, True):
        raise RuntimeError("Failed to enable frozen jaw-to-tissue contact")
    apply_pose(
        environment,
        float(frame_timestamps[0]),
        joint_offsets,
        translation_world,
    )
    environment.sim.sync_kinematic_body_interpolation()
    environment.sim.update_gaussian_transforms()
    initial_state = environment.sim.clone_embodied_gaussian_state()
    initial_positions, _velocities = local_tissue_tensors(environment)
    rest_positions = initial_positions.clone()

    handle = environment.super_tissue_soft_handle
    model = environment.sim.model
    local_tetrahedra = (
        wp.to_torch(model.tet_indices).long()[
            handle.tet_start : handle.tet_end
        ]
        - handle.particle_start
    )
    rest_inverse = wp.to_torch(model.tet_poses)[
        handle.tet_start : handle.tet_end
    ]
    inverse_mass = wp.to_torch(model.particle_inv_mass)[
        handle.particle_start : handle.particle_end
    ]
    dynamic_mask = inverse_mass > 0.0
    tissue_asset_path = Path(manifest["tissue"]["asset"]["path"])
    if not tissue_asset_path.is_absolute():
        tissue_asset_path = REPO_ROOT / tissue_asset_path
    observations = load_observations(
        surface_root=surface_root,
        tissue_asset_path=tissue_asset_path,
        manifest=manifest,
        frame_end=frame_end,
    )
    return {
        "environment": environment,
        "initial_state": initial_state,
        "manifest": manifest,
        "frame_indices": frame_indices,
        "frame_timestamps": frame_timestamps,
        "joint_offsets": joint_offsets,
        "translation_world": translation_world,
        "rest_positions": rest_positions,
        "local_tetrahedra": local_tetrahedra,
        "rest_inverse": rest_inverse,
        "dynamic_mask": dynamic_mask,
        "observation_bundle": observations,
    }


def worker_main(args: argparse.Namespace) -> None:
    spec = json.loads(args.worker_batch.read_text(encoding="utf-8"))
    candidates = spec["candidates"]
    if not candidates:
        return
    frame_end = max(int(candidate["frame_end"]) for candidate in candidates)
    cache_dir = args.warp_cache_dir / args.device.replace(":", "_")
    cache_dir.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(cache_dir)
    wp.init()
    wp.set_device(args.device)
    if args.device.startswith("cuda:"):
        torch.cuda.set_device(int(args.device.split(":", 1)[1]))
    context = build_worker_context(
        manifest_path=args.manifest,
        surface_root=args.surface_root,
        frame_end=frame_end,
        device=args.device,
    )
    surface_report_path = args.surface_root / "stage_b_report.json"
    report_dir = args.output_dir / "candidate_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        report_path = report_dir / f"{candidate['candidate_id']}.json"
        if args.resume and report_path.is_file():
            existing = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                existing.get("candidate") == candidate_signature(candidate)
                and existing.get("inputs", {})
                .get("stage_a_manifest", {})
                .get("sha256")
                == sha256_file(args.manifest)
                and existing.get("inputs", {})
                .get("stage_b_surface_report", {})
                .get("sha256")
                == sha256_file(surface_report_path)
            ):
                print(
                    f"[stage-D:{args.device}] reuse "
                    f"{candidate['candidate_id']}",
                    flush=True,
                )
                continue
        print(
            f"[stage-D:{args.device}] start {candidate['candidate_id']} "
            f"E={candidate['young_modulus_pa']:g} Pa "
            f"damping={candidate['velocity_damping_per_second']:g}/s "
            f"end={candidate['frame_end']}",
            flush=True,
        )
        report = evaluate_candidate(
            candidate=candidate,
            manifest_path=args.manifest,
            surface_report_path=surface_report_path,
            **context,
        )
        write_json(report_path, report)
        print(
            f"[stage-D:{args.device}] done {candidate['candidate_id']} "
            f"loss={report['objective']['calibration_objective_mm']:.6f} mm "
            f"time={report['runtime']['wall_seconds']:.1f}s",
            flush=True,
        )


def parameter_key(young_modulus_pa: float, damping: float) -> tuple[float, float]:
    return (round(math.log(young_modulus_pa), 12), round(math.log(damping), 12))


def make_candidate(
    prefix: str,
    index: int,
    young_modulus_pa: float,
    damping: float,
    frame_end: int,
    dense_stability: bool = False,
) -> dict[str, Any]:
    return {
        "candidate_id": f"{prefix}_{index:02d}",
        "young_modulus_pa": float(young_modulus_pa),
        "velocity_damping_per_second": float(damping),
        "frame_end": int(frame_end),
        "dense_stability": bool(dense_stability),
    }


def load_candidate_report(output_dir: Path, candidate_id: str) -> dict[str, Any]:
    path = output_dir / "candidate_reports" / f"{candidate_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing candidate report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_phase(
    *,
    phase: str,
    candidates: list[dict[str, Any]],
    devices: list[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    assignments = [[] for _device in devices]
    for index, candidate in enumerate(candidates):
        assignments[index % len(devices)].append(candidate)
    processes: list[subprocess.Popen[Any]] = []
    for worker_index, (device, assigned) in enumerate(
        zip(devices, assignments, strict=True)
    ):
        if not assigned:
            continue
        spec_path = (
            args.output_dir
            / "worker_specs"
            / f"{phase}_{worker_index:02d}.json"
        )
        write_json(spec_path, {"phase": phase, "candidates": assigned})
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-batch",
            str(spec_path),
            "--device",
            device,
            "--manifest",
            str(args.manifest),
            "--surface-root",
            str(args.surface_root),
            "--output-dir",
            str(args.output_dir),
            "--warp-cache-dir",
            str(args.warp_cache_dir),
        ]
        command.append("--resume" if args.resume else "--no-resume")
        processes.append(
            subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                env=os.environ.copy(),
            )
        )
    return_codes = [process.wait() for process in processes]
    if any(code != 0 for code in return_codes):
        raise RuntimeError(
            f"Stage-D {phase} worker failure codes: {return_codes}"
        )
    return [
        load_candidate_report(args.output_dir, candidate["candidate_id"])
        for candidate in candidates
    ]


def ranked_passed(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    passed = [report for report in reports if report.get("passed", False)]
    return sorted(
        passed,
        key=lambda report: float(
            report["objective"]["calibration_objective_mm"]
        ),
    )


def controller_main(args: argparse.Namespace) -> None:
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("At least one worker device is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    coarse_parameters = [
        (young, damping)
        for young in (20.0, 100.0, 500.0, 2500.0)
        for damping in (5.0, 20.0, 80.0)
    ]
    coarse_parameters.append((50.0, 30.0))
    coarse = [
        make_candidate(
            "coarse",
            index,
            young,
            damping,
            CALIBRATION_END,
        )
        for index, (young, damping) in enumerate(coarse_parameters)
    ]
    print(
        f"[stage-D] coarse search: {len(coarse)} candidates on {devices}",
        flush=True,
    )
    coarse_reports = run_phase(
        phase="coarse",
        candidates=coarse,
        devices=devices,
        args=args,
    )
    coarse_ranked = ranked_passed(coarse_reports)
    if not coarse_ranked:
        raise RuntimeError("No physically valid coarse candidate")
    coarse_best = coarse_ranked[0]
    best_young = float(coarse_best["candidate"]["young_modulus_pa"])
    best_damping = float(
        coarse_best["candidate"]["velocity_damping_per_second"]
    )

    prior_keys = {
        parameter_key(
            float(report["candidate"]["young_modulus_pa"]),
            float(report["candidate"]["velocity_damping_per_second"]),
        )
        for report in coarse_reports
    }
    local_parameters: list[tuple[float, float]] = []
    for young_factor in (0.63, 1.0, 1.59):
        for damping_factor in (0.63, 1.0, 1.59):
            young = float(np.clip(best_young * young_factor, 10.0, 5000.0))
            damping = float(
                np.clip(best_damping * damping_factor, 2.0, 120.0)
            )
            key = parameter_key(young, damping)
            if key not in prior_keys:
                prior_keys.add(key)
                local_parameters.append((young, damping))
    local = [
        make_candidate(
            "local",
            index,
            young,
            damping,
            CALIBRATION_END,
        )
        for index, (young, damping) in enumerate(local_parameters)
    ]
    print(
        f"[stage-D] local search: {len(local)} candidates around "
        f"E={best_young:g}, damping={best_damping:g}",
        flush=True,
    )
    local_reports = run_phase(
        phase="local",
        candidates=local,
        devices=devices,
        args=args,
    )
    train_ranked = ranked_passed(coarse_reports + local_reports)
    if len(train_ranked) < 3:
        raise RuntimeError("Fewer than three valid calibration candidates")

    finalists = [
        make_candidate(
            "final",
            index,
            float(report["candidate"]["young_modulus_pa"]),
            float(report["candidate"]["velocity_damping_per_second"]),
            TEST_END,
        )
        for index, report in enumerate(train_ranked[:3])
    ]
    print(
        "[stage-D] full future validation/test: 3 calibration-selected finalists",
        flush=True,
    )
    finalist_reports = run_phase(
        phase="final",
        candidates=finalists,
        devices=devices,
        args=args,
    )
    finalist_ranked = ranked_passed(finalist_reports)
    if not finalist_ranked:
        raise RuntimeError("No physically valid full-sequence finalist")
    selected = finalist_ranked[0]
    selected_young = float(selected["candidate"]["young_modulus_pa"])
    selected_damping = float(
        selected["candidate"]["velocity_damping_per_second"]
    )

    verification = [
        make_candidate(
            "verify",
            index,
            selected_young,
            selected_damping,
            TEST_END,
            dense_stability=True,
        )
        for index in range(2)
    ]
    print(
        "[stage-D] dense deterministic verification: 2 full selected replays",
        flush=True,
    )
    verification_reports = run_phase(
        phase="verification",
        candidates=verification,
        devices=[devices[0]],
        args=args,
    )
    verify_a, verify_b = verification_reports
    exact_objective = (
        verify_a["objective"]["calibration_objective_mm"]
        == verify_b["objective"]["calibration_objective_mm"]
    )
    exact_hashes = verify_a["state_hashes"] == verify_b["state_hashes"]
    dense_stable = all(
        report["passed"]
        and report["stability"]["finite"]
        and report["stability"]["inverted_tetrahedra_max"] == 0
        and report["stability"]["minimum_tetrahedron_volume_ratio"] > 0.0
        and report["stability"]["checked_frame_count"] == TEST_END + 1
        for report in verification_reports
    )

    baseline = next(
        report
        for report in coarse_reports
        if math.isclose(
            float(report["candidate"]["young_modulus_pa"]), 50.0
        )
        and math.isclose(
            float(
                report["candidate"]["velocity_damping_per_second"]
            ),
            30.0,
        )
    )
    selected_loss = float(
        verify_a["objective"]["calibration_objective_mm"]
    )
    baseline_loss = float(
        baseline["objective"]["calibration_objective_mm"]
    )
    improvement_fraction = (
        (baseline_loss - selected_loss) / baseline_loss
        if baseline_loss > 0.0
        else 0.0
    )
    near_optimal = [
        {
            "young_modulus_pa": float(
                report["candidate"]["young_modulus_pa"]
            ),
            "velocity_damping_per_second": float(
                report["candidate"]["velocity_damping_per_second"]
            ),
            "calibration_objective_mm": float(
                report["objective"]["calibration_objective_mm"]
            ),
        }
        for report in train_ranked
        if float(report["objective"]["calibration_objective_mm"])
        <= selected_loss * 1.10
    ]
    passed = bool(
        dense_stable
        and exact_objective
        and exact_hashes
        and selected_loss <= baseline_loss
    )
    material_config = {
        "schema": "super_tissue_calibrated_material_v1",
        "stage": "D",
        "passed": passed,
        "young_modulus_pa": selected_young,
        "velocity_damping_per_second": selected_damping,
        "poisson_ratio": float(
            verify_a["fixed_parameters"]["poisson_ratio"]
        ),
        "gravity_m_s2": float(
            verify_a["fixed_parameters"]["gravity_m_s2"]
        ),
        "source_report": "stage_d_calibration_report.json",
        "usage": (
            "Apply these two calibrated scalars to paper_soft before replay; "
            "keep all other stage-A settings frozen."
        ),
    }
    write_json(args.output_dir / "calibrated_material.json", material_config)

    report = {
        "schema": "super_tissue_stage_d_material_calibration_report_v1",
        "stage": "D_global_material_calibration",
        "passed": passed,
        "method": {
            "paper": "arXiv:2309.11656",
            "search": (
                "logarithmic global grid followed by local logarithmic grid"
            ),
            "selection_rule": (
                "select only by continuous calibration interval; validation "
                "and test remain future-only reports"
            ),
            "optimized_parameters": [
                "global Young's modulus",
                "velocity damping",
            ],
            "loss": (
                "0.70 robust first-contact keypoint displacement RMSE + "
                "0.30 motion-weighted visible-surface one-way RMSE + "
                "hard physical-validity penalty"
            ),
            "visual_forces": "disabled",
            "residual_mapping": "disabled",
        },
        "continuous_time_splits_inclusive": {
            name: [start, end]
            for name, (start, end, _stride) in SPLIT_SPECS.items()
        },
        "candidate_counts": {
            "coarse": len(coarse_reports),
            "local": len(local_reports),
            "full_sequence_finalists": len(finalist_reports),
            "dense_selected_verification": len(verification_reports),
            "total_replays": (
                len(coarse_reports)
                + len(local_reports)
                + len(finalist_reports)
                + len(verification_reports)
            ),
        },
        "selected_material": material_config,
        "selected_metrics": verify_a["split_metrics"],
        "baseline": {
            "young_modulus_pa": 50.0,
            "velocity_damping_per_second": 30.0,
            "calibration_objective_mm": baseline_loss,
        },
        "calibration_improvement_fraction_vs_baseline": improvement_fraction,
        "near_optimal_within_10_percent": near_optimal,
        "determinism": {
            "exact_calibration_objective": exact_objective,
            "exact_landmark_and_final_particle_hashes": exact_hashes,
            "run_a_state_hashes": verify_a["state_hashes"],
            "run_b_state_hashes": verify_b["state_hashes"],
        },
        "dense_physics_validation": {
            "passed": dense_stable,
            "run_a": verify_a["stability"],
            "run_b": verify_b["stability"],
        },
        "rankings": {
            "calibration_search": [
                {
                    "candidate_id": report["candidate_id"],
                    **report["candidate"],
                    "calibration_objective_mm": report["objective"][
                        "calibration_objective_mm"
                    ],
                }
                for report in train_ranked
            ],
            "full_sequence_finalists": [
                {
                    "candidate_id": report["candidate_id"],
                    **report["candidate"],
                    "calibration_objective_mm": report["objective"][
                        "calibration_objective_mm"
                    ],
                    "validation_objective_mm": report["split_metrics"][
                        "validation"
                    ]["objective_mm"],
                    "test_objective_mm": report["split_metrics"]["test"][
                        "objective_mm"
                    ],
                }
                for report in finalist_ranked
            ],
        },
        "inputs": verify_a["inputs"],
        "outputs": {
            "calibrated_material": str(
                (args.output_dir / "calibrated_material.json").resolve()
            ),
            "candidate_reports": str(
                (args.output_dir / "candidate_reports").resolve()
            ),
        },
        "runtime": {
            "devices": devices,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    write_json(args.output_dir / "stage_d_calibration_report.json", report)
    print(
        f"[stage-D] {'PASS' if passed else 'FAIL'} selected "
        f"E={selected_young:g} Pa damping={selected_damping:g}/s; "
        f"calibration={selected_loss:.6f} mm, "
        f"validation={verify_a['split_metrics']['validation']['objective_mm']:.6f} mm, "
        f"test={verify_a['split_metrics']['test']['objective_mm']:.6f} mm",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.worker_batch is not None:
        worker_main(args)
    else:
        controller_main(args)


if __name__ == "__main__":
    main()
