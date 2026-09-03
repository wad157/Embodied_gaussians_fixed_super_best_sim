#!/usr/bin/env python3
"""Deterministic physics-only replay for SUPER tissue calibration stage C.

The replay consumes the stage-A manifest as its only calibration contract.  It
keeps the frozen absolute PSM trajectory and manual correction, advances the
soft tissue on a fixed physics clock, and never evaluates visual forces or
observation residuals.

The first run stores the complete tissue-particle trajectory.  A second run is
compared frame-by-frame against it, including contact statistics, Gaussian
skinning state, and landmark renders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from embodied_environments.super_embodied.super_embodied import (  # noqa: E402
    apply_psm_lnd_pose,
    build_environment,
    set_psm_tissue_collisions,
)


DEFAULT_MANIFEST = (
    REPO_ROOT
    / "data/super/tissue_calibration_v1/stage_a_frozen_manifest.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "data/super/tissue_calibration_v1/stage_c_offline_replay"
)

METRIC_INT_KEYS = (
    "contact_count",
    "jaw_triangle_contact_count",
    "top_barrier_contact_count",
    "top_support_entry_count",
    "persistent_grip_active",
    "persistent_grip_particle_count",
)
METRIC_FLOAT_KEYS = (
    "minimum_signed_distance_m",
    "top_minimum_signed_distance_m",
    "maximum_penetration_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run and repeat the manifest-driven, no-visual-force SUPER tissue "
            "replay."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--render-width", type=int, default=480)
    parser.add_argument("--render-height", type=int, default=270)
    parser.add_argument(
        "--no-full-trajectory",
        action="store_true",
        help="Do not store the full particle/Gaussian arrays (for short tests).",
    )
    parser.add_argument(
        "--warp-cache-dir",
        type=Path,
        default=Path("/tmp/warp-super-tissue-stage-c"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


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


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("passed", False):
        raise RuntimeError("Stage-A manifest is not in a passed state")
    if manifest.get("stage") not in {"A", "A_frozen_inputs"}:
        raise ValueError(f"Expected stage-A manifest, got {manifest.get('stage')}")
    return manifest


def resolve_manifest_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def camera_to_world_translation(
    manifest: dict[str, Any],
) -> np.ndarray:
    correction = manifest["psm_trajectory"]["manual_corrections"]
    translation_camera = (
        np.asarray(
            [
                correction["manual_image_x_mm_right_positive"],
                correction["manual_image_y_mm_down_positive"],
                correction["manual_camera_z_mm_far_positive"],
            ],
            dtype=np.float64,
        )
        / 1000.0
    )
    # cameras.json stores a Blender-convention X_WC.  The GUI first converts
    # it to OpenCV (+x right, +y down, +z far) and then maps the manual camera
    # translation into world coordinates.
    x_world_camera_blender = np.asarray(
        manifest["camera"]["X_world_camera"]["stereo_left"],
        dtype=np.float64,
    )
    blender_to_opencv = np.diag([1.0, -1.0, -1.0, 1.0])
    x_world_camera_opencv = x_world_camera_blender @ blender_to_opencv
    return x_world_camera_opencv[:3, :3] @ translation_camera


def render_camera(
    manifest: dict[str, Any],
    device: torch.device,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_width, source_height = manifest["camera"]["image_size_wh"]
    if width < 1 or height < 1:
        raise ValueError("Render dimensions must be positive")
    x_world_camera_blender = np.asarray(
        manifest["camera"]["X_world_camera"]["stereo_left"],
        dtype=np.float64,
    )
    blender_to_opencv = np.diag([1.0, -1.0, -1.0, 1.0])
    x_camera_world_opencv = np.linalg.inv(
        x_world_camera_blender @ blender_to_opencv
    )
    intrinsic = np.asarray(
        manifest["camera"]["K_left_rect"], dtype=np.float64
    ).copy()
    intrinsic[0] *= float(width) / float(source_width)
    intrinsic[1] *= float(height) / float(source_height)
    return (
        torch.as_tensor(
            x_camera_world_opencv[None], dtype=torch.float32, device=device
        ),
        torch.as_tensor(intrinsic[None], dtype=torch.float32, device=device),
    )


def selected_frames(
    manifest: dict[str, Any], frame_start: int, frame_end: int | None
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    metadata_path = resolve_manifest_path(
        manifest["sequence"]["offline_stereo"]["left_metadata"]["path"]
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    all_timestamps = np.asarray(metadata["timestamps"], dtype=np.float64)
    frozen_start, frozen_end = manifest["sequence"]["offline_stereo"][
        "playback_left_frame_range_inclusive"
    ]
    actual_start = max(int(frame_start), int(frozen_start))
    actual_end = int(frozen_end) if frame_end is None else min(
        int(frame_end), int(frozen_end)
    )
    if actual_start < frozen_start or actual_end < actual_start:
        raise ValueError(
            f"Invalid frame range {actual_start}..{actual_end}; frozen range "
            f"is {frozen_start}..{frozen_end}"
        )
    frame_indices = np.arange(actual_start, actual_end + 1, dtype=np.int32)
    timestamps = all_timestamps[frame_indices]
    landmarks = {
        name: int(entry["left_frame"])
        for name, entry in manifest["sequence"]["landmarks"].items()
        if actual_start <= int(entry["left_frame"]) <= actual_end
    }
    return frame_indices, timestamps, landmarks


def pose_state_indices(
    pose_timestamps: np.ndarray, frame_timestamps: np.ndarray
) -> np.ndarray:
    indices = (
        np.searchsorted(pose_timestamps, frame_timestamps, side="right") - 1
    )
    return np.clip(indices, 0, len(pose_timestamps) - 1).astype(np.int32)


def apply_pose(
    environment: Any,
    pose_timestamp_s: float,
    joint_offsets: np.ndarray,
    translation_world: np.ndarray,
) -> int:
    state_index = int(
        np.searchsorted(
            environment.super_psm_lnd_timestamps,
            pose_timestamp_s,
            side="right",
        )
        - 1
    )
    state_index = max(
        0, min(state_index, len(environment.super_psm_lnd_timestamps) - 1)
    )
    apply_psm_lnd_pose(
        environment,
        state_index,
        joint_offsets=joint_offsets,
        translation_offset=translation_world,
        update_gaussians=False,
    )
    return state_index


def detect_contact(environment: Any) -> dict[str, Any]:
    settings = environment.physics_settings
    projector = environment.sim.triangle_skin_contact_projector
    projector.detect(
        environment.sim.model,
        environment.sim.state_0,
        settings.dt / settings.substeps,
        contact_margin_m=settings.triangle_skin_contact_margin_m,
        query_distance_m=settings.triangle_skin_query_distance_m,
        ccd_velocity_scale=0.0,
        friction_coefficient=0.0,
        relaxation=1.0,
    )
    return projector.metrics()


def local_tissue_tensors(environment: Any) -> tuple[torch.Tensor, torch.Tensor]:
    handle = environment.super_tissue_soft_handle
    positions = wp.to_torch(environment.sim.state_0.particle_q)[
        handle.particle_start : handle.particle_end
    ]
    velocities = wp.to_torch(environment.sim.state_0.particle_qd)[
        handle.particle_start : handle.particle_end
    ]
    return positions, velocities


def deformation_metrics(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    rest_positions: torch.Tensor,
    local_tetrahedra: torch.Tensor,
    rest_inverse: torch.Tensor,
    dynamic_mask: torch.Tensor,
) -> dict[str, float | int | bool]:
    deformation = torch.stack(
        (
            positions[local_tetrahedra[:, 1]]
            - positions[local_tetrahedra[:, 0]],
            positions[local_tetrahedra[:, 2]]
            - positions[local_tetrahedra[:, 0]],
            positions[local_tetrahedra[:, 3]]
            - positions[local_tetrahedra[:, 0]],
        ),
        dim=2,
    )
    volume_ratio = torch.linalg.det(deformation @ rest_inverse)
    displacement = torch.linalg.vector_norm(
        positions - rest_positions, dim=1
    )
    speed = torch.linalg.vector_norm(velocities, dim=1)
    anchor_mask = ~dynamic_mask
    return {
        "finite": bool(
            torch.isfinite(positions).all()
            and torch.isfinite(velocities).all()
            and torch.isfinite(volume_ratio).all()
        ),
        "inverted_tetrahedra": int((volume_ratio <= 0.0).sum().item()),
        "minimum_tetrahedron_volume_ratio": float(volume_ratio.min().item()),
        "maximum_dynamic_speed_m_s": float(speed[dynamic_mask].max().item()),
        "maximum_dynamic_displacement_m": float(
            displacement[dynamic_mask].max().item()
        ),
        "p99_dynamic_displacement_m": float(
            torch.quantile(displacement[dynamic_mask], 0.99).item()
        ),
        "maximum_anchor_drift_m": float(
            displacement[anchor_mask].max().item()
        ),
    }


def allocate_metric_arrays(frame_count: int) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "physics_step_index": np.zeros(frame_count, dtype=np.int32),
        "physics_sample_lag_s": np.zeros(frame_count, dtype=np.float64),
        "pose_state_index": np.zeros(frame_count, dtype=np.int32),
        "inverted_tetrahedra": np.zeros(frame_count, dtype=np.int32),
        "finite": np.zeros(frame_count, dtype=bool),
        "minimum_tetrahedron_volume_ratio": np.zeros(
            frame_count, dtype=np.float32
        ),
        "maximum_dynamic_speed_m_s": np.zeros(frame_count, dtype=np.float32),
        "maximum_dynamic_displacement_m": np.zeros(
            frame_count, dtype=np.float32
        ),
        "p99_dynamic_displacement_m": np.zeros(frame_count, dtype=np.float32),
        "maximum_anchor_drift_m": np.zeros(frame_count, dtype=np.float32),
        "particle_sha256": np.empty(frame_count, dtype="U64"),
        "gaussian_means_sha256": np.empty(frame_count, dtype="U64"),
        "gaussian_quats_sha256": np.empty(frame_count, dtype="U64"),
    }
    for key in METRIC_INT_KEYS:
        arrays[key] = np.zeros(frame_count, dtype=np.int32)
    for key in METRIC_FLOAT_KEYS:
        arrays[key] = np.zeros(frame_count, dtype=np.float32)
    return arrays


def create_memmaps(
    output_dir: Path,
    frame_count: int,
    particle_count: int,
    gaussian_count: int,
    enabled: bool,
) -> dict[str, np.memmap]:
    if not enabled:
        return {}
    return {
        "particle_positions": np.lib.format.open_memmap(
            output_dir / "particle_positions.npy",
            mode="w+",
            dtype=np.float32,
            shape=(frame_count, particle_count, 3),
        ),
        "gaussian_means": np.lib.format.open_memmap(
            output_dir / "gaussian_means.npy",
            mode="w+",
            dtype=np.float32,
            shape=(frame_count, gaussian_count, 3),
        ),
        "gaussian_quats": np.lib.format.open_memmap(
            output_dir / "gaussian_quats.npy",
            mode="w+",
            dtype=np.float32,
            shape=(frame_count, gaussian_count, 4),
        ),
    }


def save_landmark_render(
    output_dir: Path,
    name: str,
    frame_index: int,
    rgb_float: np.ndarray,
) -> tuple[str, str]:
    render_dir = output_dir / "landmark_renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{frame_index:04d}_{name}"
    float_path = render_dir / f"{stem}.npy"
    png_path = render_dir / f"{stem}.png"
    np.save(float_path, rgb_float.astype(np.float32, copy=False))
    rgb_u8 = np.clip(np.rint(rgb_float * 255.0), 0, 255).astype(np.uint8)
    if not cv2.imwrite(
        str(png_path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    ):
        raise RuntimeError(f"Failed to write render {png_path}")
    return array_sha256(rgb_float), array_sha256(rgb_u8)


def run_replay(
    *,
    environment: Any,
    initial_state: Any,
    manifest: dict[str, Any],
    output_dir: Path,
    frame_indices: np.ndarray,
    frame_timestamps: np.ndarray,
    landmark_by_frame: dict[int, str],
    joint_offsets: np.ndarray,
    translation_world: np.ndarray,
    rest_positions: torch.Tensor,
    local_tetrahedra: torch.Tensor,
    rest_inverse: torch.Tensor,
    dynamic_mask: torch.Tensor,
    render_view: torch.Tensor,
    render_intrinsic: torch.Tensor,
    render_width: int,
    render_height: int,
    run_index: int,
    primary_metrics: dict[str, np.ndarray] | None,
    primary_memmaps: dict[str, np.memmap] | None,
    save_full_trajectory: bool,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.memmap],
    dict[str, dict[str, Any]],
    dict[str, float | int],
]:
    environment.sim.copy_embodied_gaussian_state(initial_state)
    environment.sim.reset()
    environment.sim.clear_soft_visual_force_cache()
    environment.sim.state_0.clear_forces()
    apply_pose(
        environment,
        float(frame_timestamps[0]),
        joint_offsets,
        translation_world,
    )
    environment.sim.sync_kinematic_body_interpolation()
    environment.sim.update_gaussian_transforms()

    handle = environment.super_tissue_soft_handle
    gaussian_count = int(environment.sim.gaussian_model.num_gaussians)
    metrics = allocate_metric_arrays(len(frame_indices))
    memmaps = create_memmaps(
        output_dir,
        len(frame_indices),
        handle.particle_end - handle.particle_start,
        gaussian_count,
        enabled=save_full_trajectory and run_index == 0,
    )
    if run_index > 0 and primary_memmaps is None:
        raise ValueError("Repeated replay requires primary trajectory arrays")

    comparison = {
        "maximum_particle_position_abs_error_m": 0.0,
        "maximum_gaussian_mean_abs_error_m": 0.0,
        "maximum_gaussian_quaternion_abs_error": 0.0,
        "maximum_contact_float_abs_error": 0.0,
        "contact_integer_mismatch_count": 0,
        "particle_hash_mismatch_count": 0,
        "gaussian_hash_mismatch_count": 0,
        "render_float_hash_mismatch_count": 0,
        "render_uint8_hash_mismatch_count": 0,
    }
    render_records: dict[str, dict[str, Any]] = {}

    dt = float(environment.physics_settings.dt)
    first_timestamp = float(frame_timestamps[0])
    completed_steps = 0
    next_step_timestamp = first_timestamp + dt
    started = time.perf_counter()

    for local_frame, (frame_index, frame_timestamp) in enumerate(
        zip(frame_indices, frame_timestamps, strict=True)
    ):
        frame_timestamp = float(frame_timestamp)
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

        # Advance one fixed step past the observation if needed.  This makes
        # every sample causal and bounds the timestamp lag by one physics dt.
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

        pose_index = apply_pose(
            environment,
            frame_timestamp,
            joint_offsets,
            translation_world,
        )
        environment.sim.update_gaussian_transforms()
        contact = detect_contact(environment)
        positions, velocities = local_tissue_tensors(environment)
        dynamics = deformation_metrics(
            positions,
            velocities,
            rest_positions,
            local_tetrahedra,
            rest_inverse,
            dynamic_mask,
        )
        particle_cpu = positions.detach().cpu().numpy().copy()
        gaussian_means_cpu = (
            environment.sim.gaussian_state.means.detach().cpu().numpy().copy()
        )
        gaussian_quats_cpu = (
            environment.sim.gaussian_state.quats.detach().cpu().numpy().copy()
        )

        metrics["physics_step_index"][local_frame] = completed_steps
        metrics["physics_sample_lag_s"][local_frame] = (
            first_timestamp + completed_steps * dt - frame_timestamp
        )
        metrics["pose_state_index"][local_frame] = pose_index
        for key in METRIC_INT_KEYS:
            metrics[key][local_frame] = int(contact[key])
        for key in METRIC_FLOAT_KEYS:
            metrics[key][local_frame] = float(contact[key])
        for key, value in dynamics.items():
            metrics[key][local_frame] = value
        metrics["particle_sha256"][local_frame] = array_sha256(particle_cpu)
        metrics["gaussian_means_sha256"][local_frame] = array_sha256(
            gaussian_means_cpu
        )
        metrics["gaussian_quats_sha256"][local_frame] = array_sha256(
            gaussian_quats_cpu
        )

        if run_index == 0:
            if memmaps:
                memmaps["particle_positions"][local_frame] = particle_cpu
                memmaps["gaussian_means"][local_frame] = gaussian_means_cpu
                memmaps["gaussian_quats"][local_frame] = gaussian_quats_cpu
        else:
            if primary_metrics is None:
                raise ValueError("Missing primary metrics for repeated replay")
            if primary_memmaps:
                comparison["maximum_particle_position_abs_error_m"] = max(
                    float(
                        comparison[
                            "maximum_particle_position_abs_error_m"
                        ]
                    ),
                    float(
                        np.max(
                            np.abs(
                                particle_cpu
                                - primary_memmaps["particle_positions"][
                                    local_frame
                                ]
                            )
                        )
                    ),
                )
                comparison["maximum_gaussian_mean_abs_error_m"] = max(
                    float(
                        comparison["maximum_gaussian_mean_abs_error_m"]
                    ),
                    float(
                        np.max(
                            np.abs(
                                gaussian_means_cpu
                                - primary_memmaps["gaussian_means"][
                                    local_frame
                                ]
                            )
                        )
                    ),
                )
                comparison[
                    "maximum_gaussian_quaternion_abs_error"
                ] = max(
                    float(
                        comparison[
                            "maximum_gaussian_quaternion_abs_error"
                        ]
                    ),
                    float(
                        np.max(
                            np.abs(
                                gaussian_quats_cpu
                                - primary_memmaps["gaussian_quats"][
                                    local_frame
                                ]
                            )
                        )
                    ),
                )
            comparison["particle_hash_mismatch_count"] += int(
                metrics["particle_sha256"][local_frame]
                != primary_metrics["particle_sha256"][local_frame]
            )
            comparison["gaussian_hash_mismatch_count"] += int(
                metrics["gaussian_means_sha256"][local_frame]
                != primary_metrics["gaussian_means_sha256"][local_frame]
                or metrics["gaussian_quats_sha256"][local_frame]
                != primary_metrics["gaussian_quats_sha256"][local_frame]
            )
            for key in METRIC_INT_KEYS:
                comparison["contact_integer_mismatch_count"] += int(
                    metrics[key][local_frame]
                    != primary_metrics[key][local_frame]
                )
            for key in METRIC_FLOAT_KEYS:
                comparison["maximum_contact_float_abs_error"] = max(
                    float(comparison["maximum_contact_float_abs_error"]),
                    abs(
                        float(metrics[key][local_frame])
                        - float(primary_metrics[key][local_frame])
                    ),
                )

        landmark_name = landmark_by_frame.get(int(frame_index))
        if landmark_name is not None:
            background = torch.zeros(
                3,
                dtype=torch.float32,
                device=environment.sim.gaussian_state.means.device,
            )
            render, _alpha, _info = environment.sim.render_gaussians(
                environment.sim.gaussian_state,
                render_view,
                render_intrinsic,
                render_width,
                render_height,
                background,
            )
            wp.synchronize()
            render_cpu = render[0].detach().cpu().numpy().copy()
            if run_index == 0:
                float_hash, uint8_hash = save_landmark_render(
                    output_dir,
                    landmark_name,
                    int(frame_index),
                    render_cpu,
                )
            else:
                float_path = (
                    output_dir
                    / "landmark_renders"
                    / f"{int(frame_index):04d}_{landmark_name}.npy"
                )
                reference_render = np.load(float_path)
                float_hash = array_sha256(render_cpu)
                uint8_hash = array_sha256(
                    np.clip(np.rint(render_cpu * 255.0), 0, 255).astype(
                        np.uint8
                    )
                )
                reference_float_hash = array_sha256(reference_render)
                reference_uint8_hash = array_sha256(
                    np.clip(
                        np.rint(reference_render * 255.0), 0, 255
                    ).astype(np.uint8)
                )
                comparison["render_float_hash_mismatch_count"] += int(
                    float_hash != reference_float_hash
                )
                comparison["render_uint8_hash_mismatch_count"] += int(
                    uint8_hash != reference_uint8_hash
                )
            render_records[landmark_name] = {
                "left_frame": int(frame_index),
                "float32_sha256": float_hash,
                "uint8_sha256": uint8_hash,
            }

        if (
            local_frame == 0
            or local_frame + 1 == len(frame_indices)
            or (local_frame + 1) % 100 == 0
        ):
            print(
                f"[stage C run {run_index + 1}] "
                f"{local_frame + 1}/{len(frame_indices)} frames; "
                f"physics_steps={completed_steps}; "
                f"contacts={int(metrics['contact_count'][local_frame])}; "
                f"max_disp={float(metrics['maximum_dynamic_displacement_m'][local_frame]) * 1e3:.3f} mm"
            )

    wp.synchronize()
    elapsed = time.perf_counter() - started
    for memmap in memmaps.values():
        memmap.flush()
    comparison["physics_steps"] = completed_steps
    comparison["elapsed_seconds"] = elapsed
    comparison["milliseconds_per_physics_step"] = (
        elapsed / max(completed_steps, 1) * 1000.0
    )
    return metrics, memmaps, render_records, comparison


def summarize_metrics(
    metrics: dict[str, np.ndarray],
    frame_indices: np.ndarray,
    landmarks: dict[str, int],
) -> dict[str, Any]:
    contact_active = metrics["contact_count"] > 0
    active_indices = frame_indices[contact_active]
    landmark_metrics: dict[str, Any] = {}
    lookup = {int(frame): index for index, frame in enumerate(frame_indices)}
    for name, frame in landmarks.items():
        index = lookup[frame]
        landmark_metrics[name] = {
            "left_frame": frame,
            "pose_state_index": int(metrics["pose_state_index"][index]),
            "physics_step_index": int(metrics["physics_step_index"][index]),
            "contact_count": int(metrics["contact_count"][index]),
            "jaw_triangle_contact_count": int(
                metrics["jaw_triangle_contact_count"][index]
            ),
            "persistent_grip_active": bool(
                metrics["persistent_grip_active"][index]
            ),
            "persistent_grip_particle_count": int(
                metrics["persistent_grip_particle_count"][index]
            ),
            "minimum_signed_distance_m": float(
                metrics["minimum_signed_distance_m"][index]
            ),
            "maximum_dynamic_displacement_m": float(
                metrics["maximum_dynamic_displacement_m"][index]
            ),
            "minimum_tetrahedron_volume_ratio": float(
                metrics["minimum_tetrahedron_volume_ratio"][index]
            ),
        }
    return {
        "frames": len(frame_indices),
        "physics_steps": int(metrics["physics_step_index"][-1]),
        "physics_sample_lag_ms": {
            "minimum": float(metrics["physics_sample_lag_s"].min() * 1000.0),
            "p50": float(
                np.quantile(metrics["physics_sample_lag_s"], 0.50) * 1000.0
            ),
            "p95": float(
                np.quantile(metrics["physics_sample_lag_s"], 0.95) * 1000.0
            ),
            "maximum": float(metrics["physics_sample_lag_s"].max() * 1000.0),
        },
        "contact_active_frame_count": int(contact_active.sum()),
        "first_contact_active_left_frame": (
            int(active_indices[0]) if len(active_indices) else None
        ),
        "last_contact_active_left_frame": (
            int(active_indices[-1]) if len(active_indices) else None
        ),
        "contact_count_maximum": int(metrics["contact_count"].max()),
        "jaw_triangle_contact_count_maximum": int(
            metrics["jaw_triangle_contact_count"].max()
        ),
        "top_barrier_contact_count_maximum": int(
            metrics["top_barrier_contact_count"].max()
        ),
        "persistent_grip_active_frame_count": int(
            np.count_nonzero(metrics["persistent_grip_active"])
        ),
        "persistent_grip_particle_count_maximum": int(
            metrics["persistent_grip_particle_count"].max()
        ),
        "minimum_signed_distance_m": float(
            metrics["minimum_signed_distance_m"].min()
        ),
        "maximum_penetration_m": float(
            metrics["maximum_penetration_m"].max()
        ),
        "all_finite": bool(metrics["finite"].all()),
        "maximum_inverted_tetrahedra": int(
            metrics["inverted_tetrahedra"].max()
        ),
        "minimum_tetrahedron_volume_ratio": float(
            metrics["minimum_tetrahedron_volume_ratio"].min()
        ),
        "maximum_dynamic_speed_m_s": float(
            metrics["maximum_dynamic_speed_m_s"].max()
        ),
        "maximum_dynamic_displacement_m": float(
            metrics["maximum_dynamic_displacement_m"].max()
        ),
        "maximum_anchor_drift_m": float(
            metrics["maximum_anchor_drift_m"].max()
        ),
        "landmarks": landmark_metrics,
    }


def main() -> None:
    args = parse_args()
    if args.runs < 2:
        raise ValueError("--runs must be at least 2 for the stage-C gate")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.warp_cache_dir.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(args.warp_cache_dir)
    wp.init()

    manifest = load_manifest(args.manifest)
    frame_indices, frame_timestamps, landmarks = selected_frames(
        manifest, args.frame_start, args.frame_end
    )
    landmark_by_frame = {
        frame: name for name, frame in landmarks.items()
    }
    driver_path = resolve_manifest_path(
        manifest["psm_trajectory"]["driver"]["path"]
    )
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
        device=args.device,
        tissue_mode="paper_soft",
        psm_pose_driver_path=driver_path,
        psm_visual_tip_only=False,
    )
    environment.visual_forces_settings.iterations = 0
    environment.frames = None
    if not set_psm_tissue_collisions(environment, True):
        raise RuntimeError("Failed to enable jaw-to-tissue contact")

    expected_pose_indices = pose_state_indices(
        np.asarray(environment.super_psm_lnd_timestamps, dtype=np.float64),
        frame_timestamps,
    )
    apply_pose(
        environment,
        float(frame_timestamps[0]),
        joint_offsets,
        translation_world,
    )
    environment.sim.sync_kinematic_body_interpolation()
    environment.sim.update_gaussian_transforms()
    initial_state = environment.sim.clone_embodied_gaussian_state()
    initial_positions, _initial_velocities = local_tissue_tensors(environment)
    rest_positions = initial_positions.clone()

    handle = environment.super_tissue_soft_handle
    model = environment.sim.model
    all_tetrahedra = wp.to_torch(model.tet_indices).long()
    local_tetrahedra = (
        all_tetrahedra[handle.tet_start : handle.tet_end]
        - handle.particle_start
    )
    rest_inverse = wp.to_torch(model.tet_poses)[
        handle.tet_start : handle.tet_end
    ]
    inverse_mass = wp.to_torch(model.particle_inv_mass)[
        handle.particle_start : handle.particle_end
    ]
    dynamic_mask = inverse_mass > 0.0
    render_view, render_intrinsic = render_camera(
        manifest,
        environment.sim.gaussian_state.means.device,
        args.render_width,
        args.render_height,
    )

    # Capture all kernels before the measured/repeated runs, then restore the
    # exact initial physical and Gaussian state.
    environment.step(compute_visual_forces=False)
    wp.synchronize()
    environment.sim.copy_embodied_gaussian_state(initial_state)
    environment.sim.reset()
    apply_pose(
        environment,
        float(frame_timestamps[0]),
        joint_offsets,
        translation_world,
    )
    environment.sim.sync_kinematic_body_interpolation()
    environment.sim.update_gaussian_transforms()

    primary_metrics: dict[str, np.ndarray] | None = None
    primary_memmaps: dict[str, np.memmap] | None = None
    primary_renders: dict[str, dict[str, Any]] | None = None
    run_results: list[dict[str, Any]] = []
    comparison: dict[str, float | int] | None = None
    for run_index in range(args.runs):
        metrics, memmaps, render_records, run_comparison = run_replay(
            environment=environment,
            initial_state=initial_state,
            manifest=manifest,
            output_dir=args.output_dir,
            frame_indices=frame_indices,
            frame_timestamps=frame_timestamps,
            landmark_by_frame=landmark_by_frame,
            joint_offsets=joint_offsets,
            translation_world=translation_world,
            rest_positions=rest_positions,
            local_tetrahedra=local_tetrahedra,
            rest_inverse=rest_inverse,
            dynamic_mask=dynamic_mask,
            render_view=render_view,
            render_intrinsic=render_intrinsic,
            render_width=args.render_width,
            render_height=args.render_height,
            run_index=run_index,
            primary_metrics=primary_metrics,
            primary_memmaps=primary_memmaps,
            save_full_trajectory=not args.no_full_trajectory,
        )
        run_results.append(
            {
                "run_index": run_index,
                "elapsed_seconds": run_comparison["elapsed_seconds"],
                "physics_steps": run_comparison["physics_steps"],
                "milliseconds_per_physics_step": run_comparison[
                    "milliseconds_per_physics_step"
                ],
            }
        )
        if run_index == 0:
            primary_metrics = metrics
            primary_memmaps = memmaps
            primary_renders = render_records
            np.savez_compressed(
                args.output_dir / "trajectory_metrics.npz",
                frame_indices=frame_indices,
                frame_timestamps_s=frame_timestamps,
                **metrics,
            )
        else:
            comparison = run_comparison

    if primary_metrics is None or primary_renders is None or comparison is None:
        raise RuntimeError("Stage-C replay did not produce two comparable runs")

    summary = summarize_metrics(primary_metrics, frame_indices, landmarks)
    projector_metrics = environment.sim.triangle_skin_contact_projector.metrics()
    expected_pose_mapping_matches = bool(
        np.array_equal(
            primary_metrics["pose_state_index"], expected_pose_indices
        )
    )
    full_frozen_range = (
        int(frame_indices[0])
        == manifest["sequence"]["offline_stereo"][
            "playback_left_frame_range_inclusive"
        ][0]
        and int(frame_indices[-1])
        == manifest["sequence"]["offline_stereo"][
            "playback_left_frame_range_inclusive"
        ][1]
    )
    trajectory_files: dict[str, Any] = {
        "trajectory_metrics.npz": {
            "sha256": sha256_file(
                args.output_dir / "trajectory_metrics.npz"
            )
        }
    }
    for name in ("particle_positions", "gaussian_means", "gaussian_quats"):
        path = args.output_dir / f"{name}.npy"
        if path.exists() and name in (primary_memmaps or {}):
            trajectory_files[f"{name}.npy"] = {
                "shape": list(primary_memmaps[name].shape),
                "dtype": str(primary_memmaps[name].dtype),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    gates = {
        "full_frozen_frame_range_replayed": full_frozen_range,
        "manifest_pose_mapping_exact": expected_pose_mapping_matches,
        "fixed_60hz_clock_lag_below_one_step": (
            summary["physics_sample_lag_ms"]["maximum"]
            <= environment.physics_settings.dt * 1000.0 + 1.0e-6
        ),
        "jaw_contact_enabled": bool(
            environment.physics_settings.enable_triangle_skin_contacts
        ),
        "jaw_contact_generated": summary["contact_active_frame_count"] > 0,
        "shaft_contact_disabled": (
            set(environment.super_psm_tissue_contact_shape_ids[0])
            == set(projector_metrics["jaw_contact_shape_ids"])
        ),
        "persistent_grip_enabled": bool(
            projector_metrics["persistent_grip_constraint_enabled"]
        ),
        "persistent_grip_activated": (
            summary["persistent_grip_active_frame_count"] > 0
        ),
        "kinematic_pose_retraction_disabled": bool(
            not environment.super_psm_tissue_kinematic_gap_guard_enabled
        ),
        "particle_shape_contact_disabled": bool(
            not environment.physics_settings.enable_particle_shape_contacts
        ),
        "particle_particle_contact_disabled": bool(
            not environment.physics_settings.enable_particle_particle_contacts
        ),
        "visual_force_disabled": bool(
            environment.frames is None
            and environment.visual_forces_settings.iterations == 0
            and environment.sim.last_soft_visual_force_metrics is None
        ),
        "depth_residual_mapping_disabled": bool(
            not environment.super_tissue_residual_mapping_enabled
        ),
        "online_stiffness_optimization_disabled": bool(
            not environment.super_tissue_stiffness_optimization_enabled
        ),
        "fixed_paper_material_active": bool(
            environment.super_tissue_fixed_material_parameters
            and environment.super_tissue_mode == "paper_soft"
        ),
        "gaussian_tetra_binding_retained": bool(
            environment.sim.gaussian_model.num_soft_gaussians
            == manifest["tissue"]["counts"]["gaussians"]
            and environment.sim.gaussian_model.soft_gaussian_particle_indices.numel()
            == manifest["tissue"]["counts"]["gaussians"] * 4
        ),
        "all_particle_states_finite": summary["all_finite"],
        "no_inverted_tetrahedra": (
            summary["maximum_inverted_tetrahedra"] == 0
        ),
        "anchors_remain_fixed": (
            summary["maximum_anchor_drift_m"] <= 1.0e-9
        ),
        "bounded_tissue_displacement": (
            summary["maximum_dynamic_displacement_m"] <= 0.050
        ),
        "particle_trajectory_repeat_exact": (
            comparison["particle_hash_mismatch_count"] == 0
            and comparison["maximum_particle_position_abs_error_m"] == 0.0
        ),
        "contact_statistics_repeat_exact": (
            comparison["contact_integer_mismatch_count"] == 0
            and comparison["maximum_contact_float_abs_error"] == 0.0
        ),
        "gaussian_trajectory_repeat_exact": (
            comparison["gaussian_hash_mismatch_count"] == 0
            and comparison["maximum_gaussian_mean_abs_error_m"] == 0.0
            and comparison["maximum_gaussian_quaternion_abs_error"] == 0.0
        ),
        "landmark_renders_repeat_exact": (
            comparison["render_float_hash_mismatch_count"] == 0
            and comparison["render_uint8_hash_mismatch_count"] == 0
        ),
    }
    if args.no_full_trajectory:
        # Short developer runs still compare SHA256 at every selected frame,
        # but are not the formal full-trajectory stage-C artifact.
        gates["particle_trajectory_repeat_exact"] = (
            comparison["particle_hash_mismatch_count"] == 0
        )
        gates["gaussian_trajectory_repeat_exact"] = (
            comparison["gaussian_hash_mismatch_count"] == 0
        )

    report = {
        "schema": "super_tissue_calibration_stage_c_replay_v1",
        "stage": "C",
        "status": "passed" if all(gates.values()) else "failed",
        "passed": all(gates.values()),
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256_file(args.manifest),
        },
        "replay_contract": {
            "frame_range_inclusive": [
                int(frame_indices[0]),
                int(frame_indices[-1]),
            ],
            "frame_count": len(frame_indices),
            "physics_dt_s": environment.physics_settings.dt,
            "physics_substeps": environment.physics_settings.substeps,
            "timeline": (
                "fixed 60 Hz physics clock; stage-A pose-driver ZOH at every "
                "physics target time; causal camera samples within one dt"
            ),
            "pose_driver": manifest["psm_trajectory"]["driver_name"],
            "manual_corrections": correction,
            "manual_translation_world_m": translation_world.tolist(),
            "contact_shapes": (
                "two jaw triangle meshes only; shaft excluded"
            ),
            "visual_force": False,
            "depth_residual_mapping": False,
            "online_stiffness_optimization": False,
            "persistent_grip_or_follow_constraint": True,
            "gaussian_tetra_binding": True,
            "repeat_runs": args.runs,
        },
        "material": {
            "young_modulus_pa": environment.super_tissue_young_modulus_pa,
            "poisson_ratio": environment.super_tissue_poisson_ratio,
            "velocity_damping_per_second": (
                environment.physics_settings
                .particle_velocity_damping_per_second
            ),
            "gravity_m_s2": environment.super_tissue_gravity_m_s2,
            "material_iterations": (
                environment.physics_settings.material_iterations
            ),
            "material_relaxation": (
                environment.physics_settings.material_relaxation
            ),
        },
        "counts": {
            "tissue_particles": handle.particle_end - handle.particle_start,
            "tissue_tetrahedra": handle.tet_end - handle.tet_start,
            "all_gaussians": environment.sim.gaussian_model.num_gaussians,
            "soft_gaussians": (
                environment.sim.gaussian_model.num_soft_gaussians
            ),
        },
        "summary": summary,
        "repeatability": comparison,
        "runs": run_results,
        "landmark_renders": primary_renders,
        "artifacts": trajectory_files,
        "gates": gates,
    }
    report_path = args.output_dir / "stage_c_replay_report.json"
    report_path.write_text(
        json.dumps(json_ready(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(report), indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
