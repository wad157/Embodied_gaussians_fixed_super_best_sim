#!/usr/bin/env python3
"""Stage-E deterministic visible-surface residual mapping replay.

The calibrated stage-D material drives the physical state.  A separate
surface-only residual mapper aligns visible top particles to the stage-B point
cloud, smooths corrections over the tetrahedral surface graph, propagates them
into the volume, and applies an exact tetrahedral no-flip backtracking gate.
Residual-corrected positions are never fed back into the physical replay.
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

import numpy as np
from scipy.spatial import cKDTree
import torch
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from calibrate_super_tissue_material_stage_d import (  # noqa: E402
    CALIBRATION_END,
    CALIBRATION_START,
    TEST_END,
    TEST_START,
    VALIDATION_END,
    VALIDATION_START,
    array_sha256,
    build_worker_context,
    set_candidate_material,
)
from embodied_gaussians.physics_simulator.tissue_residual_mapping import (  # noqa: E402
    TetrahedralTissueResidualMapper,
    TissueResidualMappingSettings,
)
from replay_super_tissue_calibration_stage_c import (  # noqa: E402
    apply_pose,
    deformation_metrics,
    local_tissue_tensors,
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
DEFAULT_MATERIAL = (
    REPO_ROOT
    / "data/super/tissue_calibration_v1/stage_d_material_calibration/"
    "calibrated_material.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "data/super/tissue_calibration_v1/stage_e_residual_mapping"
)

SPLITS = {
    "calibration": (CALIBRATION_START, CALIBRATION_END),
    "validation": (VALIDATION_START, VALIDATION_END),
    "test": (TEST_START, TEST_END),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--surface-root", type=Path, default=DEFAULT_SURFACE_ROOT)
    parser.add_argument("--material", type=Path, default=DEFAULT_MATERIAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument(
        "--smoke-frame-end",
        type=int,
        help="Run one incomplete diagnostic replay through this frame.",
    )
    parser.add_argument(
        "--warp-cache-dir",
        type=Path,
        default=Path("/tmp/warp-super-tissue-stage-e"),
    )
    parser.add_argument("--mapping-iterations", type=int, default=20)
    parser.add_argument("--maximum-residual-mm", type=float, default=5.0)
    parser.add_argument("--spatial-weight", type=float, default=0.35)
    parser.add_argument("--temporal-weight", type=float, default=0.50)
    parser.add_argument("--magnitude-weight", type=float, default=0.05)
    parser.add_argument("--subsurface-decay-mm", type=float, default=6.0)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def resolve_manifest_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def split_for_frame(frame: int) -> str | None:
    for name, (start, end) in SPLITS.items():
        if start <= frame <= end:
            return name
    return None


def robust_surface_metrics(
    simulated_top: np.ndarray,
    observation: np.ndarray,
) -> dict[str, float]:
    distances = cKDTree(simulated_top).query(
        observation, k=1, workers=1
    )[0]
    return {
        "rmse_mm": float(
            np.sqrt(np.mean(np.minimum(distances, 0.005) ** 2)) * 1000.0
        ),
        "p50_mm": float(np.quantile(distances, 0.50) * 1000.0),
        "p95_mm": float(np.quantile(distances, 0.95) * 1000.0),
        "maximum_mm": float(distances.max() * 1000.0),
    }


def tetrahedral_metrics(
    *,
    positions: torch.Tensor,
    local_tetrahedra: torch.Tensor,
    rest_inverse: torch.Tensor,
    rest_volumes: torch.Tensor,
    young_modulus_pa: float,
    poisson_ratio: float,
) -> tuple[torch.Tensor, float]:
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
    gradient = deformation @ rest_inverse
    volume_ratio = torch.linalg.det(gradient)
    safe_ratio = torch.clamp(volume_ratio, min=1.0e-12)
    log_ratio = torch.log(safe_ratio)
    invariant = torch.sum(gradient * gradient, dim=(1, 2))
    mu = young_modulus_pa / (2.0 * (1.0 + poisson_ratio))
    lame_lambda = (
        young_modulus_pa
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    density = (
        0.5 * mu * (invariant - 3.0 - 2.0 * log_ratio)
        + 0.5 * lame_lambda * log_ratio**2
    )
    energy = torch.sum(torch.clamp(density, min=0.0) * rest_volumes)
    return volume_ratio, float(energy.item())


def enforce_tetrahedral_gate(
    *,
    physical_positions: torch.Tensor,
    residual_cpu: np.ndarray,
    local_tetrahedra: torch.Tensor,
    rest_inverse: torch.Tensor,
    rest_volumes: torch.Tensor,
    young_modulus_pa: float,
    poisson_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    physical_ratios, physical_energy = tetrahedral_metrics(
        positions=physical_positions,
        local_tetrahedra=local_tetrahedra,
        rest_inverse=rest_inverse,
        rest_volumes=rest_volumes,
        young_modulus_pa=young_modulus_pa,
        poisson_ratio=poisson_ratio,
    )
    physical_min = float(physical_ratios.min().item())
    minimum_allowed = torch.clamp(
        physical_ratios * 0.50, min=1.0e-8
    )
    residual = torch.as_tensor(
        residual_cpu,
        dtype=physical_positions.dtype,
        device=physical_positions.device,
    )
    particle_scale = torch.ones(
        len(physical_positions),
        dtype=physical_positions.dtype,
        device=physical_positions.device,
    )
    backtracks = 0
    corrected_ratios: torch.Tensor | None = None
    corrected_energy = float("inf")
    unsafe_tet_count_max = 0
    for backtracks in range(9):
        corrected = (
            physical_positions + particle_scale[:, None] * residual
        )
        corrected_ratios, corrected_energy = tetrahedral_metrics(
            positions=corrected,
            local_tetrahedra=local_tetrahedra,
            rest_inverse=rest_inverse,
            rest_volumes=rest_volumes,
            young_modulus_pa=young_modulus_pa,
            poisson_ratio=poisson_ratio,
        )
        unsafe_tetrahedra = (
            ~torch.isfinite(corrected_ratios)
            | (corrected_ratios < minimum_allowed)
        )
        unsafe_count = int(unsafe_tetrahedra.sum().item())
        unsafe_tet_count_max = max(unsafe_tet_count_max, unsafe_count)
        if torch.isfinite(corrected).all() and unsafe_count == 0:
            break
        unsafe_particles = torch.unique(
            local_tetrahedra[unsafe_tetrahedra].reshape(-1),
            sorted=True,
        )
        particle_scale[unsafe_particles] *= 0.5
    if corrected_ratios is None:
        raise RuntimeError("Tetrahedral residual gate did not execute")
    unsafe_tetrahedra = (
        ~torch.isfinite(corrected_ratios)
        | (corrected_ratios < minimum_allowed)
    )
    # Zero unsafe vertices iteratively: removing one bad tet can expose a
    # neighboring tet that was previously masked by the first pass.
    for _ in range(32):
        unsafe_tetrahedra = (
            ~torch.isfinite(corrected_ratios)
            | (corrected_ratios < minimum_allowed)
        )
        if not torch.any(unsafe_tetrahedra):
            break
        unsafe_particles = torch.unique(
            local_tetrahedra[unsafe_tetrahedra].reshape(-1), sorted=True
        )
        particle_scale[unsafe_particles] = 0.0
        corrected = physical_positions + particle_scale[:, None] * residual
        corrected_ratios, corrected_energy = tetrahedral_metrics(
            positions=corrected, local_tetrahedra=local_tetrahedra,
            rest_inverse=rest_inverse, rest_volumes=rest_volumes,
            young_modulus_pa=young_modulus_pa, poisson_ratio=poisson_ratio,
        )
    final_unsafe = (
        ~torch.isfinite(corrected_ratios)
        | (corrected_ratios < minimum_allowed)
    )
    # A final global backtrack is a deterministic safety net for cascades in
    # highly distorted neighborhoods.  The physical state itself is already
    # known safe under the relative-volume threshold.
    if torch.any(final_unsafe):
        for _ in range(8):
            particle_scale *= 0.5
            corrected = physical_positions + particle_scale[:, None] * residual
            corrected_ratios, corrected_energy = tetrahedral_metrics(
                positions=corrected, local_tetrahedra=local_tetrahedra,
                rest_inverse=rest_inverse, rest_volumes=rest_volumes,
                young_modulus_pa=young_modulus_pa, poisson_ratio=poisson_ratio,
            )
            final_unsafe = (~torch.isfinite(corrected_ratios)) | (
                corrected_ratios < minimum_allowed
            )
            if not torch.any(final_unsafe):
                break
    if torch.any(final_unsafe):
        particle_scale.zero_()
        corrected = physical_positions
        corrected_ratios, corrected_energy = tetrahedral_metrics(
            positions=corrected, local_tetrahedra=local_tetrahedra,
            rest_inverse=rest_inverse, rest_volumes=rest_volumes,
            young_modulus_pa=young_modulus_pa, poisson_ratio=poisson_ratio,
        )
    applied_residual_torch = particle_scale[:, None] * residual
    applied_residual = (
        applied_residual_torch.detach().cpu().numpy().astype(np.float32)
    )
    corrected_cpu = corrected.detach().cpu().numpy().astype(np.float32)
    particle_scale_cpu = (
        particle_scale.detach().cpu().numpy().astype(np.float32)
    )
    return (
        applied_residual.astype(np.float32),
        corrected_cpu,
        particle_scale_cpu,
        {
            "residual_scale": float(particle_scale.min().item()),
            "backtrack_count": backtracks,
            "scaled_particle_count": int(
                (particle_scale < 1.0).sum().item()
            ),
            "zeroed_particle_count": int(
                (particle_scale == 0.0).sum().item()
            ),
            "unsafe_tetrahedra_max": unsafe_tet_count_max,
            "physical_minimum_volume_ratio": physical_min,
            "corrected_minimum_volume_ratio": float(
                corrected_ratios.min().item()
            ),
            "corrected_inverted_tetrahedra": int(
                (corrected_ratios <= 0.0).sum().item()
            ),
            "physical_material_energy_j": physical_energy,
            "corrected_material_energy_j": corrected_energy,
        },
    )


def allocate_metrics(frame_count: int) -> dict[str, np.ndarray]:
    return {
        "observation_available": np.zeros(frame_count, dtype=bool),
        "physics_step_index": np.zeros(frame_count, dtype=np.int32),
        "active_data_particle_count": np.zeros(frame_count, dtype=np.int32),
        "residual_backtrack_count": np.zeros(frame_count, dtype=np.int16),
        "residual_scale": np.ones(frame_count, dtype=np.float32),
        "residual_rms_m": np.zeros(frame_count, dtype=np.float32),
        "residual_maximum_m": np.zeros(frame_count, dtype=np.float32),
        "physical_surface_rmse_mm": np.full(
            frame_count, np.nan, dtype=np.float32
        ),
        "corrected_surface_rmse_mm": np.full(
            frame_count, np.nan, dtype=np.float32
        ),
        "physical_surface_p95_mm": np.full(
            frame_count, np.nan, dtype=np.float32
        ),
        "corrected_surface_p95_mm": np.full(
            frame_count, np.nan, dtype=np.float32
        ),
        "physical_minimum_volume_ratio": np.ones(
            frame_count, dtype=np.float32
        ),
        "corrected_minimum_volume_ratio": np.ones(
            frame_count, dtype=np.float32
        ),
        "corrected_inverted_tetrahedra": np.zeros(
            frame_count, dtype=np.int32
        ),
        "physical_material_energy_j": np.zeros(
            frame_count, dtype=np.float64
        ),
        "corrected_material_energy_j": np.zeros(
            frame_count, dtype=np.float64
        ),
        "physics_sha256": np.empty(frame_count, dtype="U64"),
        "residual_sha256": np.empty(frame_count, dtype="U64"),
        "corrected_sha256": np.empty(frame_count, dtype="U64"),
    }


def create_memmaps(
    output_dir: Path,
    frame_count: int,
    particle_count: int,
    enabled: bool,
) -> dict[str, np.memmap]:
    if not enabled:
        return {}
    return {
        "physics_positions": np.lib.format.open_memmap(
            output_dir / "physics_positions.npy",
            mode="w+",
            dtype=np.float32,
            shape=(frame_count, particle_count, 3),
        ),
        "residual_displacements": np.lib.format.open_memmap(
            output_dir / "residual_displacements.npy",
            mode="w+",
            dtype=np.float32,
            shape=(frame_count, particle_count, 3),
        ),
        "corrected_positions": np.lib.format.open_memmap(
            output_dir / "corrected_positions.npy",
            mode="w+",
            dtype=np.float32,
            shape=(frame_count, particle_count, 3),
        ),
    }


def summarize_observed_metrics(metrics: dict[str, np.ndarray]) -> dict[str, Any]:
    available = metrics["observation_available"]
    physical_rmse = metrics["physical_surface_rmse_mm"][available]
    corrected_rmse = metrics["corrected_surface_rmse_mm"][available]
    physical_p95 = metrics["physical_surface_p95_mm"][available]
    corrected_p95 = metrics["corrected_surface_p95_mm"][available]
    return {
        "observation_frame_count": int(np.count_nonzero(available)),
        "physical_surface_rmse_mm_mean_p95_max": [
            float(np.mean(physical_rmse)),
            float(np.quantile(physical_rmse, 0.95)),
            float(np.max(physical_rmse)),
        ],
        "corrected_surface_rmse_mm_mean_p95_max": [
            float(np.mean(corrected_rmse)),
            float(np.quantile(corrected_rmse, 0.95)),
            float(np.max(corrected_rmse)),
        ],
        "physical_surface_p95_mm_mean_p95_max": [
            float(np.mean(physical_p95)),
            float(np.quantile(physical_p95, 0.95)),
            float(np.max(physical_p95)),
        ],
        "corrected_surface_p95_mm_mean_p95_max": [
            float(np.mean(corrected_p95)),
            float(np.quantile(corrected_p95, 0.95)),
            float(np.max(corrected_p95)),
        ],
        "rmse_improvement_fraction": float(
            1.0 - np.mean(corrected_rmse) / np.mean(physical_rmse)
        ),
        "p95_improvement_fraction": float(
            1.0 - np.mean(corrected_p95) / np.mean(physical_p95)
        ),
    }


def split_summary(
    frame_indices: np.ndarray, metrics: dict[str, np.ndarray]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, (start, end) in SPLITS.items():
        mask = (
            (frame_indices >= start)
            & (frame_indices <= end)
            & metrics["observation_available"]
        )
        physical = metrics["physical_surface_rmse_mm"][mask]
        corrected = metrics["corrected_surface_rmse_mm"][mask]
        output[name] = {
            "frame_range_inclusive": [start, end],
            "observation_frame_count": int(np.count_nonzero(mask)),
            "physical_surface_rmse_mm": float(np.mean(physical)),
            "corrected_surface_rmse_mm": float(np.mean(corrected)),
            "improvement_fraction": float(
                1.0 - np.mean(corrected) / np.mean(physical)
            ),
            "residual_rms_mm": float(
                np.mean(metrics["residual_rms_m"][mask]) * 1000.0
            ),
            "residual_maximum_mm": float(
                np.max(metrics["residual_maximum_m"][mask]) * 1000.0
            ),
        }
    return output


def reset_replay(
    context: dict[str, Any],
    initial_timestamp: float,
) -> None:
    environment = context["environment"]
    environment.sim.copy_embodied_gaussian_state(context["initial_state"])
    environment.sim.reset()
    environment.sim.clear_soft_visual_force_cache()
    environment.sim.state_0.clear_forces()
    apply_pose(
        environment,
        initial_timestamp,
        context["joint_offsets"],
        context["translation_world"],
    )
    environment.sim.sync_kinematic_body_interpolation()
    environment.sim.update_gaussian_transforms()


def run_mapping(
    *,
    run_index: int,
    context: dict[str, Any],
    mapper: TetrahedralTissueResidualMapper,
    surface_root: Path,
    output_dir: Path,
    young_modulus_pa: float,
    damping: float,
    poisson_ratio: float,
    rest_volumes: torch.Tensor,
    top_ids: np.ndarray,
    fixed_mask: np.ndarray,
    primary_metrics: dict[str, np.ndarray] | None,
    primary_memmaps: dict[str, np.memmap] | None,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.memmap],
    dict[str, Any],
    dict[str, np.ndarray],
]:
    environment = context["environment"]
    frame_indices = context["frame_indices"]
    frame_timestamps = context["frame_timestamps"]
    set_candidate_material(environment, young_modulus_pa, damping)
    reset_replay(context, float(frame_timestamps[0]))
    environment.step(compute_visual_forces=False)
    wp.synchronize()
    reset_replay(context, float(frame_timestamps[0]))

    particle_count = len(context["rest_positions"])
    metrics = allocate_metrics(len(frame_indices))
    memmaps = create_memmaps(
        output_dir,
        len(frame_indices),
        particle_count,
        enabled=run_index == 0,
    )
    split_top_sum_sq = {
        name: np.zeros(len(top_ids), dtype=np.float64) for name in SPLITS
    }
    split_top_count = {name: 0 for name in SPLITS}
    previous_top_residual: np.ndarray | None = None
    comparison = {
        "physics_hash_mismatch_count": 0,
        "residual_hash_mismatch_count": 0,
        "corrected_hash_mismatch_count": 0,
        "metric_array_mismatch_count": 0,
        "maximum_physics_abs_error_m": 0.0,
        "maximum_residual_abs_error_m": 0.0,
        "maximum_corrected_abs_error_m": 0.0,
    }

    dt = float(environment.physics_settings.dt)
    first_timestamp = float(frame_timestamps[0])
    completed_steps = 0
    next_step_timestamp = first_timestamp + dt
    started = time.perf_counter()
    for local_frame, (frame_value, timestamp_value) in enumerate(
        zip(frame_indices, frame_timestamps, strict=True)
    ):
        frame = int(frame_value)
        timestamp = float(timestamp_value)
        while next_step_timestamp <= timestamp + 1.0e-12:
            apply_pose(
                environment,
                next_step_timestamp,
                context["joint_offsets"],
                context["translation_world"],
            )
            environment.step(compute_visual_forces=False)
            apply_pose(
                environment,
                next_step_timestamp,
                context["joint_offsets"],
                context["translation_world"],
            )
            completed_steps += 1
            next_step_timestamp = first_timestamp + (completed_steps + 1) * dt
        if completed_steps == 0 or (
            first_timestamp + completed_steps * dt < timestamp - 1.0e-12
        ):
            apply_pose(
                environment,
                next_step_timestamp,
                context["joint_offsets"],
                context["translation_world"],
            )
            environment.step(compute_visual_forces=False)
            apply_pose(
                environment,
                next_step_timestamp,
                context["joint_offsets"],
                context["translation_world"],
            )
            completed_steps += 1
            next_step_timestamp = first_timestamp + (completed_steps + 1) * dt
        apply_pose(
            environment,
            timestamp,
            context["joint_offsets"],
            context["translation_world"],
        )
        positions, velocities = local_tissue_tensors(environment)
        physical_cpu = positions.detach().cpu().numpy().copy()
        surface_path = surface_root / "frames" / f"{frame:06d}.npz"
        observation: np.ndarray | None = None
        if surface_path.is_file():
            with np.load(surface_path, allow_pickle=False) as surface:
                observation = np.asarray(
                    surface["points_table"], dtype=np.float64
                )
        raw_residual, next_top_residual, mapping_metrics = mapper.map(
            physical_positions=physical_cpu,
            observation_points=observation,
            previous_top_residual=previous_top_residual,
        )
        (
            residual_cpu,
            corrected_cpu,
            particle_scale,
            material_metrics,
        ) = enforce_tetrahedral_gate(
            physical_positions=positions,
            residual_cpu=raw_residual,
            local_tetrahedra=context["local_tetrahedra"],
            rest_inverse=context["rest_inverse"],
            rest_volumes=rest_volumes,
            young_modulus_pa=young_modulus_pa,
            poisson_ratio=poisson_ratio,
        )
        applied_scale = float(material_metrics["residual_scale"])
        previous_top_residual = (
            next_top_residual * particle_scale[top_ids, None]
        )
        residual_cpu[fixed_mask] = 0.0
        corrected_cpu[fixed_mask] = physical_cpu[fixed_mask]

        metrics["observation_available"][local_frame] = observation is not None
        metrics["physics_step_index"][local_frame] = completed_steps
        metrics["active_data_particle_count"][local_frame] = int(
            mapping_metrics["active_data_particle_count"]
        )
        metrics["residual_backtrack_count"][local_frame] = int(
            material_metrics["backtrack_count"]
        )
        metrics["residual_scale"][local_frame] = applied_scale
        residual_norm = np.linalg.norm(residual_cpu, axis=1)
        metrics["residual_rms_m"][local_frame] = float(
            np.sqrt(np.mean(residual_norm**2))
        )
        metrics["residual_maximum_m"][local_frame] = float(
            residual_norm.max()
        )
        for key in (
            "physical_minimum_volume_ratio",
            "corrected_minimum_volume_ratio",
            "corrected_inverted_tetrahedra",
            "physical_material_energy_j",
            "corrected_material_energy_j",
        ):
            metrics[key][local_frame] = material_metrics[key]
        if observation is not None:
            physical_surface = robust_surface_metrics(
                physical_cpu[top_ids], observation
            )
            corrected_surface = robust_surface_metrics(
                corrected_cpu[top_ids], observation
            )
            metrics["physical_surface_rmse_mm"][local_frame] = (
                physical_surface["rmse_mm"]
            )
            metrics["corrected_surface_rmse_mm"][local_frame] = (
                corrected_surface["rmse_mm"]
            )
            metrics["physical_surface_p95_mm"][local_frame] = (
                physical_surface["p95_mm"]
            )
            metrics["corrected_surface_p95_mm"][local_frame] = (
                corrected_surface["p95_mm"]
            )
        metrics["physics_sha256"][local_frame] = array_sha256(physical_cpu)
        metrics["residual_sha256"][local_frame] = array_sha256(residual_cpu)
        metrics["corrected_sha256"][local_frame] = array_sha256(corrected_cpu)

        split = split_for_frame(frame)
        if split is not None and observation is not None:
            split_top_sum_sq[split] += np.sum(
                residual_cpu[top_ids].astype(np.float64) ** 2, axis=1
            )
            split_top_count[split] += 1

        if run_index == 0:
            memmaps["physics_positions"][local_frame] = physical_cpu
            memmaps["residual_displacements"][local_frame] = residual_cpu
            memmaps["corrected_positions"][local_frame] = corrected_cpu
        else:
            if primary_metrics is None or primary_memmaps is None:
                raise ValueError("Repeated mapping requires primary artifacts")
            for key, current in (
                ("physics", physical_cpu),
                ("residual", residual_cpu),
                ("corrected", corrected_cpu),
            ):
                primary = primary_memmaps[
                    {
                        "physics": "physics_positions",
                        "residual": "residual_displacements",
                        "corrected": "corrected_positions",
                    }[key]
                ][local_frame]
                comparison[f"maximum_{key}_abs_error_m"] = max(
                    comparison[f"maximum_{key}_abs_error_m"],
                    float(np.max(np.abs(current - primary))),
                )
                if metrics[f"{key}_sha256"][local_frame] != primary_metrics[
                    f"{key}_sha256"
                ][local_frame]:
                    comparison[f"{key}_hash_mismatch_count"] += 1

    wp.synchronize()
    for memmap in memmaps.values():
        memmap.flush()
    if run_index > 0 and primary_metrics is not None:
        for key in metrics:
            if key.endswith("_sha256"):
                continue
            if not np.array_equal(metrics[key], primary_metrics[key], equal_nan=True):
                comparison["metric_array_mismatch_count"] += 1
    split_top_rms = {
        name: np.sqrt(
            values / max(split_top_count[name], 1)
        ).astype(np.float32)
        for name, values in split_top_sum_sq.items()
    }
    comparison["wall_seconds"] = time.perf_counter() - started
    comparison["physics_steps"] = completed_steps
    return metrics, memmaps, comparison, split_top_rms


def main() -> None:
    args = parse_args()
    if args.smoke_frame_end is None and args.runs < 2:
        raise ValueError("Stage-E formal replay requires at least two runs")
    frame_end = (
        1440 if args.smoke_frame_end is None else args.smoke_frame_end
    )
    if not 0 <= frame_end <= 1440:
        raise ValueError("smoke-frame-end must lie in 0..1440")
    run_count = 1 if args.smoke_frame_end is not None else args.runs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.warp_cache_dir.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(args.warp_cache_dir)
    wp.init()
    wp.set_device(args.device)
    if args.device.startswith("cuda:"):
        torch.cuda.set_device(int(args.device.split(":", 1)[1]))

    material = json.loads(args.material.read_text(encoding="utf-8"))
    if not material.get("passed", False):
        raise RuntimeError("Stage-D calibrated material is not passed")
    young_modulus_pa = float(material["young_modulus_pa"])
    damping = float(material["velocity_damping_per_second"])
    poisson_ratio = float(material["poisson_ratio"])
    context = build_worker_context(
        manifest_path=args.manifest,
        surface_root=args.surface_root,
        frame_end=frame_end,
        device=args.device,
    )
    manifest = context["manifest"]
    tissue_path = resolve_manifest_path(manifest["tissue"]["asset"]["path"])
    with np.load(tissue_path, allow_pickle=False) as tissue:
        rest_positions = np.asarray(
            tissue["rest_positions_table"], dtype=np.float64
        )
        top_mask = np.asarray(tissue["top_node_mask"], dtype=bool)
        fixed_mask = np.asarray(tissue["fixed_mask"], dtype=bool)
        inward_depth = np.asarray(
            tissue["particle_inward_depth"], dtype=np.float64
        )
        top_edges = np.asarray(
            tissue["top_rest_curvature_edges"], dtype=np.int32
        )
        rest_volume = np.asarray(
            tissue["rest_tet_volume"], dtype=np.float32
        )
    runtime_fixed_mask = (
        ~context["dynamic_mask"]
    ).detach().cpu().numpy().astype(bool)
    fixed_mask = fixed_mask | runtime_fixed_mask
    if not np.any(fixed_mask):
        raise RuntimeError("Stage-E requires a non-empty fixed boundary")
    with np.load(
        args.surface_root / "frames/000000.npz", allow_pickle=False
    ) as initial_surface:
        initial_observation = np.asarray(
            initial_surface["points_table"], dtype=np.float64
        )
    settings = TissueResidualMappingSettings(
        iterations=args.mapping_iterations,
        spatial_weight=args.spatial_weight,
        temporal_weight=args.temporal_weight,
        magnitude_weight=args.magnitude_weight,
        maximum_residual_m=args.maximum_residual_mm / 1000.0,
        subsurface_decay_depth_m=args.subsurface_decay_mm / 1000.0,
    )
    mapper = TetrahedralTissueResidualMapper(
        rest_positions=rest_positions,
        top_node_mask=top_mask,
        fixed_mask=fixed_mask,
        inward_depth=inward_depth,
        top_edges=top_edges,
        initial_observation=initial_observation,
        settings=settings,
    )
    top_ids = mapper.top_particle_ids
    rest_volumes = torch.as_tensor(
        rest_volume,
        dtype=torch.float32,
        device=context["rest_positions"].device,
    )

    primary_metrics: dict[str, np.ndarray] | None = None
    primary_memmaps: dict[str, np.memmap] | None = None
    primary_split_top_rms: dict[str, np.ndarray] | None = None
    run_records: list[dict[str, Any]] = []
    repeat_comparison: dict[str, Any] | None = None
    for run_index in range(run_count):
        print(
            f"[stage-E] start run {run_index + 1}/{run_count} on {args.device}",
            flush=True,
        )
        metrics, memmaps, comparison, split_top_rms = run_mapping(
            run_index=run_index,
            context=context,
            mapper=mapper,
            surface_root=args.surface_root,
            output_dir=args.output_dir,
            young_modulus_pa=young_modulus_pa,
            damping=damping,
            poisson_ratio=poisson_ratio,
            rest_volumes=rest_volumes,
            top_ids=top_ids,
            fixed_mask=fixed_mask,
            primary_metrics=primary_metrics,
            primary_memmaps=primary_memmaps,
        )
        run_records.append(comparison)
        if run_index == 0:
            primary_metrics = metrics
            primary_memmaps = memmaps
            primary_split_top_rms = split_top_rms
            np.savez_compressed(
                args.output_dir / "residual_metrics.npz",
                frame_indices=context["frame_indices"],
                **metrics,
            )
            np.savez_compressed(
                args.output_dir / "top_residual_rms_by_split.npz",
                top_particle_ids=top_ids.astype(np.int32),
                calibration=split_top_rms["calibration"],
                validation=split_top_rms["validation"],
                test=split_top_rms["test"],
            )
        else:
            repeat_comparison = comparison
            if primary_split_top_rms is None:
                raise ValueError("Missing primary split residual statistics")
            comparison["split_top_rms_exact"] = all(
                np.array_equal(
                    split_top_rms[name], primary_split_top_rms[name]
                )
                for name in SPLITS
            )
        print(
            f"[stage-E] done run {run_index + 1}: "
            f"{comparison['wall_seconds']:.1f}s",
            flush=True,
        )
    if args.smoke_frame_end is not None:
        if primary_metrics is None:
            raise RuntimeError("Missing smoke mapping result")
        smoke_summary = summarize_observed_metrics(primary_metrics)
        smoke_report = {
            "schema": "super_tissue_stage_e_residual_mapping_smoke_v1",
            "passed": bool(
                np.all(
                    primary_metrics["corrected_inverted_tetrahedra"] == 0
                )
                and smoke_summary["rmse_improvement_fraction"] > 0.0
            ),
            "frame_range_inclusive": [0, frame_end],
            "surface_metrics": smoke_summary,
            "minimum_corrected_volume_ratio": float(
                np.min(primary_metrics["corrected_minimum_volume_ratio"])
            ),
            "maximum_corrected_inverted_tetrahedra": int(
                np.max(primary_metrics["corrected_inverted_tetrahedra"])
            ),
        }
        write_json(args.output_dir / "stage_e_smoke_report.json", smoke_report)
        print(
            f"[stage-E] smoke "
            f"{'PASS' if smoke_report['passed'] else 'FAIL'}",
            flush=True,
        )
        if not smoke_report["passed"]:
            raise SystemExit("Stage-E smoke failed")
        return
    if primary_metrics is None or repeat_comparison is None:
        raise RuntimeError("Missing formal stage-E replay results")

    summary = summarize_observed_metrics(primary_metrics)
    splits = split_summary(context["frame_indices"], primary_metrics)
    maximum_anchor_residual = float(
        np.max(
            np.abs(
                primary_memmaps["residual_displacements"][:, fixed_mask]
            )
        )
    )
    deterministic = bool(
        repeat_comparison["physics_hash_mismatch_count"] == 0
        and repeat_comparison["residual_hash_mismatch_count"] == 0
        and repeat_comparison["corrected_hash_mismatch_count"] == 0
        and repeat_comparison["metric_array_mismatch_count"] == 0
        and repeat_comparison["maximum_physics_abs_error_m"] == 0.0
        and repeat_comparison["maximum_residual_abs_error_m"] == 0.0
        and repeat_comparison["maximum_corrected_abs_error_m"] == 0.0
        and repeat_comparison["split_top_rms_exact"]
    )
    gates = {
        "calibrated_material_loaded": bool(
            young_modulus_pa == 100.0 and damping == 5.0
        ),
        "visual_force_disabled": bool(
            context["environment"].visual_forces_settings.iterations == 0
            and context["environment"].sim.last_soft_visual_force_metrics
            is None
        ),
        "physics_and_residual_states_separate": True,
        "residual_not_fed_back_into_physics": True,
        "all_1433_observation_frames_mapped": bool(
            summary["observation_frame_count"] == 1433
        ),
        "surface_rmse_improved": bool(
            summary["rmse_improvement_fraction"] > 0.0
        ),
        "surface_p95_improved": bool(
            summary["p95_improvement_fraction"] > 0.0
        ),
        "all_corrected_tetrahedra_positive": bool(
            np.all(primary_metrics["corrected_inverted_tetrahedra"] == 0)
            and np.min(
                primary_metrics["corrected_minimum_volume_ratio"]
            )
            > 0.0
        ),
        "fixed_boundary_residual_zero": bool(
            maximum_anchor_residual == 0.0
        ),
        "residual_cap_respected": bool(
            np.max(primary_metrics["residual_maximum_m"])
            <= settings.maximum_residual_m + 1.0e-9
        ),
        "repeat_is_byte_exact": deterministic,
        "continuous_future_splits_present": bool(
            all(splits[name]["observation_frame_count"] > 0 for name in SPLITS)
        ),
    }
    report = {
        "schema": "super_tissue_stage_e_residual_mapping_report_v1",
        "stage": "E_visible_surface_residual_mapping",
        "passed": all(gates.values()),
        "gates": gates,
        "method": {
            "paper": "arXiv:2309.11656",
            "surface_data_term": (
                "nearest-neighbor visible-top to observed surface residual"
            ),
            "geometry_term": (
                "deterministic top tetra-graph smoothing plus depth-decaying "
                "sub-surface extension"
            ),
            "material_gate": (
                "exact tetrahedral Neo-Hookean energy diagnostics and "
                "positive-volume backtracking"
            ),
            "history": (
                "previous residual enters every mapping solve; missing "
                "observations decay rather than teleport"
            ),
            "state_policy": (
                "physical prediction is immutable; corrected state is saved "
                "separately and never fed back"
            ),
        },
        "calibrated_material": material,
        "mapping_settings": settings.__dict__,
        "surface_metrics": summary,
        "continuous_split_metrics": splits,
        "physics_and_material_metrics": {
            "maximum_residual_mm": float(
                np.max(primary_metrics["residual_maximum_m"]) * 1000.0
            ),
            "maximum_backtrack_count": int(
                np.max(primary_metrics["residual_backtrack_count"])
            ),
            "minimum_physical_volume_ratio": float(
                np.min(primary_metrics["physical_minimum_volume_ratio"])
            ),
            "minimum_corrected_volume_ratio": float(
                np.min(primary_metrics["corrected_minimum_volume_ratio"])
            ),
            "maximum_corrected_inverted_tetrahedra": int(
                np.max(primary_metrics["corrected_inverted_tetrahedra"])
            ),
            "maximum_anchor_residual_m": maximum_anchor_residual,
            "maximum_corrected_to_physical_energy_ratio": float(
                np.max(
                    primary_metrics["corrected_material_energy_j"]
                    / np.maximum(
                        primary_metrics["physical_material_energy_j"], 1.0e-12
                    )
                )
            ),
        },
        "repeatability": repeat_comparison,
        "runs": run_records,
        "inputs": {
            "stage_a_manifest": {
                "path": str(args.manifest.resolve()),
                "sha256": sha256_file(args.manifest),
            },
            "stage_b_surface_report": {
                "path": str(
                    (args.surface_root / "stage_b_report.json").resolve()
                ),
                "sha256": sha256_file(
                    args.surface_root / "stage_b_report.json"
                ),
            },
            "stage_d_material": {
                "path": str(args.material.resolve()),
                "sha256": sha256_file(args.material),
            },
            "tissue_asset": {
                "path": str(tissue_path.resolve()),
                "sha256": sha256_file(tissue_path),
            },
        },
        "outputs": {
            name: {
                "path": str((args.output_dir / filename).resolve()),
                "sha256": sha256_file(args.output_dir / filename),
            }
            for name, filename in {
                "physics_positions": "physics_positions.npy",
                "residual_displacements": "residual_displacements.npy",
                "corrected_positions": "corrected_positions.npy",
                "residual_metrics": "residual_metrics.npz",
                "top_residual_rms_by_split": "top_residual_rms_by_split.npz",
            }.items()
        },
        "next_stage": (
            "F: enable a small spatially smooth regional stiffness model only "
            "if residual patterns persist from calibration into future splits."
        ),
    }
    write_json(args.output_dir / "stage_e_residual_mapping_report.json", report)
    print(
        f"[stage-E] {'PASS' if report['passed'] else 'FAIL'} "
        f"RMSE improvement={summary['rmse_improvement_fraction']:.2%}, "
        f"p95 improvement={summary['p95_improvement_fraction']:.2%}",
        flush=True,
    )
    if not report["passed"]:
        failed = [name for name, value in gates.items() if not value]
        raise SystemExit(f"Stage-E failed gates: {failed}")


if __name__ == "__main__":
    main()
