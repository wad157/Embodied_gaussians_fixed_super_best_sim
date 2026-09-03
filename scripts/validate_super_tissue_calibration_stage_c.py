#!/usr/bin/env python3
"""Validate the saved SUPER tissue-calibration stage-C replay artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY_DIR = (
    REPO_ROOT / "data/super/tissue_calibration_v1/stage_c_offline_replay"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify stage-C report gates, complete trajectories, per-frame "
            "state hashes, and landmark renders without rerunning CUDA."
        )
    )
    parser.add_argument(
        "--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(
        memoryview(np.ascontiguousarray(array)).cast("B")
    ).hexdigest()


def validate_array(
    replay_dir: Path,
    name: str,
    artifact: dict[str, Any],
) -> tuple[np.ndarray, dict[str, bool]]:
    path = replay_dir / name
    array = np.load(path, mmap_mode="r")
    return array, {
        f"{name}_exists": path.is_file(),
        f"{name}_shape": list(array.shape) == artifact["shape"],
        f"{name}_dtype": str(array.dtype) == artifact["dtype"],
        f"{name}_size": path.stat().st_size == artifact["size_bytes"],
        f"{name}_sha256": sha256_file(path) == artifact["sha256"],
    }


def main() -> None:
    args = parse_args()
    report_path = args.replay_dir / "stage_c_replay_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest_path = Path(report["manifest"]["path"])
    artifacts = report["artifacts"]
    metrics_path = args.replay_dir / "trajectory_metrics.npz"
    metrics = np.load(metrics_path)
    frame_indices = metrics["frame_indices"]

    gates: dict[str, bool] = {
        "stage_c_report_passed": bool(report["passed"]),
        "current_manifest_sha256": (
            manifest_path.is_file()
            and sha256_file(manifest_path) == report["manifest"]["sha256"]
        ),
        "all_stage_c_runtime_gates_passed": all(
            report["gates"].values()
        ),
        "full_frozen_frame_range_present": bool(
            len(frame_indices) == 1441
            and int(frame_indices[0]) == 0
            and int(frame_indices[-1]) == 1440
        ),
        "trajectory_metrics_sha256": (
            sha256_file(metrics_path)
            == artifacts["trajectory_metrics.npz"]["sha256"]
        ),
        "repeatability_has_zero_mismatches": all(
            report["repeatability"][key] == 0
            for key in (
                "contact_integer_mismatch_count",
                "particle_hash_mismatch_count",
                "gaussian_hash_mismatch_count",
                "render_float_hash_mismatch_count",
                "render_uint8_hash_mismatch_count",
            )
        ),
        "repeatability_has_zero_numeric_error": all(
            report["repeatability"][key] == 0.0
            for key in (
                "maximum_contact_float_abs_error",
                "maximum_particle_position_abs_error_m",
                "maximum_gaussian_mean_abs_error_m",
                "maximum_gaussian_quaternion_abs_error",
            )
        ),
    }

    particles, particle_gates = validate_array(
        args.replay_dir,
        "particle_positions.npy",
        artifacts["particle_positions.npy"],
    )
    gaussian_means, mean_gates = validate_array(
        args.replay_dir,
        "gaussian_means.npy",
        artifacts["gaussian_means.npy"],
    )
    gaussian_quats, quat_gates = validate_array(
        args.replay_dir,
        "gaussian_quats.npy",
        artifacts["gaussian_quats.npy"],
    )
    gates.update(particle_gates)
    gates.update(mean_gates)
    gates.update(quat_gates)

    particle_frame_mismatches: list[int] = []
    gaussian_frame_mismatches: list[int] = []
    for local_index, frame_index in enumerate(frame_indices):
        if (
            array_sha256(particles[local_index])
            != metrics["particle_sha256"][local_index]
        ):
            particle_frame_mismatches.append(int(frame_index))
        if (
            array_sha256(gaussian_means[local_index])
            != metrics["gaussian_means_sha256"][local_index]
            or array_sha256(gaussian_quats[local_index])
            != metrics["gaussian_quats_sha256"][local_index]
        ):
            gaussian_frame_mismatches.append(int(frame_index))
    gates["all_particle_frame_hashes_match"] = (
        not particle_frame_mismatches
    )
    gates["all_gaussian_frame_hashes_match"] = (
        not gaussian_frame_mismatches
    )

    render_mismatches: list[str] = []
    for name, entry in report["landmark_renders"].items():
        stem = f"{entry['left_frame']:04d}_{name}"
        float_path = args.replay_dir / "landmark_renders" / f"{stem}.npy"
        png_path = args.replay_dir / "landmark_renders" / f"{stem}.png"
        render_float = np.load(float_path)
        render_u8 = np.clip(
            np.rint(render_float * 255.0), 0, 255
        ).astype(np.uint8)
        decoded_bgr = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
        decoded_rgb = (
            None
            if decoded_bgr is None
            else cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
        )
        if (
            array_sha256(render_float) != entry["float32_sha256"]
            or array_sha256(render_u8) != entry["uint8_sha256"]
            or decoded_rgb is None
            or not np.array_equal(decoded_rgb, render_u8)
        ):
            render_mismatches.append(name)
    gates["all_landmark_render_hashes_match"] = not render_mismatches

    validation = {
        "schema": "super_tissue_calibration_stage_c_artifact_validation_v1",
        "stage": "C",
        "source_report": str(report_path.resolve()),
        "frame_count": len(frame_indices),
        "particle_frame_hash_mismatches": particle_frame_mismatches,
        "gaussian_frame_hash_mismatches": gaussian_frame_mismatches,
        "landmark_render_mismatches": render_mismatches,
        "gates": gates,
        "passed": all(gates.values()),
    }
    output_path = args.replay_dir / "stage_c_artifact_validation.json"
    output_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    if not validation["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
