#!/usr/bin/env python3
"""Fail-closed audit for the RGB-only, no-PSM TRACE SIM baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = REPO_ROOT / "baselines" / "trace_sim"
sys.path.insert(0, str(ADAPTER_ROOT))
from protocol import (  # noqa: E402
    ADAPTER_VERSION,
    CAMERAS,
    DATASETS,
    HOLDOUT_OFFSET,
    HOLDOUT_STRIDE,
    UPSTREAM_COMMIT,
    dataset_spec,
    future_frames,
    future_start,
    load_json,
    max_observed_time,
    reconstruction_holdouts,
    resolve_dataset,
    sha256_file,
    training_frames,
)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def exact_png_frames(directory: Path, frames: int) -> None:
    expected = ["{:06d}.png".format(frame) for frame in range(frames)]
    actual = sorted(path.name for path in directory.glob("*.png"))
    if actual != expected:
        raise ValueError("RGB sequence is incomplete: {}".format(directory))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, default=REPO_ROOT / "baselines" / "TRACE")
    parser.add_argument("--trace-python", type=Path, default=Path("/Media_HDD/jwshan/conda_envs/freegave/bin/python"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = resolve_dataset(REPO_ROOT, args.dataset_key, args.dataset)
    spec = dataset_spec(args.dataset_key)
    frames = int(spec["frames"])
    episode = load_json(dataset / "episode.json")
    if episode.get("name") != spec["name"] or int(episode.get("frames", -1)) != frames:
        raise ValueError("episode.json does not match the frozen dataset")
    if int(episode.get("fps", -1)) != 30:
        raise ValueError("TRACE SIM protocol requires the original 30 Hz sequence")
    calibration_hashes = {}
    timestamps = None
    full_k = {}
    for camera in CAMERAS:
        exact_png_frames(dataset / "rgb" / camera, frames)
        metadata_path = dataset / "videos" / (camera + ".json")
        metadata = load_json(metadata_path)
        current = np.asarray(metadata["timestamps"], dtype=np.float64)
        if current.shape != (frames,) or not np.allclose(
            current, np.arange(frames, dtype=np.float64) / 30.0, atol=1.0e-12
        ):
            raise ValueError("{} timestamps are not the frozen 30 Hz sequence".format(camera))
        if timestamps is not None and not np.array_equal(timestamps, current):
            raise ValueError("stereo timestamps differ")
        timestamps = current
        intrinsic = np.asarray(metadata["K"], dtype=np.float64)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError("invalid K for {}".format(camera))
        width, height = [int(value) for value in metadata["resolution"]]
        full_k[camera] = {
            "fx": float(intrinsic[0, 0]),
            "fy": float(intrinsic[1, 1]),
            "cx": float(intrinsic[0, 2]),
            "cy": float(intrinsic[1, 2]),
            "width": width,
            "height": height,
        }
        calibration_hashes[camera] = sha256_file(metadata_path)
    upstream = args.baseline_root.expanduser().resolve()
    if git(upstream, "rev-parse", "HEAD") != UPSTREAM_COMMIT:
        raise ValueError("TRACE checkout is not pinned to {}".format(UPSTREAM_COMMIT))
    changed = [
        line
        for line in git(
            upstream, "status", "--short", "--untracked-files=no"
        ).splitlines()
        if "/__pycache__/" not in line and not line.rstrip().endswith(".pyc")
    ]
    if changed:
        raise ValueError(
            "TRACE core checkout is modified:\n{}".format("\n".join(changed))
        )
    environment = subprocess.check_output(
        [
            str(args.trace_python),
            "-c",
            "import sys,torch; import diff_gaussian_rasterization,simple_knn; "
            "print(sys.version.split()[0]); print(torch.__version__); print(torch.version.cuda)",
        ],
        text=True,
    ).splitlines()
    if environment[:3] != ["3.7.16", "1.13.1", "11.6"]:
        raise ValueError("unexpected TRACE environment: {}".format(environment))
    train = training_frames(frames)
    holdout = reconstruction_holdouts(frames)
    future = future_frames(frames)
    if sorted(train + holdout + future) != list(range(frames)):
        raise AssertionError("protocol frame sets do not cover the sequence exactly")
    report = {
        "schema": "fixedsuperbest.trace_protocol_audit.v1",
        "passed": True,
        "adapter_version": ADAPTER_VERSION,
        "dataset_key": args.dataset_key,
        "dataset": str(dataset),
        "frames": frames,
        "future_start": future_start(frames),
        "max_observed_time": max_observed_time(frames),
        "split": {
            "holdout_stride": HOLDOUT_STRIDE,
            "holdout_offset": HOLDOUT_OFFSET,
            "training_frame_count": len(train),
            "training_view_count": len(train) * len(CAMERAS),
            "reconstruction_holdout_count": len(holdout),
            "future_frame_count": len(future),
        },
        "training_inputs": {
            "allowed": ["legal prefix stereo RGB", "K", "X_WC_ros_optical", "timestamps"],
            "forbidden": [
                "PSM poses or controls",
                "known grasp boundary",
                "depth of any kind",
                "tissue or instrument masks",
                "ground-truth trajectories",
                "evaluation point IDs/pixels",
                "held-out RGB",
                "future RGB",
            ],
            "camera_calibration_sha256": calibration_hashes,
            "full_k": full_k,
        },
        "evaluation_only": {
            "query_manifest": "evaluation/evaluation_points_30_non_grasp.json",
            "query_frame": 0,
            "controlled_boundary": "used by the unchanged scorer only, never TRACE",
        },
        "trace": {
            "root": str(upstream),
            "commit": UPSTREAM_COMMIT,
            "tracked_core_clean": True,
            "freegave": False,
            "camera_change": "adapter-side asymmetric projection matrix from full K",
            "algorithm_changes": [],
        },
        "environment": {
            "python": environment[0],
            "torch": environment[1],
            "torch_cuda": environment[2],
        },
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
