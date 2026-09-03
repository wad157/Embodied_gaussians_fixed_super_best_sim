#!/usr/bin/env python3
"""Extract a fresh first strict stereo pair for paper-LND calibration.

This stage reads only the original bag, original stereo calibration, and the
frozen raw-kinematics backbone.  It deliberately does not read any P420006
annotation, mask, correction, or GUI pose driver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from build_super_raw_psm_kinematics import (
    load_stereo_calibration,
    nearest_indices,
)
from track_super_p420006_stereo_surgicalsam2 import strict_pair_stream


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data/super/psm_raw_kinematics_v1（纯机器人学版本）"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_paper_lnd_stereo_v1"
)
PAPER_REPO = Path("/Media_HDD/jwshan/wad/online_dvrk_tracking")
UPSTREAM_COMMIT = "cb2a264167aaf05b5a9c20da885d48568f78311f"
PAPER_MESH_NAMES = (
    "low_res_shaft_multi_cylinder.ply",
    "low_res_logo_low_res_1.ply",
    "low_res_jawright_lowres.ply",
    "low_res_jawleft_lowres.ply",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a clean paper-LND first-frame stereo annotation root "
            "directly from the original raw observations."
        )
    )
    parser.add_argument(
        "--bag",
        type=Path,
        default=REPO_ROOT / "data/grasp5/grasp5.bag",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO_ROOT / "data/camera_calibration.yaml",
    )
    parser.add_argument(
        "--kinematics",
        type=Path,
        default=RAW_ROOT / "kinematics.npz",
    )
    parser.add_argument(
        "--raw-model",
        type=Path,
        default=RAW_ROOT / "model.json",
    )
    parser.add_argument(
        "--raw-report",
        type=Path,
        default=RAW_ROOT / "report.json",
    )
    parser.add_argument("--paper-repo", type=Path, default=PAPER_REPO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def verify_frozen_input(
    path: Path,
    frozen: dict[str, Any],
    label: str,
) -> None:
    stat = path.stat()
    if stat.st_size != int(frozen["size_bytes"]):
        raise RuntimeError(f"{label} size differs from the frozen raw report")
    if stat.st_mtime_ns != int(frozen["mtime_ns"]):
        raise RuntimeError(f"{label} mtime differs from the frozen raw report")


def main() -> None:
    args = parse_args()
    mesh_dir = args.paper_repo / "urdfs/dVRK/meshes"
    mesh_paths = [mesh_dir / name for name in PAPER_MESH_NAMES]
    for path in (
        args.bag,
        args.calibration,
        args.kinematics,
        args.raw_model,
        args.raw_report,
        *mesh_paths,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    import subprocess

    commit = subprocess.run(
        ["git", "-C", str(args.paper_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"Unexpected upstream commit {commit}")

    raw_report = json.loads(args.raw_report.read_text(encoding="utf-8"))
    if raw_report.get("passed") is not True:
        raise RuntimeError("Frozen raw-kinematics report has not passed")
    verify_frozen_input(args.bag, raw_report["inputs"]["bag"], "Original bag")
    verify_frozen_input(
        args.calibration,
        raw_report["inputs"]["camera_calibration"],
        "Original stereo calibration",
    )
    raw_model = json.loads(args.raw_model.read_text(encoding="utf-8"))
    T_right_left = np.asarray(
        raw_model["calibration_and_static_transforms"][
            "T_rectified_right_camera_rectified_left_camera"
        ],
        dtype=np.float64,
    )
    calibration = load_stereo_calibration(args.calibration)
    width, height = calibration.image_size

    with np.load(args.kinematics, allow_pickle=False) as raw:
        stereo_left_index = raw["stereo_left_index"].astype(np.int64)
        stereo_right_index = raw["stereo_right_index"].astype(np.int64)
        left_timestamps_ns = raw["left_timestamps_ros_ns"].astype(np.int64)
        right_timestamps_ns = raw["right_timestamps_ros_ns"].astype(np.int64)
        joint_timestamps_ns = raw["joint_timestamps_ros_ns"].astype(np.int64)
        q7 = raw["q7"].astype(np.float64)
    if len(stereo_left_index) == 0:
        raise RuntimeError("Frozen raw backbone contains no strict stereo pair")

    left_index = int(stereo_left_index[0])
    right_index = int(stereo_right_index[0])
    decoding_anomalies: list[dict[str, Any]] = []
    stream = strict_pair_stream(
        bag_path=args.bag,
        calibration=calibration,
        left_indices=np.asarray([left_index], dtype=np.int64),
        right_indices=np.asarray([right_index], dtype=np.int64),
        decoding_anomalies=decoding_anomalies,
    )
    slot, left_image, right_image = next(stream)
    if slot != 0:
        raise RuntimeError(f"Expected strict pair slot 0, received {slot}")

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} exists; pass --overwrite to recreate it"
            )
        shutil.rmtree(args.output_dir)
    for name in ("images", "annotations", "previews"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)

    image_paths: dict[str, Path] = {}
    for side, image in (("left", left_image), ("right", right_image)):
        output = (
            args.output_dir
            / "images"
            / f"keyframe_00_{side}_rectified.png"
        )
        if not cv2.imwrite(
            str(output),
            image,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        ):
            raise RuntimeError(f"Failed to write {output}")
        image_paths[side] = output

    left_ns = int(left_timestamps_ns[left_index])
    right_ns = int(right_timestamps_ns[right_index])
    pair_ns = left_ns + (right_ns - left_ns) // 2
    left_q = int(
        nearest_indices(
            np.asarray([left_ns], dtype=np.int64),
            joint_timestamps_ns,
        )[0]
    )
    right_q = int(
        nearest_indices(
            np.asarray([right_ns], dtype=np.int64),
            joint_timestamps_ns,
        )[0]
    )
    pair_q = int(
        nearest_indices(
            np.asarray([pair_ns], dtype=np.int64),
            joint_timestamps_ns,
        )[0]
    )
    pair_time_s = (pair_ns - int(joint_timestamps_ns[0])) * 1.0e-9
    row = {
        "keyframe_index": 0,
        "strict_pair_slot": 0,
        "annotation_schema": "super_paper_lnd_stereo_manual_annotation_v1",
        "left_index": left_index,
        "right_index": right_index,
        "left_timestamp_ros_ns": left_ns,
        "right_timestamp_ros_ns": right_ns,
        "pair_timestamp_ros_ns": pair_ns,
        "pair_time_s": float(pair_time_s),
        "stereo_delta_ms": float((left_ns - right_ns) * 1.0e-6),
        "left_q_index": left_q,
        "right_q_index": right_q,
        "pair_q_index": pair_q,
        "q7_mid": q7[pair_q].tolist(),
        "left_image": "images/keyframe_00_left_rectified.png",
        "right_image": "images/keyframe_00_right_rectified.png",
        "left_image_sha256": sha256(image_paths["left"]),
        "right_image_sha256": sha256(image_paths["right"]),
    }
    np.savez_compressed(
        args.output_dir / "keyframes.npz",
        schema=np.asarray("super_paper_lnd_stereo_first_frame_v1"),
        strict_pair_slot=np.asarray([0], dtype=np.int64),
        left_index=np.asarray([left_index], dtype=np.int64),
        right_index=np.asarray([right_index], dtype=np.int64),
        left_timestamp_ros_ns=np.asarray([left_ns], dtype=np.int64),
        right_timestamp_ros_ns=np.asarray([right_ns], dtype=np.int64),
        pair_timestamp_ros_ns=np.asarray([pair_ns], dtype=np.int64),
        pair_time_s=np.asarray([pair_time_s], dtype=np.float64),
        left_q_index=np.asarray([left_q], dtype=np.int64),
        right_q_index=np.asarray([right_q], dtype=np.int64),
        pair_q_index=np.asarray([pair_q], dtype=np.int64),
        q7_mid=q7[[pair_q]],
        K_left_rect=calibration.K_left_rect,
        K_right_rect=calibration.K_right_rect,
        T_rectified_right_camera_rectified_left_camera=T_right_left,
    )
    manifest = {
        "schema": "super_paper_lnd_stereo_first_frame_manifest_v1",
        "passed": True,
        "method": (
            "fresh extraction of strict raw stereo pair 0; one manual "
            "initialization for upstream SurgicalSAM2; no historical masks"
        ),
        "count": 1,
        "source_pair_count": int(len(stereo_left_index)),
        "pairing_policy": (
            "frozen maximum-cardinality monotonic raw pairing with "
            "|dt| <= 20 ms"
        ),
        "coordinate_frame": (
            "rectified OpenCV left/right camera pixels, origin top-left"
        ),
        "image_size_wh": [width, height],
        "keyframes": [row],
        "annotation_status": "fresh annotation required",
        "upstream_method": {
            "repository": (
                "https://github.com/hanyang-hu/online_dvrk_tracking"
            ),
            "commit": commit,
            "initialization": (
                "one sparse prompt set per eye, both derived only from this "
                "new human first-frame annotation"
            ),
        },
        "paper_lnd_meshes": [
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in mesh_paths
        ],
        "forbidden_inputs": [
            "raw_p420006_stereo_v1/annotations",
            "raw_p420006_stereo_v1/surgicalsam2_stereo_sequence_v1",
            "raw_p420006_stereo_v1/online_stereo_cma_v1",
            "all historical corrected pose drivers",
        ],
        "inputs": {
            "bag": {
                "path": str(args.bag.resolve()),
                "sha256": raw_report["inputs"]["bag"]["sha256"],
                "sha256_source": str(args.raw_report.resolve()),
            },
            "calibration": {
                "path": str(args.calibration.resolve()),
                "sha256": sha256(args.calibration),
            },
            "kinematics": {
                "path": str(args.kinematics.resolve()),
                "sha256": sha256(args.kinematics),
            },
            "raw_model": {
                "path": str(args.raw_model.resolve()),
                "sha256": sha256(args.raw_model),
            },
        },
        "raw_message_decoding_anomalies": decoding_anomalies,
    }
    write_json(args.output_dir / "pair_manifest.json", manifest)
    preview = np.concatenate([left_image, right_image], axis=1)
    preview_path = args.output_dir / "previews/first_strict_stereo_pair.png"
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f"Failed to write {preview_path}")
    print(f"Wrote fresh paper-LND stereo root: {args.output_dir}")
    print(f"Strict source pairs: {len(stereo_left_index)}")
    print(f"First pair source indices: left={left_index}, right={right_index}")


if __name__ == "__main__":
    main()
