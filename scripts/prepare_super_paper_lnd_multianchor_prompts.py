#!/usr/bin/env python3
"""Extract raw strict-stereo prompt anchors for paper-LND SAM2 tracking.

The output is a new, isolated visual-calibration root.  Every image is decoded
from the original ROS bag and rectified with the frozen stereo calibration.
No historical masks or corrected poses are read.
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
RAW_ROOT = (
    REPO_ROOT / "data/super/psm_raw_kinematics_v1（纯机器人学版本）"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_paper_lnd_stereo_multianchor_v2"
)
DEFAULT_SLOTS = (0, 200, 400, 600, 800, 1000, 1200, 1400)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract several strict raw stereo pairs for separate shaft and "
            "distal-tool SAM2 prompts."
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
        "--raw-report",
        type=Path,
        default=RAW_ROOT / "report.json",
    )
    parser.add_argument(
        "--slots",
        type=int,
        nargs="+",
        default=list(DEFAULT_SLOTS),
    )
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
        raise RuntimeError(f"{label} size differs from frozen raw report")
    if stat.st_mtime_ns != int(frozen["mtime_ns"]):
        raise RuntimeError(f"{label} mtime differs from frozen raw report")


def main() -> None:
    args = parse_args()
    for path in (
        args.bag,
        args.calibration,
        args.kinematics,
        args.raw_report,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    slots = np.asarray(sorted(set(args.slots)), dtype=np.int64)
    if len(slots) == 0 or int(slots[0]) != 0:
        raise ValueError("Prompt slots must be non-empty and start at slot 0")

    raw_report = json.loads(args.raw_report.read_text(encoding="utf-8"))
    if raw_report.get("passed") is not True:
        raise RuntimeError("Frozen raw-kinematics report has not passed")
    verify_frozen_input(args.bag, raw_report["inputs"]["bag"], "Original bag")
    verify_frozen_input(
        args.calibration,
        raw_report["inputs"]["camera_calibration"],
        "Stereo calibration",
    )

    with np.load(args.kinematics, allow_pickle=False) as raw:
        stereo_left_index = raw["stereo_left_index"].astype(np.int64)
        stereo_right_index = raw["stereo_right_index"].astype(np.int64)
        left_timestamps_ns = raw["left_timestamps_ros_ns"].astype(np.int64)
        right_timestamps_ns = raw["right_timestamps_ros_ns"].astype(np.int64)
        joint_timestamps_ns = raw["joint_timestamps_ros_ns"].astype(np.int64)
        q7 = raw["q7"].astype(np.float64)
    pair_count = len(stereo_left_index)
    if int(slots[-1]) >= pair_count:
        raise ValueError(
            f"Prompt slot {int(slots[-1])} exceeds {pair_count} pairs"
        )

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} exists; pass --overwrite to recreate it"
            )
        shutil.rmtree(args.output_dir)
    for name in ("images", "annotations", "previews"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)

    calibration = load_stereo_calibration(args.calibration)
    selected_left = stereo_left_index[slots]
    selected_right = stereo_right_index[slots]
    decoding_anomalies: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    preview_rows: list[np.ndarray] = []

    stream = strict_pair_stream(
        bag_path=args.bag,
        calibration=calibration,
        left_indices=selected_left,
        right_indices=selected_right,
        decoding_anomalies=decoding_anomalies,
    )
    for keyframe_index, (stream_slot, left_image, right_image) in enumerate(
        stream
    ):
        if stream_slot != keyframe_index:
            raise RuntimeError("Selected strict-pair stream lost ordering")
        strict_slot = int(slots[keyframe_index])
        left_index = int(selected_left[keyframe_index])
        right_index = int(selected_right[keyframe_index])
        image_paths: dict[str, Path] = {}
        for side, image in (("left", left_image), ("right", right_image)):
            output = (
                args.output_dir
                / "images"
                / f"anchor_{keyframe_index:02d}_{side}_rectified.png"
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
        pair_q = int(
            nearest_indices(
                np.asarray([pair_ns], dtype=np.int64),
                joint_timestamps_ns,
            )[0]
        )
        rows.append(
            {
                "anchor_index": keyframe_index,
                "strict_pair_slot": strict_slot,
                "left_index": left_index,
                "right_index": right_index,
                "left_timestamp_ros_ns": left_ns,
                "right_timestamp_ros_ns": right_ns,
                "pair_timestamp_ros_ns": pair_ns,
                "stereo_delta_ms": float((left_ns - right_ns) * 1.0e-6),
                "pair_q_index": pair_q,
                "q7_mid": q7[pair_q].tolist(),
                "left_image": str(
                    image_paths["left"].relative_to(args.output_dir)
                ),
                "right_image": str(
                    image_paths["right"].relative_to(args.output_dir)
                ),
                "left_image_sha256": sha256(image_paths["left"]),
                "right_image_sha256": sha256(image_paths["right"]),
            }
        )

        scale = 480.0 / left_image.shape[1]
        panels = []
        for side, image in (("L", left_image), ("R", right_image)):
            panel = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
            cv2.rectangle(panel, (0, 0), (panel.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(
                panel,
                f"slot {strict_slot:04d} {side}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            panels.append(panel)
        preview_rows.append(np.concatenate(panels, axis=1))

    manifest = {
        "schema": "super_paper_lnd_stereo_multianchor_prompt_manifest_v2",
        "passed": True,
        "method": (
            "selected strict stereo pairs decoded from the original bag and "
            "rectified with the frozen calibration; no historical masks or "
            "corrected poses"
        ),
        "source_pair_count": pair_count,
        "prompt_anchor_count": len(rows),
        "prompt_slots": slots.tolist(),
        "coordinate_frame": (
            "rectified OpenCV left/right camera pixels, origin top-left"
        ),
        "image_size_wh": list(calibration.image_size),
        "anchors": rows,
        "raw_message_decoding_anomalies": decoding_anomalies,
        "inputs": {
            "bag": str(args.bag),
            "bag_sha256": sha256(args.bag),
            "calibration": str(args.calibration),
            "calibration_sha256": sha256(args.calibration),
            "kinematics": str(args.kinematics),
            "kinematics_sha256": sha256(args.kinematics),
        },
    }
    write_json(args.output_dir / "prompt_manifest.json", manifest)
    contact_sheet = args.output_dir / "previews/prompt_anchor_pairs.png"
    if not cv2.imwrite(str(contact_sheet), np.concatenate(preview_rows, axis=0)):
        raise RuntimeError(f"Failed to write {contact_sheet}")
    print(f"Manifest: {args.output_dir / 'prompt_manifest.json'}")
    print(f"Anchor overview: {contact_sheet}")


if __name__ == "__main__":
    main()
