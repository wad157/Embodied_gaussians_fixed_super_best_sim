#!/usr/bin/env python3
"""Track the raw P420006 stereo sequence with upstream SurgicalSAM2.

This follows online_dvrk_tracking's ``calibrate-online-videos`` observation
path: sparse foreground/background clicks initialize the first frame, then a
SAM2 camera predictor propagates one whole-instrument mask causally.  The only
method extension in this stage is to run one predictor for each rectified eye.

The ten manual polygon pairs are never copied into the output masks.  The
default is the upstream single-initialization behavior: only the first stereo
pair supplies sparse prompts, and both eyes then propagate causally for the
complete sequence.  ``--anchor-policy all`` is retained only as an explicit
diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

from annotate_super_p420006_stereo_keyframes import rasterize_view
from build_super_raw_psm_kinematics import (
    LEFT_TOPIC,
    RIGHT_TOPIC,
    image_message_to_bgr,
    load_stereo_calibration,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data/super/psm_raw_kinematics_v1（纯机器人学版本）"
VISUAL_ROOT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_p420006_stereo_v1"
)
PAPER_REPO = Path("/Media_HDD/jwshan/wad/online_dvrk_tracking")
UPSTREAM_COMMIT = "cb2a264167aaf05b5a9c20da885d48568f78311f"
CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_s.yaml"
CHECKPOINT_NAME = "sam2.1_hiera_s_endo18.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Causally propagate a whole-P420006 mask in both rectified raw "
            "camera streams using the upstream SurgicalSAM2 camera predictor."
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
    parser.add_argument("--visual-root", type=Path, default=VISUAL_ROOT)
    parser.add_argument(
        "--instrument",
        choices=("p420006", "paper_lnd"),
        default="p420006",
        help="Controls provenance/schema labels; tracking itself is CAD-free.",
    )
    parser.add_argument("--paper-repo", type=Path, default=PAPER_REPO)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            PAPER_REPO
            / f"SurgicalSAM2/checkpoints/{CHECKPOINT_NAME}"
        ),
    )
    parser.add_argument("--config", default=CONFIG_NAME)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VISUAL_ROOT / "surgicalsam2_stereo_sequence_v1",
    )
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--cuda-device", type=int, default=1)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Debug only; 0 processes all 1631 strict stereo pairs.",
    )
    parser.add_argument("--preview-interval", type=int, default=100)
    parser.add_argument(
        "--anchor-policy",
        choices=("first", "all"),
        default="first",
        help=(
            "'first' is the exact upstream single-initialization behavior; "
            "'all' restarts the same causal method at each of 10 anchors."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def robust_center_points(mask: np.ndarray, count: int) -> np.ndarray:
    """Place sparse clicks along the principal extent of a manual region."""

    y_values, x_values = np.where(mask)
    if len(x_values) == 0:
        raise ValueError("Cannot sample prompts from an empty mask")
    xy = np.column_stack([x_values, y_values]).astype(np.float32)
    center = xy.mean(axis=0)
    covariance = np.cov(xy - center, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    coordinate = (xy - center) @ axis
    distance = cv2.distanceTransform(
        mask.astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    edges = np.quantile(
        coordinate,
        np.linspace(0.08, 0.92, count + 1),
    )
    points: list[list[float]] = []
    for index in range(count):
        active = (coordinate >= edges[index]) & (
            coordinate <= edges[index + 1]
        )
        candidates = xy[active]
        if len(candidates) == 0:
            continue
        clearances = distance[
            candidates[:, 1].astype(int),
            candidates[:, 0].astype(int),
        ]
        point = candidates[int(np.argmax(clearances))]
        points.append(point.astype(float).tolist())
    if not points:
        maximum = cv2.minMaxLoc(distance)[3]
        points.append([float(maximum[0]), float(maximum[1])])
    return np.asarray(points, dtype=np.float32)


def background_prompt_points(
    instrument_mask: np.ndarray,
    count: int = 8,
) -> np.ndarray:
    """Place background clicks around, but not on, the instrument."""

    height, width = instrument_mask.shape
    y_values, x_values = np.where(instrument_mask)
    x0, x1 = int(x_values.min()), int(x_values.max())
    y0, y1 = int(y_values.min()), int(y_values.max())
    extent = max(x1 - x0 + 1, y1 - y0 + 1)
    padding = max(40, int(round(0.18 * extent)))
    allowed = np.zeros_like(instrument_mask, dtype=np.uint8)
    allowed[
        max(0, y0 - padding) : min(height, y1 + padding + 1),
        max(0, x0 - padding) : min(width, x1 + padding + 1),
    ] = 1
    exclusion = cv2.dilate(
        instrument_mask.astype(np.uint8),
        np.ones((31, 31), dtype=np.uint8),
    )
    allowed[exclusion > 0] = 0
    clearance = cv2.distanceTransform(
        allowed,
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    points: list[list[float]] = []
    suppression_radius = max(25, extent // 8)
    for _ in range(count):
        _, maximum, _, location = cv2.minMaxLoc(clearance)
        if maximum <= 0:
            break
        x, y = location
        points.append([float(x), float(y)])
        cv2.circle(
            clearance,
            (x, y),
            suppression_radius,
            0.0,
            -1,
        )
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def prompts_from_manual_view(
    view: dict[str, Any],
    width: int,
    height: int,
    downsample_factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    manual_labels = rasterize_view(view, width, height)
    positive = np.concatenate(
        [
            robust_center_points(manual_labels == 1, 5),
            robust_center_points(manual_labels == 2, 2),
            robust_center_points(manual_labels == 3, 2),
        ],
        axis=0,
    )
    negative = background_prompt_points(manual_labels > 0)
    points = np.concatenate([positive, negative], axis=0)
    labels = np.concatenate(
        [
            np.ones(len(positive), dtype=np.int64),
            np.zeros(len(negative), dtype=np.int64),
        ]
    )
    return (
        points / float(downsample_factor),
        labels,
        manual_labels,
    )


def union_manual_mask(
    annotation: dict[str, Any],
    side: str,
) -> np.ndarray:
    width, height = annotation["image_size_wh"]
    return (
        rasterize_view(annotation["views"][side], width, height) > 0
    )


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(a & b) / union)


def strict_pair_stream(
    *,
    bag_path: Path,
    calibration: Any,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    decoding_anomalies: list[dict[str, Any]],
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield strict raw pairs in pair-slot order without retaining the bag."""

    left_to_slot = {
        int(source): slot for slot, source in enumerate(left_indices)
    }
    right_to_slot = {
        int(source): slot for slot, source in enumerate(right_indices)
    }
    counters = {LEFT_TOPIC: 0, RIGHT_TOPIC: 0}
    pending: dict[int, dict[str, np.ndarray]] = {}
    next_slot = 0
    typestore = get_typestore(Stores.ROS1_NOETIC)
    with Reader(bag_path) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic in (LEFT_TOPIC, RIGHT_TOPIC)
        ]
        if {connection.topic for connection in connections} != {
            LEFT_TOPIC,
            RIGHT_TOPIC,
        }:
            raise RuntimeError("Original bag is missing a stereo image topic")
        for connection, timestamp_ns, raw in reader.messages(
            connections=connections
        ):
            topic = connection.topic
            index = counters[topic]
            counters[topic] += 1
            slot_by_index = (
                left_to_slot if topic == LEFT_TOPIC else right_to_slot
            )
            slot = slot_by_index.get(index)
            if slot is None:
                continue
            message = typestore.deserialize_ros1(raw, connection.msgtype)
            side = "left" if topic == LEFT_TOPIC else "right"
            message_width = int(message.width)
            message_height = int(message.height)
            message_step = int(message.step)
            encoding = str(message.encoding).lower()
            if (
                encoding == "rgb8"
                and message_step == 2 * message_width
                and len(message.data)
                == message_height * message_step
            ):
                # The original bag contains exactly one mislabeled frame:
                # left index 1441 declares rgb8 but its 2-byte/pixel payload is
                # UYVY.  Decoding the bytes as UYVY reconstructs a normal frame
                # continuous with indices 1440 and 1442.  Never silently skip
                # or duplicate this raw time.
                uyvy = np.frombuffer(message.data, dtype=np.uint8).reshape(
                    message_height,
                    message_width,
                    2,
                )
                image = cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
                decoding_anomalies.append(
                    {
                        "side": side,
                        "source_image_index": index,
                        "strict_pair_slot": slot,
                        "timestamp_ros_ns": int(timestamp_ns),
                        "declared_encoding": str(message.encoding),
                        "declared_step": message_step,
                        "payload_size_bytes": len(message.data),
                        "decoded_as": "UYVY 4:2:2",
                    }
                )
            else:
                image = image_message_to_bgr(message)
            maps = (
                calibration.left_maps
                if side == "left"
                else calibration.right_maps
            )
            rectified = cv2.remap(
                image,
                maps[0],
                maps[1],
                interpolation=cv2.INTER_LINEAR,
            )
            pending.setdefault(slot, {})[side] = rectified
            while (
                next_slot in pending
                and set(pending[next_slot]) == {"left", "right"}
            ):
                pair = pending.pop(next_slot)
                yield next_slot, pair["left"], pair["right"]
                next_slot += 1
                if next_slot == len(left_indices):
                    return
    if next_slot != len(left_indices):
        raise RuntimeError(
            f"Extracted {next_slot}/{len(left_indices)} strict raw pairs"
        )


def build_predictor(
    paper_repo: Path,
    config: str,
    checkpoint: Path,
) -> Any:
    for path in (paper_repo / "SurgicalSAM2", paper_repo):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from sam2.build_sam import build_sam2_camera_predictor

    return build_sam2_camera_predictor(
        config,
        str(checkpoint),
        vos_optimized=True,
    )


def pack_mask(mask: np.ndarray) -> np.ndarray:
    return np.packbits(mask.reshape(-1), bitorder="little")


def tint_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> np.ndarray:
    result = image.copy()
    tint = np.zeros_like(result)
    tint[mask] = color
    result[mask] = cv2.addWeighted(
        result[mask],
        0.60,
        tint[mask],
        0.40,
        0.0,
    )
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(result, contours, -1, color, 2)
    return result


def make_anchor_panel(
    image: np.ndarray,
    sam_mask_small: np.ndarray,
    manual_mask_full: np.ndarray,
    title: str,
) -> np.ndarray:
    sam_full = cv2.resize(
        sam_mask_small.astype(np.uint8),
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    manual_overlay = tint_mask(image, manual_mask_full, (50, 220, 50))
    sam_overlay = tint_mask(image, sam_full, (255, 120, 30))
    scale = 460.0 / image.shape[1]
    manual_overlay = cv2.resize(
        manual_overlay,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )
    sam_overlay = cv2.resize(
        sam_overlay,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )
    panel = np.concatenate([manual_overlay, sam_overlay], axis=1)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        panel,
        f"{title}: manual prompt reference | causal SurgicalSAM2",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def main() -> None:
    args = parse_args()
    schema_prefix = (
        "super_paper_lnd_stereo"
        if args.instrument == "paper_lnd"
        else "super_p420006_stereo"
    )
    if args.downsample_factor < 1:
        raise ValueError("--downsample-factor must be >= 1")
    for path in (
        args.bag,
        args.calibration,
        args.kinematics,
        args.visual_root / "pair_manifest.json",
        args.checkpoint,
        args.paper_repo / ".git",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    import subprocess

    commit = subprocess.run(
        ["git", "-C", str(args.paper_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"Unexpected online_dvrk_tracking commit {commit}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the upstream camera predictor")
    if not 0 <= args.cuda_device < torch.cuda.device_count():
        raise ValueError(
            f"CUDA device {args.cuda_device} is unavailable; "
            f"found {torch.cuda.device_count()} devices"
        )
    torch.cuda.set_device(args.cuda_device)

    manifest = json.loads(
        (args.visual_root / "pair_manifest.json").read_text(encoding="utf-8")
    )
    with np.load(args.kinematics, allow_pickle=False) as raw:
        left_indices = raw["stereo_left_index"].astype(np.int64)
        right_indices = raw["stereo_right_index"].astype(np.int64)
        left_timestamps = raw["left_timestamps_ros_ns"][
            left_indices
        ].astype(np.int64)
        right_timestamps = raw["right_timestamps_ros_ns"][
            right_indices
        ].astype(np.int64)
    pair_count = len(left_indices)
    if args.max_pairs > 0:
        pair_count = min(pair_count, args.max_pairs)
        left_indices = left_indices[:pair_count]
        right_indices = right_indices[:pair_count]
        left_timestamps = left_timestamps[:pair_count]
        right_timestamps = right_timestamps[:pair_count]
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} is not empty; pass --overwrite"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "previews").mkdir(exist_ok=True)

    anchor_rows = {
        int(row["strict_pair_slot"]): row
        for row in manifest["keyframes"]
        if int(row["strict_pair_slot"]) < pair_count
    }
    if 0 not in anchor_rows:
        raise RuntimeError("The first strict stereo pair is not an anchor")
    anchor_annotations: dict[int, dict[str, Any]] = {}
    anchor_annotation_paths: dict[int, Path] = {}
    prompts_by_slot: dict[
        int,
        dict[str, tuple[np.ndarray, np.ndarray]],
    ] = {}
    source_width, source_height = manifest["image_size_wh"]
    for slot, row in anchor_rows.items():
        keyframe = int(row["keyframe_index"])
        annotation_path = (
            args.visual_root
            / "annotations"
            / f"keyframe_{keyframe:02d}.json"
        )
        annotation = json.loads(
            annotation_path.read_text(encoding="utf-8")
        )
        if annotation["image_size_wh"] != [source_width, source_height]:
            raise RuntimeError(
                f"Annotation size mismatch in {annotation_path}"
            )
        anchor_annotations[slot] = annotation
        anchor_annotation_paths[slot] = annotation_path
        prompts_by_slot[slot] = {}
        for side in ("left", "right"):
            points, labels, _manual_labels = prompts_from_manual_view(
                annotation["views"][side],
                source_width,
                source_height,
                args.downsample_factor,
            )
            prompts_by_slot[slot][side] = (points, labels)
    conditioning_slots = (
        {0} if args.anchor_policy == "first" else set(anchor_rows)
    )
    if (
        source_width % args.downsample_factor != 0
        or source_height % args.downsample_factor != 0
    ):
        raise ValueError("Image size is not divisible by downsample factor")
    width = source_width // args.downsample_factor
    height = source_height // args.downsample_factor
    print(
        "Building two upstream SurgicalSAM2 camera predictors "
        f"on cuda:{args.cuda_device}",
        flush=True,
    )
    predictor_left = build_predictor(
        args.paper_repo,
        args.config,
        args.checkpoint,
    )
    predictor_right = build_predictor(
        args.paper_repo,
        args.config,
        args.checkpoint,
    )
    predictors = {"left": predictor_left, "right": predictor_right}

    packed_width = math.ceil(width * height / 8)
    masks_packed = {
        "left": np.empty((pair_count, packed_width), dtype=np.uint8),
        "right": np.empty((pair_count, packed_width), dtype=np.uint8),
    }
    mask_area = {
        "left": np.empty(pair_count, dtype=np.int32),
        "right": np.empty(pair_count, dtype=np.int32),
    }
    mask_bbox = {
        "left": np.full((pair_count, 4), -1, dtype=np.int32),
        "right": np.full((pair_count, 4), -1, dtype=np.int32),
    }
    anchor_metrics: list[dict[str, Any]] = []
    anchor_panels: list[np.ndarray] = []
    timings: list[float] = []
    decoding_anomalies: list[dict[str, Any]] = []
    calibration = load_stereo_calibration(args.calibration)

    for pair_slot, left_full, right_full in strict_pair_stream(
        bag_path=args.bag,
        calibration=calibration,
        left_indices=left_indices,
        right_indices=right_indices,
        decoding_anomalies=decoding_anomalies,
    ):
        images_full = {"left": left_full, "right": right_full}
        images_small = {
            side: cv2.resize(
                image,
                (width, height),
                interpolation=cv2.INTER_AREA,
            )
            for side, image in images_full.items()
        }
        start = time.perf_counter()
        output_masks: dict[str, np.ndarray] = {}
        with torch.inference_mode(), torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):
            for side in ("left", "right"):
                predictor = predictors[side]
                if pair_slot in conditioning_slots:
                    predictor.frame_idx = 0
                    predictor.load_first_frame(images_small[side])
                    prompt_points, prompt_labels = prompts_by_slot[
                        pair_slot
                    ][side]
                    _, _, logits = predictor.add_new_points(
                        frame_idx=0,
                        obj_id=0,
                        points=prompt_points,
                        labels=prompt_labels,
                    )
                else:
                    _, logits = predictor.track(images_small[side])
                output_masks[side] = (
                    logits.squeeze() > 0
                ).detach().cpu().numpy()
        if pair_slot > 0:
            torch.cuda.synchronize(args.cuda_device)
            timings.append(time.perf_counter() - start)
        for side in ("left", "right"):
            mask = output_masks[side].astype(bool)
            if mask.shape != (height, width):
                raise RuntimeError(
                    f"Unexpected {side} mask shape {mask.shape}"
                )
            masks_packed[side][pair_slot] = pack_mask(mask)
            mask_area[side][pair_slot] = int(mask.sum())
            y_values, x_values = np.where(mask)
            if len(x_values):
                mask_bbox[side][pair_slot] = [
                    int(x_values.min()),
                    int(y_values.min()),
                    int(x_values.max()),
                    int(y_values.max()),
                ]
        if pair_slot in anchor_rows:
            row = anchor_rows[pair_slot]
            annotation = anchor_annotations[pair_slot]
            measurement: dict[str, Any] = {
                "keyframe_index": int(row["keyframe_index"]),
                "strict_pair_slot": pair_slot,
                "used_for_prompting": pair_slot in conditioning_slots,
                "views": {},
            }
            side_panels: list[np.ndarray] = []
            for side in ("left", "right"):
                manual_full = union_manual_mask(annotation, side)
                sam_full = cv2.resize(
                    output_masks[side].astype(np.uint8),
                    (source_width, source_height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                measurement["views"][side] = {
                    "iou_with_manual_union": mask_iou(
                        sam_full,
                        manual_full,
                    ),
                    "sam2_area_px_at_full_resolution": int(
                        sam_full.sum()
                    ),
                    "manual_union_area_px": int(manual_full.sum()),
                }
                side_panels.append(
                    make_anchor_panel(
                        images_full[side],
                        output_masks[side],
                        manual_full,
                        f"KF {int(row['keyframe_index']):02d} {side}",
                    )
                )
            anchor_metrics.append(measurement)
            anchor_panels.append(np.concatenate(side_panels, axis=0))
        if (
            pair_slot == 0
            or (pair_slot + 1) % args.preview_interval == 0
            or pair_slot + 1 == pair_count
        ):
            recent = (
                float(np.mean(timings[-20:])) if timings else float("nan")
            )
            print(
                f"[{pair_slot + 1:04d}/{pair_count:04d}] "
                f"causal stereo masks; recent {recent:.3f}s/pair",
                flush=True,
            )

    masks_path = args.output_dir / "stereo_surgicalsam2_masks.npz"
    np.savez_compressed(
        masks_path,
        schema=np.asarray(
            f"{schema_prefix}_surgicalsam2_sequence_v1"
        ),
        mask_shape=np.asarray([height, width], dtype=np.int64),
        bitorder=np.asarray("little"),
        left_masks_packbits=masks_packed["left"],
        right_masks_packbits=masks_packed["right"],
        left_area_px=mask_area["left"],
        right_area_px=mask_area["right"],
        left_bbox_xyxy=mask_bbox["left"],
        right_bbox_xyxy=mask_bbox["right"],
        stereo_left_index=left_indices,
        stereo_right_index=right_indices,
        left_timestamp_ros_ns=left_timestamps,
        right_timestamp_ros_ns=right_timestamps,
    )
    contact_sheet_path = (
        args.output_dir
        / "previews/heldout_manual_vs_causal_surgicalsam2.png"
    )
    if anchor_panels:
        if not cv2.imwrite(
            str(contact_sheet_path),
            np.concatenate(anchor_panels, axis=0),
        ):
            raise RuntimeError(f"Failed to write {contact_sheet_path}")

    report = {
        "schema": f"{schema_prefix}_surgicalsam2_report_v1",
        "passed": len(anchor_metrics) > 0,
        "method_fidelity": {
            "upstream_repository": (
                "https://github.com/hanyang-hu/online_dvrk_tracking"
            ),
            "upstream_commit": commit,
            "upstream_entrypoint": "scripts/video_calibration.py",
            "preserved": [
                "SurgicalSAM2 endoscopic checkpoint",
                "SAM2 camera predictor with vos_optimized=True",
                "bfloat16 CUDA inference",
                "sparse segment-start foreground/background prompts",
                "causal frame-to-frame mask propagation",
                "one whole-instrument binary mask per camera",
            ],
            "stereo_extension": (
                "two independent predictors process timestamp-paired "
                "rectified eyes with identical settings"
            ),
            "long_sequence_extension": (
                "with anchor_policy=all, each of ten annotated times starts "
                "a new upstream-style causal segment; anchor_policy=first "
                "retains the exact single-initialization baseline"
            ),
            "not_yet_in_this_stage": (
                "CMA-ES pose/joint correction is performed by the next "
                "shared-state stereo stage"
            ),
        },
        "inputs": {
            "bag": str(args.bag),
            "bag_sha256": sha256(args.bag),
            "calibration": str(args.calibration),
            "calibration_sha256": sha256(args.calibration),
            "kinematics": str(args.kinematics),
            "manual_annotations": [
                {
                    "keyframe_index": int(
                        anchor_rows[slot]["keyframe_index"]
                    ),
                    "strict_pair_slot": slot,
                    "path": str(anchor_annotation_paths[slot]),
                    "sha256": sha256(anchor_annotation_paths[slot]),
                    "used_for_prompting": slot in conditioning_slots,
                }
                for slot in sorted(anchor_rows)
            ],
        },
        "model": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "checkpoint_size_bytes": args.checkpoint.stat().st_size,
            "config": args.config,
            "cuda_device_index": args.cuda_device,
            "cuda_device_name": torch.cuda.get_device_name(
                args.cuda_device
            ),
            "torch_version": torch.__version__,
        },
        "sequence": {
            "complete_strict_pair_sequence": pair_count
            == int(manifest["source_pair_count"]),
            "pair_count": pair_count,
            "source_image_size_wh": [source_width, source_height],
            "tracking_image_size_wh": [width, height],
            "downsample_factor": args.downsample_factor,
            "mean_seconds_per_stereo_pair_excluding_first": (
                float(np.mean(timings)) if timings else None
            ),
            "raw_message_decoding_anomalies": decoding_anomalies,
            **(
                {}
                if args.anchor_policy == "first"
                else {
                    "diagnostic_anchor_policy": args.anchor_policy,
                    "diagnostic_causal_segment_start_slots": sorted(
                        conditioning_slots
                    ),
                }
            ),
        },
        "manual_annotation_policy": {
            "first_pair": (
                "new manual part polygons are used only to place sparse "
                "foreground/background prompts in each eye"
            ),
            "remaining_pairs": (
                "no manual annotation; masks are propagated causally by "
                "the two SurgicalSAM2 camera predictors"
            ),
            "manual_pixels_copied_to_output": False,
            "confidence_downweighting": False,
        },
        "prompts": [
            {
                "keyframe_index": int(
                    anchor_rows[slot]["keyframe_index"]
                ),
                "strict_pair_slot": slot,
                "used_for_prompting": slot in conditioning_slots,
                "views": {
                    side: {
                        "points_xy_at_tracking_resolution": (
                            prompts_by_slot[slot][side][0].tolist()
                        ),
                        "labels": (
                            prompts_by_slot[slot][side][1].tolist()
                        ),
                    }
                    for side in ("left", "right")
                },
            }
            for slot in sorted(anchor_rows)
        ],
        "anchor_diagnostics": anchor_metrics,
        "outputs": {
            "packed_masks": str(masks_path),
            "contact_sheet": (
                str(contact_sheet_path) if anchor_panels else None
            ),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Packed masks: {masks_path}")
    print(f"Report: {report_path}")
    if anchor_panels:
        print(f"Anchor comparison: {contact_sheet_path}")


if __name__ == "__main__":
    main()
