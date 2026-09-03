#!/usr/bin/env python3
"""Select pre-contact SUPER frames for multiview tissue reconstruction.

This is stage A of the v10 tissue-reconstruction plan in ``PROGRESS.md``.
It does not build a scene asset or modify any active runtime input.  The
script:

1. verifies the frozen v9/table/camera/PSM hashes before writing output;
2. recovers the complete left/right image timestamps from the original bag;
3. finds a conservative pre-contact interval from the recorded jaw command;
4. chooses quality-screened candidates and provisional left-view set-cover
   keyframes; and
5. writes an auditable observation manifest and visual previews.

The keyframes are deliberately marked provisional.  Stage B must create
independent right-view tissue/tool masks and rerun the final stereo set-cover
selection before any geometry is fused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOT = REPO_ROOT / "data/super/grasp5_native"
OFFLINE_ROOT = REPO_ROOT / "data/super/grasp5_offline_demo"
TRACK_ROOT = REPO_ROOT / "data/super/psm_tracking"

LEFT_TOPIC = "/stereo/slave/left/image"
RIGHT_TOPIC = "/stereo/slave/right/image"
JOINT_TOPIC = "/dvrk/PSM1/slave/state_joint_current"

FROZEN_ASSETS = {
    "table_frame": (
        REPO_ROOT / "data/super/table_frame.json",
        "6dddc2178cdf816f5dada5febdd528f80e42d52e631076e1f5f4a952297adecf",
    ),
    "cameras": (
        OFFLINE_ROOT / "cameras.json",
        "e1e7b7e7e21ca8a9409c88a29409d2e9ad6b783a85b277e71cec0c84340ce4ef",
    ),
    "v9_tissue": (
        NATIVE_ROOT / "bodies_v9_dense_0p5mm_rigid_tissue/tissue.json",
        "21974bbb10fa4d756ba615f93b74b460952ca394fa8b679aab3cfc265785d872",
    ),
    "v9_ground": (
        NATIVE_ROOT / "bodies_v9_dense_0p5mm_rigid_tissue/ground.json",
        "8e769cd7fd92d2b5eab2f7ca306986d94206fcdf9726b02b3fc76030bc8f8ecb",
    ),
    "v9_ground_plane": (
        NATIVE_ROOT / "bodies_v9_dense_0p5mm_rigid_tissue/ground_plane.json",
        "1a4674957ce063a47f1dcb16a44781b3df77ee4af711a6c105e4e5b8de0c5d96",
    ),
    "corrected_psm_driver": (
        TRACK_ROOT / "psm_part_corrected_pose_driver.npz",
        "21df09849b08d6cef1ae47c694e7400b6df5228e0805c5c35ecf3d68e2ef648a",
    ),
    "depth_then_visual_psm_driver": (
        TRACK_ROOT / "psm_depth_then_visual_pose_driver_candidate.npz",
        "72bf8aaade1e935445c6908e0519a8c63c9d202f31c21710e598cef91938b07f",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select pre-contact frames for SUPER v10 tissue reconstruction."
    )
    parser.add_argument(
        "--bag",
        type=Path,
        default=REPO_ROOT / "data/grasp5/grasp5.bag",
    )
    parser.add_argument("--rgb-dir", type=Path, default=NATIVE_ROOT / "rgb")
    parser.add_argument(
        "--left-metadata",
        type=Path,
        default=OFFLINE_ROOT / "videos/stereo_left.json",
    )
    parser.add_argument(
        "--right-metadata",
        type=Path,
        default=OFFLINE_ROOT / "videos/stereo_right.json",
    )
    parser.add_argument(
        "--robots",
        type=Path,
        default=OFFLINE_ROOT / "robots.json",
    )
    parser.add_argument(
        "--tissue-masks",
        type=Path,
        default=NATIVE_ROOT / "visual_force_masks_v1/tissue_masks_packbits.npy",
    )
    parser.add_argument(
        "--part-masks",
        type=Path,
        default=TRACK_ROOT / "part_masks_full_sequence/part_masks_full.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=NATIVE_ROOT / "tissue_multiview_v1",
    )
    parser.add_argument("--candidate-count", type=int, default=25)
    parser.add_argument("--keyframe-min", type=int, default=6)
    parser.add_argument("--keyframe-max", type=int, default=12)
    parser.add_argument("--sample-stride", type=int, default=3)
    parser.add_argument(
        "--jaw-drop-threshold-rad",
        type=float,
        default=0.05,
        help="Jaw-command drop from the initial median that marks closing onset.",
    )
    parser.add_argument(
        "--precontact-margin-seconds",
        type=float,
        default=1.5,
        help="Safety margin removed before the first jaw-closing command.",
    )
    parser.add_argument(
        "--tool-dilation-fullres-px",
        type=int,
        default=30,
        help="Conservative full-resolution dilation around image tool masks.",
    )
    parser.add_argument(
        "--coverage-target",
        type=float,
        default=0.999,
        help="Target fraction of the candidate left-view visibility union.",
    )
    parser.add_argument(
        "--minimum-coverage-gain",
        type=float,
        default=1.0e-4,
        help="Minimum measurable marginal gain as a fraction of coverage domain.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this stage's own manifest/coverage/preview files.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def verify_frozen_assets() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    failures = []
    for label, (path, expected) in FROZEN_ASSETS.items():
        if not path.is_file():
            failures.append(f"{label}: missing {path}")
            results[label] = {
                "path": str(path.resolve()),
                "expected_sha256": expected,
                "actual_sha256": None,
                "matches": False,
            }
            continue
        actual = sha256(path)
        matches = actual == expected
        results[label] = {
            "path": str(path.resolve()),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": matches,
        }
        if not matches:
            failures.append(f"{label}: expected {expected}, got {actual}")

    plane_path = FROZEN_ASSETS["v9_ground_plane"][0]
    if plane_path.is_file():
        plane_data = read_json(plane_path)
        plane = plane_data.get("plane", plane_data.get("ground_plane"))
        exact_plane = plane == [0, 0, 1, 0] or plane == [0.0, 0.0, 1.0, 0.0]
        results["ground_plane_value"] = {
            "value": plane,
            "expected": [0.0, 0.0, 1.0, 0.0],
            "matches": exact_plane,
        }
        if not exact_plane:
            failures.append(f"ground plane is not exact z=0: {plane}")

    if failures:
        raise RuntimeError(
            "Frozen coordinate/runtime gate failed before output creation:\n- "
            + "\n- ".join(failures)
        )
    return results


def image_paths(rgb_dir: Path, side: str) -> list[Path]:
    paths = sorted(rgb_dir.glob(f"*-{side}.png"))
    expected = [f"{index:06d}-{side}.png" for index in range(len(paths))]
    actual = [path.name for path in paths]
    if actual != expected:
        raise RuntimeError(f"{side} image sequence is not contiguous from frame 0")
    return paths


def recover_image_timestamps(
    bag: Path,
    expected_left: int,
    expected_right: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Read raw message times without decoding the 28 GB image payloads."""

    from rosbags.rosbag1 import Reader

    left: list[float] = []
    right: list[float] = []
    first_timestamp_ns: int | None = None
    with Reader(bag) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic in {LEFT_TOPIC, RIGHT_TOPIC, JOINT_TOPIC}
        ]
        for connection, timestamp_ns, _raw in reader.messages(
            connections=connections
        ):
            timestamp_ns = int(timestamp_ns)
            if first_timestamp_ns is None:
                first_timestamp_ns = timestamp_ns
            relative = float(timestamp_ns - first_timestamp_ns) * 1.0e-9
            if connection.topic == LEFT_TOPIC and len(left) < expected_left:
                left.append(relative)
            elif connection.topic == RIGHT_TOPIC and len(right) < expected_right:
                right.append(relative)
            if len(left) == expected_left and len(right) == expected_right:
                break

    if first_timestamp_ns is None:
        raise RuntimeError(f"No target messages found in {bag}")
    if len(left) != expected_left or len(right) != expected_right:
        raise RuntimeError(
            "Bag/image count mismatch: "
            f"left={len(left)}/{expected_left}, right={len(right)}/{expected_right}"
        )
    return (
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
        first_timestamp_ns,
    )


def nearest_indices(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    after = np.searchsorted(reference, query, side="left")
    after = np.clip(after, 0, len(reference) - 1)
    before = np.clip(after - 1, 0, len(reference) - 1)
    use_after = np.abs(reference[after] - query) < np.abs(reference[before] - query)
    return np.where(use_after, after, before).astype(np.int32)


def unpack_tissue_mask(
    packed_masks: np.ndarray,
    frame: int,
    full_width: int,
) -> np.ndarray:
    full = np.unpackbits(
        packed_masks[frame], axis=1, count=full_width
    ).astype(bool)
    return full[::2, ::2]


def unpack_tool_mask(
    packed_masks: np.ndarray,
    frame: int,
    mask_shape: tuple[int, int],
) -> np.ndarray:
    height, width = mask_shape
    parts = np.unpackbits(
        packed_masks[frame],
        axis=-1,
        count=height * width,
    ).reshape(packed_masks.shape[1], height, width)
    return np.any(parts.astype(bool), axis=0)


def load_half_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    return cv2.resize(
        image,
        (width // 2, height // 2),
        interpolation=cv2.INTER_AREA,
    )


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2:
        return 0.0
    left_std = float(left.std())
    right_std = float(right.std())
    if left_std < 1.0e-8 or right_std < 1.0e-8:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def mask_centroid(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [float(xs.mean()), float(ys.mean())]


def analyze_frame(
    frame: int,
    right_frame: int,
    left_path: Path,
    right_path: Path,
    tissue_packed: np.ndarray,
    tool_packed: np.ndarray,
    tool_shape: tuple[int, int],
    dilation_kernel: np.ndarray,
    reference_gray: np.ndarray,
    reference_tissue: np.ndarray,
    reference_tool_dilated: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    left = load_half_image(left_path)
    right = load_half_image(right_path)
    if left.shape != right.shape:
        raise RuntimeError(
            f"Left/right image shape mismatch at {frame}/{right_frame}"
        )
    tissue = unpack_tissue_mask(tissue_packed, frame, left.shape[1] * 2)
    tool = unpack_tool_mask(tool_packed, frame, tool_shape)
    if tissue.shape != tool.shape or tissue.shape != left.shape[:2]:
        raise RuntimeError(
            f"Mask/image shape mismatch: tissue={tissue.shape}, "
            f"tool={tool.shape}, image={left.shape[:2]}"
        )
    tool_dilated = cv2.dilate(
        tool.astype(np.uint8), dilation_kernel
    ).astype(bool)
    # Stage A is selecting complementary tool-occlusion positions while the
    # tissue is still static.  The propagated SAM2 tissue mask slowly shrinks
    # even in this interval, so using it as the set-cover domain would reward
    # segmentation drift.  Keep the manually verified frame-0 tissue support
    # fixed here; per-frame tissue masks remain part of the quality audit.
    visible = reference_tissue & ~tool_dilated

    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    stable_region = (
        reference_tissue
        & tissue
        & ~reference_tool_dilated
        & ~tool_dilated
    )
    reference_values = reference_gray[stable_region].astype(np.float32)
    current_values = left_gray[stable_region].astype(np.float32)
    if len(current_values):
        current_values += float(
            np.median(reference_values) - np.median(current_values)
        )
        difference = np.abs(current_values - reference_values)
        difference_p50 = float(np.quantile(difference, 0.50))
        difference_p90 = float(np.quantile(difference, 0.90))
    else:
        difference_p50 = float("inf")
        difference_p90 = float("inf")

    left_laplacian = cv2.Laplacian(left_gray, cv2.CV_32F)
    right_laplacian = cv2.Laplacian(right_gray, cv2.CV_32F)
    right_roi = cv2.dilate(
        tissue.astype(np.uint8), np.ones((11, 11), np.uint8)
    ).astype(bool)
    tissue_union = np.logical_or(reference_tissue, tissue)
    tissue_intersection = np.logical_and(reference_tissue, tissue)
    metrics = {
        "left_frame": int(frame),
        "right_frame": int(right_frame),
        "tissue_area_halfres_px": int(tissue.sum(dtype=np.int64)),
        "tool_area_halfres_px": int(tool.sum(dtype=np.int64)),
        "visible_tissue_halfres_px": int(visible.sum(dtype=np.int64)),
        "tool_centroid_halfres_xy": mask_centroid(tool),
        "tissue_iou_with_frame0": float(
            tissue_intersection.sum(dtype=np.int64)
            / max(tissue_union.sum(dtype=np.int64), 1)
        ),
        "frame0_stable_region_halfres_px": int(
            stable_region.sum(dtype=np.int64)
        ),
        "frame0_gray_correlation": safe_correlation(
            reference_values, current_values
        ),
        "frame0_gray_absdiff_p50": difference_p50,
        "frame0_gray_absdiff_p90": difference_p90,
        "left_laplacian_variance": float(left_laplacian[visible].var()),
        "right_laplacian_variance": float(right_laplacian[right_roi].var()),
        "left_saturated_fraction": float(
            (left[visible].max(axis=1) >= 250).mean()
        ),
        "right_saturated_fraction": float(
            (right[right_roi].max(axis=1) >= 250).mean()
        ),
    }
    return metrics, visible, tissue, tool_dilated


def quality_gate_and_score(metrics: dict[str, Any]) -> tuple[bool, float, list[str]]:
    reasons = []
    if metrics["frame0_gray_correlation"] < 0.94:
        reasons.append("frame0_gray_correlation<0.94")
    if metrics["left_laplacian_variance"] < 60.0:
        reasons.append("left_laplacian_variance<60")
    if metrics["right_laplacian_variance"] < 60.0:
        reasons.append("right_laplacian_variance<60")
    if metrics["left_saturated_fraction"] > 0.02:
        reasons.append("left_saturated_fraction>0.02")
    if metrics["right_saturated_fraction"] > 0.02:
        reasons.append("right_saturated_fraction>0.02")
    if metrics["visible_tissue_halfres_px"] <= 0:
        reasons.append("empty_visible_tissue")

    correlation = np.clip(
        (metrics["frame0_gray_correlation"] - 0.94) / 0.06, 0.0, 1.0
    )
    blur = np.clip(
        min(
            metrics["left_laplacian_variance"],
            metrics["right_laplacian_variance"],
        )
        / 100.0,
        0.0,
        1.0,
    )
    saturation = 1.0 - np.clip(
        max(
            metrics["left_saturated_fraction"],
            metrics["right_saturated_fraction"],
        )
        / 0.02,
        0.0,
        1.0,
    )
    score = float(0.50 * correlation + 0.30 * blur + 0.20 * saturation)
    return not reasons, score, reasons


def select_temporal_candidates(
    records: list[dict[str, Any]],
    count: int,
    first_frame: int,
) -> list[dict[str, Any]]:
    eligible = [record for record in records if record["quality_gate_passed"]]
    if len(eligible) < count:
        raise RuntimeError(
            f"Only {len(eligible)} quality-passing samples for {count} candidates"
        )

    selected: list[dict[str, Any]] = []
    times = np.asarray([record["left_timestamp"] for record in eligible])
    edges = np.linspace(times.min(), times.max() + 1.0e-12, count + 1)
    used: set[int] = set()
    for bin_index in range(count):
        if bin_index == count - 1:
            in_bin = [
                record
                for record in eligible
                if edges[bin_index] <= record["left_timestamp"] <= edges[bin_index + 1]
                and record["left_frame"] not in used
            ]
        else:
            in_bin = [
                record
                for record in eligible
                if edges[bin_index] <= record["left_timestamp"] < edges[bin_index + 1]
                and record["left_frame"] not in used
            ]
        if bin_index == 0:
            first = [
                record
                for record in in_bin
                if record["left_frame"] == first_frame
            ]
            if first:
                in_bin = first
        if not in_bin:
            midpoint = 0.5 * (edges[bin_index] + edges[bin_index + 1])
            in_bin = sorted(
                (
                    record
                    for record in eligible
                    if record["left_frame"] not in used
                ),
                key=lambda record: abs(record["left_timestamp"] - midpoint),
            )[:1]
        chosen = max(
            in_bin,
            key=lambda record: (
                record["quality_score"],
                record["visible_tissue_halfres_px"],
            ),
        )
        selected.append(chosen)
        used.add(chosen["left_frame"])

    if len(selected) != count or len(used) != count:
        raise RuntimeError("Candidate selection did not produce unique requested frames")
    return sorted(selected, key=lambda record: record["left_frame"])


def greedy_set_cover(
    candidates: list[dict[str, Any]],
    visibility_by_frame: dict[int, np.ndarray],
    minimum_count: int,
    maximum_count: int,
    target: float,
    minimum_gain: float,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    domain = np.zeros_like(next(iter(visibility_by_frame.values())), dtype=bool)
    for candidate in candidates:
        domain |= visibility_by_frame[candidate["left_frame"]]
    domain_count = int(domain.sum(dtype=np.int64))
    if domain_count == 0:
        raise RuntimeError("Candidate visibility union is empty")

    covered = np.zeros_like(domain)
    remaining = {candidate["left_frame"]: candidate for candidate in candidates}
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < maximum_count:
        scored = []
        for frame, candidate in remaining.items():
            visibility = visibility_by_frame[frame]
            gain = int((visibility & domain & ~covered).sum(dtype=np.int64))
            scored.append(
                (
                    gain,
                    candidate["quality_score"],
                    candidate["visible_tissue_halfres_px"],
                    -frame,
                    frame,
                    candidate,
                )
            )
        gain, _, _, _, frame, candidate = max(scored)
        gain_fraction = float(gain / domain_count)
        if len(selected) >= minimum_count and (
            gain_fraction < minimum_gain
            or float((covered & domain).sum(dtype=np.int64) / domain_count)
            >= target
        ):
            break
        if gain_fraction < minimum_gain:
            raise RuntimeError(
                "Could not reach minimum keyframe count with measurable gains: "
                f"selected={len(selected)}, next_gain={gain_fraction:.8f}"
            )
        covered |= visibility_by_frame[frame]
        cumulative = float((covered & domain).sum(dtype=np.int64) / domain_count)
        selected.append(
            {
                "selection_order": len(selected),
                "left_frame": int(frame),
                "right_frame": int(candidate["right_frame"]),
                "left_timestamp": float(candidate["left_timestamp"]),
                "right_timestamp": float(candidate["right_timestamp"]),
                "right_minus_left_ms": float(
                    candidate["right_minus_left_ms"]
                ),
                "quality_score": float(candidate["quality_score"]),
                "coverage_gain_halfres_px": int(gain),
                "coverage_gain_fraction": gain_fraction,
                "cumulative_left_proxy_coverage": cumulative,
            }
        )
        del remaining[frame]

    if len(selected) < minimum_count:
        raise RuntimeError(
            f"Selected only {len(selected)} keyframes, need {minimum_count}"
        )
    return selected, domain


def annotated_pair(
    left_path: Path,
    right_path: Path,
    tissue: np.ndarray,
    tool_dilated: np.ndarray,
    lines: list[str],
    width_each: int = 480,
) -> np.ndarray:
    left = load_half_image(left_path)
    right = load_half_image(right_path)
    overlay = left.copy()
    visible = tissue & ~tool_dilated
    tint = np.zeros_like(overlay)
    tint[:, :] = (40, 170, 40)
    overlay[visible] = cv2.addWeighted(
        overlay, 0.72, tint, 0.28, 0.0
    )[visible]
    contours, _ = cv2.findContours(
        tissue.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    tool_contours, _ = cv2.findContours(
        tool_dilated.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, tool_contours, -1, (0, 80, 255), 1)

    scale = width_each / overlay.shape[1]
    height = int(round(overlay.shape[0] * scale))
    overlay = cv2.resize(overlay, (width_each, height), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, (width_each, height), interpolation=cv2.INTER_AREA)
    pair = np.concatenate([overlay, right], axis=1)
    header_height = 24 + 18 * len(lines)
    canvas = np.zeros((height + header_height, pair.shape[1], 3), dtype=np.uint8)
    canvas[header_height:] = pair
    for line_index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (8, 20 + 18 * line_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        "left: green visible tissue / red dilated tool",
        (8, header_height - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (120, 230, 120),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "timestamp-matched right",
        (width_each + 8, header_height - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas


def contact_sheet(tiles: list[np.ndarray], columns: int) -> np.ndarray:
    if not tiles:
        raise ValueError("No tiles for contact sheet")
    height = max(tile.shape[0] for tile in tiles)
    width = max(tile.shape[1] for tile in tiles)
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet[
            row * height : row * height + tile.shape[0],
            column * width : column * width + tile.shape[1],
        ] = tile
    return sheet


def main() -> None:
    args = parse_args()
    if args.candidate_count < 20 or args.candidate_count > 30:
        raise ValueError("--candidate-count must be in the planned [20, 30] range")
    if not (6 <= args.keyframe_min <= args.keyframe_max <= 12):
        raise ValueError("Require 6 <= keyframe-min <= keyframe-max <= 12")
    if args.sample_stride < 1:
        raise ValueError("--sample-stride must be positive")
    if not 0.0 < args.coverage_target <= 1.0:
        raise ValueError("--coverage-target must be in (0, 1]")

    print("[phase-a] verifying frozen v9/table/camera/PSM assets", flush=True)
    frozen_results = verify_frozen_assets()

    manifest_path = args.output_dir / "observations.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{manifest_path} already exists; pass --overwrite to replace stage-A files"
        )

    left_paths = image_paths(args.rgb_dir, "left")
    right_paths = image_paths(args.rgb_dir, "right")
    paired_left_metadata = read_json(args.left_metadata)
    paired_right_metadata = read_json(args.right_metadata)

    left_timestamp_cache = args.output_dir / "timestamps_left_native.npy"
    right_timestamp_cache = args.output_dir / "timestamps_right_native.npy"
    cache_valid = False
    if (
        args.overwrite
        and manifest_path.is_file()
        and left_timestamp_cache.is_file()
        and right_timestamp_cache.is_file()
    ):
        previous = read_json(manifest_path)
        previous_bag = previous.get("inputs", {}).get("bag_identity", {})
        cache_valid = bool(
            previous_bag.get("size_bytes") == args.bag.stat().st_size
            and previous_bag.get("mtime_ns") == args.bag.stat().st_mtime_ns
        )
        if cache_valid:
            left_timestamps = np.load(left_timestamp_cache)
            right_timestamps = np.load(right_timestamp_cache)
            first_ros_timestamp_ns = int(
                previous_bag["first_ros_timestamp_ns"]
            )
            cache_valid = bool(
                left_timestamps.shape == (len(left_paths),)
                and right_timestamps.shape == (len(right_paths),)
                and array_sha256(left_timestamps)
                == previous_bag["left_timestamps_float64_sha256"]
                and array_sha256(right_timestamps)
                == previous_bag["right_timestamps_float64_sha256"]
            )
    if cache_valid:
        print(
            "[phase-a] reusing verified native timestamp cache "
            f"(left={len(left_paths)}, right={len(right_paths)})",
            flush=True,
        )
    else:
        print(
            "[phase-a] recovering complete native timestamps from bag "
            f"(left={len(left_paths)}, right={len(right_paths)})",
            flush=True,
        )
        left_timestamps, right_timestamps, first_ros_timestamp_ns = (
            recover_image_timestamps(args.bag, len(left_paths), len(right_paths))
        )
    paired_left_timestamps = np.asarray(
        paired_left_metadata["timestamps"], dtype=np.float64
    )
    paired_right_timestamps = np.asarray(
        paired_right_metadata["timestamps"], dtype=np.float64
    )
    left_metadata_error = float(
        np.max(
            np.abs(
                left_timestamps[: len(paired_left_timestamps)]
                - paired_left_timestamps
            )
        )
    )
    right_metadata_error = float(
        np.max(
            np.abs(
                right_timestamps[: len(paired_right_timestamps)]
                - paired_right_timestamps
            )
        )
    )
    if left_metadata_error > 1.0e-12 or right_metadata_error > 1.0e-12:
        raise RuntimeError(
            "Recovered bag timestamps do not match frozen paired metadata: "
            f"left={left_metadata_error}, right={right_metadata_error}"
        )

    robots = read_json(args.robots)
    if len(robots) != 1:
        raise RuntimeError(f"Expected one robot, got {sorted(robots)}")
    robot_name = next(iter(robots))
    robot = robots[robot_name]
    joint_names = list(robot["states"][0]["names"])
    if "jaw" not in joint_names:
        raise RuntimeError(f"jaw not found in robot joints: {joint_names}")
    jaw_index = joint_names.index("jaw")
    joint_timestamps = np.asarray(
        robot["control_timestamps"], dtype=np.float64
    )
    controls = np.asarray(robot["control"], dtype=np.float64)
    joint_indices = np.searchsorted(
        joint_timestamps, left_timestamps, side="right"
    ) - 1
    joint_indices = np.clip(joint_indices, 0, len(joint_timestamps) - 1)
    jaw_by_left_frame = controls[joint_indices, jaw_index]
    initial_count = min(60, len(jaw_by_left_frame))
    open_jaw_reference = float(np.median(jaw_by_left_frame[:initial_count]))
    closing = np.flatnonzero(
        jaw_by_left_frame
        < open_jaw_reference - args.jaw_drop_threshold_rad
    )
    if not len(closing):
        raise RuntimeError("Could not find jaw-closing onset")
    closing_frame = int(closing[0])
    closing_timestamp = float(left_timestamps[closing_frame])
    safe_end_timestamp = closing_timestamp - args.precontact_margin_seconds
    safe_end_frame = int(
        np.searchsorted(left_timestamps, safe_end_timestamp, side="right") - 1
    )
    if safe_end_frame <= 0:
        raise RuntimeError("Pre-contact interval is empty after safety margin")

    tissue_packed = np.load(args.tissue_masks, mmap_mode="r")
    if tissue_packed.shape[0] != len(left_paths):
        raise RuntimeError(
            f"Tissue mask count {tissue_packed.shape[0]} != left images {len(left_paths)}"
        )
    with np.load(args.part_masks) as part_data:
        tool_packed = np.asarray(part_data["masks_packbits"])
        tool_shape = tuple(int(value) for value in part_data["mask_shape"])
        part_names = [str(value) for value in part_data["part_names"]]
        part_confidence = np.asarray(part_data["confidence"], dtype=np.float32)
    if tool_packed.shape[0] != len(left_paths):
        raise RuntimeError(
            f"Tool mask count {tool_packed.shape[0]} != left images {len(left_paths)}"
        )

    full_resolution = tuple(int(value) for value in paired_left_metadata["resolution"])
    if full_resolution != (
        tissue_packed.shape[2] * 8,
        tissue_packed.shape[1],
    ):
        raise RuntimeError(
            "Packed tissue shape does not match left resolution: "
            f"packed={tissue_packed.shape}, resolution={full_resolution}"
        )
    if tool_shape != (
        full_resolution[1] // 2,
        full_resolution[0] // 2,
    ):
        raise RuntimeError(
            f"Part-mask shape {tool_shape} is not half of {full_resolution}"
        )

    right_matches = nearest_indices(right_timestamps, left_timestamps)
    half_dilation = max(1, int(round(args.tool_dilation_fullres_px / 2)))
    kernel_size = 2 * half_dilation + 1
    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    reference_left = load_half_image(left_paths[0])
    reference_gray = cv2.cvtColor(reference_left, cv2.COLOR_BGR2GRAY)
    reference_tissue = unpack_tissue_mask(
        tissue_packed, 0, full_resolution[0]
    )
    reference_tool = unpack_tool_mask(tool_packed, 0, tool_shape)
    reference_tool_dilated = cv2.dilate(
        reference_tool.astype(np.uint8), dilation_kernel
    ).astype(bool)

    sampled_frames = list(range(0, safe_end_frame + 1, args.sample_stride))
    if sampled_frames[-1] != safe_end_frame:
        sampled_frames.append(safe_end_frame)
    records: list[dict[str, Any]] = []
    visibility_by_frame: dict[int, np.ndarray] = {}
    tissue_by_frame: dict[int, np.ndarray] = {}
    tool_by_frame: dict[int, np.ndarray] = {}
    print(
        f"[phase-a] scoring {len(sampled_frames)} frames in conservative "
        f"pre-contact interval 0..{safe_end_frame}",
        flush=True,
    )
    for sample_index, frame in enumerate(sampled_frames):
        right_frame = int(right_matches[frame])
        metrics, visible, tissue, tool_dilated = analyze_frame(
            frame,
            right_frame,
            left_paths[frame],
            right_paths[right_frame],
            tissue_packed,
            tool_packed,
            tool_shape,
            dilation_kernel,
            reference_gray,
            reference_tissue,
            reference_tool_dilated,
        )
        passed, score, reasons = quality_gate_and_score(metrics)
        metrics.update(
            {
                "left_timestamp": float(left_timestamps[frame]),
                "right_timestamp": float(right_timestamps[right_frame]),
                "right_minus_left_ms": float(
                    (right_timestamps[right_frame] - left_timestamps[frame])
                    * 1000.0
                ),
                "jaw_command_rad": float(jaw_by_left_frame[frame]),
                "part_mask_confidence": [
                    float(value) for value in part_confidence[frame]
                ],
                "quality_gate_passed": bool(passed),
                "quality_gate_failures": reasons,
                "quality_score": score,
            }
        )
        records.append(metrics)
        visibility_by_frame[frame] = visible
        tissue_by_frame[frame] = tissue
        tool_by_frame[frame] = tool_dilated
        if sample_index % 40 == 0 or sample_index == len(sampled_frames) - 1:
            print(
                f"[phase-a] scored {sample_index + 1}/{len(sampled_frames)}",
                flush=True,
            )

    candidates = select_temporal_candidates(
        records, args.candidate_count, first_frame=0
    )
    keyframes, coverage_domain = greedy_set_cover(
        candidates,
        visibility_by_frame,
        args.keyframe_min,
        args.keyframe_max,
        args.coverage_target,
        args.minimum_coverage_gain,
    )
    candidate_frames = [record["left_frame"] for record in candidates]
    keyframe_order = [record["left_frame"] for record in keyframes]
    keyframe_order_lookup = {
        record["left_frame"]: record for record in keyframes
    }

    output_files = [
        manifest_path,
        args.output_dir / "timestamps_left_native.npy",
        args.output_dir / "timestamps_right_native.npy",
        args.output_dir / "coverage/left_proxy_candidate_visibility.npz",
        args.output_dir / "coverage/left_proxy_observation_count.png",
        args.output_dir / "previews/phase_a_candidates.png",
        args.output_dir / "previews/phase_a_provisional_keyframes.png",
    ]
    if not args.overwrite:
        collisions = [path for path in output_files if path.exists()]
        if collisions:
            raise FileExistsError(
                "Stage-A output files already exist:\n- "
                + "\n- ".join(str(path) for path in collisions)
            )

    print("[phase-a] writing isolated tissue_multiview_v1 outputs", flush=True)
    (args.output_dir / "coverage").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "previews").mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "timestamps_left_native.npy", left_timestamps)
    np.save(args.output_dir / "timestamps_right_native.npy", right_timestamps)

    candidate_visibility = np.stack(
        [visibility_by_frame[frame] for frame in candidate_frames]
    )
    packed_candidate_visibility = np.packbits(candidate_visibility, axis=2)
    np.savez_compressed(
        args.output_dir / "coverage/left_proxy_candidate_visibility.npz",
        left_frame_ids=np.asarray(candidate_frames, dtype=np.int32),
        masks_packbits=packed_candidate_visibility,
        mask_shape=np.asarray(candidate_visibility.shape[1:], dtype=np.int32),
        keyframe_selection_order=np.asarray(keyframe_order, dtype=np.int32),
    )
    observation_count = candidate_visibility.sum(axis=0).astype(np.uint8)
    heat = cv2.applyColorMap(
        np.round(
            observation_count.astype(np.float32)
            / max(float(observation_count.max()), 1.0)
            * 255.0
        ).astype(np.uint8),
        cv2.COLORMAP_VIRIDIS,
    )
    heat[~coverage_domain] = 0
    cv2.imwrite(
        str(args.output_dir / "coverage/left_proxy_observation_count.png"),
        heat,
    )

    candidate_tiles = []
    for candidate in candidates:
        frame = candidate["left_frame"]
        candidate_tiles.append(
            annotated_pair(
                left_paths[frame],
                right_paths[candidate["right_frame"]],
                tissue_by_frame[frame],
                tool_by_frame[frame],
                [
                    f"L{frame:06d} t={candidate['left_timestamp']:.3f}s | "
                    f"R{candidate['right_frame']:06d} "
                    f"dt={candidate['right_minus_left_ms']:+.2f}ms",
                    f"quality={candidate['quality_score']:.3f} "
                    f"corr0={candidate['frame0_gray_correlation']:.4f}",
                ],
                width_each=320,
            )
        )
    cv2.imwrite(
        str(args.output_dir / "previews/phase_a_candidates.png"),
        contact_sheet(candidate_tiles, columns=5),
    )

    keyframe_tiles = []
    for frame in keyframe_order:
        selection = keyframe_order_lookup[frame]
        candidate = next(
            record for record in candidates if record["left_frame"] == frame
        )
        keyframe_tiles.append(
            annotated_pair(
                left_paths[frame],
                right_paths[candidate["right_frame"]],
                tissue_by_frame[frame],
                tool_by_frame[frame],
                [
                    f"set-cover #{selection['selection_order']} | "
                    f"L{frame:06d} R{candidate['right_frame']:06d}",
                    f"gain={selection['coverage_gain_fraction'] * 100:.4f}% "
                    f"cumulative={selection['cumulative_left_proxy_coverage'] * 100:.4f}%",
                ],
                width_each=400,
            )
        )
    cv2.imwrite(
        str(args.output_dir / "previews/phase_a_provisional_keyframes.png"),
        contact_sheet(keyframe_tiles, columns=3),
    )

    candidate_manifest = []
    for candidate in candidates:
        frame = candidate["left_frame"]
        right_frame = candidate["right_frame"]
        candidate_manifest.append(
            {
                **candidate,
                "left_image": {
                    "path": str(left_paths[frame].resolve()),
                    "sha256": sha256(left_paths[frame]),
                },
                "right_image": {
                    "path": str(right_paths[right_frame].resolve()),
                    "sha256": sha256(right_paths[right_frame]),
                },
            }
        )

    input_hashes = {
        "left_metadata": {
            "path": str(args.left_metadata.resolve()),
            "sha256": sha256(args.left_metadata),
        },
        "right_metadata": {
            "path": str(args.right_metadata.resolve()),
            "sha256": sha256(args.right_metadata),
        },
        "robots": {
            "path": str(args.robots.resolve()),
            "sha256": sha256(args.robots),
        },
        "tissue_masks": {
            "path": str(args.tissue_masks.resolve()),
            "sha256": sha256(args.tissue_masks),
        },
        "part_masks": {
            "path": str(args.part_masks.resolve()),
            "sha256": sha256(args.part_masks),
        },
        "bag_identity": {
            "path": str(args.bag.resolve()),
            "size_bytes": int(args.bag.stat().st_size),
            "mtime_ns": int(args.bag.stat().st_mtime_ns),
            "first_ros_timestamp_ns": int(first_ros_timestamp_ns),
            "left_timestamps_float64_sha256": array_sha256(left_timestamps),
            "right_timestamps_float64_sha256": array_sha256(right_timestamps),
        },
    }
    final_coverage = keyframes[-1]["cumulative_left_proxy_coverage"]
    gates = {
        "frozen_hashes_match": bool(
            all(result["matches"] for result in frozen_results.values())
        ),
        "ground_plane_exact_z0": bool(
            frozen_results["ground_plane_value"]["matches"]
        ),
        "native_counts_are_left_1441_right_1443": bool(
            len(left_paths) == 1441 and len(right_paths) == 1443
        ),
        "recovered_timestamps_match_paired_metadata": bool(
            left_metadata_error <= 1.0e-12
            and right_metadata_error <= 1.0e-12
        ),
        "candidate_count_in_planned_range": bool(
            20 <= len(candidates) <= 30
        ),
        "all_candidates_pass_quality_gate": bool(
            all(record["quality_gate_passed"] for record in candidates)
        ),
        "keyframe_count_in_planned_range": bool(
            6 <= len(keyframes) <= 12
        ),
        "every_provisional_keyframe_has_measurable_left_gain": bool(
            all(
                record["coverage_gain_fraction"]
                >= args.minimum_coverage_gain
                for record in keyframes
            )
        ),
        "left_proxy_coverage_target_reached": bool(
            final_coverage >= args.coverage_target
        ),
        "active_runtime_not_modified": True,
        "right_view_masks_required_before_final_set_cover": False,
    }
    report = {
        "stage": "v10_multiview_rigid_tissue_stage_a_frame_selection",
        "status": "provisional_left_proxy_complete_right_masks_pending",
        "runtime_switch_performed": False,
        "candidate_asset_created": False,
        "active_runtime": (
            "data/super/grasp5_native/"
            "bodies_v9_dense_0p5mm_rigid_tissue"
        ),
        "frozen_assets": frozen_results,
        "inputs": input_hashes,
        "native_streams": {
            "left_image_count": len(left_paths),
            "right_image_count": len(right_paths),
            "paired_left_metadata_count": len(paired_left_timestamps),
            "paired_right_metadata_count": len(paired_right_timestamps),
            "paired_metadata_max_abs_error_seconds": {
                "left": left_metadata_error,
                "right": right_metadata_error,
            },
            "timestamp_note": (
                "The frozen offline manifests intentionally contain the first "
                "1441 paired frames. Stage A recovered all native bag times, "
                "including right frames 1441 and 1442, without modifying the "
                "frozen manifests."
            ),
        },
        "precontact_interval": {
            "method": (
                "first dataset jaw-command drop from the initial median, minus "
                "a conservative time margin"
            ),
            "robot": robot_name,
            "jaw_joint_index": jaw_index,
            "initial_open_reference_rad": open_jaw_reference,
            "jaw_drop_threshold_rad": args.jaw_drop_threshold_rad,
            "first_closing_left_frame": closing_frame,
            "first_closing_timestamp": closing_timestamp,
            "margin_seconds": args.precontact_margin_seconds,
            "safe_last_left_frame": safe_end_frame,
            "safe_last_timestamp": float(left_timestamps[safe_end_frame]),
        },
        "selection_parameters": {
            "sample_stride": args.sample_stride,
            "sampled_frame_count": len(sampled_frames),
            "candidate_count": args.candidate_count,
            "keyframe_min": args.keyframe_min,
            "keyframe_max": args.keyframe_max,
            "tool_dilation_fullres_px": args.tool_dilation_fullres_px,
            "coverage_target": args.coverage_target,
            "minimum_coverage_gain": args.minimum_coverage_gain,
            "candidate_method": (
                "best quality-passing sampled frame per equal-duration bin, "
                "with frame 0 retained as the manual left-mask anchor"
            ),
            "keyframe_method": (
                "greedy set cover of the fixed manual frame-0 tissue support "
                "minus dilated per-frame image-tool masks; propagated SAM2 "
                "tissue masks are quality audit only, and the result remains "
                "provisional until independent right masks exist"
            ),
            "quality_gates": {
                "frame0_gray_correlation_min": 0.94,
                "left_laplacian_variance_min": 60.0,
                "right_laplacian_variance_min": 60.0,
                "left_saturated_fraction_max": 0.02,
                "right_saturated_fraction_max": 0.02,
            },
        },
        "part_mask_names": part_names,
        "candidate_frames": candidate_manifest,
        "provisional_keyframes_set_cover_order": keyframes,
        "provisional_keyframes_temporal_order": sorted(
            keyframes, key=lambda record: record["left_frame"]
        ),
        "coverage": {
            "analysis_resolution_wh": [
                int(candidate_visibility.shape[2]),
                int(candidate_visibility.shape[1]),
            ],
            "left_proxy_domain_halfres_px": int(
                coverage_domain.sum(dtype=np.int64)
            ),
            "final_left_proxy_coverage_fraction": float(final_coverage),
            "right_proxy_coverage_fraction": None,
            "combined_stereo_coverage_fraction": None,
            "limitation": (
                "Right-view semantic/tool masks do not exist yet. These "
                "keyframes are suitable inputs to stage B, but are not the "
                "final stereo set-cover result."
            ),
        },
        "outputs": {
            "manifest": str(manifest_path.resolve()),
            "left_native_timestamps": str(
                (args.output_dir / "timestamps_left_native.npy").resolve()
            ),
            "right_native_timestamps": str(
                (args.output_dir / "timestamps_right_native.npy").resolve()
            ),
            "candidate_visibility": str(
                (
                    args.output_dir
                    / "coverage/left_proxy_candidate_visibility.npz"
                ).resolve()
            ),
            "coverage_preview": str(
                (
                    args.output_dir
                    / "coverage/left_proxy_observation_count.png"
                ).resolve()
            ),
            "candidate_preview": str(
                (
                    args.output_dir / "previews/phase_a_candidates.png"
                ).resolve()
            ),
            "keyframe_preview": str(
                (
                    args.output_dir
                    / "previews/phase_a_provisional_keyframes.png"
                ).resolve()
            ),
        },
        "gates": gates,
        "passed_for_stage_b": bool(
            all(
                value
                for name, value in gates.items()
                if name != "right_view_masks_required_before_final_set_cover"
            )
        ),
        "next_required_step": (
            "Stage B: independently propagate/verify right-view tissue masks "
            "and build per-view tool masks for these candidates, then rerun "
            "the final stereo greedy set-cover selection."
        ),
    }
    write_json(manifest_path, report)
    print(
        "[phase-a] selected candidates="
        f"{candidate_frames}; provisional keyframes={keyframe_order}",
        flush=True,
    )
    print(
        "[phase-a] left proxy coverage="
        f"{final_coverage * 100.0:.5f}%; "
        f"manifest={manifest_path}",
        flush=True,
    )
    if not report["passed_for_stage_b"]:
        raise SystemExit("Stage-A gate failed; see observations.json")


if __name__ == "__main__":
    main()
