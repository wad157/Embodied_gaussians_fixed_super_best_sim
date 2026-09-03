#!/usr/bin/env python3

"""Depth-first, bounded visual refinement for selected SUPER PSM frames.

The timestamp-matched registered LND state is the pose prior.  FoundationStereo
depth is used in several small camera-z updates, while RAFT-Stereo is an
independent consistency check.  The resulting pose then initializes the
online_dvrk_tracking differentiable CAD renderer and part-aware visual loss.

This script is intentionally a keyframe experiment.  It never writes a runtime
driver and never changes the frozen camera/table coordinate assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
PAPER_REPO = Path("/Media_HDD/jwshan/wad/online_dvrk_tracking")
TRACK_ROOT = REPO / "data/super/psm_tracking"
DEFAULT_DEPTH_ROOT = (
    TRACK_ROOT / "psm_stereo_3d_corrected_v1/depth_observations"
)
DEFAULT_OUTPUT_ROOT = TRACK_ROOT / "psm_depth_then_visual_v1"
DEFAULT_OLD_STATES = (
    TRACK_ROOT
    / "part_pose_correction_expanded_3mm_8deg_12deg_25deg"
    / "tracking_states_part_corrected.npz"
)
DEFAULT_REGISTRATION = (
    TRACK_ROOT
    / "part_pose_correction_expanded_3mm_8deg_12deg_25deg"
    / "paper_to_gui_registration.npz"
)
DEFAULT_FRAMES = "0,141,160,320,480,800,1120,1280,1360"
EXPECTED_COORDINATE_SHA256 = {
    "table_frame": "6dddc2178cdf816f5dada5febdd528f80e42d52e631076e1f5f4a952297adecf",
    "cameras": "e1e7b7e7e21ca8a9409c88a29409d2e9ad6b783a85b277e71cec0c84340ce4ef",
    "active_corrected_driver": "21df09849b08d6cef1ae47c694e7400b6df5228e0805c5c35ecf3d68e2ef648a",
}

sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(PAPER_REPO)]

from propagate_super_psm_part_masks import (  # noqa: E402
    load_paper_meshes,
    paper_prior_sequence,
)
from refine_super_psm_part_keyframes import (  # noqa: E402
    PART_COLORS,
    PART_NAMES,
    render_state_numpy,
    state_metrics,
    target_panel,
    tint_rendered_parts,
)
from super_psm_tracking_common import (  # noqa: E402
    TrackingInputs,
    _paper_component_transforms,
    ctr_to_matrix,
    matrix_to_ctr,
    pose_to_matrix,
)
from track_super_psm_paper_and_hybrid import ctrnet_args  # noqa: E402
from track_super_psm_part_corrected import (  # noqa: E402
    apply_parameters,
    optimize_batch,
    unpack_mask_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Iterative Foundation/RAFT camera-z initialization followed by "
            "bounded online_dvrk visual correction on selected keyframes."
        )
    )
    parser.add_argument("--frames", default=DEFAULT_FRAMES)
    parser.add_argument(
        "--states",
        type=Path,
        default=TRACK_ROOT / "tracking_states_paper_exact.npz",
    )
    parser.add_argument(
        "--old-corrected-states", type=Path, default=DEFAULT_OLD_STATES
    )
    parser.add_argument(
        "--registration", type=Path, default=DEFAULT_REGISTRATION
    )
    parser.add_argument(
        "--part-masks",
        type=Path,
        default=TRACK_ROOT / "part_masks_full_sequence/part_masks_full.npz",
    )
    parser.add_argument("--depth-dir", type=Path, default=DEFAULT_DEPTH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--left-video",
        type=Path,
        default=REPO
        / "data/super/grasp5_offline_demo/videos/stereo_left.mp4",
    )
    parser.add_argument(
        "--right-video",
        type=Path,
        default=REPO
        / "data/super/grasp5_offline_demo/videos/stereo_right.mp4",
    )
    parser.add_argument(
        "--right-metadata",
        type=Path,
        default=REPO
        / "data/super/grasp5_offline_demo/videos/stereo_right.json",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO / "data/super/grasp5_native/calib_rectified.json",
    )
    parser.add_argument("--depth-height", type=int, default=270)
    parser.add_argument("--depth-width", type=int, default=480)
    parser.add_argument("--depth-rounds", type=int, default=4)
    parser.add_argument("--depth-step-mm", type=float, default=0.3)
    parser.add_argument("--max-total-depth-mm", type=float, default=1.2)
    parser.add_argument("--stop-step-mm", type=float, default=0.1)
    parser.add_argument("--stop-improvement-mm", type=float, default=0.05)
    parser.add_argument("--max-body-dice-drop", type=float, default=0.005)
    parser.add_argument("--model-agreement-mm", type=float, default=3.0)
    parser.add_argument("--min-depth-samples", type=int, default=80)
    parser.add_argument("--visual-iterations", type=int, default=120)
    parser.add_argument("--visual-learning-rate", type=float, default=0.05)
    parser.add_argument("--visual-max-xy-mm", type=float, default=3.0)
    parser.add_argument("--visual-max-z-mm", type=float, default=0.4)
    parser.add_argument("--visual-max-rotation-deg", type=float, default=8.0)
    parser.add_argument("--visual-max-wrist-deg", type=float, default=12.0)
    parser.add_argument("--visual-max-jaw-deg", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scaled_intrinsics(K: np.ndarray, scale: float) -> np.ndarray:
    output = np.asarray(K, dtype=np.float64).copy()
    output[0] *= scale
    output[1] *= scale
    output[2, 2] = 1.0
    return output


def rasterize_depth(
    vertices: np.ndarray,
    faces: np.ndarray,
    transform: np.ndarray,
    K: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize a perspective-correct front-surface z-buffer on the CPU."""
    height, width = shape
    camera = vertices @ transform[:3, :3].T + transform[:3, 3]
    projected = np.empty((len(camera), 2), dtype=np.float64)
    projected[:, 0] = K[0, 0] * camera[:, 0] / camera[:, 2] + K[0, 2]
    projected[:, 1] = K[1, 1] * camera[:, 1] / camera[:, 2] + K[1, 2]
    zbuffer = np.full(shape, np.inf, dtype=np.float32)
    for face in faces:
        triangle_z = camera[face, 2]
        if np.any(triangle_z <= 1.0e-5):
            continue
        triangle = projected[face]
        x0 = max(0, int(math.floor(float(np.min(triangle[:, 0])))))
        x1 = min(width - 1, int(math.ceil(float(np.max(triangle[:, 0])))))
        y0 = max(0, int(math.floor(float(np.min(triangle[:, 1])))))
        y1 = min(height - 1, int(math.ceil(float(np.max(triangle[:, 1])))))
        if x1 < x0 or y1 < y0:
            continue
        ax, ay = triangle[0]
        bx, by = triangle[1]
        cx, cy = triangle[2]
        denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(denominator) < 1.0e-10:
            continue
        yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        sample_x = xx.astype(np.float64) + 0.5
        sample_y = yy.astype(np.float64) + 0.5
        w0 = (
            (by - cy) * (sample_x - cx)
            + (cx - bx) * (sample_y - cy)
        ) / denominator
        w1 = (
            (cy - ay) * (sample_x - cx)
            + (ax - cx) * (sample_y - cy)
        ) / denominator
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1.0e-6) & (w1 >= -1.0e-6) & (w2 >= -1.0e-6)
        if not np.any(inside):
            continue
        inverse_z = w0 / triangle_z[0] + w1 / triangle_z[1] + w2 / triangle_z[2]
        depth = np.full_like(inverse_z, np.inf, dtype=np.float64)
        valid = inside & (inverse_z > 0.0)
        depth[valid] = 1.0 / inverse_z[valid]
        region = zbuffer[y0 : y1 + 1, x0 : x1 + 1]
        np.minimum(region, depth.astype(np.float32), out=region)
    zbuffer[~np.isfinite(zbuffer)] = np.nan
    return zbuffer


def render_body_depth(
    ctr: np.ndarray,
    joints: np.ndarray,
    vertices: tuple[np.ndarray, ...],
    faces: tuple[np.ndarray, ...],
    K: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    camera = ctr_to_matrix(ctr)
    components = _paper_component_transforms(joints)
    body_depths = []
    for mesh_id, component_id in ((0, 0), (1, 1)):
        body_depths.append(
            rasterize_depth(
                vertices[mesh_id],
                faces[mesh_id],
                camera @ components[component_id],
                K,
                shape,
            )
        )
    stacked = np.stack(body_depths)
    finite = np.any(np.isfinite(stacked), axis=0)
    output = np.min(np.where(np.isfinite(stacked), stacked, np.inf), axis=0)
    output[~finite] = np.nan
    return output.astype(np.float32)


def dice_score(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=bool)
    target = np.asarray(target, dtype=bool)
    return float(
        (2 * np.count_nonzero(predicted & target) + 1.0e-6)
        / (np.count_nonzero(predicted) + np.count_nonzero(target) + 1.0e-6)
    )


def robust_depth_values(
    observed: np.ndarray, rendered: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    residual_mm = (observed[valid] - rendered[valid]) * 1000.0
    residual_mm = residual_mm[np.isfinite(residual_mm)]
    if not len(residual_mm):
        return residual_mm
    median = float(np.median(residual_mm))
    mad = float(np.median(np.abs(residual_mm - median)))
    radius = max(1.0, 3.5 * 1.4826 * mad)
    return residual_mm[np.abs(residual_mm - median) <= radius]


def depth_metrics(
    *,
    rendered: np.ndarray,
    foundation: np.ndarray,
    raft: np.ndarray,
    valid_observation: np.ndarray,
    target_body: np.ndarray,
) -> dict[str, float | int]:
    valid = valid_observation & np.isfinite(rendered)
    foundation_values = robust_depth_values(foundation, rendered, valid)
    raft_values = robust_depth_values(raft, rendered, valid)

    def summarize(values: np.ndarray, prefix: str) -> dict[str, float | int]:
        if not len(values):
            return {
                f"{prefix}_sample_count": 0,
                f"{prefix}_median_mm": float("nan"),
                f"{prefix}_absolute_median_mm": float("nan"),
                f"{prefix}_mad_mm": float("nan"),
            }
        median = float(np.median(values))
        return {
            f"{prefix}_sample_count": int(len(values)),
            f"{prefix}_median_mm": median,
            f"{prefix}_absolute_median_mm": float(np.median(np.abs(values))),
            f"{prefix}_mad_mm": float(np.median(np.abs(values - median))),
        }

    return {
        **summarize(foundation_values, "foundation"),
        **summarize(raft_values, "raft"),
        "body_dice": dice_score(np.isfinite(rendered), target_body),
        "rendered_pixel_count": int(np.count_nonzero(np.isfinite(rendered))),
        "valid_overlap_count": int(np.count_nonzero(valid)),
    }


def prepare_depth_observation(
    *,
    frame: int,
    depth_dir: Path,
    body_mask_half: np.ndarray,
    shape: tuple[int, int],
    agreement_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    prefix = depth_dir / f"{frame:06d}"
    foundation_path = Path(f"{prefix}-depth.npy")
    raft_path = Path(f"{prefix}-raft_depth.npy")
    confidence_path = Path(f"{prefix}-confidence.npz")
    for path in (foundation_path, raft_path, confidence_path):
        if not path.exists():
            raise FileNotFoundError(path)
    foundation_full = np.load(foundation_path)
    raft_full = np.load(raft_path)
    confidence = np.load(confidence_path)
    height, width = shape
    foundation = cv2.resize(
        foundation_full, (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(np.float32)
    raft = cv2.resize(
        raft_full, (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(np.float32)
    target = cv2.resize(
        body_mask_half.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    eroded = cv2.erode(
        target.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1
    ).astype(bool)
    high = cv2.resize(
        confidence["high_confidence"].astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    model_difference = cv2.resize(
        confidence["absolute_model_difference_mm"],
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    foundation_lr = cv2.resize(
        confidence["foundation_lr_valid"].astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    raft_valid = cv2.resize(
        confidence["raft_valid"].astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    strict = (
        eroded
        & high
        & np.isfinite(model_difference)
        & (model_difference <= agreement_mm)
        & np.isfinite(foundation)
        & np.isfinite(raft)
    )
    relaxed = (
        eroded
        & foundation_lr
        & raft_valid
        & np.isfinite(model_difference)
        & (model_difference <= agreement_mm)
        & np.isfinite(foundation)
        & np.isfinite(raft)
    )
    if np.count_nonzero(strict) >= 80:
        valid = strict
        tier = "foundation_raft_high_confidence"
    else:
        valid = relaxed
        tier = "foundation_raft_relaxed_agreement"
    return foundation, raft, valid, target, tier


def shift_ctr_z(ctr: np.ndarray, delta_m: float) -> np.ndarray:
    matrix = ctr_to_matrix(ctr)
    matrix[2, 3] += delta_m
    return matrix_to_ctr(matrix)


def iterative_depth_refinement(
    *,
    args: argparse.Namespace,
    frame: int,
    prior_ctr: np.ndarray,
    joints: np.ndarray,
    vertices: tuple[np.ndarray, ...],
    faces: tuple[np.ndarray, ...],
    K: np.ndarray,
    foundation: np.ndarray,
    raft: np.ndarray,
    valid_observation: np.ndarray,
    target_body: np.ndarray,
    confidence_tier: str,
) -> tuple[np.ndarray, list[dict[str, object]], np.ndarray]:
    shape = (args.depth_height, args.depth_width)
    current_ctr = np.asarray(prior_ctr, dtype=np.float32).copy()
    current_depth = render_body_depth(
        current_ctr, joints, vertices, faces, K, shape
    )
    current = depth_metrics(
        rendered=current_depth,
        foundation=foundation,
        raft=raft,
        valid_observation=valid_observation,
        target_body=target_body,
    )
    history: list[dict[str, object]] = [
        {
            "round": 0,
            "accepted": True,
            "cumulative_camera_z_mm": 0.0,
            "confidence_tier": confidence_tier,
            **current,
        }
    ]
    initial_dice = float(current["body_dice"])
    cumulative_mm = 0.0
    for round_index in range(1, args.depth_rounds + 1):
        residual_mm = float(current["foundation_median_mm"])
        if (
            int(current["foundation_sample_count"]) < args.min_depth_samples
            or not np.isfinite(residual_mm)
        ):
            history.append(
                {
                    "round": round_index,
                    "accepted": False,
                    "reason": "insufficient_reliable_body_depth",
                    "cumulative_camera_z_mm": cumulative_mm,
                }
            )
            break
        available_low = -args.max_total_depth_mm - cumulative_mm
        available_high = args.max_total_depth_mm - cumulative_mm
        step_mm = float(
            np.clip(
                residual_mm,
                max(-args.depth_step_mm, available_low),
                min(args.depth_step_mm, available_high),
            )
        )
        if abs(step_mm) < args.stop_step_mm:
            history.append(
                {
                    "round": round_index,
                    "accepted": False,
                    "reason": "step_below_stop_threshold",
                    "proposed_step_mm": step_mm,
                    "cumulative_camera_z_mm": cumulative_mm,
                }
            )
            break
        candidate_ctr = shift_ctr_z(current_ctr, step_mm / 1000.0)
        candidate_depth = render_body_depth(
            candidate_ctr, joints, vertices, faces, K, shape
        )
        candidate = depth_metrics(
            rendered=candidate_depth,
            foundation=foundation,
            raft=raft,
            valid_observation=valid_observation,
            target_body=target_body,
        )
        old_error = abs(float(current["foundation_median_mm"]))
        new_error = abs(float(candidate["foundation_median_mm"]))
        improvement = old_error - new_error
        raft_ok = abs(float(candidate["raft_median_mm"])) <= (
            abs(float(current["raft_median_mm"])) + 0.10
        )
        silhouette_ok = float(candidate["body_dice"]) >= (
            initial_dice - args.max_body_dice_drop
        )
        sample_ok = (
            int(candidate["foundation_sample_count"]) >= args.min_depth_samples
        )
        accepted = bool(
            np.isfinite(new_error)
            and improvement > 0.0
            and raft_ok
            and silhouette_ok
            and sample_ok
        )
        item: dict[str, object] = {
            "round": round_index,
            "accepted": accepted,
            "proposed_step_mm": step_mm,
            "depth_improvement_mm": improvement,
            "raft_non_regression": raft_ok,
            "silhouette_non_regression": silhouette_ok,
            "sample_gate": sample_ok,
            "cumulative_camera_z_mm": (
                cumulative_mm + step_mm if accepted else cumulative_mm
            ),
            **candidate,
        }
        history.append(item)
        if not accepted:
            break
        current_ctr = candidate_ctr
        current_depth = candidate_depth
        current = candidate
        cumulative_mm += step_mm
        if improvement < args.stop_improvement_mm:
            break
        if abs(cumulative_mm) >= args.max_total_depth_mm - 1.0e-9:
            break
    print(
        f"depth frame={frame:04d} z={cumulative_mm:+.3f}mm "
        f"foundation={history[0]['foundation_median_mm']:+.3f}->"
        f"{current['foundation_median_mm']:+.3f}mm "
        f"samples={current['foundation_sample_count']}"
    )
    return current_ctr, history, current_depth


def visual_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        iterations=args.visual_iterations,
        learning_rate=args.visual_learning_rate,
        render_height=135,
        render_width=240,
        max_translation_mm=max(args.visual_max_xy_mm, args.visual_max_z_mm),
        max_translation_xy_mm_per_axis=args.visual_max_xy_mm,
        max_translation_z_mm_per_axis=args.visual_max_z_mm,
        max_rotation_deg=args.visual_max_rotation_deg,
        max_wrist_deg=args.visual_max_wrist_deg,
        max_jaw_deg=args.visual_max_jaw_deg,
    )


def visual_gate(
    candidate: dict[str, object],
    old: dict[str, object],
    candidate_depth: dict[str, float | int],
    old_depth: dict[str, float | int],
    observation_confidence: np.ndarray,
    prompt_source: int,
) -> tuple[bool, list[str]]:
    reasons = []
    candidate_parts = candidate["parts"]
    old_parts = old["parts"]
    body_dice_ok = (
        float(candidate_parts["body"]["dice"])
        >= float(old_parts["body"]["dice"]) - 0.005
    )
    body_contour_ok = (
        float(candidate_parts["body"]["contour_distance_px_p95"])
        <= float(old_parts["body"]["contour_distance_px_p95"]) + 1.5
    )
    candidate_jaw_dice = np.mean(
        [float(candidate_parts[name]["dice"]) for name in PART_NAMES[1:]]
    )
    old_jaw_dice = np.mean(
        [float(old_parts[name]["dice"]) for name in PART_NAMES[1:]]
    )
    jaw_dice_ok = candidate_jaw_dice >= old_jaw_dice - 0.02
    candidate_tip = np.mean(
        [float(candidate["tips"][name]["error_px"]) for name in PART_NAMES[1:]]
    )
    old_tip = np.mean(
        [float(old["tips"][name]["error_px"]) for name in PART_NAMES[1:]]
    )
    tip_ok = candidate_tip <= old_tip + 1.5
    candidate_depth_error = abs(float(candidate_depth["foundation_median_mm"]))
    old_depth_error = abs(float(old_depth["foundation_median_mm"]))
    depth_ok = candidate_depth_error <= old_depth_error + 0.15
    for ok, reason in (
        (body_dice_ok, "body_dice"),
        (body_contour_ok, "body_contour"),
        (jaw_dice_ok, "jaw_dice"),
        (tip_ok, "ordered_jaw_tips"),
        (depth_ok, "foundation_depth"),
    ):
        if not ok:
            reasons.append(reason)
    # Source 1 is a periodically injected CAD identity anchor, not an image
    # observation.  It can stabilize temporal tracking but cannot validate a
    # new depth/pose candidate.  Likewise, one missing jaw observation is not
    # enough evidence to certify physically plausible two-jaw image alignment.
    if prompt_source == 1:
        reasons.append("synthetic_cad_observation")
    if float(np.min(observation_confidence[1:])) < 0.15:
        reasons.append("unreliable_jaw_observation")
    return not reasons, reasons


def read_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read video frame {frame_index}")
    return cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)


def right_ctr(ctr: np.ndarray, baseline_m: float) -> np.ndarray:
    matrix = ctr_to_matrix(ctr)
    matrix[0, 3] -= baseline_m
    return matrix_to_ctr(matrix)


def rendered_panel(
    image: np.ndarray, rendered: dict[str, np.ndarray], title: str
) -> np.ndarray:
    output = image.copy()
    for name in PART_NAMES:
        predicted = rendered[name] >= 0.25
        tint = np.zeros_like(output)
        tint[predicted] = PART_COLORS[name]
        output = cv2.addWeighted(output, 1.0, tint, 0.42, 0.0)
        contours, _ = cv2.findContours(
            predicted.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(output, contours, -1, PART_COLORS[name], 2)
    cv2.rectangle(output, (0, 0), (output.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        output,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def residual_panel(
    image: np.ndarray,
    rendered_depth: np.ndarray,
    observed_depth: np.ndarray,
    valid_observation: np.ndarray,
    title: str,
) -> np.ndarray:
    valid = valid_observation & np.isfinite(rendered_depth)
    residual_mm = np.zeros_like(rendered_depth, dtype=np.float32)
    residual_mm[valid] = (
        observed_depth[valid] - rendered_depth[valid]
    ) * 1000.0
    normalized = np.clip((residual_mm + 3.0) / 6.0 * 255.0, 0, 255).astype(
        np.uint8
    )
    heat = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    base = cv2.resize(
        image, (rendered_depth.shape[1], rendered_depth.shape[0]), cv2.INTER_AREA
    )
    mixed = base.copy()
    mixed[valid] = cv2.addWeighted(base, 0.25, heat, 0.75, 0.0)[valid]
    mixed = cv2.resize(mixed, (960, 540), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(mixed, (0, 0), (mixed.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        mixed,
        title + " (blue=near, red=far)",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return mixed


def finite_json(value: object) -> object:
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    frame_indices = [int(value) for value in args.frames.split(",") if value]
    if not frame_indices or len(frame_indices) != len(set(frame_indices)):
        raise ValueError("Frames must be a non-empty unique comma-separated list")
    if args.depth_rounds <= 0 or args.depth_step_mm <= 0.0:
        raise ValueError("Depth rounds and step must be positive")
    required = (
        args.states,
        args.old_corrected_states,
        args.registration,
        args.part_masks,
        args.left_video,
        args.right_video,
        args.right_metadata,
        args.calibration,
        REPO / "data/super/table_frame.json",
        REPO / "data/super/grasp5_offline_demo/cameras.json",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    coordinate_guard_paths = {
        "table_frame": REPO / "data/super/table_frame.json",
        "cameras": REPO / "data/super/grasp5_offline_demo/cameras.json",
        "active_corrected_driver": TRACK_ROOT
        / "psm_part_corrected_pose_driver.npz",
    }
    coordinate_hashes = {
        name: sha256(path) for name, path in coordinate_guard_paths.items()
    }
    mismatches = {
        name: {"expected": EXPECTED_COORDINATE_SHA256[name], "actual": value}
        for name, value in coordinate_hashes.items()
        if value != EXPECTED_COORDINATE_SHA256[name]
    }
    if mismatches:
        raise RuntimeError(f"Frozen coordinate/driver hash mismatch: {mismatches}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = TrackingInputs.load()
    exact = np.load(args.states)
    old_states = np.load(args.old_corrected_states)
    mask_asset = np.load(args.part_masks)
    mask_shape = tuple(int(value) for value in mask_asset["mask_shape"])
    frame_count = min(
        len(inputs.video_timestamps),
        len(exact["pure_ctr"]),
        len(old_states["corrected_ctr"]),
        len(mask_asset["confidence"]),
    )
    if min(frame_indices) < 0 or max(frame_indices) >= frame_count:
        raise IndexError(f"Requested frames must be in [0, {frame_count - 1}]")
    prior_ctr_full, prior_joints_full = paper_prior_sequence(
        inputs, exact, frame_count
    )
    packed_masks = mask_asset["masks_packbits"]
    selected_masks = np.concatenate(
        [
            unpack_mask_batch(packed_masks, mask_shape, frame, frame + 1)
            for frame in frame_indices
        ],
        axis=0,
    )
    selected_prior_ctr = prior_ctr_full[frame_indices]
    selected_prior_joints = prior_joints_full[frame_indices]
    vertices, faces = load_paper_meshes()
    depth_scale = args.depth_width / float(mask_shape[1])
    K_depth = scaled_intrinsics(inputs.K_half, depth_scale)

    depth_ctr = np.empty_like(selected_prior_ctr)
    depth_histories: dict[str, object] = {}
    observations = []
    depth_buffers = []
    for local_index, frame in enumerate(frame_indices):
        observation = prepare_depth_observation(
            frame=frame,
            depth_dir=args.depth_dir,
            body_mask_half=selected_masks[local_index, 0],
            shape=(args.depth_height, args.depth_width),
            agreement_mm=args.model_agreement_mm,
        )
        foundation, raft, valid, target_body, tier = observation
        refined, history, rendered_depth = iterative_depth_refinement(
            args=args,
            frame=frame,
            prior_ctr=selected_prior_ctr[local_index],
            joints=selected_prior_joints[local_index],
            vertices=vertices,
            faces=faces,
            K=K_depth,
            foundation=foundation,
            raft=raft,
            valid_observation=valid,
            target_body=target_body,
            confidence_tier=tier,
        )
        depth_ctr[local_index] = refined
        depth_histories[str(frame)] = history
        observations.append(observation)
        depth_buffers.append(rendered_depth)

    from diffcali.models.CtRNet import CtRNet

    model = CtRNet(ctrnet_args(inputs))
    mesh_dir = PAPER_REPO / "urdfs/dVRK/meshes"
    mesh_files = [
        mesh_dir / "low_res_shaft_multi_cylinder.ply",
        mesh_dir / "low_res_logo_low_res_1.ply",
        mesh_dir / "low_res_jawright_lowres.ply",
        mesh_dir / "low_res_jawleft_lowres.ply",
    ]
    renderer = model.setup_robot_renderer(
        [str(path) for path in mesh_files], downscale_factor=2
    )
    renderer.set_mesh_visibility([True, True, True, True])
    K_torch = torch.as_tensor(
        inputs.K_half, device=model.device, dtype=torch.float32
    )
    selected = np.asarray(frame_indices, dtype=np.int64)
    parameters, initial_losses, optimized_losses = optimize_batch(
        args=visual_args(args),
        model=model,
        renderer=renderer,
        K=K_torch,
        masks=selected_masks,
        confidence=mask_asset["confidence"][selected],
        observed_tips=mask_asset["jaw_tips_xy"][selected],
        observed_areas=mask_asset["areas_px"][selected],
        prompt_source=mask_asset["prompt_source"][selected],
        prior_ctr=depth_ctr,
        prior_joints_np=selected_prior_joints,
    )
    visual_ctr, visual_joints = apply_parameters(
        depth_ctr, selected_prior_joints, parameters
    )

    old_ctr = old_states["corrected_ctr"][selected]
    old_joints = old_states["corrected_joints"][selected]
    accepted_ctr = visual_ctr.copy()
    accepted_joints = visual_joints.copy()
    metrics_report: dict[str, object] = {}
    rendered_variants: list[dict[str, dict[str, np.ndarray]]] = []
    depth_variant_buffers: list[dict[str, np.ndarray]] = []
    for local_index, frame in enumerate(frame_indices):
        masks = {
            name: selected_masks[local_index, part_index]
            for part_index, name in enumerate(PART_NAMES)
        }
        tips = {
            "jaw_left": mask_asset["jaw_tips_xy"][frame, 0],
            "jaw_right": mask_asset["jaw_tips_xy"][frame, 1],
        }
        states = {
            "kinematic": (selected_prior_ctr[local_index], selected_prior_joints[local_index]),
            "after_depth": (depth_ctr[local_index], selected_prior_joints[local_index]),
            "after_visual_raw": (visual_ctr[local_index], visual_joints[local_index]),
            "old_corrected": (old_ctr[local_index], old_joints[local_index]),
        }
        rendered = {
            name: render_state_numpy(
                model=model,
                renderer=renderer,
                K=K_torch,
                ctr=state[0],
                joints=state[1],
                render_size=mask_shape,
            )
            for name, state in states.items()
        }
        state_scores = {
            name: state_metrics(value, masks, tips)
            for name, value in rendered.items()
        }
        foundation, raft, valid, target_body, _ = observations[local_index]
        variant_depths = {
            name: render_body_depth(
                state[0],
                state[1],
                vertices,
                faces,
                K_depth,
                (args.depth_height, args.depth_width),
            )
            for name, state in states.items()
        }
        depth_scores = {
            name: depth_metrics(
                rendered=value,
                foundation=foundation,
                raft=raft,
                valid_observation=valid,
                target_body=target_body,
            )
            for name, value in variant_depths.items()
        }
        accepted, reasons = visual_gate(
            state_scores["after_visual_raw"],
            state_scores["old_corrected"],
            depth_scores["after_visual_raw"],
            depth_scores["old_corrected"],
            mask_asset["confidence"][frame],
            int(mask_asset["prompt_source"][frame]),
        )
        if not accepted:
            accepted_ctr[local_index] = old_ctr[local_index]
            accepted_joints[local_index] = old_joints[local_index]
            rendered["accepted"] = rendered["old_corrected"]
            variant_depths["accepted"] = variant_depths["old_corrected"]
            state_scores["accepted"] = state_scores["old_corrected"]
            depth_scores["accepted"] = depth_scores["old_corrected"]
        else:
            rendered["accepted"] = rendered["after_visual_raw"]
            variant_depths["accepted"] = variant_depths["after_visual_raw"]
            state_scores["accepted"] = state_scores["after_visual_raw"]
            depth_scores["accepted"] = depth_scores["after_visual_raw"]
        metrics_report[str(frame)] = {
            "visual_candidate_accepted": accepted,
            "rejection_reasons": reasons,
            "observation_confidence": mask_asset["confidence"][frame].tolist(),
            "prompt_source": int(mask_asset["prompt_source"][frame]),
            "online_visual_loss": {
                "initial": float(initial_losses[local_index]),
                "optimized": float(optimized_losses[local_index]),
            },
            "visual": state_scores,
            "depth": depth_scores,
        }
        rendered_variants.append(rendered)
        depth_variant_buffers.append(variant_depths)
        print(
            f"visual frame={frame:04d} loss={initial_losses[local_index]:.3f}->"
            f"{optimized_losses[local_index]:.3f} accepted={accepted} "
            f"reasons={','.join(reasons) if reasons else 'none'}"
        )

    full_accepted_ctr = prior_ctr_full.copy()
    full_accepted_joints = prior_joints_full.copy()
    full_accepted_ctr[selected] = accepted_ctr
    full_accepted_joints[selected] = accepted_joints
    registration = np.load(args.registration)
    conversion_options = {
        "registration_ctr": registration["registration_ctr"],
        "registration_joints": registration["registration_joints"],
        "camera_alignment": registration["T_rectified_camera_alignment"],
    }
    visual_poses = inputs.paper_states_to_visual_poses(
        full_accepted_ctr, full_accepted_joints, **conversion_options
    )
    wrist_angles = inputs.gui_wrist_angles_from_paper_residual(
        prior_joints_full, full_accepted_joints
    )
    jaw_angles = inputs.gui_jaw_angles_from_paper_residual(
        prior_joints_full, full_accepted_joints
    )
    visual_poses = inputs.enforce_urdf_distal_kinematics(
        visual_poses, wrist_angles, jaw_angles
    )
    jaw_names = (
        "PSM1_tool_wrist_sca_ee_link_1",
        "PSM1_tool_wrist_sca_ee_link_2",
    )
    jaw_indices = tuple(inputs.link_names.index(name) for name in jaw_names)
    pivot_gaps = []
    axis_errors = []
    for frame in frame_indices:
        left = pose_to_matrix(visual_poses[frame, jaw_indices[0]])
        right = pose_to_matrix(visual_poses[frame, jaw_indices[1]])
        pivot_gaps.append(float(np.linalg.norm(left[:3, 3] - right[:3, 3])))
        cosine = float(np.clip(np.dot(left[:3, 2], right[:3, 2]), -1.0, 1.0))
        axis_errors.append(math.degrees(math.acos(cosine)))
    jaw_gate = {
        "shared_pivot_gap_max_m": max(pivot_gaps),
        "hinge_axis_error_max_deg": max(axis_errors),
        "passed": bool(max(pivot_gaps) < 1.0e-7 and max(axis_errors) < 1.0e-3),
        "method": "literal URDF distal-chain rebuild with symmetric +/-jaw/2",
    }

    calibration = read_json(args.calibration)
    baseline_m = float(calibration["baseline_m"])
    right_timestamps = np.asarray(
        read_json(args.right_metadata)["timestamps"], dtype=np.float64
    )
    left_cap = cv2.VideoCapture(str(args.left_video))
    right_cap = cv2.VideoCapture(str(args.right_video))
    comparison_rows = []
    for local_index, frame in enumerate(frame_indices):
        left_image = read_frame(left_cap, frame)
        right_index = int(
            np.argmin(np.abs(right_timestamps - inputs.video_timestamps[frame]))
        )
        right_image = read_frame(right_cap, right_index)
        masks = {
            name: selected_masks[local_index, part_index]
            for part_index, name in enumerate(PART_NAMES)
        }
        tips = {
            "jaw_left": mask_asset["jaw_tips_xy"][frame, 0],
            "jaw_right": mask_asset["jaw_tips_xy"][frame, 1],
        }
        rendered = rendered_variants[local_index]
        status = metrics_report[str(frame)]["visual_candidate_accepted"]
        top = [target_panel(left_image, masks, tips, frame)]
        labels = {
            "kinematic": "1 LND kinematic prior",
            "after_depth": "2 iterative stereo depth",
            "after_visual_raw": "3 online_dvrk visual raw",
            "old_corrected": "A/B old corrected",
            "accepted": f"accepted output ({'new' if status else 'old fallback'})",
        }
        for name in (
            "kinematic",
            "after_depth",
            "after_visual_raw",
            "old_corrected",
            "accepted",
        ):
            top.append(
                tint_rendered_parts(left_image, rendered[name], masks, tips, labels[name])
            )
        foundation, _, valid, _, _ = observations[local_index]
        depth_buffers_for_frame = depth_variant_buffers[local_index]
        bottom = [
            residual_panel(
                left_image,
                depth_buffers_for_frame[name],
                foundation,
                valid,
                f"{name} Foundation-model z residual",
            )
            for name in (
                "kinematic",
                "after_depth",
                "after_visual_raw",
                "old_corrected",
            )
        ]
        right_states = {
            "old corrected right": (old_ctr[local_index], old_joints[local_index]),
            "accepted right": (accepted_ctr[local_index], accepted_joints[local_index]),
        }
        for title, state in right_states.items():
            right_render = render_state_numpy(
                model=model,
                renderer=renderer,
                K=K_torch,
                ctr=right_ctr(state[0], baseline_m),
                joints=state[1],
                render_size=mask_shape,
            )
            bottom.append(rendered_panel(right_image, right_render, title))
        comparison = np.vstack((np.hstack(top), np.hstack(bottom)))
        path = args.output_dir / f"frame{frame:06d}_comparison.png"
        cv2.imwrite(str(path), comparison)
        comparison_rows.append(cv2.resize(comparison, (1920, 360)))
    left_cap.release()
    right_cap.release()
    if comparison_rows:
        cv2.imwrite(
            str(args.output_dir / "contact_sheet.png"), np.vstack(comparison_rows)
        )

    np.savez_compressed(
        args.output_dir / "keyframe_states.npz",
        frame_indices=selected,
        kinematic_ctr=selected_prior_ctr,
        kinematic_joints=selected_prior_joints,
        depth_ctr=depth_ctr,
        visual_raw_ctr=visual_ctr,
        visual_raw_joints=visual_joints,
        accepted_ctr=accepted_ctr,
        accepted_joints=accepted_joints,
        old_corrected_ctr=old_ctr,
        old_corrected_joints=old_joints,
        visual_parameters=parameters,
        visual_initial_losses=initial_losses,
        visual_optimized_losses=optimized_losses,
    )
    np.savez_compressed(
        args.output_dir / "keyframe_visual_poses_urdf.npz",
        frame_indices=selected,
        timestamps=inputs.video_timestamps[selected],
        link_names=np.asarray(inputs.link_names),
        poses_rect_camera_xyz_xyzw=visual_poses[selected],
        gui_wrist_angles=wrist_angles[selected],
        gui_jaw_angles=jaw_angles[selected],
    )
    coordinate_paths = {
        "table_frame": REPO / "data/super/table_frame.json",
        "cameras": REPO / "data/super/grasp5_offline_demo/cameras.json",
        "old_corrected_states": args.old_corrected_states,
        "active_corrected_driver": TRACK_ROOT / "psm_part_corrected_pose_driver.npz",
    }
    report = {
        "method": (
            "timestamp-matched registered LND -> iterative bounded camera-z "
            "FoundationStereo refinement with RAFT check -> online_dvrk "
            "differentiable part visual correction -> per-frame A/B gate -> "
            "literal URDF distal-chain rebuild"
        ),
        "runtime_driver_written": False,
        "frame_indices": frame_indices,
        "depth_limits": {
            "rounds": args.depth_rounds,
            "step_mm": args.depth_step_mm,
            "total_mm": args.max_total_depth_mm,
            "camera_coordinate_axis": "+z of rectified left camera",
        },
        "visual_limits": {
            "iterations": args.visual_iterations,
            "max_rotation_deg": args.visual_max_rotation_deg,
            "max_xy_translation_mm_per_axis": args.visual_max_xy_mm,
            "max_z_translation_mm_per_axis": args.visual_max_z_mm,
            "max_wrist_deg": args.visual_max_wrist_deg,
            "max_common_jaw_deg": args.visual_max_jaw_deg,
        },
        "depth_histories": depth_histories,
        "frame_metrics": metrics_report,
        "accepted_new_frame_count": int(
            sum(
                bool(metrics_report[str(frame)]["visual_candidate_accepted"])
                for frame in frame_indices
            )
        ),
        "jaw_kinematics_gate": jaw_gate,
        "coordinate_sha256": {
            name: sha256(path) for name, path in coordinate_paths.items() if path.exists()
        },
        "coordinate_hash_gate_passed": True,
        "temporal_status": (
            "keyframe-only experiment; no interpolation and no runtime driver "
            "were generated, so full-sequence velocity/acceleration validation "
            "remains intentionally pending"
        ),
        "outputs": {
            "states": str(args.output_dir / "keyframe_states.npz"),
            "urdf_visual_poses": str(
                args.output_dir / "keyframe_visual_poses_urdf.npz"
            ),
            "contact_sheet": str(args.output_dir / "contact_sheet.png"),
        },
    }
    report = finite_json(report)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
