#!/usr/bin/env python3

"""Part-aware bounded pose correction for every SUPER grasp5 video frame.

The paper-exact result is kept as a read-only baseline.  Each frame starts from
the timestamp-matched registered LND state and optimizes only a small camera,
wrist, and common-jaw residual against the propagated three-part observations.
Large jaw masks are allowed (they still provide useful local tip evidence), but
their silhouette term is automatically reduced so that a leaked shaft cannot
move the whole instrument.  Residuals are smoothed, while the four manually
validated keyframe corrections are retained as exact temporal anchors.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.spatial.transform import Rotation


REPO = Path(__file__).resolve().parents[1]
PAPER_REPO = Path("/Media_HDD/jwshan/wad/online_dvrk_tracking")
TRACK_ROOT = REPO / "data/super/psm_tracking"
sys.path[:0] = [
    str(REPO),
    str(REPO / "scripts"),
    str(PAPER_REPO),
]

from propagate_super_psm_part_masks import paper_prior_sequence  # noqa: E402
from rebuild_super_psm_registered_drivers import (  # noqa: E402
    estimate_camera_alignment,
    strict_current_poses_at_first_video_frame,
)
from refine_super_psm_part_keyframes import (  # noqa: E402
    PART_NAMES,
    render_parts,
    render_state_numpy,
    state_metrics,
    target_panel,
    tint_rendered_parts,
)
from super_psm_tracking_common import (  # noqa: E402
    TrackingInputs,
    ctr_to_matrix,
    matrix_to_ctr,
    save_runtime_driver,
)
from track_super_psm_paper_and_hybrid import ctrnet_args  # noqa: E402


MANUAL_ANCHORS = (141, 142, 160, 320)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-sequence bounded LND/three-part PSM correction."
    )
    parser.add_argument(
        "--states",
        type=Path,
        default=TRACK_ROOT / "tracking_states_paper_exact.npz",
    )
    parser.add_argument(
        "--part-masks",
        type=Path,
        default=TRACK_ROOT / "part_masks_full_sequence/part_masks_full.npz",
    )
    parser.add_argument(
        "--keyframe-states",
        type=Path,
        default=TRACK_ROOT
        / "part_pose_refinement/corrected_keyframe_states.npz",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=REPO
        / "data/super/grasp5_offline_demo/videos/stereo_left.mp4",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TRACK_ROOT / "part_pose_correction_full",
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--render-height", type=int, default=135)
    parser.add_argument("--render-width", type=int, default=240)
    parser.add_argument("--max-translation-mm", type=float, default=3.0)
    parser.add_argument("--max-rotation-deg", type=float, default=8.0)
    parser.add_argument("--max-wrist-deg", type=float, default=12.0)
    parser.add_argument("--max-jaw-deg", type=float, default=25.0)
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore and replace an existing batch optimization checkpoint.",
    )
    parser.add_argument(
        "--registered-runtime-driver",
        type=Path,
        default=TRACK_ROOT / "psm_registered_lnd_pose_driver.npz",
        help="5458-state registered-LND GUI driver written after a full run.",
    )
    parser.add_argument(
        "--runtime-driver",
        type=Path,
        default=TRACK_ROOT / "psm_part_corrected_pose_driver.npz",
        help="5458-state corrected GUI driver written after a full run.",
    )
    parser.add_argument(
        "--diagnostic-frames",
        default="0,141,142,160,320,480,640,800,960,1120,1280,1440",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def unpack_mask_batch(
    packed: np.ndarray,
    mask_shape: tuple[int, int],
    start: int,
    stop: int,
) -> np.ndarray:
    pixel_count = int(np.prod(mask_shape))
    unpacked = np.unpackbits(packed[start:stop], axis=-1)[..., :pixel_count]
    return unpacked.reshape(stop - start, len(PART_NAMES), *mask_shape).astype(
        bool
    )


def prepare_target_batch(
    masks: np.ndarray,
    render_size: tuple[int, int],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, part_count = masks.shape[:2]
    height, width = render_size
    targets = np.empty(
        (batch_size, part_count, height, width), dtype=np.float32
    )
    outside = np.empty_like(targets)
    for batch_index in range(batch_size):
        for part_index in range(part_count):
            resized = cv2.resize(
                masks[batch_index, part_index].astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32)
            targets[batch_index, part_index] = resized
            outside[batch_index, part_index] = cv2.distanceTransform(
                (~(resized >= 0.5)).astype(np.uint8), cv2.DIST_L2, 3
            )
    return (
        torch.as_tensor(targets, device=device),
        torch.as_tensor(outside, device=device),
    )


def euler_xyz_matrices(angles: torch.Tensor) -> torch.Tensor:
    """Return Rz @ Ry @ Rx for a batch of xyz Euler residuals."""
    rx, ry, rz = angles.unbind(dim=1)
    one = torch.ones_like(rx)
    zero = torch.zeros_like(rx)
    Rx = torch.stack(
        (
            torch.stack((one, zero, zero), dim=1),
            torch.stack((zero, torch.cos(rx), -torch.sin(rx)), dim=1),
            torch.stack((zero, torch.sin(rx), torch.cos(rx)), dim=1),
        ),
        dim=1,
    )
    Ry = torch.stack(
        (
            torch.stack((torch.cos(ry), zero, torch.sin(ry)), dim=1),
            torch.stack((zero, one, zero), dim=1),
            torch.stack((-torch.sin(ry), zero, torch.cos(ry)), dim=1),
        ),
        dim=1,
    )
    Rz = torch.stack(
        (
            torch.stack((torch.cos(rz), -torch.sin(rz), zero), dim=1),
            torch.stack((torch.sin(rz), torch.cos(rz), zero), dim=1),
            torch.stack((zero, zero, one), dim=1),
        ),
        dim=1,
    )
    return Rz @ Ry @ Rx


def physical_parameters(raw: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    max_xy_per_axis = getattr(args, "max_translation_xy_mm_per_axis", None)
    max_z_per_axis = getattr(args, "max_translation_z_mm_per_axis", None)
    if max_xy_per_axis is None or max_z_per_axis is None:
        translation_limits = torch.full(
            (3,),
            args.max_translation_mm / 1000.0 / math.sqrt(3.0),
            device=raw.device,
            dtype=raw.dtype,
        )
    else:
        translation_limits = torch.as_tensor(
            [max_xy_per_axis, max_xy_per_axis, max_z_per_axis],
            device=raw.device,
            dtype=raw.dtype,
        ) / 1000.0
    return torch.cat(
        (
            math.radians(args.max_rotation_deg)
            / math.sqrt(3.0)
            * torch.tanh(raw[:, :3]),
            translation_limits * torch.tanh(raw[:, 3:6]),
            math.radians(args.max_wrist_deg) * torch.tanh(raw[:, 6:8]),
            math.radians(args.max_jaw_deg)
            * torch.tanh(raw[:, 8:9]),
        ),
        dim=1,
    )


def bounded_state_batch(
    raw: torch.Tensor,
    prior_rotation: torch.Tensor,
    prior_translation: torch.Tensor,
    prior_joints: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    parameters = physical_parameters(raw, args)
    camera_rotation = euler_xyz_matrices(parameters[:, :3]) @ prior_rotation
    camera_translation = prior_translation + parameters[:, 3:6]
    wrist = prior_joints[:, :2] + parameters[:, 6:8]
    wrist = torch.stack(
        (
            wrist[:, 0].clamp(-1.5707, 1.5707),
            wrist[:, 1].clamp(-1.3963, 1.3963),
        ),
        dim=1,
    )
    jaws = (prior_joints[:, 2:] + parameters[:, 8:9]).clamp(
        0.0, math.pi / 2.0
    )
    joints = torch.cat((wrist, jaws), dim=1)
    return camera_rotation, camera_translation, joints, parameters


def part_loss_per_sample(
    predicted: torch.Tensor,
    target: torch.Tensor,
    outside_distance: torch.Tensor,
) -> torch.Tensor:
    predicted = predicted.clamp(0.0, 1.0)
    spatial_dims = (-2, -1)
    intersection = torch.sum(predicted * target, dim=spatial_dims)
    denominator = torch.sum(predicted, dim=spatial_dims) + torch.sum(
        target, dim=spatial_dims
    )
    dice = 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    outside = torch.sum(
        predicted * outside_distance, dim=spatial_dims
    ) / (torch.sum(predicted, dim=spatial_dims) + 1e-6)
    return dice + 0.025 * outside


def project_tips_batch(
    camera_rotation: torch.Tensor,
    camera_translation: torch.Tensor,
    component_rotations: torch.Tensor,
    component_translations: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    local = torch.as_tensor(
        ((0.0, 0.0004, 0.0096), (0.0, -0.0004, 0.0096)),
        device=camera_rotation.device,
        dtype=torch.float32,
    )
    mesh_ids = (2, 3)
    output = []
    for tip_index, mesh_id in enumerate(mesh_ids):
        paper_point = (
            component_rotations[:, mesh_id]
            @ local[tip_index].view(1, 3, 1)
        ).squeeze(-1) + component_translations[:, mesh_id]
        camera_point = (
            camera_rotation @ paper_point.unsqueeze(-1)
        ).squeeze(-1) + camera_translation
        output.append(((camera_point / camera_point[:, 2:3]) @ K.T)[:, :2])
    return torch.stack(output, dim=1)


def optimize_batch(
    *,
    args: argparse.Namespace,
    model: object,
    renderer: object,
    K: torch.Tensor,
    masks: np.ndarray,
    confidence: np.ndarray,
    observed_tips: np.ndarray,
    observed_areas: np.ndarray,
    prompt_source: np.ndarray,
    prior_ctr: np.ndarray,
    prior_joints_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = model.device
    matrices = np.stack([ctr_to_matrix(value) for value in prior_ctr])
    prior_rotation = torch.as_tensor(
        matrices[:, :3, :3], device=device, dtype=torch.float32
    )
    prior_translation = torch.as_tensor(
        matrices[:, :3, 3], device=device, dtype=torch.float32
    )
    prior_joints = torch.as_tensor(
        prior_joints_np, device=device, dtype=torch.float32
    )
    targets, outside = prepare_target_batch(
        masks, (args.render_height, args.render_width), device
    )
    confidence_tensor = torch.as_tensor(
        confidence, device=device, dtype=torch.float32
    )
    area_tensor = torch.as_tensor(
        observed_areas, device=device, dtype=torch.float32
    )
    source_tensor = torch.as_tensor(prompt_source, device=device)
    tip_targets = torch.as_tensor(
        observed_tips, device=device, dtype=torch.float32
    )

    # Body always remains useful.  A jaw mask larger than roughly the complete
    # visible wrist can still locate a tip, but is not a valid jaw silhouette.
    jaw_area_gate = torch.minimum(
        torch.ones_like(area_tensor[:, 1:]),
        (6000.0 / area_tensor[:, 1:].clamp_min(1.0)) ** 2,
    )
    mask_weights = confidence_tensor.clone()
    mask_weights[:, 1:] *= jaw_area_gate
    # CAD masks injected every 25 frames are temporal identity anchors rather
    # than image observations.  Their low weight keeps the residual near zero.
    mask_weights = torch.where(
        (source_tensor == 1).view(-1, 1),
        torch.minimum(mask_weights, torch.tensor(0.20, device=device)),
        mask_weights,
    )
    tip_weights = confidence_tensor[:, 1:].clone()
    tip_weights = torch.where(
        (source_tensor == 1).view(-1, 1),
        torch.zeros_like(tip_weights),
        tip_weights,
    )

    raw = torch.nn.Parameter(
        torch.zeros((len(masks), 9), device=device, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam([raw], lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.iterations, 1), eta_min=0.002
    )
    best_loss = torch.full((len(masks),), float("inf"), device=device)
    best_raw = raw.detach().clone()
    initial_loss: torch.Tensor | None = None

    for _ in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        rotation, translation, joints, parameters = bounded_state_batch(
            raw, prior_rotation, prior_translation, prior_joints, args
        )
        rendered, component_rotations, component_translations = render_parts(
            model,
            renderer,
            rotation,
            translation,
            joints,
            (args.render_height, args.render_width),
        )
        rendered_stack = torch.stack(
            [
                rendered[name].unsqueeze(0)
                if rendered[name].ndim == 2
                else rendered[name]
                for name in PART_NAMES
            ],
            dim=1,
        )
        silhouette_losses = part_loss_per_sample(
            rendered_stack, targets, outside
        )
        part_total = (
            mask_weights[:, 0] * silhouette_losses[:, 0]
            + 0.8 * mask_weights[:, 1] * silhouette_losses[:, 1]
            + 0.8 * mask_weights[:, 2] * silhouette_losses[:, 2]
        )
        projected_tips = project_tips_batch(
            rotation,
            translation,
            component_rotations,
            component_translations,
            K,
        )
        per_tip = F.smooth_l1_loss(
            projected_tips / 10.0,
            tip_targets / 10.0,
            beta=0.5,
            reduction="none",
        ).sum(dim=2)
        tip_total = torch.sum(tip_weights * per_tip, dim=1)
        max_xy_per_axis = getattr(
            args, "max_translation_xy_mm_per_axis", args.max_translation_mm
        )
        max_z_per_axis = getattr(
            args, "max_translation_z_mm_per_axis", args.max_translation_mm
        )
        maxima = torch.as_tensor(
            [
                math.radians(args.max_rotation_deg),
                math.radians(args.max_rotation_deg),
                math.radians(args.max_rotation_deg),
                max_xy_per_axis / 1000.0,
                max_xy_per_axis / 1000.0,
                max_z_per_axis / 1000.0,
                math.radians(args.max_wrist_deg),
                math.radians(args.max_wrist_deg),
                math.radians(args.max_jaw_deg),
            ],
            device=device,
        )
        prior_loss = torch.sum((parameters / maxima) ** 2, dim=1)
        sample_loss = part_total + 1.5 * tip_total + 0.08 * prior_loss
        if initial_loss is None:
            initial_loss = sample_loss.detach().clone()
        if not torch.isfinite(sample_loss).all():
            raise FloatingPointError("Non-finite full-sequence refinement loss")
        improved = sample_loss.detach() < best_loss
        best_loss = torch.where(improved, sample_loss.detach(), best_loss)
        best_raw[improved] = raw.detach()[improved]
        sample_loss.mean().backward()
        torch.nn.utils.clip_grad_norm_([raw], 5.0)
        optimizer.step()
        scheduler.step()

    assert initial_loss is not None
    return (
        physical_parameters(best_raw, args).detach().cpu().numpy(),
        initial_loss.cpu().numpy(),
        best_loss.cpu().numpy(),
    )


def state_to_parameters(
    corrected_ctr: np.ndarray,
    corrected_joints: np.ndarray,
    prior_ctr: np.ndarray,
    prior_joints: np.ndarray,
) -> np.ndarray:
    corrected = ctr_to_matrix(corrected_ctr)
    prior = ctr_to_matrix(prior_ctr)
    residual_rotation = corrected[:3, :3] @ prior[:3, :3].T
    parameters = np.empty(9, dtype=np.float64)
    parameters[:3] = Rotation.from_matrix(residual_rotation).as_euler("xyz")
    parameters[3:6] = corrected[:3, 3] - prior[:3, 3]
    parameters[6:8] = corrected_joints[:2] - prior_joints[:2]
    parameters[8] = np.mean(corrected_joints[2:] - prior_joints[2:])
    return parameters


def smooth_with_keyframe_anchors(
    parameters: np.ndarray,
    args: argparse.Namespace,
    keyframe_path: Path,
    prior_ctr: np.ndarray,
    prior_joints: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    if len(parameters) < 3:
        return parameters.copy(), []
    filtered = median_filter(parameters, size=(5, 1), mode="nearest")
    smoothed = gaussian_filter1d(
        filtered, sigma=args.smooth_sigma, axis=0, mode="nearest"
    )
    used_anchors: list[int] = []
    if not keyframe_path.exists():
        return smoothed, used_anchors
    keyframes = np.load(keyframe_path)
    anchor_values: dict[int, np.ndarray] = {}
    for local_index, frame_index_value in enumerate(keyframes["frame_indices"]):
        frame_index = int(frame_index_value)
        if frame_index >= len(smoothed):
            continue
        anchor = state_to_parameters(
            keyframes["corrected_ctr"][local_index],
            keyframes["corrected_joints"][local_index],
            prior_ctr[frame_index],
            prior_joints[frame_index],
        )
        delta = anchor - smoothed[frame_index]
        indices = np.arange(len(smoothed), dtype=np.float64)
        kernel = np.exp(-0.5 * ((indices - frame_index) / 2.0) ** 2)
        smoothed += kernel[:, None] * delta[None, :]
        anchor_values[frame_index] = anchor
        used_anchors.append(frame_index)
    maxima = np.asarray(
        [
            *(
                [math.radians(args.max_rotation_deg) / math.sqrt(3.0)]
                * 3
            ),
            *([args.max_translation_mm / 1000.0 / math.sqrt(3.0)] * 3),
            *([math.radians(args.max_wrist_deg)] * 2),
            math.radians(args.max_jaw_deg),
        ]
    )
    smoothed = np.clip(smoothed, -maxima, maxima)
    # Neighbouring anchors (141/142) have overlapping kernels.  Reapply their
    # exact validated values after the smooth neighbourhood corrections.
    for frame_index, anchor in anchor_values.items():
        smoothed[frame_index] = np.clip(anchor, -maxima, maxima)
    return smoothed, used_anchors


def apply_parameters(
    prior_ctr: np.ndarray,
    prior_joints: np.ndarray,
    parameters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    corrected_ctr = np.empty_like(prior_ctr, dtype=np.float32)
    corrected_joints = prior_joints.astype(np.float64).copy()
    residual_rotations = Rotation.from_euler("xyz", parameters[:, :3]).as_matrix()
    for frame_index in range(len(prior_ctr)):
        prior = ctr_to_matrix(prior_ctr[frame_index])
        corrected = prior.copy()
        corrected[:3, :3] = residual_rotations[frame_index] @ prior[:3, :3]
        corrected[:3, 3] = prior[:3, 3] + parameters[frame_index, 3:6]
        corrected_ctr[frame_index] = matrix_to_ctr(corrected)
    corrected_joints[:, :2] += parameters[:, 6:8]
    corrected_joints[:, 2:] += parameters[:, 8:9]
    corrected_joints[:, 0] = np.clip(corrected_joints[:, 0], -1.5707, 1.5707)
    corrected_joints[:, 1] = np.clip(corrected_joints[:, 1], -1.3963, 1.3963)
    corrected_joints[:, 2:] = np.clip(
        corrected_joints[:, 2:], 0.0, math.pi / 2.0
    )
    return corrected_ctr, corrected_joints.astype(np.float32)


def read_video_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read video frame {frame_index}")
    return cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)


def render_diagnostics(
    *,
    args: argparse.Namespace,
    model: object,
    renderer: object,
    K: torch.Tensor,
    packed_masks: np.ndarray,
    mask_shape: tuple[int, int],
    jaw_tips: np.ndarray,
    confidence: np.ndarray,
    exact: np.lib.npyio.NpzFile,
    prior_ctr: np.ndarray,
    prior_joints: np.ndarray,
    corrected_ctr: np.ndarray,
    corrected_joints: np.ndarray,
) -> dict[str, object]:
    requested = [
        int(value) for value in args.diagnostic_frames.split(",") if value
    ]
    frame_indices = [value for value in requested if value < len(corrected_ctr)]
    cap = cv2.VideoCapture(str(args.video))
    rows = []
    metrics: dict[str, object] = {}
    for frame_index in frame_indices:
        masks_array = unpack_mask_batch(
            packed_masks, mask_shape, frame_index, frame_index + 1
        )[0]
        masks = {
            name: masks_array[part_index]
            for part_index, name in enumerate(PART_NAMES)
        }
        tips = {
            "jaw_left": jaw_tips[frame_index, 0],
            "jaw_right": jaw_tips[frame_index, 1],
        }
        variants = {
            "paper_exact": render_state_numpy(
                model=model,
                renderer=renderer,
                K=K,
                ctr=exact["pure_ctr"][frame_index],
                joints=exact["pure_joints"][frame_index],
                render_size=mask_shape,
            ),
            "lnd_prior": render_state_numpy(
                model=model,
                renderer=renderer,
                K=K,
                ctr=prior_ctr[frame_index],
                joints=prior_joints[frame_index],
                render_size=mask_shape,
            ),
            "part_corrected": render_state_numpy(
                model=model,
                renderer=renderer,
                K=K,
                ctr=corrected_ctr[frame_index],
                joints=corrected_joints[frame_index],
                render_size=mask_shape,
            ),
        }
        metrics[str(frame_index)] = {
            "confidence": confidence[frame_index].tolist(),
            **{
                name: state_metrics(rendered, masks, tips)
                for name, rendered in variants.items()
            },
        }
        image = read_video_frame(cap, frame_index)
        panels = [target_panel(image, masks, tips, frame_index)]
        labels = {
            "paper_exact": "paper exact baseline",
            "lnd_prior": "registered LND prior",
            "part_corrected": "full-sequence part correction",
        }
        for name in ("paper_exact", "lnd_prior", "part_corrected"):
            panels.append(
                tint_rendered_parts(
                    image, variants[name], masks, tips, labels[name]
                )
            )
        comparison = np.hstack(panels)
        cv2.imwrite(
            str(args.output_dir / f"frame{frame_index:06d}_comparison.png"),
            comparison,
        )
        rows.append(cv2.resize(comparison, (1920, 270)))
    cap.release()
    if rows:
        cv2.imwrite(str(args.output_dir / "contact_sheet.png"), np.vstack(rows))
    return metrics


def percentiles(values: np.ndarray) -> list[float]:
    return np.percentile(values, [0, 5, 50, 95, 100]).tolist()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.batch_size <= 0 or args.iterations <= 0:
        raise ValueError("batch size and iterations must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in (args.states, args.part_masks, args.video):
        if not path.exists():
            raise FileNotFoundError(path)

    inputs = TrackingInputs.load()
    exact = np.load(args.states)
    mask_asset = np.load(args.part_masks)
    mask_shape = tuple(int(value) for value in mask_asset["mask_shape"])
    frame_count = min(
        len(exact["pure_ctr"]),
        len(mask_asset["confidence"]),
        len(inputs.video_timestamps),
    )
    if args.max_frames is not None:
        frame_count = min(frame_count, args.max_frames)
    prior_ctr, prior_joints = paper_prior_sequence(inputs, exact, frame_count)

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
    K = torch.as_tensor(inputs.K_half, device=model.device, dtype=torch.float32)

    raw_parameters = np.empty((frame_count, 9), dtype=np.float32)
    initial_losses = np.empty(frame_count, dtype=np.float32)
    optimized_losses = np.empty(frame_count, dtype=np.float32)
    partial_path = args.output_dir / "optimization_partial.npz"
    processed_count = 0
    if partial_path.exists() and not args.restart:
        partial = np.load(partial_path)
        saved_count = int(partial["processed_count"])
        if (
            int(partial["frame_count"]) == frame_count
            and 0 <= saved_count <= frame_count
        ):
            raw_parameters[:saved_count] = partial["raw_parameters"]
            initial_losses[:saved_count] = partial["initial_losses"]
            optimized_losses[:saved_count] = partial["optimized_losses"]
            processed_count = saved_count
            print(f"resuming completed frames 0:{processed_count - 1}")
    started = time.perf_counter()
    packed_masks = mask_asset["masks_packbits"]
    for start in range(processed_count, frame_count, args.batch_size):
        stop = min(frame_count, start + args.batch_size)
        masks = unpack_mask_batch(packed_masks, mask_shape, start, stop)
        parameters, initial, optimized = optimize_batch(
            args=args,
            model=model,
            renderer=renderer,
            K=K,
            masks=masks,
            confidence=mask_asset["confidence"][start:stop],
            observed_tips=mask_asset["jaw_tips_xy"][start:stop],
            observed_areas=mask_asset["areas_px"][start:stop],
            prompt_source=mask_asset["prompt_source"][start:stop],
            prior_ctr=prior_ctr[start:stop],
            prior_joints_np=prior_joints[start:stop],
        )
        raw_parameters[start:stop] = parameters
        initial_losses[start:stop] = initial
        optimized_losses[start:stop] = optimized
        np.savez(
            partial_path,
            frame_count=np.asarray(frame_count),
            processed_count=np.asarray(stop),
            raw_parameters=raw_parameters[:stop],
            initial_losses=initial_losses[:stop],
            optimized_losses=optimized_losses[:stop],
        )
        elapsed = time.perf_counter() - started
        print(
            f"part_pose frames={start:04d}:{stop - 1:04d}/{frame_count - 1:04d} "
            f"loss={np.median(initial):.3f}->{np.median(optimized):.3f} "
            f"elapsed={elapsed:.1f}s"
        )

    smoothed_parameters, used_anchors = smooth_with_keyframe_anchors(
        raw_parameters,
        args,
        args.keyframe_states,
        prior_ctr,
        prior_joints,
    )
    corrected_ctr, corrected_joints = apply_parameters(
        prior_ctr, prior_joints, smoothed_parameters
    )
    raw_ctr, raw_joints = apply_parameters(
        prior_ctr, prior_joints, raw_parameters
    )

    states_path = args.output_dir / "tracking_states_part_corrected.npz"
    np.savez_compressed(
        states_path,
        video_timestamps=inputs.video_timestamps[:frame_count],
        corrected_ctr=corrected_ctr,
        corrected_joints=corrected_joints,
        raw_corrected_ctr=raw_ctr,
        raw_corrected_joints=raw_joints,
        prior_ctr=prior_ctr,
        prior_joints=prior_joints,
        raw_correction_parameters=raw_parameters,
        smoothed_correction_parameters=smoothed_parameters,
        initial_losses=initial_losses,
        optimized_losses=optimized_losses,
        confidence=mask_asset["confidence"][:frame_count],
        prompt_source=mask_asset["prompt_source"][:frame_count],
    )
    # The paper renderer and the GUI do not use the same instrument geometry.
    # Register the current tip Gaussians to the paper CAD once, using the
    # registered prior (never the corrected first frame), and reuse this exact
    # immutable transform for both drivers.  This prevents the first corrected
    # residual from algebraically cancelling during visual-pose conversion.
    strict_first = strict_current_poses_at_first_video_frame(inputs)
    camera_alignment, geometry_alignment_report = estimate_camera_alignment(
        prior_ctr, prior_joints, strict_first
    )
    registration_path = args.output_dir / "paper_to_gui_registration.npz"
    np.savez_compressed(
        registration_path,
        T_rectified_camera_alignment=camera_alignment,
        registration_ctr=prior_ctr[0],
        registration_joints=prior_joints[0],
    )
    conversion_options = {
        "registration_ctr": prior_ctr[0],
        "registration_joints": prior_joints[0],
        "camera_alignment": camera_alignment,
    }
    registered_video_poses = inputs.paper_states_to_visual_poses(
        prior_ctr, prior_joints, **conversion_options
    )
    video_poses = inputs.paper_states_to_visual_poses(
        corrected_ctr, corrected_joints, **conversion_options
    )
    registered_gui_wrist_angles = inputs.gui_wrist_angles_from_paper_residual(
        prior_joints, prior_joints
    )
    corrected_gui_wrist_angles = inputs.gui_wrist_angles_from_paper_residual(
        prior_joints, corrected_joints
    )
    registered_gui_jaw_angles = inputs.gui_jaw_angles_from_paper_residual(
        prior_joints, prior_joints
    )
    corrected_gui_jaw_angles = inputs.gui_jaw_angles_from_paper_residual(
        prior_joints, corrected_joints
    )
    registered_video_poses = inputs.enforce_urdf_distal_kinematics(
        registered_video_poses,
        registered_gui_wrist_angles,
        registered_gui_jaw_angles,
    )
    video_poses = inputs.enforce_urdf_distal_kinematics(
        video_poses,
        corrected_gui_wrist_angles,
        corrected_gui_jaw_angles,
    )
    registered_video_pose_path = (
        args.output_dir / "visual_poses_registered_lnd.npz"
    )
    np.savez_compressed(
        registered_video_pose_path,
        timestamps=inputs.video_timestamps[:frame_count],
        link_names=np.asarray(inputs.link_names),
        poses_rect_camera_xyz_xyzw=registered_video_poses,
    )
    video_pose_path = args.output_dir / "visual_poses_part_corrected.npz"
    np.savez_compressed(
        video_pose_path,
        timestamps=inputs.video_timestamps[:frame_count],
        link_names=np.asarray(inputs.link_names),
        poses_rect_camera_xyz_xyzw=video_poses,
    )
    registered_runtime_path = args.registered_runtime_driver
    runtime_path = args.runtime_driver
    if frame_count == len(inputs.video_timestamps):
        save_runtime_driver(
            registered_runtime_path, inputs, registered_video_poses
        )
        save_runtime_driver(runtime_path, inputs, video_poses)

    diagnostic_metrics = render_diagnostics(
        args=args,
        model=model,
        renderer=renderer,
        K=K,
        packed_masks=packed_masks,
        mask_shape=mask_shape,
        jaw_tips=mask_asset["jaw_tips_xy"][:frame_count],
        confidence=mask_asset["confidence"][:frame_count],
        exact=exact,
        prior_ctr=prior_ctr,
        prior_joints=prior_joints,
        corrected_ctr=corrected_ctr,
        corrected_joints=corrected_joints,
    )
    partial_path.unlink(missing_ok=True)
    translation_mm = np.linalg.norm(smoothed_parameters[:, 3:6], axis=1) * 1000
    rotation_deg = np.degrees(
        Rotation.from_euler("xyz", smoothed_parameters[:, :3]).magnitude()
    )
    report = {
        "frame_count": frame_count,
        "full_run": frame_count == len(inputs.video_timestamps),
        "batch_size": args.batch_size,
        "iterations": args.iterations,
        "render_size": [args.render_height, args.render_width],
        "manual_keyframe_anchors": used_anchors,
        "initial_loss_min_p05_p50_p95_max": percentiles(initial_losses),
        "optimized_loss_min_p05_p50_p95_max": percentiles(optimized_losses),
        "loss_improved_frame_count": int(
            np.count_nonzero(optimized_losses < initial_losses)
        ),
        "translation_correction_mm_min_p05_p50_p95_max": percentiles(
            translation_mm
        ),
        "rotation_correction_deg_min_p05_p50_p95_max": percentiles(
            rotation_deg
        ),
        "states": str(states_path),
        "paper_to_gui_registration": str(registration_path),
        "geometry_alignment": geometry_alignment_report,
        "jaw_kinematics": {
            "method": (
                "complete distal chain rebuilt from the tracked tool_wrist_link "
                "anchor with literal URDF wrist-pitch, wrist-yaw, and symmetric "
                "+/-jaw/2 rigid rotations; CAD registration is not inserted "
                "between URDF joints"
            ),
            "corrected_gui_jaw_deg_min_p05_p50_p95_max": percentiles(
                np.degrees(corrected_gui_jaw_angles)
            ),
            "image_jaw_correction_deg_min_p05_p50_p95_max": percentiles(
                np.degrees(
                    corrected_gui_jaw_angles - registered_gui_jaw_angles
                )
            ),
        },
        "wrist_kinematics": {
            "method": (
                "URDF wrist-pitch q4 stays on the encoder; URDF wrist-yaw q5 "
                "uses encoder baseline plus only the corrected-minus-prior "
                "paper q5 residual; the static paper joint-zero offset is not "
                "applied twice"
            ),
            "corrected_gui_wrist_deg_min_p05_p50_p95_max": np.percentile(
                np.degrees(corrected_gui_wrist_angles),
                [0, 5, 50, 95, 100],
                axis=0,
            ).T.tolist(),
        },
        "registered_video_poses": str(registered_video_pose_path),
        "video_poses": str(video_pose_path),
        "registered_runtime_driver": str(registered_runtime_path)
        if frame_count == len(inputs.video_timestamps)
        else None,
        "runtime_driver": str(runtime_path)
        if frame_count == len(inputs.video_timestamps)
        else None,
        "runtime_state_count": len(inputs.joint_timestamps)
        if frame_count == len(inputs.video_timestamps)
        else None,
        "elapsed_seconds": time.perf_counter() - started,
        "diagnostic_metrics": diagnostic_metrics,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
