#!/usr/bin/env python3

"""Refine selected SUPER PSM poses against separate body and jaw masks.

This is deliberately separate from the paper-exact tracker.  The saved exact
result remains an immutable reproduction baseline.  A timestamp-matched strict
LND state is registered into the paper CAD frame on frame 0, then only a small
bounded camera-pose/wrist/jaw residual is optimized on each diagnosed keyframe.

The image-left and image-right jaw identities stay fixed for the whole run.
They map to paper mesh 2 and paper mesh 3 respectively; unlike the upstream
permutation-invariant tip loss, identities are never swapped per frame.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation


REPO = Path(__file__).resolve().parents[1]
PAPER_REPO = Path("/Media_HDD/jwshan/wad/online_dvrk_tracking")
TRACK_ROOT = REPO / "data/super/psm_tracking"
sys.path[:0] = [
    str(REPO),
    str(REPO / "scripts"),
    str(PAPER_REPO),
    str(PAPER_REPO / "SurgicalSAM2"),
]

from super_psm_tracking_common import (  # noqa: E402
    TrackingInputs,
    ctr_to_matrix,
    matrix_to_ctr,
)
from track_super_psm_paper_and_hybrid import ctrnet_args  # noqa: E402


PART_NAMES = ("body", "jaw_left", "jaw_right")
PART_COLORS = {
    "body": (0, 180, 255),
    "jaw_left": (255, 0, 255),
    "jaw_right": (0, 0, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Part-aware bounded LND refinement on diagnosed PSM frames."
    )
    parser.add_argument("--frames", default="141,142,160,320")
    parser.add_argument(
        "--states",
        type=Path,
        default=TRACK_ROOT / "tracking_states_paper_exact.npz",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=TRACK_ROOT / "part_segmentation_experiment",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TRACK_ROOT / "part_pose_refinement",
    )
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--render-height", type=int, default=270)
    parser.add_argument("--render-width", type=int, default=480)
    parser.add_argument("--max-translation-mm", type=float, default=2.0)
    parser.add_argument("--max-rotation-deg", type=float, default=5.0)
    parser.add_argument("--max-wrist-deg", type=float, default=8.0)
    parser.add_argument("--max-jaw-deg", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def paper_lnd_prior(
    inputs: TrackingInputs,
    exact_ctr: np.ndarray,
    exact_joints: np.ndarray,
    frame_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    strict_ctr, strict_joints = inputs.strict_paper_states()
    paper_to_current_frame4 = (
        np.linalg.inv(ctr_to_matrix(exact_ctr[0]))
        @ ctr_to_matrix(strict_ctr[0])
    )
    prior_ctr = matrix_to_ctr(
        ctr_to_matrix(strict_ctr[frame_index])
        @ np.linalg.inv(paper_to_current_frame4)
    )
    prior_joints = (
        exact_joints[0] + strict_joints[frame_index] - strict_joints[0]
    ).astype(np.float32)
    prior_joints[:2] = np.clip(
        prior_joints[:2],
        np.asarray([-1.5707, -1.3963], dtype=np.float32),
        np.asarray([1.5707, 1.3963], dtype=np.float32),
    )
    prior_joints[2:] = np.clip(prior_joints[2:], 0.0, np.pi / 2.0)
    return prior_ctr, prior_joints


def load_part_observation(
    masks_dir: Path, frame_index: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    path = masks_dir / f"frame{frame_index:06d}_masks.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    asset = np.load(path)
    masks = {name: asset[name].astype(bool) for name in PART_NAMES}
    tips = {
        "jaw_left": asset["jaw_left_tip_xy"].astype(np.float32),
        "jaw_right": asset["jaw_right_tip_xy"].astype(np.float32),
    }
    return masks, tips


def assemble_part_geometry(
    renderer: object,
    rotations: torch.Tensor,
    translations: torch.Tensor,
    mesh_ids: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform selected paper meshes for a differentiable batched render."""
    vertices = []
    faces = []
    vertex_offset = 0
    for mesh_id in mesh_ids:
        local_vertices = renderer.preload_verts[mesh_id].to(
            device=rotations.device, dtype=torch.float32
        )
        local_faces = renderer.preload_faces[mesh_id].to(
            device=rotations.device, dtype=torch.int32
        )
        transformed = (
            torch.matmul(
                local_vertices.unsqueeze(0),
                rotations[:, mesh_id].transpose(1, 2),
            )
            + translations[:, mesh_id].unsqueeze(1)
        )
        vertices.append(transformed)
        faces.append(local_faces + vertex_offset)
        vertex_offset += local_vertices.shape[0]
    return torch.cat(vertices, dim=1), torch.cat(faces, dim=0)


def prepare_targets(
    masks: dict[str, np.ndarray],
    render_size: tuple[int, int],
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    targets: dict[str, torch.Tensor] = {}
    outside_distances: dict[str, torch.Tensor] = {}
    height, width = render_size
    for name, mask in masks.items():
        resized = cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        target = torch.as_tensor(resized, device=device, dtype=torch.float32)
        targets[name] = target
        binary = resized >= 0.5
        outside = cv2.distanceTransform(
            (~binary).astype(np.uint8), cv2.DIST_L2, 3
        )
        outside_distances[name] = torch.as_tensor(
            outside, device=device, dtype=torch.float32
        )
    return targets, outside_distances


def part_mask_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    outside_distance: torch.Tensor,
) -> torch.Tensor:
    predicted = predicted.clamp(0.0, 1.0)
    eps = 1e-6
    intersection = torch.sum(predicted * target)
    dice = 1.0 - (2.0 * intersection + eps) / (
        torch.sum(predicted) + torch.sum(target) + eps
    )
    outside = torch.sum(predicted * outside_distance) / (
        torch.sum(predicted) + eps
    )
    # Dice supplies bidirectional coverage while the distance term strongly
    # distinguishes a nearby silhouette from an equally sized remote one.
    return dice + 0.025 * outside


def project_ordered_tips(
    camera_rotation: torch.Tensor,
    camera_translation: torch.Tensor,
    component_rotations: torch.Tensor,
    component_translations: torch.Tensor,
    K: torch.Tensor,
) -> dict[str, torch.Tensor]:
    definitions = {
        # Paper mesh 2 appears as the image-left jaw on the calibrated anchor.
        "jaw_left": (2, (0.0, 0.0004, 0.0096)),
        "jaw_right": (3, (0.0, -0.0004, 0.0096)),
    }
    output: dict[str, torch.Tensor] = {}
    for name, (mesh_id, local_xyz) in definitions.items():
        local = torch.tensor(
            local_xyz,
            device=camera_rotation.device,
            dtype=torch.float32,
        )
        paper_point = (
            component_rotations[:, mesh_id] @ local.view(1, 3, 1)
        ).squeeze(-1) + component_translations[:, mesh_id]
        camera_point = (
            camera_rotation @ paper_point.unsqueeze(-1)
        ).squeeze(-1) + camera_translation
        normalized = camera_point / camera_point[:, 2:3]
        output[name] = (normalized @ K.T)[:, :2]
    return output


def bounded_state(
    raw: torch.Tensor,
    prior_rotation: torch.Tensor,
    prior_translation: torch.Tensor,
    prior_joints: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    max_rotation = math.radians(args.max_rotation_deg)
    rotation_vector = (
        max_rotation / math.sqrt(3.0) * torch.tanh(raw[:3])
    )
    # Small Euler residuals avoid the undefined axis derivative of an exactly
    # zero axis-angle vector in some Kornia/PyTorch combinations.
    rx, ry, rz = rotation_vector
    one = torch.ones((), device=raw.device, dtype=raw.dtype)
    zero = torch.zeros((), device=raw.device, dtype=raw.dtype)
    Rx = torch.stack(
        (
            torch.stack((one, zero, zero)),
            torch.stack((zero, torch.cos(rx), -torch.sin(rx))),
            torch.stack((zero, torch.sin(rx), torch.cos(rx))),
        )
    )
    Ry = torch.stack(
        (
            torch.stack((torch.cos(ry), zero, torch.sin(ry))),
            torch.stack((zero, one, zero)),
            torch.stack((-torch.sin(ry), zero, torch.cos(ry))),
        )
    )
    Rz = torch.stack(
        (
            torch.stack((torch.cos(rz), -torch.sin(rz), zero)),
            torch.stack((torch.sin(rz), torch.cos(rz), zero)),
            torch.stack((zero, zero, one)),
        )
    )
    residual_rotation = (Rz @ Ry @ Rx).unsqueeze(0)
    camera_rotation = residual_rotation @ prior_rotation
    translation_delta = (
        args.max_translation_mm
        / 1000.0
        / math.sqrt(3.0)
        * torch.tanh(raw[3:6])
    )
    camera_translation = prior_translation + translation_delta.unsqueeze(0)
    wrist_delta = math.radians(args.max_wrist_deg) * torch.tanh(raw[6:8])
    jaw_delta = math.radians(args.max_jaw_deg) * torch.tanh(raw[8])
    joints = prior_joints.clone()
    joints[:2] = joints[:2] + wrist_delta
    joints[2:] = joints[2:] + jaw_delta
    joints = torch.stack(
        (
            joints[0].clamp(-1.5707, 1.5707),
            joints[1].clamp(-1.3963, 1.3963),
            joints[2].clamp(0.0, math.pi / 2.0),
            joints[3].clamp(0.0, math.pi / 2.0),
        )
    )
    residuals = {
        "rotation_vector": rotation_vector,
        "translation_delta": translation_delta,
        "wrist_delta": wrist_delta,
        "jaw_delta": jaw_delta,
    }
    return camera_rotation, camera_translation, joints.unsqueeze(0), residuals


def render_parts(
    model: object,
    renderer: object,
    camera_rotation: torch.Tensor,
    camera_translation: torch.Tensor,
    joints: torch.Tensor,
    render_size: tuple[int, int],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    from diffcali.eval_dvrk.LND_fk import batch_lndFK

    component_rotations, component_translations = batch_lndFK(joints)
    geometry = {
        "body": (0, 1),
        "jaw_left": (2,),
        "jaw_right": (3,),
    }
    masks = {}
    for name, mesh_ids in geometry.items():
        vertices, faces = assemble_part_geometry(
            renderer,
            component_rotations,
            component_translations,
            mesh_ids,
        )
        masks[name] = model.render_robot_mask_batch_nvdiffrast_rotmat(
            camera_rotation,
            camera_translation,
            vertices,
            faces,
            renderer,
            resolution=render_size,
        ).squeeze(0)
    return masks, component_rotations, component_translations


def refine_frame(
    *,
    args: argparse.Namespace,
    model: object,
    renderer: object,
    K: torch.Tensor,
    masks: dict[str, np.ndarray],
    observed_tips: dict[str, np.ndarray],
    prior_ctr: np.ndarray,
    prior_joints_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    device = model.device
    prior_matrix = ctr_to_matrix(prior_ctr)
    prior_rotation = torch.as_tensor(
        prior_matrix[:3, :3], device=device, dtype=torch.float32
    ).unsqueeze(0)
    prior_translation = torch.as_tensor(
        prior_matrix[:3, 3], device=device, dtype=torch.float32
    ).unsqueeze(0)
    prior_joints = torch.as_tensor(
        prior_joints_np, device=device, dtype=torch.float32
    )
    targets, outside_distances = prepare_targets(
        masks, (args.render_height, args.render_width), device
    )
    observed_tip_tensors = {
        name: torch.as_tensor(value, device=device, dtype=torch.float32)
        for name, value in observed_tips.items()
    }

    raw = torch.nn.Parameter(torch.zeros(9, device=device, dtype=torch.float32))
    optimizer = torch.optim.Adam([raw], lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.iterations, 1), eta_min=0.002
    )
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_raw = raw.detach().clone()
    for iteration in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        camera_rotation, camera_translation, joints, residuals = bounded_state(
            raw,
            prior_rotation,
            prior_translation,
            prior_joints,
            args,
        )
        predicted, component_rotations, component_translations = render_parts(
            model,
            renderer,
            camera_rotation,
            camera_translation,
            joints,
            (args.render_height, args.render_width),
        )
        losses = {
            name: part_mask_loss(
                predicted[name], targets[name], outside_distances[name]
            )
            for name in PART_NAMES
        }
        projected_tips = project_ordered_tips(
            camera_rotation,
            camera_translation,
            component_rotations,
            component_translations,
            K,
        )
        tip_loss = sum(
            F.smooth_l1_loss(
                projected_tips[name] / 10.0,
                observed_tip_tensors[name].view(1, 2) / 10.0,
                beta=0.5,
                reduction="sum",
            )
            for name in ("jaw_left", "jaw_right")
        )
        prior_loss = (
            torch.sum(
                (
                    residuals["rotation_vector"]
                    / max(math.radians(args.max_rotation_deg), 1e-6)
                )
                ** 2
            )
            + torch.sum(
                (
                    residuals["translation_delta"]
                    / max(args.max_translation_mm / 1000.0, 1e-6)
                )
                ** 2
            )
            + torch.sum(
                (
                    residuals["wrist_delta"]
                    / max(math.radians(args.max_wrist_deg), 1e-6)
                )
                ** 2
            )
            + (
                residuals["jaw_delta"]
                / max(math.radians(args.max_jaw_deg), 1e-6)
            )
            ** 2
        )
        total = (
            1.0 * losses["body"]
            + 0.8 * losses["jaw_left"]
            + 0.8 * losses["jaw_right"]
            + 1.5 * tip_loss
            + 0.08 * prior_loss
        )
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite part refinement loss")
        total.backward()
        torch.nn.utils.clip_grad_norm_([raw], 5.0)
        optimizer.step()
        scheduler.step()
        loss_value = float(total.detach().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_raw = raw.detach().clone()
        if iteration % 25 == 0 or iteration + 1 == args.iterations:
            item = {
                "iteration": iteration,
                "total": loss_value,
                "body": float(losses["body"].detach().item()),
                "jaw_left": float(losses["jaw_left"].detach().item()),
                "jaw_right": float(losses["jaw_right"].detach().item()),
                "tip": float(tip_loss.detach().item()),
                "prior": float(prior_loss.detach().item()),
            }
            history.append(item)
            print(json.dumps(item))

    with torch.no_grad():
        camera_rotation, camera_translation, joints, _ = bounded_state(
            best_raw,
            prior_rotation,
            prior_translation,
            prior_joints,
            args,
        )
    corrected = np.eye(4, dtype=np.float64)
    corrected[:3, :3] = camera_rotation[0].cpu().numpy()
    corrected[:3, 3] = camera_translation[0].cpu().numpy()
    return (
        matrix_to_ctr(corrected),
        joints[0].cpu().numpy().astype(np.float32),
        history,
    )


@torch.no_grad()
def render_state_numpy(
    *,
    model: object,
    renderer: object,
    K: torch.Tensor,
    ctr: np.ndarray,
    joints: np.ndarray,
    render_size: tuple[int, int],
) -> dict[str, np.ndarray]:
    matrix = ctr_to_matrix(ctr)
    camera_rotation = torch.as_tensor(
        matrix[:3, :3], device=model.device, dtype=torch.float32
    ).unsqueeze(0)
    camera_translation = torch.as_tensor(
        matrix[:3, 3], device=model.device, dtype=torch.float32
    ).unsqueeze(0)
    joints_tensor = torch.as_tensor(
        joints, device=model.device, dtype=torch.float32
    ).unsqueeze(0)
    masks, component_rotations, component_translations = render_parts(
        model,
        renderer,
        camera_rotation,
        camera_translation,
        joints_tensor,
        render_size,
    )
    return {
        **{
            name: masks[name].detach().cpu().numpy()
            for name in PART_NAMES
        },
        **{
            f"{name}_tip_xy": value[0].detach().cpu().numpy()
            for name, value in project_ordered_tips(
                camera_rotation,
                camera_translation,
                component_rotations,
                component_translations,
                K,
            ).items()
        },
    }


def dice_score(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted_bool = predicted >= 0.25
    target_bool = target.astype(bool)
    return float(
        (2.0 * np.count_nonzero(predicted_bool & target_bool) + 1e-6)
        / (
            np.count_nonzero(predicted_bool)
            + np.count_nonzero(target_bool)
            + 1e-6
        )
    )


def contour_p95(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted_bool = (predicted >= 0.25).astype(np.uint8)
    target_bool = target.astype(np.uint8)
    pred_contours, _ = cv2.findContours(
        predicted_bool, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    target_contours, _ = cv2.findContours(
        target_bool, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not pred_contours or not target_contours:
        return float("inf")
    pred_boundary = np.zeros_like(predicted_bool)
    target_boundary = np.zeros_like(target_bool)
    cv2.drawContours(pred_boundary, pred_contours, -1, 1, 1)
    cv2.drawContours(target_boundary, target_contours, -1, 1, 1)
    distance_to_target = cv2.distanceTransform(
        1 - target_boundary, cv2.DIST_L2, 3
    )
    distance_to_pred = cv2.distanceTransform(
        1 - pred_boundary, cv2.DIST_L2, 3
    )
    distances = np.concatenate(
        (
            distance_to_target[pred_boundary.astype(bool)],
            distance_to_pred[target_boundary.astype(bool)],
        )
    )
    return float(np.percentile(distances, 95))


def state_metrics(
    rendered: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    observed_tips: dict[str, np.ndarray],
) -> dict[str, object]:
    part_metrics = {
        name: {
            "dice": dice_score(rendered[name], targets[name]),
            "contour_distance_px_p95": contour_p95(
                rendered[name], targets[name]
            ),
        }
        for name in PART_NAMES
    }
    tip_metrics = {
        name: {
            "predicted_xy": rendered[f"{name}_tip_xy"].tolist(),
            "observed_xy": observed_tips[name].tolist(),
            "error_px": float(
                np.linalg.norm(
                    rendered[f"{name}_tip_xy"] - observed_tips[name]
                )
            ),
        }
        for name in ("jaw_left", "jaw_right")
    }
    return {"parts": part_metrics, "tips": tip_metrics}


def tint_rendered_parts(
    image: np.ndarray,
    rendered: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    observed_tips: dict[str, np.ndarray],
    title: str,
) -> np.ndarray:
    output = image.copy()
    for name in PART_NAMES:
        predicted = rendered[name] >= 0.25
        tint = np.zeros_like(output)
        tint[predicted] = PART_COLORS[name]
        output = cv2.addWeighted(output, 1.0, tint, 0.42, 0.0)
        predicted_contours, _ = cv2.findContours(
            predicted.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            output, predicted_contours, -1, PART_COLORS[name], 2
        )
        target_contours, _ = cv2.findContours(
            targets[name].astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(output, target_contours, -1, (255, 255, 255), 1)
    for name in ("jaw_left", "jaw_right"):
        predicted_tip = tuple(
            np.rint(rendered[f"{name}_tip_xy"]).astype(int)
        )
        observed_tip = tuple(np.rint(observed_tips[name]).astype(int))
        cv2.line(output, predicted_tip, observed_tip, (0, 255, 255), 2)
        cv2.circle(output, predicted_tip, 5, (255, 255, 0), -1)
        cv2.circle(output, observed_tip, 5, (0, 255, 255), -1)
    cv2.rectangle(output, (0, 0), (output.shape[1], 68), (0, 0, 0), -1)
    cv2.putText(
        output,
        title,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    left_error = np.linalg.norm(
        rendered["jaw_left_tip_xy"] - observed_tips["jaw_left"]
    )
    right_error = np.linalg.norm(
        rendered["jaw_right_tip_xy"] - observed_tips["jaw_right"]
    )
    cv2.putText(
        output,
        f"ordered tip error L/R={left_error:.1f}/{right_error:.1f}px; white=target contour",
        (12, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def target_panel(
    image: np.ndarray,
    masks: dict[str, np.ndarray],
    observed_tips: dict[str, np.ndarray],
    frame_index: int,
) -> np.ndarray:
    output = image.copy()
    for name in PART_NAMES:
        tint = np.zeros_like(output)
        tint[masks[name]] = PART_COLORS[name]
        output = cv2.addWeighted(output, 1.0, tint, 0.42, 0.0)
        contours, _ = cv2.findContours(
            masks[name].astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(output, contours, -1, PART_COLORS[name], 2)
    for name in ("jaw_left", "jaw_right"):
        cv2.circle(
            output,
            tuple(np.rint(observed_tips[name]).astype(int)),
            6,
            (0, 255, 255),
            -1,
        )
    cv2.rectangle(output, (0, 0), (output.shape[1], 68), (0, 0, 0), -1)
    cv2.putText(
        output,
        f"frame {frame_index}: part observations",
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "body=orange, image-left jaw=magenta, image-right jaw=red",
        (12, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    frame_indices = [int(value) for value in args.frames.split(",") if value]
    if not frame_indices:
        raise ValueError("No frame indices requested")
    if args.render_height <= 0 or args.render_width <= 0:
        raise ValueError("Render size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = TrackingInputs.load()
    exact = np.load(args.states)
    if max(frame_indices) >= len(exact["pure_ctr"]):
        raise IndexError("Requested frame is outside the saved exact sequence")

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

    all_metrics: dict[str, object] = {}
    histories: dict[str, object] = {}
    corrected_ctr = []
    corrected_joints = []
    prior_ctr_list = []
    prior_joints_list = []
    sheets = []
    for frame_index in frame_indices:
        print(f"refining frame {frame_index}")
        masks, observed_tips = load_part_observation(
            args.masks_dir, frame_index
        )
        prior_ctr, prior_joints = paper_lnd_prior(
            inputs,
            exact["pure_ctr"],
            exact["pure_joints"],
            frame_index,
        )
        frame_ctr, frame_joints, history = refine_frame(
            args=args,
            model=model,
            renderer=renderer,
            K=K,
            masks=masks,
            observed_tips=observed_tips,
            prior_ctr=prior_ctr,
            prior_joints_np=prior_joints,
        )
        histories[str(frame_index)] = history
        corrected_ctr.append(frame_ctr)
        corrected_joints.append(frame_joints)
        prior_ctr_list.append(prior_ctr)
        prior_joints_list.append(prior_joints)

        render_size = masks["body"].shape
        variants = {
            "paper_exact": render_state_numpy(
                model=model,
                renderer=renderer,
                K=K,
                ctr=exact["pure_ctr"][frame_index],
                joints=exact["pure_joints"][frame_index],
                render_size=render_size,
            ),
            "lnd_prior": render_state_numpy(
                model=model,
                renderer=renderer,
                K=K,
                ctr=prior_ctr,
                joints=prior_joints,
                render_size=render_size,
            ),
            "part_corrected": render_state_numpy(
                model=model,
                renderer=renderer,
                K=K,
                ctr=frame_ctr,
                joints=frame_joints,
                render_size=render_size,
            ),
        }
        frame_metrics = {
            name: state_metrics(rendered, masks, observed_tips)
            for name, rendered in variants.items()
        }
        prior_matrix = ctr_to_matrix(prior_ctr)
        corrected_matrix = ctr_to_matrix(frame_ctr)
        frame_metrics["correction"] = {
            "translation_mm": float(
                np.linalg.norm(
                    corrected_matrix[:3, 3] - prior_matrix[:3, 3]
                )
                * 1000.0
            ),
            "rotation_deg": float(
                np.degrees(
                    Rotation.from_matrix(
                        corrected_matrix[:3, :3]
                        @ prior_matrix[:3, :3].T
                    ).magnitude()
                )
            ),
            "joint_delta_deg": np.degrees(
                frame_joints - prior_joints
            ).tolist(),
        }
        all_metrics[str(frame_index)] = frame_metrics

        native_path = (
            REPO
            / f"data/super/grasp5_native/rgb/{frame_index:06d}-left.png"
        )
        native = cv2.imread(str(native_path))
        if native is None:
            raise FileNotFoundError(native_path)
        image = cv2.resize(
            native,
            (masks["body"].shape[1], masks["body"].shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        panels = [target_panel(image, masks, observed_tips, frame_index)]
        labels = {
            "paper_exact": "paper exact (drifted observation recursion)",
            "lnd_prior": "registered strict LND prior",
            "part_corrected": "part-aware bounded correction",
        }
        for name in ("paper_exact", "lnd_prior", "part_corrected"):
            panels.append(
                tint_rendered_parts(
                    image,
                    variants[name],
                    masks,
                    observed_tips,
                    labels[name],
                )
            )
        comparison = np.hstack(panels)
        output_path = (
            args.output_dir / f"frame{frame_index:06d}_comparison.png"
        )
        cv2.imwrite(str(output_path), comparison)
        sheets.append(cv2.resize(comparison, (1920, 540)))
        print(output_path)
        print(json.dumps(frame_metrics, indent=2))

    np.savez_compressed(
        args.output_dir / "corrected_keyframe_states.npz",
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        corrected_ctr=np.asarray(corrected_ctr, dtype=np.float32),
        corrected_joints=np.asarray(corrected_joints, dtype=np.float32),
        prior_ctr=np.asarray(prior_ctr_list, dtype=np.float32),
        prior_joints=np.asarray(prior_joints_list, dtype=np.float32),
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2), encoding="utf-8"
    )
    (args.output_dir / "optimization_history.json").write_text(
        json.dumps(histories, indent=2), encoding="utf-8"
    )
    cv2.imwrite(str(args.output_dir / "contact_sheet.png"), np.vstack(sheets))
    print(args.output_dir / "contact_sheet.png")


if __name__ == "__main__":
    main()
