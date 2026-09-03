#!/usr/bin/env python3
"""Original-style online CMA-ES correction from stereo SurgicalSAM2 masks.

The observation/update structure follows online_dvrk_tracking commit
cb2a264167aaf05b5a9c20da885d48568f78311f:

* whole-instrument SAM2 masks;
* NvDiffRast silhouette MSE + area loss;
* ContourTipNet keypoints when geometrically valid;
* three CMA-ES generations per time step, initialized from the previous state;
* the upstream Kalman smoother.

The stereo extension evaluates one shared physical correction in two frozen
rectified cameras.  The raw q7 + calibration + hand-eye + LND trajectory is
retained as the backbone, so CMA-ES estimates bounded residuals instead of
reinitializing an unconstrained camera pose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from build_super_raw_psm_kinematics import (
    load_stereo_calibration,
    nearest_indices,
)
from optimize_super_p420006_stereo_keyframes import (
    LINK_MESHES,
    P420_ROOT,
    PAPER_REPO,
    RAW_ROOT,
    P420StereoRenderer,
)
from super_paper_lnd_stereo_renderer import (
    PAPER_LND_MESHES,
    PaperLNDStereoRenderer,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
VISUAL_ROOT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_p420006_stereo_v1"
)
SAM2_ROOT = VISUAL_ROOT / "surgicalsam2_stereo_sequence_v1"
UPSTREAM_COMMIT = "cb2a264167aaf05b5a9c20da885d48568f78311f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run original-style online CMA-ES on 1631 stereo SurgicalSAM2 "
            "observations while preserving the raw kinematic backbone."
        )
    )
    parser.add_argument("--visual-root", type=Path, default=VISUAL_ROOT)
    parser.add_argument(
        "--observation-mode",
        choices=("first_pair_union", "multianchor_parts"),
        default="first_pair_union",
    )
    parser.add_argument(
        "--geometry",
        choices=("p420006", "paper_lnd"),
        default="p420006",
        help=(
            "Use either the registered P420006 carrier or the upstream "
            "paper repository's exact four-part LND CAD/FK."
        ),
    )
    parser.add_argument(
        "--masks",
        type=Path,
        default=SAM2_ROOT / "stereo_surgicalsam2_masks.npz",
    )
    parser.add_argument(
        "--sam2-report",
        type=Path,
        default=SAM2_ROOT / "report.json",
    )
    parser.add_argument(
        "--first-annotation",
        type=Path,
        default=VISUAL_ROOT / "annotations/keyframe_00.json",
        help=(
            "The first stereo pair annotation. Its two manually marked jaw "
            "tips are used only for original-style first-frame initialization."
        ),
    )
    parser.add_argument(
        "--anchor-state",
        type=Path,
        default=None,
        help=(
            "Completed live-SAM2 anchor state for multianchor_parts mode."
        ),
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
        "--driver",
        type=Path,
        default=P420_ROOT / "psm_p420006_gui_pose_driver.npz",
    )
    parser.add_argument("--mesh-dir", type=Path, default=P420_ROOT / "meshes")
    parser.add_argument("--paper-repo", type=Path, default=PAPER_REPO)
    parser.add_argument(
        "--tipnet-checkpoint",
        type=Path,
        default=PAPER_REPO / "ContourTipNet/models/cnn_model.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SAM2_ROOT / "online_stereo_cma_v2",
    )
    parser.add_argument("--cuda-device", type=int, default=1)
    parser.add_argument("--render-scale", type=float, default=0.25)
    parser.add_argument(
        "--global-iterations",
        type=int,
        default=100,
        help="First-pair initialization generations, matching upstream final_iters.",
    )
    parser.add_argument(
        "--global-population",
        type=int,
        default=30,
        help="First-pair population, matching upstream min(popsize, 30).",
    )
    parser.add_argument(
        "--global-prior-weight",
        type=float,
        default=0.08,
        help=(
            "Normalized raw-backbone residual prior for first-pair "
            "initialization."
        ),
    )
    parser.add_argument("--online-iterations", type=int, default=3)
    parser.add_argument("--population", type=int, default=70)
    parser.add_argument(
        "--local-prior-weight",
        type=float,
        default=0.5,
        help=(
            "Normalized raw-backbone residual prior per online pair. This "
            "prevents an incomplete occluded mask from pulling every degree "
            "of freedom to its bound."
        ),
    )
    parser.add_argument(
        "--unbounded-translation",
        action="store_true",
        help=(
            "Remove every hard XYZ correction bound. Translation is decoded "
            "linearly and remains controlled only by the soft prior."
        ),
    )
    parser.add_argument(
        "--global-translation-scale-mm",
        type=float,
        default=5.0,
        help=(
            "Metres-per-unit conditioning scale expressed in millimetres for "
            "unbounded global XYZ; this is not a limit."
        ),
    )
    parser.add_argument(
        "--local-translation-scale-mm",
        type=float,
        default=5.0,
        help=(
            "Metres-per-unit conditioning scale expressed in millimetres for "
            "unbounded per-pair XYZ; this is not a limit."
        ),
    )
    parser.add_argument(
        "--manual-anchor-tip-weight-multiplier",
        type=float,
        default=1.0,
        help=(
            "Multiplier on the upstream 3e-3 jaw-tip loss at manually "
            "annotated stereo anchors."
        ),
    )
    parser.add_argument("--seed", type=int, default=420006)
    parser.add_argument("--tipnet-batch-size", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument(
        "--freeze-jaw-correction",
        action="store_true",
        help=(
            "Force visual q7 jaw-angle residual to zero. The raw measured "
            "jaw state, including the closed middle interval, is preserved."
        ),
    )
    parser.add_argument(
        "--freeze-all-joint-corrections",
        action="store_true",
        help=(
            "Force all visual q4..q7 residuals to exactly zero. Only the "
            "shared stereo XYZ translation and 3D rotation are optimized; "
            "every raw robot joint remains unchanged."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_mask(
    packed: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    return np.unpackbits(
        packed,
        bitorder="little",
        count=shape[0] * shape[1],
    ).reshape(shape).astype(bool)


def combine_renderer_groups(renderer: P420StereoRenderer) -> None:
    vertices: list[torch.Tensor] = []
    faces: list[torch.Tensor] = []
    link_ids: list[torch.Tensor] = []
    offset = 0
    for name in ("body", "jaw_a", "jaw_b"):
        group = renderer.groups[name]
        vertices.append(group["vertices"])
        faces.append(group["faces"] + offset)
        link_ids.append(group["link_ids"])
        offset += len(group["vertices"])
    renderer.groups["whole"] = {
        "vertices": torch.cat(vertices),
        "faces": torch.cat(faces),
        "link_ids": torch.cat(link_ids),
    }


def render_whole_and_tips(
    renderer: P420StereoRenderer,
    raw_parameters: torch.Tensor,
    raw_q7: torch.Tensor,
    T_left_base: torch.Tensor,
    side: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    links = renderer.link_transforms(
        raw_parameters,
        raw_q7,
        T_left_base,
        side,
    )
    group = renderer.groups["whole"]
    link_ids = group["link_ids"]
    rotations = links[:, link_ids, :3, :3]
    translations = links[:, link_ids, :3, 3]
    camera_vertices = (
        torch.einsum(
            "bnij,nj->bni",
            rotations,
            group["vertices"],
        )
        + translations
    )
    clip = renderer.project_clip(camera_vertices, side)
    raster, _ = renderer.dr.rasterize(
        renderer.context,
        clip,
        group["faces"],
        resolution=[renderer.height, renderer.width],
    )
    # Upstream disables antialiasing for black-box CMA-ES.
    mask = (raster[..., 3] > 0).to(torch.float32)
    return mask, renderer.project_tips(links, side)


def tip_keypoint_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Upstream permutation-invariant two-tip + centerline L1 loss."""

    target_batch = target.unsqueeze(0).expand(len(predicted), -1, -1)
    direct = torch.linalg.vector_norm(
        predicted[:, 0] - target_batch[:, 0],
        ord=1,
        dim=1,
    ) + torch.linalg.vector_norm(
        predicted[:, 1] - target_batch[:, 1],
        ord=1,
        dim=1,
    )
    swapped = torch.linalg.vector_norm(
        predicted[:, 0] - target_batch[:, 1],
        ord=1,
        dim=1,
    ) + torch.linalg.vector_norm(
        predicted[:, 1] - target_batch[:, 0],
        ord=1,
        dim=1,
    )
    assignment = torch.clamp(
        torch.minimum(direct, swapped) - threshold,
        min=0.0,
    )
    centerline = torch.clamp(
        torch.linalg.vector_norm(
            predicted.mean(dim=1) - target_batch.mean(dim=1),
            ord=1,
            dim=1,
        )
        - threshold,
        min=0.0,
    )
    return assignment + centerline


class WholeStereoObjective:
    def __init__(
        self,
        *,
        renderer: P420StereoRenderer,
        raw_q7: dict[str, torch.Tensor],
        T_left_base: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        tips: dict[str, torch.Tensor | None],
        prior_weight: float = 0.0,
        tip_weight: float = 3.0e-3,
    ) -> None:
        self.renderer = renderer
        self.raw_q7 = raw_q7
        self.T_left_base = T_left_base
        self.targets = targets
        self.tips = tips
        self.prior_weight = float(prior_weight)
        self.tip_weight = float(tip_weight)

    @torch.no_grad()
    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        losses = torch.zeros(
            len(values),
            device=values.device,
            dtype=torch.float32,
        )
        tip_count = 0
        for side in ("left", "right"):
            predicted, predicted_tips = render_whole_and_tips(
                self.renderer,
                values,
                self.raw_q7[side],
                self.T_left_base[side],
                side,
            )
            target = self.targets[side].unsqueeze(0).expand_as(predicted)
            mse = F.mse_loss(predicted, target, reduction="none").mean(
                dim=(1, 2)
            )
            area = torch.abs(
                predicted.sum(dim=(1, 2)) - target.sum(dim=(1, 2))
            )
            losses = losses + 6.0 * mse + 6.0e-6 * area
            if self.tips[side] is not None:
                losses = losses + self.tip_weight * tip_keypoint_loss(
                    predicted_tips,
                    self.tips[side],
                    threshold=5.0,
                )
                tip_count += 1
        losses = losses / 2.0
        if tip_count:
            # The division above matches the stereo mean for both mask and
            # keypoint observations.
            pass
        if self.prior_weight:
            normalized = torch.tanh(values)
            if self.renderer.translation_unbounded:
                # XYZ has no hard clamp.  Penalize its scale-normalized raw
                # coordinate quadratically so bad occlusion masks cannot get
                # a free, arbitrarily large translation.
                normalized = normalized.clone()
                normalized[:, 3:6] = values[:, 3:6]
            losses = losses + self.prior_weight * torch.mean(
                normalized**2, dim=1
            )
        return losses


def resize_target(
    packed: np.ndarray,
    mask_shape: tuple[int, int],
    render_size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    mask = unpack_mask(packed, mask_shape).astype(np.uint8)
    resized = cv2.resize(
        mask,
        (render_size[1], render_size[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return torch.as_tensor(
        resized,
        device=device,
        dtype=torch.float32,
    )


def load_manual_first_tips(
    *,
    annotation_path: Path,
    render_size_hw: tuple[int, int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    if int(annotation.get("strict_pair_slot", -1)) != 0:
        raise ValueError("The first-frame annotation must target strict pair slot 0")
    image_width, image_height = map(int, annotation["image_size_wh"])
    render_height, render_width = render_size_hw
    scale = np.asarray(
        [render_width / image_width, render_height / image_height],
        dtype=np.float32,
    )
    result: dict[str, torch.Tensor] = {}
    for side in ("left", "right"):
        tips = annotation["views"][side]["tips"]
        points: list[list[float]] = []
        for name in ("jaw_1", "jaw_2"):
            record = tips[name]
            if record.get("visible") is not True:
                raise ValueError(
                    f"First-frame {side} {name} tip is not marked visible"
                )
            points.append([float(value) for value in record["point_xy"]])
        result[side] = torch.as_tensor(
            np.asarray(points, dtype=np.float32) * scale[None],
            device=device,
            dtype=torch.float32,
        )
    return result


def load_manual_multianchor_tips(
    *,
    anchor_slots: np.ndarray,
    tip_points_xy: np.ndarray,
    mask_shape: tuple[int, int],
    render_size_hw: tuple[int, int],
    device: torch.device,
) -> dict[int, dict[str, torch.Tensor]]:
    if tip_points_xy.shape != (len(anchor_slots), 2, 2, 2):
        raise ValueError(
            f"Unexpected multianchor tip shape {tip_points_xy.shape}"
        )
    render_height, render_width = render_size_hw
    scale = np.asarray(
        [
            render_width / mask_shape[1],
            render_height / mask_shape[0],
        ],
        dtype=np.float32,
    )
    output: dict[int, dict[str, torch.Tensor]] = {}
    for anchor_index, slot in enumerate(anchor_slots):
        output[int(slot)] = {
            side: torch.as_tensor(
                tip_points_xy[anchor_index, side_index] * scale[None],
                device=device,
                dtype=torch.float32,
            )
            for side_index, side in enumerate(("left", "right"))
        }
    return output


def raw_from_correction(
    correction: np.ndarray,
    bounds: np.ndarray,
    *,
    unbounded_translation: bool = False,
    translation_scale: float = 1.0,
) -> np.ndarray:
    normalized = np.clip(correction / bounds, -0.995, 0.995)
    raw = np.arctanh(normalized).astype(np.float32)
    if unbounded_translation:
        raw[3:6] = correction[3:6] / float(translation_scale)
    return raw


def assignment_by_prediction(
    detections: np.ndarray,
    predicted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    direct = np.linalg.norm(detections - predicted, axis=1)
    swapped = np.linalg.norm(detections[::-1] - predicted, axis=1)
    if swapped.sum() < direct.sum():
        return detections[::-1].copy(), swapped, True
    return detections.copy(), direct, False


def batched_tipnet_detections(
    *,
    packed_masks: dict[str, np.ndarray],
    mask_shape: tuple[int, int],
    count: int,
    checkpoint: Path,
    paper_repo: Path,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    sys.path.insert(0, str(paper_repo))
    from diffcali.utils.contour_tip_net import (
        Tip2DNet,
        get_local_maxima,
    )

    model = Tip2DNet().to(device).eval()
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    detections = np.full((count, 2, 2, 2), np.nan, dtype=np.float32)
    requests = [
        (slot, side_index, side)
        for slot in range(count)
        for side_index, side in enumerate(("left", "right"))
    ]
    for start in range(0, len(requests), batch_size):
        chunk = requests[start : start + batch_size]
        masks_np = np.stack(
            [
                unpack_mask(
                    packed_masks[side][slot],
                    mask_shape,
                ).astype(np.float32)
                for slot, _side_index, side in chunk
            ]
        )
        masks_tensor = torch.as_tensor(masks_np, device=device).unsqueeze(1)
        masks_small = F.interpolate(
            masks_tensor,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        with torch.inference_mode():
            raw_heatmaps = model.raw_predict(masks_small)
            heatmaps = torch.sigmoid(raw_heatmaps)
        raw_np = raw_heatmaps[:, 0].detach().cpu().numpy()
        heat_np = heatmaps[:, 0].detach().cpu().numpy()
        for local_index, (slot, side_index, _side) in enumerate(chunk):
            peaks = get_local_maxima(
                heat_np[local_index],
                min_distance=3,
                min_area=1,
                threshold=0.5,
            )
            peaks = sorted(
                peaks,
                key=lambda point: raw_np[
                    local_index,
                    point[0],
                    point[1],
                ],
                reverse=True,
            )[:2]
            if len(peaks) == 2:
                detections[slot, side_index] = np.asarray(
                    [
                        [
                            x / 224.0 * mask_shape[1],
                            y / 224.0 * mask_shape[0],
                        ]
                        for y, x in peaks
                    ],
                    dtype=np.float32,
                )
        if start == 0 or start + batch_size >= len(requests):
            print(
                f"TipNet {min(start + len(chunk), len(requests))}/"
                f"{len(requests)} views",
                flush=True,
            )
    del model
    torch.cuda.empty_cache()
    return detections


def gate_tipnet_stereo(
    *,
    detections: np.ndarray,
    predicted_at_mask_scale: dict[str, np.ndarray],
    distance_gate: float,
    epipolar_gate: float,
    disparity_gate: float,
) -> tuple[dict[str, np.ndarray | None], dict[str, Any]]:
    assigned: dict[str, np.ndarray] = {}
    distances: dict[str, np.ndarray] = {}
    swapped: dict[str, bool] = {}
    finite = True
    for side_index, side in enumerate(("left", "right")):
        candidate = detections[side_index]
        if not np.all(np.isfinite(candidate)):
            finite = False
            continue
        (
            assigned[side],
            distances[side],
            swapped[side],
        ) = assignment_by_prediction(
            candidate,
            predicted_at_mask_scale[side],
        )
    accepted = finite and all(
        float(np.max(distances[side])) <= distance_gate
        for side in ("left", "right")
    )
    epipolar_errors = np.full(2, np.nan, dtype=np.float32)
    disparity_errors = np.full(2, np.nan, dtype=np.float32)
    if finite:
        epipolar_errors = np.abs(
            assigned["left"][:, 1] - assigned["right"][:, 1]
        )
        observed_disparity = (
            assigned["left"][:, 0] - assigned["right"][:, 0]
        )
        predicted_disparity = (
            predicted_at_mask_scale["left"][:, 0]
            - predicted_at_mask_scale["right"][:, 0]
        )
        disparity_errors = np.abs(
            observed_disparity - predicted_disparity
        )
        accepted = (
            accepted
            and float(np.max(epipolar_errors)) <= epipolar_gate
            and float(np.max(disparity_errors)) <= disparity_gate
        )
    tips = {
        side: assigned[side] if accepted else None
        for side in ("left", "right")
    }
    diagnostic = {
        "accepted": bool(accepted),
        "finite_two_tip_detection": bool(finite),
        "assigned_tip_distance_to_kinematic_prior_px": {
            side: (
                distances[side].tolist() if side in distances else None
            )
            for side in ("left", "right")
        },
        "assignment_swapped": {
            side: swapped.get(side) for side in ("left", "right")
        },
        "epipolar_error_px": epipolar_errors.tolist(),
        "disparity_error_vs_kinematic_prior_px": (
            disparity_errors.tolist()
        ),
    }
    return tips, diagnostic


def make_bounds(
    rotation_deg: float,
    translation_xy_mm: float,
    translation_z_mm: float,
    roll_deg: float,
    wrist_deg: float,
    jaw_deg: float,
) -> np.ndarray:
    return np.asarray(
        [
            *([math.radians(rotation_deg)] * 3),
            translation_xy_mm * 1.0e-3,
            translation_xy_mm * 1.0e-3,
            translation_z_mm * 1.0e-3,
            math.radians(roll_deg),
            math.radians(wrist_deg),
            math.radians(wrist_deg),
            math.radians(jaw_deg),
        ],
        dtype=np.float32,
    )


def render_preview_panel(
    *,
    renderer: P420StereoRenderer,
    row: dict[str, Any],
    q7: np.ndarray,
    T_left_lnd: np.ndarray,
    left_q_index: np.ndarray,
    right_q_index: np.ndarray,
    raw_parameters: np.ndarray,
    targets_packed: dict[str, np.ndarray],
    mask_shape: tuple[int, int],
    visual_root: Path,
    label: str,
) -> np.ndarray:
    slot = int(row["strict_pair_slot"])
    panels: list[np.ndarray] = []
    for side in ("left", "right"):
        q_index = (
            int(left_q_index[slot])
            if side == "left"
            else int(right_q_index[slot])
        )
        values = torch.as_tensor(
            raw_parameters[None],
            device=renderer.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            predicted, tips = render_whole_and_tips(
                renderer,
                values,
                torch.as_tensor(
                    q7[q_index],
                    device=renderer.device,
                    dtype=torch.float32,
                ),
                torch.as_tensor(
                    T_left_lnd[q_index, 0],
                    device=renderer.device,
                    dtype=torch.float32,
                ),
                side,
            )
        prediction = predicted[0].cpu().numpy() > 0.5
        target = unpack_mask(targets_packed[side][slot], mask_shape)
        image = cv2.imread(
            str(visual_root / row[f"{side}_image"]),
            cv2.IMREAD_COLOR,
        )
        image = cv2.resize(
            image,
            (renderer.width, renderer.height),
            interpolation=cv2.INTER_AREA,
        )
        target_small = cv2.resize(
            target.astype(np.uint8),
            (renderer.width, renderer.height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        tint = np.zeros_like(image)
        tint[target_small] = (255, 80, 20)
        tint[prediction] = (30, 40, 255)
        overlap = target_small & prediction
        tint[overlap] = (60, 220, 60)
        active = target_small | prediction
        image[active] = cv2.addWeighted(
            image[active],
            0.45,
            tint[active],
            0.55,
            0.0,
        )
        for point in tips[0].cpu().numpy():
            cv2.circle(
                image,
                tuple(np.rint(point).astype(int)),
                5,
                (255, 255, 255),
                2,
            )
        panels.append(image)
    panel = np.concatenate(panels, axis=1)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(
        panel,
        f"KF {int(row['keyframe_index']):02d} {label}: "
        "blue=SAM2 red=CAD green=overlap",
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def main() -> None:
    args = parse_args()
    if args.global_translation_scale_mm <= 0.0:
        raise ValueError("--global-translation-scale-mm must be positive")
    if args.local_translation_scale_mm <= 0.0:
        raise ValueError("--local-translation-scale-mm must be positive")
    if args.manual_anchor_tip_weight_multiplier <= 0.0:
        raise ValueError(
            "--manual-anchor-tip-weight-multiplier must be positive"
        )
    if args.freeze_all_joint_corrections:
        # Freezing all distal joint residuals necessarily includes the jaw.
        args.freeze_jaw_correction = True
    manifest_path = args.visual_root / (
        "prompt_manifest.json"
        if args.observation_mode == "multianchor_parts"
        else "pair_manifest.json"
    )
    required_paths = [
        manifest_path,
        args.masks,
        args.sam2_report,
        args.kinematics,
        args.raw_model,
        args.tipnet_checkpoint,
    ]
    if args.observation_mode == "first_pair_union":
        required_paths.append(args.first_annotation)
    else:
        if args.anchor_state is None:
            raise ValueError(
                "--anchor-state is required for multianchor_parts"
            )
        required_paths.extend([args.anchor_state, args.calibration])
    if args.geometry == "p420006":
        required_paths.append(args.driver)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    schema_prefix = (
        "super_paper_lnd_stereo"
        if args.geometry == "paper_lnd"
        else "super_p420006_stereo"
    )
    mesh_names = (
        PAPER_LND_MESHES
        if args.geometry == "paper_lnd"
        else LINK_MESHES
    )
    renderer_class = (
        PaperLNDStereoRenderer
        if args.geometry == "paper_lnd"
        else P420StereoRenderer
    )
    for name in mesh_names:
        if not (args.mesh_dir / name).is_file():
            raise FileNotFoundError(args.mesh_dir / name)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.cuda_device)
    device = torch.device(f"cuda:{args.cuda_device}")
    import subprocess

    commit = subprocess.run(
        ["git", "-C", str(args.paper_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"Unexpected upstream commit {commit}")
    sys.path.insert(0, str(args.paper_repo))
    from diffcali.utils.cma_es import CMAES_cus
    from diffcali.utils.pose_tracker import KalmanFilter
    from evotorch import Problem
    from torch.quasirandom import SobolEngine

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_rows = manifest.get("keyframes", manifest.get("anchors"))
    if not isinstance(manifest_rows, list) or not manifest_rows:
        raise RuntimeError("Visual manifest contains no preview anchors")
    manifest_rows = [
        {
            **row,
            "keyframe_index": int(
                row.get("keyframe_index", row.get("anchor_index", index))
            ),
        }
        for index, row in enumerate(manifest_rows)
    ]
    sam2_report = json.loads(args.sam2_report.read_text(encoding="utf-8"))
    sam2_sequence = sam2_report.get("sequence", {})
    annotation_policy = sam2_report.get("manual_annotation_policy", {})
    if args.observation_mode == "first_pair_union":
        if (
            sam2_sequence.get("complete_strict_pair_sequence") is not True
            or int(sam2_sequence.get("pair_count", -1))
            != int(manifest["source_pair_count"])
            or "anchor_policy" in sam2_sequence
            or "causal_segment_start_slots" in sam2_sequence
            or "first_pair" not in annotation_policy
            or "remaining_pairs" not in annotation_policy
            or annotation_policy.get("manual_pixels_copied_to_output")
            is not False
            or annotation_policy.get("confidence_downweighting") is not False
        ):
            raise RuntimeError(
                "Expected complete first-pair-only causal stereo SAM2"
            )
    else:
        quality_gate = sam2_report.get("quality_gate", {})
        if (
            sam2_report.get("passed") is not True
            or sam2_sequence.get("complete_strict_pair_sequence") is not True
            or int(sam2_sequence.get("pair_count", -1))
            != int(manifest["source_pair_count"])
            or int(quality_gate.get("invalid_count", -1)) != 0
            or "no confidence downweighting"
            not in str(quality_gate.get("policy", ""))
        ):
            raise RuntimeError(
                "Multianchor part observations failed their hard gate"
            )
    raw_model = json.loads(args.raw_model.read_text(encoding="utf-8"))
    with np.load(args.masks, allow_pickle=False) as mask_data:
        expected_mask_schema = (
            "super_paper_lnd_stereo_surgicalsam2_multianchor_parts_v4"
            if args.observation_mode == "multianchor_parts"
            else f"{schema_prefix}_surgicalsam2_sequence_v1"
        )
        if mask_data["schema"].item() != expected_mask_schema:
            raise RuntimeError("Unexpected SAM2 mask schema")
        mask_shape = tuple(mask_data["mask_shape"].astype(int).tolist())
        packed_masks = {
            "left": mask_data["left_masks_packbits"].copy(),
            "right": mask_data["right_masks_packbits"].copy(),
        }
        stereo_left_index = mask_data["stereo_left_index"].astype(np.int64)
        stereo_right_index = mask_data["stereo_right_index"].astype(np.int64)
        if args.observation_mode == "multianchor_parts":
            if not np.all(mask_data["quality_valid"]):
                raise RuntimeError(
                    "An invalid union mask reached multianchor optimization"
                )
            anchor_slots = mask_data["anchor_slots"].astype(np.int64)
            anchor_tip_points = mask_data[
                "anchor_tip_points_xy_at_mask_resolution"
            ].astype(np.float32)
        else:
            anchor_slots = np.asarray([0], dtype=np.int64)
            anchor_tip_points = np.empty((0, 2, 2, 2), dtype=np.float32)
    with np.load(args.kinematics, allow_pickle=False) as raw:
        q7 = raw["q7"].astype(np.float32)
        T_left_lnd = raw[
            "T_rectified_left_camera_lnd_link"
        ].astype(np.float32)
        left_timestamps_ns = raw[
            "left_timestamps_ros_ns"
        ].astype(np.int64)
        right_timestamps_ns = raw[
            "right_timestamps_ros_ns"
        ].astype(np.int64)
        joint_timestamps_ns = raw[
            "joint_timestamps_ros_ns"
        ].astype(np.int64)
        raw_stereo_left_index = raw[
            "stereo_left_index"
        ].astype(np.int64)
        raw_stereo_right_index = raw[
            "stereo_right_index"
        ].astype(np.int64)
    if not (
        np.array_equal(stereo_left_index, raw_stereo_left_index)
        and np.array_equal(stereo_right_index, raw_stereo_right_index)
    ):
        raise RuntimeError("SAM2 masks do not follow the frozen 1631-pair mapping")
    count = len(stereo_left_index)
    if args.max_pairs > 0:
        count = min(count, args.max_pairs)
        for side in ("left", "right"):
            packed_masks[side] = packed_masks[side][:count]
        stereo_left_index = stereo_left_index[:count]
        stereo_right_index = stereo_right_index[:count]
    left_pair_ns = left_timestamps_ns[stereo_left_index]
    right_pair_ns = right_timestamps_ns[stereo_right_index]
    pair_ns = left_pair_ns + (right_pair_ns - left_pair_ns) // 2
    left_q_index = nearest_indices(left_pair_ns, joint_timestamps_ns)
    right_q_index = nearest_indices(right_pair_ns, joint_timestamps_ns)
    pair_q_index = nearest_indices(pair_ns, joint_timestamps_ns)
    keyframes_path = args.visual_root / "keyframes.npz"
    if keyframes_path.is_file():
        with np.load(keyframes_path, allow_pickle=False) as keyframes:
            K_left = keyframes["K_left_rect"].astype(np.float32)
            K_right = keyframes["K_right_rect"].astype(np.float32)
            T_right_left = keyframes[
                "T_rectified_right_camera_rectified_left_camera"
            ].astype(np.float32)
    else:
        calibration = load_stereo_calibration(args.calibration)
        K_left = calibration.K_left_rect.astype(np.float32)
        K_right = calibration.K_right_rect.astype(np.float32)
        T_right_left = np.asarray(
            raw_model["calibration_and_static_transforms"][
                "T_rectified_right_camera_rectified_left_camera"
            ],
            dtype=np.float32,
        )
    if args.geometry == "p420006":
        with np.load(args.driver, allow_pickle=False) as driver:
            link_offsets = driver[
                "T_lndlink_urdf_link"
            ].astype(np.float32)
            lnd_link_ids = driver["lnd_link_ids"].astype(np.int64)
    else:
        # The paper renderer evaluates the original paper component FK
        # directly; no P420006/static GUI adapter is allowed into this branch.
        link_offsets = np.repeat(
            np.eye(4, dtype=np.float32)[None],
            7,
            axis=0,
        )
        lnd_link_ids = np.arange(7, dtype=np.int64)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} is not empty; pass --overwrite"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "previews").mkdir(exist_ok=True)

    global_bounds = make_bounds(15.0, 8.0, 4.0, 20.0, 20.0, 30.0)
    local_bounds = make_bounds(2.0, 1.0, 0.75, 5.0, 5.0, 10.0)
    renderer = renderer_class(
        mesh_dir=args.mesh_dir,
        link_offsets=link_offsets,
        lnd_link_ids=lnd_link_ids,
        dh_parameters=raw_model["lnd"]["DH_params"],
        K_left=K_left,
        K_right=K_right,
        T_right_left=T_right_left,
        image_size=tuple(manifest["image_size_wh"]),
        render_scale=args.render_scale,
        bounds=global_bounds,
        device=device,
    )
    renderer.translation_unbounded = bool(args.unbounded_translation)
    renderer.translation_scale = (
        args.global_translation_scale_mm * 1.0e-3
    )
    combine_renderer_groups(renderer)
    if args.observation_mode == "multianchor_parts":
        manual_anchor_tips = load_manual_multianchor_tips(
            anchor_slots=anchor_slots,
            tip_points_xy=anchor_tip_points,
            mask_shape=mask_shape,
            render_size_hw=(renderer.height, renderer.width),
            device=device,
        )
        if 0 not in manual_anchor_tips:
            raise RuntimeError("Multianchor tips do not include slot 0")
        manual_first_tips = manual_anchor_tips[0]
    else:
        manual_first_tips = load_manual_first_tips(
            annotation_path=args.first_annotation,
            render_size_hw=(renderer.height, renderer.width),
            device=device,
        )
        manual_anchor_tips = {0: manual_first_tips}

    print("Running upstream Tip2DNet on stereo SAM2 masks", flush=True)
    tipnet = batched_tipnet_detections(
        packed_masks=packed_masks,
        mask_shape=mask_shape,
        count=count,
        checkpoint=args.tipnet_checkpoint,
        paper_repo=args.paper_repo,
        batch_size=args.tipnet_batch_size,
        device=device,
    )

    class ProblemAdapter(Problem):
        def __init__(self, objective: Any) -> None:
            self.objective = objective
            super().__init__(
                "min",
                solution_length=10,
                device=device,
                initial_bounds=(
                    [-float("inf")] * 10,
                    [float("inf")] * 10,
                ),
            )

        def _evaluate_batch(self, batch: Any) -> None:
            batch.set_evals(self.objective(batch.values))

    class ParameterConstrainedObjective:
        def __init__(self, objective: WholeStereoObjective) -> None:
            self.objective = objective

        @torch.no_grad()
        def __call__(self, values: torch.Tensor) -> torch.Tensor:
            if not (
                args.freeze_jaw_correction
                or args.freeze_all_joint_corrections
            ):
                return self.objective(values)
            constrained = values.clone()
            if args.freeze_all_joint_corrections:
                constrained[:, 6:10] = 0.0
            else:
                constrained[:, 9] = 0.0
            return self.objective(constrained)

    def observation(
        slot: int,
        tips: dict[str, torch.Tensor | None],
        *,
        prior_weight: float,
        tip_weight: float = 3.0e-3,
    ) -> ParameterConstrainedObjective:
        return ParameterConstrainedObjective(
            WholeStereoObjective(
                renderer=renderer,
                raw_q7={
                    "left": torch.as_tensor(
                        q7[left_q_index[slot]],
                        device=device,
                    ),
                    "right": torch.as_tensor(
                        q7[right_q_index[slot]],
                        device=device,
                    ),
                },
                T_left_base={
                    "left": torch.as_tensor(
                        T_left_lnd[left_q_index[slot], 0],
                        device=device,
                    ),
                    "right": torch.as_tensor(
                        T_left_lnd[right_q_index[slot], 0],
                        device=device,
                    ),
                },
                targets={
                    side: resize_target(
                        packed_masks[side][slot],
                        mask_shape,
                        (renderer.height, renderer.width),
                        device,
                    )
                    for side in ("left", "right")
                },
                tips=tips,
                prior_weight=prior_weight,
                tip_weight=tip_weight,
            )
        )

    renderer.bounds = torch.as_tensor(global_bounds, device=device)
    zero = torch.zeros((1, 10), device=device)
    # Upstream initializes from the first manually prompted frame, then tracks
    # causally.  The stereo extension uses the same first instant in both eyes.
    global_objective = observation(
        0,
        manual_first_tips,
        prior_weight=args.global_prior_weight,
        tip_weight=(
            3.0e-3 * args.manual_anchor_tip_weight_multiplier
        ),
    )

    class GlobalObjective:
        @torch.no_grad()
        def __call__(self, values: torch.Tensor) -> torch.Tensor:
            return global_objective(values)

    global_initial = float(GlobalObjective()(zero)[0].item())
    global_problem = ProblemAdapter(GlobalObjective())
    torch.manual_seed(args.seed)
    global_searcher = CMAES_cus(
        global_problem,
        center_init=torch.zeros(10, device=device),
        stdev_init=0.35,
        popsize=args.global_population,
        mu_size=min(15, args.global_population // 2),
        sobol=SobolEngine(10, scramble=True, seed=args.seed - 1),
    )
    global_best = torch.zeros(10, device=device)
    global_best_loss = global_initial
    for iteration in range(args.global_iterations):
        global_searcher.step()
        candidate_loss = float(global_searcher.status["pop_best_eval"])
        if candidate_loss < global_best_loss:
            global_best_loss = candidate_loss
            global_best = (
                global_searcher.status["pop_best"].values.detach().clone()
            )
        print(
            f"GLOBAL {iteration + 1:02d}/{args.global_iterations}: "
            f"{candidate_loss:.6f}",
            flush=True,
        )
    if args.freeze_all_joint_corrections:
        global_best[6:10] = 0.0
    elif args.freeze_jaw_correction:
        global_best[9] = 0.0
    global_correction = renderer.decode(global_best[None])[0]
    renderer.base_correction = global_correction.detach()
    print(
        f"Global SAM2 registration: {global_initial:.6f} -> "
        f"{global_best_loss:.6f}",
        flush=True,
    )

    renderer.bounds = torch.as_tensor(local_bounds, device=device)
    renderer.translation_scale = args.local_translation_scale_mm * 1.0e-3
    raw_measurements = np.zeros((count, 10), dtype=np.float32)
    measured_corrections = np.zeros((count, 10), dtype=np.float32)
    filtered_corrections = np.zeros((count, 10), dtype=np.float32)
    initial_losses = np.empty(count, dtype=np.float32)
    measured_losses = np.empty(count, dtype=np.float32)
    filtered_losses = np.empty(count, dtype=np.float32)
    tip_accepted = np.zeros(count, dtype=bool)
    tip_diagnostics: list[dict[str, Any]] = []
    mask_to_render_scale = np.asarray(
        [
            renderer.width / mask_shape[1],
            renderer.height / mask_shape[0],
        ],
        dtype=np.float32,
    )
    kalman = KalmanFilter(
        process_noise_pos=np.asarray(
            [
                2e-5,
                1e-4,
                2e-5,
                2e-5,
                2e-5,
                2e-5,
                1e-4,
                1e-4,
                1e-4,
                1e-4,
            ]
        ),
        process_noise_vel=np.asarray(
            [
                2e-4,
                1e-3,
                2e-4,
                2e-4,
                2e-4,
                2e-4,
                1e-3,
                1e-3,
                1e-3,
                1e-3,
            ]
        )
        * 10.0,
        measurement_noise=np.asarray(
            [
                2e-3,
                1e-2,
                2e-3,
                2e-3,
                2e-3,
                2e-3,
                5e-3,
                5e-3,
                5e-3,
                5e-3,
            ]
        ),
        joint_angles_lb=-local_bounds[6:10],
        joint_angles_ub=local_bounds[6:10],
    )
    center = torch.zeros(10, device=device)
    sobol = SobolEngine(10, scramble=True, seed=args.seed)
    for slot in range(count):
        if slot in manual_anchor_tips:
            observed_tips: dict[str, torch.Tensor | None] = {
                side: manual_anchor_tips[slot][side]
                for side in ("left", "right")
            }
            accepted_views = ["left", "right"]
            tip_source = (
                "manual_live_sam2_anchor"
                if args.observation_mode == "multianchor_parts"
                else "manual_first_pair"
            )
        else:
            observed_tips = {}
            accepted_views = []
            for side_index, side in enumerate(("left", "right")):
                candidate = tipnet[slot, side_index]
                if np.all(np.isfinite(candidate)):
                    observed_tips[side] = torch.as_tensor(
                        candidate * mask_to_render_scale[None],
                        device=device,
                        dtype=torch.float32,
                    )
                    accepted_views.append(side)
                else:
                    observed_tips[side] = None
            tip_source = "upstream_contour_tip_net"
        tip_accepted[slot] = len(accepted_views) == 2
        tip_diagnostics.append(
            {
                "strict_pair_slot": slot,
                "source": tip_source,
                "accepted_views": accepted_views,
                "both_views_accepted": bool(tip_accepted[slot]),
            }
        )
        objective = observation(
            slot,
            observed_tips,
            prior_weight=args.local_prior_weight,
            tip_weight=(
                3.0e-3 * args.manual_anchor_tip_weight_multiplier
                if slot in manual_anchor_tips
                else 3.0e-3
            ),
        )
        initial_losses[slot] = float(objective(center[None])[0].item())
        problem = ProblemAdapter(objective)
        searcher = CMAES_cus(
            problem,
            center_init=center,
            stdev_init=0.35,
            popsize=args.population,
            mu_size=min(15, args.population // 2),
            sobol=sobol,
        )
        best_raw = center.detach().clone()
        best_loss = initial_losses[slot]
        for _iteration in range(args.online_iterations):
            searcher.step()
            candidate_loss = float(searcher.status["pop_best_eval"])
            if candidate_loss < best_loss:
                best_loss = candidate_loss
                best_raw = (
                    searcher.status["pop_best"].values.detach().clone()
                )
        if args.freeze_all_joint_corrections:
            best_raw[6:10] = 0.0
        elif args.freeze_jaw_correction:
            best_raw[9] = 0.0
        measurement = renderer.decode(best_raw[None])[0].cpu().numpy()
        if slot == 0:
            kalman.reset(measurement)
            filtered = measurement.copy()
        else:
            kalman.update(measurement)
            filtered = np.asarray(kalman.get_x_hat(), dtype=np.float32)
        if args.unbounded_translation:
            bounded_indices = np.asarray([0, 1, 2, 6, 7, 8, 9])
            filtered[bounded_indices] = np.clip(
                filtered[bounded_indices],
                -local_bounds[bounded_indices],
                local_bounds[bounded_indices],
            )
        else:
            filtered = np.clip(filtered, -local_bounds, local_bounds)
        if args.freeze_all_joint_corrections:
            measurement[6:10] = 0.0
            filtered[6:10] = 0.0
        elif args.freeze_jaw_correction:
            measurement[9] = 0.0
            filtered[9] = 0.0
        center_np = raw_from_correction(
            filtered,
            local_bounds,
            unbounded_translation=args.unbounded_translation,
            translation_scale=renderer.translation_scale,
        )
        center = torch.as_tensor(center_np, device=device)
        filtered_loss = float(objective(center[None])[0].item())
        raw_measurements[slot] = best_raw.cpu().numpy()
        measured_corrections[slot] = measurement
        filtered_corrections[slot] = filtered
        measured_losses[slot] = best_loss
        filtered_losses[slot] = filtered_loss
        if (
            slot == 0
            or (slot + 1) % 100 == 0
            or slot + 1 == count
        ):
            print(
                f"[{slot + 1:04d}/{count:04d}] "
                f"{initial_losses[slot]:.5f} -> "
                f"{best_loss:.5f} -> KF {filtered_loss:.5f}; "
                f"tips={int(tip_accepted[slot])}",
                flush=True,
            )

    output_path = args.output_dir / "online_stereo_corrections.npz"
    np.savez_compressed(
        output_path,
        schema=np.asarray(
            f"{schema_prefix}_sam2_online_corrections_v1"
        ),
        strict_pair_slot=np.arange(count, dtype=np.int64),
        pair_timestamp_ros_ns=pair_ns,
        stereo_left_index=stereo_left_index,
        stereo_right_index=stereo_right_index,
        left_q_index=left_q_index,
        right_q_index=right_q_index,
        pair_q_index=pair_q_index,
        global_raw_cma_parameters=global_best.cpu().numpy(),
        global_correction=global_correction.cpu().numpy(),
        local_raw_cma_measurements=raw_measurements,
        local_measured_corrections=measured_corrections,
        local_filtered_corrections=filtered_corrections,
        global_parameter_bounds=global_bounds,
        local_parameter_bounds=local_bounds,
        translation_hard_bounded=np.asarray(
            not args.unbounded_translation
        ),
        global_translation_scale_m=np.asarray(
            args.global_translation_scale_mm * 1.0e-3,
            dtype=np.float32,
        ),
        local_translation_scale_m=np.asarray(
            args.local_translation_scale_mm * 1.0e-3,
            dtype=np.float32,
        ),
        initial_loss=initial_losses,
        measured_loss=measured_losses,
        filtered_loss=filtered_losses,
        tipnet_detections_xy_at_mask_resolution=tipnet,
        tipnet_stereo_gate_accepted=tip_accepted,
        all_joint_corrections_frozen=np.asarray(
            args.freeze_all_joint_corrections
        ),
    )

    preview_rows = [
        row
        for row in manifest_rows
        if int(row["strict_pair_slot"]) < count
    ]
    preview_panels: list[np.ndarray] = []
    for row in preview_rows:
        slot = int(row["strict_pair_slot"])
        preview_panels.append(
            render_preview_panel(
                renderer=renderer,
                row=row,
                q7=q7,
                T_left_lnd=T_left_lnd,
                left_q_index=left_q_index,
                right_q_index=right_q_index,
                raw_parameters=raw_from_correction(
                    filtered_corrections[slot],
                    local_bounds,
                    unbounded_translation=args.unbounded_translation,
                    translation_scale=renderer.translation_scale,
                ),
                targets_packed=packed_masks,
                mask_shape=mask_shape,
                visual_root=args.visual_root,
                label="online",
            )
        )
    preview_path = args.output_dir / "previews/anchor_online_overlay.png"
    if preview_panels and not cv2.imwrite(
        str(preview_path),
        np.concatenate(preview_panels, axis=0),
    ):
        raise RuntimeError(f"Failed to write {preview_path}")

    bounded_indices = (
        np.asarray([0, 1, 2, 6, 7, 8, 9])
        if args.unbounded_translation
        else np.arange(10)
    )
    global_fraction = np.abs(
        global_correction.cpu().numpy()[bounded_indices]
        / global_bounds[bounded_indices]
    )
    local_fraction = np.abs(
        filtered_corrections[:, bounded_indices]
        / local_bounds[None, bounded_indices]
    )
    report = {
        "schema": f"{schema_prefix}_sam2_online_optimization_v1",
        "passed": bool(
            count > 0
            and np.all(np.isfinite(filtered_corrections))
            and np.mean(measured_losses) <= np.mean(initial_losses)
            and np.max(global_fraction) < 0.995
            and np.max(local_fraction) < 0.995
            and (
                not args.freeze_jaw_correction
                or (
                    float(abs(global_correction[9].item())) == 0.0
                    and np.count_nonzero(filtered_corrections[:, 9]) == 0
                )
            )
            and (
                not args.freeze_all_joint_corrections
                or (
                    np.count_nonzero(
                        global_correction.cpu().numpy()[6:10]
                    )
                    == 0
                    and np.count_nonzero(filtered_corrections[:, 6:10]) == 0
                )
            )
        ),
        "upstream_fidelity": {
            "repository": (
                "https://github.com/hanyang-hu/online_dvrk_tracking"
            ),
            "commit": commit,
            "preserved": [
                "whole-instrument SurgicalSAM2 observations",
                "NvDiffRast hard silhouettes",
                "mse_weight=6",
                "appearance_area_weight=6e-6",
                "ContourTipNet with pts_weight=3e-3",
                "CMA-ES population 70 and 3 generations per frame",
                "previous-state initialization",
                "upstream Kalman filter",
            ],
            "stereo_extension": (
                "one shared residual is evaluated in both calibrated "
                "rectified eyes; right pose is frozen T_right_left times left"
            ),
            "raw_backbone_extension": (
                "raw q7 + hand-eye + LND supplies each absolute state; "
                "the optimizer estimates only bounded distal residuals"
            ),
            "instrument_geometry": (
                "exact four-part low-resolution paper LND CAD and "
                "diffcali/eval_dvrk/LND_fk.py component transforms"
                if args.geometry == "paper_lnd"
                else "registered P420006 seven-link carrier"
            ),
            "tip_safety_extension": (
                "non-finite ContourTipNet outputs are omitted per eye; finite "
                "outputs otherwise retain the upstream permutation-invariant "
                "tip loss without a kinematic-prior rejection gate"
            ),
            "jaw_state_constraint": (
                "all visual q4..q7 residuals are fixed to exactly zero; "
                "the complete raw seven-joint trajectory is preserved"
                if args.freeze_all_joint_corrections
                else "visual q7 jaw-angle residual is fixed to exactly zero; "
                "raw q7 preserves the measured closed middle interval"
                if args.freeze_jaw_correction
                else "visual q7 jaw-angle residual is optimized"
            ),
        },
        "inputs": {
            "masks": str(args.masks),
            "masks_sha256": sha256(args.masks),
            "sam2_report": str(args.sam2_report),
            "sam2_report_sha256": sha256(args.sam2_report),
            **(
                {
                    "anchor_state": str(args.anchor_state),
                    "anchor_state_sha256": sha256(args.anchor_state),
                }
                if args.observation_mode == "multianchor_parts"
                else {
                    "first_annotation": str(args.first_annotation),
                    "first_annotation_sha256": sha256(
                        args.first_annotation
                    ),
                }
            ),
            "kinematics": str(args.kinematics),
            "kinematics_sha256": sha256(args.kinematics),
            "geometry": args.geometry,
            "mesh_dir": str(args.mesh_dir),
            "mesh_sha256": {
                name: sha256(args.mesh_dir / name)
                for name in mesh_names
            },
            "tipnet_checkpoint": str(args.tipnet_checkpoint),
            "tipnet_checkpoint_sha256": sha256(
                args.tipnet_checkpoint
            ),
        },
        "sequence": {
            "pair_count": count,
            "complete_strict_pair_sequence": count
            == int(manifest["source_pair_count"]),
            "render_size_wh": [renderer.width, renderer.height],
        },
        "optimization": {
            "global_iterations": args.global_iterations,
            "global_population": args.global_population,
            "global_prior_weight": args.global_prior_weight,
            "online_iterations": args.online_iterations,
            "population": args.population,
            "local_prior_weight": args.local_prior_weight,
            "translation_hard_bounded": not args.unbounded_translation,
            "global_translation_scale_mm_not_a_limit": (
                args.global_translation_scale_mm
            ),
            "local_translation_scale_mm_not_a_limit": (
                args.local_translation_scale_mm
            ),
            "manual_anchor_tip_weight_multiplier": (
                args.manual_anchor_tip_weight_multiplier
            ),
            "observation_mode": args.observation_mode,
            "freeze_jaw_correction": args.freeze_jaw_correction,
            "freeze_all_joint_corrections": (
                args.freeze_all_joint_corrections
            ),
            "active_parameters": ["rx", "ry", "rz", "tx", "ty", "tz"],
            "active_parameter_count": (
                6 if args.freeze_all_joint_corrections else 9
                if args.freeze_jaw_correction else 10
            ),
            "cma_state_length": 10,
            "global_initial_loss": global_initial,
            "global_best_loss": global_best_loss,
            "mean_initial_loss": float(np.mean(initial_losses)),
            "mean_measured_loss": float(np.mean(measured_losses)),
            "mean_filtered_loss": float(np.mean(filtered_losses)),
            "global_correction": {
                "rotation_vector_deg": np.degrees(
                    global_correction.cpu().numpy()[:3]
                ).tolist(),
                "translation_mm": (
                    global_correction.cpu().numpy()[3:6] * 1000.0
                ).tolist(),
                "q4_q7_deg": np.degrees(
                    global_correction.cpu().numpy()[6:10]
                ).tolist(),
            },
            "maximum_global_boundary_fraction": float(
                np.max(global_fraction)
            ),
            "maximum_local_boundary_fraction": float(
                np.max(local_fraction)
            ),
        },
        "hard_constraints": {
            "all_q7_offsets_rad": (
                [0.0] * 7 if args.freeze_all_joint_corrections else None
            ),
            "global_q4_q7_residual_exact_zero": bool(
                np.count_nonzero(
                    global_correction.cpu().numpy()[6:10]
                )
                == 0
            ),
            "all_local_q4_q7_residuals_exact_zero": bool(
                np.count_nonzero(filtered_corrections[:, 6:10]) == 0
            ),
        },
        "tip_observations": {
            "manual_anchor_slots": sorted(manual_anchor_tips),
            "manual_policy": (
                "manual stereo jaw tips at every live-SAM2 anchor"
                if args.observation_mode == "multianchor_parts"
                else "manual stereo jaw tips at first pair"
            ),
            "remaining_pairs": "upstream ContourTipNet independently per eye",
            "both_views_valid_pair_count": int(np.count_nonzero(tip_accepted)),
            "both_views_valid_pair_fraction": float(np.mean(tip_accepted)),
            "diagnostics": tip_diagnostics,
        },
        "outputs": {
            "corrections": str(output_path),
            "anchor_preview": str(preview_path),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["optimization"], indent=2))
    print(
        f"TipNet accepted {int(np.count_nonzero(tip_accepted))}/{count}"
    )
    print(f"Report: {report_path}")
    print(f"Corrections: {output_path}")
    print(f"Preview: {preview_path}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
