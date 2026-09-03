#!/usr/bin/env python3
"""Bounded joint stereo visual refinement of raw P420006 keyframes.

This adapts the NvDiffRast + CMA-ES online calibration method from
online_dvrk_tracking commit cb2a264167aaf05b5a9c20da885d48568f78311f.
Unlike the upstream monocular script, every candidate is one shared physical
state.  The left and right observations use their actual raw q7 timestamps,
and the right camera pose is always derived from the frozen stereo transform.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from annotate_super_p420006_stereo_keyframes import rasterize_view


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_REPO = Path("/Media_HDD/jwshan/wad/online_dvrk_tracking")
RAW_ROOT = REPO_ROOT / "data/super/psm_raw_kinematics_v1（纯机器人学版本）"
P420_ROOT = RAW_ROOT / "gui_p420006_v1"
DEFAULT_ROOT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_p420006_stereo_v1"
)
UPSTREAM_COMMIT = "cb2a264167aaf05b5a9c20da885d48568f78311f"
LINK_MESHES = (
    "tool_main_link.stl",
    "tool_wrist_link.stl",
    "tool_wrist_shaft_link.stl",
    "tool_wrist_scal_link.stl",
    "tool_wrist_sca_shaft_link.stl",
    "tool_wrist_sca_ee_link_1.stl",
    "tool_wrist_sca_ee_link_2.stl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joint stereo P420006 refinement from manual annotations."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--kinematics", type=Path, default=RAW_ROOT / "kinematics.npz")
    parser.add_argument("--raw-model", type=Path, default=RAW_ROOT / "model.json")
    parser.add_argument(
        "--driver",
        type=Path,
        default=P420_ROOT / "psm_p420006_gui_pose_driver.npz",
    )
    parser.add_argument("--mesh-dir", type=Path, default=P420_ROOT / "meshes")
    parser.add_argument("--paper-repo", type=Path, default=PAPER_REPO)
    parser.add_argument("--render-scale", type=float, default=0.25)
    parser.add_argument("--global-iterations", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--population", type=int, default=70)
    parser.add_argument("--seed", type=int, default=420006)
    parser.add_argument("--max-rotation-deg", type=float, default=3.0)
    parser.add_argument("--max-translation-xy-mm", type=float, default=1.5)
    parser.add_argument("--max-translation-z-mm", type=float, default=0.5)
    parser.add_argument("--max-roll-deg", type=float, default=8.0)
    parser.add_argument("--max-wrist-deg", type=float, default=6.0)
    parser.add_argument("--max-jaw-deg", type=float, default=12.0)
    parser.add_argument("--global-max-rotation-deg", type=float, default=20.0)
    parser.add_argument(
        "--global-max-translation-xy-mm", type=float, default=8.0
    )
    parser.add_argument(
        "--global-max-translation-z-mm", type=float, default=4.0
    )
    parser.add_argument("--global-max-roll-deg", type=float, default=25.0)
    parser.add_argument("--global-max-wrist-deg", type=float, default=25.0)
    parser.add_argument("--global-max-jaw-deg", type=float, default=35.0)
    parser.add_argument("--global-prior-weight", type=float, default=0.08)
    parser.add_argument("--local-prior-weight", type=float, default=0.50)
    parser.add_argument("--local-temporal-weight", type=float, default=0.15)
    parser.add_argument(
        "--max-global-boundary-fraction",
        type=float,
        default=0.995,
        help="Reject a solution that is effectively clipped by a global bound.",
    )
    parser.add_argument(
        "--max-local-boundary-fraction",
        type=float,
        default=0.98,
        help="Reject a solution that is effectively clipped by a local bound.",
    )
    parser.add_argument(
        "--coarse-axis-search",
        action="store_true",
        help=(
            "Diagnostic only: test 24 proper CAD-LND axis rotations before "
            "bounded refinement. The strict raw registration remains the default."
        ),
    )
    parser.add_argument("--prior-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load binary STL and exactly deduplicate its triangle vertices."""
    with path.open("rb") as handle:
        header = handle.read(80)
        if len(header) != 80:
            raise RuntimeError(f"Truncated STL header: {path}")
        count_bytes = handle.read(4)
        if len(count_bytes) != 4:
            raise RuntimeError(f"Truncated STL triangle count: {path}")
        triangle_count = struct.unpack("<I", count_bytes)[0]
        records = np.fromfile(
            handle,
            dtype=np.dtype(
                [
                    ("normal", "<f4", (3,)),
                    ("vertices", "<f4", (3, 3)),
                    ("attribute", "<u2"),
                ]
            ),
            count=triangle_count,
        )
    if len(records) != triangle_count:
        raise RuntimeError(f"Truncated binary STL body: {path}")
    flat = np.asarray(records["vertices"], dtype=np.float32).reshape(-1, 3)
    vertices, inverse = np.unique(flat, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int32)
    return vertices.astype(np.float32), faces


def transform_matrix(
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    batch = rotation.shape[0]
    output = torch.zeros(
        (batch, 4, 4),
        dtype=rotation.dtype,
        device=rotation.device,
    )
    output[:, :3, :3] = rotation
    output[:, :3, 3] = translation
    output[:, 3, 3] = 1.0
    return output


def rot_x(angle: torch.Tensor) -> torch.Tensor:
    count = len(angle)
    output = torch.eye(4, device=angle.device, dtype=angle.dtype).repeat(
        count, 1, 1
    )
    cosine, sine = torch.cos(angle), torch.sin(angle)
    output[:, 1, 1] = cosine
    output[:, 1, 2] = -sine
    output[:, 2, 1] = sine
    output[:, 2, 2] = cosine
    return output


def rot_z(angle: torch.Tensor) -> torch.Tensor:
    count = len(angle)
    output = torch.eye(4, device=angle.device, dtype=angle.dtype).repeat(
        count, 1, 1
    )
    cosine, sine = torch.cos(angle), torch.sin(angle)
    output[:, 0, 0] = cosine
    output[:, 0, 1] = -sine
    output[:, 1, 0] = sine
    output[:, 1, 1] = cosine
    return output


def trans_x(distance: torch.Tensor) -> torch.Tensor:
    output = torch.eye(
        4, device=distance.device, dtype=distance.dtype
    ).repeat(len(distance), 1, 1)
    output[:, 0, 3] = distance
    return output


def trans_z(distance: torch.Tensor) -> torch.Tensor:
    output = torch.eye(
        4, device=distance.device, dtype=distance.dtype
    ).repeat(len(distance), 1, 1)
    output[:, 2, 3] = distance
    return output


def lnd_fk_batch(
    q7: torch.Tensor,
    dh_parameters: list[dict[str, Any]],
) -> torch.Tensor:
    batch = q7.shape[0]
    transforms = torch.empty(
        (batch, 9, 4, 4), device=q7.device, dtype=q7.dtype
    )
    current = torch.eye(4, device=q7.device, dtype=q7.dtype).repeat(
        batch, 1, 1
    )
    transforms[:, 0] = current
    zeros = torch.zeros(batch, device=q7.device, dtype=q7.dtype)
    for index, dh in enumerate(dh_parameters, start=1):
        alpha = zeros + float(dh.get("alpha", 0.0))
        a = zeros + float(dh.get("A", 0.0))
        theta0 = float(dh.get("theta", 0.0))
        d0 = float(dh.get("D", 0.0))
        offset = float(dh.get("offset", 0.0))
        if dh["type"] == "revolute":
            theta = q7[:, index - 1] + theta0 + offset
            distance = zeros + d0
        elif dh["type"] == "prismatic":
            theta = zeros + theta0
            distance = q7[:, index - 1] + d0 + offset
        else:
            raise RuntimeError(f"Unsupported LND joint type: {dh['type']}")
        modified_dh = (
            rot_x(alpha)
            @ trans_x(a)
            @ rot_z(theta)
            @ trans_z(distance)
        )
        current = current @ modified_dh
        transforms[:, index] = current
    transforms[:, 7] = current @ rot_z(0.5 * q7[:, 6])
    transforms[:, 8] = current @ rot_z(-0.5 * q7[:, 6])
    return transforms


def axis_angle_matrix(rotation_vector: torch.Tensor) -> torch.Tensor:
    x, y, z = rotation_vector.unbind(dim=1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        [
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ],
        dim=1,
    ).reshape(-1, 3, 3)
    return torch.matrix_exp(skew)


class P420StereoRenderer:
    def __init__(
        self,
        *,
        mesh_dir: Path,
        link_offsets: np.ndarray,
        lnd_link_ids: np.ndarray,
        dh_parameters: list[dict[str, Any]],
        K_left: np.ndarray,
        K_right: np.ndarray,
        T_right_left: np.ndarray,
        image_size: tuple[int, int],
        render_scale: float,
        bounds: np.ndarray,
        device: torch.device,
    ) -> None:
        import nvdiffrast.torch as dr

        self.dr = dr
        self.context = dr.RasterizeCudaContext(device=device)
        self.device = device
        self.dtype = torch.float32
        self.link_offsets = torch.as_tensor(
            link_offsets, device=device, dtype=self.dtype
        )
        self.lnd_link_ids = torch.as_tensor(
            lnd_link_ids, device=device, dtype=torch.long
        )
        self.dh_parameters = dh_parameters
        self.T_right_left = torch.as_tensor(
            T_right_left, device=device, dtype=self.dtype
        )
        self.width = int(round(image_size[0] * render_scale))
        self.height = int(round(image_size[1] * render_scale))
        if self.width < 64 or self.height < 64:
            raise ValueError("Render resolution is too small")
        self.K = {
            "left": torch.as_tensor(
                K_left * np.asarray(
                    [[render_scale, render_scale, render_scale],
                     [render_scale, render_scale, render_scale],
                     [1.0, 1.0, 1.0]],
                    dtype=np.float64,
                ),
                device=device,
                dtype=self.dtype,
            ),
            "right": torch.as_tensor(
                K_right * np.asarray(
                    [[render_scale, render_scale, render_scale],
                     [render_scale, render_scale, render_scale],
                     [1.0, 1.0, 1.0]],
                    dtype=np.float64,
                ),
                device=device,
                dtype=self.dtype,
            ),
        }
        self.bounds = torch.as_tensor(bounds, device=device, dtype=self.dtype)
        # Online visual correction normally uses tanh-bounded parameters.  A
        # caller may opt into an unbounded camera-frame XYZ parameterization;
        # ``translation_scale`` then controls numerical conditioning only and
        # is not a physical limit.
        self.translation_unbounded = False
        self.translation_scale = 1.0
        self.base_rotation = torch.eye(
            3, device=device, dtype=self.dtype
        )
        self.base_correction = torch.zeros(
            10, device=device, dtype=self.dtype
        )

        loaded = [load_binary_stl(mesh_dir / name) for name in LINK_MESHES]
        self.groups = {
            "body": self.make_group(loaded, range(5)),
            "jaw_a": self.make_group(loaded, (5,)),
            "jaw_b": self.make_group(loaded, (6,)),
        }
        self.tip_local = torch.tensor(
            [0.0, 0.0, 0.009002951],
            device=device,
            dtype=self.dtype,
        )

    def make_group(
        self,
        loaded: list[tuple[np.ndarray, np.ndarray]],
        link_indices: Any,
    ) -> dict[str, torch.Tensor]:
        vertices: list[np.ndarray] = []
        faces: list[np.ndarray] = []
        link_ids: list[np.ndarray] = []
        offset = 0
        for link_index in link_indices:
            local_vertices, local_faces = loaded[link_index]
            vertices.append(local_vertices)
            faces.append(local_faces + offset)
            link_ids.append(
                np.full(len(local_vertices), link_index, dtype=np.int64)
            )
            offset += len(local_vertices)
        return {
            "vertices": torch.as_tensor(
                np.concatenate(vertices),
                device=self.device,
                dtype=self.dtype,
            ),
            "faces": torch.as_tensor(
                np.concatenate(faces),
                device=self.device,
                dtype=torch.int32,
            ),
            "link_ids": torch.as_tensor(
                np.concatenate(link_ids),
                device=self.device,
                dtype=torch.long,
            ),
        }

    def decode(self, raw_parameters: torch.Tensor) -> torch.Tensor:
        decoded = torch.tanh(raw_parameters) * self.bounds
        if self.translation_unbounded:
            decoded = decoded.clone()
            decoded[:, 3:6] = (
                raw_parameters[:, 3:6] * self.translation_scale
            )
        return decoded

    def link_transforms(
        self,
        raw_parameters: torch.Tensor,
        raw_q7: torch.Tensor,
        T_left_base: torch.Tensor,
        side: str,
        base_rotation_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        correction = self.decode(raw_parameters)
        q7 = raw_q7.unsqueeze(0).expand(len(correction), -1).clone()
        q7[:, 3:7] += (
            self.base_correction[None, 6:10] + correction[:, 6:10]
        )
        fk = lnd_fk_batch(q7, self.dh_parameters)
        T_base = T_left_base.unsqueeze(0).expand(len(correction), -1, -1)
        links = (
            T_base[:, None]
            @ fk[:, self.lnd_link_ids]
            @ self.link_offsets[None]
        )

        small_rotation = axis_angle_matrix(correction[:, :3])
        base_correction_rotation = axis_angle_matrix(
            self.base_correction[None, :3]
        ).expand(len(correction), -1, -1)
        base_rotation = (
            self.base_rotation.unsqueeze(0).expand(len(correction), -1, -1)
            if base_rotation_override is None
            else base_rotation_override
        )
        delta_rotation = (
            small_rotation @ base_correction_rotation @ base_rotation
        )
        delta_translation = (
            self.base_correction[None, 3:6] + correction[:, 3:6]
        )
        # Apply the small rigid residual about the raw distal hinge rather than
        # rotating the tool about the distant robot base or camera origin.
        anchor = (T_base @ fk[:, 6])[:, :3, 3]
        link_rotation = links[:, :, :3, :3]
        link_translation = links[:, :, :3, 3]
        links[:, :, :3, :3] = (
            delta_rotation[:, None] @ link_rotation
        )
        links[:, :, :3, 3] = (
            torch.einsum(
                "bij,blj->bli",
                delta_rotation,
                link_translation - anchor[:, None],
            )
            + anchor[:, None]
            + delta_translation[:, None]
        )
        if side == "right":
            links = self.T_right_left[None, None] @ links
        return links

    def project_clip(
        self,
        camera_vertices: torch.Tensor,
        side: str,
    ) -> torch.Tensor:
        K = self.K[side]
        x, y, z = camera_vertices.unbind(dim=2)
        near, far = 0.01, 2.0
        x_clip = (
            2.0 * K[0, 0] / self.width * x
            + (2.0 * K[0, 2] / self.width - 1.0) * z
        )
        # NvDiffRast's CUDA raster output is bottom-up in clip-space y.  This
        # is the same sign convention used by CtRNet's projection matrix in
        # online_dvrk_tracking.  Using the ordinary OpenCV top-down sign here
        # mirrors the rendered mesh while leaving separately projected tips
        # unmirrored.
        y_clip = (
            2.0 * K[1, 1] / self.height * y
            + (2.0 * K[1, 2] / self.height - 1.0) * z
        )
        z_clip = (far + near) / (far - near) * z - (
            2.0 * far * near / (far - near)
        )
        return torch.stack([x_clip, y_clip, z_clip, z], dim=2)

    def render_group(
        self,
        links: torch.Tensor,
        side: str,
        group_name: str,
    ) -> torch.Tensor:
        group = self.groups[group_name]
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
        clip = self.project_clip(camera_vertices, side)
        raster, _ = self.dr.rasterize(
            self.context,
            clip,
            group["faces"],
            resolution=[self.height, self.width],
        )
        hard = (raster[..., 3:] > 0).to(self.dtype)
        antialiased = self.dr.antialias(
            hard,
            raster,
            clip,
            group["faces"],
        )
        return antialiased[..., 0].clamp(0.0, 1.0)

    def project_tips(
        self,
        links: torch.Tensor,
        side: str,
    ) -> torch.Tensor:
        jaw_links = links[:, 5:7]
        points = (
            torch.einsum(
                "bnij,j->bni",
                jaw_links[:, :, :3, :3],
                self.tip_local,
            )
            + jaw_links[:, :, :3, 3]
        )
        K = self.K[side]
        normalized = points / points[:, :, 2:3]
        return (normalized @ K.T)[:, :, :2]

    def render_state(
        self,
        raw_parameters: torch.Tensor,
        raw_q7: torch.Tensor,
        T_left_base: torch.Tensor,
        side: str,
        base_rotation_override: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        links = self.link_transforms(
            raw_parameters,
            raw_q7,
            T_left_base,
            side,
            base_rotation_override,
        )
        masks = {
            name: self.render_group(links, side, name)
            for name in ("body", "jaw_a", "jaw_b")
        }
        return masks, self.project_tips(links, side)

    def render_body_state(
        self,
        raw_parameters: torch.Tensor,
        raw_q7: torch.Tensor,
        T_left_base: torch.Tensor,
        side: str,
        base_rotation_override: torch.Tensor,
    ) -> torch.Tensor:
        links = self.link_transforms(
            raw_parameters,
            raw_q7,
            T_left_base,
            side,
            base_rotation_override,
        )
        return self.render_group(links, side, "body")


def mask_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    outside_distance: torch.Tensor,
) -> torch.Tensor:
    eps = 1.0e-6
    target_batch = target.unsqueeze(0)
    intersection = (predicted * target_batch).sum(dim=(1, 2))
    dice = 1.0 - (2.0 * intersection + eps) / (
        predicted.sum(dim=(1, 2)) + target.sum() + eps
    )
    area = torch.abs(predicted.sum(dim=(1, 2)) - target.sum()) / (
        target.sum() + eps
    )
    outside = (
        predicted * outside_distance.unsqueeze(0)
    ).sum(dim=(1, 2)) / (predicted.sum(dim=(1, 2)) + eps)
    return dice + 0.10 * area + 0.02 * outside


def target_distance(mask: np.ndarray) -> np.ndarray:
    binary = mask >= 0.5
    return cv2.distanceTransform(
        (~binary).astype(np.uint8),
        cv2.DIST_L2,
        3,
    ).astype(np.float32)


class StereoObjective:
    def __init__(
        self,
        *,
        renderer: P420StereoRenderer,
        raw_q7: dict[str, torch.Tensor],
        T_left_base: dict[str, torch.Tensor],
        targets: dict[str, dict[str, torch.Tensor]],
        distances: dict[str, dict[str, torch.Tensor]],
        tips: dict[str, dict[str, torch.Tensor | None]],
        confidence: dict[str, float],
        previous_normalized: torch.Tensor | None,
        prior_weight: float,
        temporal_weight: float,
    ) -> None:
        self.renderer = renderer
        self.raw_q7 = raw_q7
        self.T_left_base = T_left_base
        self.targets = targets
        self.distances = distances
        self.tips = tips
        self.confidence = confidence
        self.previous_normalized = previous_normalized
        self.prior_weight = float(prior_weight)
        self.temporal_weight = float(temporal_weight)

    @torch.no_grad()
    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        side_losses: dict[str, dict[str, torch.Tensor]] = {}
        predicted_tips: dict[str, torch.Tensor] = {}
        for side in ("left", "right"):
            predicted, tip_pixels = self.renderer.render_state(
                values,
                self.raw_q7[side],
                self.T_left_base[side],
                side,
            )
            predicted_tips[side] = tip_pixels
            side_losses[side] = {
                "body": mask_loss(
                    predicted["body"],
                    self.targets[side]["body"],
                    self.distances[side]["body"],
                ),
                "a_to_1": mask_loss(
                    predicted["jaw_a"],
                    self.targets[side]["jaw_1"],
                    self.distances[side]["jaw_1"],
                ),
                "a_to_2": mask_loss(
                    predicted["jaw_a"],
                    self.targets[side]["jaw_2"],
                    self.distances[side]["jaw_2"],
                ),
                "b_to_1": mask_loss(
                    predicted["jaw_b"],
                    self.targets[side]["jaw_1"],
                    self.distances[side]["jaw_1"],
                ),
                "b_to_2": mask_loss(
                    predicted["jaw_b"],
                    self.targets[side]["jaw_2"],
                    self.distances[side]["jaw_2"],
                ),
            }

        body = sum(side_losses[side]["body"] for side in ("left", "right"))
        direct_masks = sum(
            side_losses[side]["a_to_1"] + side_losses[side]["b_to_2"]
            for side in ("left", "right")
        )
        swapped_masks = sum(
            side_losses[side]["a_to_2"] + side_losses[side]["b_to_1"]
            for side in ("left", "right")
        )
        direct_tips = torch.zeros_like(body)
        swapped_tips = torch.zeros_like(body)
        for side in ("left", "right"):
            for predicted_index, target_name in ((0, "jaw_1"), (1, "jaw_2")):
                target_tip = self.tips[side][target_name]
                if target_tip is not None:
                    direct_tips = direct_tips + torch.linalg.vector_norm(
                        predicted_tips[side][:, predicted_index] - target_tip,
                        dim=1,
                    ) / 20.0
            for predicted_index, target_name in ((0, "jaw_2"), (1, "jaw_1")):
                target_tip = self.tips[side][target_name]
                if target_tip is not None:
                    swapped_tips = swapped_tips + torch.linalg.vector_norm(
                        predicted_tips[side][:, predicted_index] - target_tip,
                        dim=1,
                    ) / 20.0

        direct = (
            0.75 * self.confidence["jaw_masks"] * direct_masks
            + 0.30 * self.confidence["jaw_tips"] * direct_tips
        )
        swapped = (
            0.75 * self.confidence["jaw_masks"] * swapped_masks
            + 0.30 * self.confidence["jaw_tips"] * swapped_tips
        )

        normalized = torch.tanh(values)
        prior = torch.mean(normalized**2, dim=1)
        temporal = torch.zeros_like(prior)
        if self.previous_normalized is not None:
            temporal = torch.mean(
                (normalized - self.previous_normalized.unsqueeze(0)) ** 2,
                dim=1,
            )
        return (
            self.confidence["body"] * body
            + torch.minimum(direct, swapped)
            + self.prior_weight * prior
            + self.temporal_weight * temporal
        )


def load_targets(
    root: Path,
    keyframe: int,
    render_size: tuple[int, int],
    scale: float,
    device: torch.device,
) -> tuple[
    dict[str, dict[str, torch.Tensor]],
    dict[str, dict[str, torch.Tensor]],
    dict[str, dict[str, torch.Tensor | None]],
]:
    path = root / "annotations" / f"keyframe_{keyframe:02d}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}; finish stereo manual annotation and validation first"
        )
    annotation = json.loads(path.read_text(encoding="utf-8"))
    width, height = annotation["image_size_wh"]
    render_height, render_width = render_size
    targets: dict[str, dict[str, torch.Tensor]] = {}
    distances: dict[str, dict[str, torch.Tensor]] = {}
    tips: dict[str, dict[str, torch.Tensor | None]] = {}
    for side in ("left", "right"):
        full = rasterize_view(annotation["views"][side], width, height)
        labels = cv2.resize(
            full,
            (render_width, render_height),
            interpolation=cv2.INTER_NEAREST,
        )
        targets[side] = {}
        distances[side] = {}
        for class_id, name in ((1, "body"), (2, "jaw_1"), (3, "jaw_2")):
            mask = (labels == class_id).astype(np.float32)
            targets[side][name] = torch.as_tensor(mask, device=device)
            distances[side][name] = torch.as_tensor(
                target_distance(mask),
                device=device,
            )
        tips[side] = {}
        for name in ("jaw_1", "jaw_2"):
            tip = annotation["views"][side]["tips"][name]
            tips[side][name] = (
                torch.as_tensor(
                    np.asarray(tip["point_xy"], dtype=np.float32) * scale,
                    device=device,
                )
                if tip["visible"]
                else None
            )
    return targets, distances, tips


def load_annotation_confidence(
    root: Path,
    keyframe: int,
) -> dict[str, float]:
    path = root / "annotation_confidence.json"
    if not path.is_file():
        return {"body": 1.0, "jaw_masks": 1.0, "jaw_tips": 1.0}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "super_p420006_stereo_annotation_confidence_v1":
        raise RuntimeError("Unexpected annotation confidence schema")
    output = {
        name: float(value)
        for name, value in payload["default"].items()
    }
    output.update(
        {
            name: float(value)
            for name, value in payload.get("keyframes", {})
            .get(str(keyframe), {})
            .items()
            if name in output
        }
    )
    if set(output) != {"body", "jaw_masks", "jaw_tips"}:
        raise RuntimeError(f"Incomplete annotation confidence for KF {keyframe}")
    if any(not 0.0 <= value <= 1.0 for value in output.values()):
        raise ValueError(f"Invalid annotation confidence for KF {keyframe}")
    return output


def make_bounds(args: argparse.Namespace) -> np.ndarray:
    return np.asarray(
        [
            *([math.radians(args.max_rotation_deg)] * 3),
            args.max_translation_xy_mm * 1.0e-3,
            args.max_translation_xy_mm * 1.0e-3,
            args.max_translation_z_mm * 1.0e-3,
            math.radians(args.max_roll_deg),
            math.radians(args.max_wrist_deg),
            math.radians(args.max_wrist_deg),
            math.radians(args.max_jaw_deg),
        ],
        dtype=np.float32,
    )


def make_global_bounds(args: argparse.Namespace) -> np.ndarray:
    return np.asarray(
        [
            *([math.radians(args.global_max_rotation_deg)] * 3),
            args.global_max_translation_xy_mm * 1.0e-3,
            args.global_max_translation_xy_mm * 1.0e-3,
            args.global_max_translation_z_mm * 1.0e-3,
            math.radians(args.global_max_roll_deg),
            math.radians(args.global_max_wrist_deg),
            math.radians(args.global_max_wrist_deg),
            math.radians(args.global_max_jaw_deg),
        ],
        dtype=np.float32,
    )


def proper_axis_rotations() -> np.ndarray:
    """Return all 24 right-handed signed axis-permutation rotations."""
    rotations: list[np.ndarray] = []
    identity = np.eye(3, dtype=np.float32)
    for permutation in itertools.permutations(range(3)):
        permuted = identity[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            candidate = permuted @ np.diag(signs).astype(np.float32)
            if np.linalg.det(candidate) > 0.5:
                rotations.append(candidate)
    unique: list[np.ndarray] = []
    for candidate in rotations:
        if not any(np.array_equal(candidate, item) for item in unique):
            unique.append(candidate)
    if len(unique) != 24:
        raise RuntimeError(f"Expected 24 proper axis rotations, got {len(unique)}")
    # Put identity first so equal-score behavior is deterministic.
    unique.sort(key=lambda value: 0 if np.array_equal(value, identity) else 1)
    return np.stack(unique)


@torch.no_grad()
def select_global_axis_rotation(
    *,
    args: argparse.Namespace,
    renderer: P420StereoRenderer,
    rows: list[dict[str, Any]],
    q7: np.ndarray,
    T_left_lnd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    candidates_np = proper_axis_rotations()
    candidates = torch.as_tensor(
        candidates_np,
        device=renderer.device,
        dtype=torch.float32,
    )
    scores = torch.zeros(
        len(candidates),
        device=renderer.device,
        dtype=torch.float32,
    )
    zeros = torch.zeros(
        (len(candidates), 10),
        device=renderer.device,
        dtype=torch.float32,
    )
    for row in rows:
        keyframe = int(row["keyframe_index"])
        targets, distances, _tips = load_targets(
            args.root,
            keyframe,
            (renderer.height, renderer.width),
            args.render_scale,
            renderer.device,
        )
        for side in ("left", "right"):
            q_index = int(row[f"{side}_q_index"])
            predicted = renderer.render_body_state(
                zeros,
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
                candidates,
            )
            scores += mask_loss(
                predicted,
                targets[side]["body"],
                distances[side]["body"],
            )
    best_index = int(torch.argmin(scores).item())
    return candidates_np[best_index], scores.cpu().numpy()


def overlay_masks(
    image: np.ndarray,
    masks: dict[str, np.ndarray],
    tips: np.ndarray,
) -> np.ndarray:
    result = image.copy()
    colors = {
        "body": (40, 220, 40),
        "jaw_a": (30, 80, 255),
        "jaw_b": (255, 160, 20),
    }
    for name, color in colors.items():
        full = cv2.resize(
            masks[name],
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        tint = np.zeros_like(result)
        tint[:] = color
        alpha = np.clip(full[..., None] * 0.45, 0.0, 0.45)
        result = np.asarray(
            result * (1.0 - alpha) + tint * alpha,
            dtype=np.uint8,
        )
    scale_x = image.shape[1] / masks["body"].shape[1]
    scale_y = image.shape[0] / masks["body"].shape[0]
    for index, point in enumerate(tips):
        pixel = (
            int(round(point[0] * scale_x)),
            int(round(point[1] * scale_y)),
        )
        cv2.drawMarker(
            result,
            pixel,
            colors["jaw_a" if index == 0 else "jaw_b"],
            cv2.MARKER_CROSS,
            28,
            4,
            cv2.LINE_AA,
        )
    return result


@torch.no_grad()
def render_preview(
    *,
    root: Path,
    rows: list[dict[str, Any]],
    renderer: P420StereoRenderer,
    q7: np.ndarray,
    T_left_lnd: np.ndarray,
    raw_parameters: np.ndarray,
    name: str,
) -> Path:
    panels: list[np.ndarray] = []
    panel_width = 800
    for local_index, row in enumerate(rows):
        side_panels: list[np.ndarray] = []
        parameters = torch.as_tensor(
            raw_parameters[local_index : local_index + 1],
            device=renderer.device,
            dtype=torch.float32,
        )
        for side in ("left", "right"):
            q_index = int(row[f"{side}_q_index"])
            masks, tips = renderer.render_state(
                parameters,
                torch.as_tensor(q7[q_index], device=renderer.device, dtype=torch.float32),
                torch.as_tensor(
                    T_left_lnd[q_index, 0],
                    device=renderer.device,
                    dtype=torch.float32,
                ),
                side,
            )
            masks_np = {
                key: value[0].cpu().numpy() for key, value in masks.items()
            }
            image = cv2.imread(str(root / row[f"{side}_image"]))
            overlay = overlay_masks(
                image,
                masks_np,
                tips[0].cpu().numpy(),
            )
            scale = panel_width / (2.0 * overlay.shape[1])
            side_panels.append(
                cv2.resize(
                    overlay,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
            )
        panel = np.concatenate(side_panels, axis=1)
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(
            panel,
            f"KF {local_index:02d}  {name}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)
    sheet = np.concatenate(panels, axis=0)
    path = root / "previews" / f"{name}_overlay.png"
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"Failed to write {path}")
    return path


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("NvDiffRast stereo optimization requires CUDA")
    if not 0.05 <= args.render_scale <= 1.0:
        raise ValueError("--render-scale must be in [0.05, 1]")
    for path in (
        args.root / "pair_manifest.json",
        args.root / "keyframes.npz",
        args.kinematics,
        args.raw_model,
        args.driver,
        args.paper_repo,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    for name in LINK_MESHES:
        if not (args.mesh_dir / name).is_file():
            raise FileNotFoundError(args.mesh_dir / name)

    sys.path.insert(0, str(args.paper_repo))
    from diffcali.utils.cma_es import CMAES_cus
    from evotorch import Problem
    from torch.quasirandom import SobolEngine

    manifest = json.loads(
        (args.root / "pair_manifest.json").read_text(encoding="utf-8")
    )
    rows = manifest["keyframes"]
    raw_model = json.loads(args.raw_model.read_text(encoding="utf-8"))
    with np.load(args.root / "keyframes.npz", allow_pickle=False) as keyframes:
        K_left = keyframes["K_left_rect"].astype(np.float32)
        K_right = keyframes["K_right_rect"].astype(np.float32)
        T_right_left = keyframes[
            "T_rectified_right_camera_rectified_left_camera"
        ].astype(np.float32)
    with np.load(args.kinematics, allow_pickle=False) as raw:
        q7 = raw["q7"].astype(np.float32)
        T_left_lnd = raw[
            "T_rectified_left_camera_lnd_link"
        ].astype(np.float32)
    with np.load(args.driver, allow_pickle=False) as driver:
        link_offsets = driver["T_lndlink_urdf_link"].astype(np.float32)
        lnd_link_ids = driver["lnd_link_ids"].astype(np.int64)

    device = torch.device("cuda")
    bounds = make_bounds(args)
    renderer = P420StereoRenderer(
        mesh_dir=args.mesh_dir,
        link_offsets=link_offsets,
        lnd_link_ids=lnd_link_ids,
        dh_parameters=raw_model["lnd"]["DH_params"],
        K_left=K_left,
        K_right=K_right,
        T_right_left=T_right_left,
        image_size=tuple(manifest["image_size_wh"]),
        render_scale=args.render_scale,
        bounds=bounds,
        device=device,
    )
    zero_parameters = np.zeros((len(rows), 10), dtype=np.float32)
    prior_preview = render_preview(
        root=args.root,
        rows=rows,
        renderer=renderer,
        q7=q7,
        T_left_lnd=T_left_lnd,
        raw_parameters=zero_parameters,
        name="raw_p420006_prior",
    )
    print(f"Raw prior preview: {prior_preview}")
    if args.prior_only:
        return

    validation_path = args.root / "previews/annotation_validation.json"
    if not validation_path.is_file():
        raise FileNotFoundError(
            f"{validation_path}; run annotation validation first"
        )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("passed") is not True:
        raise RuntimeError("Manual stereo annotations have not passed validation")

    if args.coarse_axis_search:
        print("Selecting one global CAD-LND axis rotation from 24 proper candidates")
        registration_rotation, axis_scores = select_global_axis_rotation(
            args=args,
            renderer=renderer,
            rows=rows,
            q7=q7,
            T_left_lnd=T_left_lnd,
        )
    else:
        registration_rotation = np.eye(3, dtype=np.float32)
        axis_scores = np.asarray([np.nan], dtype=np.float32)
    renderer.base_rotation = torch.as_tensor(
        registration_rotation,
        device=device,
        dtype=torch.float32,
    )
    axis_preview = render_preview(
        root=args.root,
        rows=rows,
        renderer=renderer,
        q7=q7,
        T_left_lnd=T_left_lnd,
        raw_parameters=zero_parameters,
        name="axis_registered_prior",
    )
    print(f"Axis-registered prior preview: {axis_preview}")

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

    observations: list[dict[str, Any]] = []
    for row in rows:
        keyframe = int(row["keyframe_index"])
        targets, distances, tips = load_targets(
            args.root,
            keyframe,
            (renderer.height, renderer.width),
            args.render_scale,
            device,
        )
        observations.append(
            {
                "targets": targets,
                "distances": distances,
                "tips": tips,
                "confidence": load_annotation_confidence(
                    args.root, keyframe
                ),
                "raw_q7": {
                    side: torch.as_tensor(
                        q7[int(row[f"{side}_q_index"])],
                        device=device,
                        dtype=torch.float32,
                    )
                    for side in ("left", "right")
                },
                "T_left_base": {
                    side: torch.as_tensor(
                        T_left_lnd[int(row[f"{side}_q_index"]), 0],
                        device=device,
                        dtype=torch.float32,
                    )
                    for side in ("left", "right")
                },
            }
        )

    def objective_for(
        local_index: int,
        previous: torch.Tensor | None,
        *,
        prior_weight: float,
        temporal_weight: float,
    ) -> StereoObjective:
        observation = observations[local_index]
        return StereoObjective(
            renderer=renderer,
            raw_q7=observation["raw_q7"],
            T_left_base=observation["T_left_base"],
            targets=observation["targets"],
            distances=observation["distances"],
            tips=observation["tips"],
            confidence=observation["confidence"],
            previous_normalized=previous,
            prior_weight=prior_weight,
            temporal_weight=temporal_weight,
        )

    global_bounds = make_global_bounds(args)
    renderer.bounds = torch.as_tensor(
        global_bounds, device=device, dtype=torch.float32
    )
    global_objectives = [
        objective_for(
            local_index,
            None,
            prior_weight=args.global_prior_weight,
            temporal_weight=0.0,
        )
        for local_index in range(len(rows))
    ]
    zero_batch = torch.zeros((1, 10), device=device, dtype=torch.float32)
    raw_loss = np.asarray(
        [
            float(objective(zero_batch)[0].item())
            for objective in global_objectives
        ],
        dtype=np.float32,
    )

    class GlobalObjective:
        @torch.no_grad()
        def __call__(self, values: torch.Tensor) -> torch.Tensor:
            return torch.stack(
                [objective(values) for objective in global_objectives],
                dim=0,
            ).mean(dim=0)

    global_problem = ProblemAdapter(GlobalObjective())
    global_center = torch.zeros(10, device=device, dtype=torch.float32)
    global_searcher = CMAES_cus(
        global_problem,
        center_init=global_center,
        stdev_init=0.35,
        popsize=args.population,
        mu_size=min(15, args.population // 2),
        sobol=SobolEngine(10, scramble=True, seed=args.seed - 1),
    )
    global_best = global_center.clone()
    global_best_loss = float(np.mean(raw_loss))
    for iteration in range(args.global_iterations):
        global_searcher.step()
        candidate_loss = float(global_searcher.status["pop_best_eval"])
        if candidate_loss < global_best_loss:
            global_best_loss = candidate_loss
            global_best = (
                global_searcher.status["pop_best"].values.detach().clone()
            )
        print(
            f"GLOBAL iter {iteration + 1:02d}/{args.global_iterations}: "
            f"{candidate_loss:.5f}"
        )
    global_correction = (
        torch.tanh(global_best)
        * torch.as_tensor(global_bounds, device=device)
    )
    renderer.base_correction = global_correction.detach()
    global_loss = np.asarray(
        [
            float(objective(zero_batch)[0].item())
            for objective in global_objectives
        ],
        dtype=np.float32,
    )
    global_preview = render_preview(
        root=args.root,
        rows=rows,
        renderer=renderer,
        q7=q7,
        T_left_lnd=T_left_lnd,
        raw_parameters=zero_parameters,
        name="global_registered_prior",
    )
    print(
        f"Global registration: {raw_loss.mean():.5f} -> "
        f"{global_loss.mean():.5f}; preview={global_preview}"
    )

    # Per-keyframe residuals are deliberately much smaller than the shared
    # global calibration residual.
    renderer.bounds = torch.as_tensor(
        bounds, device=device, dtype=torch.float32
    )
    best_raw = np.zeros((len(rows), 10), dtype=np.float32)
    best_loss = np.empty(len(rows), dtype=np.float32)
    initial_loss = np.empty(len(rows), dtype=np.float32)
    previous_normalized: torch.Tensor | None = None
    center = torch.zeros(10, device=device, dtype=torch.float32)
    for local_index, row in enumerate(rows):
        keyframe = int(row["keyframe_index"])
        objective = objective_for(
            local_index,
            previous_normalized,
            prior_weight=args.local_prior_weight,
            temporal_weight=args.local_temporal_weight,
        )
        initial_loss[local_index] = float(
            objective(torch.zeros((1, 10), device=device))[0].item()
        )
        problem = ProblemAdapter(objective)
        searcher = CMAES_cus(
            problem,
            center_init=center,
            stdev_init=0.35,
            popsize=args.population,
            mu_size=min(15, args.population // 2),
            sobol=SobolEngine(10, scramble=True, seed=args.seed + keyframe),
        )
        frame_best_loss = initial_loss[local_index]
        frame_best = torch.zeros_like(center)
        for iteration in range(args.iterations):
            searcher.step()
            candidate_loss = float(searcher.status["pop_best_eval"])
            if candidate_loss < frame_best_loss:
                frame_best_loss = candidate_loss
                frame_best = (
                    searcher.status["pop_best"].values.detach().clone()
                )
            print(
                f"KF {keyframe:02d} iter {iteration + 1:02d}/"
                f"{args.iterations}: {candidate_loss:.5f}"
            )
        best_raw[local_index] = frame_best.cpu().numpy()
        best_loss[local_index] = frame_best_loss
        previous_normalized = torch.tanh(frame_best).detach()
        center = frame_best.detach()
        print(
            f"KF {keyframe:02d}: "
            f"{initial_loss[local_index]:.5f} -> {best_loss[local_index]:.5f}"
        )

    corrections = np.tanh(best_raw) * bounds[None]
    global_correction_np = global_correction.cpu().numpy()
    global_boundary_fraction = np.abs(
        global_correction_np / global_bounds
    )
    local_boundary_fraction = np.abs(corrections / bounds[None])
    output_path = args.root / "keyframe_corrections.npz"
    np.savez_compressed(
        output_path,
        schema=np.asarray("super_p420006_stereo_keyframe_corrections_v2"),
        pair_timestamp_ros_ns=np.asarray(
            [row["pair_timestamp_ros_ns"] for row in rows],
            dtype=np.int64,
        ),
        strict_pair_slot=np.asarray(
            [row["strict_pair_slot"] for row in rows],
            dtype=np.int64,
        ),
        global_raw_cma_parameters=global_best.cpu().numpy(),
        global_correction_rotvec_camera_rad=global_correction_np[:3],
        global_correction_translation_camera_m=global_correction_np[3:6],
        global_correction_q3_q6_rad=global_correction_np[6:10],
        local_raw_cma_parameters=best_raw,
        local_correction_rotvec_camera_rad=corrections[:, :3],
        local_correction_translation_camera_m=corrections[:, 3:6],
        local_correction_q3_q6_rad=corrections[:, 6:10],
        global_registration_rotation_camera=registration_rotation,
        global_axis_candidate_scores=axis_scores,
        global_parameter_bounds=global_bounds,
        local_parameter_bounds=bounds,
        raw_loss=raw_loss,
        global_registered_loss=global_loss,
        local_initial_objective=initial_loss,
        optimized_loss=best_loss,
    )
    optimized_preview = render_preview(
        root=args.root,
        rows=rows,
        renderer=renderer,
        q7=q7,
        T_left_lnd=T_left_lnd,
        raw_parameters=best_raw,
        name="stereo_optimized",
    )
    global_boundary_ok = bool(
        np.max(global_boundary_fraction)
        <= args.max_global_boundary_fraction
    )
    local_boundary_ok = bool(
        np.max(local_boundary_fraction)
        <= args.max_local_boundary_fraction
    )
    passed = bool(
        np.mean(global_loss) < np.mean(raw_loss)
        and np.all(best_loss <= initial_loss + 1.0e-6)
        and np.all(best_loss < raw_loss)
        and global_boundary_ok
        and local_boundary_ok
    )
    report = {
        "schema": "super_p420006_stereo_keyframe_optimization_v2",
        "passed": passed,
        "method": (
            "hierarchical online_dvrk_tracking NvDiffRast + CMA-ES: one "
            "shared global calibration residual, then bounded temporal "
            "keyframe residuals, each evaluated by two calibrated cameras"
        ),
        "upstream": {
            "repository": str(args.paper_repo.resolve()),
            "commit": UPSTREAM_COMMIT,
        },
        "stereo_state_rule": (
            "left/right use actual timestamp-matched raw q7; right pose is "
            "T_rectRight_rectLeft times the corrected left-camera state"
        ),
        "global_bounds": {
            "rotation_deg": args.global_max_rotation_deg,
            "translation_xy_mm": args.global_max_translation_xy_mm,
            "translation_z_mm": args.global_max_translation_z_mm,
            "roll_deg": args.global_max_roll_deg,
            "wrist_deg": args.global_max_wrist_deg,
            "jaw_deg": args.global_max_jaw_deg,
        },
        "local_bounds": {
            "rotation_deg": args.max_rotation_deg,
            "translation_xy_mm": args.max_translation_xy_mm,
            "translation_z_mm": args.max_translation_z_mm,
            "roll_deg": args.max_roll_deg,
            "wrist_deg": args.max_wrist_deg,
            "jaw_deg": args.max_jaw_deg,
        },
        "regularization": {
            "global_prior_weight": args.global_prior_weight,
            "local_prior_weight": args.local_prior_weight,
            "local_temporal_weight": args.local_temporal_weight,
        },
        "acceptance_gates": {
            "maximum_global_boundary_fraction": (
                args.max_global_boundary_fraction
            ),
            "maximum_local_boundary_fraction": (
                args.max_local_boundary_fraction
            ),
            "global_boundary_passed": global_boundary_ok,
            "local_boundary_passed": local_boundary_ok,
        },
        "coarse_axis_registration": {
            "enabled": args.coarse_axis_search,
            "candidate_count": 24 if args.coarse_axis_search else 1,
            "selected_rotation_camera": registration_rotation.tolist(),
            "candidate_body_scores": axis_scores.tolist(),
            "preview": str(axis_preview.resolve()),
        },
        "render_size_wh": [renderer.width, renderer.height],
        "annotation_confidence": {
            str(row["keyframe_index"]): load_annotation_confidence(
                args.root,
                int(row["keyframe_index"]),
            )
            for row in rows
        },
        "global_iterations": args.global_iterations,
        "local_iterations": args.iterations,
        "population": args.population,
        "raw_loss": raw_loss.tolist(),
        "global_registered_loss": global_loss.tolist(),
        "local_initial_objective": initial_loss.tolist(),
        "optimized_loss": best_loss.tolist(),
        "raw_to_global_improvement": (raw_loss - global_loss).tolist(),
        "global_to_local_improvement": (initial_loss - best_loss).tolist(),
        "raw_to_optimized_improvement": (raw_loss - best_loss).tolist(),
        "global_correction": {
            "rotation_vector_deg": np.degrees(
                global_correction_np[:3]
            ).tolist(),
            "translation_mm": (
                global_correction_np[3:6] * 1.0e3
            ).tolist(),
            "q3_q6_deg": np.degrees(
                global_correction_np[6:10]
            ).tolist(),
            "maximum_boundary_fraction": float(
                np.max(global_boundary_fraction)
            ),
        },
        "local_corrections": {
            "maximum_boundary_fraction": float(
                np.max(local_boundary_fraction)
            ),
            "per_parameter_maximum_boundary_fraction": np.max(
                local_boundary_fraction, axis=0
            ).tolist(),
        },
        "outputs": {
            "corrections": {
                "path": str(output_path.resolve()),
                "sha256": sha256(output_path),
            },
            "global_preview": str(global_preview.resolve()),
            "preview": str(optimized_preview.resolve()),
        },
    }
    report_path = args.root / "optimization_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(
            "Hierarchical stereo correction failed its loss improvement gate"
        )


if __name__ == "__main__":
    main()
