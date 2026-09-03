"""Paper-LND renderer with the upstream four-mesh kinematic convention."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from optimize_super_p420006_stereo_keyframes import (
    P420StereoRenderer,
    axis_angle_matrix,
    lnd_fk_batch,
)
from paper_lnd_mesh_io import load_binary_ply_triangles


PAPER_LND_MESHES = (
    "low_res_shaft_multi_cylinder.ply",
    "low_res_logo_low_res_1.ply",
    "low_res_jawright_lowres.ply",
    "low_res_jawleft_lowres.ply",
)


def load_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    return load_binary_ply_triangles(path)


def _constant_transform(
    rows: list[list[float]],
    *,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.tensor(
        rows,
        device=device,
        dtype=dtype,
    ).unsqueeze(0).expand(batch, -1, -1)


def paper_component_transforms(
    joints: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return seven GUI-contract transforms and the shared jaw frame.

    Visible mesh slots are 0=shaft, 4=logo, 5=right jaw and 6=left jaw.
    Slots 1..3 retain useful paper frames but carry no geometry.  The four
    visible transforms reproduce ``diffcali/eval_dvrk/LND_fk.py`` exactly.
    """

    batch = len(joints)
    device, dtype = joints.device, joints.dtype
    theta0, theta1, theta2, theta3 = joints.unbind(dim=1)
    zeros = torch.zeros_like(theta0)
    ones = torch.ones_like(theta0)

    def wrist_transform(
        theta: torch.Tensor,
        translation_x: float,
    ) -> torch.Tensor:
        output = torch.stack(
            [
                torch.stack(
                    [
                        torch.sin(theta),
                        torch.cos(theta),
                        zeros,
                        zeros + translation_x,
                    ],
                    dim=1,
                ),
                torch.stack([zeros, zeros, ones, zeros], dim=1),
                torch.stack(
                    [torch.cos(theta), -torch.sin(theta), zeros, zeros],
                    dim=1,
                ),
                torch.stack([zeros, zeros, zeros, ones], dim=1),
            ],
            dim=1,
        )
        return output

    def jaw_rotation(theta: torch.Tensor) -> torch.Tensor:
        output = torch.eye(
            4,
            device=device,
            dtype=dtype,
        ).unsqueeze(0).repeat(batch, 1, 1)
        output[:, 0, 0] = torch.cos(theta)
        output[:, 0, 1] = -torch.sin(theta)
        output[:, 1, 0] = torch.sin(theta)
        output[:, 1, 1] = torch.cos(theta)
        return output

    T45 = wrist_transform(theta0, 0.0)
    T56 = wrist_transform(theta1, 0.0091)
    T4_mesh = _constant_transform(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        batch=batch,
        device=device,
        dtype=dtype,
    )
    T5_mesh = _constant_transform(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        batch=batch,
        device=device,
        dtype=dtype,
    )
    T7_mesh = _constant_transform(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        batch=batch,
        device=device,
        dtype=dtype,
    )
    identity = torch.eye(
        4,
        device=device,
        dtype=dtype,
    ).unsqueeze(0).expand(batch, -1, -1)
    T6 = T45 @ T56
    components = torch.stack(
        [
            T4_mesh,
            identity,
            T45,
            T6,
            T45 @ T5_mesh,
            T6 @ jaw_rotation(-theta2) @ T7_mesh,
            T6 @ jaw_rotation(theta3) @ T7_mesh,
        ],
        dim=1,
    )
    return components, T6


class PaperLNDStereoRenderer(P420StereoRenderer):
    """NvDiffRast stereo renderer for the paper repository's real LND CAD."""

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
        del link_offsets, lnd_link_ids
        import nvdiffrast.torch as dr

        self.dr = dr
        self.context = dr.RasterizeCudaContext(device=device)
        self.device = device
        self.dtype = torch.float32
        self.dh_parameters = dh_parameters
        self.T_right_left = torch.as_tensor(
            T_right_left,
            device=device,
            dtype=self.dtype,
        )
        self.width = int(round(image_size[0] * render_scale))
        self.height = int(round(image_size[1] * render_scale))
        if self.width < 64 or self.height < 64:
            raise ValueError("Render resolution is too small")
        intrinsic_scale = np.asarray(
            [
                [render_scale, render_scale, render_scale],
                [render_scale, render_scale, render_scale],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.K = {
            "left": torch.as_tensor(
                K_left * intrinsic_scale,
                device=device,
                dtype=self.dtype,
            ),
            "right": torch.as_tensor(
                K_right * intrinsic_scale,
                device=device,
                dtype=self.dtype,
            ),
        }
        self.bounds = torch.as_tensor(bounds, device=device, dtype=self.dtype)
        self.base_rotation = torch.eye(3, device=device, dtype=self.dtype)
        self.base_correction = torch.zeros(
            10,
            device=device,
            dtype=self.dtype,
        )

        loaded = [load_ply(mesh_dir / name) for name in PAPER_LND_MESHES]
        self.groups = {
            "body": self.make_paper_group(loaded, ((0, 0), (1, 4))),
            "jaw_a": self.make_paper_group(loaded, ((2, 5),)),
            "jaw_b": self.make_paper_group(loaded, ((3, 6),)),
        }
        self.tip_local = torch.tensor(
            [
                [0.0, 0.0004, 0.0096],
                [0.0, -0.0004, 0.0096],
            ],
            device=device,
            dtype=self.dtype,
        )

    def make_paper_group(
        self,
        loaded: list[tuple[np.ndarray, np.ndarray]],
        mesh_and_link_indices: tuple[tuple[int, int], ...],
    ) -> dict[str, torch.Tensor]:
        vertices: list[np.ndarray] = []
        faces: list[np.ndarray] = []
        link_ids: list[np.ndarray] = []
        offset = 0
        for mesh_index, link_index in mesh_and_link_indices:
            local_vertices, local_faces = loaded[mesh_index]
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
        paper_joints = torch.stack(
            [q7[:, 4], q7[:, 5], 0.5 * q7[:, 6], 0.5 * q7[:, 6]],
            dim=1,
        )
        components, T_frame6 = paper_component_transforms(paper_joints)
        T_base = T_left_base.unsqueeze(0).expand(len(correction), -1, -1)
        T_camera_frame4 = T_base @ fk[:, 4]
        links = T_camera_frame4[:, None] @ components

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
        anchor = (T_camera_frame4 @ T_frame6)[:, :3, 3]
        link_rotation = links[:, :, :3, :3]
        link_translation = links[:, :, :3, 3]
        links[:, :, :3, :3] = delta_rotation[:, None] @ link_rotation
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

    def project_tips(
        self,
        links: torch.Tensor,
        side: str,
    ) -> torch.Tensor:
        jaw_links = links[:, 5:7]
        points = (
            torch.einsum(
                "bnij,nj->bni",
                jaw_links[:, :, :3, :3],
                self.tip_local,
            )
            + jaw_links[:, :, :3, 3]
        )
        K = self.K[side]
        normalized = points / points[:, :, 2:3]
        return (normalized @ K.T)[:, :, :2]
