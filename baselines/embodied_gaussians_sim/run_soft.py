#!/usr/bin/env python3
"""Run the reconstructed deformable Embodied Gaussians algorithm on SIM.

The method path is intentionally isolated from FixedSuperBest's soft-body and
tracking modules.  It implements only oriented particle integration, sphere
collision, local-neighbour shape matching, nearest-particle Gaussian bonds,
and the paper's RGB-to-visual-force correction.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import warp as wp
from gsplat.rendering import rasterization
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


BASELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BASELINE_ROOT.parents[1]
sys.path.insert(0, str(BASELINE_ROOT))
from protocol import (  # noqa: E402
    CAMERAS,
    PAPER_PARAMETERS,
    UPSTREAM_COMMIT,
    dataset_spec,
    observation_allowed,
    read_json,
    rendering_frames,
    resolve_dataset,
    sha256_file,
)
from shape_matching import (  # noqa: E402
    OrientedShapeMatcher,
    build_particle_neighbour_clusters,
)
from embodied_gaussians.scene_builders.domain import Body  # noqa: E402


@wp.kernel
def _self_collision_kernel(
    grid: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    radius: float,
    relaxation: float,
    deltas: wp.array(dtype=wp.vec3),
):
    ordered_id = wp.tid()
    particle_id = wp.hash_grid_point_id(grid, ordered_id)
    if particle_id < 0:
        return
    point = positions[particle_id]
    query = wp.hash_grid_query(grid, point, 2.0 * radius)
    other = int(0)
    delta = wp.vec3(0.0)
    while wp.hash_grid_query_next(query, other):
        if other != particle_id:
            difference = point - positions[other]
            distance = wp.length(difference)
            penetration = 2.0 * radius - distance
            if penetration > 0.0 and distance > 1.0e-12:
                # Equal masses: each particle receives half of Eq. (3).
                delta += relaxation * 0.5 * penetration * difference / distance
    deltas[particle_id] = delta


@wp.kernel
def _robot_collision_kernel(
    robot_grid: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    robot_positions: wp.array(dtype=wp.vec3),
    tissue_radius: float,
    robot_radius: float,
    relaxation: float,
    deltas: wp.array(dtype=wp.vec3),
):
    particle_id = wp.tid()
    point = positions[particle_id]
    query = wp.hash_grid_query(robot_grid, point, tissue_radius + robot_radius)
    robot_id = int(0)
    delta = wp.vec3(0.0)
    count = float(0.0)
    while wp.hash_grid_query_next(query, robot_id):
        difference = point - robot_positions[robot_id]
        distance = wp.length(difference)
        penetration = tissue_radius + robot_radius - distance
        if penetration > 0.0 and distance > 1.0e-12:
            # The robot particle is kinematic (inverse mass zero), so the
            # deformable particle receives the full paper collision update.
            delta += relaxation * penetration * difference / distance
            count += 1.0
    if count > 0.0:
        deltas[particle_id] = delta / count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=("sim01", "sim02", "sim03"), required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--actuation",
        choices=("shared-prescribed-boundary", "psm-fk-collision-only"),
        default="psm-fk-collision-only",
    )
    parser.add_argument("--online-width", type=int, default=int(PAPER_PARAMETERS["online_width"]))
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=-1,
        help="工程诊断；正式运行必须为 -1。",
    )
    return parser.parse_args()


def _normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion / torch.linalg.norm(quaternion, dim=-1, keepdim=True).clamp_min(1.0e-12)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to scalar-first quaternions."""
    shape = matrix.shape[:-2]
    flat = matrix.reshape(-1, 3, 3)
    result = torch.empty((len(flat), 4), dtype=matrix.dtype, device=matrix.device)
    trace = flat[:, 0, 0] + flat[:, 1, 1] + flat[:, 2, 2]
    positive = trace > 0.0
    if bool(positive.any()):
        values = flat[positive]
        scale = torch.sqrt(trace[positive] + 1.0) * 2.0
        result[positive, 0] = 0.25 * scale
        result[positive, 1] = (values[:, 2, 1] - values[:, 1, 2]) / scale
        result[positive, 2] = (values[:, 0, 2] - values[:, 2, 0]) / scale
        result[positive, 3] = (values[:, 1, 0] - values[:, 0, 1]) / scale
    remaining = ~positive
    for diagonal in range(3):
        first = diagonal
        second = (diagonal + 1) % 3
        third = (diagonal + 2) % 3
        selected = remaining & (flat[:, first, first] >= flat[:, second, second]) & (
            flat[:, first, first] >= flat[:, third, third]
        )
        if not bool(selected.any()):
            continue
        values = flat[selected]
        scale = torch.sqrt(
            1.0
            + values[:, first, first]
            - values[:, second, second]
            - values[:, third, third]
        ).clamp_min(1.0e-12) * 2.0
        result[selected, first + 1] = 0.25 * scale
        result[selected, 0] = (values[:, third, second] - values[:, second, third]) / scale
        result[selected, second + 1] = (values[:, second, first] + values[:, first, second]) / scale
        result[selected, third + 1] = (values[:, third, first] + values[:, first, third]) / scale
        remaining &= ~selected
    if bool(remaining.any()):
        raise RuntimeError("rotation matrix 到 quaternion 转换失败")
    return _normalize_quaternion(result).reshape(*shape, 4)


def rotation_matrix_from_rotation_vector(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Rodrigues map for world-frame angular increments."""
    angle = torch.linalg.norm(rotation_vector, dim=-1, keepdim=True)
    axis = rotation_vector / angle.clamp_min(1.0e-12)
    x, y, z = axis.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(*rotation_vector.shape[:-1], 3, 3)
    identity = torch.eye(
        3, dtype=rotation_vector.dtype, device=rotation_vector.device
    ).expand_as(skew)
    sine = torch.sin(angle)[..., None]
    cosine = torch.cos(angle)[..., None]
    result = identity + sine * skew + (1.0 - cosine) * (skew @ skew)
    return torch.where((angle < 1.0e-8)[..., None], identity, result)


def rotation_vector_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Shortest axis-angle vector used by Algorithm 2's omega update."""
    quaternion = matrix_to_quaternion(matrix)
    quaternion = torch.where(
        (quaternion[..., :1] < 0.0), -quaternion, quaternion
    )
    vector = quaternion[..., 1:]
    sine_half = torch.linalg.norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sine_half, quaternion[..., :1].clamp_min(0.0))
    return torch.where(
        sine_half > 1.0e-8,
        vector * angle / sine_half.clamp_min(1.0e-12),
        2.0 * vector,
    )


def quaternion_to_matrix_numpy(quaternions_wxyz: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions_wxyz, dtype=np.float64)
    return Rotation.from_quat(quaternions, scalar_first=True).as_matrix().astype(np.float32)


class TissueGaussianBody:
    def __init__(self, body_path: Path, device: torch.device):
        self.path = body_path.expanduser().resolve()
        body = Body.model_validate_json(self.path.read_text(encoding="utf-8"))
        if body.particles is None or body.gaussians is None:
            raise ValueError("EG body 必须同时包含 particles 与 gaussians")
        transform = np.asarray(body.X_WB, dtype=np.float64)
        rotation_wb = transform[:3, :3]
        translation_wb = transform[:3, 3]
        particle_local = np.asarray(body.particles.means, dtype=np.float64)
        self.particle_radii_np = np.asarray(body.particles.radii, dtype=np.float32)
        gaussian_local = np.asarray(body.gaussians.means, dtype=np.float64)
        particle_world = (rotation_wb @ particle_local.T).T + translation_wb
        gaussian_world = (rotation_wb @ gaussian_local.T).T + translation_wb
        particle_rotations = rotation_wb[None] @ quaternion_to_matrix_numpy(
            np.asarray(body.particles.quats, dtype=np.float64)
        )
        gaussian_rotations = rotation_wb[None] @ quaternion_to_matrix_numpy(
            np.asarray(body.gaussians.quats, dtype=np.float64)
        )
        _, parents = cKDTree(particle_world).query(gaussian_world, k=1, workers=-1)
        offsets_world = gaussian_world - particle_world[parents]
        local_offsets = np.einsum(
            "gji,gj->gi", particle_rotations[parents], offsets_world
        )
        local_rotations = np.einsum(
            "gji,gjk->gik", particle_rotations[parents], gaussian_rotations
        )

        self.rest_positions_np = particle_world.astype(np.float32)
        self.positions = torch.as_tensor(
            self.rest_positions_np, device=device
        ).contiguous()
        self.velocities = torch.zeros_like(self.positions)
        self.rotations = torch.as_tensor(
            particle_rotations, dtype=torch.float32, device=device
        ).contiguous()
        self.angular_velocities = torch.zeros_like(self.positions)
        self.parents = torch.as_tensor(parents, dtype=torch.long, device=device)
        self.local_offsets = torch.as_tensor(local_offsets, dtype=torch.float32, device=device)
        self.local_rotations = torch.as_tensor(local_rotations, dtype=torch.float32, device=device)
        self.scales = torch.as_tensor(body.gaussians.scales, dtype=torch.float32, device=device)
        colors = torch.as_tensor(body.gaussians.colors, dtype=torch.float32, device=device).clamp(1.0e-5, 1.0 - 1.0e-5)
        opacities = torch.as_tensor(body.gaussians.opacities, dtype=torch.float32, device=device).clamp(1.0e-5, 1.0 - 1.0e-5)
        self.color_logits = torch.logit(colors)
        self.opacity_logits = torch.logit(opacities)
        self.external_force = torch.zeros_like(self.positions)

    def gaussian_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        parent_rotations = self.rotations[self.parents]
        means = self.positions[self.parents] + torch.einsum(
            "gij,gj->gi", parent_rotations, self.local_offsets
        )
        rotation_matrices = parent_rotations @ self.local_rotations
        quaternions = matrix_to_quaternion(rotation_matrices)
        return means, quaternions, self.color_logits.sigmoid(), self.opacity_logits.sigmoid()


class StereoInputs:
    def __init__(self, dataset: Path, online_width: int):
        cameras = read_json(dataset / "cameras.json")
        self.dataset = dataset
        self.online_width = int(online_width)
        self.calibration = {}
        self.packed_masks = {}
        self.mask_width = {}
        for name in CAMERAS:
            metadata = read_json(dataset / "videos" / f"{name}.json")
            width, height = (int(value) for value in metadata["resolution"])
            intrinsic = np.asarray(metadata["K"], dtype=np.float32)
            world_from_camera = np.asarray(cameras[name]["X_WC_ros_optical"], dtype=np.float32)
            self.calibration[name] = {
                "K": intrinsic,
                "X_WC": world_from_camera,
                "X_CW": np.linalg.inv(world_from_camera).astype(np.float32),
                "resolution": (width, height),
                "timestamps": np.asarray(metadata["timestamps"], dtype=np.float64),
            }
            mask_root = dataset / "gui_assets" / "visual_force_masks" / name
            report = read_json(mask_root / "report.json")
            self.mask_width[name] = int(report["resolution_wh"][0])
            self.packed_masks[name] = np.load(
                mask_root / "tissue_masks_packbits.npy", mmap_mode="r"
            )

    def load_online(self, camera: str, frame: int, device: torch.device):
        image = np.asarray(
            Image.open(self.dataset / "rgb" / camera / f"{frame:06d}.png").convert("RGB"),
            dtype=np.float32,
        ) / 255.0
        packed = np.asarray(self.packed_masks[camera][frame])
        mask = np.unpackbits(packed, axis=1, bitorder="big")[:, : self.mask_width[camera]] > 0
        source_height, source_width = image.shape[:2]
        target_height = int(round(source_height * self.online_width / source_width))
        size = (self.online_width, target_height)
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        image = image * mask[..., None].astype(np.float32)
        intrinsic = self.calibration[camera]["K"].copy()
        intrinsic[0, :] *= self.online_width / source_width
        intrinsic[1, :] *= target_height / source_height
        intrinsic[2, :] = (0.0, 0.0, 1.0)
        return (
            torch.as_tensor(np.ascontiguousarray(image), device=device),
            torch.as_tensor(intrinsic, device=device).unsqueeze(0),
            torch.as_tensor(self.calibration[camera]["X_CW"], device=device).unsqueeze(0),
            size,
        )

    def full_camera_tensors(self, camera: str, device: torch.device):
        calibration = self.calibration[camera]
        return (
            torch.as_tensor(calibration["K"], device=device).unsqueeze(0),
            torch.as_tensor(calibration["X_CW"], device=device).unsqueeze(0),
            calibration["resolution"],
        )


class PrescribedBoundary:
    def __init__(self, dataset: Path, rest_positions: np.ndarray, device: torch.device):
        self.path = dataset / "task_inputs" / "known_grasp_region_boundary.npz"
        with np.load(self.path, allow_pickle=False) as archive:
            self.positions = torch.as_tensor(
                archive["trajectory_positions_world"], dtype=torch.float32, device=device
            )
            self.velocities = torch.as_tensor(
                archive["trajectory_velocities_world"], dtype=torch.float32, device=device
            )
            self.grasped = np.asarray(archive["grasped"], dtype=bool)
        first = self.positions[0].cpu().numpy()
        _, particle_ids = cKDTree(rest_positions).query(first, k=1, workers=-1)
        if len(np.unique(particle_ids)) != len(particle_ids):
            raise ValueError("EG 粒子过稀，无法一一映射共享夹持边界")
        self.particle_ids = torch.as_tensor(particle_ids, dtype=torch.long, device=device)

    def apply(self, frame: int, body: TissueGaussianBody) -> None:
        if self.grasped[frame]:
            body.positions[self.particle_ids] = self.positions[frame]
            body.velocities[self.particle_ids] = self.velocities[frame]


class PSMCollisionParticles:
    def __init__(self, dataset: Path, device: torch.device):
        pose_path = dataset / "task_inputs" / "psm_link_poses.npz"
        mesh_path = dataset / "gui_assets" / "official_psm_tip_meshes_v2.npz"
        with np.load(pose_path, allow_pickle=False) as poses:
            self.pose_link_names = [str(value) for value in poses["link_names"].tolist()]
            self.world_from_links = np.asarray(poses["X_WL"], dtype=np.float32)
        samples = []
        sample_links = []
        with np.load(mesh_path, allow_pickle=False) as meshes:
            for link_name in meshes["link_names"].tolist():
                link_name = str(link_name)
                vertices = np.asarray(meshes[f"{link_name}__vertices"], dtype=np.float32)
                # Use the recorded official mesh vertices directly.  A voxel
                # spacing would be another unpublished adapter parameter.
                selected = np.unique(vertices, axis=0)
                samples.append(selected)
                sample_links.append(
                    np.full(len(selected), self.pose_link_names.index(link_name), dtype=np.int64)
                )
        self.local = torch.as_tensor(np.concatenate(samples), device=device)
        self.link_ids = torch.as_tensor(np.concatenate(sample_links), dtype=torch.long, device=device)
        self.device = device

    def positions_at(self, frame: int) -> torch.Tensor:
        transforms = torch.as_tensor(self.world_from_links[frame], device=self.device)
        selected = transforms[self.link_ids]
        return torch.einsum("nij,nj->ni", selected[:, :3, :3], self.local) + selected[:, :3, 3]


class PaperSoftPhysics:
    def __init__(
        self,
        body: TissueGaussianBody,
        matcher: OrientedShapeMatcher,
        *,
        shape_stiffness: float,
        particle_radius: float,
        device: torch.device,
    ):
        self.body = body
        self.matcher = matcher
        self.shape_stiffness = float(shape_stiffness)
        self.radius = float(particle_radius)
        self.device = device
        self.self_grid = wp.HashGrid(128, 128, 128, device=str(device))
        self.robot_grid = wp.HashGrid(128, 128, 128, device=str(device))
        self.deltas = torch.zeros_like(body.positions)

    def _collide_self(self) -> None:
        positions_wp = wp.from_torch(self.body.positions, dtype=wp.vec3)
        self.self_grid.build(positions_wp, 2.0 * self.radius)
        self.deltas.zero_()
        wp.launch(
            _self_collision_kernel,
            dim=len(self.body.positions),
            inputs=[self.self_grid.id, positions_wp, self.radius, 1.0, wp.from_torch(self.deltas, dtype=wp.vec3)],
            device=str(self.device),
        )
        self.body.positions.add_(self.deltas)

    def _collide_robot(self, robot_positions: torch.Tensor | None) -> None:
        if robot_positions is None or len(robot_positions) == 0:
            return
        robot_wp = wp.from_torch(robot_positions.contiguous(), dtype=wp.vec3)
        self.robot_grid.build(robot_wp, 2.0 * self.radius)
        self.deltas.zero_()
        wp.launch(
            _robot_collision_kernel,
            dim=len(self.body.positions),
            inputs=[
                self.robot_grid.id,
                wp.from_torch(self.body.positions, dtype=wp.vec3),
                robot_wp,
                self.radius,
                self.radius,
                1.0,
                wp.from_torch(self.deltas, dtype=wp.vec3),
            ],
            device=str(self.device),
        )
        self.body.positions.add_(self.deltas)

    @torch.no_grad()
    def step(
        self,
        frame: int,
        boundary: PrescribedBoundary | None,
        robot_positions: torch.Tensor | None,
    ) -> None:
        frame_dt = float(PAPER_PARAMETERS["dt_s"])
        substeps = int(PAPER_PARAMETERS["substeps"])
        substep_dt = frame_dt / substeps
        mass = float(PAPER_PARAMETERS["particle_mass_kg"])
        gravity = torch.tensor(
            (0.0, 0.0, float(PAPER_PARAMETERS["gravity_m_s2"])),
            dtype=self.body.positions.dtype,
            device=self.device,
        )
        for _ in range(substeps):
            old_positions = self.body.positions.clone()
            old_rotations = self.body.rotations.clone()
            self.body.velocities.add_(
                substep_dt * (self.body.external_force / mass + gravity)
            )
            self.body.positions.add_(substep_dt * self.body.velocities)
            rotation_increment = rotation_matrix_from_rotation_vector(
                substep_dt * self.body.angular_velocities
            )
            self.body.rotations.copy_(rotation_increment @ self.body.rotations)
            for _ in range(int(PAPER_PARAMETERS["jacobi_iterations"])):
                self._collide_self()
                self._collide_robot(robot_positions)
                projected_positions, projected_rotations = self.matcher.project(
                    self.body.positions,
                    self.body.rotations,
                    self.shape_stiffness,
                )
                self.body.positions.copy_(projected_positions)
                self.body.rotations.copy_(projected_rotations)
                # Paper Algorithm 2 resolves the ground constraint in every
                # solver iteration.  Particle spheres may touch but not cross
                # the public builder's z=0 ground plane.
                self.body.positions[:, 2].clamp_(min=self.radius)
                if boundary is not None:
                    boundary.apply(frame, self.body)
            self.body.velocities.copy_((self.body.positions - old_positions) / substep_dt)
            rotation_delta = self.body.rotations @ old_rotations.transpose(-1, -2)
            self.body.angular_velocities.copy_(
                rotation_vector_from_matrix(rotation_delta) / substep_dt
            )
            if boundary is not None:
                boundary.apply(frame, self.body)
        self.body.velocities.mul_(float(PAPER_PARAMETERS["velocity_damping_per_frame"]))
        self.body.angular_velocities.mul_(
            float(PAPER_PARAMETERS["velocity_damping_per_frame"])
        )
        self.body.external_force.zero_()


def render_tissue(
    body: TissueGaussianBody,
    view_matrix: torch.Tensor,
    intrinsic: torch.Tensor,
    size_wh: tuple[int, int],
):
    means, quaternions, colors, opacities = body.gaussian_state()
    width, height = size_wh
    rendered, alpha, _ = rasterization(
        means=means,
        quats=quaternions,
        scales=body.scales,
        colors=colors,
        opacities=opacities,
        viewmats=view_matrix,
        Ks=intrinsic,
        width=width,
        height=height,
        camera_model="pinhole",
        render_mode="RGB",
        backgrounds=torch.zeros((1, 3), dtype=torch.float32, device=means.device),
        near_plane=0.01,
        far_plane=2.0,
        packed=False,
    )
    return rendered, alpha


def compute_visual_force(
    body: TissueGaussianBody,
    stereo: StereoInputs,
    frame: int,
    rng: random.Random,
    device: torch.device,
) -> dict[str, float]:
    means, quaternions, _, old_opacities = body.gaussian_state()
    desired_means = torch.nn.Parameter(means.detach().clone())
    desired_quaternions = torch.nn.Parameter(quaternions.detach().clone())
    colors = torch.nn.Parameter(body.color_logits.detach().clone())
    opacities = torch.nn.Parameter(body.opacity_logits.detach().clone())
    optimizer = torch.optim.Adam(
        [
            {"params": [desired_means], "lr": float(PAPER_PARAMETERS["visual_lr_position"])},
            {"params": [desired_quaternions], "lr": float(PAPER_PARAMETERS["visual_lr_rotation"])},
            {"params": [colors], "lr": float(PAPER_PARAMETERS["visual_lr_color"])},
            {"params": [opacities], "lr": float(PAPER_PARAMETERS["visual_lr_opacity"])},
        ]
    )
    camera_inputs = {
        camera: stereo.load_online(camera, frame, device) for camera in CAMERAS
    }
    losses = []
    for _ in range(int(PAPER_PARAMETERS["visual_iterations"])):
        camera = CAMERAS[rng.randrange(len(CAMERAS))]
        target, intrinsic, view_matrix, size = camera_inputs[camera]
        rendered, _, _ = rasterization(
            means=desired_means,
            quats=_normalize_quaternion(desired_quaternions),
            scales=body.scales,
            colors=colors.sigmoid(),
            opacities=opacities.sigmoid(),
            viewmats=view_matrix,
            Ks=intrinsic,
            width=size[0],
            height=size[1],
            camera_model="pinhole",
            render_mode="RGB",
            backgrounds=torch.zeros((1, 3), dtype=torch.float32, device=device),
            near_plane=0.01,
            far_plane=2.0,
            packed=False,
        )
        # L_rgb is the absolute photometric loss defined by paper Eq. (8).
        loss = torch.nn.functional.l1_loss(rendered[0], target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    with torch.no_grad():
        displacement = desired_means - means
        displacement_norm = torch.linalg.norm(displacement, dim=1)
        active = displacement_norm >= float(PAPER_PARAMETERS["visual_displacement_deadzone_m"])
        gaussian_force = (
            float(PAPER_PARAMETERS["visual_kp"])
            * old_opacities[:, None]
            * displacement
            * active[:, None]
        )
        body.external_force.zero_()
        body.external_force.index_add_(0, body.parents, gaussian_force)
        body.color_logits.copy_(colors)
        body.opacity_logits.copy_(opacities)
    return {
        "loss_mean": float(np.mean(losses)),
        "active_gaussians": int(active.sum().item()),
        "maximum_displacement_mm": 1000.0 * float(displacement_norm.max().item()),
        "maximum_particle_force_n": float(torch.linalg.norm(body.external_force, dim=1).max().item()),
    }


def save_render(path: Path, tensor: torch.Tensor, *, alpha: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = tensor.detach().clamp(0.0, 1.0)
    if alpha:
        array = value.mul(255.0).add(0.5).byte().cpu().numpy()
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        Image.fromarray(array, mode="L").save(path)
    else:
        array = value.mul(255.0).add(0.5).byte().cpu().numpy()
        Image.fromarray(array, mode="RGB").save(path)


def main() -> None:
    args = parse_args()
    if args.seed < 0:
        raise ValueError("seed 参数非法")
    if args.frame_limit < 0 and args.online_width != int(PAPER_PARAMETERS["online_width"]):
        raise ValueError("正式 rollout 不允许覆盖论文在线分辨率")
    dataset = resolve_dataset(REPO_ROOT, args.dataset_key, args.dataset)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有 EG rollout：{output}")
    output.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("正式 EG visual-force rollout 需要 CUDA")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)
    wp.init()
    body_path = args.body.expanduser().resolve()
    body = TissueGaussianBody(body_path, device)
    expected_radius = float(PAPER_PARAMETERS["particle_radius_m"])
    if not np.allclose(body.particle_radii_np, expected_radius, rtol=0.0, atol=1.0e-7):
        raise ValueError("body 粒子半径不是公开代码默认/论文范围内的固定 6 mm")
    clusters = build_particle_neighbour_clusters(
        body.rest_positions_np,
        particle_mass=float(PAPER_PARAMETERS["particle_mass_kg"]),
    )
    matcher = OrientedShapeMatcher(clusters, device)
    physics = PaperSoftPhysics(
        body,
        matcher,
        shape_stiffness=float(PAPER_PARAMETERS["shape_constraint_projection"]),
        particle_radius=float(PAPER_PARAMETERS["particle_radius_m"]),
        device=device,
    )
    stereo = StereoInputs(dataset, args.online_width)
    episode = read_json(dataset / "episode.json")
    formal_frame_count = int(dataset_spec(args.dataset_key)["frames"])
    if int(episode["frames"]) != formal_frame_count:
        raise ValueError("dataset frame count 与固定协议不一致")
    frame_count = formal_frame_count if args.frame_limit < 0 else min(args.frame_limit, formal_frame_count)
    boundary = None
    robot = None
    if args.actuation == "shared-prescribed-boundary":
        boundary = PrescribedBoundary(dataset, body.rest_positions_np, device)
    else:
        robot = PSMCollisionParticles(dataset, device)

    particle_positions = np.empty((frame_count, len(body.positions), 3), dtype=np.float32)
    particle_rotations = np.empty((frame_count, len(body.positions), 3, 3), dtype=np.float32)
    diagnostics = []
    render_set = set(rendering_frames(formal_frame_count))
    start = time.perf_counter()
    for frame in range(frame_count):
        frame_start = time.perf_counter()
        robot_positions = robot.positions_at(frame) if robot is not None else None
        physics.step(frame, boundary, robot_positions)
        particle_positions[frame] = body.positions.detach().cpu().numpy()
        particle_rotations[frame] = body.rotations.detach().cpu().numpy()

        frame_diagnostic: dict[str, object] = {
            "frame": frame,
            "observation_allowed": observation_allowed(frame, formal_frame_count),
        }
        if frame in render_set and not args.no_render:
            for camera in CAMERAS:
                intrinsic, view_matrix, size = stereo.full_camera_tensors(camera, device)
                rendered, alpha = render_tissue(body, view_matrix, intrinsic, size)
                save_render(output / "rgb" / camera / f"{frame:06d}.png", rendered[0])
                save_render(output / "alpha" / camera / f"{frame:06d}.png", alpha[0], alpha=True)
        if observation_allowed(frame, formal_frame_count):
            frame_diagnostic.update(compute_visual_force(body, stereo, frame, rng, device))
        else:
            body.external_force.zero_()
        frame_diagnostic["elapsed_s"] = time.perf_counter() - frame_start
        diagnostics.append(frame_diagnostic)
        print(
            f"[EG-Soft] {args.dataset_key} frame {frame + 1:04d}/{frame_count:04d} "
            f"observe={frame_diagnostic['observation_allowed']} "
            f"elapsed={frame_diagnostic['elapsed_s']:.3f}s",
            flush=True,
        )

    raw_path = output / "raw_particle_rollout.npz"
    timestamps = np.asarray(stereo.calibration[CAMERAS[0]]["timestamps"][:frame_count])
    np.savez_compressed(
        raw_path,
        frame_indices=np.arange(frame_count, dtype=np.int32),
        timestamps=timestamps,
        particle_positions_world=particle_positions,
        particle_rotations_world=particle_rotations,
    )
    (output / "frame_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "schema": "fixedsuperbest.embodied_gaussians_soft_rollout.v1",
        "formal": args.frame_limit < 0 and not args.no_render,
        "method": "EG-Soft paper equations reconstructed on pinned public reference initializer",
        "dataset_key": args.dataset_key,
        "dataset": str(dataset),
        "seed": args.seed,
        "upstream_commit": UPSTREAM_COMMIT,
        "body": str(body_path),
        "body_sha256": sha256_file(body_path),
        "actuation": args.actuation,
        "actuation_disclosure": (
            "protocol-level prescribed grasp boundary; derived from the SIM generator and shared with A/B/C"
            if boundary is not None
            else "known PSM link poses plus paper sphere collision only; no prescribed tissue trajectory"
        ),
        "particles": len(body.positions),
        "gaussians": len(body.parents),
        "shape_cluster_construction": (
            "one particle plus its parameter-free Delaunay-adjacent particles"
        ),
        "shape_constraint_projection": float(
            PAPER_PARAMETERS["shape_constraint_projection"]
        ),
        "shape_parameter_disclosure": (
            "paper omits a numerical deformable k_S and neighbourhood rule; no value is fitted: "
            "Eq. (5) is fully projected (k_S=1), with Delaunay geometric adjacency"
        ),
        "parameters": PAPER_PARAMETERS,
        "gravity": [0.0, 0.0, float(PAPER_PARAMETERS["gravity_m_s2"])],
        "ground_constraint": "paper sphere-ground constraint at public z=0 plane",
        "forbidden_method_components": [
            "tetrahedral topology",
            "distance constraint",
            "volume constraint",
            "triangle/tetra barycentric Gaussian binding",
            "CoTracker/AllTracker",
            "FoundationStereo online correction",
            "online stiffness/damping identification",
            "GT material parameters",
        ],
        "online_rgb_adapter": (
            f"two available stereo cameras; {args.online_width}-pixel-wide tissue layer; "
            "packed tissue masks are the same "
            "GT-mask protocol adapter supplied to all external baselines"
        ),
        "evaluation_truth_opened": False,
        "frame_count": frame_count,
        "total_elapsed_s": time.perf_counter() - start,
        "raw_rollout_sha256": sha256_file(raw_path),
    }
    (output / "rollout_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
