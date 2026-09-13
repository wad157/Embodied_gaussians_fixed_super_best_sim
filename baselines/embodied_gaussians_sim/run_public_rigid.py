#!/usr/bin/env python3
"""Run the pinned public rigid-only EG implementation on SIM.

This audit variant uses upstream ``EmbodiedGaussiansEnvironment``, rigid Warp
dynamics, upstream RGB MSE visual forces, and the public default visual-force
parameters.  It is not presented as the paper's deformable method.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp
import warp.sim
from scipy.spatial.transform import Rotation


BASELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BASELINE_ROOT.parents[1]
sys.path.insert(0, str(BASELINE_ROOT))
from protocol import (  # noqa: E402
    CAMERAS,
    DATASETS,
    PAPER_PARAMETERS,
    UPSTREAM_COMMIT,
    observation_allowed,
    read_json,
    rendering_frames,
    resolve_dataset,
    sha256_file,
)
from data_adapter import StereoInputs, save_render  # noqa: E402

from embodied_gaussians.embodied_simulator.builder import EmbodiedGaussiansBuilder  # noqa: E402
from embodied_gaussians.embodied_simulator.frames import FramesBuilder  # noqa: E402
from embodied_gaussians.environments.embodied_environment import EmbodiedGaussiansEnvironment  # noqa: E402
from embodied_gaussians.scene_builders.domain import Body  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--online-width", type=int, default=int(PAPER_PARAMETERS["online_width"]))
    parser.add_argument("--frame-limit", type=int, default=-1)
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def warp_transform(matrix: np.ndarray):
    value = np.asarray(matrix, dtype=np.float64)
    quaternion = wp.quat_from_matrix(value[:3, :3])
    return wp.transformf(*value[:3, 3], *quaternion)


class RigidSimBuilder(EmbodiedGaussiansBuilder):
    def add_dataset_tissue(self, body: Body) -> int:
        if body.particles is None or body.gaussians is None:
            raise ValueError("rigid-only EG body 缺少 particles 或 gaussians")
        body_id = self.add_body(origin=warp_transform(np.asarray(body.X_WB)), name=body.name)
        self.bodies_affected_by_visual_forces.append(body_id)
        shape_ids = []
        first_group = max(self.last_collision_group + 1, 1)
        for particle_id, (position, quaternion, radius) in enumerate(
            zip(body.particles.means, body.particles.quats, body.particles.radii)
        ):
            # Same-body spheres are collision geometry for one rigid compound,
            # not mutually colliding particles.  Avoid constructing quadratic
            # filter pairs while preserving the public builder's density.
            self.body_shapes[body_id] = []
            shape_id = self.add_shape_sphere(
                body=body_id,
                radius=float(radius),
                pos=position,
                rot=[quaternion[1], quaternion[2], quaternion[3], quaternion[0]],
                mu=0.0,
                density=1000.0,
                collision_group=first_group + particle_id,
                has_ground_collision=True,
            )
            shape_ids.append(int(shape_id))
        self.body_shapes[body_id] = shape_ids
        self.gaussian_means.extend(body.gaussians.means)
        self.gaussian_quats.extend(body.gaussians.quats)
        self.gaussian_scales.extend(body.gaussians.scales)
        self.gaussian_opacities.extend(body.gaussians.opacities)
        self.gaussian_colors.extend(body.gaussians.colors)
        self.gaussian_body_ids.extend([body_id] * len(body.gaussians.means))
        return int(body_id)

    def add_psm_collision_meshes(self, dataset: Path) -> tuple[list[int], list[str]]:
        mesh_path = dataset / "gui_assets" / "official_psm_tip_meshes_v2.npz"
        poses_path = dataset / "task_inputs" / "psm_link_poses.npz"
        with np.load(poses_path, allow_pickle=False) as poses:
            pose_names = [str(value) for value in poses["link_names"].tolist()]
            initial_poses = np.asarray(poses["X_WL"][0], dtype=np.float32)
        body_ids = []
        body_names = []
        with np.load(mesh_path, allow_pickle=False) as meshes:
            for raw_name in meshes["link_names"].tolist():
                name = str(raw_name)
                pose_index = pose_names.index(name)
                body_id = self.add_body(
                    origin=warp_transform(initial_poses[pose_index]), name=name
                )
                mesh = warp.sim.Mesh(
                    vertices=np.asarray(meshes[f"{name}__vertices"], dtype=np.float32),
                    indices=np.asarray(meshes[f"{name}__faces"], dtype=np.int32).reshape(-1),
                )
                self.add_shape_mesh(
                    body=body_id,
                    mesh=mesh,
                    density=0.0,
                    mu=0.0,
                    has_ground_collision=False,
                    # Warp group -1 collides with every tissue sphere group.
                    # The tissue spheres use distinct groups only to avoid an
                    # O(N^2) same-rigid-body candidate-pair construction.
                    collision_group=-1,
                )
                body_ids.append(int(body_id))
                body_names.append(name)
        return body_ids, body_names


def build_frames(stereo: StereoInputs, device: str):
    sample_name = CAMERAS[0]
    source_width, source_height = stereo.calibration[sample_name]["resolution"]
    target_width = stereo.online_width
    target_height = int(round(source_height * target_width / source_width))
    builder = FramesBuilder(target_width, target_height)
    blender_flip = np.diag([1.0, -1.0, -1.0, 1.0])
    for camera in CAMERAS:
        calibration = stereo.calibration[camera]
        intrinsic = calibration["K"].copy()
        intrinsic[0, :] *= target_width / source_width
        intrinsic[1, :] *= target_height / source_height
        intrinsic[2, :] = (0.0, 0.0, 1.0)
        world_from_camera_blender = calibration["X_WC"] @ blender_flip
        builder.add_camera(camera, intrinsic, world_from_camera_blender)
    return builder.finalize(device)


def pose_rows(world_from_links: np.ndarray, names: list[str], all_names: list[str]) -> np.ndarray:
    rows = []
    for name in names:
        matrix = world_from_links[all_names.index(name)]
        quaternion_xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        rows.append(np.concatenate((matrix[:3, 3], quaternion_xyzw)))
    return np.asarray(rows, dtype=np.float32)


def tissue_virtual_particles(
    state_body_q: torch.Tensor,
    local_positions: np.ndarray,
    local_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    row = state_body_q.detach().cpu().numpy()
    translation = row[:3]
    rotation = Rotation.from_quat(row[3:7]).as_matrix().astype(np.float32)
    positions = (rotation @ local_positions.T).T + translation
    rotations = rotation[None] @ local_rotations
    return positions.astype(np.float32), rotations.astype(np.float32)


def main() -> None:
    args = parse_args()
    dataset = resolve_dataset(REPO_ROOT, args.dataset_key, args.dataset)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有 public-rigid rollout：{output}")
    output.mkdir(parents=True)
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("public rigid EG rollout 需要 CUDA")
    body_path = args.body.expanduser().resolve()
    body = Body.model_validate_json(body_path.read_text(encoding="utf-8"))
    if body.particles is None:
        raise ValueError("body 缺少 particles")

    builder = RigidSimBuilder(
        up_vector=(0.0, 0.0, 1.0),
        gravity=float(PAPER_PARAMETERS["gravity_m_s2"]),
    )
    robot_body_ids, robot_names = builder.add_psm_collision_meshes(dataset)
    tissue_body_id = builder.add_dataset_tissue(body)
    environment = EmbodiedGaussiansEnvironment(builder, device=args.device)
    environment.physics_settings.dt = float(PAPER_PARAMETERS["dt_s"])
    environment.physics_settings.substeps = int(PAPER_PARAMETERS["substeps"])
    environment.physics_settings.xpbd_iterations = int(PAPER_PARAMETERS["jacobi_iterations"])
    model = environment.sim.model
    wp.to_torch(model.body_inv_mass)[robot_body_ids] = 0.0
    wp.to_torch(model.body_inv_inertia)[robot_body_ids] = 0.0

    with np.load(dataset / "task_inputs" / "psm_link_poses.npz", allow_pickle=False) as poses:
        all_robot_names = [str(value) for value in poses["link_names"].tolist()]
        world_from_links = np.asarray(poses["X_WL"], dtype=np.float32)
    stereo = StereoInputs(dataset, args.online_width)
    frames = build_frames(stereo, args.device)
    formal_frames = int(DATASETS[args.dataset_key]["frames"])
    frame_count = formal_frames if args.frame_limit < 0 else min(args.frame_limit, formal_frames)
    render_set = set(rendering_frames(formal_frames))
    local_positions = np.asarray(body.particles.means, dtype=np.float32)
    local_rotations = Rotation.from_quat(
        np.asarray(body.particles.quats, dtype=np.float32), scalar_first=True
    ).as_matrix().astype(np.float32)
    particle_positions = np.empty((frame_count, len(local_positions), 3), dtype=np.float32)
    particle_rotations = np.empty((frame_count, len(local_positions), 3, 3), dtype=np.float32)
    diagnostics = []
    start = time.perf_counter()

    for frame in range(frame_count):
        frame_start = time.perf_counter()
        robot_rows = torch.as_tensor(
            pose_rows(world_from_links[frame], robot_names, all_robot_names),
            device=args.device,
        )
        for state in (environment.sim.state_0, environment.sim.state_1):
            body_q = wp.to_torch(state.body_q)
            body_q[robot_body_ids] = robot_rows
            wp.to_torch(state.body_qd)[robot_body_ids] = 0.0

        allowed = observation_allowed(frame, formal_frames)
        if allowed:
            for camera in CAMERAS:
                target, _, _, _ = stereo.load_online(camera, frame, torch.device(args.device))
                frames.update_colors(
                    camera,
                    float(stereo.calibration[camera]["timestamps"][frame]),
                    target,
                )
            environment.frames = frames
        else:
            environment.frames = None
        environment.step()
        tissue_q = wp.to_torch(environment.sim.state_0.body_q)[tissue_body_id]
        positions, rotations = tissue_virtual_particles(
            tissue_q, local_positions, local_rotations
        )
        particle_positions[frame] = positions
        particle_rotations[frame] = rotations

        if frame in render_set and not args.no_render:
            for camera in CAMERAS:
                intrinsic, view_matrix, size = stereo.full_camera_tensors(
                    camera, torch.device(args.device)
                )
                rendered, alpha, _ = environment.sim.render_gaussians(
                    environment.sim.gaussian_state,
                    view_matrix,
                    intrinsic,
                    size[0],
                    size[1],
                    torch.zeros(3, device=args.device),
                    near_plane=0.01,
                    far_plane=2.0,
                )
                save_render(output / "rgb" / camera / f"{frame:06d}.png", rendered[0])
                save_render(output / "alpha" / camera / f"{frame:06d}.png", alpha[0], alpha=True)
        diagnostics.append(
            {
                "frame": frame,
                "observation_allowed": allowed,
                "elapsed_s": time.perf_counter() - frame_start,
            }
        )
        print(
            f"[EG-Public-Rigid] {args.dataset_key} frame {frame + 1:04d}/{frame_count:04d} "
            f"observe={allowed} elapsed={diagnostics[-1]['elapsed_s']:.3f}s",
            flush=True,
        )

    raw_path = output / "raw_particle_rollout.npz"
    np.savez_compressed(
        raw_path,
        frame_indices=np.arange(frame_count, dtype=np.int32),
        timestamps=np.asarray(stereo.calibration[CAMERAS[0]]["timestamps"][:frame_count]),
        particle_positions_world=particle_positions,
        particle_rotations_world=particle_rotations,
    )
    (output / "frame_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    visual = environment.visual_forces_settings
    metadata = {
        "schema": "fixedsuperbest.embodied_gaussians_public_rigid_rollout.v1",
        "formal": args.frame_limit < 0 and not args.no_render,
        "method": "official public rigid-only reference implementation capability audit",
        "dataset_key": args.dataset_key,
        "upstream_commit": UPSTREAM_COMMIT,
        "body": str(body_path),
        "body_sha256": sha256_file(body_path),
        "actuation": "psm-fk-collision-only",
        "representation": "one rigid tissue body; zero deformable degrees of freedom",
        "visual_force_settings": {
            "iterations": visual.iterations,
            "lr_means": visual.lr_means,
            "lr_quats": visual.lr_quats,
            "lr_color": visual.lr_color,
            "lr_opacity": visual.lr_opacity,
            "lr_scale": visual.lr_scale,
            "kp": visual.kp,
            "loss": "upstream full-batch RGB MSE",
        },
        "timing_adapter": {
            "dt_s": environment.physics_settings.dt,
            "substeps": environment.physics_settings.substeps,
            "xpbd_iterations": environment.physics_settings.xpbd_iterations,
        },
        "robot_collision_adapter": (
            "known PSM meshes use Warp collision group -1 so they collide with every "
            "single-shape tissue group; same rigid tissue shapes remain filtered"
        ),
        "online_width": args.online_width,
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
