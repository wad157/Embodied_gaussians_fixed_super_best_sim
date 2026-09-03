#!/usr/bin/env python3
"""Stage-C gates for tetrahedron-to-Gaussian soft-tissue skinning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_gaussians.embodied_simulator.builder import (  # noqa: E402
    EmbodiedGaussiansBuilder,
)
from embodied_gaussians.embodied_simulator.simulator import (  # noqa: E402
    update_gaussian_transforms,
)
from embodied_gaussians.scene_builders.domain import SoftBody  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-C soft skinning gates.")
    parser.add_argument(
        "--asset",
        type=Path,
        default=REPO_ROOT
        / "data/super/grasp5_native/soft_tissue_v1/tissue_soft.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "data/super/grasp5_native/soft_tissue_v1/skinning_gate_report.json",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--warp-cache-dir", type=Path, default=Path("/tmp/warp-super-c-cache")
    )
    return parser.parse_args()


def quat_multiply_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quaternion_angle_error(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    actual = torch.nn.functional.normalize(actual, dim=-1)
    expected = torch.nn.functional.normalize(expected, dim=-1)
    dot = torch.abs(torch.sum(actual * expected, dim=-1)).clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def run_skinning(gaussian_model, model, physics_state, state) -> None:
    state.means.zero_()
    state.quats.zero_()
    update_gaussian_transforms(
        gaussian_model,
        physics_state.body_q,
        state,
        particle_q=physics_state.particle_q,
    )
    wp.synchronize_device(model.device)


def main() -> None:
    args = parse_args()
    wp.config.kernel_cache_dir = str(args.warp_cache_dir)
    wp.init()
    if args.device == "auto":
        args.device = "cuda" if wp.is_cuda_available() else "cpu"

    soft_body = SoftBody.from_npz(args.asset, name="super_tissue")
    soft_builder = EmbodiedGaussiansBuilder(up_vector=(0.0, 0.0, 1.0))
    local_handle = soft_builder.add_soft_body(
        soft_body,
        young_modulus_pa=15000.0,
        poisson_ratio=0.45,
        anchor_mode="support_candidate",
        add_gaussians=True,
    )

    # Exercise offsets used when SUPER copies a per-environment builder into the
    # final scene.  The sentinels must remain untouched by soft skinning.
    builder = EmbodiedGaussiansBuilder(up_vector=(0.0, 0.0, 1.0))
    builder.add_particle((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), 0.0, radius=0.001)
    builder.gaussian_means.append([1.0, 1.0, 1.0])
    builder.gaussian_quats.append([1.0, 0.0, 0.0, 0.0])
    builder.gaussian_scales.append([0.001, 0.001, 0.001])
    builder.gaussian_opacities.append(0.5)
    builder.gaussian_colors.append([0.5, 0.5, 0.5])
    builder.gaussian_body_ids.append(-1)
    builder.add_builder(soft_builder)
    handle = builder.soft_body_handles[0]
    model = builder.finalize(device=args.device)
    gaussian_model = builder.gaussian_model
    gaussian_state = gaussian_model.state()
    physics_state = model.state()

    if handle.particle_start != 1 or handle.gaussian_start != 1:
        raise RuntimeError("add_builder did not offset soft particle/Gaussian ranges")
    if local_handle.gaussian_end - local_handle.gaussian_start != 2490:
        raise RuntimeError("Unexpected stage-A soft Gaussian count")

    rest_particles = wp.to_torch(model.particle_q).clone()
    soft_slice = slice(handle.gaussian_start, handle.gaussian_end)
    rest_means = gaussian_model.means[soft_slice]
    rest_quats = gaussian_model.quats[soft_slice]

    run_skinning(gaussian_model, model, physics_state, gaussian_state)
    rest_mean_error = torch.linalg.vector_norm(
        gaussian_state.means[soft_slice] - rest_means, dim=1
    )
    rest_quat_error = quaternion_angle_error(
        gaussian_state.quats[soft_slice], rest_quats
    )
    sentinel_unchanged = bool(
        torch.equal(gaussian_state.means[0], torch.zeros_like(gaussian_state.means[0]))
        and torch.equal(
            gaussian_state.quats[0], torch.zeros_like(gaussian_state.quats[0])
        )
    )

    axis = torch.tensor([0.3, -0.5, 0.8], device=args.device, dtype=torch.float32)
    axis = axis / torch.linalg.vector_norm(axis)
    angle = torch.tensor(0.47, device=args.device, dtype=torch.float32)
    half = 0.5 * angle
    q_delta = torch.cat((torch.cos(half).reshape(1), axis * torch.sin(half)))
    w, x, y, z = q_delta
    rotation = torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w))),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w))),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))),
        )
    )
    translation = torch.tensor(
        [0.012, -0.007, 0.018], device=args.device, dtype=torch.float32
    )
    transformed_particles = rest_particles @ rotation.T + translation
    wp.copy(physics_state.particle_q, wp.from_torch(transformed_particles))
    run_skinning(gaussian_model, model, physics_state, gaussian_state)
    expected_means = rest_means @ rotation.T + translation
    expected_quats = quat_multiply_wxyz(q_delta.expand_as(rest_quats), rest_quats)
    rigid_mean_error = torch.linalg.vector_norm(
        gaussian_state.means[soft_slice] - expected_means, dim=1
    )
    rigid_quat_error = quaternion_angle_error(
        gaussian_state.quats[soft_slice], expected_quats
    )

    local_particles = rest_particles.clone()
    soft_particles = local_particles[handle.particle_start : handle.particle_end]
    xy_center = torch.mean(soft_particles[:, :2], dim=0)
    radial_distance = torch.linalg.vector_norm(
        soft_particles[:, :2] - xy_center, dim=1
    )
    radial_weight = torch.exp(-0.5 * (radial_distance / 0.012) ** 2)
    z_min = torch.min(soft_particles[:, 2])
    soft_particles[:, 2] -= 0.08 * (soft_particles[:, 2] - z_min) * radial_weight
    wp.copy(physics_state.particle_q, wp.from_torch(local_particles))
    run_skinning(gaussian_model, model, physics_state, gaussian_state)
    local_means = gaussian_state.means[soft_slice]
    local_displacement = torch.linalg.vector_norm(local_means - rest_means, dim=1)

    rest_means_cpu = rest_means.detach().cpu().numpy()
    _, nearest_indices = cKDTree(rest_means_cpu).query(rest_means_cpu, k=2)
    nearest_indices_t = torch.as_tensor(
        nearest_indices[:, 1], device=args.device, dtype=torch.long
    )
    neighbor_jump = torch.abs(
        local_displacement - local_displacement[nearest_indices_t]
    )
    gaussian_radius = torch.linalg.vector_norm(
        rest_means[:, :2] - xy_center, dim=1
    )
    near = local_displacement[gaussian_radius < 0.008]
    far = local_displacement[gaussian_radius > 0.025]
    quat_norm_error = torch.abs(
        torch.linalg.vector_norm(gaussian_state.quats[soft_slice], dim=1) - 1.0
    )

    gates = {
        "builder_offsets": bool(
            handle.particle_start == 1
            and handle.gaussian_start == 1
            and int(gaussian_model.soft_gaussian_ids.min().item()) == 1
            and int(gaussian_model.soft_gaussian_particle_indices.min().item()) >= 1
        ),
        "rest_pose": bool(
            rest_mean_error.max().item() < 1.0e-6
            and rest_quat_error.max().item() < 1.0e-3
            and sentinel_unchanged
        ),
        "rigid_equivariance": bool(
            rigid_mean_error.max().item() < 2.0e-6
            and rigid_quat_error.max().item() < 1.0e-3
        ),
        "local_compression": bool(
            torch.isfinite(local_means).all().item()
            and torch.isfinite(gaussian_state.quats[soft_slice]).all().item()
            and near.mean().item() > 2.0 * far.mean().item()
            and torch.quantile(neighbor_jump, 0.95).item() < 2.5e-4
            and quat_norm_error.max().item() < 2.0e-5
        ),
        "scale_unchanged": bool(
            torch.equal(
                gaussian_state.scale_log[soft_slice],
                torch.log(gaussian_model.scales[soft_slice]),
            )
        ),
    }
    report = {
        "stage": "C_gaussian_skinning",
        "device": str(model.device),
        "counts": {
            "particles": int(model.particle_count),
            "tetrahedra": int(model.tet_count),
            "gaussians": int(gaussian_model.num_gaussians),
            "soft_gaussians": int(gaussian_model.num_soft_gaussians),
        },
        "rest_pose": {
            "max_mean_error_m": float(rest_mean_error.max().item()),
            "max_quaternion_angle_error_rad": float(rest_quat_error.max().item()),
            "static_sentinel_untouched": sentinel_unchanged,
        },
        "rigid_transform": {
            "max_mean_error_m": float(rigid_mean_error.max().item()),
            "max_quaternion_angle_error_rad": float(rigid_quat_error.max().item()),
        },
        "local_compression": {
            "max_displacement_m": float(local_displacement.max().item()),
            "near_mean_displacement_m": float(near.mean().item()),
            "far_mean_displacement_m": float(far.mean().item()),
            "neighbor_jump_p95_m": float(torch.quantile(neighbor_jump, 0.95).item()),
            "max_quaternion_norm_error": float(quat_norm_error.max().item()),
        },
        "gates": gates,
        "passed": bool(all(gates.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Stage-C skinning gate failed")


if __name__ == "__main__":
    main()
