#!/usr/bin/env python3
"""Deterministic invariance tests for the paper shape-matching reconstruction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baselines" / "embodied_gaussians_sim"))
from shape_matching import OrientedShapeMatcher, build_particle_neighbour_clusters  # noqa: E402


def main() -> None:
    rest = np.asarray(
        [
            [-0.02, -0.01, 0.00],
            [0.02, -0.01, 0.00],
            [-0.02, 0.01, 0.00],
            [0.02, 0.01, 0.00],
            [0.00, 0.00, 0.01],
            [0.00, 0.00, -0.01],
        ],
        dtype=np.float32,
    )
    clusters = build_particle_neighbour_clusters(
        rest, particle_mass=0.3
    )
    matcher = OrientedShapeMatcher(clusters, "cpu")
    identity = torch.eye(3).repeat(len(rest), 1, 1)

    unchanged, unchanged_rotations = matcher.project(
        torch.from_numpy(rest), identity, stiffness=1.0
    )
    if not torch.allclose(unchanged, torch.from_numpy(rest), atol=2.0e-6):
        raise AssertionError("静止构型不是 shape-matching 不动点")
    if not torch.allclose(unchanged_rotations, identity, atol=2.0e-6):
        raise AssertionError("静止方向不是 shape-matching 不动点")

    rigid_rotation = torch.from_numpy(
        Rotation.from_euler("zyx", [20.0, -7.0, 11.0], degrees=True)
        .as_matrix()
        .astype(np.float32)
    )
    translation = torch.tensor([0.1, -0.04, 0.03])
    rigid_positions = torch.from_numpy(rest) @ rigid_rotation.T + translation
    rigid_orientations = rigid_rotation.repeat(len(rest), 1, 1)
    projected, projected_rotations = matcher.project(
        rigid_positions, rigid_orientations, stiffness=1.0
    )
    if not torch.allclose(projected, rigid_positions, atol=2.0e-6):
        raise AssertionError("shape matching 错误改变了刚体变换")
    if not torch.allclose(projected_rotations, rigid_orientations, atol=2.0e-6):
        raise AssertionError("shape matching 错误改变了刚体方向")

    deformed = rigid_positions.clone()
    deformed[0] += torch.tensor([0.0, 0.0, 0.01])
    before = torch.linalg.norm(deformed - rigid_positions)
    after_positions, _ = matcher.project(deformed, rigid_orientations, stiffness=0.5)
    after = torch.linalg.norm(after_positions - rigid_positions)
    if not after < before:
        raise AssertionError("局部 shape matching 没有减小非刚性扰动")
    print("PASS: EG oriented clustered shape matching invariants")


if __name__ == "__main__":
    main()
