#!/usr/bin/env python3
"""Oriented-particle clustered shape matching from EG equations (4)--(5).

This module deliberately has no tetrahedral, distance, volume, FEM, tracker,
depth-correction, or material-identification dependency.  Every particle owns
one deformable shape made from itself and its nearest neighbours, exactly the
construction stated by the Embodied Gaussians paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import Delaunay


@dataclass(frozen=True)
class ShapeClusters:
    members: np.ndarray
    valid: np.ndarray
    rest_positions: np.ndarray
    rest_centers: np.ndarray
    masses: np.ndarray

    @property
    def cluster_count(self) -> int:
        return int(self.members.shape[0])


def build_particle_neighbour_clusters(
    rest_positions: np.ndarray,
    *,
    particle_mass: float,
) -> ShapeClusters:
    """Create one shape per particle from parameter-free Delaunay adjacency."""
    rest = np.asarray(rest_positions, dtype=np.float64)
    if rest.ndim != 2 or rest.shape[1] != 3 or len(rest) < 4:
        raise ValueError(f"rest_positions 必须是至少四个三维点，收到 {rest.shape}")
    if particle_mass <= 0.0:
        raise ValueError("particle_mass 必须为正")

    # The paper says each particle and its neighbours form a deformable shape,
    # but publishes no k or distance threshold.  Delaunay adjacency supplies
    # a rest-state geometric neighbourhood without introducing such a
    # tunable value.  QJ handles the builder's nearly planar double layer.
    tessellation = Delaunay(rest, qhull_options="Qbb Qc Q12 QJ")
    neighbours = [{index} for index in range(len(rest))]
    for simplex in tessellation.simplices:
        ids = [int(value) for value in simplex if int(value) < len(rest)]
        for first in ids:
            neighbours[first].update(value for value in ids if value != first)
    rows = []
    for center, adjacent in enumerate(neighbours):
        if len(adjacent) < 4:
            raise RuntimeError(f"粒子 {center} 的 Delaunay shape 少于四个成员")
        rows.append(
            [center]
            + sorted(
                (value for value in adjacent if value != center),
                key=lambda value: (
                    float(np.linalg.norm(rest[value] - rest[center])),
                    int(value),
                ),
            )
        )
    width = max(len(row) for row in rows)
    members = np.empty((len(rows), width), dtype=np.int64)
    valid = np.zeros((len(rows), width), dtype=bool)
    for center, row in enumerate(rows):
        members[center] = center
        members[center, : len(row)] = row
        valid[center, : len(row)] = True
    gathered = rest[members]
    masses = valid.astype(np.float64) * float(particle_mass)
    centers = (gathered * masses[..., None]).sum(axis=1) / masses.sum(axis=1)[:, None]
    return ShapeClusters(
        members=members,
        valid=valid,
        rest_positions=rest.astype(np.float32),
        rest_centers=centers.astype(np.float32),
        masses=masses.astype(np.float32),
    )


def _proper_polar(matrix: torch.Tensor) -> torch.Tensor:
    """Return the closest proper rotation for a batch of 3x3 matrices."""
    u, _, vh = torch.linalg.svd(matrix)
    rotation = u @ vh
    reflected = torch.det(rotation) < 0.0
    if bool(reflected.any()):
        fix = torch.ones((*matrix.shape[:-2], 3), dtype=matrix.dtype, device=matrix.device)
        fix[..., 2] = torch.where(reflected, -1.0, 1.0)
        rotation = u @ torch.diag_embed(fix) @ vh
    return rotation


class OrientedShapeMatcher:
    """Batched Jacobi projector for the paper's deformable shape construction."""

    def __init__(self, clusters: ShapeClusters, device: torch.device | str):
        self.device = torch.device(device)
        self.members = torch.as_tensor(clusters.members, dtype=torch.long, device=self.device)
        self.valid = torch.as_tensor(clusters.valid, dtype=torch.bool, device=self.device)
        self.rest = torch.as_tensor(clusters.rest_positions, dtype=torch.float32, device=self.device)
        self.rest_centers = torch.as_tensor(
            clusters.rest_centers, dtype=torch.float32, device=self.device
        )
        self.masses = torch.as_tensor(clusters.masses, dtype=torch.float32, device=self.device)
        self.total_mass = self.masses.sum(dim=1)
        flat_members = self.members.reshape(-1)
        self.flat_members = flat_members[self.valid.reshape(-1)]
        self.contribution_count = torch.bincount(
            self.flat_members, minlength=len(self.rest)
        ).clamp_min(1).to(torch.float32)

    def project(
        self,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        stiffness: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.shape != self.rest.shape:
            raise ValueError(f"positions 形状错误：{positions.shape} != {self.rest.shape}")
        if rotations.shape != (len(self.rest), 3, 3):
            raise ValueError("rotations 必须是 particle_count x 3 x 3")
        if not 0.0 <= stiffness <= 1.0:
            raise ValueError("shape stiffness 必须位于 [0,1]")

        current = positions[self.members]
        member_rotations = rotations[self.members]
        weights = self.masses[..., None]
        current_centers = (current * weights).sum(dim=1) / self.total_mass[:, None]
        rest_members = self.rest[self.members]

        # Eq. (4): A_S = sum(1/5 m_i R_i + p_i pbar_i^T)
        #                    - M c_S cbar_S^T.
        position_outer = torch.einsum(
            "cm,cmi,cmj->cij", self.masses, current, rest_members
        )
        orientation_term = (
            self.masses[..., None, None] * member_rotations / 5.0
        ).sum(dim=1)
        centroid_outer = self.total_mass[:, None, None] * torch.einsum(
            "ci,cj->cij", current_centers, self.rest_centers
        )
        shape_rotations = _proper_polar(
            orientation_term + position_outer - centroid_outer
        )

        rest_relative = rest_members - self.rest_centers[:, None, :]
        goals = (
            torch.einsum("cij,cmj->cmi", shape_rotations, rest_relative)
            + current_centers[:, None, :]
        )
        member_deltas = (goals - current).reshape(-1, 3)[self.valid.reshape(-1)]
        accumulated = torch.zeros_like(positions)
        accumulated.index_add_(0, self.flat_members, member_deltas)
        projected_positions = positions + float(stiffness) * (
            accumulated / self.contribution_count[:, None]
        )

        # Algorithm 2 also updates particle orientations in shapeMatching().
        # Average all incident shape rotations with the same Jacobi incidence,
        # blend by k_S, then re-project onto SO(3).
        rotation_contributions = shape_rotations[:, None].expand(
            -1, self.members.shape[1], -1, -1
        ).reshape(-1, 3, 3)[self.valid.reshape(-1)]
        rotation_sum = torch.zeros_like(rotations)
        rotation_sum.index_add_(0, self.flat_members, rotation_contributions)
        averaged = rotation_sum / self.contribution_count[:, None, None]
        blended = (1.0 - float(stiffness)) * rotations + float(stiffness) * averaged
        projected_rotations = _proper_polar(blended)
        return projected_positions, projected_rotations
