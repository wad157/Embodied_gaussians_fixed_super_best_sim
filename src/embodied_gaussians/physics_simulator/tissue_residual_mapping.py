"""Deterministic visible-surface residual mapping for volumetric tissue.

The mapper keeps the physical prediction immutable.  It estimates a separate
residual displacement by matching camera-visible top particles to a surface
point cloud, smoothing that displacement over the tetrahedral surface graph,
and propagating it to sub-surface particles with a depth-decaying harmonic
extension.  Fixed particles always receive zero residual.

This is the project-scale counterpart of the volumetric residual mapping in
arXiv:2309.11656: the observation term is surface-only, while unobserved
particles are informed by mesh geometry.  Exact tetrahedral material energy
and no-flip checks are applied by the replay that consumes this mapper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class TissueResidualMappingSettings:
    iterations: int = 20
    spatial_weight: float = 0.35
    temporal_weight: float = 0.50
    magnitude_weight: float = 0.05
    data_distance_scale_m: float = 0.003
    data_gate_m: float = 0.010
    maximum_residual_m: float = 0.005
    subsurface_decay_depth_m: float = 0.006

    def validate(self) -> None:
        if self.iterations < 1:
            raise ValueError("Residual mapping iterations must be positive")
        if min(
            self.spatial_weight,
            self.temporal_weight,
            self.magnitude_weight,
        ) < 0.0:
            raise ValueError("Residual mapping weights must be non-negative")
        if (
            self.data_distance_scale_m <= 0.0
            or self.data_gate_m <= 0.0
            or self.maximum_residual_m <= 0.0
            or self.subsurface_decay_depth_m <= 0.0
        ):
            raise ValueError("Residual mapping metric scales must be positive")


def _sorted_surface_adjacency(
    top_particle_ids: np.ndarray,
    top_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    top_particle_ids = np.asarray(top_particle_ids, dtype=np.int64)
    top_edges = np.asarray(top_edges, dtype=np.int64)
    local_by_particle = np.full(
        int(top_particle_ids.max()) + 1, -1, dtype=np.int64
    )
    local_by_particle[top_particle_ids] = np.arange(
        len(top_particle_ids), dtype=np.int64
    )
    local_edges = local_by_particle[top_edges]
    if np.any(local_edges < 0):
        raise ValueError("Top-surface edge references a non-top particle")
    directed = np.concatenate(
        (local_edges, local_edges[:, ::-1]), axis=0
    )
    counts = np.bincount(directed[:, 0], minlength=len(top_particle_ids))
    isolated = np.flatnonzero(counts == 0)
    if len(isolated):
        directed = np.concatenate(
            (directed, np.column_stack((isolated, isolated))), axis=0
        )
    order = np.lexsort((directed[:, 1], directed[:, 0]))
    directed = directed[order]
    counts = np.bincount(directed[:, 0], minlength=len(top_particle_ids))
    offsets = np.zeros(len(top_particle_ids) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    return offsets, directed[:, 1].astype(np.int64, copy=False)


class TetrahedralTissueResidualMapper:
    """Map visible point-cloud residuals into a full volumetric displacement."""

    def __init__(
        self,
        *,
        rest_positions: np.ndarray,
        top_node_mask: np.ndarray,
        fixed_mask: np.ndarray,
        inward_depth: np.ndarray,
        top_edges: np.ndarray,
        initial_observation: np.ndarray,
        initial_visibility_distance_m: float = 0.002,
        settings: TissueResidualMappingSettings | None = None,
    ) -> None:
        self.settings = settings or TissueResidualMappingSettings()
        self.settings.validate()
        self.rest_positions = np.asarray(
            rest_positions, dtype=np.float64
        )
        self.top_node_mask = np.asarray(top_node_mask, dtype=bool)
        self.fixed_mask = np.asarray(fixed_mask, dtype=bool)
        self.inward_depth = np.asarray(inward_depth, dtype=np.float64)
        if (
            self.rest_positions.ndim != 2
            or self.rest_positions.shape[1] != 3
        ):
            raise ValueError("rest_positions must have shape [N,3]")
        particle_count = len(self.rest_positions)
        if any(
            len(array) != particle_count
            for array in (
                self.top_node_mask,
                self.fixed_mask,
                self.inward_depth,
            )
        ):
            raise ValueError("Residual mapper particle arrays disagree")

        self.top_particle_ids = np.flatnonzero(self.top_node_mask).astype(
            np.int64
        )
        self.top_rest_positions = self.rest_positions[self.top_particle_ids]
        self.top_offsets, self.top_neighbors = _sorted_surface_adjacency(
            self.top_particle_ids, top_edges
        )
        self.top_degrees = np.diff(self.top_offsets).astype(np.float64)
        initial_observation = np.asarray(
            initial_observation, dtype=np.float64
        )
        initial_distances = cKDTree(initial_observation).query(
            self.top_rest_positions, k=1, workers=1
        )[0]
        self.visible_top_mask = (
            initial_distances <= float(initial_visibility_distance_m)
        )
        minimum_visible = min(100, max(4, len(self.top_particle_ids) // 10))
        if np.count_nonzero(self.visible_top_mask) < minimum_visible:
            raise ValueError("Too few initially visible top particles")

        # A deterministic nearest-column extension is sufficient because the
        # asset already stores each particle's inward depth.  Surface graph
        # smoothing happens before this extension.
        top_xy_tree = cKDTree(self.top_rest_positions[:, :2])
        self.particle_to_top_local = top_xy_tree.query(
            self.rest_positions[:, :2], k=1, workers=1
        )[1].astype(np.int64)
        nonnegative_depth = np.maximum(self.inward_depth, 0.0)
        self.subsurface_attenuation = np.exp(
            -nonnegative_depth
            / self.settings.subsurface_decay_depth_m
        )
        self.subsurface_attenuation[self.top_node_mask] = 1.0
        self.subsurface_attenuation[self.fixed_mask] = 0.0

    def _neighbor_average(self, values: np.ndarray) -> np.ndarray:
        gathered = values[self.top_neighbors]
        sums = np.add.reduceat(gathered, self.top_offsets[:-1], axis=0)
        return sums / self.top_degrees[:, None]

    @staticmethod
    def _clip_vectors(vectors: np.ndarray, maximum_norm: float) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1)
        scale = np.minimum(1.0, maximum_norm / np.maximum(norms, 1.0e-12))
        return vectors * scale[:, None]

    def map(
        self,
        *,
        physical_positions: np.ndarray,
        observation_points: np.ndarray | None,
        previous_top_residual: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
        physical_positions = np.asarray(
            physical_positions, dtype=np.float64
        )
        if physical_positions.shape != self.rest_positions.shape:
            raise ValueError("Physical particle state has the wrong shape")
        top_positions = physical_positions[self.top_particle_ids]
        data_targets = np.zeros_like(top_positions)
        data_weights = np.zeros(len(top_positions), dtype=np.float64)
        matched_distance = np.full(
            len(top_positions), np.inf, dtype=np.float64
        )
        if observation_points is not None and len(observation_points):
            observation_points = np.asarray(
                observation_points, dtype=np.float64
            )
            observation_tree = cKDTree(observation_points)
            matched_distance, matched_ids = observation_tree.query(
                top_positions, k=1, workers=1
            )
            active = self.visible_top_mask & (
                matched_distance <= self.settings.data_gate_m
            )
            simulation_to_observation = (
                observation_points[matched_ids[active]]
                - top_positions[active]
            )
            scaled_distance = (
                matched_distance[active]
                / self.settings.data_distance_scale_m
            )
            simulation_weights = 1.0 / (1.0 + scaled_distance**2)
            target_numerator = np.zeros_like(top_positions)
            target_numerator[active] = (
                simulation_weights[:, None] * simulation_to_observation
            )
            data_weights[active] = simulation_weights

            # Complete the symmetric Chamfer data term.  Observation points
            # splat their residuals into the nearest visible top particle;
            # accumulation is on CPU and follows point-cloud storage order.
            top_tree = cKDTree(top_positions)
            observation_distance, observation_top_ids = top_tree.query(
                observation_points, k=1, workers=1
            )
            observation_active = (
                observation_distance <= self.settings.data_gate_m
            ) & self.visible_top_mask[observation_top_ids]
            observation_ids = observation_top_ids[observation_active]
            observation_scaled = (
                observation_distance[observation_active]
                / self.settings.data_distance_scale_m
            )
            observation_weights = 1.0 / (
                1.0 + observation_scaled**2
            )
            observation_residuals = (
                observation_points[observation_active]
                - top_positions[observation_ids]
            )
            np.add.at(
                target_numerator,
                observation_ids,
                observation_weights[:, None] * observation_residuals,
            )
            np.add.at(data_weights, observation_ids, observation_weights)
            has_data = data_weights > 0.0
            data_targets[has_data] = (
                target_numerator[has_data] / data_weights[has_data, None]
            )
            # Avoid making denser image regions mechanically stronger.
            data_weights = np.minimum(data_weights, 4.0)
            data_targets = self._clip_vectors(
                data_targets, self.settings.maximum_residual_m
            )

        if previous_top_residual is None:
            previous = np.zeros_like(top_positions)
        else:
            previous = np.asarray(
                previous_top_residual, dtype=np.float64
            )
            if previous.shape != top_positions.shape:
                raise ValueError("Previous top residual has the wrong shape")
        current = previous.copy()
        denominator = (
            data_weights
            + self.settings.temporal_weight
            + self.settings.spatial_weight
            + self.settings.magnitude_weight
        )
        for _ in range(self.settings.iterations):
            neighbor_average = self._neighbor_average(current)
            current = (
                data_weights[:, None] * data_targets
                + self.settings.temporal_weight * previous
                + self.settings.spatial_weight * neighbor_average
            ) / denominator[:, None]
            current = self._clip_vectors(
                current, self.settings.maximum_residual_m
            )

        full_residual = (
            current[self.particle_to_top_local]
            * self.subsurface_attenuation[:, None]
        )
        full_residual[self.top_particle_ids] = current
        full_residual[self.fixed_mask] = 0.0
        full_residual = self._clip_vectors(
            full_residual, self.settings.maximum_residual_m
        )
        active_distances = matched_distance[np.isfinite(matched_distance)]
        metrics: dict[str, float | int] = {
            "visible_top_particle_count": int(
                np.count_nonzero(self.visible_top_mask)
            ),
            "active_data_particle_count": int(
                np.count_nonzero(data_weights > 0.0)
            ),
            "maximum_residual_m": float(
                np.linalg.norm(full_residual, axis=1).max()
            ),
            "rms_residual_m": float(
                np.sqrt(np.mean(np.sum(full_residual**2, axis=1)))
            ),
            "maximum_top_residual_m": float(
                np.linalg.norm(current, axis=1).max()
            ),
            "matched_distance_p95_m": (
                float(np.quantile(active_distances, 0.95))
                if len(active_distances)
                else float("inf")
            ),
        }
        return (
            full_residual.astype(np.float32),
            current.astype(np.float64),
            metrics,
        )
