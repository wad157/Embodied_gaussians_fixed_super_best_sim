"""Flow/depth observations bound to fixed physical-surface particle ranges.

The first implementation deliberately separates binding from state mutation:
an image track is lifted once with depth, associated with a compact range of
surface particles, and keeps those particle ids and weights for its lifetime.
Later frames may update the observation, but never re-run nearest-neighbour
association and therefore cannot slide the material identity over the tissue.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FlowDepthRangeBindingSettings:
    """Fixed V1 settings for one track's material support."""

    radius_m: float = 0.006
    fallback_radius_m: float = 0.008
    sigma_m: float = 0.003
    minimum_movable_particles: int = 3
    maximum_depth_sampling_radius_px: int = 2

    def validate(self) -> None:
        if self.radius_m <= 0.0:
            raise ValueError("range-binding radius must be positive")
        if self.fallback_radius_m < self.radius_m:
            raise ValueError("fallback radius must not be smaller than primary radius")
        if self.sigma_m <= 0.0:
            raise ValueError("range-binding sigma must be positive")
        if self.minimum_movable_particles < 1:
            raise ValueError("minimum movable-particle count must be positive")
        if self.maximum_depth_sampling_radius_px < 0:
            raise ValueError("maximum depth sampling radius must be non-negative")


@dataclass(frozen=True)
class FlowDepthTriangleBindingSettings:
    """Fixed binding from a lifted image track to one physical surface face."""

    primary_maximum_surface_distance_m: float = 0.001
    fallback_maximum_surface_distance_m: float = 0.002
    minimum_movable_vertices: int = 3
    maximum_depth_sampling_radius_px: int = 2

    def validate(self) -> None:
        if self.primary_maximum_surface_distance_m <= 0.0:
            raise ValueError("primary triangle distance must be positive")
        if (
            self.fallback_maximum_surface_distance_m
            < self.primary_maximum_surface_distance_m
        ):
            raise ValueError(
                "fallback triangle distance must not be smaller than primary"
            )
        if not 1 <= self.minimum_movable_vertices <= 3:
            raise ValueError("minimum movable triangle vertices must lie in [1, 3]")
        if self.maximum_depth_sampling_radius_px < 0:
            raise ValueError("maximum depth sampling radius must be non-negative")


@dataclass(frozen=True)
class FlowDepthParticleRangeBindings:
    """Padded, deterministic track-to-particle material bindings."""

    track_valid: np.ndarray
    particle_ids: np.ndarray
    particle_weights: np.ndarray
    support_counts: np.ndarray
    movable_support_counts: np.ndarray
    binding_radius_m: np.ndarray
    initial_depth_m: np.ndarray
    depth_sampling_radius_px: np.ndarray
    initial_points_table: np.ndarray
    nearest_surface_distance_m: np.ndarray
    binding_method: str = "particle_range"
    surface_face_ids: np.ndarray | None = None
    surface_projection_points_table: np.ndarray | None = None

    def validate(
        self,
        *,
        surface_mask: np.ndarray | None = None,
        fixed_mask: np.ndarray | None = None,
    ) -> None:
        track_count = len(self.track_valid)
        expected_vectors = {
            "support_counts": self.support_counts,
            "movable_support_counts": self.movable_support_counts,
            "binding_radius_m": self.binding_radius_m,
            "initial_depth_m": self.initial_depth_m,
            "depth_sampling_radius_px": self.depth_sampling_radius_px,
            "nearest_surface_distance_m": self.nearest_surface_distance_m,
        }
        for name, values in expected_vectors.items():
            if values.shape != (track_count,):
                raise ValueError(f"{name} must have one value per track")
        if self.initial_points_table.shape != (track_count, 3):
            raise ValueError("initial table points must be shaped (tracks, 3)")
        if self.surface_face_ids is not None and self.surface_face_ids.shape != (
            track_count,
        ):
            raise ValueError("surface face ids must have one value per track")
        if (
            self.surface_projection_points_table is not None
            and self.surface_projection_points_table.shape != (track_count, 3)
        ):
            raise ValueError(
                "surface projection points must be shaped (tracks, 3)"
            )
        if self.particle_ids.ndim != 2 or self.particle_ids.shape[0] != track_count:
            raise ValueError("particle ids must be a padded track-by-support matrix")
        if self.particle_weights.shape != self.particle_ids.shape:
            raise ValueError("particle weights disagree with particle ids")
        if not np.array_equal(
            self.support_counts,
            np.sum(self.particle_ids >= 0, axis=1).astype(np.int32),
        ):
            raise ValueError("stored support counts disagree with padded ids")
        if np.any(self.particle_weights[self.particle_ids < 0] != 0.0):
            raise ValueError("padded particle slots must have zero weight")
        for track_id in np.flatnonzero(self.track_valid):
            count = int(self.support_counts[track_id])
            ids = self.particle_ids[track_id, :count]
            weights = self.particle_weights[track_id, :count]
            if count < 1 or np.any(ids < 0):
                raise ValueError("valid track has no material support")
            if (
                not np.isfinite(weights).all()
                or np.any(weights < 0.0)
                or not np.any(weights > 0.0)
            ):
                raise ValueError("valid track has invalid material weights")
            if not np.isclose(float(weights.sum()), 1.0, atol=2.0e-6):
                raise ValueError("valid track weights do not sum to one")
            if surface_mask is not None and not np.all(surface_mask[ids]):
                raise ValueError("range binding contains a non-surface particle")
            if fixed_mask is not None:
                movable = int(np.count_nonzero(~fixed_mask[ids]))
                if movable != int(self.movable_support_counts[track_id]):
                    raise ValueError("movable support count is inconsistent")

    def valid_weight_sums(self) -> np.ndarray:
        return self.particle_weights[self.track_valid].sum(axis=1)


@dataclass(frozen=True)
class FlowDepthObservationSettings:
    """Validity limits for one tracked flow/depth observation pair."""

    maximum_depth_sampling_radius_px: int = 2
    minimum_depth_m: float = 0.035
    maximum_depth_m: float = 0.250
    maximum_depth_change_m: float = 0.015
    maximum_observed_flow_m: float = 0.020

    def validate(self) -> None:
        if self.maximum_depth_sampling_radius_px < 0:
            raise ValueError("maximum depth sampling radius must be non-negative")
        if self.minimum_depth_m <= 0.0:
            raise ValueError("minimum depth must be positive")
        if self.maximum_depth_m <= self.minimum_depth_m:
            raise ValueError("maximum depth must be greater than minimum depth")
        if self.maximum_depth_change_m <= 0.0:
            raise ValueError("maximum depth change must be positive")
        if self.maximum_observed_flow_m <= 0.0:
            raise ValueError("maximum observed flow must be positive")


@dataclass(frozen=True)
class FlowDepthTrackObservation:
    """A causal 3D scene-flow observation for each fixed material track."""

    track_valid: np.ndarray
    confidence: np.ndarray
    current_depth_m: np.ndarray
    next_depth_m: np.ndarray
    current_depth_sampling_radius_px: np.ndarray
    next_depth_sampling_radius_px: np.ndarray
    current_points_table: np.ndarray
    next_points_table: np.ndarray
    observed_flow_table: np.ndarray

    def validate(self) -> None:
        track_count = len(self.track_valid)
        for name, values in {
            "confidence": self.confidence,
            "current_depth_m": self.current_depth_m,
            "next_depth_m": self.next_depth_m,
            "current_depth_sampling_radius_px": (
                self.current_depth_sampling_radius_px
            ),
            "next_depth_sampling_radius_px": self.next_depth_sampling_radius_px,
        }.items():
            if values.shape != (track_count,):
                raise ValueError(f"{name} must have one value per track")
        for name, values in {
            "current_points_table": self.current_points_table,
            "next_points_table": self.next_points_table,
            "observed_flow_table": self.observed_flow_table,
        }.items():
            if values.shape != (track_count, 3):
                raise ValueError(f"{name} must be shaped (tracks, 3)")
        if np.any(self.confidence < 0.0) or np.any(self.confidence > 1.0):
            raise ValueError("track confidence must stay in [0, 1]")
        valid = self.track_valid
        if not np.isfinite(self.observed_flow_table[valid]).all():
            raise ValueError("valid tracks must have finite observed flow")
        if np.any(self.confidence[~valid] != 0.0):
            raise ValueError("invalid tracks must have zero confidence")


@dataclass(frozen=True)
class FlowDepthStateUpdateSettings:
    """Bounded absolute-position/velocity observer after an XPBD prediction."""

    position_gain: float = 0.60
    velocity_gain: float = 0.30
    absolute_position_weight: float = 0.80
    solver_regularization: float = 0.02
    solver_iterations: int = 16
    robust_residual_scale_m: float = 0.020
    compliance_inverse_mass: float = 0.0
    maximum_position_correction_m: float = 0.020
    maximum_velocity_correction_m_s: float = 0.25

    def validate(self) -> None:
        if not 0.0 <= self.position_gain <= 1.0:
            raise ValueError("position gain must stay in [0, 1]")
        if not 0.0 <= self.velocity_gain <= 1.0:
            raise ValueError("velocity gain must stay in [0, 1]")
        if not 0.0 <= self.absolute_position_weight <= 1.0:
            raise ValueError("absolute-position weight must stay in [0, 1]")
        if self.solver_regularization <= 0.0:
            raise ValueError("solver regularization must be positive")
        if self.solver_iterations < 1:
            raise ValueError("solver iterations must be positive")
        if self.robust_residual_scale_m <= 0.0:
            raise ValueError("robust residual scale must be positive")
        if self.compliance_inverse_mass < 0.0:
            raise ValueError("flow-depth compliance must be non-negative")
        if self.maximum_position_correction_m <= 0.0:
            raise ValueError("maximum position correction must be positive")
        if self.maximum_velocity_correction_m_s <= 0.0:
            raise ValueError("maximum velocity correction must be positive")


@dataclass(frozen=True)
class FlowDepthParticleStateUpdate:
    """Non-mutating result ready for physical safety validation/writeback."""

    corrected_positions: np.ndarray
    corrected_velocities: np.ndarray
    position_correction: np.ndarray
    velocity_correction: np.ndarray
    predicted_track_flow: np.ndarray
    innovation: np.ndarray
    track_valid: np.ndarray
    particle_confidence_support: np.ndarray
    particle_track_support_count: np.ndarray

    def validate(self) -> None:
        particle_count = len(self.corrected_positions)
        if self.corrected_positions.shape != (particle_count, 3):
            raise ValueError("corrected positions must be shaped (particles, 3)")
        for name, values in {
            "corrected_velocities": self.corrected_velocities,
            "position_correction": self.position_correction,
            "velocity_correction": self.velocity_correction,
        }.items():
            if values.shape != (particle_count, 3):
                raise ValueError(f"{name} must be shaped (particles, 3)")
        track_count = len(self.track_valid)
        for name, values in {
            "predicted_track_flow": self.predicted_track_flow,
            "innovation": self.innovation,
        }.items():
            if values.shape != (track_count, 3):
                raise ValueError(f"{name} must be shaped (tracks, 3)")
        if self.particle_confidence_support.shape != (particle_count,):
            raise ValueError("particle confidence support has the wrong shape")
        if self.particle_track_support_count.shape != (particle_count,):
            raise ValueError("particle track support count has the wrong shape")
        if not np.isfinite(self.corrected_positions).all():
            raise ValueError("corrected positions contain non-finite values")
        if not np.isfinite(self.corrected_velocities).all():
            raise ValueError("corrected velocities contain non-finite values")


@dataclass(frozen=True)
class FlowDepthObservationSequence:
    """Compact precomputed observations indexed by their destination frame."""

    current_source_frames: np.ndarray
    next_source_frames: np.ndarray
    track_valid: np.ndarray
    confidence: np.ndarray
    observed_flow_table: np.ndarray
    current_points_table: np.ndarray
    next_points_table: np.ndarray
    current_depth_m: np.ndarray
    next_depth_m: np.ndarray

    def validate(self) -> None:
        pair_count = len(self.current_source_frames)
        if self.next_source_frames.shape != (pair_count,):
            raise ValueError("next source frames have the wrong shape")
        if np.any(self.next_source_frames <= self.current_source_frames):
            raise ValueError("flow-depth frame pairs must advance in time")
        if len(np.unique(self.next_source_frames)) != pair_count:
            raise ValueError("flow-depth destination frames must be unique")
        if self.track_valid.ndim != 2 or self.track_valid.shape[0] != pair_count:
            raise ValueError("track validity must be shaped (pairs, tracks)")
        pair_track_shape = self.track_valid.shape
        for name, values in {
            "confidence": self.confidence,
            "current_depth_m": self.current_depth_m,
            "next_depth_m": self.next_depth_m,
        }.items():
            if values.shape != pair_track_shape:
                raise ValueError(f"{name} must be shaped (pairs, tracks)")
        vector_shape = (*pair_track_shape, 3)
        for name, values in {
            "observed_flow_table": self.observed_flow_table,
            "current_points_table": self.current_points_table,
            "next_points_table": self.next_points_table,
        }.items():
            if values.shape != vector_shape:
                raise ValueError(f"{name} must be shaped (pairs, tracks, 3)")
        if np.any(self.confidence < 0.0) or np.any(self.confidence > 1.0):
            raise ValueError("sequence confidence must stay in [0, 1]")
        if not np.isfinite(self.observed_flow_table[self.track_valid]).all():
            raise ValueError("valid sequence flow must be finite")

    def pair_index_for_next_frame(self, frame_index: int) -> int | None:
        matches = np.flatnonzero(self.next_source_frames == int(frame_index))
        return None if not len(matches) else int(matches[0])

    def observation(self, pair_index: int) -> FlowDepthTrackObservation:
        if not 0 <= pair_index < len(self.current_source_frames):
            raise IndexError(pair_index)
        valid = self.track_valid[pair_index]
        radii = np.where(valid, 0, -1).astype(np.int16)
        result = FlowDepthTrackObservation(
            track_valid=valid.copy(),
            confidence=self.confidence[pair_index].copy(),
            current_depth_m=self.current_depth_m[pair_index].copy(),
            next_depth_m=self.next_depth_m[pair_index].copy(),
            current_depth_sampling_radius_px=radii,
            next_depth_sampling_radius_px=radii.copy(),
            current_points_table=self.current_points_table[pair_index].copy(),
            next_points_table=self.next_points_table[pair_index].copy(),
            observed_flow_table=self.observed_flow_table[pair_index].copy(),
        )
        result.validate()
        return result


def load_fixed_particle_range_bindings(
    path: str | Path,
) -> FlowDepthParticleRangeBindings:
    """Load the immutable binding artifact written by the SUPER diagnostic."""

    with np.load(Path(path), allow_pickle=False) as loaded:
        result = FlowDepthParticleRangeBindings(
            track_valid=loaded["track_valid"].astype(bool),
            particle_ids=loaded["particle_ids"].astype(np.int32),
            particle_weights=loaded["particle_weights"].astype(np.float32),
            support_counts=loaded["support_counts"].astype(np.int32),
            movable_support_counts=loaded["movable_support_counts"].astype(
                np.int32
            ),
            binding_radius_m=loaded["binding_radius_m"].astype(np.float32),
            initial_depth_m=loaded["initial_depth_m"].astype(np.float32),
            depth_sampling_radius_px=loaded[
                "depth_sampling_radius_px"
            ].astype(np.int16),
            initial_points_table=loaded["initial_points_table"].astype(
                np.float32
            ),
            nearest_surface_distance_m=loaded[
                "nearest_surface_distance_m"
            ].astype(np.float32),
            binding_method=(
                str(loaded["binding_method"].item())
                if "binding_method" in loaded.files
                else "particle_range"
            ),
            surface_face_ids=(
                loaded["surface_face_ids"].astype(np.int32)
                if "surface_face_ids" in loaded.files
                else None
            ),
            surface_projection_points_table=(
                loaded["surface_projection_points_table"].astype(np.float32)
                if "surface_projection_points_table" in loaded.files
                else None
            ),
        )
    result.validate()
    return result


def load_flow_depth_observation_sequence(
    path: str | Path,
) -> FlowDepthObservationSequence:
    """Load a causal precomputed observation sequence without simulator state."""

    with np.load(Path(path), allow_pickle=False) as loaded:
        result = FlowDepthObservationSequence(
            current_source_frames=loaded["current_source_frames"].astype(
                np.int32
            ),
            next_source_frames=loaded["next_source_frames"].astype(np.int32),
            track_valid=loaded["track_valid"].astype(bool),
            confidence=loaded["confidence"].astype(np.float32),
            observed_flow_table=loaded["observed_flow_table"].astype(np.float32),
            current_points_table=loaded["current_points_table"].astype(np.float32),
            next_points_table=loaded["next_points_table"].astype(np.float32),
            current_depth_m=loaded["current_depth_m"].astype(np.float32),
            next_depth_m=loaded["next_depth_m"].astype(np.float32),
        )
    result.validate()
    return result


def sample_depth_near_pixels(
    depth: np.ndarray,
    pixels_uv: np.ndarray,
    *,
    maximum_radius_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample robust finite depth, preferring the exact pixel.

    At each radius the median of finite positive values in the square window
    is returned.  The first radius with any valid sample wins.
    """

    depth = np.asarray(depth)
    pixels_uv = np.asarray(pixels_uv, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth must be a 2D array")
    if pixels_uv.ndim != 2 or pixels_uv.shape[1] != 2:
        raise ValueError("pixels must be shaped (tracks, 2)")
    if maximum_radius_px < 0:
        raise ValueError("maximum sampling radius must be non-negative")
    height, width = depth.shape
    values = np.full(len(pixels_uv), np.nan, dtype=np.float64)
    radii = np.full(len(pixels_uv), -1, dtype=np.int16)
    for track_id, pixel in enumerate(pixels_uv):
        if not np.isfinite(pixel).all():
            continue
        u = int(round(float(pixel[0])))
        v = int(round(float(pixel[1])))
        if not (0 <= u < width and 0 <= v < height):
            continue
        for radius in range(maximum_radius_px + 1):
            x0, x1 = max(0, u - radius), min(width, u + radius + 1)
            y0, y1 = max(0, v - radius), min(height, v + radius + 1)
            patch = np.asarray(depth[y0:y1, x0:x1], dtype=np.float64)
            finite = np.isfinite(patch) & (patch > 0.0)
            if finite.any():
                values[track_id] = float(np.median(patch[finite]))
                radii[track_id] = radius
                break
    return values, radii


def backproject_pixels(
    pixels_uv: np.ndarray,
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    x_table_camera: np.ndarray,
) -> np.ndarray:
    """Backproject rectified OpenCV pixels into the SUPER table frame."""

    pixels_uv = np.asarray(pixels_uv, dtype=np.float64)
    depth_m = np.asarray(depth_m, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    x_table_camera = np.asarray(x_table_camera, dtype=np.float64)
    if pixels_uv.ndim != 2 or pixels_uv.shape[1] != 2:
        raise ValueError("pixels must be shaped (tracks, 2)")
    if depth_m.shape != (len(pixels_uv),):
        raise ValueError("depth must have one value per pixel")
    if intrinsic.shape != (3, 3):
        raise ValueError("camera intrinsic must be 3x3")
    if x_table_camera.shape != (4, 4):
        raise ValueError("camera-to-table transform must be 4x4")
    z = depth_m
    camera = np.column_stack(
        (
            (pixels_uv[:, 0] - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (pixels_uv[:, 1] - intrinsic[1, 2]) * z / intrinsic[1, 1],
            z,
            np.ones(len(pixels_uv), dtype=np.float64),
        )
    )
    return (camera @ x_table_camera.T)[:, :3]


def observe_flow_depth_tracks(
    *,
    current_pixels_uv: np.ndarray,
    next_pixels_uv: np.ndarray,
    current_depth: np.ndarray,
    next_depth: np.ndarray,
    intrinsic: np.ndarray,
    x_table_camera: np.ndarray,
    current_track_valid: np.ndarray | None = None,
    next_track_valid: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    settings: FlowDepthObservationSettings | None = None,
) -> FlowDepthTrackObservation:
    """Lift two corresponding 2D track samples into causal 3D scene flow.

    Masks, tool occlusion and CoTracker visibility remain caller-owned signals.
    They enter through the two validity arrays and are intersected with depth,
    depth-change and maximum-flow checks here.
    """

    settings = settings or FlowDepthObservationSettings()
    settings.validate()
    current_pixels_uv = np.asarray(current_pixels_uv, dtype=np.float64)
    next_pixels_uv = np.asarray(next_pixels_uv, dtype=np.float64)
    if current_pixels_uv.ndim != 2 or current_pixels_uv.shape[1] != 2:
        raise ValueError("current pixels must be shaped (tracks, 2)")
    if next_pixels_uv.shape != current_pixels_uv.shape:
        raise ValueError("current and next pixels must have identical shape")
    track_count = len(current_pixels_uv)

    def validity(values: np.ndarray | None, name: str) -> np.ndarray:
        if values is None:
            return np.ones(track_count, dtype=bool)
        result = np.asarray(values, dtype=bool)
        if result.shape != (track_count,):
            raise ValueError(f"{name} must have one value per track")
        return result

    requested = validity(current_track_valid, "current validity") & validity(
        next_track_valid, "next validity"
    )
    if confidence is None:
        track_confidence = np.ones(track_count, dtype=np.float64)
    else:
        track_confidence = np.asarray(confidence, dtype=np.float64)
        if track_confidence.shape != (track_count,):
            raise ValueError("confidence must have one value per track")
        if np.any(~np.isfinite(track_confidence)):
            raise ValueError("confidence must be finite")
        if np.any(track_confidence < 0.0) or np.any(track_confidence > 1.0):
            raise ValueError("confidence must stay in [0, 1]")

    current_depth_m, current_radius = sample_depth_near_pixels(
        current_depth,
        current_pixels_uv,
        maximum_radius_px=settings.maximum_depth_sampling_radius_px,
    )
    next_depth_m, next_radius = sample_depth_near_pixels(
        next_depth,
        next_pixels_uv,
        maximum_radius_px=settings.maximum_depth_sampling_radius_px,
    )
    depth_valid = (
        np.isfinite(current_depth_m)
        & np.isfinite(next_depth_m)
        & (current_depth_m >= settings.minimum_depth_m)
        & (current_depth_m <= settings.maximum_depth_m)
        & (next_depth_m >= settings.minimum_depth_m)
        & (next_depth_m <= settings.maximum_depth_m)
        & (
            np.abs(next_depth_m - current_depth_m)
            <= settings.maximum_depth_change_m
        )
    )
    safe_current_depth = np.where(depth_valid, current_depth_m, 1.0)
    safe_next_depth = np.where(depth_valid, next_depth_m, 1.0)
    current_points = backproject_pixels(
        current_pixels_uv,
        safe_current_depth,
        intrinsic,
        x_table_camera,
    )
    next_points = backproject_pixels(
        next_pixels_uv,
        safe_next_depth,
        intrinsic,
        x_table_camera,
    )
    observed_flow = next_points - current_points
    finite_flow = np.isfinite(observed_flow).all(axis=1)
    flow_norm = np.linalg.norm(
        np.where(finite_flow[:, None], observed_flow, 0.0), axis=1
    )
    valid = (
        requested
        & depth_valid
        & finite_flow
        & (flow_norm <= settings.maximum_observed_flow_m)
        & (track_confidence > 0.0)
    )
    current_points[~valid] = np.nan
    next_points[~valid] = np.nan
    observed_flow[~valid] = 0.0
    track_confidence = np.where(valid, track_confidence, 0.0)
    result = FlowDepthTrackObservation(
        track_valid=valid,
        confidence=track_confidence.astype(np.float32),
        current_depth_m=current_depth_m.astype(np.float32),
        next_depth_m=next_depth_m.astype(np.float32),
        current_depth_sampling_radius_px=current_radius,
        next_depth_sampling_radius_px=next_radius,
        current_points_table=current_points.astype(np.float32),
        next_points_table=next_points.astype(np.float32),
        observed_flow_table=observed_flow.astype(np.float32),
    )
    result.validate()
    return result


def observe_flow_depth_track_samples(
    *,
    current_pixels_uv: np.ndarray,
    next_pixels_uv: np.ndarray,
    current_depth_m: np.ndarray,
    next_depth_m: np.ndarray,
    intrinsic: np.ndarray,
    x_table_camera: np.ndarray,
    current_track_valid: np.ndarray | None = None,
    next_track_valid: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    current_depth_sampling_radius_px: np.ndarray | None = None,
    next_depth_sampling_radius_px: np.ndarray | None = None,
    settings: FlowDepthObservationSettings | None = None,
) -> FlowDepthTrackObservation:
    """Lift already sampled, causally held depths into one 3D flow pair.

    This variant is intended for sparse stereo-depth schedules.  A caller may
    forward-hold the most recent depth sample between expensive stereo frames,
    while confidence records the sample age.  No future depth is interpolated.
    """

    settings = settings or FlowDepthObservationSettings()
    settings.validate()
    current_pixels_uv = np.asarray(current_pixels_uv, dtype=np.float64)
    next_pixels_uv = np.asarray(next_pixels_uv, dtype=np.float64)
    if current_pixels_uv.ndim != 2 or current_pixels_uv.shape[1] != 2:
        raise ValueError("current pixels must be shaped (tracks, 2)")
    if next_pixels_uv.shape != current_pixels_uv.shape:
        raise ValueError("current and next pixels must have identical shape")
    track_count = len(current_pixels_uv)

    def vector(values: np.ndarray, name: str, dtype: type) -> np.ndarray:
        result = np.asarray(values, dtype=dtype)
        if result.shape != (track_count,):
            raise ValueError(f"{name} must have one value per track")
        return result

    current_depth_m = vector(current_depth_m, "current depth", np.float64)
    next_depth_m = vector(next_depth_m, "next depth", np.float64)
    current_valid = (
        np.ones(track_count, dtype=bool)
        if current_track_valid is None
        else vector(current_track_valid, "current validity", bool)
    )
    next_valid = (
        np.ones(track_count, dtype=bool)
        if next_track_valid is None
        else vector(next_track_valid, "next validity", bool)
    )
    track_confidence = (
        np.ones(track_count, dtype=np.float64)
        if confidence is None
        else vector(confidence, "confidence", np.float64)
    )
    if np.any(~np.isfinite(track_confidence)):
        raise ValueError("confidence must be finite")
    if np.any(track_confidence < 0.0) or np.any(track_confidence > 1.0):
        raise ValueError("confidence must stay in [0, 1]")
    current_radius = (
        np.full(track_count, -1, dtype=np.int16)
        if current_depth_sampling_radius_px is None
        else vector(
            current_depth_sampling_radius_px,
            "current sampling radius",
            np.int16,
        )
    )
    next_radius = (
        np.full(track_count, -1, dtype=np.int16)
        if next_depth_sampling_radius_px is None
        else vector(
            next_depth_sampling_radius_px,
            "next sampling radius",
            np.int16,
        )
    )
    depth_valid = (
        np.isfinite(current_depth_m)
        & np.isfinite(next_depth_m)
        & (current_depth_m >= settings.minimum_depth_m)
        & (current_depth_m <= settings.maximum_depth_m)
        & (next_depth_m >= settings.minimum_depth_m)
        & (next_depth_m <= settings.maximum_depth_m)
        & (
            np.abs(next_depth_m - current_depth_m)
            <= settings.maximum_depth_change_m
        )
    )
    safe_current_depth = np.where(depth_valid, current_depth_m, 1.0)
    safe_next_depth = np.where(depth_valid, next_depth_m, 1.0)
    current_points = backproject_pixels(
        current_pixels_uv,
        safe_current_depth,
        intrinsic,
        x_table_camera,
    )
    next_points = backproject_pixels(
        next_pixels_uv,
        safe_next_depth,
        intrinsic,
        x_table_camera,
    )
    observed_flow = next_points - current_points
    finite_flow = np.isfinite(observed_flow).all(axis=1)
    flow_norm = np.linalg.norm(
        np.where(finite_flow[:, None], observed_flow, 0.0), axis=1
    )
    valid = (
        current_valid
        & next_valid
        & depth_valid
        & finite_flow
        & (flow_norm <= settings.maximum_observed_flow_m)
        & (track_confidence > 0.0)
    )
    current_points[~valid] = np.nan
    next_points[~valid] = np.nan
    observed_flow[~valid] = 0.0
    track_confidence = np.where(valid, track_confidence, 0.0)
    result = FlowDepthTrackObservation(
        track_valid=valid,
        confidence=track_confidence.astype(np.float32),
        current_depth_m=current_depth_m.astype(np.float32),
        next_depth_m=next_depth_m.astype(np.float32),
        current_depth_sampling_radius_px=current_radius,
        next_depth_sampling_radius_px=next_radius,
        current_points_table=current_points.astype(np.float32),
        next_points_table=next_points.astype(np.float32),
        observed_flow_table=observed_flow.astype(np.float32),
    )
    result.validate()
    return result


def fixed_range_centers(
    positions: np.ndarray,
    bindings: FlowDepthParticleRangeBindings,
) -> np.ndarray:
    """Evaluate every fixed track range center without reassociation."""

    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("particle positions must be shaped (particles, 3)")
    centers = np.full((len(bindings.track_valid), 3), np.nan, dtype=np.float64)
    for track_id in np.flatnonzero(bindings.track_valid):
        count = int(bindings.support_counts[track_id])
        ids = bindings.particle_ids[track_id, :count]
        if np.any(ids >= len(positions)):
            raise ValueError("range binding references a missing particle")
        weights = bindings.particle_weights[track_id, :count].astype(
            np.float64
        )
        centers[track_id] = np.sum(positions[ids] * weights[:, None], axis=0)
    return centers


def _clip_vector_norms(vectors: np.ndarray, maximum_norm: float) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    scales = np.minimum(1.0, maximum_norm / np.maximum(norms, 1.0e-12))
    return vectors * scales


def _apply_binding_linear_map(
    particle_vectors: np.ndarray,
    bindings: FlowDepthParticleRangeBindings,
    track_mask: np.ndarray,
) -> np.ndarray:
    """Apply the fixed track-range matrix ``A`` to particle vectors."""

    result = np.zeros((len(bindings.track_valid), 3), dtype=np.float64)
    for track_id in np.flatnonzero(track_mask):
        count = int(bindings.support_counts[track_id])
        ids = bindings.particle_ids[track_id, :count]
        weights = bindings.particle_weights[track_id, :count].astype(
            np.float64
        )
        result[track_id] = np.sum(
            particle_vectors[ids] * weights[:, None], axis=0
        )
    return result


def _apply_binding_transpose(
    track_vectors: np.ndarray,
    bindings: FlowDepthParticleRangeBindings,
    track_mask: np.ndarray,
    particle_count: int,
) -> np.ndarray:
    """Apply ``A.T`` without materialising a dense track/particle matrix."""

    result = np.zeros((particle_count, 3), dtype=np.float64)
    for track_id in np.flatnonzero(track_mask):
        count = int(bindings.support_counts[track_id])
        ids = bindings.particle_ids[track_id, :count]
        weights = bindings.particle_weights[track_id, :count].astype(
            np.float64
        )
        np.add.at(
            result,
            ids,
            weights[:, None] * track_vectors[track_id],
        )
    return result


def _solve_joint_binding_residual(
    *,
    bindings: FlowDepthParticleRangeBindings,
    track_mask: np.ndarray,
    track_residual: np.ndarray,
    track_confidence: np.ndarray,
    effective_inverse_mass: np.ndarray,
    regularization: float,
    iterations: int,
    robust_residual_scale_m: float,
) -> np.ndarray:
    """Solve all overlapping range constraints in one robust least square.

    The previous implementation solved every track independently and then
    averaged the particle updates.  Overlapping ranges therefore changed one
    another's centres, sometimes in the opposite direction.  Here ``A`` is the
    fixed range-weight matrix and the three coordinate systems solve

    ``argmin_d sum_j c_j ||A_j d-r_j||^2 + regularization ||d||^2``.

    The square-root inverse-mass change of variables preserves fixed particles
    while keeping the normal operator symmetric positive definite for CG.
    """

    particle_count = len(effective_inverse_mass)
    usable = np.asarray(track_mask, dtype=bool).copy()
    for track_id in np.flatnonzero(usable):
        count = int(bindings.support_counts[track_id])
        ids = bindings.particle_ids[track_id, :count]
        if not np.any(effective_inverse_mass[ids] > 0.0):
            usable[track_id] = False
    if not np.any(usable):
        return np.zeros((particle_count, 3), dtype=np.float64)

    positive_mass = effective_inverse_mass > 0.0
    mobility = np.zeros(particle_count, dtype=np.float64)
    reference_mobility = float(
        np.median(effective_inverse_mass[positive_mass])
    )
    mobility[positive_mass] = np.clip(
        effective_inverse_mass[positive_mass]
        / max(reference_mobility, 1.0e-15),
        0.25,
        4.0,
    )
    sqrt_mobility = np.sqrt(mobility)

    residual_norm = np.linalg.norm(track_residual, axis=1)
    robust_weight = np.minimum(
        1.0,
        robust_residual_scale_m / np.maximum(residual_norm, 1.0e-12),
    )
    weights = np.where(
        usable,
        np.asarray(track_confidence, dtype=np.float64) * robust_weight,
        0.0,
    )
    usable &= weights > 0.0
    if not np.any(usable):
        return np.zeros((particle_count, 3), dtype=np.float64)

    # Confidence ranks competing constraints, but does not attenuate an
    # isolated material correspondence.  With AllTracker's direct one-particle
    # bindings, confidence on both sides of the normal equation therefore
    # cancels (apart from the small Tikhonov term), allowing a reliable unique
    # observation to reach its absolute target instead of applying c^2 twice.
    weighted_residual = weights[:, None] * track_residual
    rhs = sqrt_mobility[:, None] * _apply_binding_transpose(
        weighted_residual,
        bindings,
        usable,
        particle_count,
    )

    def normal_operator(values: np.ndarray) -> np.ndarray:
        particle_values = sqrt_mobility[:, None] * values
        track_values = _apply_binding_linear_map(
            particle_values, bindings, usable
        )
        scattered = _apply_binding_transpose(
            weights[:, None] * track_values,
            bindings,
            usable,
            particle_count,
        )
        return sqrt_mobility[:, None] * scattered + regularization * values

    solution = np.zeros_like(rhs)
    conjugate_residual = rhs.copy()
    direction = conjugate_residual.copy()
    squared_residual = np.sum(conjugate_residual * conjugate_residual, axis=0)
    for _ in range(iterations):
        if np.all(squared_residual <= 1.0e-24):
            break
        normal_direction = normal_operator(direction)
        denominator = np.sum(direction * normal_direction, axis=0)
        alpha = np.divide(
            squared_residual,
            denominator,
            out=np.zeros(3, dtype=np.float64),
            where=np.abs(denominator) > 1.0e-24,
        )
        solution += direction * alpha[None, :]
        conjugate_residual -= normal_direction * alpha[None, :]
        next_squared_residual = np.sum(
            conjugate_residual * conjugate_residual, axis=0
        )
        beta = np.divide(
            next_squared_residual,
            squared_residual,
            out=np.zeros(3, dtype=np.float64),
            where=squared_residual > 1.0e-24,
        )
        direction = conjugate_residual + direction * beta[None, :]
        squared_residual = next_squared_residual

    correction = sqrt_mobility[:, None] * solution
    correction[~positive_mass] = 0.0
    return correction


def compute_flow_depth_particle_state_update(
    *,
    bindings: FlowDepthParticleRangeBindings,
    observation: FlowDepthTrackObservation,
    current_positions: np.ndarray,
    predicted_positions: np.ndarray,
    predicted_velocities: np.ndarray,
    particle_inverse_masses: np.ndarray,
    observation_dt_s: float,
    reference_range_centers: np.ndarray | None = None,
    dynamic_exclusion_mask: np.ndarray | None = None,
    settings: FlowDepthStateUpdateSettings | None = None,
) -> FlowDepthParticleStateUpdate:
    """Fuse absolute observed targets with flow and jointly solve particle q/qd.

    ``reference_range_centers`` fixes the material-frame centres at the query
    frame.  The observed absolute target is the reference centre plus the
    track's displacement from its initial 3D point.  This closes accumulated
    position error; a flow-only observer cannot do so when XPBD has the right
    local velocity but the wrong absolute state.
    """

    settings = settings or FlowDepthStateUpdateSettings()
    settings.validate()
    observation.validate()
    if observation_dt_s <= 0.0:
        raise ValueError("observation dt must be positive")
    current_positions = np.asarray(current_positions, dtype=np.float64)
    predicted_positions = np.asarray(predicted_positions, dtype=np.float64)
    predicted_velocities = np.asarray(predicted_velocities, dtype=np.float64)
    if current_positions.ndim != 2 or current_positions.shape[1] != 3:
        raise ValueError("current positions must be shaped (particles, 3)")
    particle_count = len(current_positions)
    if predicted_positions.shape != (particle_count, 3):
        raise ValueError("predicted positions disagree with current positions")
    if predicted_velocities.shape != (particle_count, 3):
        raise ValueError("predicted velocities disagree with current positions")
    inverse_masses = np.asarray(particle_inverse_masses, dtype=np.float64)
    if inverse_masses.shape != (particle_count,):
        raise ValueError("inverse masses must have one value per particle")
    if np.any(~np.isfinite(inverse_masses)) or np.any(inverse_masses < 0.0):
        raise ValueError("inverse masses must be finite and non-negative")
    if dynamic_exclusion_mask is None:
        exclusion = np.zeros(particle_count, dtype=bool)
    else:
        exclusion = np.asarray(dynamic_exclusion_mask, dtype=bool)
        if exclusion.shape != (particle_count,):
            raise ValueError("dynamic exclusion mask has the wrong shape")
    effective_inverse_mass = inverse_masses.copy()
    effective_inverse_mass[exclusion] = 0.0

    current_centers = fixed_range_centers(current_positions, bindings)
    predicted_centers = fixed_range_centers(predicted_positions, bindings)
    predicted_track_flow = predicted_centers - current_centers
    track_valid = bindings.track_valid & observation.track_valid
    if reference_range_centers is None:
        # Backward-compatible stateless fallback.  Runtime evaluation passes a
        # fixed query-frame reference, while isolated callers retain a valid
        # flow observer instead of accidentally anchoring to an unknown state.
        reference_centers = current_centers - (
            observation.current_points_table - bindings.initial_points_table
        )
    else:
        reference_centers = np.asarray(
            reference_range_centers, dtype=np.float64
        )
        if reference_centers.shape != predicted_centers.shape:
            raise ValueError("reference range centres have the wrong shape")
    observed_target_centers = reference_centers + (
        observation.next_points_table - bindings.initial_points_table
    )
    absolute_innovation = np.zeros_like(predicted_track_flow)
    absolute_innovation[track_valid] = (
        observed_target_centers[track_valid]
        - predicted_centers[track_valid]
    )
    flow_innovation = np.zeros_like(predicted_track_flow)
    flow_innovation[track_valid] = (
        observation.observed_flow_table[track_valid]
        - predicted_track_flow[track_valid]
    )
    innovation = np.zeros_like(predicted_track_flow)
    innovation[track_valid] = (
        settings.absolute_position_weight
        * absolute_innovation[track_valid]
        + (1.0 - settings.absolute_position_weight)
        * flow_innovation[track_valid]
    )

    confidence_support = np.zeros(particle_count, dtype=np.float64)
    track_support_count = np.zeros(particle_count, dtype=np.int32)
    usable_track = track_valid.copy()
    for track_id in np.flatnonzero(track_valid):
        count = int(bindings.support_counts[track_id])
        ids = bindings.particle_ids[track_id, :count]
        masses = effective_inverse_mass[ids]
        if not np.any(masses > 0.0):
            usable_track[track_id] = False
            innovation[track_id] = 0.0
            continue
        confidence = float(observation.confidence[track_id])
        movable = masses > 0.0
        confidence_support[ids[movable]] += confidence
        track_support_count[ids[movable]] += 1

    joint_position = _solve_joint_binding_residual(
        bindings=bindings,
        track_mask=usable_track,
        track_residual=innovation,
        track_confidence=observation.confidence,
        effective_inverse_mass=effective_inverse_mass,
        regularization=(
            settings.solver_regularization
            + settings.compliance_inverse_mass
        ),
        iterations=settings.solver_iterations,
        robust_residual_scale_m=settings.robust_residual_scale_m,
    )
    position_correction = settings.position_gain * joint_position
    position_correction = _clip_vector_norms(
        position_correction,
        settings.maximum_position_correction_m,
    )
    position_correction[effective_inverse_mass <= 0.0] = 0.0

    predicted_track_velocity = fixed_range_centers(
        predicted_velocities, bindings
    )
    velocity_innovation = np.zeros_like(predicted_track_velocity)
    velocity_innovation[usable_track] = (
        observation.observed_flow_table[usable_track] / observation_dt_s
        - predicted_track_velocity[usable_track]
    )
    joint_velocity = _solve_joint_binding_residual(
        bindings=bindings,
        track_mask=usable_track,
        track_residual=velocity_innovation,
        track_confidence=observation.confidence,
        effective_inverse_mass=effective_inverse_mass,
        regularization=(
            settings.solver_regularization
            + settings.compliance_inverse_mass
        ),
        iterations=settings.solver_iterations,
        robust_residual_scale_m=(
            settings.robust_residual_scale_m / observation_dt_s
        ),
    )
    velocity_correction = settings.velocity_gain * joint_velocity
    velocity_correction = _clip_vector_norms(
        velocity_correction,
        settings.maximum_velocity_correction_m_s,
    )
    velocity_correction[effective_inverse_mass <= 0.0] = 0.0
    result = FlowDepthParticleStateUpdate(
        corrected_positions=(predicted_positions + position_correction).astype(
            np.float32
        ),
        corrected_velocities=(
            predicted_velocities + velocity_correction
        ).astype(np.float32),
        position_correction=position_correction.astype(np.float32),
        velocity_correction=velocity_correction.astype(np.float32),
        predicted_track_flow=predicted_track_flow.astype(np.float32),
        innovation=innovation.astype(np.float32),
        track_valid=usable_track,
        particle_confidence_support=confidence_support.astype(np.float32),
        particle_track_support_count=track_support_count,
    )
    result.validate()
    return result


def build_fixed_particle_range_bindings(
    *,
    initial_pixels_uv: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    x_table_camera: np.ndarray,
    rest_positions_table: np.ndarray,
    surface_mask: np.ndarray,
    fixed_mask: np.ndarray,
    initial_track_valid: np.ndarray | None = None,
    settings: FlowDepthRangeBindingSettings | None = None,
) -> FlowDepthParticleRangeBindings:
    """Create fixed Gaussian-distance particle ranges for initial tracks."""

    settings = settings or FlowDepthRangeBindingSettings()
    settings.validate()
    initial_pixels_uv = np.asarray(initial_pixels_uv, dtype=np.float64)
    rest_positions_table = np.asarray(rest_positions_table, dtype=np.float64)
    surface_mask = np.asarray(surface_mask, dtype=bool)
    fixed_mask = np.asarray(fixed_mask, dtype=bool)
    if initial_pixels_uv.ndim != 2 or initial_pixels_uv.shape[1] != 2:
        raise ValueError("initial pixels must be shaped (tracks, 2)")
    if rest_positions_table.ndim != 2 or rest_positions_table.shape[1] != 3:
        raise ValueError("rest positions must be shaped (particles, 3)")
    particle_count = len(rest_positions_table)
    if surface_mask.shape != (particle_count,) or fixed_mask.shape != (
        particle_count,
    ):
        raise ValueError("surface/fixed masks must have one value per particle")
    if initial_track_valid is None:
        requested = np.ones(len(initial_pixels_uv), dtype=bool)
    else:
        requested = np.asarray(initial_track_valid, dtype=bool)
        if requested.shape != (len(initial_pixels_uv),):
            raise ValueError("initial validity must have one value per track")

    sampled_depth, sampling_radius = sample_depth_near_pixels(
        depth,
        initial_pixels_uv,
        maximum_radius_px=settings.maximum_depth_sampling_radius_px,
    )
    finite_depth = np.isfinite(sampled_depth) & (sampled_depth > 0.0)
    backproject_depth = sampled_depth.copy()
    backproject_depth[~finite_depth] = 1.0
    initial_points = backproject_pixels(
        initial_pixels_uv,
        backproject_depth,
        intrinsic,
        x_table_camera,
    )
    initial_points[~finite_depth] = np.nan

    surface_ids = np.flatnonzero(surface_mask)
    if not len(surface_ids):
        raise ValueError("physical asset has no surface particles")
    surface_positions = rest_positions_table[surface_ids]
    valid = requested & finite_depth & np.isfinite(initial_points).all(axis=1)
    selected_ids: list[np.ndarray] = []
    selected_weights: list[np.ndarray] = []
    support_counts = np.zeros(len(initial_pixels_uv), dtype=np.int32)
    movable_counts = np.zeros(len(initial_pixels_uv), dtype=np.int32)
    used_radius = np.full(len(initial_pixels_uv), np.nan, dtype=np.float64)
    nearest_distance = np.full(len(initial_pixels_uv), np.nan, dtype=np.float64)

    for track_id in range(len(initial_pixels_uv)):
        if not valid[track_id]:
            selected_ids.append(np.empty(0, dtype=np.int32))
            selected_weights.append(np.empty(0, dtype=np.float64))
            continue
        distances = np.linalg.norm(
            surface_positions - initial_points[track_id][None, :],
            axis=1,
        )
        nearest_distance[track_id] = float(distances.min())
        chosen = np.flatnonzero(distances <= settings.radius_m)
        radius = settings.radius_m
        movable = int(np.count_nonzero(~fixed_mask[surface_ids[chosen]]))
        if movable < settings.minimum_movable_particles:
            chosen = np.flatnonzero(distances <= settings.fallback_radius_m)
            radius = settings.fallback_radius_m
            movable = int(np.count_nonzero(~fixed_mask[surface_ids[chosen]]))
        if movable < settings.minimum_movable_particles:
            valid[track_id] = False
            selected_ids.append(np.empty(0, dtype=np.int32))
            selected_weights.append(np.empty(0, dtype=np.float64))
            continue
        ids = surface_ids[chosen]
        chosen_distances = distances[chosen]
        order = np.lexsort((ids, chosen_distances))
        ids = ids[order].astype(np.int32, copy=False)
        chosen_distances = chosen_distances[order]
        unnormalized = np.exp(
            -(chosen_distances * chosen_distances)
            / (2.0 * settings.sigma_m * settings.sigma_m)
        )
        weights = unnormalized / unnormalized.sum()
        selected_ids.append(ids)
        selected_weights.append(weights)
        support_counts[track_id] = len(ids)
        movable_counts[track_id] = movable
        used_radius[track_id] = radius

    maximum_support = max((len(ids) for ids in selected_ids), default=0)
    padded_ids = np.full(
        (len(initial_pixels_uv), maximum_support), -1, dtype=np.int32
    )
    padded_weights = np.zeros_like(padded_ids, dtype=np.float32)
    for track_id, (ids, weights) in enumerate(
        zip(selected_ids, selected_weights, strict=True)
    ):
        padded_ids[track_id, : len(ids)] = ids
        padded_weights[track_id, : len(weights)] = weights.astype(np.float32)

    bindings = FlowDepthParticleRangeBindings(
        track_valid=valid,
        particle_ids=padded_ids,
        particle_weights=padded_weights,
        support_counts=support_counts,
        movable_support_counts=movable_counts,
        binding_radius_m=used_radius.astype(np.float32),
        initial_depth_m=sampled_depth.astype(np.float32),
        depth_sampling_radius_px=sampling_radius,
        initial_points_table=initial_points.astype(np.float32),
        nearest_surface_distance_m=nearest_distance.astype(np.float32),
    )
    bindings.validate(surface_mask=surface_mask, fixed_mask=fixed_mask)
    return bindings


def build_fixed_particle_triangle_bindings(
    *,
    initial_pixels_uv: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    x_table_camera: np.ndarray,
    rest_positions_table: np.ndarray,
    surface_faces: np.ndarray,
    surface_mask: np.ndarray,
    fixed_mask: np.ndarray,
    initial_track_valid: np.ndarray | None = None,
    settings: FlowDepthTriangleBindingSettings | None = None,
) -> FlowDepthParticleRangeBindings:
    """Project each lifted track onto one physical face and bind barycentrically.

    Association is performed once in the reference frame.  The three physical
    vertex ids and convex barycentric weights then remain fixed for the entire
    sequence, so a track cannot slide to a neighbouring material patch.
    """

    settings = settings or FlowDepthTriangleBindingSettings()
    settings.validate()
    initial_pixels_uv = np.asarray(initial_pixels_uv, dtype=np.float64)
    rest_positions_table = np.asarray(rest_positions_table, dtype=np.float64)
    surface_faces = np.asarray(surface_faces, dtype=np.int32)
    surface_mask = np.asarray(surface_mask, dtype=bool)
    fixed_mask = np.asarray(fixed_mask, dtype=bool)
    if initial_pixels_uv.ndim != 2 or initial_pixels_uv.shape[1] != 2:
        raise ValueError("initial pixels must be shaped (tracks, 2)")
    if rest_positions_table.ndim != 2 or rest_positions_table.shape[1] != 3:
        raise ValueError("rest positions must be shaped (particles, 3)")
    particle_count = len(rest_positions_table)
    if surface_mask.shape != (particle_count,) or fixed_mask.shape != (
        particle_count,
    ):
        raise ValueError("surface/fixed masks must have one value per particle")
    if surface_faces.ndim != 2 or surface_faces.shape[1] != 3:
        raise ValueError("surface faces must be shaped (faces, 3)")
    if not len(surface_faces):
        raise ValueError("physical asset has no surface triangles")
    if np.any(surface_faces < 0) or np.any(surface_faces >= particle_count):
        raise ValueError("surface face contains an out-of-range particle id")
    if np.any(
        (surface_faces[:, 0] == surface_faces[:, 1])
        | (surface_faces[:, 1] == surface_faces[:, 2])
        | (surface_faces[:, 2] == surface_faces[:, 0])
    ):
        raise ValueError("surface mesh contains a degenerate index triangle")
    if initial_track_valid is None:
        requested = np.ones(len(initial_pixels_uv), dtype=bool)
    else:
        requested = np.asarray(initial_track_valid, dtype=bool)
        if requested.shape != (len(initial_pixels_uv),):
            raise ValueError("initial validity must have one value per track")

    sampled_depth, sampling_radius = sample_depth_near_pixels(
        depth,
        initial_pixels_uv,
        maximum_radius_px=settings.maximum_depth_sampling_radius_px,
    )
    finite_depth = np.isfinite(sampled_depth) & (sampled_depth > 0.0)
    backproject_depth = sampled_depth.copy()
    backproject_depth[~finite_depth] = 1.0
    initial_points = backproject_pixels(
        initial_pixels_uv,
        backproject_depth,
        intrinsic,
        x_table_camera,
    )
    initial_points[~finite_depth] = np.nan

    face_surface = np.all(surface_mask[surface_faces], axis=1)
    face_movable_counts = np.count_nonzero(
        ~fixed_mask[surface_faces], axis=1
    )
    eligible = face_surface & (
        face_movable_counts >= settings.minimum_movable_vertices
    )
    eligible_face_ids = np.flatnonzero(eligible)
    if not len(eligible_face_ids):
        raise ValueError("physical asset has no eligible movable surface triangle")
    eligible_faces = surface_faces[eligible_face_ids]

    # Open3D's tensor raycasting scene returns the exact closest point on the
    # triangle (including edge/vertex regions) plus primitive barycentric UV.
    # Keep the dependency local so loading an already-built artifact remains a
    # lightweight NumPy-only operation.
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(rest_positions_table),
        o3d.utility.Vector3iVector(eligible_faces),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    track_count = len(initial_pixels_uv)
    valid = requested & finite_depth & np.isfinite(initial_points).all(axis=1)
    candidate_track_ids = np.flatnonzero(valid)
    particle_ids = np.full((track_count, 3), -1, dtype=np.int32)
    particle_weights = np.zeros((track_count, 3), dtype=np.float32)
    support_counts = np.zeros(track_count, dtype=np.int32)
    movable_counts = np.zeros(track_count, dtype=np.int32)
    used_distance_limit = np.full(track_count, np.nan, dtype=np.float32)
    nearest_distance = np.full(track_count, np.nan, dtype=np.float32)
    face_ids = np.full(track_count, -1, dtype=np.int32)
    projected_points = np.full((track_count, 3), np.nan, dtype=np.float32)

    if len(candidate_track_ids):
        closest = scene.compute_closest_points(
            o3d.core.Tensor(
                initial_points[candidate_track_ids].astype(np.float32)
            )
        )
        closest_points = closest["points"].numpy().astype(np.float64)
        local_face_ids = closest["primitive_ids"].numpy().astype(np.int64)
        primitive_uvs = closest["primitive_uvs"].numpy().astype(np.float64)
        candidate_face_ids = eligible_face_ids[local_face_ids]
        candidate_faces = surface_faces[candidate_face_ids]
        barycentric = np.column_stack(
            (
                1.0 - primitive_uvs[:, 0] - primitive_uvs[:, 1],
                primitive_uvs[:, 0],
                primitive_uvs[:, 1],
            )
        )
        barycentric = np.clip(barycentric, 0.0, 1.0)
        barycentric /= np.maximum(
            barycentric.sum(axis=1, keepdims=True), 1.0e-12
        )
        distances = np.linalg.norm(
            initial_points[candidate_track_ids] - closest_points, axis=1
        )
        accepted = (
            distances <= settings.fallback_maximum_surface_distance_m
        )
        rejected_track_ids = candidate_track_ids[~accepted]
        valid[rejected_track_ids] = False
        nearest_distance[candidate_track_ids] = distances.astype(np.float32)
        face_ids[candidate_track_ids] = candidate_face_ids.astype(np.int32)
        projected_points[candidate_track_ids] = closest_points.astype(np.float32)

        accepted_track_ids = candidate_track_ids[accepted]
        accepted_faces = candidate_faces[accepted]
        accepted_weights = barycentric[accepted]
        particle_ids[accepted_track_ids] = accepted_faces
        particle_weights[accepted_track_ids] = accepted_weights.astype(np.float32)
        support_counts[accepted_track_ids] = 3
        movable_counts[accepted_track_ids] = np.count_nonzero(
            ~fixed_mask[accepted_faces], axis=1
        ).astype(np.int32)
        accepted_distances = distances[accepted]
        used_distance_limit[accepted_track_ids] = np.where(
            accepted_distances
            <= settings.primary_maximum_surface_distance_m,
            settings.primary_maximum_surface_distance_m,
            settings.fallback_maximum_surface_distance_m,
        ).astype(np.float32)

    bindings = FlowDepthParticleRangeBindings(
        track_valid=valid,
        particle_ids=particle_ids,
        particle_weights=particle_weights,
        support_counts=support_counts,
        movable_support_counts=movable_counts,
        binding_radius_m=used_distance_limit,
        initial_depth_m=sampled_depth.astype(np.float32),
        depth_sampling_radius_px=sampling_radius,
        initial_points_table=initial_points.astype(np.float32),
        nearest_surface_distance_m=nearest_distance,
        binding_method="surface_triangle_barycentric",
        surface_face_ids=face_ids,
        surface_projection_points_table=projected_points,
    )
    bindings.validate(surface_mask=surface_mask, fixed_mask=fixed_mask)
    return bindings
