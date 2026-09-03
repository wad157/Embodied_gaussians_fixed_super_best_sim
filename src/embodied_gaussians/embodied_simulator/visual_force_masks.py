from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from embodied_gaussians.embodied_simulator.frames import Frames


def _smoothstep01(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def tool_conditioned_tissue_weights(
    tissue_mask: np.ndarray,
    instrument_mask: np.ndarray,
    *,
    interaction_mask: np.ndarray | None = None,
    tool_near_radius_px: float = 120.0,
    tool_far_radius_px: float = 360.0,
    tool_falloff_power: float = 2.0,
    tissue_edge_zero_px: float = 24.0,
    tissue_edge_full_px: float = 64.0,
    tool_occlusion_radius_px: float = 6.0,
    image_border_zero_px: float = 48.0,
    image_border_full_px: float = 96.0,
    posterior_full_reach_px: float | None = 140.0,
    posterior_zero_reach_px: float | None = 280.0,
) -> np.ndarray:
    """Return a broad tool-local tissue weight with protected image regions.

    Radii are expressed at ``tissue_mask`` resolution.  The distance fields
    are evaluated at quarter image resolution and the resulting smooth field
    is resized once.  This preserves the broad 24--360 pixel bands without
    adding two full-HD distance transforms to every decoded video frame.

    For the fixed SUPER endoscope view, negative image-y from the distal/jaw
    mask points toward the anatomically posterior (far-camera) tissue.  The
    posterior field therefore remains full for ``posterior_full_reach_px``
    above the median interaction-mask row, fades smoothly, and is exactly zero
    at ``posterior_zero_reach_px``.  Other directions keep the broad isotropic
    tool SDF so local shape correction is not globally narrowed.
    """
    tissue_mask = np.asarray(tissue_mask, dtype=bool)
    instrument_mask = np.asarray(instrument_mask, dtype=bool)
    interaction_mask = (
        instrument_mask
        if interaction_mask is None
        else np.asarray(interaction_mask, dtype=bool)
    )
    if tissue_mask.ndim != 2 or instrument_mask.ndim != 2:
        raise ValueError("Tissue and instrument masks must both be 2D")
    if interaction_mask.shape != instrument_mask.shape:
        raise ValueError("Interaction and full instrument masks must have one shape")
    if tool_near_radius_px < 0.0 or tool_far_radius_px <= tool_near_radius_px:
        raise ValueError("Tool distance radii must satisfy 0 <= near < far")
    if not np.isfinite(tool_falloff_power) or tool_falloff_power <= 0.0:
        raise ValueError("Tool falloff power must be finite and positive")
    if tissue_edge_zero_px < 0.0 or tissue_edge_full_px <= tissue_edge_zero_px:
        raise ValueError("Tissue edge radii must satisfy 0 <= zero < full")
    if tool_occlusion_radius_px < 0.0:
        raise ValueError("Tool occlusion radius must be non-negative")
    if image_border_zero_px < 0.0 or image_border_full_px <= image_border_zero_px:
        raise ValueError("Image border radii must satisfy 0 <= zero < full")
    posterior_disabled = (
        posterior_full_reach_px is None and posterior_zero_reach_px is None
    )
    if not posterior_disabled and (
        posterior_full_reach_px is None
        or posterior_zero_reach_px is None
        or posterior_full_reach_px < 0.0
        or posterior_zero_reach_px <= posterior_full_reach_px
    ):
        raise ValueError(
            "Posterior reaches must both be None or satisfy 0 <= full < zero"
        )

    full_height, full_width = tissue_mask.shape
    instrument_height, instrument_width = instrument_mask.shape
    instrument_scale_x = instrument_width / float(full_width)
    instrument_scale_y = instrument_height / float(full_height)
    if not np.isclose(
        instrument_scale_x, instrument_scale_y, rtol=0.0, atol=1.0e-6
    ):
        raise ValueError("Tissue and instrument mask aspect ratios disagree")
    field_scale = min(0.25, instrument_scale_x)
    field_width = max(1, int(round(full_width * field_scale)))
    field_height = max(1, int(round(full_height * field_scale)))
    tissue_field = cv2.resize(
        tissue_mask.astype(np.uint8),
        (field_width, field_height),
        interpolation=cv2.INTER_NEAREST,
    )
    instrument_field = cv2.resize(
        instrument_mask.astype(np.uint8),
        (field_width, field_height),
        interpolation=cv2.INTER_AREA,
    ) > 0
    interaction_field = cv2.resize(
        interaction_mask.astype(np.uint8),
        (field_width, field_height),
        interpolation=cv2.INTER_AREA,
    ) > 0

    tool_distance = cv2.distanceTransform(
        (~interaction_field).astype(np.uint8), cv2.DIST_L2, 5
    )
    instrument_distance = cv2.distanceTransform(
        (~instrument_field).astype(np.uint8), cv2.DIST_L2, 5
    )
    tissue_edge_distance = cv2.distanceTransform(
        tissue_field, cv2.DIST_L2, 5
    )
    tool_near = float(tool_near_radius_px) * field_scale
    tool_far = float(tool_far_radius_px) * field_scale
    edge_zero = float(tissue_edge_zero_px) * field_scale
    edge_full = float(tissue_edge_full_px) * field_scale
    occlusion = float(tool_occlusion_radius_px) * field_scale
    image_border_zero = float(image_border_zero_px) * field_scale
    image_border_full = float(image_border_full_px) * field_scale

    tool_falloff = _smoothstep01(
        (tool_far - tool_distance) / max(tool_far - tool_near, 1.0e-6)
    )
    tool_falloff = np.power(
        tool_falloff, float(tool_falloff_power), dtype=np.float32
    )
    tool_falloff[tool_distance <= occlusion] = 0.0
    edge_confidence = _smoothstep01(
        (tissue_edge_distance - edge_zero)
        / max(edge_full - edge_zero, 1.0e-6)
    )
    yy, xx = np.ogrid[:field_height, :field_width]
    image_border_distance = np.minimum.reduce(
        (
            np.broadcast_to(xx, (field_height, field_width)),
            np.broadcast_to(yy, (field_height, field_width)),
            np.broadcast_to(field_width - 1 - xx, (field_height, field_width)),
            np.broadcast_to(field_height - 1 - yy, (field_height, field_width)),
        )
    ).astype(np.float32)
    image_border_confidence = _smoothstep01(
        (image_border_distance - image_border_zero)
        / max(image_border_full - image_border_zero, 1.0e-6)
    )
    posterior_confidence = np.ones_like(tool_falloff, dtype=np.float32)
    posterior_reference_y_full: float | None = None
    if not posterior_disabled:
        interaction_rows = np.nonzero(interaction_field)[0]
        if len(interaction_rows) > 0:
            posterior_reference_y_field = float(np.median(interaction_rows))
            posterior_reference_y_full = (
                float(np.median(np.nonzero(interaction_mask)[0]))
                / instrument_scale_y
            )
            posterior_distance = np.maximum(
                posterior_reference_y_field
                - np.arange(field_height, dtype=np.float32)[:, None],
                0.0,
            )
            posterior_full = float(posterior_full_reach_px) * field_scale
            posterior_zero = float(posterior_zero_reach_px) * field_scale
            posterior_confidence = 1.0 - _smoothstep01(
                (posterior_distance - posterior_full)
                / max(posterior_zero - posterior_full, 1.0e-6)
            )
        else:
            posterior_confidence.fill(0.0)
    weights_field = (
        tissue_field.astype(np.float32)
        * tool_falloff.astype(np.float32)
        * edge_confidence.astype(np.float32)
        * image_border_confidence.astype(np.float32)
        * posterior_confidence.astype(np.float32)
    )
    weights = cv2.resize(
        weights_field,
        (full_width, full_height),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)

    # Re-apply exact binary support after interpolation.  This guarantees that
    # neither the background nor the instrument/occlusion region receives a
    # small nonzero value from bilinear resizing.
    occlusion_full = cv2.resize(
        (instrument_distance <= occlusion).astype(np.uint8),
        (full_width, full_height),
        interpolation=cv2.INTER_NEAREST,
    )
    weights[~tissue_mask] = 0.0
    weights[occlusion_full.astype(bool)] = 0.0
    if posterior_reference_y_full is not None:
        posterior_zero_rows = (
            posterior_reference_y_full
            - np.arange(full_height, dtype=np.float32)
            >= float(posterior_zero_reach_px)
        )
        weights[posterior_zero_rows, :] = 0.0
    return np.clip(weights, 0.0, 1.0)


class PackedStereoInstrumentMasks:
    """Exact camera-frame lookup for the existing packed stereo PSM masks."""

    _SIDE_BY_CAMERA = {"stereo_left": "left", "stereo_right": "right"}

    def __init__(self, asset: Path | str, maximum_frame_gap: int = 1):
        self.asset = Path(asset)
        if maximum_frame_gap < 0:
            raise ValueError("maximum_frame_gap must be non-negative")
        self.maximum_frame_gap = int(maximum_frame_gap)
        with np.load(self.asset, allow_pickle=False) as archive:
            self.mask_shape = tuple(int(value) for value in archive["mask_shape"])
            self.bitorder = str(archive["bitorder"].item())
            self.quality_valid = np.asarray(archive["quality_valid"], dtype=bool)
            self.packed_by_camera = {
                camera: np.asarray(
                    archive[f"{side}_masks_packbits"], dtype=np.uint8
                )
                for camera, side in self._SIDE_BY_CAMERA.items()
            }
            self.packed_interaction_by_camera = {
                camera: np.asarray(
                    archive[f"{side}_distal_masks_packbits"], dtype=np.uint8
                )
                for camera, side in self._SIDE_BY_CAMERA.items()
            }
            self.frame_indices_by_camera = {
                camera: np.asarray(
                    archive[f"stereo_{side}_index"], dtype=np.int64
                )
                for camera, side in self._SIDE_BY_CAMERA.items()
            }
        expected_bytes = (self.mask_shape[0] * self.mask_shape[1] + 7) // 8
        for camera, packed in self.packed_by_camera.items():
            if packed.shape != (len(self.quality_valid), expected_bytes):
                raise ValueError(
                    f"Unexpected packed instrument mask shape for {camera}: "
                    f"{packed.shape}"
                )
            if self.packed_interaction_by_camera[camera].shape != packed.shape:
                raise ValueError(
                    f"Unexpected packed distal mask shape for {camera}: "
                    f"{self.packed_interaction_by_camera[camera].shape}"
                )
        self.slot_lookup_by_camera: dict[str, dict[int, int]] = {
            camera: {
                int(frame_index): int(slot)
                for slot, frame_index in enumerate(frame_indices)
                if self.quality_valid[slot]
            }
            for camera, frame_indices in self.frame_indices_by_camera.items()
        }
        self.last_slot = -1
        self.last_source_frame_index = -1
        self.last_frame_gap = -1

    def masks_for_camera_frame(
        self, camera_name: str, frame_index: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        lookup = self.slot_lookup_by_camera.get(camera_name)
        if lookup is None:
            return None
        source_frame = int(frame_index)
        slot = lookup.get(source_frame)
        gap = 0
        if slot is None:
            for candidate_gap in range(1, self.maximum_frame_gap + 1):
                candidates = (
                    source_frame - candidate_gap,
                    source_frame + candidate_gap,
                )
                matched = next((value for value in candidates if value in lookup), None)
                if matched is not None:
                    source_frame = int(matched)
                    slot = lookup[source_frame]
                    gap = candidate_gap
                    break
        if slot is None:
            self.last_slot = -1
            self.last_source_frame_index = -1
            self.last_frame_gap = -1
            return None
        masks = []
        for packed_by_camera in (
            self.packed_by_camera,
            self.packed_interaction_by_camera,
        ):
            packed = packed_by_camera[camera_name][slot]
            masks.append(
                np.unpackbits(
                    packed,
                    bitorder=self.bitorder,
                    count=self.mask_shape[0] * self.mask_shape[1],
                ).reshape(self.mask_shape).astype(bool)
            )
        self.last_slot = int(slot)
        self.last_source_frame_index = source_frame
        self.last_frame_gap = gap
        return masks[0], masks[1]

    def mask_for_camera_frame(
        self, camera_name: str, frame_index: int
    ) -> np.ndarray | None:
        masks = self.masks_for_camera_frame(camera_name, frame_index)
        return None if masks is None else masks[0]


class PackedTissueVisualForceWeights:
    """Turn timestamped packed tissue masks into conservative pixel weights."""

    def __init__(
        self,
        asset_dir: Path | str,
        camera_name: str = "stereo_left",
        erosion_radius_px: int = 7,
        highlight_value_threshold: float = 0.90,
        highlight_channel_spread: float = 0.18,
        highlight_weight: float = 0.10,
    ):
        self.asset_dir = Path(asset_dir)
        with (self.asset_dir / "report.json").open("r") as file:
            self.report = json.load(file)
        if not self.report.get("passed", False):
            raise ValueError("Refusing visual-force masks whose propagation gate failed")
        self.timestamps = np.load(self.asset_dir / "timestamps.npy", mmap_mode="r")
        self.packed_masks = np.load(
            self.asset_dir / "tissue_masks_packbits.npy", mmap_mode="r"
        )
        self.width, self.height = map(int, self.report["resolution_wh"])
        expected_shape = (len(self.timestamps), self.height, (self.width + 7) // 8)
        if self.packed_masks.shape != expected_shape:
            raise ValueError(
                f"Packed mask shape {self.packed_masks.shape} != {expected_shape}"
            )
        if erosion_radius_px < 0:
            raise ValueError("erosion_radius_px must be non-negative")
        if not (0.0 <= highlight_weight <= 1.0):
            raise ValueError("highlight_weight must be in [0, 1]")
        self.camera_name = camera_name
        self.erosion_radius_px = int(erosion_radius_px)
        self.highlight_value_threshold = float(highlight_value_threshold)
        self.highlight_channel_spread = float(highlight_channel_spread)
        self.highlight_weight = float(highlight_weight)
        self.last_frame_index = -1
        self.last_valid_pixels = 0
        self.last_highlight_pixels = 0
        self.last_nonzero_weight_pixels = 0
        self.last_mean_nonzero_weight = 0.0
        self.last_timestamp_in_range = False
        if len(self.timestamps) > 1:
            spacing = np.diff(np.asarray(self.timestamps, dtype=np.float64))
            spacing = spacing[np.isfinite(spacing) & (spacing > 0.0)]
            self.timestamp_tolerance = (
                0.5 * float(np.median(spacing)) + 1.0e-6
                if len(spacing) > 0
                else 1.0e-6
            )
        else:
            self.timestamp_tolerance = 1.0e-6

    def covers_timestamp(self, timestamp: float) -> bool:
        """Return whether this asset contains a mask near ``timestamp``."""
        if len(self.timestamps) == 0 or not np.isfinite(timestamp):
            return False
        return bool(
            float(self.timestamps[0]) - self.timestamp_tolerance
            <= timestamp
            <= float(self.timestamps[-1]) + self.timestamp_tolerance
        )

    def frame_index(self, timestamp: float) -> int:
        right = int(np.searchsorted(self.timestamps, timestamp, side="left"))
        if right <= 0:
            return 0
        if right >= len(self.timestamps):
            return len(self.timestamps) - 1
        left = right - 1
        if abs(float(self.timestamps[left]) - timestamp) <= abs(
            float(self.timestamps[right]) - timestamp
        ):
            return left
        return right

    def raw_mask_at_index(self, frame_index: int) -> np.ndarray:
        packed = np.asarray(self.packed_masks[frame_index])
        return np.unpackbits(packed, axis=1, count=self.width).astype(bool)

    def mask_at_index(self, frame_index: int) -> np.ndarray:
        mask = self.raw_mask_at_index(frame_index).astype(np.uint8)
        if self.erosion_radius_px > 0:
            size = 2 * self.erosion_radius_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            mask = cv2.erode(mask, kernel, iterations=1)
        return mask.astype(bool)

    def camera_weights(
        self,
        color: torch.Tensor,
        timestamp: float,
        *,
        require_timestamp_coverage: bool = False,
        instrument_mask: np.ndarray | None = None,
        interaction_mask: np.ndarray | None = None,
        tool_near_radius_px: float = 120.0,
        tool_far_radius_px: float = 360.0,
        tool_falloff_power: float = 2.0,
        tissue_edge_zero_px: float = 24.0,
        tissue_edge_full_px: float = 64.0,
        tool_occlusion_radius_px: float = 6.0,
        image_border_zero_px: float = 48.0,
        image_border_full_px: float = 96.0,
        posterior_full_reach_px: float | None = 140.0,
        posterior_zero_reach_px: float | None = 280.0,
    ) -> torch.Tensor | None:
        """Build one camera's safe tissue weights at its decoded timestamp."""
        self.last_timestamp_in_range = self.covers_timestamp(timestamp)
        if require_timestamp_coverage and not self.last_timestamp_in_range:
            self.last_frame_index = -1
            self.last_valid_pixels = 0
            self.last_highlight_pixels = 0
            self.last_nonzero_weight_pixels = 0
            self.last_mean_nonzero_weight = 0.0
            return None
        frame_index = self.frame_index(timestamp)
        if instrument_mask is None:
            weight_values = self.mask_at_index(frame_index).astype(np.float32)
        else:
            weight_values = tool_conditioned_tissue_weights(
                self.raw_mask_at_index(frame_index),
                instrument_mask,
                interaction_mask=interaction_mask,
                tool_near_radius_px=tool_near_radius_px,
                tool_far_radius_px=tool_far_radius_px,
                tool_falloff_power=tool_falloff_power,
                tissue_edge_zero_px=tissue_edge_zero_px,
                tissue_edge_full_px=tissue_edge_full_px,
                tool_occlusion_radius_px=tool_occlusion_radius_px,
                image_border_zero_px=image_border_zero_px,
                image_border_full_px=image_border_full_px,
                posterior_full_reach_px=posterior_full_reach_px,
                posterior_zero_reach_px=posterior_zero_reach_px,
            )
        weights = torch.from_numpy(weight_values).to(device=color.device)
        safe_mask = weights > 0.0
        channel_max = color.max(dim=-1).values
        channel_min = color.min(dim=-1).values
        highlight = (channel_max >= self.highlight_value_threshold) & (
            channel_max - channel_min <= self.highlight_channel_spread
        )
        weights[highlight & safe_mask] *= self.highlight_weight
        self.last_frame_index = frame_index
        self.last_valid_pixels = int(safe_mask.sum().item())
        self.last_highlight_pixels = int((highlight & safe_mask).sum().item())
        self.last_nonzero_weight_pixels = self.last_valid_pixels
        self.last_mean_nonzero_weight = (
            float(weights[safe_mask].mean().item())
            if self.last_valid_pixels > 0
            else 0.0
        )
        return weights

    def update_frames(self, frames: Frames, timestamp: float) -> torch.Tensor:
        if frames.width != self.width or frames.height != self.height:
            raise ValueError(
                f"Frame resolution {(frames.width, frames.height)} does not match "
                f"mask resolution {(self.width, self.height)}"
            )
        weights = torch.zeros(
            (len(frames.names), self.height, self.width),
            device=frames.colors_gpu.device,
            dtype=torch.float32,
        )
        if self.camera_name not in frames.names:
            frames.set_loss_weights(weights)
            return weights
        camera_index = frames.names.index(self.camera_name)
        camera_timestamp = (
            float(frames.timestamps[camera_index])
            if np.isfinite(frames.timestamps[camera_index])
            else float(timestamp)
        )
        camera_weights = self.camera_weights(
            frames.colors_gpu[camera_index],
            camera_timestamp,
        )
        assert camera_weights is not None
        weights[camera_index] = camera_weights
        frames.set_loss_weights(weights)
        return weights


class MultiCameraPackedTissueVisualForceWeights:
    """Combine independent packed-mask assets without cross-camera clamping.

    A camera contributes only while its own timestamp lies inside its mask
    asset. This is important for partially propagated assets: later frames must
    receive zero weight instead of silently reusing the final available mask.
    """

    def __init__(
        self,
        camera_assets: dict[str, Path | str],
        erosion_radius_px: int = 7,
        highlight_value_threshold: float = 0.90,
        highlight_channel_spread: float = 0.18,
        highlight_weight: float = 0.10,
        instrument_mask_asset: Path | str | None = None,
        tool_near_radius_px: float = 120.0,
        tool_far_radius_px: float = 360.0,
        tool_falloff_power: float = 2.0,
        tissue_edge_zero_px: float = 24.0,
        tissue_edge_full_px: float = 64.0,
        tool_occlusion_radius_px: float = 6.0,
        image_border_zero_px: float = 48.0,
        image_border_full_px: float = 96.0,
        posterior_full_reach_px: float | None = 140.0,
        posterior_zero_reach_px: float | None = 280.0,
        instrument_maximum_frame_gap: int = 1,
    ):
        if not camera_assets:
            raise ValueError("At least one camera mask asset is required")
        self.providers = {
            camera_name: PackedTissueVisualForceWeights(
                asset_dir,
                camera_name=camera_name,
                erosion_radius_px=erosion_radius_px,
                highlight_value_threshold=highlight_value_threshold,
                highlight_channel_spread=highlight_channel_spread,
                highlight_weight=highlight_weight,
            )
            for camera_name, asset_dir in camera_assets.items()
        }
        resolutions = {
            (provider.width, provider.height)
            for provider in self.providers.values()
        }
        if len(resolutions) != 1:
            raise ValueError(
                f"Camera visual-force mask resolutions disagree: {resolutions}"
            )
        self.width, self.height = next(iter(resolutions))
        self.instrument_masks = (
            PackedStereoInstrumentMasks(
                instrument_mask_asset,
                maximum_frame_gap=instrument_maximum_frame_gap,
            )
            if instrument_mask_asset is not None
            else None
        )
        self.tool_near_radius_px = float(tool_near_radius_px)
        self.tool_far_radius_px = float(tool_far_radius_px)
        self.tool_falloff_power = float(tool_falloff_power)
        self.tissue_edge_zero_px = float(tissue_edge_zero_px)
        self.tissue_edge_full_px = float(tissue_edge_full_px)
        self.tool_occlusion_radius_px = float(tool_occlusion_radius_px)
        self.image_border_zero_px = float(image_border_zero_px)
        self.image_border_full_px = float(image_border_full_px)
        self.posterior_full_reach_px = posterior_full_reach_px
        self.posterior_zero_reach_px = posterior_zero_reach_px
        self.last_active_cameras: list[str] = []
        self.last_camera_statistics: dict[str, dict[str, int | float | bool]] = {}

    def update_frames(self, frames: Frames, timestamp: float) -> torch.Tensor:
        if frames.width != self.width or frames.height != self.height:
            raise ValueError(
                f"Frame resolution {(frames.width, frames.height)} does not match "
                f"mask resolution {(self.width, self.height)}"
            )
        weights = torch.zeros(
            (len(frames.names), self.height, self.width),
            device=frames.colors_gpu.device,
            dtype=torch.float32,
        )
        self.last_active_cameras = []
        self.last_camera_statistics = {}
        for camera_name, provider in self.providers.items():
            if camera_name not in frames.names:
                continue
            camera_index = frames.names.index(camera_name)
            camera_timestamp = (
                float(frames.timestamps[camera_index])
                if np.isfinite(frames.timestamps[camera_index])
                else float(timestamp)
            )
            instrument_pair = (
                self.instrument_masks.masks_for_camera_frame(
                    camera_name, provider.frame_index(camera_timestamp)
                )
                if self.instrument_masks is not None
                else None
            )
            if self.instrument_masks is not None and instrument_pair is None:
                self.last_camera_statistics[camera_name] = {
                    "timestamp": camera_timestamp,
                    "timestamp_in_range": provider.covers_timestamp(
                        camera_timestamp
                    ),
                    "frame_index": provider.frame_index(camera_timestamp),
                    "valid_pixels": 0,
                    "highlight_pixels": 0,
                    "mean_nonzero_weight": 0.0,
                    "instrument_slot": -1,
                    "instrument_source_frame_index": -1,
                    "instrument_frame_gap": -1,
                }
                continue
            instrument_mask = (
                instrument_pair[0] if instrument_pair is not None else None
            )
            interaction_mask = (
                instrument_pair[1] if instrument_pair is not None else None
            )
            camera_weights = provider.camera_weights(
                frames.colors_gpu[camera_index],
                camera_timestamp,
                require_timestamp_coverage=True,
                instrument_mask=instrument_mask,
                interaction_mask=interaction_mask,
                tool_near_radius_px=self.tool_near_radius_px,
                tool_far_radius_px=self.tool_far_radius_px,
                tool_falloff_power=self.tool_falloff_power,
                tissue_edge_zero_px=self.tissue_edge_zero_px,
                tissue_edge_full_px=self.tissue_edge_full_px,
                tool_occlusion_radius_px=self.tool_occlusion_radius_px,
                image_border_zero_px=self.image_border_zero_px,
                image_border_full_px=self.image_border_full_px,
                posterior_full_reach_px=self.posterior_full_reach_px,
                posterior_zero_reach_px=self.posterior_zero_reach_px,
            )
            if camera_weights is not None:
                weights[camera_index] = camera_weights
                self.last_active_cameras.append(camera_name)
            self.last_camera_statistics[camera_name] = {
                "timestamp": camera_timestamp,
                "timestamp_in_range": provider.last_timestamp_in_range,
                "frame_index": provider.last_frame_index,
                "valid_pixels": provider.last_valid_pixels,
                "highlight_pixels": provider.last_highlight_pixels,
                "mean_nonzero_weight": provider.last_mean_nonzero_weight,
                "instrument_slot": (
                    self.instrument_masks.last_slot
                    if self.instrument_masks is not None
                    else -1
                ),
                "instrument_source_frame_index": (
                    self.instrument_masks.last_source_frame_index
                    if self.instrument_masks is not None
                    else -1
                ),
                "instrument_frame_gap": (
                    self.instrument_masks.last_frame_gap
                    if self.instrument_masks is not None
                    else -1
                ),
            }
        frames.set_loss_weights(weights)
        return weights
