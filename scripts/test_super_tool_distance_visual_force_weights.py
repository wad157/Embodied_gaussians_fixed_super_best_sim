#!/usr/bin/env python3
"""Gate the broad tool-SDF visual-force pixel weights on synthetic and real masks."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_gaussians.embodied_simulator.visual_force_masks import (  # noqa: E402
    PackedStereoInstrumentMasks,
    PackedTissueVisualForceWeights,
    tool_conditioned_tissue_weights,
)


MULTIVIEW_ROOT = REPO_ROOT / "data/super/grasp5_native"
INSTRUMENT_MASKS = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_paper_lnd_stereo_dense_contact_v4/"
    "surgicalsam2_multianchor_parts_dense_contact_v6/"
    "stereo_multianchor_part_masks.npz"
)


def synthetic_gate() -> tuple[dict[str, bool], dict[str, float]]:
    tissue = np.zeros((200, 320), dtype=bool)
    tissue[10:200, 20:300] = True
    instrument = np.zeros((100, 160), dtype=bool)
    instrument[45:55, 35:45] = True
    instrument[0:46, 39:41] = True
    interaction = np.zeros_like(instrument)
    interaction[45:55, 35:45] = True
    weights = tool_conditioned_tissue_weights(
        tissue,
        instrument,
        interaction_mask=interaction,
        tool_near_radius_px=20.0,
        tool_far_radius_px=100.0,
        tool_falloff_power=2.0,
        tissue_edge_zero_px=8.0,
        tissue_edge_full_px=20.0,
        tool_occlusion_radius_px=4.0,
        image_border_zero_px=8.0,
        image_border_full_px=20.0,
        posterior_full_reach_px=20.0,
        posterior_zero_reach_px=60.0,
    )
    linear_weights = tool_conditioned_tissue_weights(
        tissue,
        instrument,
        interaction_mask=interaction,
        tool_near_radius_px=20.0,
        tool_far_radius_px=100.0,
        tool_falloff_power=1.0,
        tissue_edge_zero_px=8.0,
        tissue_edge_full_px=20.0,
        tool_occlusion_radius_px=4.0,
        image_border_zero_px=8.0,
        image_border_full_px=20.0,
        posterior_full_reach_px=20.0,
        posterior_zero_reach_px=60.0,
    )
    samples = {
        "instrument": float(weights[100, 80]),
        "near": float(weights[100, 105]),
        "transition": float(weights[100, 140]),
        "linear_transition": float(linear_weights[100, 140]),
        "far": float(weights[100, 220]),
        "tissue_edge": float(weights[100, 24]),
        "background": float(weights[5, 100]),
        "image_border": float(weights[198, 100]),
        "posterior_zero": float(weights[30, 105]),
        "posterior_transition": float(weights[60, 105]),
        "camera_near_side": float(weights[140, 105]),
        "minimum": float(weights.min()),
        "maximum": float(weights.max()),
    }
    gates = {
        "instrument_occlusion_is_zero": samples["instrument"] == 0.0,
        "near_tool_region_has_full_weight": samples["near"] > 0.99,
        "tool_distance_transition_is_smooth": 0.0 < samples["transition"] < 1.0,
        "quadratic_falloff_reduces_transition_weight": (
            samples["transition"] < samples["linear_transition"]
        ),
        "far_region_is_zero": samples["far"] == 0.0,
        "tissue_edge_overrides_tool_radius": samples["tissue_edge"] == 0.0,
        "background_is_zero": samples["background"] == 0.0,
        "image_border_is_zero_when_tissue_touches_frame": (
            samples["image_border"] == 0.0
        ),
        "posterior_far_region_is_zero": samples["posterior_zero"] == 0.0,
        "posterior_transition_is_smooth": (
            0.0 < samples["posterior_transition"] < 1.0
        ),
        "camera_near_side_keeps_broad_support": samples["camera_near_side"] > 0.0,
        "weights_are_bounded": samples["minimum"] >= 0.0
        and samples["maximum"] <= 1.0,
    }
    return gates, samples


def dilated_tool_mask(mask: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    height, width = output_shape
    full = cv2.resize(
        mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    return cv2.dilate(full, kernel, iterations=1).astype(bool)


def real_asset_gate() -> tuple[
    dict[str, bool], list[dict[str, int | float | str]], float
]:
    tissue_assets = {
        "stereo_left": MULTIVIEW_ROOT / "visual_force_masks_v1",
        "stereo_right": MULTIVIEW_ROOT / "visual_force_masks_right_v1",
    }
    providers = {
        camera: PackedTissueVisualForceWeights(asset, erosion_radius_px=7)
        for camera, asset in tissue_assets.items()
    }
    instrument = PackedStereoInstrumentMasks(INSTRUMENT_MASKS, maximum_frame_gap=1)
    samples: list[dict[str, int | float | str]] = []
    all_finite_and_bounded = True
    all_have_support = True
    all_occlusions_zero = True
    all_reduce_tissue_area = True
    all_posterior_regions_zero = True
    all_new_support_is_subset_of_old = True
    maximum_camera_near_core_reduction = 0.0
    all_weight_values_not_increased = True
    maximum_frame_gap = 0
    for frame_index in (0, 420, 498, 1000, 1440):
        for camera, provider in providers.items():
            tool_pair = instrument.masks_for_camera_frame(camera, frame_index)
            if tool_pair is None:
                raise RuntimeError(
                    f"No instrument mask within one frame of {camera}:{frame_index}"
                )
            tool, interaction = tool_pair
            tissue = provider.raw_mask_at_index(frame_index)
            weights = tool_conditioned_tissue_weights(
                tissue, tool, interaction_mask=interaction
            )
            old_weights = tool_conditioned_tissue_weights(
                tissue,
                tool,
                interaction_mask=interaction,
                tool_falloff_power=1.0,
                posterior_full_reach_px=None,
                posterior_zero_reach_px=None,
            )
            nonzero = int(np.count_nonzero(weights > 0.0))
            tissue_area = int(np.count_nonzero(tissue))
            occlusion = dilated_tool_mask(tool, tissue.shape)
            occlusion_max = float(weights[occlusion].max(initial=0.0))
            all_finite_and_bounded &= bool(
                np.isfinite(weights).all()
                and weights.min() >= 0.0
                and weights.max() <= 1.0
            )
            all_have_support &= nonzero > 0
            all_occlusions_zero &= occlusion_max == 0.0
            all_reduce_tissue_area &= nonzero < tissue_area
            interaction_scale_y = interaction.shape[0] / float(tissue.shape[0])
            interaction_reference_y = float(
                np.median(np.nonzero(interaction)[0]) / interaction_scale_y
            )
            rows = np.arange(tissue.shape[0])[:, None]
            posterior_region = tissue & (
                interaction_reference_y - rows >= 280.0
            )
            camera_near_region = (
                tissue
                & (rows >= interaction_reference_y)
                & (old_weights > 0.999)
            )
            posterior_old_pixels = int(
                np.count_nonzero(old_weights[posterior_region] > 0.0)
            )
            posterior_new_pixels = int(
                np.count_nonzero(weights[posterior_region] > 0.0)
            )
            all_posterior_regions_zero &= posterior_new_pixels == 0
            all_new_support_is_subset_of_old &= bool(
                np.all((weights > 0.0) <= (old_weights > 0.0))
            )
            all_weight_values_not_increased &= bool(
                np.all(weights <= old_weights + 1.0e-6)
            )
            maximum_camera_near_core_reduction = max(
                maximum_camera_near_core_reduction,
                float(
                    np.max(
                        old_weights[camera_near_region]
                        - weights[camera_near_region],
                        initial=0.0,
                    )
                ),
            )
            maximum_frame_gap = max(maximum_frame_gap, instrument.last_frame_gap)
            samples.append(
                {
                    "camera": camera,
                    "frame_index": frame_index,
                    "instrument_source_frame_index": (
                        instrument.last_source_frame_index
                    ),
                    "instrument_frame_gap": instrument.last_frame_gap,
                    "tissue_pixels": tissue_area,
                    "weighted_pixels": nonzero,
                    "weighted_fraction_of_tissue": nonzero / tissue_area,
                    "mean_nonzero_weight": float(weights[weights > 0.0].mean()),
                    "occlusion_maximum_weight": occlusion_max,
                    "posterior_reference_y": interaction_reference_y,
                    "posterior_old_weighted_pixels": posterior_old_pixels,
                    "posterior_new_weighted_pixels": posterior_new_pixels,
                }
            )
    gates = {
        "real_weights_are_finite_and_bounded": all_finite_and_bounded,
        "every_sample_has_local_tissue_support": all_have_support,
        "instrument_and_six_pixel_occlusion_are_zero": all_occlusions_zero,
        "distance_and_edge_fields_reduce_whole_tissue_mask": (
            all_reduce_tissue_area
        ),
        "posterior_beyond_280px_is_exactly_zero": all_posterior_regions_zero,
        "posterior_gate_only_removes_existing_support": (
            all_new_support_is_subset_of_old
        ),
        "quadratic_far_falloff_never_increases_old_weight": (
            all_weight_values_not_increased
        ),
        "camera_near_full_weight_core_reduction_below_0p0011": (
            maximum_camera_near_core_reduction <= 1.1e-3
        ),
        "missing_strict_pair_frames_use_at_most_one_frame_neighbor": (
            maximum_frame_gap <= 1
        ),
    }
    return gates, samples, maximum_camera_near_core_reduction


def main() -> None:
    synthetic_gates, synthetic_samples = synthetic_gate()
    (
        real_gates,
        real_samples,
        maximum_camera_near_core_reduction,
    ) = real_asset_gate()
    gates = {**synthetic_gates, **real_gates}
    report = {
        "stage": "tool_distance_visual_force_pixel_weight_gate",
        "parameters_at_1920x1080": {
            "tool_near_radius_px": 120.0,
            "tool_far_radius_px": 360.0,
            "tool_falloff_power": 2.0,
            "tissue_edge_zero_px": 24.0,
            "tissue_edge_full_px": 64.0,
            "tool_occlusion_radius_px": 6.0,
            "image_border_zero_px": 48.0,
            "image_border_full_px": 96.0,
            "posterior_full_reach_px": 140.0,
            "posterior_zero_reach_px": 280.0,
        },
        "synthetic_samples": synthetic_samples,
        "real_samples": real_samples,
        "maximum_camera_near_core_reduction": (
            maximum_camera_near_core_reduction
        ),
        "gates": gates,
        "passed": bool(all(gates.values())),
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Tool-distance visual-force pixel-weight gate failed")


if __name__ == "__main__":
    main()
