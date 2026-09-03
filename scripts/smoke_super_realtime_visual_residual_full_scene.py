#!/usr/bin/env python3
"""Gate one accepted real stereo residual inside the complete SUPER scene."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import warp as wp


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from embodied_environments.super_embodied.super_embodied import (  # noqa: E402
    PSM_POSE_DRIVER_PATHS,
    build_environment,
)
from embodied_gaussians import DatasetManager  # noqa: E402
from embodied_gaussians.embodied_simulator.visual_force_masks import (  # noqa: E402
    MultiCameraPackedTissueVisualForceWeights,
)
from example_embodied_super_offline import (  # noqa: E402
    RIGHT_VISUAL_FORCE_MASK_DIR,
    VISUAL_FORCE_INSTRUMENT_MASKS,
    VISUAL_FORCE_MASK_DIR,
    SuperPlaybackControls,
    build_visual_tissue_residual_mapper,
)


def main() -> None:
    wp.init()
    environment = build_environment(
        psm_pose_driver_path=PSM_POSE_DRIVER_PATHS[
            "raw_paper_lnd_sam2_dense_contact_unbounded_xyz"
        ],
        psm_visual_tip_only=False,
        tissue_mode="paper_pbd",
    )
    dataset = DatasetManager(REPO_ROOT / "data/super/grasp5_offline_demo")
    dataset.keep_only_cameras(["stereo_left", "stereo_right"])
    dataset.set_visual_force_weight_provider(
        MultiCameraPackedTissueVisualForceWeights(
            {
                "stereo_left": VISUAL_FORCE_MASK_DIR,
                "stereo_right": RIGHT_VISUAL_FORCE_MASK_DIR,
            },
            erosion_radius_px=7,
            highlight_weight=0.10,
            instrument_mask_asset=VISUAL_FORCE_INSTRUMENT_MASKS,
            tool_near_radius_px=120.0,
            tool_far_radius_px=360.0,
            tool_falloff_power=2.0,
            tissue_edge_zero_px=24.0,
            tissue_edge_full_px=64.0,
            tool_occlusion_radius_px=6.0,
            image_border_zero_px=48.0,
            image_border_full_px=96.0,
            posterior_full_reach_px=140.0,
            posterior_zero_reach_px=280.0,
        )
    )
    dataset.update_frames(0.0)
    environment.frames = dataset.frames
    mapper = build_visual_tissue_residual_mapper(
        environment, iterations=4, learning_rate_m=2.5e-5
    )
    controls = SuperPlaybackControls(
        environment,
        dataset,
        fps=30,
        visual_force_update_interval=4,
        visual_feedback_mode="residual",
        visual_residual_mapper=mapper,
        enable_psm_tissue_contact=True,
    )
    controls.reset()
    controls.go_to_frame(420)
    environment.step(compute_visual_forces=False)
    controls.apply_current_psm_pose()

    velocities_before = wp.to_torch(
        environment.sim.state_0.particle_qd
    ).detach().clone()
    previous_residual = None
    elapsed_ms = []
    accepted_updates = []
    result = None
    for _ in range(2):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = environment.sim.solve_visual_tissue_residual(
            mapper,
            environment.frames,
            previous_residual=previous_residual,
            observations_are_bgr=True,
        )
        accepted = environment.sim.apply_visual_tissue_residual(result)
        torch.cuda.synchronize()
        elapsed_ms.append((time.perf_counter() - start) * 1000.0)
        accepted_updates.append(accepted)
        previous_residual = result.residual.detach().clone() if accepted else None
    assert result is not None
    velocities_after = wp.to_torch(environment.sim.state_0.particle_qd)
    gates = {
        "accepted": all(accepted_updates),
        "visual_loss_decreased": (
            result.final_visual_loss < result.initial_visual_loss
        ),
        "no_inverted_tetrahedra": result.inverted_tetrahedra == 0,
        "particle_velocity_unchanged": torch.equal(
            velocities_before, velocities_after
        ),
        "soft_only_backward_render": (
            environment.sim.gaussian_model.num_soft_gaussians
            < environment.sim.gaussian_model.num_gaussians
        ),
    }
    contact_metrics = environment.sim.triangle_skin_contact_metrics()
    report = {
        "stage": "complete_scene_realtime_visual_residual_gate",
        "frame": 420,
        "full_gaussians": environment.sim.gaussian_model.num_gaussians,
        "backward_render_gaussians": (
            environment.sim.gaussian_model.num_soft_gaussians
        ),
        "initial_visual_loss": result.initial_visual_loss,
        "final_visual_loss": result.final_visual_loss,
        "maximum_residual_mm": result.maximum_residual_m * 1000.0,
        "minimum_volume_ratio": result.minimum_volume_ratio,
        "solve_and_apply_elapsed_ms_cold_to_warm": elapsed_ms,
        "contact": {
            "configured": contact_metrics is not None,
            "candidate_count": (
                int(contact_metrics["contact_count"])
                if contact_metrics is not None
                else 0
            ),
            "maximum_penetration_mm": (
                float(contact_metrics["maximum_penetration_m"]) * 1000.0
                if contact_metrics is not None
                else 0.0
            ),
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    output = (
        REPO_ROOT
        / "outputs/visual_tissue_residual_mapping/full_scene_frame_0420.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Complete-scene realtime residual gate failed")


if __name__ == "__main__":
    main()
