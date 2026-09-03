#!/usr/bin/env python3
"""Run one real stereo frame through particle-space visual residual mapping."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import warp as wp


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embodied_gaussians import DatasetManager  # noqa: E402
from embodied_gaussians.embodied_simulator.builder import (  # noqa: E402
    EmbodiedGaussiansBuilder,
)
from embodied_gaussians.embodied_simulator.simulator import (  # noqa: E402
    EmbodiedGaussiansSimulator,
)
from embodied_gaussians.embodied_simulator.visual_force_masks import (  # noqa: E402
    MultiCameraPackedTissueVisualForceWeights,
)
from embodied_gaussians.physics_simulator.visual_tissue_residual_mapping import (  # noqa: E402
    TetrahedralGaussianVisualResidualMapper,
    VisualTissueResidualMappingSettings,
)
from embodied_gaussians.scene_builders.domain import SoftBody  # noqa: E402


DEFAULT_ASSET = (
    REPO_ROOT
    / "data/super/grasp5_native/tissue_multiview_v1/"
    "paper_pbd_tissue_v15_denser_mild_paper_constraints_centroid_ellipsoids/"
    "tissue_paper_pbd_centroid_gaussians.npz"
)
DEFAULT_DATASET = REPO_ROOT / "data/super/grasp5_offline_demo"
LEFT_MASKS = REPO_ROOT / "data/super/grasp5_native/visual_force_masks_v1"
RIGHT_MASKS = (
    REPO_ROOT / "data/super/grasp5_native/visual_force_masks_right_v1"
)
INSTRUMENT_MASKS = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_paper_lnd_stereo_dense_contact_v4/"
    "surgicalsam2_multianchor_parts_dense_contact_v6/"
    "stereo_multianchor_part_masks.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--frame", type=int, default=420)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--learning-rate-m", type=float, default=2.5e-5)
    parser.add_argument("--image-scale", type=float, default=0.25)
    parser.add_argument("--maximum-residual-m", type=float, default=0.00025)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--feedback-steps",
        type=int,
        default=0,
        help="Also apply this many consecutive residual updates to the state.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs/visual_tissue_residual_mapping/real_frame_0420.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wp.init()
    soft_body = SoftBody.from_npz(args.asset, name="super_tissue_visual_residual")
    builder = EmbodiedGaussiansBuilder(gravity=0.0)
    builder.add_soft_body(
        soft_body,
        young_modulus_pa=15_000.0,
        poisson_ratio=0.45,
        anchor_mode="support_candidate",
        add_gaussians=True,
        add_collision_skin=False,
    )
    simulator = EmbodiedGaussiansSimulator(builder, device=args.device)
    simulator.update_gaussian_transforms()

    dataset = DatasetManager(args.dataset)
    dataset.keep_only_cameras(["stereo_left", "stereo_right"])
    provider = MultiCameraPackedTissueVisualForceWeights(
        {
            "stereo_left": LEFT_MASKS,
            "stereo_right": RIGHT_MASKS,
        },
        erosion_radius_px=7,
        highlight_weight=0.10,
        instrument_mask_asset=INSTRUMENT_MASKS,
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
    dataset.set_visual_force_weight_provider(provider)
    left_timestamps = dataset.offline_cameras.cameras["stereo_left"].timestamps
    if not 0 <= args.frame < len(left_timestamps):
        raise ValueError("Requested left frame is out of range")
    dataset.update_frames(float(left_timestamps[args.frame]))

    model = simulator.gaussian_model
    mesh = soft_body.tetra_mesh
    settings = VisualTissueResidualMappingSettings(
        iterations=args.iterations,
        learning_rate_m=args.learning_rate_m,
        image_scale=args.image_scale,
        maximum_residual_m=args.maximum_residual_m,
    )
    mapper = TetrahedralGaussianVisualResidualMapper.from_visual_face_centroid_bindings(
        rest_positions=torch.as_tensor(
            mesh.rest_positions, device=args.device, dtype=torch.float32
        ),
        tet_indices=torch.as_tensor(
            mesh.tet_indices, device=args.device, dtype=torch.long
        ),
        fixed_mask=(
            wp.to_torch(simulator.model.particle_inv_mass).detach() == 0.0
        ),
        soft_gaussian_ids=model.soft_gaussian_ids,
        visual_vertex_particle_indices=(
            model.soft_gaussian_visual_vertex_particle_indices
        ),
        visual_vertex_weights=model.soft_gaussian_visual_vertex_weights,
        settings=settings,
    )
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    solve_elapsed_s: list[float] = []
    result = None
    for _ in range(args.repeats):
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        solve_start = time.perf_counter()
        result = simulator.solve_visual_tissue_residual(
            mapper,
            dataset.frames,
            observations_are_bgr=True,
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        solve_elapsed_s.append(time.perf_counter() - solve_start)
    assert result is not None
    fixed_residual_max = float(
        torch.linalg.vector_norm(
            result.residual[mapper.fixed_mask], dim=1
        ).max().item()
    )
    gates = {
        "both_cameras_have_weights": bool(
            torch.all(dataset.frames.loss_weights_gpu.sum(dim=(1, 2)) > 0.0).item()
        ),
        "visual_loss_is_finite": bool(
            torch.isfinite(torch.tensor(result.final_visual_loss)).item()
        ),
        "visual_loss_decreased": result.final_visual_loss
        < result.initial_visual_loss,
        "fixed_boundary_residual_is_zero": fixed_residual_max == 0.0,
        "residual_cap_respected": result.maximum_residual_m
        <= settings.maximum_residual_m + 1.0e-9,
        "all_corrected_tetrahedra_positive": result.inverted_tetrahedra == 0
        and result.minimum_volume_ratio > 0.0,
    }
    feedback_history = []
    if args.feedback_steps < 0:
        raise ValueError("--feedback-steps must be non-negative")
    if args.feedback_steps:
        feedback_start_positions = wp.to_torch(
            simulator.state_0.particle_q
        ).detach().clone()
        feedback_start_velocities = wp.to_torch(
            simulator.state_0.particle_qd
        ).detach().clone()
        previous_residual = None
        for feedback_index in range(args.feedback_steps):
            feedback_result = simulator.solve_visual_tissue_residual(
                mapper,
                dataset.frames,
                previous_residual=previous_residual,
                observations_are_bgr=True,
            )
            accepted = simulator.apply_visual_tissue_residual(feedback_result)
            feedback_history.append(
                {
                    "step": feedback_index + 1,
                    "accepted": accepted,
                    "initial_visual_loss": feedback_result.initial_visual_loss,
                    "final_visual_loss": feedback_result.final_visual_loss,
                    "maximum_residual_mm": (
                        feedback_result.maximum_residual_m * 1000.0
                    ),
                    "minimum_volume_ratio": (
                        feedback_result.minimum_volume_ratio
                    ),
                }
            )
            previous_residual = (
                feedback_result.residual.detach().clone()
                if accepted
                else None
            )
        feedback_positions = wp.to_torch(simulator.state_0.particle_q)
        feedback_velocities = wp.to_torch(simulator.state_0.particle_qd)
        gates["feedback_updates_all_accepted"] = all(
            item["accepted"] for item in feedback_history
        )
        gates["feedback_does_not_inject_velocity"] = bool(
            torch.equal(feedback_velocities, feedback_start_velocities)
        )
        feedback_cumulative_max_mm = float(
            torch.linalg.vector_norm(
                feedback_positions - feedback_start_positions, dim=1
            ).max().item()
            * 1000.0
        )
    else:
        feedback_cumulative_max_mm = 0.0
    report = {
        "stage": "real_stereo_gaussian_visual_tissue_residual_mapping_smoke",
        "paper_adaptation": (
            "replace visible-surface point-cloud Chamfer with masked stereo "
            "Gaussian RGB residual; optimize physical particle residual"
        ),
        "frame": args.frame,
        "camera_names": dataset.frames.names,
        "settings": settings.__dict__,
        "mesh": {
            "particles": int(len(mesh.rest_positions)),
            "tetrahedra": int(len(mesh.tet_indices)),
            "soft_gaussians": int(model.num_soft_gaussians),
        },
        "initial_visual_loss": result.initial_visual_loss,
        "final_visual_loss": result.final_visual_loss,
        "visual_loss_reduction_fraction": (
            (result.initial_visual_loss - result.final_visual_loss)
            / max(result.initial_visual_loss, 1.0e-12)
        ),
        "final_loss_terms": {
            "visual": result.final_visual_loss,
            "distance": result.final_distance_loss,
            "volume": result.final_volume_loss,
            "shape": result.final_shape_loss,
            "spatial": result.final_spatial_loss,
            "temporal": result.final_temporal_loss,
            "magnitude": result.final_magnitude_loss,
            "weighted_total": result.final_total_loss,
        },
        "maximum_residual_mm": result.maximum_residual_m * 1000.0,
        "rms_residual_mm": result.rms_residual_m * 1000.0,
        "minimum_volume_ratio": result.minimum_volume_ratio,
        "inverted_tetrahedra": result.inverted_tetrahedra,
        "backtrack_count": result.backtrack_count,
        "solve_elapsed_s_cold_to_warm": solve_elapsed_s,
        "solve_elapsed_s_warm": solve_elapsed_s[-1],
        "feedback_history": feedback_history,
        "feedback_cumulative_maximum_mm": feedback_cumulative_max_mm,
        "gates": gates,
        "passed": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Real visual tissue residual mapping smoke failed")


if __name__ == "__main__":
    main()
