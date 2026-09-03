"""Privileged synthetic benchmark export for the simulation reconstruction.

The exporter is deliberately downstream of estimation.  Ground-truth motion is
never read by the runtime controller: only first-frame evaluation points are
embedded once into the reconstructed top surface.  Their persistent barycentric
coordinates are then evaluated on the predicted PBD mesh at every frame.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import warp as wp


CAMERAS = ("stereo_left", "stereo_right")


def _surface_embedding(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Embed nearly planar top-surface points with persistent triangle weights."""
    triangles = vertices[faces]
    a = triangles[:, 0, :2]
    v0 = triangles[:, 1, :2] - a
    v1 = triangles[:, 2, :2] - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    denominator = d00 * d11 - d01 * d01
    valid_faces = np.abs(denominator) > 1.0e-16
    if not bool(valid_faces.any()):
        raise ValueError("重建顶面没有非退化三角形")

    selected_faces: list[int] = []
    selected_weights: list[np.ndarray] = []
    embedded: list[np.ndarray] = []
    errors: list[float] = []
    for point in np.asarray(points, dtype=np.float64):
        v2 = point[None, :2] - a
        d20 = np.einsum("ij,ij->i", v2, v0)
        d21 = np.einsum("ij,ij->i", v2, v1)
        weight_1 = np.zeros(len(faces), dtype=np.float64)
        weight_2 = np.zeros(len(faces), dtype=np.float64)
        weight_1[valid_faces] = (
            d11[valid_faces] * d20[valid_faces]
            - d01[valid_faces] * d21[valid_faces]
        ) / denominator[valid_faces]
        weight_2[valid_faces] = (
            d00[valid_faces] * d21[valid_faces]
            - d01[valid_faces] * d20[valid_faces]
        ) / denominator[valid_faces]
        weights = np.stack(
            (1.0 - weight_1 - weight_2, weight_1, weight_2), axis=1
        )
        inside = valid_faces & (weights.min(axis=1) >= -1.0e-6)
        if bool(inside.any()):
            candidates = np.flatnonzero(inside)
            candidate_points = np.einsum(
                "fi,fij->fj", weights[candidates], triangles[candidates]
            )
            face_index = int(
                candidates[np.argmin(np.linalg.norm(candidate_points - point, axis=1))]
            )
            chosen_weights = weights[face_index]
        else:
            # This is used only for rare boundary points just outside the
            # reconstructed scan hull.  Select the least extrapolating face,
            # clamp it to the triangle, and report the resulting initialization
            # error in metadata instead of silently aligning the trajectory.
            score = np.where(valid_faces, weights.min(axis=1), -np.inf)
            face_index = int(np.argmax(score))
            chosen_weights = np.clip(weights[face_index], 0.0, None)
            chosen_weights /= max(float(chosen_weights.sum()), 1.0e-12)
        embedded_point = np.einsum(
            "i,ij->j", chosen_weights, triangles[face_index]
        )
        selected_faces.append(face_index)
        selected_weights.append(chosen_weights)
        embedded.append(embedded_point)
        errors.append(float(np.linalg.norm(embedded_point - point)))
    return (
        np.asarray(selected_faces, dtype=np.int32),
        np.asarray(selected_weights, dtype=np.float32),
        np.asarray(embedded, dtype=np.float32),
        np.asarray(errors, dtype=np.float32),
    )


def _project_points(
    points_world: np.ndarray,
    X_CW_opencv: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    camera = (
        np.asarray(X_CW_opencv[:3, :3], dtype=np.float64)
        @ np.asarray(points_world, dtype=np.float64).T
    ).T + np.asarray(X_CW_opencv[:3, 3], dtype=np.float64)
    depth = camera[:, 2]
    uv_h = (np.asarray(intrinsic, dtype=np.float64) @ camera.T).T
    uv = np.full((len(points_world), 2), np.nan, dtype=np.float32)
    projectable = np.abs(depth) > 1.0e-8
    uv[projectable] = (
        uv_h[projectable, :2] / depth[projectable, None]
    ).astype(
        np.float32
    )
    # Do not discard a bad prediction merely because it leaves the image.
    # GT visibility is the evaluation mask; any finite predicted projection
    # remains an error-bearing sample, including off-screen coordinates.
    valid = projectable & np.isfinite(uv).all(axis=1)
    return uv, valid


class SimBenchmarkArtifactWriter:
    """Export trajectories, tissue-only renders, and material diagnostics."""

    def __init__(
        self,
        playback_controls: Any,
        output_directory: Path,
        *,
        start_frame: int,
        end_frame_exclusive: int,
        label: str,
        render_images: bool,
        open_loop_start_frame: int | None,
        physics_steps_per_frame: int,
        holdout_stride: int | None,
        holdout_offset: int,
        render_frame_mode: str,
    ) -> None:
        self.controls = playback_controls
        self.environment = playback_controls.environment
        self.dataset_root = Path(playback_controls.dataset_manager.path).resolve()
        self.output_directory = Path(output_directory).resolve()
        if self.output_directory.exists() and any(self.output_directory.iterdir()):
            raise FileExistsError(
                f"拒绝覆盖已有仿真评估产物：{self.output_directory}"
            )
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.label = str(label)
        self.render_images = bool(render_images)
        # Trajectory-only system-identification probes can skip the expensive
        # Gaussian alignment render. Formal PSNR/SSIM/LPIPS runs leave this
        # unset and retain the original RGB diagnostics.
        self.compute_rgb_alignment = (
            os.environ.get("SIM_BENCHMARK_SKIP_RGB_ALIGNMENT", "0") != "1"
        )
        self.open_loop_start_frame = open_loop_start_frame
        self.physics_steps_per_frame = int(physics_steps_per_frame)
        self.holdout_stride = holdout_stride
        self.holdout_offset = int(holdout_offset)
        self.render_frame_mode = str(render_frame_mode)
        self.frame_indices = np.arange(
            int(start_frame), int(end_frame_exclusive), dtype=np.int32
        )
        self.timestamps = np.asarray(
            playback_controls.playback_timestamps[self.frame_indices],
            dtype=np.float64,
        )
        self._frame_to_slot = {
            int(frame): slot for slot, frame in enumerate(self.frame_indices)
        }
        self._written = np.zeros(len(self.frame_indices), dtype=bool)

        reference = np.load(
            self.dataset_root / "ground_truth/trajectories_3d.npz"
        )
        self.node_ids = np.asarray(
            reference["tissue_evaluation_node_ids"], dtype=np.int32
        )
        reference_ids = np.asarray(reference["tissue_node_ids"], dtype=np.int32)
        lookup = {int(node_id): index for index, node_id in enumerate(reference_ids)}
        reference_columns = np.asarray(
            [lookup[int(node_id)] for node_id in self.node_ids], dtype=np.int64
        )
        self.reference_first_positions = np.asarray(
            reference["tissue_positions_world"][0, reference_columns],
            dtype=np.float32,
        )

        tissue_path = Path(self.environment.super_tissue_asset_path)
        with np.load(tissue_path, allow_pickle=False) as tissue:
            rest = np.asarray(tissue["rest_positions_table"], dtype=np.float32)
            surface_faces = np.asarray(tissue["surface_faces"], dtype=np.int32)
            top_mask = np.asarray(tissue["top_node_mask"], dtype=bool)
        top_faces = surface_faces[np.all(top_mask[surface_faces], axis=1)]
        if len(top_faces) == 0:
            raise ValueError("重建资产没有完整顶面三角形")
        (
            local_face_ids,
            self.surface_weights,
            self.initial_embedded_positions,
            self.initial_embedding_errors_m,
        ) = _surface_embedding(self.reference_first_positions, rest, top_faces)
        self.surface_particle_ids_local = top_faces[local_face_ids]
        handle = self.environment.super_tissue_soft_handle
        if handle is None:
            raise ValueError("仿真评估需要可变形组织 handle")
        self.surface_particle_ids_global = (
            self.surface_particle_ids_local + int(handle.particle_start)
        ).astype(np.int64)

        frame_count = len(self.frame_indices)
        node_count = len(self.node_ids)
        self.positions = np.full(
            (frame_count, node_count, 3), np.nan, dtype=np.float32
        )
        self.projected_uv = {
            camera: np.full(
                (frame_count, node_count, 2), np.nan, dtype=np.float32
            )
            for camera in CAMERAS
        }
        self.projected_valid = {
            camera: np.zeros((frame_count, node_count), dtype=bool)
            for camera in CAMERAS
        }
        self.distance_stats = np.full((frame_count, 3), np.nan, dtype=np.float32)
        self.shape_stats = np.full((frame_count, 3), np.nan, dtype=np.float32)
        projector = self.environment.sim.material_projector
        if projector is None:
            self.material_particle_ids = np.empty(0, dtype=np.int32)
            self.distance_fields = np.empty(
                (frame_count, 0), dtype=np.float32
            )
            self.shape_fields = np.empty((frame_count, 0), dtype=np.float32)
        else:
            material_particle_count = int(
                wp.to_torch(projector.paper_distance_stiffness).numel()
            )
            self.material_particle_ids = np.arange(
                material_particle_count, dtype=np.int32
            )
            self.distance_fields = np.full(
                (frame_count, material_particle_count),
                np.nan,
                dtype=np.float32,
            )
            self.shape_fields = np.full(
                (frame_count, material_particle_count),
                np.nan,
                dtype=np.float32,
            )
        self.stiffness_signal_float_names = (
            "residual_norm_m",
            "deformation_norm_m",
            "vector_signal",
            "strain_signal",
            "strain_confidence",
            "blended_signal_before_smoothing",
            "smoothed_signal",
            "candidate_ema",
            "log_step",
        )
        self.stiffness_signal_mask_names = (
            "quality_valid_mask",
            "supervision_valid_mask",
            "control_exclusion_mask",
            "control_frozen_mask",
            "eligible_mask",
            "vector_active_mask",
            "strain_active_mask",
            "material_active_mask",
        )
        material_particle_count = len(self.material_particle_ids)
        self.stiffness_signal_floats = {
            name: np.full(
                (frame_count, material_particle_count),
                np.nan,
                dtype=np.float32,
            )
            for name in self.stiffness_signal_float_names
        }
        self.stiffness_signal_masks = {
            name: np.zeros(
                (frame_count, material_particle_count), dtype=bool
            )
            for name in self.stiffness_signal_mask_names
        }
        self.stiffness_signal_proposal_count = np.full(
            frame_count, -1, dtype=np.int32
        )
        self.stiffness_signal_committed_update_count = np.zeros(
            frame_count, dtype=np.int32
        )
        self.stiffness_signal_status = np.full(frame_count, "none", dtype="<U32")
        self.global_system_id_metric_names = (
            "global_distance_scale",
            "global_velocity_damping_per_second",
            "global_coupling_gain",
            "autograd_gradient_log_distance",
            "autograd_gradient_log_damping",
            "autograd_gradient_log_coupling",
            "autograd_raw_gradient_log_distance",
            "autograd_raw_gradient_log_damping",
            "autograd_raw_gradient_log_coupling",
            "autograd_gradient_finite_count",
            "autograd_gradient_nonfinite_count",
            "warp_fd_gradient_log_distance",
            "warp_fd_gradient_log_damping",
            "warp_fd_gradient_log_coupling",
            "gradient_direction_cosine",
            "global_position_point_mean_loss",
            "global_position_median_band_loss",
            "global_position_region_mean_loss",
            "global_position_tail_region_loss",
            "global_relative_track_loss",
            "global_relative_track_weight",
            "global_relative_track_pair_count",
            "global_cauchy_scale",
            "global_track_loss_count",
            "global_coupling_fixed",
            "global_horizon_weight_h1",
            "global_horizon_weight_h2",
            "global_horizon_weight_h3",
            "global_position_point_mean_weight",
            "global_position_region_balance_weight",
            "global_position_tail_region_weight",
            "warp_adam_step_maximum",
            "warp_gradient_accumulator_count",
            "warp_gradient_consensus_norm",
            "candidate_parameter_step_maximum",
            "gradient_cosine_minimum",
            "warp_fd_gradient_norm",
            "warp_fd_local_gradient_minimum",
            "warp_fd_local_gradient_median",
            "warp_fd_local_gradient_maximum",
            "warp_fd_local_gradient_sum",
            "warp_fd_short_horizon_gradient_norm",
            "warp_fd_long_horizon_gradient_norm",
            "warp_fd_short_long_gradient_cosine",
            "short_constraint_pre_directional_derivative",
            "short_constraint_target_directional_derivative",
            "short_constraint_post_directional_derivative",
            "short_constraint_projection_applied",
            "short_constraint_projection_norm",
            "short_horizon_descent_margin",
            "local_relative_track_loss",
            "local_relative_track_pair_count",
            "local_track_loss_count",
            "local_spatial_prior_loss",
            "local_weighted_particle_log_mean",
            "hierarchical_global_log_distance",
            "local_particle_log_minimum",
            "local_particle_log_median",
            "local_particle_log_maximum",
            "local_horizon_weight_h1",
            "local_horizon_weight_h3",
            "local_horizon_weight_h5",
            "local_parameter_excluded_particles",
            "local_velocity_damping_per_second",
            "graph_lm_cg_residual",
            "graph_lm_damping",
            "graph_lm_spatial_hessian_weight",
            "graph_lm_step_maximum",
            "graph_global_log_distance",
            "graph_particle_log_minimum",
            "graph_particle_log_median",
            "graph_particle_log_maximum",
            "graph_distance_minimum",
            "graph_distance_median",
            "graph_distance_maximum",
            "graph_shape_minimum",
            "graph_shape_median",
            "graph_shape_maximum",
            "graph_directional_probe_count",
            "graph_directional_sign_agreement",
            "graph_autograd_directional_norm",
            "graph_warp_directional_norm",
            "observable_allowed",
            "observable_transition_count",
            "observable_used_transition_count",
            "observable_segment_count",
            "observable_residual_count",
            "observable_jacobian_norm",
            "observable_singular_value_maximum",
            "observable_singular_value_minimum",
            "observable_singular_value_ratio",
            "observable_singular_ratio_minimum",
            "observable_fd_scale_cosine",
            "observable_fd_scale_cosine_minimum",
            "observable_gradient_log_distance",
            "observable_gradient_log_damping",
            "observable_lm_damping",
            "observable_parameter_prior_weight",
            "observable_predicted_reduction",
            "observable_log_distance_step",
            "observable_log_damping_step",
            "observable_used_short_transition_count",
            "observable_short_residual_count",
            "observable_short_horizon_weight",
            "observable_long_horizon_weight",
            "observable_short_gradient_log_distance",
            "observable_short_gradient_log_damping",
            "observable_short_directional_before",
            "observable_short_directional_after",
            "observable_short_directional_uncertainty",
            "observable_long_directional_after",
            "observable_short_descent_projected",
            "warp_lm_step_maximum",
        )
        self.global_system_id_metrics = {
            name: np.full(frame_count, np.nan, dtype=np.float64)
            for name in self.global_system_id_metric_names
        }
        self._last_exported_signal_proposal_count = 0
        self.rgb_alignment_loss = np.full(frame_count, np.nan, dtype=np.float64)
        self.rgb_camera_alignment_loss = np.full(
            (frame_count, len(CAMERAS)), np.nan, dtype=np.float64
        )
        self.rgb_alignment_reused_residual_validation = np.zeros(
            frame_count, dtype=bool
        )

        frames = self.environment.frames
        if frames is None:
            raise ValueError("仿真评估需要离线相机帧")
        missing_cameras = [camera for camera in CAMERAS if camera not in frames.names]
        if missing_cameras:
            raise ValueError(f"仿真评估缺少相机：{missing_cameras}")
        self.camera_indices = {
            camera: frames.names.index(camera) for camera in CAMERAS
        }
        self.camera_intrinsics = {
            camera: frames.Ks_cpu[index].detach().cpu().numpy().astype(np.float64)
            for camera, index in self.camera_indices.items()
        }
        self.camera_X_CW = {
            camera: frames.X_CWs_opencv_gpu[index]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
            for camera, index in self.camera_indices.items()
        }
        self.width = int(frames.width)
        self.height = int(frames.height)
        if self.render_images:
            for camera in CAMERAS:
                (self.output_directory / "renders/rgb" / camera).mkdir(
                    parents=True, exist_ok=True
                )
                (self.output_directory / "renders/alpha" / camera).mkdir(
                    parents=True, exist_ok=True
                )

    def _is_holdout_frame(self, frame_index: int) -> bool:
        return bool(
            self.holdout_stride is not None
            and int(frame_index) % self.holdout_stride == self.holdout_offset
        )

    def _should_render_frame(self, frame_index: int) -> bool:
        if not self.render_images:
            return False
        if self.render_frame_mode == "all":
            return True
        if self.render_frame_mode == "holdout":
            return self._is_holdout_frame(frame_index)
        if self.render_frame_mode == "future":
            assert self.open_loop_start_frame is not None
            return int(frame_index) >= int(self.open_loop_start_frame)
        raise ValueError(f"未知渲染帧模式：{self.render_frame_mode}")

    @staticmethod
    def _three_stats(values: torch.Tensor) -> np.ndarray:
        values = values.detach().float()
        return np.asarray(
            (
                float(values.min().item()),
                float(values.median().item()),
                float(values.max().item()),
            ),
            dtype=np.float32,
        )

    def _render_tissue(self, frame_index: int) -> None:
        simulator = self.environment.sim
        frames = self.environment.frames
        assert frames is not None
        soft_ids = simulator.gaussian_model.soft_gaussian_ids.long()
        tissue_state = simulator.gaussian_state.slice(soft_ids)
        selected = torch.as_tensor(
            [self.camera_indices[camera] for camera in CAMERAS],
            device=frames.Ks_gpu.device,
            dtype=torch.long,
        )
        rendered, alpha, _info = simulator.render_gaussians(
            tissue_state,
            frames.X_CWs_opencv_gpu[selected],
            frames.Ks_gpu[selected],
            self.width,
            self.height,
            torch.zeros(3, dtype=torch.float32, device=frames.Ks_gpu.device),
        )
        rgb = (
            rendered[..., :3]
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        for camera_index, camera in enumerate(CAMERAS):
            Image.fromarray(rgb[camera_index], mode="RGB").save(
                self.output_directory
                / "renders/rgb"
                / camera
                / f"{frame_index:06d}.png"
            )
            alpha_image = (
                alpha[camera_index]
                .squeeze(-1)
                .clamp(0.0, 1.0)
                .mul(255.0)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            Image.fromarray(alpha_image, mode="L").save(
                self.output_directory
                / "renders/alpha"
                / camera
                / f"{frame_index:06d}.png"
            )

    def write_frame(self, frame_index: int) -> None:
        if frame_index not in self._frame_to_slot:
            raise ValueError(f"帧 {frame_index} 不在当前评估计划中")
        slot = self._frame_to_slot[frame_index]
        if self._written[slot]:
            raise RuntimeError(f"评估帧 {frame_index} 被重复写入")
        particle_positions = wp.to_torch(
            self.environment.sim.state_0.particle_q
        ).detach()
        ids = torch.as_tensor(
            self.surface_particle_ids_global,
            device=particle_positions.device,
            dtype=torch.long,
        )
        weights = torch.as_tensor(
            self.surface_weights,
            device=particle_positions.device,
            dtype=particle_positions.dtype,
        )
        mapped = (
            particle_positions[ids] * weights[..., None]
        ).sum(dim=1).cpu().numpy().astype(np.float32)
        self.positions[slot] = mapped
        for camera in CAMERAS:
            uv, valid = _project_points(
                mapped,
                self.camera_X_CW[camera],
                self.camera_intrinsics[camera],
                self.width,
                self.height,
            )
            self.projected_uv[camera][slot] = uv
            self.projected_valid[camera][slot] = valid

        projector = self.environment.sim.material_projector
        if projector is not None:
            distance = wp.to_torch(projector.paper_distance_stiffness)
            shape = wp.to_torch(projector.paper_shape_stiffness)
            self.distance_stats[slot] = self._three_stats(distance)
            self.shape_stats[slot] = self._three_stats(shape)
            self.distance_fields[slot] = (
                distance.detach().float().cpu().numpy().astype(np.float32)
            )
            self.shape_fields[slot] = (
                shape.detach().float().cpu().numpy().astype(np.float32)
            )
        updater = self.controls.stiffness_updater
        if updater is not None:
            self.stiffness_signal_committed_update_count[slot] = int(
                updater.update_count
            )
            proposal_count = int(updater.last_proposal_candidate_count)
            diagnostics = updater.last_proposal_diagnostics
            if (
                diagnostics is not None
                and proposal_count > self._last_exported_signal_proposal_count
            ):
                self.stiffness_signal_proposal_count[slot] = proposal_count
                metrics = updater.last_metrics or {}
                self.stiffness_signal_status[slot] = str(
                    metrics.get("status", "candidate")
                )
                for name in self.global_system_id_metric_names:
                    value = metrics.get(name)
                    if value is not None:
                        self.global_system_id_metrics[name][slot] = float(value)
                for name in self.stiffness_signal_float_names:
                    self.stiffness_signal_floats[name][slot] = (
                        diagnostics[name]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                for name in self.stiffness_signal_mask_names:
                    self.stiffness_signal_masks[name][slot] = (
                        diagnostics[name]
                        .detach()
                        .to(dtype=torch.bool)
                        .cpu()
                        .numpy()
                    )
                self._last_exported_signal_proposal_count = proposal_count
            if updater.settings.update_mode in {
                "differentiable_hierarchical_relative",
                "differentiable_particle_graph_lm",
            }:
                # Export the verified coefficient every frame. Candidate
                # metrics may contain a proposal that was later rejected.
                self.global_system_id_metrics[
                    "hierarchical_global_log_distance"
                ][slot] = float(
                    updater.low_dimensional_log_coefficients[0].item()
                )
        mapper = self.controls.visual_residual_mapper
        feedback_observation_allowed = (
            (
                self.open_loop_start_frame is None
                or int(frame_index) < int(self.open_loop_start_frame)
            )
            and not self._is_holdout_frame(frame_index)
        )
        if (
            mapper is not None
            and feedback_observation_allowed
            and self.compute_rgb_alignment
        ):
            metrics = self.environment.sim.last_visual_tissue_residual_metrics
            if (
                metrics is not None
                and int(metrics.get("frame_index", -1)) == int(frame_index)
            ):
                self.rgb_alignment_loss[slot] = float(
                    metrics["exact_final_visual_loss"]
                )
                self.rgb_camera_alignment_loss[slot] = np.asarray(
                    metrics["exact_final_camera_visual_losses"],
                    dtype=np.float64,
                )
                self.rgb_alignment_reused_residual_validation[slot] = True
            else:
                # RGB-only diagnostic render: it is exported after estimation
                # and never enters the state or material update. It reads no
                # depth, trajectory ground truth, or material ground truth.
                alignment = self.environment.sim.evaluate_visual_tissue_alignment(
                    mapper,
                    self.environment.frames,
                    observations_are_bgr=True,
                )
                self.rgb_alignment_loss[slot] = alignment.loss
                self.rgb_camera_alignment_loss[slot] = np.asarray(
                    alignment.camera_losses, dtype=np.float64
                )
        if self._should_render_frame(frame_index):
            self._render_tissue(frame_index)
        self._written[slot] = True

    def close(self) -> None:
        missing = self.frame_indices[~self._written]
        if len(missing):
            raise RuntimeError(
                f"评估产物缺少 {len(missing)} 帧，前十帧：{missing[:10].tolist()}"
            )
        np.savez_compressed(
            self.output_directory / "predicted_trajectories.npz",
            schema=np.asarray("fixedsuperbest.sim_prediction_trajectories.v1"),
            frame_indices=self.frame_indices,
            timestamps=self.timestamps,
            tissue_node_ids=self.node_ids,
            tissue_positions_world=self.positions,
            stereo_left_tissue_uv_pixels=self.projected_uv["stereo_left"],
            stereo_right_tissue_uv_pixels=self.projected_uv["stereo_right"],
            stereo_left_tissue_valid=self.projected_valid["stereo_left"],
            stereo_right_tissue_valid=self.projected_valid["stereo_right"],
            reconstruction_surface_particle_ids=(
                self.surface_particle_ids_local.astype(np.int32)
            ),
            reconstruction_surface_barycentric_weights=self.surface_weights,
            reference_first_positions_world=self.reference_first_positions,
            embedded_first_positions_world=self.initial_embedded_positions,
        )
        np.savez_compressed(
            self.output_directory / "material_diagnostics.npz",
            frame_indices=self.frame_indices,
            timestamps=self.timestamps,
            statistic_names=np.asarray(("minimum", "median", "maximum")),
            distance_stiffness=self.distance_stats,
            shape_stiffness=self.shape_stats,
            reconstruction_particle_ids=self.material_particle_ids,
            distance_stiffness_per_particle=self.distance_fields,
            shape_stiffness_per_particle=self.shape_fields,
        )
        np.savez_compressed(
            self.output_directory / "stiffness_signal_diagnostics.npz",
            schema=np.asarray("fixedsuperbest.stiffness_signal_diagnostics.v1"),
            frame_indices=self.frame_indices,
            timestamps=self.timestamps,
            reconstruction_particle_ids=self.material_particle_ids,
            proposal_candidate_count=self.stiffness_signal_proposal_count,
            committed_update_count=self.stiffness_signal_committed_update_count,
            proposal_status=self.stiffness_signal_status,
            **self.global_system_id_metrics,
            **self.stiffness_signal_floats,
            **self.stiffness_signal_masks,
        )
        np.savez_compressed(
            self.output_directory / "rgb_alignment_diagnostics.npz",
            schema=np.asarray("fixedsuperbest.sim_rgb_alignment.v1"),
            frame_indices=self.frame_indices,
            timestamps=self.timestamps,
            camera_names=np.asarray(CAMERAS),
            equal_camera_rgb_loss=self.rgb_alignment_loss,
            per_camera_rgb_loss=self.rgb_camera_alignment_loss,
            reused_residual_exact_validation=(
                self.rgb_alignment_reused_residual_validation
            ),
        )
        embedding_mm = self.initial_embedding_errors_m.astype(np.float64) * 1.0e3
        mapper = self.controls.visual_residual_mapper
        residual_settings = mapper.settings if mapper is not None else None
        updater = self.controls.stiffness_updater
        stiffness_settings = updater.settings if updater is not None else None
        metadata = {
            "schema": "fixedsuperbest.sim_ablation_artifacts.v1",
            "label": self.label,
            "reference_dataset": str(self.dataset_root),
            "frames": int(len(self.frame_indices)),
            "frame_start": int(self.frame_indices[0]),
            "frame_end_inclusive": int(self.frame_indices[-1]),
            "physics_steps_per_frame": self.physics_steps_per_frame,
            "open_loop_start_frame": self.open_loop_start_frame,
            "holdout_stride": self.holdout_stride,
            "holdout_offset": (
                self.holdout_offset
                if self.holdout_stride is not None
                else None
            ),
            "holdout_frame_indices": [
                int(frame)
                for frame in self.frame_indices
                if self._is_holdout_frame(int(frame))
            ],
            "render_frame_mode": self.render_frame_mode,
            "visual_feedback_mode": self.controls.visual_feedback_mode,
            "flow_depth_initial_alignment": getattr(
                self.controls,
                "flow_depth_initial_alignment",
                None,
            ),
            "online_stiffness_update": self.controls.stiffness_updater
            is not None,
            "visual_residual_settings": (
                {
                    "iterations": residual_settings.iterations,
                    "learning_rate_m": residual_settings.learning_rate_m,
                    "maximum_residual_m": residual_settings.maximum_residual_m,
                    "robust_loss_beta": residual_settings.robust_loss_beta,
                    "data_weight": residual_settings.data_weight,
                    "distance_weight": residual_settings.distance_weight,
                    "volume_weight": residual_settings.volume_weight,
                    "shape_weight": residual_settings.shape_weight,
                    "spatial_weight": residual_settings.spatial_weight,
                    "temporal_weight": residual_settings.temporal_weight,
                    "magnitude_weight": residual_settings.magnitude_weight,
                    "reference_delta_calibration": (
                        getattr(
                            residual_settings,
                            "reference_delta_calibration",
                            False,
                        )
                    ),
                    "minimum_actionable_visual_loss": (
                        getattr(
                            residual_settings,
                            "minimum_actionable_visual_loss",
                            0.0,
                        )
                    ),
                    "task_gate": (
                        "frame_0_initialization_then_recorded_grasp_frames_only; "
                        "disabled_at_open_loop_split"
                    ),
                }
                if residual_settings is not None
                else None
            ),
            "online_stiffness_settings": (
                {
                    "log_learning_rate": stiffness_settings.log_learning_rate,
                    "maximum_log_step": stiffness_settings.maximum_log_step,
                    "signal_ema_decay": stiffness_settings.signal_ema_decay,
                    "spatial_smoothing_iterations": (
                        stiffness_settings.spatial_smoothing_iterations
                    ),
                    "spatial_smoothing_blend": (
                        stiffness_settings.spatial_smoothing_blend
                    ),
                    "strain_signal_weight": (
                        stiffness_settings.strain_signal_weight
                    ),
                    "update_mode": stiffness_settings.update_mode,
                    "autograd_unroll_steps": (
                        stiffness_settings.autograd_unroll_steps
                    ),
                    "autograd_region_count": (
                        stiffness_settings.autograd_region_count
                    ),
                    "gradient_cosine_minimum": (
                        stiffness_settings.gradient_cosine_minimum
                    ),
                    "local_finite_difference_log_epsilon": (
                        stiffness_settings.local_finite_difference_log_epsilon
                    ),
                    "autograd_frame_dt": (
                        stiffness_settings.autograd_frame_dt
                    ),
                    "autograd_material_iterations": (
                        stiffness_settings.autograd_material_iterations
                    ),
                    "autograd_shape_update_gain": (
                        stiffness_settings.autograd_shape_update_gain
                    ),
                    "autograd_maximum_log_offset": (
                        stiffness_settings.autograd_maximum_log_offset
                    ),
                    "local_horizon_weights_h1_h3_h5": list(
                        stiffness_settings.local_horizon_weights
                    ),
                    "local_update_interval_frames": (
                        stiffness_settings.local_update_interval_frames
                    ),
                    "local_spatial_prior_weight": (
                        stiffness_settings.local_spatial_prior_weight
                    ),
                    "graph_lm_damping": stiffness_settings.graph_lm_damping,
                    "graph_lm_spatial_hessian_weight": (
                        stiffness_settings.graph_lm_spatial_hessian_weight
                    ),
                    "graph_lm_cg_iterations": (
                        stiffness_settings.graph_lm_cg_iterations
                    ),
                    "graph_observability_floor": (
                        stiffness_settings.graph_observability_floor
                    ),
                    "graph_shape_log_coupling": (
                        stiffness_settings.graph_shape_log_coupling
                    ),
                    "graph_directional_probe_count": (
                        stiffness_settings.graph_directional_probe_count
                    ),
                    "graph_directional_log_epsilon": (
                        stiffness_settings.graph_directional_log_epsilon
                    ),
                    "graph_directional_cosine_minimum": (
                        stiffness_settings.graph_directional_cosine_minimum
                    ),
                    "graph_update_interval_frames": (
                        stiffness_settings.graph_update_interval_frames
                    ),
                    "hierarchical_short_horizon_descent_margin": (
                        stiffness_settings.hierarchical_short_horizon_descent_margin
                    ),
                    "observable_window_size": (
                        stiffness_settings.observable_window_size
                    ),
                    "observable_minimum_transitions": (
                        stiffness_settings.observable_minimum_transitions
                    ),
                    "observable_update_interval_frames": (
                        stiffness_settings.observable_update_interval_frames
                    ),
                    "observable_minimum_singular_ratio": (
                        stiffness_settings.observable_minimum_singular_ratio
                    ),
                    "observable_lm_damping": (
                        stiffness_settings.observable_lm_damping
                    ),
                    "observable_parameter_prior_weight": (
                        stiffness_settings.observable_parameter_prior_weight
                    ),
                    "observable_short_horizon_weight": (
                        stiffness_settings.observable_short_horizon_weight
                    ),
                    "observable_long_horizon_weight": (
                        stiffness_settings.observable_long_horizon_weight
                    ),
                    "autograd_adam_step": updater.adam_step,
                    "update_policy": (
                        "one global mean plus one zero-mean log-distance value "
                        "per non-controlled particle; weak tied shape and fixed "
                        "volume/damping/coupling; observation-weighted graph "
                        "diagonal GN/LM; deterministic Warp directional-FD "
                        "check; no future, GT material, or parameter selection"
                        if stiffness_settings.update_mode
                        == "differentiable_particle_graph_lm"
                        else
                        "joint global-mean plus zero-mean regional distance "
                        "field from common uniform initialization; damping "
                        "and grasp coupling fixed; Cauchy-relative pre-residual "
                        "H3/H5 causal Warp-gradient Adam projected into an H1 "
                        "reconstruction-descent half-space; no GT material, "
                        "no branch selection"
                        if stiffness_settings.update_mode
                        == "differentiable_hierarchical_relative"
                        else
                        "local zero-mean distance field; fixed global mean, "
                        "damping and grasp coupling; Cauchy-relative "
                        "pre-residual H1/H3/H5 causal Warp-gradient Adam; "
                        "no branch selection"
                        if stiffness_settings.update_mode
                        == "differentiable_local_relative"
                        else
                        "delayed causal low-dimensional differentiable "
                        "distance+volume+shape PBD Adam over observed frames; "
                        "no branch selection"
                        if stiffness_settings.update_mode
                        == "differentiable_low_dim"
                        else
                        "causal long-primary 20-transition multi-shooting "
                        "actual-Warp finite-difference Jacobian; H1 "
                        "epsilon-vs-2epsilon uncertainty-constrained "
                        "Hessian projection; global distance+damping "
                        "observable robust LM; shape, volume and grasp "
                        "fixed; no future and no branch selection"
                        if stiffness_settings.update_mode
                        == "differentiable_global_mhe"
                        else "causal RGB-corrected edge strain; no branch selection"
                    ),
                    "candidate_count": updater.candidate_count,
                    "committed_update_count": updater.update_count,
                    "rejected_update_count": updater.rejected_count,
                    "distance_bounds": [
                        stiffness_settings.distance_minimum,
                        stiffness_settings.distance_maximum,
                    ],
                    "shape_bounds": [
                        stiffness_settings.shape_minimum,
                        stiffness_settings.shape_maximum,
                    ],
                }
                if stiffness_settings is not None
                else None
            ),
            "stiffness_signal_diagnostic": {
                "path": "stiffness_signal_diagnostics.npz",
                "schema": "fixedsuperbest.stiffness_signal_diagnostics.v1",
                "float_fields": list(self.stiffness_signal_float_names),
                "mask_fields": list(self.stiffness_signal_mask_names),
                "semantics": (
                    "每个新刚度候选保存RGB残差幅值、物理形变幅值、向量信号、"
                    "边应变信号及置信度、平滑前后信号、EMA、log步长和全部门禁掩码"
                ),
            },
            "known_grasp_boundary": {
                "mode": getattr(
                    self.environment,
                    "super_sim_grasp_boundary_mode",
                    None,
                ),
                "schema": getattr(
                    self.environment,
                    "super_sim_grasp_boundary_schema",
                    None,
                ),
                "path": getattr(
                    self.environment,
                    "super_sim_grasp_boundary_path",
                    None,
                ),
                "controlled_particle_count": getattr(
                    self.environment,
                    "super_sim_grasp_boundary_particle_count",
                    None,
                ),
            },
            "future_feedback_policy": (
                "进入 open_loop_start_frame 前取消尚未完成验证的刚度候选；"
                "从该帧起关闭视觉残差、禁止新刚度候选并保持已提交材料场；"
                "PSM 条件和所选已知夹持区域位移边界继续输入"
                if self.open_loop_start_frame is not None
                else "未启用未来开环切分"
            ),
            "reconstruction_holdout_policy": (
                "每连续 8 帧的前 7 帧允许因果 RGB/刚度更新；第 8 帧"
                "禁止视觉残差、刚度提议和未完成候选验证，只保存预测用于测试"
                if self.holdout_stride is not None
                else "未启用周期重建留出"
            ),
            "trajectory_mapping": (
                "首帧真值评估点到重建顶面三角形的一次性重心坐标映射；"
                "后续不读取真值运动且不做逐帧配准"
            ),
            "evaluation_nodes": int(len(self.node_ids)),
            "initial_embedding_error_mm": {
                "mean": float(embedding_mm.mean()),
                "median": float(np.median(embedding_mm)),
                "p95": float(np.percentile(embedding_mm, 95.0)),
                "maximum": float(embedding_mm.max()),
            },
            "render_images": self.render_images,
            "rgb_alignment_diagnostic": {
                "path": "rgb_alignment_diagnostics.npz",
                "input": (
                    "RGB only before open_loop_start_frame; no future RGB, "
                    "depth, trajectory GT, or material GT"
                ),
                "resolution_scale": (
                    residual_settings.image_scale
                    if residual_settings is not None
                    else None
                ),
                "loss": "equal-camera masked robust RGB loss",
            },
            "render_layer": (
                "仅组织 Gaussian、黑色背景，并同时保存 alpha；用于与真值组织层比较"
                if self.render_images
                else None
            ),
            "cameras": list(CAMERAS),
            "resolution": [self.width, self.height],
        }
        (self.output_directory / "artifact_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
