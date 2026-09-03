# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
import math

import numpy as np
import torch
import warp as wp
import warp.sim
from scipy.spatial.transform import Rotation

from embodied_gaussians.physics_simulator.builder import ModelBuilder
from embodied_gaussians.physics_simulator.integrator import (
    MaterialTetrahedronXPBDProjector,
    XPBDIntegrator,
)
from embodied_gaussians.physics_simulator.triangle_skin_contact import (
    PersistentGripSnapshot,
    TriangleSkinContactProjector,
)
from embodied_gaussians.utils.physics_utils import (
    clone_control,
    clone_state,
    copy_control,
    copy_state,
    cuda_graph_capture,
    synchronize_control,
    synchronize_state,
    transform_from_matrix,
    transform_to_matrix,
)


@dataclass
class PhysicsRolloutAuxiliaryState:
    """Mutable simulator-owned state not included in ``warp.sim.State``."""

    sim_time: float
    kinematic_interpolation_start_q: torch.Tensor | None
    kinematic_interpolation_target_q: torch.Tensor | None
    persistent_grip: PersistentGripSnapshot | None
    paper_distance_stiffness: torch.Tensor | None
    paper_shape_stiffness: torch.Tensor | None


@wp.kernel(enable_backward=False)
def mask_particle_shape_contact_geometries(
    shape_geo_type: wp.array(dtype=wp.int32),
    contact_shape_enabled: wp.array(dtype=wp.int32),
    saved_shape_geo_type: wp.array(dtype=wp.int32),
):
    """Temporarily hide non-participating shapes from Warp soft collide()."""
    shape_id = wp.tid()
    saved_shape_geo_type[shape_id] = shape_geo_type[shape_id]
    if contact_shape_enabled[shape_id] == 0:
        shape_geo_type[shape_id] = wp.sim.GEO_NONE


@wp.kernel(enable_backward=False)
def restore_particle_shape_contact_geometries(
    shape_geo_type: wp.array(dtype=wp.int32),
    saved_shape_geo_type: wp.array(dtype=wp.int32),
):
    shape_id = wp.tid()
    shape_geo_type[shape_id] = saved_shape_geo_type[shape_id]


@wp.kernel(enable_backward=False)
def gather_kinematic_body_poses(
    body_ids: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    gathered_q: wp.array(dtype=wp.transform),
):
    local_id = wp.tid()
    gathered_q[local_id] = body_q[body_ids[local_id]]


@wp.kernel(enable_backward=False)
def interpolate_kinematic_body_poses(
    body_ids: wp.array(dtype=wp.int32),
    start_q: wp.array(dtype=wp.transform),
    target_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    alpha: float,
    frame_dt: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    local_id = wp.tid()
    start = start_q[local_id]
    target = target_q[local_id]
    start_position = wp.transform_get_translation(start)
    target_position = wp.transform_get_translation(target)
    start_rotation = wp.transform_get_rotation(start)
    target_rotation = wp.transform_get_rotation(target)
    # Pick the quaternion representative used by shortest-path slerp so the
    # derived angular velocity obeys the same physical rotation.
    rotation_dot = (
        start_rotation[0] * target_rotation[0]
        + start_rotation[1] * target_rotation[1]
        + start_rotation[2] * target_rotation[2]
        + start_rotation[3] * target_rotation[3]
    )
    if rotation_dot < 0.0:
        target_rotation = wp.quat(
            -target_rotation[0],
            -target_rotation[1],
            -target_rotation[2],
            -target_rotation[3],
        )
    position = start_position + alpha * (target_position - start_position)
    rotation = wp.quat_slerp(start_rotation, target_rotation, alpha)
    body_id = body_ids[local_id]
    body_q[body_id] = wp.transform(position, rotation)

    # Directly written kinematic poses otherwise retain zero body velocity,
    # so jaw closure has no tangential pushing effect in contact.  Store the
    # frame-consistent COM twist used by the triangle-skin friction solve.
    if frame_dt > 0.0:
        delta_rotation = wp.mul(
            target_rotation, wp.quat_inverse(start_rotation)
        )
        rotation_axis = wp.vec3()
        rotation_angle = wp.float32(0.0)
        wp.quat_to_axis_angle(
            delta_rotation, rotation_axis, rotation_angle
        )
        angular_velocity = rotation_axis * (rotation_angle / frame_dt)
        start_com = wp.transform_point(start, body_com[body_id])
        target_with_short_rotation = wp.transform(
            target_position, target_rotation
        )
        target_com = wp.transform_point(
            target_with_short_rotation, body_com[body_id]
        )
        com_velocity = (target_com - start_com) / frame_dt
        body_qd[body_id] = wp.spatial_vector(
            angular_velocity, com_velocity
        )
    else:
        body_qd[body_id] = wp.spatial_vector()


@dataclass
class PhysicsSettings:
    substeps: int = 30
    xpbd_iterations: int = 8
    dt: float = 1.0 / 60.0
    use_project_material_tetrahedra: bool = False
    material_iterations: int = 20
    material_relaxation: float = 0.15
    material_compliance_scale: float = 1.0
    material_min_volume_ratio: float = 1.0e-4
    tetrahedral_constraint_model: str = "neo_hookean"
    paper_distance_stiffness: float = 0.2
    paper_volume_stiffness: float = 1.0e10
    paper_shape_stiffness: float = 0.005
    preserve_spatial_paper_stiffness: bool = False
    particle_velocity_damping_per_second: float = 12.0
    material_projection_velocity_scale: float = 1.0
    contact_projection_velocity_scale: float = 1.0
    particle_ground_relaxation: float = 0.9
    particle_shape_contact_relaxation: float = 0.9
    particle_shape_contact_max_correction_m: float = 0.0
    particle_shape_contact_min_volume_ratio: float = 0.05
    enable_particle_shape_contacts: bool = True
    enable_particle_particle_contacts: bool = True
    enable_triangle_skin_contacts: bool = False
    triangle_skin_contact_margin_m: float = 0.0004
    triangle_skin_query_distance_m: float = 0.010
    triangle_skin_ccd_velocity_scale: float = 1.0
    triangle_skin_friction: float = 0.35
    triangle_skin_contact_relaxation: float = 0.7
    triangle_skin_contact_max_correction_m: float = 0.0004
    # Zero inherits the general contact cap.  A smaller positive value softens
    # only the oriented top-sheet normal barrier, leaving jaw tangential
    # closure and grip contact on the general budget.
    triangle_skin_top_barrier_max_correction_m: float = 0.0
    triangle_skin_contact_iterations: int = 1
    triangle_skin_post_contact_material_iterations: int = 0
    triangle_skin_final_barrier_max_correction_m: float = 0.0
    triangle_skin_contact_min_volume_ratio: float = 0.05
    triangle_skin_contact_substep_stride: int = 1
    preserve_static_body_poses: bool = False


class Simulator:
    def __init__(
        self, builder: ModelBuilder, device: str = "cuda", requires_grad: bool = False
    ):
        self.builder = builder
        self.device = device
        self.model = builder.finalize(device, requires_grad)
        self.num_envs = self.model.num_envs
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control: warp.sim.Control = self.model.control()
        self.material_projector = (
            MaterialTetrahedronXPBDProjector(self.model)
            if self.model.tet_count > 0
            else None
        )
        self.particle_shape_contact_filter = None
        self._saved_shape_geo_type = None
        self.triangle_skin_contact_projector = None
        self.kinematic_interpolation_body_ids = None
        self.kinematic_interpolation_start_q = None
        self.kinematic_interpolation_target_q = None
        self.sim_time = 0.0
        self.eval_fk()

    def configure_kinematic_body_interpolation(
        self, body_ids: list[int] | tuple[int, ...] | None
    ) -> None:
        """Continuously advance selected kinematic bodies across physics substeps."""
        if body_ids is None:
            self.kinematic_interpolation_body_ids = None
            self.kinematic_interpolation_start_q = None
            self.kinematic_interpolation_target_q = None
        else:
            ids = np.asarray(body_ids, dtype=np.int32)
            if ids.ndim != 1 or len(ids) == 0:
                raise ValueError("Kinematic interpolation requires body ids")
            if ids.min() < 0 or ids.max() >= self.model.body_count:
                raise ValueError("Kinematic interpolation body id is invalid")
            self.kinematic_interpolation_body_ids = wp.array(
                ids, dtype=wp.int32, device=self.model.device
            )
            self.kinematic_interpolation_start_q = wp.empty(
                len(ids), dtype=wp.transform, device=self.model.device
            )
            self.kinematic_interpolation_target_q = wp.empty(
                len(ids), dtype=wp.transform, device=self.model.device
            )
            self.sync_kinematic_body_interpolation()
        if hasattr(self, "_physics_step_cache"):
            delattr(self, "_physics_step_cache")

    def sync_kinematic_body_interpolation(self) -> None:
        """Reset interpolation history to the currently displayed body poses."""
        if self.kinematic_interpolation_body_ids is None:
            return
        wp.launch(
            kernel=gather_kinematic_body_poses,
            dim=len(self.kinematic_interpolation_body_ids),
            inputs=[
                self.kinematic_interpolation_body_ids,
                self.state_0.body_q,
            ],
            outputs=[self.kinematic_interpolation_start_q],
            device=self.model.device,
        )
        wp.copy(
            self.kinematic_interpolation_target_q,
            self.kinematic_interpolation_start_q,
        )

    def _capture_kinematic_body_targets(self) -> None:
        if self.kinematic_interpolation_body_ids is None:
            return
        wp.launch(
            kernel=gather_kinematic_body_poses,
            dim=len(self.kinematic_interpolation_body_ids),
            inputs=[
                self.kinematic_interpolation_body_ids,
                self.state_0.body_q,
            ],
            outputs=[self.kinematic_interpolation_target_q],
            device=self.model.device,
        )

    def configure_triangle_skin_contacts(
        self,
        tool_shape_ids: list[int] | tuple[int, ...] | None,
        *,
        sample_spacing_m: float = 0.00035,
        spread_layers: int = 0,
        top_support_lateral_radius_m: float = 0.004,
        top_support_depth_m: float = 0.012,
        top_support_weight_scale: float = 0.15,
        top_pressure_shoulder_lateral_radius_m: float = 0.0045,
        top_pressure_shoulder_depth_m: float = 0.002,
        top_pressure_shoulder_upward_scale: float = 1.0,
        top_pressure_shoulder_outward_scale: float = 0.35,
        top_pressure_shoulder_bias_direction_world: tuple[float, float, float] = (
            0.0,
            0.0,
            0.0,
        ),
        top_pressure_shoulder_bias_start_m: float = 0.0008,
        top_barrier_lateral_tolerance_m: float = 0.00045,
        top_barrier_contact_patch_radius_m: float = 0.0,
        top_barrier_clearance_m: float = 0.0004,
        top_barrier_shape_ids: list[int] | tuple[int, ...] | None = None,
        jaw_contact_shape_ids: list[int] | tuple[int, ...] | None = None,
        jaw_contact_distal_length_m: float = 0.0,
        top_barrier_distal_length_m: float = 0.0,
        top_barrier_tip_allowance_m: float = 0.0,
        jaw_friction_coefficient: float = 1.5,
        persistent_grip_enabled: bool = False,
        persistent_grip_minimum_contact_samples_per_jaw: int = 8,
        persistent_grip_nearest_surface_particles: int = 4,
        persistent_grip_maximum_jaw_patch_separation_m: float = 0.006,
        persistent_grip_activation_steps: int = 3,
        persistent_grip_maximum_capture_penetration_m: float = 0.0025,
        persistent_grip_minimum_capture_volume_ratio: float = 0.01,
        persistent_grip_closed_angle_max_rad: float = 0.10,
        persistent_grip_release_angle_min_rad: float = 0.15,
        persistent_grip_release_angle_delta_rad: float = 0.08,
        persistent_grip_wide_open_angle_rad: float = 0.50,
        persistent_grip_angle_motion_epsilon_rad: float = 1.0e-4,
        persistent_grip_compliance_m_per_n: float = 25.0,
        persistent_grip_relaxation: float = 0.35,
        persistent_grip_maximum_correction_m: float = 0.0001,
        persistent_grip_transfer_layers: int = 3,
        persistent_grip_minimum_volume_ratio: float = 1.0e-8,
        persistent_grip_support_radius_m: float = 0.0,
        persistent_grip_support_generations: int = 4,
    ) -> None:
        """Configure deforming triangle-skin contact against tool mesh proxies."""
        if tool_shape_ids is None:
            self.triangle_skin_contact_projector = None
        else:
            if self.num_envs != 1:
                raise ValueError(
                    "Triangle-skin contact currently supports one environment"
                )
            faces = np.asarray(
                getattr(self.builder, "soft_collision_skin_faces", []),
                dtype=np.int32,
            ).reshape(-1, 3)
            if len(faces) == 0:
                raise ValueError(
                    "No enabled soft collision-skin faces were added to the builder"
                )
            face_markers = np.asarray(
                getattr(
                    self.builder,
                    "soft_collision_skin_face_markers",
                    [],
                ),
                dtype=np.uint8,
            )
            if face_markers.shape != (len(faces),):
                raise ValueError(
                    "Soft collision-skin face markers do not match faces"
                )
            top_faces = (
                faces[face_markers == 1]
                if top_barrier_shape_ids
                else np.empty((0, 3), dtype=np.int32)
            )
            self.triangle_skin_contact_projector = (
                TriangleSkinContactProjector(
                    self.model,
                    faces,
                    tool_shape_ids,
                    sample_spacing_m=sample_spacing_m,
                    spread_layers=spread_layers,
                    top_skin_faces=top_faces,
                    top_support_lateral_radius_m=(
                        top_support_lateral_radius_m
                    ),
                    top_support_depth_m=top_support_depth_m,
                    top_support_weight_scale=top_support_weight_scale,
                    top_pressure_shoulder_lateral_radius_m=(
                        top_pressure_shoulder_lateral_radius_m
                    ),
                    top_pressure_shoulder_depth_m=(
                        top_pressure_shoulder_depth_m
                    ),
                    top_pressure_shoulder_upward_scale=(
                        top_pressure_shoulder_upward_scale
                    ),
                    top_pressure_shoulder_outward_scale=(
                        top_pressure_shoulder_outward_scale
                    ),
                    top_pressure_shoulder_bias_direction_world=(
                        top_pressure_shoulder_bias_direction_world
                    ),
                    top_pressure_shoulder_bias_start_m=(
                        top_pressure_shoulder_bias_start_m
                    ),
                    top_barrier_lateral_tolerance_m=(
                        top_barrier_lateral_tolerance_m
                    ),
                    top_barrier_contact_patch_radius_m=(
                        top_barrier_contact_patch_radius_m
                    ),
                    top_barrier_clearance_m=top_barrier_clearance_m,
                    top_barrier_shape_ids=top_barrier_shape_ids,
                    jaw_contact_shape_ids=jaw_contact_shape_ids,
                    jaw_contact_distal_length_m=(
                        jaw_contact_distal_length_m
                    ),
                    top_barrier_distal_length_m=(
                        top_barrier_distal_length_m
                    ),
                    top_barrier_tip_allowance_m=(
                        top_barrier_tip_allowance_m
                    ),
                    jaw_friction_coefficient=(
                        jaw_friction_coefficient
                    ),
                    persistent_grip_enabled=persistent_grip_enabled,
                    persistent_grip_minimum_contact_samples_per_jaw=(
                        persistent_grip_minimum_contact_samples_per_jaw
                    ),
                    persistent_grip_nearest_surface_particles=(
                        persistent_grip_nearest_surface_particles
                    ),
                    persistent_grip_maximum_jaw_patch_separation_m=(
                        persistent_grip_maximum_jaw_patch_separation_m
                    ),
                    persistent_grip_activation_steps=(
                        persistent_grip_activation_steps
                    ),
                    persistent_grip_maximum_capture_penetration_m=(
                        persistent_grip_maximum_capture_penetration_m
                    ),
                    persistent_grip_minimum_capture_volume_ratio=(
                        persistent_grip_minimum_capture_volume_ratio
                    ),
                    persistent_grip_closed_angle_max_rad=(
                        persistent_grip_closed_angle_max_rad
                    ),
                    persistent_grip_release_angle_min_rad=(
                        persistent_grip_release_angle_min_rad
                    ),
                    persistent_grip_release_angle_delta_rad=(
                        persistent_grip_release_angle_delta_rad
                    ),
                    persistent_grip_wide_open_angle_rad=(
                        persistent_grip_wide_open_angle_rad
                    ),
                    persistent_grip_angle_motion_epsilon_rad=(
                        persistent_grip_angle_motion_epsilon_rad
                    ),
                    persistent_grip_compliance_m_per_n=(
                        persistent_grip_compliance_m_per_n
                    ),
                    persistent_grip_relaxation=persistent_grip_relaxation,
                    persistent_grip_maximum_correction_m=(
                        persistent_grip_maximum_correction_m
                    ),
                    persistent_grip_transfer_layers=(
                        persistent_grip_transfer_layers
                    ),
                    persistent_grip_minimum_volume_ratio=(
                        persistent_grip_minimum_volume_ratio
                    ),
                    persistent_grip_support_radius_m=(
                        persistent_grip_support_radius_m
                    ),
                    persistent_grip_support_generations=(
                        persistent_grip_support_generations
                    ),
                )
            )
        if hasattr(self, "_physics_step_cache"):
            delattr(self, "_physics_step_cache")

    def triangle_skin_contact_metrics(self) -> dict | None:
        if self.triangle_skin_contact_projector is None:
            return None
        metrics = self.triangle_skin_contact_projector.metrics()
        if self.material_projector is not None:
            metrics["material_safety_step_scale"] = float(
                self.material_projector.material_step_scale.numpy()[0]
            )
            metrics["material_safety_unsafe_tetrahedra"] = int(
                self.material_projector.material_unsafe_tet_count.numpy()[0]
            )
        projector = self.triangle_skin_contact_projector
        grip_body = projector.persistent_grip_particle_body.numpy()
        grip_direct = projector.persistent_grip_particle_direct.numpy() != 0
        direct_ids = np.flatnonzero((grip_body >= 0) & grip_direct)
        if len(direct_ids):
            particle_q = self.state_0.particle_q.numpy()
            body_q = self.state_0.body_q.numpy()
            local = projector.persistent_grip_particle_local.numpy()[direct_ids]
            body_ids = grip_body[direct_ids]
            targets = np.empty_like(local)
            for body_id in np.unique(body_ids):
                selected = body_ids == body_id
                pose = body_q[int(body_id)]
                targets[selected] = (
                    Rotation.from_quat(pose[3:7]).apply(local[selected])
                    + pose[:3]
                )
            errors = np.linalg.norm(particle_q[direct_ids] - targets, axis=1)
            metrics["persistent_grip_anchor_error_rms_m"] = float(
                np.sqrt(np.mean(errors * errors))
            )
            metrics["persistent_grip_anchor_error_maximum_m"] = float(
                errors.max()
            )
        else:
            metrics["persistent_grip_anchor_error_rms_m"] = 0.0
            metrics["persistent_grip_anchor_error_maximum_m"] = 0.0
        return metrics

    def triangle_skin_jaw_closure_resistance_metrics(self) -> dict | None:
        """Return only the bilateral closure feedback needed by the jaw motor."""
        if self.triangle_skin_contact_projector is None:
            return None
        return (
            self.triangle_skin_contact_projector
            .jaw_closure_resistance_metrics()
        )

    def set_triangle_skin_jaw_signal(
        self,
        jaw_angle_rad: float,
        timestamp_s: float,
        *,
        contact_limited_actuator_enabled: bool = False,
        closure_blocked_by_tissue: bool = False,
        closing_requested: bool = False,
    ) -> None:
        """Forward the timestamped q7[6] signal to persistent grip."""
        if self.triangle_skin_contact_projector is None:
            return
        self.triangle_skin_contact_projector.set_persistent_grip_jaw_signal(
            jaw_angle_rad,
            timestamp_s,
            contact_limited_actuator_enabled=(
                contact_limited_actuator_enabled
            ),
            closure_blocked_by_tissue=closure_blocked_by_tissue,
            closing_requested=closing_requested,
        )

    def configure_particle_shape_contact_shapes(
        self, shape_ids: list[int] | tuple[int, ...] | None
    ) -> None:
        """Restrict particle-shape contacts to an explicit shape allow-list.

        Warp's soft-contact broad phase otherwise tests every particle against
        every non-ground shape and does not consult the rigid collision flags.
        Passing ``None`` restores Warp's unfiltered behavior; an empty sequence
        intentionally allows no particle-shape contacts.
        """
        if shape_ids is None:
            self.particle_shape_contact_filter = None
            self._saved_shape_geo_type = None
        else:
            unique_shape_ids = sorted({int(shape_id) for shape_id in shape_ids})
            invalid = [
                shape_id
                for shape_id in unique_shape_ids
                if shape_id < 0 or shape_id >= self.model.shape_count
            ]
            if invalid:
                raise ValueError(
                    "Particle-shape contact ids are outside the model: "
                    f"{invalid} (shape_count={self.model.shape_count})"
                )
            enabled = np.zeros(self.model.shape_count, dtype=np.int32)
            enabled[unique_shape_ids] = 1
            self.particle_shape_contact_filter = wp.array(
                enabled, dtype=wp.int32, device=self.model.device
            )
            self._saved_shape_geo_type = wp.empty(
                self.model.shape_count, dtype=wp.int32, device=self.model.device
            )
        # The contact-filter launches and their arrays are captured into the
        # physics CUDA graph, so a runtime reconfiguration needs recapture.
        if hasattr(self, "_physics_step_cache"):
            delattr(self, "_physics_step_cache")

    def clone_control(self):
        return clone_control(self.control)

    def clone_state(self):
        return clone_state(self.state_0)

    def clone_rollout_auxiliary_state(self) -> PhysicsRolloutAuxiliaryState:
        """Clone stateful projectors and interpolation history for rollouts."""
        interpolation_start = (
            None
            if self.kinematic_interpolation_start_q is None
            else wp.to_torch(self.kinematic_interpolation_start_q)
            .detach()
            .clone()
        )
        interpolation_target = (
            None
            if self.kinematic_interpolation_target_q is None
            else wp.to_torch(self.kinematic_interpolation_target_q)
            .detach()
            .clone()
        )
        grip = (
            None
            if self.triangle_skin_contact_projector is None
            else self.triangle_skin_contact_projector.clone_persistent_grip_state()
        )
        material = self.material_projector
        distance = (
            None
            if material is None
            else wp.to_torch(material.paper_distance_stiffness)
            .detach()
            .clone()
        )
        shape = (
            None
            if material is None
            else wp.to_torch(material.paper_shape_stiffness)
            .detach()
            .clone()
        )
        return PhysicsRolloutAuxiliaryState(
            sim_time=float(self.sim_time),
            kinematic_interpolation_start_q=interpolation_start,
            kinematic_interpolation_target_q=interpolation_target,
            persistent_grip=grip,
            paper_distance_stiffness=distance,
            paper_shape_stiffness=shape,
        )

    def restore_rollout_auxiliary_state(
        self, snapshot: PhysicsRolloutAuxiliaryState
    ) -> None:
        """Restore non-Warp state paired with a cloned physics state."""

        def restore_optional(
            destination,
            source: torch.Tensor | None,
            label: str,
        ) -> None:
            if destination is None or source is None:
                if destination is not None or source is not None:
                    raise ValueError(f"Rollout snapshot disagrees for {label}")
                return
            destination_torch = wp.to_torch(destination)
            source = source.to(
                device=destination_torch.device,
                dtype=destination_torch.dtype,
            )
            if source.shape != destination_torch.shape:
                raise ValueError(
                    f"Rollout snapshot shape mismatch for {label}: "
                    f"{source.shape} != {destination_torch.shape}"
                )
            with torch.no_grad():
                destination_torch.copy_(source)

        self.sim_time = float(snapshot.sim_time)
        restore_optional(
            self.kinematic_interpolation_start_q,
            snapshot.kinematic_interpolation_start_q,
            "kinematic interpolation start",
        )
        restore_optional(
            self.kinematic_interpolation_target_q,
            snapshot.kinematic_interpolation_target_q,
            "kinematic interpolation target",
        )
        if self.triangle_skin_contact_projector is None:
            if snapshot.persistent_grip is not None:
                raise ValueError("Rollout snapshot has unexpected grip state")
        elif snapshot.persistent_grip is None:
            raise ValueError("Rollout snapshot is missing grip state")
        else:
            self.triangle_skin_contact_projector.restore_persistent_grip_state(
                snapshot.persistent_grip
            )
        material = self.material_projector
        if material is None:
            if (
                snapshot.paper_distance_stiffness is not None
                or snapshot.paper_shape_stiffness is not None
            ):
                raise ValueError("Rollout snapshot has unexpected material state")
        else:
            restore_optional(
                material.paper_distance_stiffness,
                snapshot.paper_distance_stiffness,
                "paper distance stiffness",
            )
            restore_optional(
                material.paper_shape_stiffness,
                snapshot.paper_shape_stiffness,
                "paper shape stiffness",
            )

    def set_state(self, state: warp.sim.State):
        copy_state(self.state_0, state)

    def set_control(self, control: warp.sim.Control):
        copy_control(self.control, control)

    def synchronize_state(self, state: warp.sim.State):
        synchronize_state(
            dst_state=self.state_0,
            src_state=state,
        )

    def synchronize_control(self, control: warp.sim.Control):
        synchronize_control(dst_control=self.control, src_control=control)

    def reset(self):
        self.sim_time = 0.0
        if self.triangle_skin_contact_projector is not None:
            self.triangle_skin_contact_projector.reset_persistent_grip()

    def get_time(self):
        return self.sim_time

    def eval_fk(self, mask=None):
        if self.model.joint_count > 0:
            warp.sim.eval_fk(
                self.model,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                mask,
                self.state_0,
            )
            # wp.copy(self.control.joint_act, self.model.joint_q)  # type: ignore

    def eval_ik(self):
        if self.model.joint_count > 0:
            warp.sim.eval_ik(
                self.model, self.state_0, self.state_0.joint_q, self.state_0.joint_qd
            )

    def set_body_q(self, body_id: int, X_WO: np.ndarray):
        s = wp.to_torch(self.state_0.body_q)
        T = transform_from_matrix(X_WO)
        s[body_id] = torch.from_numpy(T).float().to(self.device)

    def get_body_q(self, body_id: int):
        s = wp.to_torch(self.state_0.body_q)[body_id].cpu().numpy()
        return transform_to_matrix(s)

    def get_articulation_q(self, index: int, num_joints: int):
        assert index < self.builder.articulation_count
        joint_start = self.builder.articulation_start[index]
        joint_q = wp.to_torch(self.state_0.joint_q)
        return joint_q[joint_start : joint_start + num_joints]
    
    def get_articulation_qd(self, index: int, num_joints: int):
        assert index < self.builder.articulation_count
        joint_start = self.builder.articulation_start[index]
        joint_qd = wp.to_torch(self.state_0.joint_qd)
        return joint_qd[joint_start : joint_start + num_joints]
    
    def check_articulation_healthy(self, index: int):
        assert index < self.builder.articulation_count
        joint_qd = wp.to_torch(self.state_0.joint_qd)
        return torch.isfinite(joint_qd).all()

    def set_articulation_q(self, index: int, q: torch.Tensor):
        assert index < self.builder.articulation_count
        if q.ndim == 1:
            q = q.unsqueeze(0) # replicate q for all envs
        q = q.to(self.device)
        joint_start = self.builder.articulation_start[index]
        given_joints = q.shape[1]

        joint_q = wp.to_torch(self.state_0.joint_q).reshape((self.num_envs, -1))
        joint_q[:, joint_start : joint_start + given_joints] = q
        joint_act = wp.to_torch(self.control.joint_act).reshape((self.num_envs, -1))
        joint_act[:, joint_start : joint_start + given_joints] = q

        warp.sim.eval_fk(
            self.model, self.state_0.joint_q, self.state_0.joint_qd, None, self.state_0
        )

    def set_articulation_control_q(self, index: int, q: torch.Tensor):
        assert index < self.builder.articulation_count
        if q.ndim == 1:
            q = q.unsqueeze(0) # replicate q for all envs
        q = q.to(self.device)
        joint_start = self.builder.articulation_start[index]
        given_joints = q.shape[1]
        joint_act = wp.to_torch(self.control.joint_act).reshape((self.num_envs, -1))
        joint_act[:, joint_start : joint_start + given_joints] = q

    def get_joint_act(self) -> torch.Tensor:
        return wp.to_torch(self.control.joint_act).reshape((self.num_envs, -1))

    def set_joint_act(self, joint_act: torch.Tensor):
        assert joint_act.shape[0] == self.num_envs
        ja = self.get_joint_act()
        ja.copy_(joint_act)

    def physics_step(self, settings: PhysicsSettings):
        self._capture_kinematic_body_targets()
        self._physics_step(settings)
        self.sim_time += settings.dt

    @cuda_graph_capture
    def _physics_step(self, settings: PhysicsSettings):
        self.integrator = XPBDIntegrator(
            iterations=settings.xpbd_iterations,
            soft_contact_relaxation=settings.particle_shape_contact_relaxation,
            preserve_static_body_poses=settings.preserve_static_body_poses,
        )
        use_material_projector = settings.use_project_material_tetrahedra
        if use_material_projector:
            if self.material_projector is None:
                raise ValueError(
                    "use_project_material_tetrahedra=True requires tetrahedra"
                )
            self.material_projector.iterations = settings.material_iterations
            self.material_projector.relaxation = settings.material_relaxation
            self.material_projector.compliance_scale = (
                settings.material_compliance_scale
            )
            self.material_projector.configure_constraint_model(
                settings.tetrahedral_constraint_model,
                settings.paper_distance_stiffness,
                settings.paper_volume_stiffness,
                settings.paper_shape_stiffness,
                preserve_spatial_stiffness=(
                    settings.preserve_spatial_paper_stiffness
                ),
            )
        if settings.enable_triangle_skin_contacts:
            if not use_material_projector:
                raise ValueError(
                    "Triangle-skin contact requires the material projector"
                )
            if self.triangle_skin_contact_projector is None:
                raise ValueError(
                    "Triangle-skin contact is enabled but not configured"
                )
            if settings.triangle_skin_contact_substep_stride < 1:
                raise ValueError(
                    "triangle_skin_contact_substep_stride must be at least one"
                )
            if settings.triangle_skin_contact_iterations < 1:
                raise ValueError(
                    "triangle_skin_contact_iterations must be at least one"
                )
            if settings.triangle_skin_post_contact_material_iterations < 0:
                raise ValueError(
                    "triangle_skin_post_contact_material_iterations cannot be negative"
                )
            if not (
                0.0
                <= settings.triangle_skin_final_barrier_max_correction_m
                <= settings.triangle_skin_contact_max_correction_m
            ):
                raise ValueError(
                    "triangle_skin_final_barrier_max_correction_m must lie "
                    "between zero and the main contact correction"
                )
            if not (
                0.0
                <= settings.triangle_skin_top_barrier_max_correction_m
                <= settings.triangle_skin_contact_max_correction_m
            ):
                raise ValueError(
                    "triangle_skin_top_barrier_max_correction_m must be zero "
                    "(inherit) or lie below the main contact correction"
                )
        filter_particle_shape_contacts = (
            settings.enable_particle_shape_contacts
            and self.particle_shape_contact_filter is not None
        )
        if filter_particle_shape_contacts:
            wp.launch(
                kernel=mask_particle_shape_contact_geometries,
                dim=self.model.shape_count,
                inputs=[
                    self.model.shape_geo.type,
                    self.particle_shape_contact_filter,
                    self._saved_shape_geo_type,
                ],
                device=self.model.device,
            )
        native_particle_count = self.model.particle_count
        if not settings.enable_particle_shape_contacts:
            # collide() generates every particle-versus-shape candidate without
            # consulting rigid shape collision flags.  Stage C must not create
            # PSM contacts at all, so skip only its soft-contact generation.
            self.model.particle_count = 0
        try:
            warp.sim.collide(self.model, self.state_0)
        finally:
            self.model.particle_count = native_particle_count
            if filter_particle_shape_contacts:
                wp.launch(
                    kernel=restore_particle_shape_contact_geometries,
                    dim=self.model.shape_count,
                    inputs=[
                        self.model.shape_geo.type,
                        self._saved_shape_geo_type,
                    ],
                    device=self.model.device,
                )
        if (
            not settings.enable_particle_shape_contacts
            and self.model.soft_contact_count is not None
        ):
            self.model.soft_contact_count.zero_()
        substep_dt = settings.dt / settings.substeps
        velocity_damping = math.exp(
            -settings.particle_velocity_damping_per_second * substep_dt
        )
        for substep_index in range(settings.substeps):
            if self.kinematic_interpolation_body_ids is not None:
                interpolation_alpha = float(
                    substep_index + 1
                ) / float(settings.substeps)
                for state in (self.state_0, self.state_1):
                    wp.launch(
                        kernel=interpolate_kinematic_body_poses,
                        dim=len(self.kinematic_interpolation_body_ids),
                        inputs=[
                            self.kinematic_interpolation_body_ids,
                            self.kinematic_interpolation_start_q,
                            self.kinematic_interpolation_target_q,
                            self.model.body_com,
                            interpolation_alpha,
                            settings.dt,
                        ],
                        outputs=[state.body_q, state.body_qd],
                        device=self.model.device,
                    )
            if use_material_projector:
                self.material_projector.capture_previous_positions(self.state_0)
            native_tet_count = self.model.tet_count
            native_particle_max_radius = self.model.particle_max_radius
            native_shape_count = self.model.shape_count
            if use_material_projector:
                # Warp 1.7's native solve_tetrahedra ignores the stored Lamé
                # parameters.  Suppress only that launch; rigid/joint XPBD and
                # particle integration continue through the standard solver.
                self.model.tet_count = 0
                if settings.enable_particle_shape_contacts:
                    # The project-local material projector performs the jaw
                    # contact after material projection with a bounded,
                    # inversion-aware correction. Avoid applying Warp's
                    # unbounded particle-shape solve a second time here.
                    self.model.shape_count = 1
            if not settings.enable_particle_particle_contacts:
                self.model.particle_max_radius = 0.0
            try:
                self.integrator.simulate(
                    self.model,
                    self.state_0,
                    self.state_1,
                    substep_dt,
                    self.control,
                )
            finally:
                self.model.tet_count = native_tet_count
                self.model.particle_max_radius = native_particle_max_radius
                self.model.shape_count = native_shape_count
            if use_material_projector:
                triangle_skin_contact_stride = (
                    settings.triangle_skin_contact_substep_stride
                )
                solve_triangle_skin_contacts = (
                    settings.enable_triangle_skin_contacts
                    and (
                        (substep_index + 1)
                        % triangle_skin_contact_stride
                        == 0
                        or substep_index == settings.substeps - 1
                    )
                )
                self.material_projector.project_particles(
                    self.model,
                    self.state_1,
                    substep_dt,
                    velocity_damping=velocity_damping,
                    solve_ground=bool(self.model.ground),
                    ground_relaxation=settings.particle_ground_relaxation,
                    solve_particle_shapes=settings.enable_particle_shape_contacts,
                    particle_shape_relaxation=(
                        settings.particle_shape_contact_relaxation
                    ),
                    particle_shape_max_correction=(
                        settings.particle_shape_contact_max_correction_m
                    ),
                    particle_shape_min_volume_ratio=(
                        settings.particle_shape_contact_min_volume_ratio
                    ),
                    solve_triangle_skin_contacts=(
                        solve_triangle_skin_contacts
                    ),
                    triangle_skin_contact_projector=(
                        self.triangle_skin_contact_projector
                    ),
                    triangle_skin_contact_margin_m=(
                        settings.triangle_skin_contact_margin_m
                    ),
                    triangle_skin_query_distance_m=(
                        settings.triangle_skin_query_distance_m
                    ),
                    triangle_skin_ccd_velocity_scale=(
                        settings.triangle_skin_ccd_velocity_scale
                    ),
                    triangle_skin_friction=(
                        settings.triangle_skin_friction
                    ),
                    triangle_skin_relaxation=(
                        settings.triangle_skin_contact_relaxation
                    ),
                    triangle_skin_max_correction_m=(
                        settings.triangle_skin_contact_max_correction_m
                    ),
                    triangle_skin_top_barrier_max_correction_m=(
                        settings
                        .triangle_skin_top_barrier_max_correction_m
                    ),
                    triangle_skin_contact_iterations=(
                        settings.triangle_skin_contact_iterations
                    ),
                    triangle_skin_post_contact_material_iterations=(
                        settings.triangle_skin_post_contact_material_iterations
                    ),
                    triangle_skin_final_barrier_max_correction_m=(
                        settings.triangle_skin_final_barrier_max_correction_m
                    ),
                    triangle_skin_min_volume_ratio=(
                        settings.triangle_skin_contact_min_volume_ratio
                    ),
                    triangle_skin_contact_time_scale=float(
                        triangle_skin_contact_stride
                    ),
                    material_min_volume_ratio=(
                        settings.material_min_volume_ratio
                    ),
                    projection_velocity_scale=(
                        settings.material_projection_velocity_scale
                    ),
                    contact_projection_velocity_scale=(
                        settings.contact_projection_velocity_scale
                    ),
                )
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.state_0.clear_forces()
        if self.kinematic_interpolation_body_ids is not None:
            wp.copy(
                self.kinematic_interpolation_start_q,
                self.kinematic_interpolation_target_q,
            )
        self.eval_ik()  # xpbd does not update the state of the joints since it operates on body_q. Do so manually.
