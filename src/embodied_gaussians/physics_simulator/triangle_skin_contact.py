"""Deformable triangle-skin contact against kinematic tool meshes.

The mechanics particles are zero-radius nodes.  Dense tool-surface samples
query the continuously deforming tissue triangles, and each correction is
distributed barycentrically to the hit triangle.  SUPER currently enables
this path only for the two jaws and uses Coulomb friction while contact
exists.  An optional bilateral grasp constraint is gated by recorded q7[6]:
it captures only while closing/closed and releases once q7[6] opens a
meaningful amount relative to its value at capture.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import warp as wp
import warp.sim
from scipy.spatial import cKDTree
from warp.sim.model import PARTICLE_FLAG_ACTIVE


@dataclass
class PersistentGripSnapshot:
    """Restorable state of the persistent-grip controller.

    Warp ``State`` objects contain particle and rigid-body dynamics, but the
    bilateral grasp controller also owns attachment ids, jaw-local targets and
    a small capture/release state machine.  Shadow rollouts must restore both
    parts or an apparently identical particle snapshot can use different
    positional controls.
    """

    arrays: dict[str, torch.Tensor]
    previous_jaw_angle: float | None
    jaw_motion_state: str


def _halton(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0
    while index > 0:
        factor /= base
        result += factor * (index % base)
        index //= base
    return result


def sample_triangle_mesh_surface(
    vertices: np.ndarray,
    triangles: np.ndarray,
    spacing_m: float,
) -> np.ndarray:
    """Deterministically sample mesh vertices and triangle interiors."""
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected mesh vertices Nx3, got {vertices.shape}")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"Expected mesh triangles Mx3, got {triangles.shape}")
    if spacing_m <= 0.0:
        raise ValueError("Tool-surface sampling spacing must be positive")
    if triangles.size and (
        triangles.min() < 0 or triangles.max() >= len(vertices)
    ):
        raise ValueError("Tool mesh triangle index is outside its vertex array")

    samples: list[np.ndarray] = [point.copy() for point in vertices]
    target_area_per_sample = math.sqrt(3.0) * spacing_m * spacing_m / 4.0
    triangle_points = vertices[triangles]
    triangle_areas = 0.5 * np.linalg.norm(
        np.cross(
            triangle_points[:, 1] - triangle_points[:, 0],
            triangle_points[:, 2] - triangle_points[:, 0],
        ),
        axis=1,
    )
    total_area = float(triangle_areas.sum())
    interior_count = (
        int(math.ceil(total_area / target_area_per_sample))
        if total_area > 0.0
        else 0
    )
    if interior_count:
        cumulative_area = np.cumsum(triangle_areas)
        area_targets = (
            (np.arange(interior_count, dtype=np.float64) + 0.5)
            * total_area
            / interior_count
        )
        sampled_triangles = np.searchsorted(
            cumulative_area, area_targets, side="left"
        )
        for sample_index, triangle_index in enumerate(sampled_triangles):
            point_a, point_b, point_c = triangle_points[triangle_index]
            first = _halton(sample_index + 1, 2)
            second = _halton(sample_index + 1, 3)
            root = math.sqrt(first)
            bary_a = 1.0 - root
            bary_b = root * (1.0 - second)
            bary_c = root * second
            samples.append(
                bary_a * point_a + bary_b * point_b + bary_c * point_c
            )

    points = np.asarray(samples, dtype=np.float64)
    # STL meshes commonly repeat vertices. Quantize only far below the contact
    # margin so distinct nearby tool surfaces are never merged.
    quantization_m = min(1.0e-7, spacing_m * 1.0e-3)
    keys = np.round(points / quantization_m).astype(np.int64)
    _, unique_indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(unique_indices)].astype(np.float32)


@wp.kernel(enable_backward=False)
def gather_triangle_skin_points(
    positions: wp.array(dtype=wp.vec3),
    skin_nodes: wp.array(dtype=int),
    skin_points: wp.array(dtype=wp.vec3),
):
    local_index = wp.tid()
    skin_points[local_index] = positions[skin_nodes[local_index]]


@wp.kernel(enable_backward=False)
def accumulate_triangle_skin_contact_deltas(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
    skin_faces: wp.array(dtype=int, ndim=2),
    skin_mesh: wp.uint64,
    tool_sample_local: wp.array(dtype=wp.vec3),
    tool_sample_shape: wp.array(dtype=int),
    surface_sample_enabled: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    query_distance: float,
    contact_margin: float,
    ccd_velocity_scale: float,
    friction_coefficient: float,
    dt: float,
    relaxation: float,
    deltas: wp.array(dtype=wp.vec3),
    delta_counts: wp.array(dtype=float),
    contact_count: wp.array(dtype=int),
    contact_count_by_shape: wp.array(dtype=int),
    maximum_penetration: wp.array(dtype=float),
    minimum_signed_distance: wp.array(dtype=float),
):
    sample_index = wp.tid()
    if surface_sample_enabled[sample_index] == 0:
        return
    shape_index = tool_sample_shape[sample_index]
    body_index = shape_body[shape_index]
    if body_index < 0:
        return

    X_wb = body_q[body_index]
    X_bs = shape_transform[shape_index]
    X_ws = wp.transform_multiply(X_wb, X_bs)
    sample_scaled = wp.cw_mul(
        tool_sample_local[sample_index], shape_scale[shape_index]
    )
    tool_point = wp.transform_point(X_ws, sample_scaled)
    query = wp.mesh_query_point_sign_normal(
        skin_mesh, tool_point, query_distance
    )
    if not query.result:
        return

    skin_point = wp.mesh_eval_position(
        skin_mesh, query.face, query.u, query.v
    )
    difference = tool_point - skin_point
    distance = wp.length(difference)
    index_a = skin_faces[query.face, 0]
    index_b = skin_faces[query.face, 1]
    index_c = skin_faces[query.face, 2]
    bary_a = wp.clamp(query.u, 0.0, 1.0)
    bary_b = wp.clamp(query.v, 0.0, 1.0)
    bary_c = wp.clamp(1.0 - bary_a - bary_b, 0.0, 1.0)
    bary_sum = bary_a + bary_b + bary_c
    if bary_sum <= 1.0e-8:
        return
    bary_a /= bary_sum
    bary_b /= bary_sum
    bary_c /= bary_sum

    outward_normal = wp.vec3()
    if distance > 1.0e-9:
        outward_normal = wp.normalize(difference) * query.sign
    else:
        edge_ab = positions[index_b] - positions[index_a]
        edge_ac = positions[index_c] - positions[index_a]
        outward_normal = wp.normalize(wp.cross(edge_ab, edge_ac))
    signed_distance = distance * query.sign
    wp.atomic_min(minimum_signed_distance, 0, signed_distance)

    skin_velocity = (
        bary_a * velocities[index_a]
        + bary_b * velocities[index_b]
        + bary_c * velocities[index_c]
    )
    spatial_velocity = body_qd[body_index]
    angular_velocity = wp.spatial_top(spatial_velocity)
    com_velocity = wp.spatial_bottom(spatial_velocity)
    world_com = wp.transform_point(X_wb, body_com[body_index])
    tool_velocity = com_velocity + wp.cross(
        angular_velocity, tool_point - world_com
    )
    tool_relative_velocity = tool_velocity - skin_velocity
    inward_speed = wp.max(
        0.0, -wp.dot(tool_relative_velocity, outward_normal)
    )
    effective_margin = (
        contact_margin + ccd_velocity_scale * inward_speed * dt
    )
    penetration = effective_margin - signed_distance
    if penetration <= 0.0:
        return

    inverse_mass_a = inverse_mass[index_a]
    inverse_mass_b = inverse_mass[index_b]
    inverse_mass_c = inverse_mass[index_c]
    denominator = (
        inverse_mass_a * bary_a * bary_a
        + inverse_mass_b * bary_b * bary_b
        + inverse_mass_c * bary_c * bary_c
    )
    if denominator <= 1.0e-12:
        return

    normal_correction = -outward_normal * penetration
    tangential_velocity = tool_relative_velocity - (
        wp.dot(tool_relative_velocity, outward_normal) * outward_normal
    )
    tangential_correction = tangential_velocity * dt
    tangential_length = wp.length(tangential_correction)
    friction_limit = friction_coefficient * penetration
    if tangential_length > friction_limit and tangential_length > 1.0e-12:
        tangential_correction *= friction_limit / tangential_length
    contact_correction = relaxation * (
        normal_correction + tangential_correction
    )

    coefficient_a = inverse_mass_a * bary_a / denominator
    coefficient_b = inverse_mass_b * bary_b / denominator
    coefficient_c = inverse_mass_c * bary_c / denominator
    if (
        inverse_mass_a > 0.0
        and (particle_flags[index_a] & PARTICLE_FLAG_ACTIVE) != 0
    ):
        wp.atomic_add(deltas, index_a, coefficient_a * contact_correction)
        wp.atomic_add(delta_counts, index_a, 1.0)
    if (
        inverse_mass_b > 0.0
        and (particle_flags[index_b] & PARTICLE_FLAG_ACTIVE) != 0
    ):
        wp.atomic_add(deltas, index_b, coefficient_b * contact_correction)
        wp.atomic_add(delta_counts, index_b, 1.0)
    if (
        inverse_mass_c > 0.0
        and (particle_flags[index_c] & PARTICLE_FLAG_ACTIVE) != 0
    ):
        wp.atomic_add(deltas, index_c, coefficient_c * contact_correction)
        wp.atomic_add(delta_counts, index_c, 1.0)

    wp.atomic_add(contact_count, 0, 1)
    wp.atomic_add(contact_count_by_shape, shape_index, 1)
    wp.atomic_max(maximum_penetration, 0, penetration)


@wp.kernel(enable_backward=False)
def accumulate_jaw_triangle_skin_contact_deltas(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
    skin_faces: wp.array(dtype=int, ndim=2),
    skin_face_is_top: wp.array(dtype=int),
    skin_mesh: wp.uint64,
    tool_sample_local: wp.array(dtype=wp.vec3),
    tool_sample_shape: wp.array(dtype=int),
    jaw_surface_sample_enabled: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    query_distance: float,
    contact_margin: float,
    ccd_velocity_scale: float,
    friction_coefficient: float,
    dt: float,
    relaxation: float,
    jaw_shape_a: int,
    jaw_shape_b: int,
    contribution_nodes: wp.array(dtype=int),
    contribution_deltas: wp.array(dtype=wp.vec3),
    contact_count: wp.array(dtype=int),
    contact_count_by_shape: wp.array(dtype=int),
    jaw_triangle_contact_count: wp.array(dtype=int),
    jaw_closing_contact_count_by_shape: wp.array(dtype=int),
    jaw_closing_resistance_by_shape: wp.array(dtype=float),
    maximum_penetration: wp.array(dtype=float),
    minimum_signed_distance: wp.array(dtype=float),
):
    """Project jaw-surface samples onto the continuous tissue triangle skin.

    Each hit is distributed to all three vertices with barycentric inverse-mass
    weights.  ``query_distance`` is only the broad nearest-triangle search
    radius; ``contact_margin`` independently defines the desired surface gap.
    A jaw sample that crossed deeper than the margin therefore remains a
    penetrating contact instead of disappearing from the solver.
    """
    sample_index = wp.tid()
    contribution_start = sample_index * 3
    contribution_nodes[contribution_start] = -1
    contribution_nodes[contribution_start + 1] = -1
    contribution_nodes[contribution_start + 2] = -1
    contribution_deltas[contribution_start] = wp.vec3()
    contribution_deltas[contribution_start + 1] = wp.vec3()
    contribution_deltas[contribution_start + 2] = wp.vec3()
    if jaw_surface_sample_enabled[sample_index] == 0:
        return
    shape_index = tool_sample_shape[sample_index]
    body_index = shape_body[shape_index]
    if body_index < 0:
        return

    X_wb = body_q[body_index]
    X_ws = wp.transform_multiply(
        X_wb, shape_transform[shape_index]
    )
    sample_scaled = wp.cw_mul(
        tool_sample_local[sample_index], shape_scale[shape_index]
    )
    tool_point = wp.transform_point(X_ws, sample_scaled)
    query = wp.mesh_query_point_no_sign(
        skin_mesh, tool_point, query_distance
    )
    if not query.result:
        return

    skin_point = wp.mesh_eval_position(
        skin_mesh, query.face, query.u, query.v
    )
    index_a = skin_faces[query.face, 0]
    index_b = skin_faces[query.face, 1]
    index_c = skin_faces[query.face, 2]
    edge_ab = positions[index_b] - positions[index_a]
    edge_ac = positions[index_c] - positions[index_a]
    normal_unnormalized = wp.cross(edge_ab, edge_ac)
    if wp.length_sq(normal_unnormalized) <= 1.0e-20:
        return
    outward_normal = wp.normalize(normal_unnormalized)
    difference = tool_point - skin_point
    signed_distance = wp.dot(difference, outward_normal)
    wp.atomic_min(minimum_signed_distance, 0, signed_distance)

    bary_a = wp.clamp(query.u, 0.0, 1.0)
    bary_b = wp.clamp(query.v, 0.0, 1.0)
    bary_c = wp.clamp(1.0 - bary_a - bary_b, 0.0, 1.0)
    bary_sum = bary_a + bary_b + bary_c
    if bary_sum <= 1.0e-8:
        return
    bary_a /= bary_sum
    bary_b /= bary_sum
    bary_c /= bary_sum

    skin_velocity = (
        bary_a * velocities[index_a]
        + bary_b * velocities[index_b]
        + bary_c * velocities[index_c]
    )
    spatial_velocity = body_qd[body_index]
    angular_velocity = wp.spatial_top(spatial_velocity)
    com_velocity = wp.spatial_bottom(spatial_velocity)
    world_com = wp.transform_point(X_wb, body_com[body_index])
    tool_velocity = com_velocity + wp.cross(
        angular_velocity, tool_point - world_com
    )
    tool_relative_velocity = tool_velocity - skin_velocity
    inward_speed = wp.max(
        0.0, -wp.dot(tool_relative_velocity, outward_normal)
    )
    effective_margin = (
        contact_margin + ccd_velocity_scale * inward_speed * dt
    )
    penetration = effective_margin - signed_distance
    if penetration <= 0.0:
        return

    # q7 opens the two jaws by rotating them symmetrically about their shared
    # local-z hinge: jaw A uses +q7/2 and jaw B uses -q7/2.  Measure only
    # contacts whose normal opposes a *decrease* of q7.  This excludes samples
    # that merely touch the top sheet while sliding tangentially and gives the
    # jaw actuator a geometric tissue-resistance signal instead of a fabricated
    # minimum opening angle.
    opening_sign = float(0.0)
    if shape_index == jaw_shape_a:
        opening_sign = 1.0
    elif shape_index == jaw_shape_b:
        opening_sign = -1.0
    hinge_pivot = wp.transform_get_translation(X_wb)
    hinge_axis = wp.quat_rotate(
        wp.transform_get_rotation(X_wb), wp.vec3(0.0, 0.0, 1.0)
    )
    closing_motion_per_rad = (
        -0.5
        * opening_sign
        * wp.cross(hinge_axis, tool_point - hinge_pivot)
    )
    closing_leverage = wp.max(
        0.0, -wp.dot(closing_motion_per_rad, outward_normal)
    )
    # Ignore samples extremely close to the hinge and near-tangential grazing
    # contacts.  0.25 mm/rad is small relative to the jaw-tip lever arm while
    # preventing top-sheet contact noise from stopping the motor.
    if closing_leverage >= 0.00025:
        wp.atomic_add(
            jaw_closing_contact_count_by_shape, shape_index, 1
        )
        wp.atomic_add(
            jaw_closing_resistance_by_shape,
            shape_index,
            closing_leverage * penetration,
        )

    inverse_mass_a = inverse_mass[index_a]
    inverse_mass_b = inverse_mass[index_b]
    inverse_mass_c = inverse_mass[index_c]
    denominator = (
        inverse_mass_a * bary_a * bary_a
        + inverse_mass_b * bary_b * bary_b
        + inverse_mass_c * bary_c * bary_c
    )
    if denominator <= 1.0e-12:
        return

    # The oriented top-sheet barrier owns the normal correction on top faces.
    # Keep only Coulomb tangential motion here so the same jaw sample cannot
    # push the surface once along the triangle normal and a second time along
    # the unilateral top-plane direction.
    normal_correction = wp.vec3()
    if skin_face_is_top[query.face] == 0:
        normal_correction = -outward_normal * penetration
    tangential_velocity = tool_relative_velocity - (
        wp.dot(tool_relative_velocity, outward_normal)
        * outward_normal
    )
    tangential_correction = tangential_velocity * dt
    tangential_length = wp.length(tangential_correction)
    friction_limit = friction_coefficient * penetration
    # On the upper tissue sheet the nearest-triangle normal is approximately
    # vertical, whereas q7 closes the jaw faces mostly sideways.  Treating
    # that motion only as Coulomb friction makes the allowable push vanish
    # with the tiny vertical penetration.  The normalized q7-closing tangent
    # is the kinematic non-slip direction of the jaw face: while it is moving
    # into the tissue, preserve that component of the real tool velocity.
    # The ordinary per-particle contact cap and no-flip repair still bound the
    # accepted correction; this does not manufacture motion outside contact.
    closing_motion_length = wp.length(closing_motion_per_rad)
    if closing_motion_length > 1.0e-10:
        closing_direction = closing_motion_per_rad / closing_motion_length
        closing_speed = wp.max(
            0.0, wp.dot(tool_relative_velocity, closing_direction)
        )
        closing_tangent = closing_direction - wp.dot(
            closing_direction, outward_normal
        ) * outward_normal
        closing_tangent_length = wp.length(closing_tangent)
        if closing_tangent_length > 1.0e-10:
            closing_no_slip_distance = (
                closing_speed * dt * closing_tangent_length
            )
            friction_limit = wp.max(
                friction_limit, closing_no_slip_distance
            )
    if (
        tangential_length > friction_limit
        and tangential_length > 1.0e-12
    ):
        tangential_correction *= friction_limit / tangential_length
    # The oriented top-sheet barrier is the sole owner of pre-grasp vertical
    # indentation.  Once a jaw has crossed the reconstructed top sheet, a
    # closed-volume nearest-face query can select a bottom or steep side face;
    # its outward normal then points upward and the generic non-penetration
    # correction inflates the tissue as the tool descends.  Keep ordinary jaw
    # contact in the table plane for both normal and q7/no-slip components.
    # After capture, the four u_t anchors and common-wrist support ring own the
    # full 3-D lift, so no intended grasp motion is lost here.
    normal_correction[2] = 0.0
    tangential_correction[2] = 0.0
    contact_correction = relaxation * (
        normal_correction + tangential_correction
    )

    coefficient_a = inverse_mass_a * bary_a / denominator
    coefficient_b = inverse_mass_b * bary_b / denominator
    coefficient_c = inverse_mass_c * bary_c / denominator
    if (
        inverse_mass_a > 0.0
        and (particle_flags[index_a] & PARTICLE_FLAG_ACTIVE) != 0
    ):
        contribution_nodes[contribution_start] = index_a
        contribution_deltas[contribution_start] = (
            coefficient_a * contact_correction
        )
    if (
        inverse_mass_b > 0.0
        and (particle_flags[index_b] & PARTICLE_FLAG_ACTIVE) != 0
    ):
        contribution_nodes[contribution_start + 1] = index_b
        contribution_deltas[contribution_start + 1] = (
            coefficient_b * contact_correction
        )
    if (
        inverse_mass_c > 0.0
        and (particle_flags[index_c] & PARTICLE_FLAG_ACTIVE) != 0
    ):
        contribution_nodes[contribution_start + 2] = index_c
        contribution_deltas[contribution_start + 2] = (
            coefficient_c * contact_correction
        )

    wp.atomic_add(contact_count, 0, 1)
    wp.atomic_add(contact_count_by_shape, shape_index, 1)
    wp.atomic_add(jaw_triangle_contact_count, 0, 1)
    wp.atomic_max(maximum_penetration, 0, penetration)


@wp.kernel
def count_persistent_grip_candidates(
    contribution_nodes: wp.array(dtype=int),
    tool_sample_shape: wp.array(dtype=int),
    jaw_shape_a: int,
    jaw_shape_b: int,
    jaw_a_counts: wp.array(dtype=int),
    jaw_b_counts: wp.array(dtype=int),
):
    contribution_id = wp.tid()
    node = contribution_nodes[contribution_id]
    if node < 0:
        return
    shape_id = tool_sample_shape[contribution_id // 3]
    if shape_id == jaw_shape_a:
        wp.atomic_add(jaw_a_counts, node, 1)
    elif shape_id == jaw_shape_b:
        wp.atomic_add(jaw_b_counts, node, 1)


@wp.kernel
def accumulate_persistent_grip_contact_patch(
    positions: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
    jaw_a_counts: wp.array(dtype=int),
    jaw_b_counts: wp.array(dtype=int),
    contact_patch_sums: wp.array(dtype=float),
    between_jaw_candidate_count: wp.array(dtype=int),
):
    """Accumulate the two jaw-side surface patches without a per-node gate."""
    particle_id = wp.tid()
    if inverse_mass[particle_id] <= 0.0:
        return
    if (particle_flags[particle_id] & PARTICLE_FLAG_ACTIVE) == 0:
        return
    point = positions[particle_id]
    touched_a = jaw_a_counts[particle_id] > 0
    touched_b = jaw_b_counts[particle_id] > 0
    if touched_a:
        wp.atomic_add(contact_patch_sums, 0, point[0])
        wp.atomic_add(contact_patch_sums, 1, point[1])
        wp.atomic_add(contact_patch_sums, 2, point[2])
        wp.atomic_add(contact_patch_sums, 3, 1.0)
    if touched_b:
        wp.atomic_add(contact_patch_sums, 4, point[0])
        wp.atomic_add(contact_patch_sums, 5, point[1])
        wp.atomic_add(contact_patch_sums, 6, point[2])
        wp.atomic_add(contact_patch_sums, 7, 1.0)
    if touched_a and touched_b:
        wp.atomic_add(between_jaw_candidate_count, 0, 1)


@wp.kernel
def accumulate_particle_minimum_tet_volume_ratio(
    positions: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=int, ndim=2),
    inverse_rest_matrix: wp.array(dtype=wp.mat33),
    particle_minimum_ratio: wp.array(dtype=float),
):
    """Accumulate current incident-tet quality for grasp candidate filtering."""
    tet_id = wp.tid()
    index_a = tet_indices[tet_id, 0]
    index_b = tet_indices[tet_id, 1]
    index_c = tet_indices[tet_id, 2]
    index_d = tet_indices[tet_id, 3]
    current_matrix = wp.matrix_from_cols(
        positions[index_b] - positions[index_a],
        positions[index_c] - positions[index_a],
        positions[index_d] - positions[index_a],
    )
    ratio = wp.determinant(current_matrix * inverse_rest_matrix[tet_id])
    wp.atomic_min(particle_minimum_ratio, index_a, ratio)
    wp.atomic_min(particle_minimum_ratio, index_b, ratio)
    wp.atomic_min(particle_minimum_ratio, index_c, ratio)
    wp.atomic_min(particle_minimum_ratio, index_d, ratio)


@wp.kernel
def select_nearest_persistent_grip_surface_particles(
    positions: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
    skin_nodes: wp.array(dtype=int),
    skin_node_count: int,
    jaw_a_counts: wp.array(dtype=int),
    jaw_b_counts: wp.array(dtype=int),
    contact_patch_sums: wp.array(dtype=float),
    particle_minimum_tet_volume_ratio: wp.array(dtype=float),
    minimum_capture_volume_ratio: float,
    grasp_center: wp.array(dtype=wp.vec3),
    jaw_patch_separation: wp.array(dtype=float),
    selected_particle_ids: wp.array(dtype=int),
    selected_particle_count: wp.array(dtype=int),
):
    """Select two contacted surface particles from each jaw-side patch.

    This is deliberately a one-thread, fixed-size selection.  It is cheap for
    the current collision skin, deterministic, and remains inside a captured
    Warp graph (no CPU readback in the simulation loop).  Output slots 0--1
    belong to jaw A and slots 2--3 belong to jaw B.  Unlike the former global
    nearest-four search, every positional-control anchor is backed by an
    actual contribution from the jaw that will drive it.
    """
    if wp.tid() != 0:
        return
    count_a = contact_patch_sums[3]
    count_b = contact_patch_sums[7]
    selected_particle_count[0] = 0
    selected_particle_ids[0] = -1
    selected_particle_ids[1] = -1
    selected_particle_ids[2] = -1
    selected_particle_ids[3] = -1
    jaw_patch_separation[0] = 1.0e6
    if count_a <= 0.0 or count_b <= 0.0:
        return

    center_a = wp.vec3(
        contact_patch_sums[0],
        contact_patch_sums[1],
        contact_patch_sums[2],
    ) / count_a
    center_b = wp.vec3(
        contact_patch_sums[4],
        contact_patch_sums[5],
        contact_patch_sums[6],
    ) / count_b
    center = 0.5 * (center_a + center_b)
    grasp_center[0] = center
    jaw_patch_separation[0] = wp.length(center_a - center_b)

    best_a_distance_0 = float(1.0e30)
    best_a_distance_1 = float(1.0e30)
    best_a_id_0 = int(-1)
    best_a_id_1 = int(-1)
    for local_index in range(skin_node_count):
        particle_id = skin_nodes[local_index]
        if inverse_mass[particle_id] <= 0.0:
            continue
        if (particle_flags[particle_id] & PARTICLE_FLAG_ACTIVE) == 0:
            continue
        # A near-collapsed incident tet would reduce the shared no-flip line
        # search to almost zero and make a logically active grasp appear
        # immobile. Select healthy contacted surface nodes instead of weakening
        # the non-inversion constraint after capture.
        if (
            particle_minimum_tet_volume_ratio[particle_id]
            < minimum_capture_volume_ratio
        ):
            continue
        if jaw_a_counts[particle_id] <= 0:
            continue
        offset = positions[particle_id] - center_a
        distance_squared = wp.dot(offset, offset)
        if distance_squared < best_a_distance_0:
            best_a_distance_1 = best_a_distance_0
            best_a_id_1 = best_a_id_0
            best_a_distance_0 = distance_squared
            best_a_id_0 = particle_id
        elif distance_squared < best_a_distance_1:
            best_a_distance_1 = distance_squared
            best_a_id_1 = particle_id

    best_b_distance_0 = float(1.0e30)
    best_b_distance_1 = float(1.0e30)
    best_b_id_0 = int(-1)
    best_b_id_1 = int(-1)
    for local_index in range(skin_node_count):
        particle_id = skin_nodes[local_index]
        if inverse_mass[particle_id] <= 0.0:
            continue
        if (particle_flags[particle_id] & PARTICLE_FLAG_ACTIVE) == 0:
            continue
        if (
            particle_minimum_tet_volume_ratio[particle_id]
            < minimum_capture_volume_ratio
        ):
            continue
        if jaw_b_counts[particle_id] <= 0:
            continue
        # A node touched by both proxy surfaces may only represent one control
        # point. Keep the four anchors unique instead of assigning one physical
        # particle to two incompatible jaw frames.
        if particle_id == best_a_id_0 or particle_id == best_a_id_1:
            continue
        offset = positions[particle_id] - center_b
        distance_squared = wp.dot(offset, offset)
        if distance_squared < best_b_distance_0:
            best_b_distance_1 = best_b_distance_0
            best_b_id_1 = best_b_id_0
            best_b_distance_0 = distance_squared
            best_b_id_0 = particle_id
        elif distance_squared < best_b_distance_1:
            best_b_distance_1 = distance_squared
            best_b_id_1 = particle_id

    selected_particle_ids[0] = best_a_id_0
    selected_particle_ids[1] = best_a_id_1
    selected_particle_ids[2] = best_b_id_0
    selected_particle_ids[3] = best_b_id_1
    count = int(0)
    if best_a_id_0 >= 0:
        count += 1
    if best_a_id_1 >= 0:
        count += 1
    if best_b_id_0 >= 0:
        count += 1
    if best_b_id_1 >= 0:
        count += 1
    selected_particle_count[0] = count


@wp.kernel
def update_persistent_grip_state(
    contact_count_by_shape: wp.array(dtype=int),
    jaw_shape_a: int,
    jaw_shape_b: int,
    selected_particle_count: wp.array(dtype=int),
    jaw_patch_separation: wp.array(dtype=float),
    maximum_penetration: wp.array(dtype=float),
    minimum_contact_samples_per_jaw: int,
    maximum_jaw_patch_separation: float,
    activation_steps: int,
    maximum_capture_penetration: float,
    release_angle_delta_rad: float,
    current_jaw_angle: wp.array(dtype=float),
    current_signal_timestamp: wp.array(dtype=float),
    capture_allowed: wp.array(dtype=int),
    release_requested: wp.array(dtype=int),
    grip_state: wp.array(dtype=int),
    grip_capture_jaw_angle: wp.array(dtype=float),
    grip_capture_timestamp: wp.array(dtype=float),
):
    """Gate a sustained bilateral four-anchor capture with timestamped q7."""
    grip_state[2] = 0
    active = grip_state[0]
    if active != 0:
        opened_since_capture = (
            current_jaw_angle[0]
            > grip_capture_jaw_angle[0] + release_angle_delta_rad
        )
        if release_requested[0] != 0 or opened_since_capture:
            grip_state[0] = 0
            grip_state[1] = 0
        return
    if capture_allowed[0] == 0:
        grip_state[1] = 0
        return
    bilateral_contact = (
        contact_count_by_shape[jaw_shape_a]
        >= minimum_contact_samples_per_jaw
        and contact_count_by_shape[jaw_shape_b]
        >= minimum_contact_samples_per_jaw
    )
    four_surface_particles = selected_particle_count[0] == 4
    coherent_patch = (
        jaw_patch_separation[0] <= maximum_jaw_patch_separation
    )
    shallow_contact = maximum_penetration[0] <= maximum_capture_penetration
    if (
        bilateral_contact
        and four_surface_particles
        and coherent_patch
        and shallow_contact
    ):
        grip_state[1] = grip_state[1] + 1
        if grip_state[1] >= activation_steps:
            grip_state[0] = 1
            grip_state[2] = 1
            grip_capture_jaw_angle[0] = current_jaw_angle[0]
            grip_capture_timestamp[0] = current_signal_timestamp[0]
    else:
        grip_state[1] = 0


@wp.kernel
def capture_or_release_persistent_grip_particles(
    positions: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
    body_q: wp.array(dtype=wp.transform),
    jaw_body_a: int,
    jaw_body_b: int,
    selected_particle_ids: wp.array(dtype=int),
    grip_state: wp.array(dtype=int),
    particle_grip_body: wp.array(dtype=int),
    particle_grip_local: wp.array(dtype=wp.vec3),
    particle_grip_direct: wp.array(dtype=int),
    particle_grip_weight: wp.array(dtype=float),
    particle_grip_level: wp.array(dtype=int),
):
    particle_id = wp.tid()
    if grip_state[0] == 0:
        particle_grip_body[particle_id] = -1
        particle_grip_direct[particle_id] = 0
        particle_grip_weight[particle_id] = 0.0
        particle_grip_level[particle_id] = -1
        return
    if grip_state[2] == 0:
        return
    if (
        inverse_mass[particle_id] <= 0.0
        or (particle_flags[particle_id] & PARTICLE_FLAG_ACTIVE) == 0
    ):
        return
    selected_slot = int(-1)
    for selected_index in range(4):
        if selected_particle_ids[selected_index] == particle_id:
            selected_slot = selected_index
    if selected_slot < 0:
        return
    if particle_grip_direct[particle_id] != 0:
        return
    body_id = jaw_body_a
    if selected_slot >= 2:
        body_id = jaw_body_b
    jaw_transform = body_q[body_id]
    # This is the explicit point-position control u_t: two anchors retain
    # their coordinates in each physical jaw frame and receive a target from
    # that jaw's recorded pose at every solve.
    particle_grip_body[particle_id] = body_id
    particle_grip_local[particle_id] = wp.transform_point(
        wp.transform_inverse(jaw_transform), positions[particle_id]
    )
    particle_grip_direct[particle_id] = 1
    particle_grip_weight[particle_id] = 1.0
    particle_grip_level[particle_id] = 0


@wp.kernel
def advance_persistent_grip_support_generation(
    grip_state: wp.array(dtype=int),
    maximum_generation: int,
    support_generation: wp.array(dtype=int),
):
    if grip_state[0] == 0:
        support_generation[0] = 0
    elif support_generation[0] < maximum_generation:
        support_generation[0] = support_generation[0] + 1


@wp.kernel
def propagate_persistent_grip_support_particles(
    positions: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
    body_q: wp.array(dtype=wp.transform),
    support_body_id: int,
    grip_state: wp.array(dtype=int),
    support_offsets: wp.array(dtype=int),
    support_sources: wp.array(dtype=int),
    support_weights: wp.array(dtype=float),
    particle_grip_direct: wp.array(dtype=int),
    support_generation: wp.array(dtype=int),
    particle_grip_level: wp.array(dtype=int),
    particle_grip_body: wp.array(dtype=int),
    particle_grip_local: wp.array(dtype=wp.vec3),
    particle_grip_weight: wp.array(dtype=float),
):
    """Attach one softly weighted ring around the four direct grip anchors."""
    particle_id = wp.tid()
    if (
        grip_state[0] == 0
        or particle_grip_direct[particle_id] != 0
        or particle_grip_body[particle_id] >= 0
        or inverse_mass[particle_id] <= 0.0
        or (particle_flags[particle_id] & PARTICLE_FLAG_ACTIVE) == 0
    ):
        return
    generation = support_generation[0]
    if generation <= 0:
        return
    source = int(-1)
    weight = float(0.0)
    start = support_offsets[particle_id]
    end = support_offsets[particle_id + 1]
    for support_id in range(start, end):
        candidate = support_sources[support_id]
        candidate_level = particle_grip_level[candidate]
        if candidate_level >= 0 and candidate_level < generation:
            source = candidate
            weight = (
                particle_grip_weight[candidate]
                * support_weights[support_id]
            )
            break
    if source < 0:
        return
    source_body_id = particle_grip_body[source]
    if source_body_id < 0 or support_body_id < 0:
        return
    # Outside-ring particles follow the common wrist frame, not either
    # individual jaw. Binding them to jaw A/B made continued q7 closure pull
    # the ring in two directions and created an artificial bulge.
    particle_grip_body[particle_id] = support_body_id
    particle_grip_local[particle_id] = wp.transform_point(
        wp.transform_inverse(body_q[support_body_id]), positions[particle_id]
    )
    particle_grip_weight[particle_id] = weight
    particle_grip_level[particle_id] = generation


@wp.kernel
def accumulate_persistent_grip_constraint_deltas(
    positions: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
    body_q: wp.array(dtype=wp.transform),
    grip_state: wp.array(dtype=int),
    particle_grip_body: wp.array(dtype=int),
    particle_grip_local: wp.array(dtype=wp.vec3),
    particle_grip_weight: wp.array(dtype=float),
    dt: float,
    compliance: float,
    relaxation: float,
    maximum_correction: float,
    grip_deltas: wp.array(dtype=wp.vec3),
    grip_delta_counts: wp.array(dtype=float),
):
    particle_id = wp.tid()
    if (
        grip_state[0] == 0
        or (particle_flags[particle_id] & PARTICLE_FLAG_ACTIVE) == 0
    ):
        return
    body_id = particle_grip_body[particle_id]
    if body_id < 0:
        return
    jaw_transform = body_q[body_id]
    target = wp.transform_point(
        jaw_transform, particle_grip_local[particle_id]
    )
    # Zero-history compliant-XPBD projection.  Resetting the multiplier each
    # contact solve avoids stale warm-start forces because contact is evaluated
    # only every Nth material substep.  compliance=0 recovers a hard attachment.
    inverse_particle_mass = inverse_mass[particle_id]
    alpha = compliance / wp.max(dt * dt, 1.0e-12)
    compliant_fraction = inverse_particle_mass / (
        inverse_particle_mass + alpha
    )
    delta = (
        relaxation
        * particle_grip_weight[particle_id]
        * compliant_fraction
        * (target - positions[particle_id])
    )
    length = wp.length(delta)
    if length > maximum_correction and length > 1.0e-12:
        delta *= maximum_correction / length
    grip_deltas[particle_id] = delta
    grip_delta_counts[particle_id] = 1.0


@wp.kernel(enable_backward=False)
def filter_persistent_grip_jaw_deltas(
    input_deltas: wp.array(dtype=wp.vec3),
    input_weights: wp.array(dtype=float),
    particle_grip_body: wp.array(dtype=int),
    jaw_body_id: int,
    output_deltas: wp.array(dtype=wp.vec3),
    output_weights: wp.array(dtype=float),
):
    """Extract one jaw's controls before finite-volume propagation."""
    particle_id = wp.tid()
    if (
        input_weights[particle_id] <= 0.0
        or particle_grip_body[particle_id] != jaw_body_id
    ):
        return
    output_deltas[particle_id] = input_deltas[particle_id]
    output_weights[particle_id] = input_weights[particle_id]


@wp.kernel(enable_backward=False)
def prepare_jaw_contact_contribution_sort(
    contribution_nodes: wp.array(dtype=int),
    contribution_count: int,
    sort_keys: wp.array(dtype=int),
    sort_values: wp.array(dtype=int),
):
    contribution_index = wp.tid()
    particle_index = contribution_nodes[contribution_index]
    key = 2147483647
    if particle_index >= 0:
        # A unique secondary contribution id fixes the floating-point sum
        # order even when many samples land on the same skin vertex.
        key = (
            particle_index * contribution_count
            + contribution_index
        )
    sort_keys[contribution_index] = key
    sort_values[contribution_index] = contribution_index


@wp.kernel(enable_backward=False)
def reduce_sorted_jaw_contact_contributions(
    contribution_deltas: wp.array(dtype=wp.vec3),
    contribution_count: int,
    sorted_keys: wp.array(dtype=int),
    sorted_values: wp.array(dtype=int),
    deltas: wp.array(dtype=wp.vec3),
    delta_counts: wp.array(dtype=float),
):
    """Reduce jaw contributions in the unique sorted-key order."""
    particle_index = wp.tid()
    lower_key = particle_index * contribution_count
    upper_key = lower_key + contribution_count

    lower = int(0)
    upper = int(contribution_count)
    while lower < upper:
        middle = (lower + upper) // 2
        if sorted_keys[middle] < lower_key:
            lower = middle + 1
        else:
            upper = middle

    total = wp.vec3()
    count = int(0)
    contribution_offset = int(lower)
    while (
        contribution_offset < contribution_count
        and sorted_keys[contribution_offset] < upper_key
    ):
        source_index = sorted_values[contribution_offset]
        total += contribution_deltas[source_index]
        count += 1
        contribution_offset += 1
    if count > 0:
        deltas[particle_index] += total
        delta_counts[particle_index] += float(count)


@wp.kernel(enable_backward=False)
def accumulate_oriented_top_skin_barrier_deltas(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
    top_faces: wp.array(dtype=int, ndim=2),
    top_support_offsets: wp.array(dtype=int),
    top_support_nodes: wp.array(dtype=int),
    top_support_weights: wp.array(dtype=float),
    top_pressure_shoulder_offsets: wp.array(dtype=int),
    top_pressure_shoulder_nodes: wp.array(dtype=int),
    top_pressure_shoulder_weights: wp.array(dtype=float),
    top_mesh: wp.uint64,
    tool_sample_local: wp.array(dtype=wp.vec3),
    tool_sample_shape_center_local: wp.array(dtype=wp.vec3),
    tool_sample_shape: wp.array(dtype=int),
    top_barrier_sample_enabled: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    query_distance: float,
    lateral_tolerance: float,
    contact_patch_radius: float,
    contact_margin: float,
    ccd_velocity_scale: float,
    dt: float,
    relaxation: float,
    top_pressure_shoulder_upward_scale: float,
    top_pressure_shoulder_outward_scale: float,
    top_pressure_shoulder_bias_direction_world: wp.vec3,
    top_pressure_shoulder_bias_start_m: float,
    persistent_grip_contact_patch_sums: wp.array(dtype=float),
    persistent_grip_grasp_center: wp.array(dtype=wp.vec3),
    surface_downward_deltas: wp.array(dtype=float),
    pressure_shoulder_deltas: wp.array(dtype=wp.vec3),
    pressure_shoulder_counts: wp.array(dtype=float),
    pressure_shoulder_direct_override: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_count_by_shape: wp.array(dtype=int),
    top_barrier_contact_count: wp.array(dtype=int),
    maximum_penetration: wp.array(dtype=float),
    minimum_signed_distance: wp.array(dtype=float),
    top_minimum_signed_distance: wp.array(dtype=float),
):
    """Keep tool samples above the oriented deforming top sheet.

    Unlike a closed-volume sign query, this barrier cannot declare a tool
    sample safe merely because it has already crossed through the thin top
    sheet and exited the closed volume on the other side.
    """
    sample_index = wp.tid()
    if top_barrier_sample_enabled[sample_index] == 0:
        return
    shape_index = tool_sample_shape[sample_index]
    body_index = shape_body[shape_index]
    if body_index < 0:
        return

    X_wb = body_q[body_index]
    X_bs = shape_transform[shape_index]
    X_ws = wp.transform_multiply(X_wb, X_bs)
    sample_scaled = wp.cw_mul(
        tool_sample_local[sample_index], shape_scale[shape_index]
    )
    tool_point = wp.transform_point(X_ws, sample_scaled)
    query = wp.mesh_query_point_no_sign(
        top_mesh, tool_point, query_distance
    )
    if not query.result:
        return

    skin_point = wp.mesh_eval_position(
        top_mesh, query.face, query.u, query.v
    )
    index_a = top_faces[query.face, 0]
    index_b = top_faces[query.face, 1]
    index_c = top_faces[query.face, 2]
    edge_ab = positions[index_b] - positions[index_a]
    edge_ac = positions[index_c] - positions[index_a]
    normal_unnormalized = wp.cross(edge_ab, edge_ac)
    if wp.length_sq(normal_unnormalized) <= 1.0e-20:
        return
    top_normal = wp.normalize(normal_unnormalized)
    shape_center_scaled = wp.cw_mul(
        tool_sample_shape_center_local[sample_index],
        shape_scale[shape_index],
    )
    shape_center = wp.transform_point(X_ws, shape_center_scaled)
    # A closed jaw proxy contains upper, side, and lower surface samples.  If
    # all of them independently project onto the same top sheet, one physical
    # jaw becomes several stacked indenters and produces a large artificial
    # bulge.  Only the half facing the tissue with respect to the *current*
    # deformed top normal owns the unilateral barrier.  This remains valid as
    # the tool rotates and still covers the complete jaw length, so a small tip
    # penetration is possible without allowing the whole jaw body through.
    if wp.dot(tool_point - shape_center, top_normal) > 0.0:
        return

    # Before bilateral contact exists, clip direct downward motion around each
    # actual hit point. After both jaw patches exist, use their measured center
    # so the complete direct patch stays bounded through closure. This radius
    # limits prescribed contact motion only; XPBD may still respond naturally
    # outside it.
    center_difference = skin_point - persistent_grip_grasp_center[0]
    center_tangent = center_difference - (
        wp.dot(center_difference, top_normal) * top_normal
    )
    bilateral_patch_valid = (
        persistent_grip_contact_patch_sums[3] > 0.0
        and persistent_grip_contact_patch_sums[7] > 0.0
    )
    if (
        bilateral_patch_valid
        and
        contact_patch_radius > 0.0
        and wp.length(center_tangent) > contact_patch_radius
    ):
        return
    patch_center = skin_point
    if bilateral_patch_valid:
        patch_center = persistent_grip_grasp_center[0]
    vertex_a_difference = positions[index_a] - patch_center
    vertex_b_difference = positions[index_b] - patch_center
    vertex_c_difference = positions[index_c] - patch_center
    vertex_a_tangent = vertex_a_difference - (
        wp.dot(vertex_a_difference, top_normal) * top_normal
    )
    vertex_b_tangent = vertex_b_difference - (
        wp.dot(vertex_b_difference, top_normal) * top_normal
    )
    vertex_c_tangent = vertex_c_difference - (
        wp.dot(vertex_c_difference, top_normal) * top_normal
    )
    vertex_a_distance = wp.length(vertex_a_tangent)
    vertex_b_distance = wp.length(vertex_b_tangent)
    vertex_c_distance = wp.length(vertex_c_tangent)
    any_vertex_in_patch = (
        vertex_a_distance <= contact_patch_radius
        or vertex_b_distance <= contact_patch_radius
        or vertex_c_distance <= contact_patch_radius
    )
    # A coarse or skinny surface triangle can have its closest point inside the
    # configured tool core while every vertex lies just outside it. Keep
    # exactly the nearest vertex in that case so the unilateral barrier never
    # silently disappears, without restoring the full-triangle footprint.
    vertex_a_is_nearest = (
        vertex_a_distance <= vertex_b_distance
        and vertex_a_distance <= vertex_c_distance
    )
    vertex_b_is_nearest = (
        vertex_b_distance < vertex_a_distance
        and vertex_b_distance <= vertex_c_distance
    )
    vertex_c_is_nearest = (
        vertex_c_distance < vertex_a_distance
        and vertex_c_distance < vertex_b_distance
    )
    vertex_a_in_patch = (
        contact_patch_radius <= 0.0
        or vertex_a_distance <= contact_patch_radius
        or (not any_vertex_in_patch and vertex_a_is_nearest)
    )
    vertex_b_in_patch = (
        contact_patch_radius <= 0.0
        or vertex_b_distance <= contact_patch_radius
        or (not any_vertex_in_patch and vertex_b_is_nearest)
    )
    vertex_c_in_patch = (
        contact_patch_radius <= 0.0
        or vertex_c_distance <= contact_patch_radius
        or (not any_vertex_in_patch and vertex_c_is_nearest)
    )
    difference = tool_point - skin_point
    signed_distance = wp.dot(difference, top_normal)
    tangential_difference = difference - signed_distance * top_normal
    if wp.length(tangential_difference) > lateral_tolerance:
        return
    wp.atomic_min(minimum_signed_distance, 0, signed_distance)
    wp.atomic_min(top_minimum_signed_distance, 0, signed_distance)

    bary_a = wp.clamp(query.u, 0.0, 1.0)
    bary_b = wp.clamp(query.v, 0.0, 1.0)
    bary_c = wp.clamp(1.0 - bary_a - bary_b, 0.0, 1.0)
    bary_sum = bary_a + bary_b + bary_c
    if bary_sum <= 1.0e-8:
        return
    bary_a /= bary_sum
    bary_b /= bary_sum
    bary_c /= bary_sum

    skin_velocity = (
        bary_a * velocities[index_a]
        + bary_b * velocities[index_b]
        + bary_c * velocities[index_c]
    )
    spatial_velocity = body_qd[body_index]
    angular_velocity = wp.spatial_top(spatial_velocity)
    com_velocity = wp.spatial_bottom(spatial_velocity)
    world_com = wp.transform_point(X_wb, body_com[body_index])
    tool_velocity = com_velocity + wp.cross(
        angular_velocity, tool_point - world_com
    )
    inward_speed = wp.max(
        0.0, -wp.dot(tool_velocity - skin_velocity, top_normal)
    )
    effective_margin = (
        contact_margin + ccd_velocity_scale * inward_speed * dt
    )
    penetration = effective_margin - signed_distance
    if penetration <= 0.0:
        return

    inverse_mass_a = inverse_mass[index_a]
    inverse_mass_b = inverse_mass[index_b]
    inverse_mass_c = inverse_mass[index_c]
    correction = -relaxation * top_normal * penetration
    downward_correction = wp.min(correction[2], 0.0)
    biased_shoulder = (
        wp.length_sq(top_pressure_shoulder_bias_direction_world) > 0.0
        and wp.dot(
            skin_point - persistent_grip_grasp_center[0],
            top_pressure_shoulder_bias_direction_world,
        ) > top_pressure_shoulder_bias_start_m
    )
    if biased_shoulder:
        biased_delta = (
            top_pressure_shoulder_upward_scale
            * wp.max(0.0, -downward_correction)
            * top_normal
        )
        if (
            vertex_a_in_patch
            and
            inverse_mass_a > 0.0
            and (particle_flags[index_a] & PARTICLE_FLAG_ACTIVE) != 0
        ):
            wp.atomic_add(pressure_shoulder_deltas, index_a, biased_delta)
            wp.atomic_add(pressure_shoulder_counts, index_a, 1.0)
            wp.atomic_max(pressure_shoulder_direct_override, index_a, 1)
        if (
            vertex_b_in_patch
            and
            inverse_mass_b > 0.0
            and (particle_flags[index_b] & PARTICLE_FLAG_ACTIVE) != 0
        ):
            wp.atomic_add(pressure_shoulder_deltas, index_b, biased_delta)
            wp.atomic_add(pressure_shoulder_counts, index_b, 1.0)
            wp.atomic_max(pressure_shoulder_direct_override, index_b, 1)
        if (
            vertex_c_in_patch
            and
            inverse_mass_c > 0.0
            and (particle_flags[index_c] & PARTICLE_FLAG_ACTIVE) != 0
        ):
            wp.atomic_add(pressure_shoulder_deltas, index_c, biased_delta)
            wp.atomic_add(pressure_shoulder_counts, index_c, 1.0)
            wp.atomic_max(pressure_shoulder_direct_override, index_c, 1)
    else:
        if (
            vertex_a_in_patch
            and
            inverse_mass_a > 0.0
            and (particle_flags[index_a] & PARTICLE_FLAG_ACTIVE) != 0
        ):
            wp.atomic_min(surface_downward_deltas, index_a, downward_correction)
        if (
            vertex_b_in_patch
            and
            inverse_mass_b > 0.0
            and (particle_flags[index_b] & PARTICLE_FLAG_ACTIVE) != 0
        ):
            wp.atomic_min(surface_downward_deltas, index_b, downward_correction)
        if (
            vertex_c_in_patch
            and
            inverse_mass_c > 0.0
            and (particle_flags[index_c] & PARTICLE_FLAG_ACTIVE) != 0
        ):
            wp.atomic_min(surface_downward_deltas, index_c, downward_correction)

    support_start = top_support_offsets[query.face]
    support_end = top_support_offsets[query.face + 1]
    for support_offset in range(support_start, support_end):
        support_node = top_support_nodes[support_offset]
        if (
            inverse_mass[support_node] > 0.0
            and (
                particle_flags[support_node] & PARTICLE_FLAG_ACTIVE
            ) != 0
        ):
            support_weight = top_support_weights[support_offset]
            wp.atomic_min(
                surface_downward_deltas,
                support_node,
                support_weight * downward_correction,
            )

    # A nearly incompressible solid cannot make every surface vertex around a
    # descending indenter follow the indenter.  Move the non-contact shoulder
    # candidates away from this pressure source.  Contributions from all hit
    # faces are averaged later, so their tangential directions combine into a
    # radial displacement away from the complete contact patch rather than
    # away from one arbitrary tool sample.
    shoulder_start = top_pressure_shoulder_offsets[query.face]
    shoulder_end = top_pressure_shoulder_offsets[query.face + 1]
    pressure_depth = wp.max(0.0, -downward_correction)
    for shoulder_offset in range(shoulder_start, shoulder_end):
        shoulder_node = top_pressure_shoulder_nodes[shoulder_offset]
        if (
            inverse_mass[shoulder_node] > 0.0
            and (
                particle_flags[shoulder_node] & PARTICLE_FLAG_ACTIVE
            ) != 0
        ):
            shoulder_difference = positions[shoulder_node] - skin_point
            shoulder_tangent = shoulder_difference - wp.dot(
                shoulder_difference, top_normal
            ) * top_normal
            tangent_length = wp.length(shoulder_tangent)
            outward_direction = wp.vec3()
            if tangent_length > 1.0e-10:
                outward_direction = shoulder_tangent / tangent_length
            shoulder_weight = top_pressure_shoulder_weights[
                shoulder_offset
            ]
            shoulder_delta = shoulder_weight * pressure_depth * (
                top_pressure_shoulder_upward_scale * top_normal
                + top_pressure_shoulder_outward_scale * outward_direction
            )
            wp.atomic_add(
                pressure_shoulder_deltas, shoulder_node, shoulder_delta
            )
            wp.atomic_add(pressure_shoulder_counts, shoulder_node, 1.0)
    wp.atomic_add(contact_count, 0, 1)
    wp.atomic_add(contact_count_by_shape, shape_index, 1)
    wp.atomic_add(top_barrier_contact_count, 0, 1)
    wp.atomic_max(maximum_penetration, 0, penetration)


@wp.kernel(enable_backward=False)
def average_and_clamp_triangle_skin_contact_deltas(
    deltas: wp.array(dtype=wp.vec3),
    delta_counts: wp.array(dtype=float),
    maximum_correction: float,
):
    particle_index = wp.tid()
    count = delta_counts[particle_index]
    if count <= 0.0:
        return
    delta = deltas[particle_index] / count
    length = wp.length(delta)
    if maximum_correction > 0.0 and length > maximum_correction:
        delta *= maximum_correction / length
    deltas[particle_index] = delta


@wp.kernel(enable_backward=False)
def mask_jaw_contact_deltas_to_bilateral_patch(
    positions: wp.array(dtype=wp.vec3),
    contact_patch_sums: wp.array(dtype=float),
    grasp_center: wp.array(dtype=wp.vec3),
    radius: float,
    deltas: wp.array(dtype=wp.vec3),
    delta_counts: wp.array(dtype=float),
):
    """Limit direct jaw motion after a bilateral patch has been observed."""
    particle_id = wp.tid()
    if radius <= 0.0:
        return
    # Before both jaws touch, do not suppress the contacts needed to establish
    # the midpoint itself.  As soon as both patches exist, direct jaw motion is
    # local; the tetrahedral material alone may transmit it farther.
    if contact_patch_sums[3] <= 0.0 or contact_patch_sums[7] <= 0.0:
        return
    difference = positions[particle_id] - grasp_center[0]
    if wp.length(difference) > radius:
        deltas[particle_id] = wp.vec3()
        delta_counts[particle_id] = 0.0


@wp.kernel(enable_backward=False)
def merge_oriented_top_surface_constraint_deltas(
    deltas: wp.array(dtype=wp.vec3),
    delta_counts: wp.array(dtype=float),
    surface_downward_deltas: wp.array(dtype=float),
    top_barrier_maximum_correction: float,
    maximum_correction: float,
):
    """Merge the deepest unilateral top-surface constraint per particle.

    Averaging thousands of overlapping tool samples can dilute the deepest
    penetration until the visible skin barely moves.  The top sheet is a
    unilateral constraint instead: each affected particle keeps the largest
    required downward correction, while the precomputed support weights form
    a smooth local volume patch below and around the hit triangle.
    """
    particle_index = wp.tid()
    surface_downward = surface_downward_deltas[particle_index]
    if surface_downward >= 0.0:
        return
    # The top-sheet normal barrier is deliberately softer than the ordinary
    # jaw contact.  Limit only its downward component here so q7 tangential
    # closure can retain the larger general contact budget.
    if (
        top_barrier_maximum_correction > 0.0
        and surface_downward < -top_barrier_maximum_correction
    ):
        surface_downward = -top_barrier_maximum_correction
    delta = deltas[particle_index]
    if delta_counts[particle_index] <= 0.0:
        delta = wp.vec3(0.0, 0.0, surface_downward)
    else:
        delta[2] = wp.min(delta[2], surface_downward)
    length = wp.length(delta)
    if maximum_correction > 0.0 and length > maximum_correction:
        delta *= maximum_correction / length
    deltas[particle_index] = delta
    delta_counts[particle_index] = wp.max(
        delta_counts[particle_index], 1.0
    )


@wp.kernel(enable_backward=False)
def merge_oriented_top_pressure_shoulder_deltas(
    deltas: wp.array(dtype=wp.vec3),
    delta_counts: wp.array(dtype=float),
    surface_downward_deltas: wp.array(dtype=float),
    pressure_shoulder_deltas: wp.array(dtype=wp.vec3),
    pressure_shoulder_counts: wp.array(dtype=float),
    pressure_shoulder_direct_override: wp.array(dtype=int),
    persistent_grip_state: wp.array(dtype=int),
    maximum_correction: float,
):
    """Apply squeeze displacement only to the outside of the direct footprint.

    A node touched by either the ordinary triangle contact or the oriented top
    barrier remains a direct contact node and must obey that unilateral
    constraint.  Only untouched candidates form the raised shoulder.  This
    prevents the pressure-transfer term from fighting the jaw at the center.
    """
    particle_index = wp.tid()
    # Once four particles are captured the tool is gripping/pulling, not
    # continuing to indent the top sheet. Keeping the squeeze source active in
    # that phase over-pressurizes the anchor tetrahedra and fights the grasp.
    if persistent_grip_state[0] != 0:
        return
    count = pressure_shoulder_counts[particle_index]
    if (
        count <= 0.0
        or (
            delta_counts[particle_index] > 0.0
            and pressure_shoulder_direct_override[particle_index] == 0
        )
        or (
            surface_downward_deltas[particle_index] < 0.0
            and pressure_shoulder_direct_override[particle_index] == 0
        )
    ):
        return
    delta = pressure_shoulder_deltas[particle_index] / count
    length = wp.length(delta)
    if maximum_correction > 0.0 and length > maximum_correction:
        delta *= maximum_correction / length
    deltas[particle_index] = delta
    delta_counts[particle_index] = 1.0


@wp.kernel(enable_backward=False)
def prepare_oriented_top_pressure_shoulder_deltas(
    direct_counts: wp.array(dtype=float),
    surface_downward_deltas: wp.array(dtype=float),
    pressure_shoulder_deltas: wp.array(dtype=wp.vec3),
    pressure_shoulder_counts: wp.array(dtype=float),
    pressure_shoulder_direct_override: wp.array(dtype=int),
    persistent_grip_state: wp.array(dtype=int),
    maximum_correction: float,
    output_deltas: wp.array(dtype=wp.vec3),
    output_weights: wp.array(dtype=float),
):
    """Prepare the post-material shoulder pass without touching direct nodes."""
    particle_index = wp.tid()
    if persistent_grip_state[0] != 0:
        return
    count = pressure_shoulder_counts[particle_index]
    if (
        count <= 0.0
        or (
            direct_counts[particle_index] > 0.0
            and pressure_shoulder_direct_override[particle_index] == 0
        )
        or (
            surface_downward_deltas[particle_index] < 0.0
            and pressure_shoulder_direct_override[particle_index] == 0
        )
    ):
        return
    delta = pressure_shoulder_deltas[particle_index] / count
    length = wp.length(delta)
    if maximum_correction > 0.0 and length > maximum_correction:
        delta *= maximum_correction / length
    output_deltas[particle_index] = delta
    output_weights[particle_index] = 1.0


@wp.kernel(enable_backward=False)
def seed_triangle_skin_contact_spread(
    direct_deltas: wp.array(dtype=wp.vec3),
    direct_counts: wp.array(dtype=float),
    spread_deltas: wp.array(dtype=wp.vec3),
    spread_weights: wp.array(dtype=float),
):
    particle_index = wp.tid()
    if direct_counts[particle_index] > 0.0:
        spread_deltas[particle_index] = direct_deltas[particle_index]
        spread_weights[particle_index] = 1.0
    else:
        spread_deltas[particle_index] = wp.vec3()
        spread_weights[particle_index] = 0.0


@wp.kernel(enable_backward=False)
def accumulate_triangle_skin_contact_spread(
    tet_indices: wp.array(dtype=int, ndim=2),
    spread_tet_ids: wp.array(dtype=int),
    input_deltas: wp.array(dtype=wp.vec3),
    input_weights: wp.array(dtype=float),
    inverse_mass: wp.array(dtype=float),
    output_deltas: wp.array(dtype=wp.vec3),
    output_weights: wp.array(dtype=float),
):
    """Extend a contact displacement through one incident-tetrahedron layer."""
    tet_id = spread_tet_ids[wp.tid()]
    index_a = tet_indices[tet_id, 0]
    index_b = tet_indices[tet_id, 1]
    index_c = tet_indices[tet_id, 2]
    index_d = tet_indices[tet_id, 3]

    total_delta = wp.vec3()
    source_count = 0.0
    if input_weights[index_a] > 0.0:
        total_delta += input_deltas[index_a]
        source_count += 1.0
    if input_weights[index_b] > 0.0:
        total_delta += input_deltas[index_b]
        source_count += 1.0
    if input_weights[index_c] > 0.0:
        total_delta += input_deltas[index_c]
        source_count += 1.0
    if input_weights[index_d] > 0.0:
        total_delta += input_deltas[index_d]
        source_count += 1.0
    if source_count <= 0.0:
        return

    average_delta = total_delta / source_count
    if inverse_mass[index_a] > 0.0:
        wp.atomic_add(output_deltas, index_a, average_delta)
        wp.atomic_add(output_weights, index_a, 1.0)
    if inverse_mass[index_b] > 0.0:
        wp.atomic_add(output_deltas, index_b, average_delta)
        wp.atomic_add(output_weights, index_b, 1.0)
    if inverse_mass[index_c] > 0.0:
        wp.atomic_add(output_deltas, index_c, average_delta)
        wp.atomic_add(output_weights, index_c, 1.0)
    if inverse_mass[index_d] > 0.0:
        wp.atomic_add(output_deltas, index_d, average_delta)
        wp.atomic_add(output_weights, index_d, 1.0)


@wp.kernel(enable_backward=False)
def normalize_triangle_skin_contact_spread(
    direct_deltas: wp.array(dtype=wp.vec3),
    direct_counts: wp.array(dtype=float),
    spread_deltas: wp.array(dtype=wp.vec3),
    spread_weights: wp.array(dtype=float),
):
    """Keep the exact skin correction and normalize only its volume support."""
    particle_index = wp.tid()
    if direct_counts[particle_index] > 0.0:
        spread_deltas[particle_index] = direct_deltas[particle_index]
        spread_weights[particle_index] = 1.0
        return
    weight = spread_weights[particle_index]
    if weight > 0.0:
        spread_deltas[particle_index] /= weight
        spread_weights[particle_index] = 1.0


@wp.kernel(enable_backward=False)
def remove_persistent_grip_direct_from_support(
    direct_counts: wp.array(dtype=float),
    support_deltas: wp.array(dtype=wp.vec3),
    support_weights: wp.array(dtype=float),
):
    """Keep the four grasp anchors out of the separately guarded support pass."""
    particle_index = wp.tid()
    if direct_counts[particle_index] > 0.0:
        support_deltas[particle_index] = wp.vec3()
        support_weights[particle_index] = 0.0


@wp.kernel(enable_backward=False)
def snapshot_persistent_grip_support_positions(
    positions: wp.array(dtype=wp.vec3),
    snapshot: wp.array(dtype=wp.vec3),
):
    particle_index = wp.tid()
    snapshot[particle_index] = positions[particle_index]


@wp.kernel(enable_backward=False)
def mark_unsafe_persistent_grip_support_particles(
    previous_positions: wp.array(dtype=wp.vec3),
    proposed_positions: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=int, ndim=2),
    inverse_rest_matrix: wp.array(dtype=wp.mat33),
    safety_tet_ids: wp.array(dtype=int),
    support_weights: wp.array(dtype=float),
    minimum_volume_ratio: float,
    unsafe_particles: wp.array(dtype=int),
    unsafe_tet_count: wp.array(dtype=int),
):
    """Flag only moved support nodes of a newly unsafe tetrahedron."""
    tet_id = safety_tet_ids[wp.tid()]
    index_a = tet_indices[tet_id, 0]
    index_b = tet_indices[tet_id, 1]
    index_c = tet_indices[tet_id, 2]
    index_d = tet_indices[tet_id, 3]
    if (
        support_weights[index_a] <= 0.0
        and support_weights[index_b] <= 0.0
        and support_weights[index_c] <= 0.0
        and support_weights[index_d] <= 0.0
    ):
        return

    rest_inverse = inverse_rest_matrix[tet_id]
    previous_a = previous_positions[index_a]
    previous_b = previous_positions[index_b]
    previous_c = previous_positions[index_c]
    previous_d = previous_positions[index_d]
    proposed_a = proposed_positions[index_a]
    proposed_b = proposed_positions[index_b]
    proposed_c = proposed_positions[index_c]
    proposed_d = proposed_positions[index_d]
    previous_matrix = wp.matrix_from_cols(
        previous_b - previous_a,
        previous_c - previous_a,
        previous_d - previous_a,
    )
    proposed_matrix = wp.matrix_from_cols(
        proposed_b - proposed_a,
        proposed_c - proposed_a,
        proposed_d - proposed_a,
    )
    previous_ratio = wp.determinant(previous_matrix * rest_inverse)
    proposed_ratio = wp.determinant(proposed_matrix * rest_inverse)
    if not (
        proposed_ratio <= minimum_volume_ratio
        and proposed_ratio < previous_ratio
    ):
        return

    wp.atomic_add(unsafe_tet_count, 0, 1)
    if support_weights[index_a] > 0.0:
        wp.atomic_max(unsafe_particles, index_a, 1)
    if support_weights[index_b] > 0.0:
        wp.atomic_max(unsafe_particles, index_b, 1)
    if support_weights[index_c] > 0.0:
        wp.atomic_max(unsafe_particles, index_c, 1)
    if support_weights[index_d] > 0.0:
        wp.atomic_max(unsafe_particles, index_d, 1)


@wp.kernel(enable_backward=False)
def revert_unsafe_persistent_grip_support_particles(
    previous_positions: wp.array(dtype=wp.vec3),
    positions: wp.array(dtype=wp.vec3),
    support_weights: wp.array(dtype=float),
    unsafe_particles: wp.array(dtype=int),
):
    particle_index = wp.tid()
    if (
        support_weights[particle_index] > 0.0
        and unsafe_particles[particle_index] != 0
    ):
        positions[particle_index] = previous_positions[particle_index]


@wp.kernel(enable_backward=False)
def find_safe_triangle_skin_contact_scale(
    positions: wp.array(dtype=wp.vec3),
    deltas: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=int, ndim=2),
    inverse_rest_matrix: wp.array(dtype=wp.mat33),
    contact_tet_ids: wp.array(dtype=int),
    minimum_volume_ratio: float,
    global_scale: wp.array(dtype=float),
):
    tet_id = contact_tet_ids[wp.tid()]
    index_a = tet_indices[tet_id, 0]
    index_b = tet_indices[tet_id, 1]
    index_c = tet_indices[tet_id, 2]
    index_d = tet_indices[tet_id, 3]
    rest_inverse = inverse_rest_matrix[tet_id]

    point_a = positions[index_a]
    point_b = positions[index_b]
    point_c = positions[index_c]
    point_d = positions[index_d]
    delta_a = deltas[index_a]
    delta_b = deltas[index_b]
    delta_c = deltas[index_c]
    delta_d = deltas[index_d]
    # A previously compressed surface tet must not freeze unrelated contacts.
    # If this proposal does not touch any of its vertices, its volume cannot
    # change and it has no bearing on the safe scale for the current pass.
    delta_energy = (
        wp.dot(delta_a, delta_a)
        + wp.dot(delta_b, delta_b)
        + wp.dot(delta_c, delta_c)
        + wp.dot(delta_d, delta_d)
    )
    if delta_energy <= 1.0e-20:
        return
    current_matrix = wp.matrix_from_cols(
        point_b - point_a, point_c - point_a, point_d - point_a
    )
    current_ratio = wp.determinant(current_matrix * rest_inverse)

    proposed_matrix = wp.matrix_from_cols(
        (point_b + delta_b) - (point_a + delta_a),
        (point_c + delta_c) - (point_a + delta_a),
        (point_d + delta_d) - (point_a + delta_a),
    )
    proposed_ratio = wp.determinant(proposed_matrix * rest_inverse)
    # Do not give an already compressed but still positive element veto power
    # over the grasp.  Its admissible floor is a fraction of its current
    # positive volume, so the search can still guarantee non-inversion while
    # permitting a local patch to move away from a pre-existing sliver.  A
    # genuinely inverted element may only stay unchanged or improve.
    admissible_ratio = minimum_volume_ratio
    if current_ratio <= 0.0:
        if proposed_ratio >= current_ratio:
            return
        wp.atomic_min(global_scale, 0, 0.0)
        return
    if current_ratio <= minimum_volume_ratio:
        admissible_ratio = wp.max(1.0e-12, 0.25 * current_ratio)
    if proposed_ratio >= admissible_ratio:
        return

    lower = float(0.0)
    upper = float(1.0)
    # Twenty bisections retain a small legal attachment step instead of
    # quantizing scales below 1/4096 to zero on a thin boundary element.
    for _ in range(20):
        middle = 0.5 * (lower + upper)
        middle_matrix = wp.matrix_from_cols(
            (point_b + middle * delta_b)
            - (point_a + middle * delta_a),
            (point_c + middle * delta_c)
            - (point_a + middle * delta_a),
            (point_d + middle * delta_d)
            - (point_a + middle * delta_a),
        )
        middle_ratio = wp.determinant(middle_matrix * rest_inverse)
        if middle_ratio >= admissible_ratio:
            lower = middle
        else:
            upper = middle
    wp.atomic_min(global_scale, 0, lower)


@wp.kernel(enable_backward=False)
def find_safe_persistent_grip_jaw_scale(
    positions: wp.array(dtype=wp.vec3),
    deltas: wp.array(dtype=wp.vec3),
    particle_grip_body: wp.array(dtype=int),
    jaw_body_id: int,
    tet_indices: wp.array(dtype=int, ndim=2),
    inverse_rest_matrix: wp.array(dtype=wp.mat33),
    contact_tet_ids: wp.array(dtype=int),
    minimum_volume_ratio: float,
    scale_index: int,
    jaw_scales: wp.array(dtype=float),
):
    """Find a safe scale for the controls attached to one physical jaw."""
    tet_id = contact_tet_ids[wp.tid()]
    index_a = tet_indices[tet_id, 0]
    index_b = tet_indices[tet_id, 1]
    index_c = tet_indices[tet_id, 2]
    index_d = tet_indices[tet_id, 3]
    rest_inverse = inverse_rest_matrix[tet_id]

    point_a = positions[index_a]
    point_b = positions[index_b]
    point_c = positions[index_c]
    point_d = positions[index_d]
    delta_a = wp.vec3()
    delta_b = wp.vec3()
    delta_c = wp.vec3()
    delta_d = wp.vec3()
    if particle_grip_body[index_a] == jaw_body_id:
        delta_a = deltas[index_a]
    if particle_grip_body[index_b] == jaw_body_id:
        delta_b = deltas[index_b]
    if particle_grip_body[index_c] == jaw_body_id:
        delta_c = deltas[index_c]
    if particle_grip_body[index_d] == jaw_body_id:
        delta_d = deltas[index_d]
    delta_energy = (
        wp.dot(delta_a, delta_a)
        + wp.dot(delta_b, delta_b)
        + wp.dot(delta_c, delta_c)
        + wp.dot(delta_d, delta_d)
    )
    if delta_energy <= 1.0e-20:
        return

    current_matrix = wp.matrix_from_cols(
        point_b - point_a, point_c - point_a, point_d - point_a
    )
    current_ratio = wp.determinant(current_matrix * rest_inverse)
    proposed_matrix = wp.matrix_from_cols(
        (point_b + delta_b) - (point_a + delta_a),
        (point_c + delta_c) - (point_a + delta_a),
        (point_d + delta_d) - (point_a + delta_a),
    )
    proposed_ratio = wp.determinant(proposed_matrix * rest_inverse)

    if current_ratio <= 0.0:
        if proposed_ratio >= current_ratio:
            return
        wp.atomic_min(jaw_scales, scale_index, 0.0)
        return
    # Below the preferred floor, u_t may improve or preserve the element but
    # must not consume more volume. Healthy elements keep the absolute floor.
    admissible_ratio = minimum_volume_ratio
    if current_ratio <= minimum_volume_ratio:
        admissible_ratio = current_ratio
    if proposed_ratio >= admissible_ratio:
        return

    lower = float(0.0)
    upper = float(1.0)
    for _ in range(20):
        middle = 0.5 * (lower + upper)
        middle_matrix = wp.matrix_from_cols(
            (point_b + middle * delta_b)
            - (point_a + middle * delta_a),
            (point_c + middle * delta_c)
            - (point_a + middle * delta_a),
            (point_d + middle * delta_d)
            - (point_a + middle * delta_a),
        )
        middle_ratio = wp.determinant(middle_matrix * rest_inverse)
        if middle_ratio >= admissible_ratio:
            lower = middle
        else:
            upper = middle
    wp.atomic_min(jaw_scales, scale_index, lower)


@wp.kernel(enable_backward=False)
def apply_persistent_grip_jaw_deltas(
    positions: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
    deltas: wp.array(dtype=wp.vec3),
    delta_counts: wp.array(dtype=float),
    particle_grip_body: wp.array(dtype=int),
    jaw_body_id: int,
    scale_index: int,
    jaw_scales: wp.array(dtype=float),
):
    particle_index = wp.tid()
    if (
        delta_counts[particle_index] <= 0.0
        or particle_grip_body[particle_index] != jaw_body_id
        or inverse_mass[particle_index] <= 0.0
        or (particle_flags[particle_index] & PARTICLE_FLAG_ACTIVE) == 0
    ):
        return
    positions[particle_index] = (
        positions[particle_index]
        + jaw_scales[scale_index] * deltas[particle_index]
    )


@wp.kernel(enable_backward=False)
def reduce_persistent_grip_jaw_scales(
    jaw_scales: wp.array(dtype=float),
    minimum_scale: wp.array(dtype=float),
):
    if wp.tid() == 0:
        minimum_scale[0] = wp.min(jaw_scales[0], jaw_scales[1])


@wp.kernel(enable_backward=False)
def apply_triangle_skin_contact_deltas(
    positions: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
    deltas: wp.array(dtype=wp.vec3),
    spread_weights: wp.array(dtype=float),
    global_scale: wp.array(dtype=float),
):
    particle_index = wp.tid()
    if (
        spread_weights[particle_index] <= 0.0
        or inverse_mass[particle_index] <= 0.0
        or (particle_flags[particle_index] & PARTICLE_FLAG_ACTIVE) == 0
    ):
        return
    positions[particle_index] = (
        positions[particle_index]
        + global_scale[0] * deltas[particle_index]
    )


class TriangleSkinContactProjector:
    """Projects kinematic tool-mesh samples against a deforming closed skin."""

    def __init__(
        self,
        model: warp.sim.Model,
        skin_faces: np.ndarray,
        tool_shape_ids: list[int] | tuple[int, ...],
        sample_spacing_m: float = 0.00035,
        spread_layers: int = 0,
        top_skin_faces: np.ndarray | None = None,
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
    ):
        faces = np.asarray(skin_faces, dtype=np.int32)
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise ValueError(
                f"Triangle-skin contact requires non-empty Mx3 faces, got {faces.shape}"
            )
        if faces.min() < 0 or faces.max() >= model.particle_count:
            raise ValueError("Triangle-skin face index is outside model particles")
        unique_shape_ids = sorted({int(value) for value in tool_shape_ids})
        if not unique_shape_ids:
            raise ValueError("Triangle-skin contact requires tool shape ids")
        invalid_shape_ids = [
            shape_id
            for shape_id in unique_shape_ids
            if shape_id < 0 or shape_id >= model.shape_count
        ]
        if invalid_shape_ids:
            raise ValueError(
                f"Tool shape ids are outside model: {invalid_shape_ids}"
            )
        if spread_layers < 0:
            raise ValueError("Triangle-skin contact spread layers cannot be negative")
        if top_support_lateral_radius_m < 0.0:
            raise ValueError("Top-contact support radius cannot be negative")
        if top_support_depth_m < 0.0:
            raise ValueError("Top-contact support depth cannot be negative")
        if not 0.0 <= top_support_weight_scale <= 1.0:
            raise ValueError(
                "Top-contact support weight scale must be in [0, 1]"
            )
        if top_pressure_shoulder_lateral_radius_m < 0.0:
            raise ValueError("Top pressure-shoulder radius cannot be negative")
        if top_pressure_shoulder_depth_m < 0.0:
            raise ValueError("Top pressure-shoulder depth cannot be negative")
        if top_pressure_shoulder_upward_scale < 0.0:
            raise ValueError("Top pressure-shoulder upward scale cannot be negative")
        if top_pressure_shoulder_outward_scale < 0.0:
            raise ValueError("Top pressure-shoulder outward scale cannot be negative")
        shoulder_bias_direction = np.asarray(
            top_pressure_shoulder_bias_direction_world, dtype=np.float64
        )
        if shoulder_bias_direction.shape != (3,):
            raise ValueError("Top pressure-shoulder bias direction must have three values")
        shoulder_bias_length = np.linalg.norm(shoulder_bias_direction)
        if shoulder_bias_length > 0.0:
            shoulder_bias_direction /= shoulder_bias_length
        if top_pressure_shoulder_bias_start_m < 0.0:
            raise ValueError("Top pressure-shoulder bias start cannot be negative")
        if top_barrier_lateral_tolerance_m <= 0.0:
            raise ValueError("Top-barrier lateral tolerance must be positive")
        if top_barrier_contact_patch_radius_m < 0.0:
            raise ValueError("Top-barrier contact-patch radius cannot be negative")
        if top_barrier_clearance_m < 0.0:
            raise ValueError("Top-barrier clearance cannot be negative")
        if jaw_friction_coefficient < 0.0:
            raise ValueError("Jaw friction coefficient cannot be negative")
        if jaw_contact_distal_length_m < 0.0:
            raise ValueError("Jaw contact distal length cannot be negative")
        if top_barrier_distal_length_m < 0.0:
            raise ValueError("Top-barrier distal length cannot be negative")
        if top_barrier_tip_allowance_m < 0.0:
            raise ValueError("Top-barrier tip allowance cannot be negative")
        effective_top_barrier_distal_length_m = (
            top_barrier_distal_length_m
            if top_barrier_distal_length_m > 0.0
            else jaw_contact_distal_length_m
        )
        if (
            effective_top_barrier_distal_length_m > 0.0
            and top_barrier_tip_allowance_m
            >= effective_top_barrier_distal_length_m
        ):
            raise ValueError(
                "Top-barrier tip allowance must be shorter than the jaw "
                "top-barrier distal length"
            )
        if (
            jaw_contact_distal_length_m > 0.0
            and effective_top_barrier_distal_length_m
            > jaw_contact_distal_length_m
        ):
            raise ValueError(
                "Top-barrier distal length cannot exceed the jaw contact "
                "distal length"
            )
        if persistent_grip_enabled and not jaw_contact_shape_ids:
            raise ValueError("Persistent grip requires two jaw contact shapes")
        if persistent_grip_minimum_contact_samples_per_jaw < 1:
            raise ValueError(
                "Persistent-grip minimum contact samples per jaw must be positive"
            )
        if persistent_grip_nearest_surface_particles != 4:
            raise ValueError(
                "The point-controlled persistent grip requires exactly four surface particles"
            )
        if persistent_grip_maximum_jaw_patch_separation_m <= 0.0:
            raise ValueError(
                "Persistent-grip jaw-patch separation must be positive"
            )
        if not (
            persistent_grip_closed_angle_max_rad
            < persistent_grip_release_angle_min_rad
            <= persistent_grip_wide_open_angle_rad
        ):
            raise ValueError(
                "Persistent-grip q7 thresholds must satisfy "
                "closed < release <= wide-open"
            )
        if persistent_grip_angle_motion_epsilon_rad <= 0.0:
            raise ValueError(
                "Persistent-grip q7 motion epsilon must be positive"
            )
        if persistent_grip_maximum_capture_penetration_m <= 0.0:
            raise ValueError(
                "Persistent-grip maximum capture penetration must be positive"
            )
        if persistent_grip_minimum_capture_volume_ratio <= 0.0:
            raise ValueError(
                "Persistent-grip minimum capture volume ratio must be positive"
            )
        if persistent_grip_compliance_m_per_n < 0.0:
            raise ValueError(
                "Persistent-grip attachment compliance cannot be negative"
            )
        if persistent_grip_transfer_layers < 1:
            raise ValueError(
                "Persistent-grip tetrahedron transfer layers must be positive"
            )
        if persistent_grip_minimum_volume_ratio <= 0.0:
            raise ValueError(
                "Persistent-grip minimum volume ratio must be positive"
            )
        if persistent_grip_support_radius_m < 0.0:
            raise ValueError(
                "Persistent-grip support radius cannot be negative"
            )
        if persistent_grip_support_generations < 1:
            raise ValueError(
                "Persistent-grip support generations must be positive"
            )
        if persistent_grip_release_angle_delta_rad <= 0.0:
            raise ValueError(
                "Persistent-grip q7 release delta must be positive"
            )
        # No oriented top mesh means no top-only barrier path.  Treating every
        # tool shape as a barrier in that case disables the ordinary surface
        # samples and silently produces zero contact.
        barrier_shape_ids = (
            []
            if top_barrier_shape_ids is None
            else sorted({int(value) for value in top_barrier_shape_ids})
        )
        invalid_barrier_shape_ids = sorted(
            set(barrier_shape_ids).difference(unique_shape_ids)
        )
        if invalid_barrier_shape_ids:
            raise ValueError(
                "Top-barrier shape ids must be tool shapes: "
                f"{invalid_barrier_shape_ids}"
            )
        jaw_shape_ids = (
            ()
            if jaw_contact_shape_ids is None
            else tuple(int(value) for value in jaw_contact_shape_ids)
        )
        if jaw_shape_ids and (
            len(jaw_shape_ids) != 2
            or jaw_shape_ids[0] == jaw_shape_ids[1]
        ):
            raise ValueError(
                "Jaw triangle contact requires two distinct shape ids"
            )
        invalid_jaw_shape_ids = sorted(
            set(jaw_shape_ids).difference(unique_shape_ids)
        )
        if invalid_jaw_shape_ids:
            raise ValueError(
                "Jaw contact shape ids must be tool shapes: "
                f"{invalid_jaw_shape_ids}"
            )
        top_faces = (
            np.empty((0, 3), dtype=np.int32)
            if top_skin_faces is None
            else np.asarray(top_skin_faces, dtype=np.int32)
        )
        if top_faces.size and (
            top_faces.ndim != 2
            or top_faces.shape[1] != 3
            or top_faces.min() < 0
            or top_faces.max() >= model.particle_count
        ):
            raise ValueError(
                "Oriented top-skin faces must be valid Mx3 particle indices"
            )

        geometry_types = model.shape_geo.type.numpy()
        sample_points: list[np.ndarray] = []
        sample_shape_centers: list[np.ndarray] = []
        sample_shape_max_z: list[np.ndarray] = []
        sample_shape_ids: list[np.ndarray] = []
        samples_per_shape: dict[int, int] = {}
        for shape_id in unique_shape_ids:
            if int(geometry_types[shape_id]) != int(wp.sim.GEO_MESH):
                raise ValueError(
                    "Triangle-skin tool proxy currently requires mesh shapes; "
                    f"shape {shape_id} has type {geometry_types[shape_id]}"
                )
            source = model.shape_geo_src[shape_id]
            vertices = np.asarray(source.vertices, dtype=np.float64)
            triangles = np.asarray(source.indices, dtype=np.int32).reshape(-1, 3)
            points = sample_triangle_mesh_surface(
                vertices, triangles, sample_spacing_m
            )
            sample_points.append(points)
            shape_center = 0.5 * (
                vertices.min(axis=0) + vertices.max(axis=0)
            )
            # Store the centre of the jaw cross-section at each sample's own
            # longitudinal z.  The runtime tissue-facing test must classify
            # upper/lower jaw surfaces, not accidentally split the 12 mm jaw
            # into proximal/distal halves when the tool is tilted.
            cross_section_centers = np.repeat(
                shape_center[None, :], len(points), axis=0
            )
            cross_section_centers[:, 2] = points[:, 2]
            sample_shape_centers.append(cross_section_centers)
            sample_shape_max_z.append(
                np.full(len(points), vertices[:, 2].max(), dtype=np.float64)
            )
            sample_shape_ids.append(
                np.full(len(points), shape_id, dtype=np.int32)
            )
            samples_per_shape[shape_id] = len(points)

        all_sample_points = np.concatenate(sample_points, axis=0)
        all_sample_shape_centers = np.concatenate(
            sample_shape_centers, axis=0
        )
        all_sample_shape_max_z = np.concatenate(sample_shape_max_z, axis=0)
        all_sample_shape_ids = np.concatenate(sample_shape_ids, axis=0)
        device = model.device
        skin_nodes, compact_face_indices = np.unique(
            faces.reshape(-1), return_inverse=True
        )
        compact_faces = compact_face_indices.reshape(-1, 3).astype(
            np.int32
        )
        model_tets = model.tet_indices.numpy().astype(np.int32, copy=False)
        skin_node_mask = np.zeros(model.particle_count, dtype=bool)
        skin_node_mask[skin_nodes] = True
        spread_node_mask = skin_node_mask.copy()
        spread_tet_mask = skin_node_mask[model_tets].any(axis=1)
        for _ in range(spread_layers):
            spread_tet_mask = spread_node_mask[model_tets].any(axis=1)
            spread_node_mask[model_tets[spread_tet_mask].reshape(-1)] = True
        contact_tet_ids = np.flatnonzero(
            spread_node_mask[model_tets].any(axis=1)
        ).astype(np.int32)
        spread_tet_ids = np.flatnonzero(spread_tet_mask).astype(np.int32)
        grip_transfer_node_mask = skin_node_mask.copy()
        grip_transfer_tet_mask = np.zeros(len(model_tets), dtype=bool)
        for _ in range(persistent_grip_transfer_layers):
            grip_layer_tet_mask = grip_transfer_node_mask[model_tets].any(
                axis=1
            )
            grip_transfer_tet_mask |= grip_layer_tet_mask
            grip_transfer_node_mask[
                model_tets[grip_layer_tet_mask].reshape(-1)
            ] = True
        grip_transfer_tet_ids = np.flatnonzero(
            grip_transfer_tet_mask
        ).astype(np.int32)
        grip_safety_tet_mask = grip_transfer_node_mask[model_tets].any(
            axis=1
        )
        grip_safety_tet_ids = np.flatnonzero(
            grip_safety_tet_mask
        ).astype(np.int32)
        if len(contact_tet_ids) == 0:
            raise ValueError(
                "Collision skin has no incident tetrahedra"
            )
        self.skin_faces = wp.array(
            faces, dtype=int, ndim=2, device=device
        )
        top_face_keys = {
            tuple(int(particle_id) for particle_id in face)
            for face in top_faces
        }
        self.skin_face_is_top = wp.array(
            np.asarray(
                [
                    int(
                        tuple(int(particle_id) for particle_id in face)
                        in top_face_keys
                    )
                    for face in faces
                ],
                dtype=np.int32,
            ),
            dtype=int,
            device=device,
        )
        self.skin_nodes = wp.array(
            skin_nodes.astype(np.int32), dtype=int, device=device
        )
        self.contact_tet_ids = wp.array(
            contact_tet_ids, dtype=int, device=device
        )
        self.spread_tet_ids = wp.array(
            spread_tet_ids, dtype=int, device=device
        )
        self.persistent_grip_transfer_tet_ids = wp.array(
            grip_transfer_tet_ids, dtype=int, device=device
        )
        self.persistent_grip_safety_tet_ids = wp.array(
            grip_safety_tet_ids, dtype=int, device=device
        )
        self.skin_points = wp.empty(
            len(skin_nodes), dtype=wp.vec3, device=device
        )
        wp.launch(
            kernel=gather_triangle_skin_points,
            dim=len(skin_nodes),
            inputs=[model.particle_q, self.skin_nodes],
            outputs=[self.skin_points],
            device=device,
        )
        self.skin_mesh = wp.Mesh(
            points=self.skin_points,
            indices=wp.array(
                compact_faces.reshape(-1), dtype=int, device=device
            ),
        )
        self.top_faces = None
        self.top_nodes = None
        self.top_points = None
        self.top_mesh = None
        self.top_support_lateral_radius_m = float(
            top_support_lateral_radius_m
        )
        self.top_support_depth_m = float(top_support_depth_m)
        self.top_support_weight_scale = float(top_support_weight_scale)
        self.top_pressure_shoulder_lateral_radius_m = float(
            top_pressure_shoulder_lateral_radius_m
        )
        self.top_pressure_shoulder_depth_m = float(
            top_pressure_shoulder_depth_m
        )
        self.top_pressure_shoulder_upward_scale = float(
            top_pressure_shoulder_upward_scale
        )
        self.top_pressure_shoulder_outward_scale = float(
            top_pressure_shoulder_outward_scale
        )
        self.top_pressure_shoulder_bias_direction_world = tuple(
            float(value) for value in shoulder_bias_direction
        )
        self.top_pressure_shoulder_bias_start_m = float(
            top_pressure_shoulder_bias_start_m
        )
        self.top_barrier_lateral_tolerance_m = float(
            top_barrier_lateral_tolerance_m
        )
        self.top_barrier_contact_patch_radius_m = float(
            top_barrier_contact_patch_radius_m
        )
        self.top_barrier_clearance_m = float(top_barrier_clearance_m)
        self.top_support_entry_count = 0
        self.top_pressure_shoulder_entry_count = 0
        top_nodes_np = np.empty(0, dtype=np.int32)
        if len(top_faces):
            rest_positions = model.particle_q.numpy().astype(
                np.float64, copy=False
            )
            shoulder_offsets = [0]
            shoulder_nodes: list[int] = []
            shoulder_weights: list[float] = []
            shoulder_enabled = bool(
                self.top_pressure_shoulder_lateral_radius_m > 0.0
                and self.top_pressure_shoulder_depth_m > 0.0
                and (
                    self.top_pressure_shoulder_upward_scale > 0.0
                    or self.top_pressure_shoulder_outward_scale > 0.0
                )
            )
            shoulder_radius = max(
                self.top_pressure_shoulder_lateral_radius_m, 1.0e-12
            )
            shoulder_sigma = 0.65 * shoulder_radius
            shoulder_top_nodes = np.unique(top_faces.reshape(-1)).astype(
                np.int32, copy=False
            )
            shoulder_tree = cKDTree(rest_positions[shoulder_top_nodes])
            for face in top_faces:
                face_points = rest_positions[face]
                face_center = face_points.mean(axis=0)
                face_normal = np.cross(
                    face_points[1] - face_points[0],
                    face_points[2] - face_points[0],
                )
                normal_length = np.linalg.norm(face_normal)
                if not shoulder_enabled or normal_length <= 1.0e-12:
                    shoulder_offsets.append(len(shoulder_nodes))
                    continue
                face_normal /= normal_length
                compact_candidates = shoulder_tree.query_ball_point(
                    face_center, shoulder_radius
                )
                candidate_ids = shoulder_top_nodes[
                    np.asarray(compact_candidates, dtype=np.int32)
                ]
                candidate_ids = candidate_ids[
                    ~np.isin(candidate_ids, face)
                ]
                if len(candidate_ids) == 0:
                    shoulder_offsets.append(len(shoulder_nodes))
                    continue
                difference = rest_positions[candidate_ids] - face_center
                normal_depth = difference @ face_normal
                tangent = (
                    difference
                    - normal_depth[:, None] * face_normal[None, :]
                )
                lateral_distance = np.linalg.norm(tangent, axis=1)
                selected = (
                    (np.abs(normal_depth) <= self.top_pressure_shoulder_depth_m)
                    & (lateral_distance <= shoulder_radius)
                    & (lateral_distance > 1.0e-8)
                )
                selected_ids = candidate_ids[selected]
                selected_lateral = lateral_distance[selected]
                weights = np.exp(
                    -0.5 * (selected_lateral / shoulder_sigma) ** 2
                )
                shoulder_nodes.extend(selected_ids.tolist())
                shoulder_weights.extend(weights.tolist())
                shoulder_offsets.append(len(shoulder_nodes))
            support_enabled = bool(
                self.top_support_lateral_radius_m > 0.0
                and self.top_support_depth_m > 0.0
                and self.top_support_weight_scale > 0.0
            )
            lateral_radius = max(
                self.top_support_lateral_radius_m, 1.0e-12
            )
            depth_limit = max(self.top_support_depth_m, 1.0e-12)
            lateral_sigma = 0.75 * lateral_radius

            # The old support was a radius/depth cylinder containing interior
            # tetrahedral nodes.  Every one of those nodes received the same
            # downward-signed correction, so a jaw press manufactured a bowl
            # instead of letting volume preservation create a lateral bulge.
            # Keep only the third vertex of edge-adjacent top triangles.  A
            # manifold face therefore has at most three weak support nodes;
            # its own three vertices still receive the full barrier motion.
            edge_faces: dict[tuple[int, int], list[int]] = {}
            for face_id, face in enumerate(top_faces):
                for edge_a, edge_b in (
                    (int(face[0]), int(face[1])),
                    (int(face[1]), int(face[2])),
                    (int(face[2]), int(face[0])),
                ):
                    edge = (
                        min(edge_a, edge_b),
                        max(edge_a, edge_b),
                    )
                    edge_faces.setdefault(edge, []).append(face_id)
            face_neighbors: list[set[int]] = [
                set() for _ in range(len(top_faces))
            ]
            for incident_faces in edge_faces.values():
                for face_id in incident_faces:
                    face_neighbors[face_id].update(incident_faces)
                    face_neighbors[face_id].discard(face_id)
            support_offsets = [0]
            support_nodes: list[int] = []
            support_weights: list[float] = []
            for face_id, face in enumerate(top_faces):
                face_points = rest_positions[face]
                face_center = face_points.mean(axis=0)
                face_normal = np.cross(
                    face_points[1] - face_points[0],
                    face_points[2] - face_points[0],
                )
                face_normal /= np.linalg.norm(face_normal)
                neighbor_face_ids = sorted(face_neighbors[face_id])
                if neighbor_face_ids:
                    candidate_ids = np.unique(
                        top_faces[neighbor_face_ids].reshape(-1)
                    ).astype(np.int32, copy=False)
                    candidate_ids = candidate_ids[
                        ~np.isin(candidate_ids, face)
                    ]
                else:
                    candidate_ids = np.empty(0, dtype=np.int32)
                if not support_enabled or len(candidate_ids) == 0:
                    support_offsets.append(len(support_nodes))
                    continue
                difference = (
                    face_center[None, :]
                    - rest_positions[candidate_ids]
                )
                inward_depth = difference @ face_normal
                tangent = (
                    difference
                    - inward_depth[:, None] * face_normal[None, :]
                )
                lateral_distance = np.linalg.norm(tangent, axis=1)
                selected = (
                    (np.abs(inward_depth) <= depth_limit)
                    & (lateral_distance <= lateral_radius)
                )
                selected_ids = candidate_ids[selected]
                selected_lateral = lateral_distance[selected]
                weights = self.top_support_weight_scale * np.exp(
                    -0.5 * (selected_lateral / lateral_sigma) ** 2
                )
                support_nodes.extend(selected_ids.tolist())
                support_weights.extend(weights.tolist())
                support_offsets.append(len(support_nodes))
            top_nodes, compact_top_face_indices = np.unique(
                top_faces.reshape(-1), return_inverse=True
            )
            top_nodes_np = top_nodes.astype(np.int32, copy=False)
            compact_top_faces = compact_top_face_indices.reshape(-1, 3).astype(
                np.int32
            )
            self.top_faces = wp.array(
                top_faces, dtype=int, ndim=2, device=device
            )
            self.top_support_offsets = wp.array(
                np.asarray(support_offsets, dtype=np.int32),
                dtype=int,
                device=device,
            )
            self.top_support_nodes = wp.array(
                np.asarray(support_nodes, dtype=np.int32),
                dtype=int,
                device=device,
            )
            self.top_support_weights = wp.array(
                np.asarray(support_weights, dtype=np.float32),
                dtype=float,
                device=device,
            )
            self.top_support_entry_count = len(support_nodes)
            self.top_pressure_shoulder_offsets = wp.array(
                np.asarray(shoulder_offsets, dtype=np.int32),
                dtype=int,
                device=device,
            )
            self.top_pressure_shoulder_nodes = wp.array(
                np.asarray(shoulder_nodes, dtype=np.int32),
                dtype=int,
                device=device,
            )
            self.top_pressure_shoulder_weights = wp.array(
                np.asarray(shoulder_weights, dtype=np.float32),
                dtype=float,
                device=device,
            )
            self.top_pressure_shoulder_entry_count = len(shoulder_nodes)
            shoulder_counts = np.diff(
                np.asarray(shoulder_offsets, dtype=np.int32)
            )
            self.top_pressure_shoulder_max_nodes_per_face = int(
                shoulder_counts.max(initial=0)
            )
            self.top_pressure_shoulder_mean_nodes_per_face = float(
                shoulder_counts.mean() if len(shoulder_counts) else 0.0
            )
            support_counts = np.diff(
                np.asarray(support_offsets, dtype=np.int32)
            )
            self.top_support_max_nodes_per_face = int(
                support_counts.max(initial=0)
            )
            self.top_support_mean_nodes_per_face = float(
                support_counts.mean() if len(support_counts) else 0.0
            )
            self.top_nodes = wp.array(
                top_nodes_np, dtype=int, device=device
            )
            self.top_points = wp.empty(
                len(top_nodes), dtype=wp.vec3, device=device
            )
            wp.launch(
                kernel=gather_triangle_skin_points,
                dim=len(top_nodes),
                inputs=[model.particle_q, self.top_nodes],
                outputs=[self.top_points],
                device=device,
            )
            self.top_mesh = wp.Mesh(
                points=self.top_points,
                indices=wp.array(
                    compact_top_faces.reshape(-1),
                    dtype=int,
                    device=device,
                ),
            )
        else:
            self.top_support_max_nodes_per_face = 0
            self.top_support_mean_nodes_per_face = 0.0
            self.top_pressure_shoulder_max_nodes_per_face = 0
            self.top_pressure_shoulder_mean_nodes_per_face = 0.0
        self.tool_sample_local = wp.array(
            all_sample_points, dtype=wp.vec3, device=device
        )
        self.tool_sample_shape_center_local = wp.array(
            all_sample_shape_centers, dtype=wp.vec3, device=device
        )
        self.tool_sample_shape = wp.array(
            all_sample_shape_ids, dtype=int, device=device
        )
        jaw_shape_id_set = set(jaw_shape_ids)
        barrier_shape_id_set = set(barrier_shape_ids)
        if jaw_contact_distal_length_m > 0.0:
            distal_min_z = (
                all_sample_shape_max_z - jaw_contact_distal_length_m
            )
            jaw_distal_mask = all_sample_points[:, 2] >= distal_min_z
        else:
            jaw_distal_mask = np.ones(len(all_sample_points), dtype=bool)
        if effective_top_barrier_distal_length_m > 0.0:
            barrier_distal_min_z = (
                all_sample_shape_max_z
                - effective_top_barrier_distal_length_m
            )
            barrier_band_mask = (
                (all_sample_points[:, 2] >= barrier_distal_min_z)
                & (
                    all_sample_points[:, 2]
                    <= all_sample_shape_max_z
                    - top_barrier_tip_allowance_m
                )
            )
        else:
            barrier_band_mask = jaw_distal_mask
        self.surface_sample_enabled = wp.array(
            np.asarray(
                [
                    int(
                        shape_id not in jaw_shape_id_set
                        and shape_id not in barrier_shape_id_set
                    )
                    for shape_id in all_sample_shape_ids
                ],
                dtype=np.int32,
            ),
            dtype=int,
            device=device,
        )
        self.jaw_surface_sample_enabled = wp.array(
            np.asarray(
                [
                    int(
                        shape_id in jaw_shape_id_set
                        and jaw_distal_mask[sample_id]
                    )
                    for sample_id, shape_id in enumerate(
                        all_sample_shape_ids
                    )
                ],
                dtype=np.int32,
            ),
            dtype=int,
            device=device,
        )
        self.top_barrier_sample_enabled = wp.array(
            (
                np.isin(
                    all_sample_shape_ids,
                    np.asarray(barrier_shape_ids, dtype=np.int32),
                )
                & barrier_band_mask
            ).astype(np.int32),
            dtype=int,
            device=device,
        )
        self.jaw_contribution_count = len(all_sample_points) * 3
        self.jaw_contact_distal_length_m = float(
            jaw_contact_distal_length_m
        )
        self.top_barrier_distal_length_m = float(
            effective_top_barrier_distal_length_m
        )
        self.top_barrier_tip_allowance_m = float(
            top_barrier_tip_allowance_m
        )
        self.jaw_contact_enabled_sample_count = int(
            np.count_nonzero(
                np.isin(
                    all_sample_shape_ids,
                    np.asarray(jaw_shape_ids, dtype=np.int32),
                )
                & jaw_distal_mask
            )
        )
        self.top_barrier_enabled_sample_count = int(
            np.count_nonzero(
                np.isin(
                    all_sample_shape_ids,
                    np.asarray(barrier_shape_ids, dtype=np.int32),
                )
                & barrier_band_mask
            )
        )
        if (
            jaw_shape_ids
            and model.particle_count * self.jaw_contribution_count
            >= np.iinfo(np.int32).max
        ):
            raise ValueError(
                "Jaw contribution sort key exceeds int32 capacity"
            )
        self.jaw_contribution_nodes = wp.empty(
            self.jaw_contribution_count, dtype=int, device=device
        )
        self.jaw_contribution_deltas = wp.empty(
            self.jaw_contribution_count, dtype=wp.vec3, device=device
        )
        self.jaw_sort_keys = wp.empty(
            2 * self.jaw_contribution_count, dtype=int, device=device
        )
        self.jaw_sort_values = wp.empty(
            2 * self.jaw_contribution_count, dtype=int, device=device
        )
        self.deltas = wp.zeros_like(model.particle_q)
        self.delta_counts = wp.zeros(
            model.particle_count, dtype=float, device=device
        )
        self.surface_downward_deltas = wp.zeros(
            model.particle_count, dtype=float, device=device
        )
        self.pressure_shoulder_deltas = wp.zeros_like(model.particle_q)
        self.pressure_shoulder_counts = wp.zeros(
            model.particle_count, dtype=float, device=device
        )
        self.pressure_shoulder_direct_override = wp.zeros(
            model.particle_count, dtype=int, device=device
        )
        self.pressure_shoulder_apply_deltas = wp.zeros_like(
            model.particle_q
        )
        self.pressure_shoulder_apply_weights = wp.zeros(
            model.particle_count, dtype=float, device=device
        )
        self.spread_deltas_a = wp.zeros_like(model.particle_q)
        self.spread_deltas_b = wp.zeros_like(model.particle_q)
        self.spread_weights_a = wp.zeros(
            model.particle_count, dtype=float, device=device
        )
        self.spread_weights_b = wp.zeros(
            model.particle_count, dtype=float, device=device
        )
        self.global_scale = wp.ones(1, dtype=float, device=device)
        self.contact_pre_projection_positions = wp.zeros_like(
            model.particle_q
        )
        self.contact_unsafe_particles = wp.zeros(
            model.particle_count, dtype=int, device=device
        )
        self.contact_unsafe_tet_count = wp.zeros(
            1, dtype=int, device=device
        )
        self.contact_count = wp.zeros(1, dtype=int, device=device)
        self.contact_count_by_shape = wp.zeros(
            model.shape_count, dtype=int, device=device
        )
        self.top_barrier_contact_count = wp.zeros(
            1, dtype=int, device=device
        )
        self.jaw_triangle_contact_count = wp.zeros(
            1, dtype=int, device=device
        )
        self.jaw_closing_contact_count_by_shape = wp.zeros(
            model.shape_count, dtype=int, device=device
        )
        self.jaw_closing_resistance_by_shape = wp.zeros(
            model.shape_count, dtype=float, device=device
        )
        self.maximum_penetration = wp.zeros(
            1, dtype=float, device=device
        )
        self.minimum_signed_distance = wp.zeros(
            1, dtype=float, device=device
        )
        self.top_minimum_signed_distance = wp.zeros(
            1, dtype=float, device=device
        )
        self.jaw_contact_shape_ids = tuple(jaw_shape_ids)
        self.jaw_friction_coefficient = float(
            jaw_friction_coefficient
        )
        self.persistent_grip_enabled = bool(persistent_grip_enabled)
        self.persistent_grip_minimum_contact_samples_per_jaw = int(
            persistent_grip_minimum_contact_samples_per_jaw
        )
        self.persistent_grip_nearest_surface_particles = int(
            persistent_grip_nearest_surface_particles
        )
        self.persistent_grip_maximum_jaw_patch_separation_m = float(
            persistent_grip_maximum_jaw_patch_separation_m
        )
        self.persistent_grip_activation_steps = int(
            persistent_grip_activation_steps
        )
        self.persistent_grip_maximum_capture_penetration_m = float(
            persistent_grip_maximum_capture_penetration_m
        )
        self.persistent_grip_minimum_capture_volume_ratio = float(
            persistent_grip_minimum_capture_volume_ratio
        )
        self.persistent_grip_closed_angle_max_rad = float(
            persistent_grip_closed_angle_max_rad
        )
        self.persistent_grip_release_angle_min_rad = float(
            persistent_grip_release_angle_min_rad
        )
        self.persistent_grip_release_angle_delta_rad = float(
            persistent_grip_release_angle_delta_rad
        )
        self.persistent_grip_wide_open_angle_rad = float(
            persistent_grip_wide_open_angle_rad
        )
        self.persistent_grip_angle_motion_epsilon_rad = float(
            persistent_grip_angle_motion_epsilon_rad
        )
        self.persistent_grip_compliance_m_per_n = float(
            persistent_grip_compliance_m_per_n
        )
        self.persistent_grip_relaxation = float(
            persistent_grip_relaxation
        )
        self.persistent_grip_maximum_correction_m = float(
            persistent_grip_maximum_correction_m
        )
        self.persistent_grip_transfer_layers = int(
            persistent_grip_transfer_layers
        )
        self.persistent_grip_minimum_volume_ratio = float(
            persistent_grip_minimum_volume_ratio
        )
        self.persistent_grip_transfer_tet_count = len(
            grip_transfer_tet_ids
        )
        self.persistent_grip_safety_tet_count = len(
            grip_safety_tet_ids
        )
        self.persistent_grip_support_radius_m = float(
            persistent_grip_support_radius_m
        )
        self.persistent_grip_support_generations = int(
            persistent_grip_support_generations
        )
        # [active, activation counter, activated on this solve]
        self.persistent_grip_state = wp.zeros(
            3, dtype=int, device=device
        )
        self.persistent_grip_current_jaw_angle = wp.zeros(
            1, dtype=float, device=device
        )
        self.persistent_grip_current_signal_timestamp = wp.zeros(
            1, dtype=float, device=device
        )
        self.persistent_grip_capture_allowed = wp.zeros(
            1, dtype=int, device=device
        )
        self.persistent_grip_release_requested = wp.ones(
            1, dtype=int, device=device
        )
        self.persistent_grip_capture_jaw_angle = wp.zeros(
            1, dtype=float, device=device
        )
        self.persistent_grip_capture_timestamp = wp.zeros(
            1, dtype=float, device=device
        )
        self.persistent_grip_previous_jaw_angle: float | None = None
        self.persistent_grip_jaw_motion_state = "unknown"
        self.persistent_grip_jaw_a_counts = wp.zeros(
            model.particle_count, dtype=int, device=device
        )
        self.persistent_grip_jaw_b_counts = wp.zeros(
            model.particle_count, dtype=int, device=device
        )
        self.persistent_grip_contact_patch_sums = wp.zeros(
            8, dtype=float, device=device
        )
        self.persistent_grip_grasp_center = wp.zeros(
            1, dtype=wp.vec3, device=device
        )
        self.persistent_grip_jaw_patch_separation = wp.full(
            1, value=1.0e6, dtype=float, device=device
        )
        self.persistent_grip_selected_particle_ids = wp.full(
            4, value=-1, dtype=int, device=device
        )
        self.persistent_grip_selected_particle_count = wp.zeros(
            1, dtype=int, device=device
        )
        self.persistent_grip_between_jaw_candidate_count = wp.zeros(
            1, dtype=int, device=device
        )
        self.persistent_grip_particle_minimum_tet_volume_ratio = wp.full(
            model.particle_count, value=1.0e6, dtype=float, device=device
        )
        shape_body_np = model.shape_body.numpy().astype(
            np.int32, copy=False
        )
        self.persistent_grip_jaw_body_ids = (
            tuple(int(shape_body_np[shape_id]) for shape_id in jaw_shape_ids)
            if jaw_shape_ids
            else ()
        )
        if self.persistent_grip_enabled and (
            len(self.persistent_grip_jaw_body_ids) != 2
            or min(self.persistent_grip_jaw_body_ids) < 0
        ):
            raise ValueError(
                "Persistent grip requires both jaw shapes to belong to rigid bodies"
            )
        self.persistent_grip_virtual_body_id = -1
        if self.persistent_grip_enabled:
            joint_children = model.joint_child.numpy().astype(
                np.int32, copy=False
            )
            joint_parents = model.joint_parent.numpy().astype(
                np.int32, copy=False
            )
            jaw_parent_ids: list[int] = []
            for jaw_body_id in self.persistent_grip_jaw_body_ids:
                matches = np.flatnonzero(joint_children == jaw_body_id)
                if len(matches) != 1:
                    raise ValueError(
                        "Each persistent-grip jaw must have one articulation parent"
                    )
                jaw_parent_ids.append(int(joint_parents[matches[0]]))
            if jaw_parent_ids[0] != jaw_parent_ids[1] or jaw_parent_ids[0] < 0:
                raise ValueError(
                    "Persistent-grip jaws must share one central parent body"
                )
            # The common jaw-hinge parent is the physical, kinematically valid
            # virtual grasp frame.  It follows the wrist but is independent of
            # q7 opening/closing, unlike averaging the two moving jaw rotations.
            self.persistent_grip_virtual_body_id = jaw_parent_ids[0]
        self.persistent_grip_particle_body = wp.full(
            model.particle_count, value=-1, dtype=int, device=device
        )
        self.persistent_grip_particle_local = wp.zeros(
            model.particle_count, dtype=wp.vec3, device=device
        )
        self.persistent_grip_particle_direct = wp.zeros(
            model.particle_count, dtype=int, device=device
        )
        self.persistent_grip_particle_weight = wp.zeros(
            model.particle_count, dtype=float, device=device
        )
        self.persistent_grip_particle_level = wp.full(
            model.particle_count, value=-1, dtype=int, device=device
        )
        # Four direct anchor corrections are transferred through one incident
        # tetrahedron layer.  These support nodes are not extra grasp anchors;
        # they receive the local rigid component of the four-anchor motion so
        # a light surface vertex cannot collapse its incident tetrahedra alone.
        self.persistent_grip_deltas = wp.zeros_like(model.particle_q)
        self.persistent_grip_delta_counts = wp.zeros(
            model.particle_count, dtype=float, device=device
        )
        self.persistent_grip_jaw_direct_deltas = wp.zeros_like(
            model.particle_q
        )
        self.persistent_grip_jaw_direct_weights = wp.zeros(
            model.particle_count, dtype=float, device=device
        )
        self.persistent_grip_spread_deltas_a = wp.zeros_like(
            model.particle_q
        )
        self.persistent_grip_spread_weights_a = wp.zeros(
            model.particle_count, dtype=float, device=device
        )
        self.persistent_grip_spread_deltas_b = wp.zeros_like(
            model.particle_q
        )
        self.persistent_grip_spread_weights_b = wp.zeros(
            model.particle_count, dtype=float, device=device
        )
        self.persistent_grip_global_scale = wp.ones(
            1, dtype=float, device=device
        )
        self.persistent_grip_direct_scale = wp.ones(
            1, dtype=float, device=device
        )
        self.persistent_grip_jaw_scales = wp.ones(
            2, dtype=float, device=device
        )
        self.persistent_grip_pre_support_positions = wp.zeros_like(
            model.particle_q
        )
        self.persistent_grip_support_unsafe_particles = wp.zeros(
            model.particle_count, dtype=int, device=device
        )
        self.persistent_grip_support_unsafe_tet_count = wp.zeros(
            1, dtype=int, device=device
        )
        self.persistent_grip_support_generation = wp.zeros(
            1, dtype=int, device=device
        )
        rest_positions = model.particle_q.numpy().astype(
            np.float64, copy=False
        )
        support_offsets = [0]
        support_sources: list[int] = []
        support_weights: list[float] = []
        if (
            self.persistent_grip_enabled
            and self.persistent_grip_support_radius_m > 0.0
        ):
            support_sigma = 0.5 * self.persistent_grip_support_radius_m
            # Build one geodesic surface ring from collision-skin edges. A
            # pure Euclidean radius can jump across a thin sheet and capture
            # the bottom skin; edge adjacency instead follows the same local
            # surface on which the four direct u_t anchors were selected.
            surface_neighbors: list[set[int]] = [
                set() for _ in range(model.particle_count)
            ]
            for face in faces:
                for index_a, index_b in (
                    (int(face[0]), int(face[1])),
                    (int(face[1]), int(face[2])),
                    (int(face[2]), int(face[0])),
                ):
                    distance = np.linalg.norm(
                        rest_positions[index_b] - rest_positions[index_a]
                    )
                    if distance <= self.persistent_grip_support_radius_m:
                        surface_neighbors[index_a].add(index_b)
                        surface_neighbors[index_b].add(index_a)
            for particle_id, neighbors in enumerate(surface_neighbors):
                if neighbors:
                    source_ids = np.asarray(
                        sorted(neighbors), dtype=np.int32
                    )
                    distances = np.linalg.norm(
                        rest_positions[source_ids]
                        - rest_positions[particle_id],
                        axis=1,
                    )
                    order = np.argsort(distances, kind="stable")
                    support_sources.extend(source_ids[order].tolist())
                    support_weights.extend(
                        np.exp(
                            -0.5 * (distances[order] / support_sigma) ** 2
                        ).tolist()
                    )
                support_offsets.append(len(support_sources))
        else:
            support_offsets.extend([0] * model.particle_count)
        self.persistent_grip_support_offsets = wp.array(
            np.asarray(support_offsets, dtype=np.int32),
            dtype=int,
            device=device,
        )
        self.persistent_grip_support_sources = wp.array(
            np.asarray(support_sources, dtype=np.int32),
            dtype=int,
            device=device,
        )
        self.persistent_grip_support_weights = wp.array(
            np.asarray(support_weights, dtype=np.float32),
            dtype=float,
            device=device,
        )
        self.persistent_grip_support_candidate_count = len(
            support_sources
        )
        self.tool_shape_ids = tuple(unique_shape_ids)
        self.top_barrier_shape_ids = tuple(barrier_shape_ids)
        self.samples_per_shape = samples_per_shape
        self.sample_count = len(all_sample_points)
        self.sample_spacing_m = float(sample_spacing_m)
        self.skin_node_count = len(skin_nodes)
        self.top_face_count = len(top_faces)
        self.top_node_count = (
            0 if self.top_nodes is None else len(top_nodes)
        )
        self.contact_tet_count = len(contact_tet_ids)
        self.spread_tet_count = len(spread_tet_ids)
        self.spread_layers = int(spread_layers)
        self.last_global_scale = 1.0

    def project(
        self,
        model: warp.sim.Model,
        state: warp.sim.State,
        dt: float,
        *,
        contact_margin_m: float,
        query_distance_m: float,
        ccd_velocity_scale: float,
        friction_coefficient: float,
        relaxation: float,
        maximum_correction_m: float,
        minimum_volume_ratio: float,
        top_barrier_maximum_correction_m: float | None = None,
        enable_top_pressure_shoulder: bool = True,
    ) -> None:
        self.detect(
            model,
            state,
            dt,
            contact_margin_m=contact_margin_m,
            query_distance_m=query_distance_m,
            ccd_velocity_scale=ccd_velocity_scale,
            friction_coefficient=friction_coefficient,
            relaxation=relaxation,
        )
        wp.launch(
            kernel=average_and_clamp_triangle_skin_contact_deltas,
            dim=model.particle_count,
            inputs=[
                self.deltas,
                self.delta_counts,
                maximum_correction_m,
            ],
            device=model.device,
        )
        if self.top_mesh is not None:
            top_barrier_maximum_correction = (
                maximum_correction_m
                if top_barrier_maximum_correction_m is None
                else top_barrier_maximum_correction_m
            )
            wp.launch(
                kernel=merge_oriented_top_surface_constraint_deltas,
                dim=model.particle_count,
                inputs=[
                    self.deltas,
                    self.delta_counts,
                    self.surface_downward_deltas,
                    top_barrier_maximum_correction,
                    maximum_correction_m,
                ],
                device=model.device,
            )
            if enable_top_pressure_shoulder:
                wp.launch(
                    kernel=merge_oriented_top_pressure_shoulder_deltas,
                    dim=model.particle_count,
                    inputs=[
                        self.deltas,
                        self.delta_counts,
                        self.surface_downward_deltas,
                        self.pressure_shoulder_deltas,
                        self.pressure_shoulder_counts,
                        self.pressure_shoulder_direct_override,
                        self.persistent_grip_state,
                        maximum_correction_m,
                    ],
                    device=model.device,
                )
        wp.launch(
            kernel=seed_triangle_skin_contact_spread,
            dim=model.particle_count,
            inputs=[
                self.deltas,
                self.delta_counts,
                self.spread_deltas_a,
                self.spread_weights_a,
            ],
            device=model.device,
        )
        spread_deltas = self.spread_deltas_a
        spread_weights = self.spread_weights_a
        next_deltas = self.spread_deltas_b
        next_weights = self.spread_weights_b
        for _ in range(self.spread_layers):
            next_deltas.zero_()
            next_weights.zero_()
            wp.launch(
                kernel=accumulate_triangle_skin_contact_spread,
                dim=self.spread_tet_count,
                inputs=[
                    model.tet_indices,
                    self.spread_tet_ids,
                    spread_deltas,
                    spread_weights,
                    model.particle_inv_mass,
                ],
                outputs=[next_deltas, next_weights],
                device=model.device,
            )
            wp.launch(
                kernel=normalize_triangle_skin_contact_spread,
                dim=model.particle_count,
                inputs=[
                    self.deltas,
                    self.delta_counts,
                    next_deltas,
                    next_weights,
                ],
                device=model.device,
            )
            spread_deltas, next_deltas = next_deltas, spread_deltas
            spread_weights, next_weights = next_weights, spread_weights
        # Apply the full bounded press proposal, then repair only moved nodes
        # belonging to an unsafe local tetrahedron.  The former single global
        # scale let one thin boundary element suppress the complete indentation
        # patch, which looked like the kinematic jaw passing through a rigid,
        # motionless tissue surface.
        wp.launch(
            kernel=snapshot_persistent_grip_support_positions,
            dim=model.particle_count,
            inputs=[
                state.particle_q,
                self.contact_pre_projection_positions,
            ],
            device=model.device,
        )
        self.global_scale.fill_(1.0)
        # Do not run the old mesh-wide pre-scale here. A single inverted or
        # visually pre-compressed tet touched by this broad safety set could
        # set the scale to zero and disable every otherwise valid jaw contact.
        # Apply the bounded proposal first, then the loop below reverts only
        # moved vertices of tetrahedra whose volume actually gets worse. The
        # persistent-grip anchors retain their separate incident-tet line
        # search; this change is limited to ordinary surface pressing.
        wp.launch(
            kernel=apply_triangle_skin_contact_deltas,
            dim=model.particle_count,
            inputs=[
                state.particle_q,
                model.particle_inv_mass,
                model.particle_flags,
                spread_deltas,
                spread_weights,
                self.global_scale,
            ],
            device=model.device,
        )
        self.contact_unsafe_tet_count.zero_()
        if minimum_volume_ratio > 0.0:
            for _ in range(8):
                self.contact_unsafe_particles.zero_()
                self.contact_unsafe_tet_count.zero_()
                wp.launch(
                    kernel=mark_unsafe_persistent_grip_support_particles,
                    dim=self.contact_tet_count,
                    inputs=[
                        self.contact_pre_projection_positions,
                        state.particle_q,
                        model.tet_indices,
                        model.tet_poses,
                        self.contact_tet_ids,
                        spread_weights,
                        minimum_volume_ratio,
                    ],
                    outputs=[
                        self.contact_unsafe_particles,
                        self.contact_unsafe_tet_count,
                    ],
                    device=model.device,
                )
                wp.launch(
                    kernel=revert_unsafe_persistent_grip_support_particles,
                    dim=model.particle_count,
                    inputs=[
                        self.contact_pre_projection_positions,
                        state.particle_q,
                        spread_weights,
                        self.contact_unsafe_particles,
                    ],
                    device=model.device,
                )

    def apply_top_pressure_shoulder(
        self,
        model: warp.sim.Model,
        state: warp.sim.State,
        *,
        maximum_correction_m: float,
        minimum_volume_ratio: float,
    ) -> None:
        """Apply one raised outer-ring pass after material recovery.

        The contact detector has already accumulated the pressure source and
        direct footprint.  Delaying this outer-ring correction until after the
        distance/volume/shape iterations prevents those iterations from
        folding the requested bulge back into the grasp core.  The same local
        no-flip repair used by ordinary contact remains active.
        """
        if self.top_mesh is None:
            return
        self.pressure_shoulder_apply_deltas.zero_()
        self.pressure_shoulder_apply_weights.zero_()
        wp.launch(
            kernel=prepare_oriented_top_pressure_shoulder_deltas,
            dim=model.particle_count,
            inputs=[
                self.delta_counts,
                self.surface_downward_deltas,
                self.pressure_shoulder_deltas,
                self.pressure_shoulder_counts,
                self.pressure_shoulder_direct_override,
                self.persistent_grip_state,
                maximum_correction_m,
            ],
            outputs=[
                self.pressure_shoulder_apply_deltas,
                self.pressure_shoulder_apply_weights,
            ],
            device=model.device,
        )
        wp.launch(
            kernel=snapshot_persistent_grip_support_positions,
            dim=model.particle_count,
            inputs=[
                state.particle_q,
                self.contact_pre_projection_positions,
            ],
            device=model.device,
        )
        self.global_scale.fill_(1.0)
        wp.launch(
            kernel=apply_triangle_skin_contact_deltas,
            dim=model.particle_count,
            inputs=[
                state.particle_q,
                model.particle_inv_mass,
                model.particle_flags,
                self.pressure_shoulder_apply_deltas,
                self.pressure_shoulder_apply_weights,
                self.global_scale,
            ],
            device=model.device,
        )
        self.contact_unsafe_tet_count.zero_()
        if minimum_volume_ratio > 0.0:
            for _ in range(8):
                self.contact_unsafe_particles.zero_()
                self.contact_unsafe_tet_count.zero_()
                wp.launch(
                    kernel=mark_unsafe_persistent_grip_support_particles,
                    dim=self.contact_tet_count,
                    inputs=[
                        self.contact_pre_projection_positions,
                        state.particle_q,
                        model.tet_indices,
                        model.tet_poses,
                        self.contact_tet_ids,
                        self.pressure_shoulder_apply_weights,
                        minimum_volume_ratio,
                    ],
                    outputs=[
                        self.contact_unsafe_particles,
                        self.contact_unsafe_tet_count,
                    ],
                    device=model.device,
                )
                wp.launch(
                    kernel=revert_unsafe_persistent_grip_support_particles,
                    dim=model.particle_count,
                    inputs=[
                        self.contact_pre_projection_positions,
                        state.particle_q,
                        self.pressure_shoulder_apply_weights,
                        self.contact_unsafe_particles,
                    ],
                    device=model.device,
                )

    def apply_persistent_grip(
        self,
        model: warp.sim.Model,
        state: warp.sim.State,
        dt: float,
    ) -> None:
        """Project the active four-anchor grasp without rerunning contact."""
        if not self.persistent_grip_enabled:
            return
        self.persistent_grip_deltas.zero_()
        self.persistent_grip_delta_counts.zero_()
        wp.launch(
            kernel=accumulate_persistent_grip_constraint_deltas,
            dim=model.particle_count,
            inputs=[
                state.particle_q,
                model.particle_inv_mass,
                model.particle_flags,
                state.body_q,
                self.persistent_grip_state,
                self.persistent_grip_particle_body,
                self.persistent_grip_particle_local,
                self.persistent_grip_particle_weight,
                dt,
                self.persistent_grip_compliance_m_per_n,
                self.persistent_grip_relaxation,
                self.persistent_grip_maximum_correction_m,
            ],
            outputs=[
                self.persistent_grip_deltas,
                self.persistent_grip_delta_counts,
            ],
            device=model.device,
        )
        # Each jaw first transfers the local rigid component of its two u_t
        # controls into one finite-volume incident-tet patch. The propagated
        # nodes are not extra anchors: they only precondition the volume before
        # the two real surface controls are projected. Jaw-local construction
        # avoids averaging the opposite q7 motions into one cancelling field.
        self.persistent_grip_jaw_scales.fill_(1.0)
        self.persistent_grip_global_scale.fill_(1.0)
        for jaw_index, jaw_body_id in enumerate(
            self.persistent_grip_jaw_body_ids
        ):
            self.persistent_grip_jaw_direct_deltas.zero_()
            self.persistent_grip_jaw_direct_weights.zero_()
            wp.launch(
                kernel=filter_persistent_grip_jaw_deltas,
                dim=model.particle_count,
                inputs=[
                    self.persistent_grip_deltas,
                    self.persistent_grip_delta_counts,
                    self.persistent_grip_particle_body,
                    jaw_body_id,
                ],
                outputs=[
                    self.persistent_grip_jaw_direct_deltas,
                    self.persistent_grip_jaw_direct_weights,
                ],
                device=model.device,
            )
            input_deltas = self.persistent_grip_jaw_direct_deltas
            input_weights = self.persistent_grip_jaw_direct_weights
            output_deltas = self.persistent_grip_spread_deltas_a
            output_weights = self.persistent_grip_spread_weights_a
            alternate_deltas = self.persistent_grip_spread_deltas_b
            alternate_weights = self.persistent_grip_spread_weights_b
            for _ in range(self.persistent_grip_transfer_layers):
                output_deltas.zero_()
                output_weights.zero_()
                wp.launch(
                    kernel=accumulate_triangle_skin_contact_spread,
                    dim=self.persistent_grip_transfer_tet_count,
                    inputs=[
                        model.tet_indices,
                        self.persistent_grip_transfer_tet_ids,
                        input_deltas,
                        input_weights,
                        model.particle_inv_mass,
                    ],
                    outputs=[output_deltas, output_weights],
                    device=model.device,
                )
                wp.launch(
                    kernel=normalize_triangle_skin_contact_spread,
                    dim=model.particle_count,
                    inputs=[
                        self.persistent_grip_jaw_direct_deltas,
                        self.persistent_grip_jaw_direct_weights,
                        output_deltas,
                        output_weights,
                    ],
                    device=model.device,
                )
                input_deltas = output_deltas
                input_weights = output_weights
                output_deltas, alternate_deltas = (
                    alternate_deltas,
                    output_deltas,
                )
                output_weights, alternate_weights = (
                    alternate_weights,
                    output_weights,
                )
            if self.persistent_grip_transfer_layers > 0:
                # No direct anchor, including one belonging to the other jaw,
                # may be displaced by the finite-volume support pass.
                wp.launch(
                    kernel=remove_persistent_grip_direct_from_support,
                    dim=model.particle_count,
                    inputs=[
                        self.persistent_grip_delta_counts,
                        input_deltas,
                        input_weights,
                    ],
                    device=model.device,
                )
                wp.launch(
                    kernel=snapshot_persistent_grip_support_positions,
                    dim=model.particle_count,
                    inputs=[
                        state.particle_q,
                        self.persistent_grip_pre_support_positions,
                    ],
                    device=model.device,
                )
                wp.launch(
                    kernel=apply_triangle_skin_contact_deltas,
                    dim=model.particle_count,
                    inputs=[
                        state.particle_q,
                        model.particle_inv_mass,
                        model.particle_flags,
                        input_deltas,
                        input_weights,
                        self.persistent_grip_global_scale,
                    ],
                    device=model.device,
                )
                # Locally undo only support nodes that would worsen a tet below
                # the same J floor used by direct u_t and surface contact.
                for _ in range(8):
                    self.persistent_grip_support_unsafe_particles.zero_()
                    self.persistent_grip_support_unsafe_tet_count.zero_()
                    wp.launch(
                        kernel=mark_unsafe_persistent_grip_support_particles,
                        dim=self.persistent_grip_safety_tet_count,
                        inputs=[
                            self.persistent_grip_pre_support_positions,
                            state.particle_q,
                            model.tet_indices,
                            model.tet_poses,
                            self.persistent_grip_safety_tet_ids,
                            input_weights,
                            self.persistent_grip_minimum_volume_ratio,
                        ],
                        outputs=[
                            self.persistent_grip_support_unsafe_particles,
                            self.persistent_grip_support_unsafe_tet_count,
                        ],
                        device=model.device,
                    )
                    wp.launch(
                        kernel=revert_unsafe_persistent_grip_support_particles,
                        dim=model.particle_count,
                        inputs=[
                            self.persistent_grip_pre_support_positions,
                            state.particle_q,
                            input_weights,
                            self.persistent_grip_support_unsafe_particles,
                        ],
                        device=model.device,
                    )
            if self.persistent_grip_transfer_tet_count > 0:
                wp.launch(
                    kernel=find_safe_persistent_grip_jaw_scale,
                    dim=self.persistent_grip_transfer_tet_count,
                    inputs=[
                        state.particle_q,
                        self.persistent_grip_deltas,
                        self.persistent_grip_particle_body,
                        jaw_body_id,
                        model.tet_indices,
                        model.tet_poses,
                        self.persistent_grip_transfer_tet_ids,
                        self.persistent_grip_minimum_volume_ratio,
                        jaw_index,
                        self.persistent_grip_jaw_scales,
                    ],
                    device=model.device,
                )
            wp.launch(
                kernel=apply_persistent_grip_jaw_deltas,
                dim=model.particle_count,
                inputs=[
                    state.particle_q,
                    model.particle_inv_mass,
                    model.particle_flags,
                    self.persistent_grip_deltas,
                    self.persistent_grip_delta_counts,
                    self.persistent_grip_particle_body,
                    jaw_body_id,
                    jaw_index,
                    self.persistent_grip_jaw_scales,
                ],
                device=model.device,
            )
        # The optional outside surface ring follows only the common jaw-hinge
        # frame. It is excluded from both per-jaw finite-volume source fields
        # above, so continued q7 closure cannot stretch the ring apart. Apply
        # its compliant target once after the two real jaw controls, with a
        # fresh local volume line search.
        if (
            self.persistent_grip_support_radius_m > 0.0
            and self.persistent_grip_virtual_body_id >= 0
            and self.persistent_grip_transfer_tet_count > 0
        ):
            self.persistent_grip_global_scale.fill_(1.0)
            wp.launch(
                kernel=find_safe_persistent_grip_jaw_scale,
                dim=self.persistent_grip_transfer_tet_count,
                inputs=[
                    state.particle_q,
                    self.persistent_grip_deltas,
                    self.persistent_grip_particle_body,
                    self.persistent_grip_virtual_body_id,
                    model.tet_indices,
                    model.tet_poses,
                    self.persistent_grip_transfer_tet_ids,
                    self.persistent_grip_minimum_volume_ratio,
                    0,
                    self.persistent_grip_global_scale,
                ],
                device=model.device,
            )
            wp.launch(
                kernel=apply_persistent_grip_jaw_deltas,
                dim=model.particle_count,
                inputs=[
                    state.particle_q,
                    model.particle_inv_mass,
                    model.particle_flags,
                    self.persistent_grip_deltas,
                    self.persistent_grip_delta_counts,
                    self.persistent_grip_particle_body,
                    self.persistent_grip_virtual_body_id,
                    0,
                    self.persistent_grip_global_scale,
                ],
                device=model.device,
            )
        wp.launch(
            kernel=reduce_persistent_grip_jaw_scales,
            dim=1,
            inputs=[self.persistent_grip_jaw_scales],
            outputs=[self.persistent_grip_direct_scale],
            device=model.device,
        )

    def detect(
        self,
        model: warp.sim.Model,
        state: warp.sim.State,
        dt: float,
        *,
        contact_margin_m: float,
        query_distance_m: float,
        ccd_velocity_scale: float,
        friction_coefficient: float,
        relaxation: float,
    ) -> None:
        """Generate contact corrections and metrics without moving the skin."""
        wp.launch(
            kernel=gather_triangle_skin_points,
            dim=self.skin_node_count,
            inputs=[state.particle_q, self.skin_nodes],
            outputs=[self.skin_points],
            device=model.device,
        )
        self.skin_mesh.refit()
        self.deltas.zero_()
        self.delta_counts.zero_()
        self.surface_downward_deltas.zero_()
        self.pressure_shoulder_deltas.zero_()
        self.pressure_shoulder_counts.zero_()
        self.pressure_shoulder_direct_override.zero_()
        self.contact_count.zero_()
        self.contact_count_by_shape.zero_()
        self.top_barrier_contact_count.zero_()
        self.jaw_triangle_contact_count.zero_()
        self.jaw_closing_contact_count_by_shape.zero_()
        self.jaw_closing_resistance_by_shape.zero_()
        self.maximum_penetration.zero_()
        self.minimum_signed_distance.fill_(query_distance_m)
        self.top_minimum_signed_distance.fill_(query_distance_m)
        wp.launch(
            kernel=accumulate_triangle_skin_contact_deltas,
            dim=self.sample_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                model.particle_inv_mass,
                model.particle_flags,
                self.skin_faces,
                self.skin_mesh.id,
                self.tool_sample_local,
                self.tool_sample_shape,
                self.surface_sample_enabled,
                state.body_q,
                state.body_qd,
                model.body_com,
                model.shape_transform,
                model.shape_body,
                model.shape_geo.scale,
                query_distance_m,
                contact_margin_m,
                ccd_velocity_scale,
                friction_coefficient,
                dt,
                relaxation,
            ],
            outputs=[
                self.deltas,
                self.delta_counts,
                self.contact_count,
                self.contact_count_by_shape,
                self.maximum_penetration,
                self.minimum_signed_distance,
            ],
            device=model.device,
        )
        if self.jaw_contact_shape_ids:
            wp.launch(
                kernel=accumulate_jaw_triangle_skin_contact_deltas,
                dim=self.sample_count,
                inputs=[
                    state.particle_q,
                    state.particle_qd,
                    model.particle_inv_mass,
                    model.particle_flags,
                    self.skin_faces,
                    self.skin_face_is_top,
                    self.skin_mesh.id,
                    self.tool_sample_local,
                    self.tool_sample_shape,
                    self.jaw_surface_sample_enabled,
                    state.body_q,
                    state.body_qd,
                    model.body_com,
                    model.shape_transform,
                    model.shape_body,
                    model.shape_geo.scale,
                    query_distance_m,
                    contact_margin_m,
                    ccd_velocity_scale,
                    self.jaw_friction_coefficient,
                    dt,
                    relaxation,
                    self.jaw_contact_shape_ids[0],
                    self.jaw_contact_shape_ids[1],
                ],
                outputs=[
                    self.jaw_contribution_nodes,
                    self.jaw_contribution_deltas,
                    self.contact_count,
                    self.contact_count_by_shape,
                    self.jaw_triangle_contact_count,
                    self.jaw_closing_contact_count_by_shape,
                    self.jaw_closing_resistance_by_shape,
                    self.maximum_penetration,
                    self.minimum_signed_distance,
                ],
                device=model.device,
            )
            wp.launch(
                kernel=prepare_jaw_contact_contribution_sort,
                dim=self.jaw_contribution_count,
                inputs=[
                    self.jaw_contribution_nodes,
                    self.jaw_contribution_count,
                ],
                outputs=[self.jaw_sort_keys, self.jaw_sort_values],
                device=model.device,
            )
            wp.utils.radix_sort_pairs(
                self.jaw_sort_keys,
                self.jaw_sort_values,
                count=self.jaw_contribution_count,
            )
            wp.launch(
                kernel=reduce_sorted_jaw_contact_contributions,
                dim=model.particle_count,
                inputs=[
                    self.jaw_contribution_deltas,
                    self.jaw_contribution_count,
                    self.jaw_sort_keys,
                    self.jaw_sort_values,
                ],
                outputs=[self.deltas, self.delta_counts],
                device=model.device,
            )
            if self.persistent_grip_enabled:
                self.persistent_grip_jaw_a_counts.zero_()
                self.persistent_grip_jaw_b_counts.zero_()
                self.persistent_grip_contact_patch_sums.zero_()
                self.persistent_grip_between_jaw_candidate_count.zero_()
                jaw_a, jaw_b = self.jaw_contact_shape_ids
                wp.launch(
                    kernel=count_persistent_grip_candidates,
                    dim=self.jaw_contribution_count,
                    inputs=[
                        self.jaw_contribution_nodes,
                        self.tool_sample_shape,
                        jaw_a,
                        jaw_b,
                    ],
                    outputs=[
                        self.persistent_grip_jaw_a_counts,
                        self.persistent_grip_jaw_b_counts,
                    ],
                    device=model.device,
                )
                wp.launch(
                    kernel=accumulate_persistent_grip_contact_patch,
                    dim=model.particle_count,
                    inputs=[
                        state.particle_q,
                        model.particle_inv_mass,
                        model.particle_flags,
                        self.persistent_grip_jaw_a_counts,
                        self.persistent_grip_jaw_b_counts,
                    ],
                    outputs=[
                        self.persistent_grip_contact_patch_sums,
                        self.persistent_grip_between_jaw_candidate_count
                    ],
                    device=model.device,
                )
                self.persistent_grip_particle_minimum_tet_volume_ratio.fill_(
                    1.0e6
                )
                wp.launch(
                    kernel=accumulate_particle_minimum_tet_volume_ratio,
                    dim=model.tet_count,
                    inputs=[
                        state.particle_q,
                        model.tet_indices,
                        model.tet_poses,
                    ],
                    outputs=[
                        self.persistent_grip_particle_minimum_tet_volume_ratio
                    ],
                    device=model.device,
                )
                wp.launch(
                    kernel=select_nearest_persistent_grip_surface_particles,
                    dim=1,
                    inputs=[
                        state.particle_q,
                        model.particle_inv_mass,
                        model.particle_flags,
                        self.skin_nodes,
                        self.skin_node_count,
                        self.persistent_grip_jaw_a_counts,
                        self.persistent_grip_jaw_b_counts,
                        self.persistent_grip_contact_patch_sums,
                        self.persistent_grip_particle_minimum_tet_volume_ratio,
                        self.persistent_grip_minimum_capture_volume_ratio,
                    ],
                    outputs=[
                        self.persistent_grip_grasp_center,
                        self.persistent_grip_jaw_patch_separation,
                        self.persistent_grip_selected_particle_ids,
                        self.persistent_grip_selected_particle_count,
                    ],
                    device=model.device,
                )
                wp.launch(
                    kernel=mask_jaw_contact_deltas_to_bilateral_patch,
                    dim=model.particle_count,
                    inputs=[
                        state.particle_q,
                        self.persistent_grip_contact_patch_sums,
                        self.persistent_grip_grasp_center,
                        self.top_barrier_contact_patch_radius_m,
                        self.deltas,
                        self.delta_counts,
                    ],
                    device=model.device,
                )
                wp.launch(
                    kernel=update_persistent_grip_state,
                    dim=1,
                    inputs=[
                        self.contact_count_by_shape,
                        jaw_a,
                        jaw_b,
                        self.persistent_grip_selected_particle_count,
                        self.persistent_grip_jaw_patch_separation,
                        self.maximum_penetration,
                        self.persistent_grip_minimum_contact_samples_per_jaw,
                        self.persistent_grip_maximum_jaw_patch_separation_m,
                        self.persistent_grip_activation_steps,
                        self.persistent_grip_maximum_capture_penetration_m,
                        self.persistent_grip_release_angle_delta_rad,
                        self.persistent_grip_current_jaw_angle,
                        self.persistent_grip_current_signal_timestamp,
                        self.persistent_grip_capture_allowed,
                        self.persistent_grip_release_requested,
                    ],
                    outputs=[
                        self.persistent_grip_state,
                        self.persistent_grip_capture_jaw_angle,
                        self.persistent_grip_capture_timestamp,
                    ],
                    device=model.device,
                )
                wp.launch(
                    kernel=capture_or_release_persistent_grip_particles,
                    dim=model.particle_count,
                    inputs=[
                        state.particle_q,
                        model.particle_inv_mass,
                        model.particle_flags,
                        state.body_q,
                        self.persistent_grip_jaw_body_ids[0],
                        self.persistent_grip_jaw_body_ids[1],
                        self.persistent_grip_selected_particle_ids,
                        self.persistent_grip_state,
                    ],
                    outputs=[
                        self.persistent_grip_particle_body,
                        self.persistent_grip_particle_local,
                        self.persistent_grip_particle_direct,
                        self.persistent_grip_particle_weight,
                        self.persistent_grip_particle_level,
                    ],
                    device=model.device,
                )
                if self.persistent_grip_support_radius_m > 0.0:
                    wp.launch(
                        kernel=advance_persistent_grip_support_generation,
                        dim=1,
                        inputs=[
                            self.persistent_grip_state,
                            self.persistent_grip_support_generations,
                        ],
                        outputs=[self.persistent_grip_support_generation],
                        device=model.device,
                    )
                    wp.launch(
                        kernel=propagate_persistent_grip_support_particles,
                        dim=model.particle_count,
                        inputs=[
                            state.particle_q,
                            model.particle_inv_mass,
                            model.particle_flags,
                            state.body_q,
                            self.persistent_grip_virtual_body_id,
                            self.persistent_grip_state,
                            self.persistent_grip_support_offsets,
                            self.persistent_grip_support_sources,
                            self.persistent_grip_support_weights,
                            self.persistent_grip_particle_direct,
                            self.persistent_grip_support_generation,
                            self.persistent_grip_particle_level,
                        ],
                        outputs=[
                            self.persistent_grip_particle_body,
                            self.persistent_grip_particle_local,
                            self.persistent_grip_particle_weight,
                        ],
                        device=model.device,
                    )
        if self.top_mesh is not None:
            wp.launch(
                kernel=gather_triangle_skin_points,
                dim=self.top_node_count,
                inputs=[state.particle_q, self.top_nodes],
                outputs=[self.top_points],
                device=model.device,
            )
            self.top_mesh.refit()
            wp.launch(
                kernel=accumulate_oriented_top_skin_barrier_deltas,
                dim=self.sample_count,
                inputs=[
                    state.particle_q,
                    state.particle_qd,
                    model.particle_inv_mass,
                    model.particle_flags,
                    self.top_faces,
                    self.top_support_offsets,
                    self.top_support_nodes,
                    self.top_support_weights,
                    self.top_pressure_shoulder_offsets,
                    self.top_pressure_shoulder_nodes,
                    self.top_pressure_shoulder_weights,
                    self.top_mesh.id,
                    self.tool_sample_local,
                    self.tool_sample_shape_center_local,
                    self.tool_sample_shape,
                    self.top_barrier_sample_enabled,
                    state.body_q,
                    state.body_qd,
                    model.body_com,
                    model.shape_transform,
                    model.shape_body,
                    model.shape_geo.scale,
                    query_distance_m,
                    self.top_barrier_lateral_tolerance_m,
                    self.top_barrier_contact_patch_radius_m,
                    self.top_barrier_clearance_m,
                    ccd_velocity_scale,
                    dt,
                    relaxation,
                    self.top_pressure_shoulder_upward_scale,
                    self.top_pressure_shoulder_outward_scale,
                    wp.vec3(
                        *self.top_pressure_shoulder_bias_direction_world
                    ),
                    self.top_pressure_shoulder_bias_start_m,
                    self.persistent_grip_contact_patch_sums,
                    self.persistent_grip_grasp_center,
                ],
                outputs=[
                    self.surface_downward_deltas,
                    self.pressure_shoulder_deltas,
                    self.pressure_shoulder_counts,
                    self.pressure_shoulder_direct_override,
                    self.contact_count,
                    self.contact_count_by_shape,
                    self.top_barrier_contact_count,
                    self.maximum_penetration,
                    self.minimum_signed_distance,
                    self.top_minimum_signed_distance,
                ],
                device=model.device,
            )

    def metrics(self) -> dict:
        counts = self.contact_count_by_shape.numpy()
        closing_counts = self.jaw_closing_contact_count_by_shape.numpy()
        closing_resistance = self.jaw_closing_resistance_by_shape.numpy()
        selected_particle_ids = self.persistent_grip_selected_particle_ids.numpy()
        selected_particle_ids = selected_particle_ids[
            selected_particle_ids >= 0
        ]
        particle_minimum_ratios = (
            self.persistent_grip_particle_minimum_tet_volume_ratio.numpy()
        )
        selected_minimum_ratio = (
            float(np.min(particle_minimum_ratios[selected_particle_ids]))
            if len(selected_particle_ids)
            else float("inf")
        )
        grip_body_ids = self.persistent_grip_particle_body.numpy()
        grip_direct = self.persistent_grip_particle_direct.numpy() != 0
        return {
            "sample_count": self.sample_count,
            "skin_node_count": self.skin_node_count,
            "top_barrier_face_count": self.top_face_count,
            "top_barrier_node_count": self.top_node_count,
            "top_barrier_contact_count": int(
                self.top_barrier_contact_count.numpy()[0]
            ),
            "jaw_triangle_contact_count": int(
                self.jaw_triangle_contact_count.numpy()[0]
            ),
            "jaw_contact_shape_ids": self.jaw_contact_shape_ids,
            "jaw_contact_distal_length_m": (
                self.jaw_contact_distal_length_m
            ),
            "top_barrier_distal_length_m": (
                self.top_barrier_distal_length_m
            ),
            "top_barrier_tip_allowance_m": (
                self.top_barrier_tip_allowance_m
            ),
            "jaw_contact_enabled_sample_count": (
                self.jaw_contact_enabled_sample_count
            ),
            "top_barrier_enabled_sample_count": (
                self.top_barrier_enabled_sample_count
            ),
            "jaw_closing_contact_count_by_shape": {
                int(shape_id): int(closing_counts[shape_id])
                for shape_id in self.jaw_contact_shape_ids
            },
            "jaw_closing_resistance_by_shape_m2": {
                int(shape_id): float(closing_resistance[shape_id])
                for shape_id in self.jaw_contact_shape_ids
            },
            "jaw_friction_coefficient": (
                self.jaw_friction_coefficient
            ),
            "persistent_grip_constraint_enabled": (
                self.persistent_grip_enabled
            ),
            "persistent_grip_control_mode": (
                "two_contact_points_per_jaw_finite_volume_u_t_with_local_lift_support"
                if self.persistent_grip_support_radius_m > 0.0
                else "two_contact_points_per_jaw_finite_volume_u_t"
            ),
            "persistent_grip_active": bool(
                self.persistent_grip_state.numpy()[0]
            ),
            "persistent_grip_activation_counter": int(
                self.persistent_grip_state.numpy()[1]
            ),
            "persistent_grip_activation_steps": (
                self.persistent_grip_activation_steps
            ),
            "persistent_grip_between_jaw_candidate_count": int(
                self.persistent_grip_between_jaw_candidate_count.numpy()[0]
            ),
            "persistent_grip_minimum_contact_samples_per_jaw": (
                self.persistent_grip_minimum_contact_samples_per_jaw
            ),
            "persistent_grip_nearest_surface_particles": (
                self.persistent_grip_nearest_surface_particles
            ),
            "persistent_grip_virtual_body_id": (
                self.persistent_grip_virtual_body_id
            ),
            "persistent_grip_selected_particle_count": int(
                self.persistent_grip_selected_particle_count.numpy()[0]
            ),
            "persistent_grip_selected_particle_ids": tuple(
                int(particle_id)
                for particle_id in selected_particle_ids
            ),
            "persistent_grip_minimum_capture_volume_ratio": (
                self.persistent_grip_minimum_capture_volume_ratio
            ),
            "persistent_grip_selected_minimum_tet_volume_ratio": (
                selected_minimum_ratio
            ),
            "persistent_grip_grasp_center_m": tuple(
                float(value)
                for value in self.persistent_grip_grasp_center.numpy()[0]
            ),
            "persistent_grip_jaw_patch_separation_m": float(
                self.persistent_grip_jaw_patch_separation.numpy()[0]
            ),
            "persistent_grip_maximum_jaw_patch_separation_m": (
                self.persistent_grip_maximum_jaw_patch_separation_m
            ),
            "persistent_grip_particle_count": int(
                np.count_nonzero(
                    self.persistent_grip_particle_body.numpy() >= 0
                )
            ),
            "persistent_grip_direct_particle_count": int(
                np.count_nonzero(
                    grip_direct
                )
            ),
            "persistent_grip_direct_particle_count_by_jaw_body": {
                int(body_id): int(
                    np.count_nonzero(grip_direct & (grip_body_ids == body_id))
                )
                for body_id in self.persistent_grip_jaw_body_ids
            },
            "persistent_grip_support_particle_count": int(
                np.count_nonzero(
                    (self.persistent_grip_particle_body.numpy() >= 0)
                    & (self.persistent_grip_particle_direct.numpy() == 0)
                )
            ),
            "persistent_grip_support_radius_m": (
                self.persistent_grip_support_radius_m
            ),
            "persistent_grip_compliance_m_per_n": (
                self.persistent_grip_compliance_m_per_n
            ),
            "persistent_grip_transfer_layers": (
                self.persistent_grip_transfer_layers
            ),
            "persistent_grip_transfer_tet_count": (
                self.persistent_grip_transfer_tet_count
            ),
            "persistent_grip_safety_tet_count": (
                self.persistent_grip_safety_tet_count
            ),
            "persistent_grip_relaxation": self.persistent_grip_relaxation,
            "persistent_grip_maximum_correction_m": (
                self.persistent_grip_maximum_correction_m
            ),
            "persistent_grip_minimum_volume_ratio": (
                self.persistent_grip_minimum_volume_ratio
            ),
            "persistent_grip_inversion_safe_scale": float(
                self.persistent_grip_global_scale.numpy()[0]
            ),
            "persistent_grip_support_safe_scale": float(
                self.persistent_grip_global_scale.numpy()[0]
            ),
            "persistent_grip_direct_safe_scale": float(
                self.persistent_grip_direct_scale.numpy()[0]
            ),
            "persistent_grip_jaw_safe_scales": tuple(
                float(value)
                for value in self.persistent_grip_jaw_scales.numpy()
            ),
            "persistent_grip_support_unsafe_tetrahedra": int(
                self.persistent_grip_support_unsafe_tet_count.numpy()[0]
            ),
            "persistent_grip_support_candidate_count": (
                self.persistent_grip_support_candidate_count
            ),
            "persistent_grip_support_generation": int(
                self.persistent_grip_support_generation.numpy()[0]
            ),
            "persistent_grip_support_generations": (
                self.persistent_grip_support_generations
            ),
            "persistent_grip_q7_angle_rad": float(
                self.persistent_grip_current_jaw_angle.numpy()[0]
            ),
            "persistent_grip_q7_timestamp_s": float(
                self.persistent_grip_current_signal_timestamp.numpy()[0]
            ),
            "persistent_grip_q7_motion_state": (
                self.persistent_grip_jaw_motion_state
            ),
            "persistent_grip_capture_allowed": bool(
                self.persistent_grip_capture_allowed.numpy()[0]
            ),
            "persistent_grip_capture_jaw_angle_rad": float(
                self.persistent_grip_capture_jaw_angle.numpy()[0]
            ),
            "persistent_grip_capture_timestamp_s": float(
                self.persistent_grip_capture_timestamp.numpy()[0]
            ),
            "persistent_grip_release_requested": bool(
                self.persistent_grip_release_requested.numpy()[0]
            ),
            "persistent_grip_closed_angle_max_rad": (
                self.persistent_grip_closed_angle_max_rad
            ),
            "persistent_grip_release_angle_min_rad": (
                self.persistent_grip_release_angle_min_rad
            ),
            "persistent_grip_release_angle_delta_rad": (
                self.persistent_grip_release_angle_delta_rad
            ),
            "persistent_grip_maximum_capture_penetration_m": (
                self.persistent_grip_maximum_capture_penetration_m
            ),
            "top_barrier_shape_ids": self.top_barrier_shape_ids,
            "top_support_lateral_radius_m": (
                self.top_support_lateral_radius_m
            ),
            "top_support_depth_m": self.top_support_depth_m,
            "top_support_weight_scale": self.top_support_weight_scale,
            "top_pressure_shoulder_lateral_radius_m": (
                self.top_pressure_shoulder_lateral_radius_m
            ),
            "top_pressure_shoulder_depth_m": (
                self.top_pressure_shoulder_depth_m
            ),
            "top_pressure_shoulder_upward_scale": (
                self.top_pressure_shoulder_upward_scale
            ),
            "top_pressure_shoulder_outward_scale": (
                self.top_pressure_shoulder_outward_scale
            ),
            "top_pressure_shoulder_bias_direction_world": (
                self.top_pressure_shoulder_bias_direction_world
            ),
            "top_pressure_shoulder_bias_start_m": (
                self.top_pressure_shoulder_bias_start_m
            ),
            "top_pressure_shoulder_entry_count": (
                self.top_pressure_shoulder_entry_count
            ),
            "top_pressure_shoulder_max_nodes_per_face": (
                self.top_pressure_shoulder_max_nodes_per_face
            ),
            "top_pressure_shoulder_mean_nodes_per_face": (
                self.top_pressure_shoulder_mean_nodes_per_face
            ),
            "top_barrier_lateral_tolerance_m": (
                self.top_barrier_lateral_tolerance_m
            ),
            "top_barrier_contact_patch_radius_m": (
                self.top_barrier_contact_patch_radius_m
            ),
            "top_barrier_clearance_m": self.top_barrier_clearance_m,
            "top_support_entry_count": self.top_support_entry_count,
            "top_support_max_nodes_per_face": (
                self.top_support_max_nodes_per_face
            ),
            "top_support_mean_nodes_per_face": (
                self.top_support_mean_nodes_per_face
            ),
            "top_support_id_weight_bytes": (
                self.top_support_entry_count * 8
            ),
            "contact_safety_tet_count": self.contact_tet_count,
            "contact_spread_layers": self.spread_layers,
            "contact_spread_tet_count": self.spread_tet_count,
            "samples_per_shape": dict(self.samples_per_shape),
            "contact_count": int(self.contact_count.numpy()[0]),
            "contact_count_by_shape": {
                int(shape_id): int(counts[shape_id])
                for shape_id in self.tool_shape_ids
            },
            "maximum_penetration_m": float(
                self.maximum_penetration.numpy()[0]
            ),
            "minimum_signed_distance_m": float(
                self.minimum_signed_distance.numpy()[0]
            ),
            "top_minimum_signed_distance_m": float(
                self.top_minimum_signed_distance.numpy()[0]
            ),
            "global_inversion_safe_scale": float(
                self.global_scale.numpy()[0]
            ),
            "contact_local_unsafe_tetrahedra": int(
                self.contact_unsafe_tet_count.numpy()[0]
            ),
        }

    def jaw_closure_resistance_metrics(self) -> dict:
        """Return the small feedback packet needed by the jaw actuator."""
        closing_counts = self.jaw_closing_contact_count_by_shape.numpy()
        closing_resistance = self.jaw_closing_resistance_by_shape.numpy()
        return {
            "jaw_shape_ids": self.jaw_contact_shape_ids,
            "contact_counts": tuple(
                int(closing_counts[shape_id])
                for shape_id in self.jaw_contact_shape_ids
            ),
            "resistance_m2": tuple(
                float(closing_resistance[shape_id])
                for shape_id in self.jaw_contact_shape_ids
            ),
        }

    _PERSISTENT_GRIP_SNAPSHOT_ARRAYS = (
        "persistent_grip_state",
        "persistent_grip_current_jaw_angle",
        "persistent_grip_current_signal_timestamp",
        "persistent_grip_capture_allowed",
        "persistent_grip_release_requested",
        "persistent_grip_capture_jaw_angle",
        "persistent_grip_capture_timestamp",
        "persistent_grip_jaw_a_counts",
        "persistent_grip_jaw_b_counts",
        "persistent_grip_contact_patch_sums",
        "persistent_grip_grasp_center",
        "persistent_grip_jaw_patch_separation",
        "persistent_grip_selected_particle_ids",
        "persistent_grip_selected_particle_count",
        "persistent_grip_between_jaw_candidate_count",
        "persistent_grip_particle_minimum_tet_volume_ratio",
        "persistent_grip_particle_body",
        "persistent_grip_particle_local",
        "persistent_grip_particle_direct",
        "persistent_grip_particle_weight",
        "persistent_grip_particle_level",
        "persistent_grip_support_generation",
    )

    def clone_persistent_grip_state(self) -> PersistentGripSnapshot:
        """Deep-copy every persistent value that affects future grip solves."""
        arrays = {
            name: wp.to_torch(getattr(self, name)).detach().clone()
            for name in self._PERSISTENT_GRIP_SNAPSHOT_ARRAYS
        }
        return PersistentGripSnapshot(
            arrays=arrays,
            previous_jaw_angle=self.persistent_grip_previous_jaw_angle,
            jaw_motion_state=str(self.persistent_grip_jaw_motion_state),
        )

    def restore_persistent_grip_state(
        self, snapshot: PersistentGripSnapshot
    ) -> None:
        """Restore a snapshot without changing immutable contact geometry."""
        expected = set(self._PERSISTENT_GRIP_SNAPSHOT_ARRAYS)
        if set(snapshot.arrays) != expected:
            missing = sorted(expected.difference(snapshot.arrays))
            extra = sorted(set(snapshot.arrays).difference(expected))
            raise ValueError(
                "Persistent-grip snapshot fields disagree: "
                f"missing={missing}, extra={extra}"
            )
        with torch.no_grad():
            for name in self._PERSISTENT_GRIP_SNAPSHOT_ARRAYS:
                destination = wp.to_torch(getattr(self, name))
                source = snapshot.arrays[name].to(
                    device=destination.device, dtype=destination.dtype
                )
                if source.shape != destination.shape:
                    raise ValueError(
                        f"Persistent-grip snapshot shape mismatch for {name}: "
                        f"{source.shape} != {destination.shape}"
                    )
                destination.copy_(source)
        self.persistent_grip_previous_jaw_angle = (
            None
            if snapshot.previous_jaw_angle is None
            else float(snapshot.previous_jaw_angle)
        )
        self.persistent_grip_jaw_motion_state = str(
            snapshot.jaw_motion_state
        )

    def freeze_persistent_grip_state_machine(self) -> None:
        """Keep existing attachments while preventing capture or release."""
        self.persistent_grip_capture_allowed.zero_()
        self.persistent_grip_release_requested.zero_()

    def reset_persistent_grip(self) -> None:
        self.persistent_grip_state.zero_()
        self.persistent_grip_capture_jaw_angle.zero_()
        self.persistent_grip_capture_timestamp.zero_()
        self.persistent_grip_current_jaw_angle.zero_()
        self.persistent_grip_current_signal_timestamp.zero_()
        self.persistent_grip_capture_allowed.zero_()
        self.persistent_grip_release_requested.fill_(1)
        self.persistent_grip_jaw_a_counts.zero_()
        self.persistent_grip_jaw_b_counts.zero_()
        self.persistent_grip_contact_patch_sums.zero_()
        self.persistent_grip_grasp_center.zero_()
        self.persistent_grip_jaw_patch_separation.fill_(1.0e6)
        self.persistent_grip_selected_particle_ids.fill_(-1)
        self.persistent_grip_selected_particle_count.zero_()
        self.persistent_grip_between_jaw_candidate_count.zero_()
        self.persistent_grip_particle_body.fill_(-1)
        self.persistent_grip_particle_local.zero_()
        self.persistent_grip_particle_direct.zero_()
        self.persistent_grip_particle_weight.zero_()
        self.persistent_grip_particle_level.fill_(-1)
        self.persistent_grip_deltas.zero_()
        self.persistent_grip_delta_counts.zero_()
        self.persistent_grip_spread_deltas_a.zero_()
        self.persistent_grip_spread_weights_a.zero_()
        self.persistent_grip_spread_deltas_b.zero_()
        self.persistent_grip_spread_weights_b.zero_()
        self.persistent_grip_support_generation.zero_()
        self.persistent_grip_previous_jaw_angle = None
        self.persistent_grip_jaw_motion_state = "unknown"

    def set_persistent_grip_jaw_signal(
        self,
        jaw_angle_rad: float,
        timestamp_s: float,
        *,
        contact_limited_actuator_enabled: bool = False,
        closure_blocked_by_tissue: bool = False,
        closing_requested: bool = False,
    ) -> None:
        """Update the q7[6] capture/release gate for the next contact solve."""
        angle = float(jaw_angle_rad)
        timestamp = float(timestamp_s)
        if not np.isfinite(angle) or not np.isfinite(timestamp):
            raise ValueError("Persistent-grip jaw signal must be finite")
        previous = self.persistent_grip_previous_jaw_angle
        epsilon = self.persistent_grip_angle_motion_epsilon_rad
        closing = previous is not None and angle < previous - epsilon
        opening = previous is not None and angle > previous + epsilon
        closed = angle <= self.persistent_grip_closed_angle_max_rad
        near_closed_while_closing = (
            closing and angle <= self.persistent_grip_release_angle_min_rad
        )
        wide_open = angle >= self.persistent_grip_wide_open_angle_rad
        release = wide_open or (
            opening and angle >= self.persistent_grip_release_angle_min_rad
        )
        # The current paper_soft path follows raw q7 and enables capture only
        # near/at closure.  The contact-limited branch remains available for
        # older callers but is intentionally disabled by the SUPER GUI.
        capture_allowed = (
            bool(closure_blocked_by_tissue and closing_requested)
            if contact_limited_actuator_enabled
            else bool(closed or near_closed_while_closing)
        )
        if wide_open:
            motion_state = "open"
        elif opening:
            motion_state = "opening"
        elif closing:
            motion_state = "closing"
        elif closed:
            motion_state = "closed"
        else:
            motion_state = "holding"
        self.persistent_grip_current_jaw_angle.fill_(angle)
        self.persistent_grip_current_signal_timestamp.fill_(timestamp)
        self.persistent_grip_capture_allowed.fill_(int(capture_allowed))
        self.persistent_grip_release_requested.fill_(int(release))
        self.persistent_grip_previous_jaw_angle = angle
        self.persistent_grip_jaw_motion_state = motion_state
