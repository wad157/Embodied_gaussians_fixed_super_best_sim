# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from typing_extensions import override
import numpy as np
import warp as wp
import warp.sim
from warp.sim.integrator import integrate_particles
from warp.sim.integrator_xpbd import (
    solve_particle_ground_contacts,
    solve_particle_shape_contacts,
)
from warp.sim.model import PARTICLE_FLAG_ACTIVE


@wp.func
def integrate_rigid_body(
    q: wp.transform,
    qd: wp.spatial_vector,
    f: wp.spatial_vector,
    com: wp.vec3,
    inertia: wp.mat33,
    inv_mass: float,
    inv_inertia: wp.mat33,
    gravity: wp.vec3,
    angular_damping: float,
    dt: float,
):
    # unpack transform
    x0 = wp.transform_get_translation(q)
    r0 = wp.transform_get_rotation(q)

    # unpack spatial twist
    w0 = wp.spatial_top(qd)
    v0 = wp.spatial_bottom(qd)

    # unpack spatial wrench
    t0 = wp.spatial_top(f)
    f0 = wp.spatial_bottom(f)

    x_com = x0 + wp.quat_rotate(r0, com)

    # linear part
    v1 = v0 + (f0 * inv_mass + gravity * wp.nonzero(inv_mass)) * dt
    x1 = x_com + v1 * dt

    # angular part (compute in body frame)
    wb = wp.quat_rotate_inv(r0, w0)
    tb = wp.quat_rotate_inv(r0, t0) - wp.cross(wb, inertia * wb)  # coriolis forces

    w1 = wp.quat_rotate(r0, wb + inv_inertia * tb * dt)
    r1 = wp.normalize(r0 + wp.quat(w1, 0.0) * r0 * 0.5 * dt)

    # angular damping
    w1 *= 1.0 - angular_damping * dt

    q_new = wp.transform(x1 - wp.quat_rotate(r1, com), r1)
    qd_new = wp.spatial_vector(w1, v1)

    return q_new, qd_new


# semi-implicit euler integration
@wp.kernel
def integrate_bodies(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    i: wp.array(dtype=wp.mat33),
    inv_m: wp.array(dtype=float),
    inv_i: wp.array(dtype=wp.mat33),
    gravity_factor: wp.array(dtype=float),
    gravity: wp.vec3,
    angular_damping: float,
    preserve_static_body_poses: bool,
    dt: float,
    # outputs
    body_q_new: wp.array(dtype=wp.transform),
    body_qd_new: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    # positions
    q = body_q[tid]
    qd = body_qd[tid]
    f = body_f[tid]

    # masses
    inv_mass = inv_m[tid]  # 1 / mass

    if preserve_static_body_poses and inv_mass == 0.0:
        # Externally driven kinematic bodies keep their supplied pose while
        # retaining body_qd for particle-shape friction calculations.
        body_q_new[tid] = q
        body_qd_new[tid] = qd
        return

    inertia = i[tid]
    inv_inertia = inv_i[tid]  # inverse of 3x3 inertia matrix

    com = body_com[tid]

    q_new, qd_new = integrate_rigid_body(
        q,
        qd,
        f,
        com,
        inertia,
        inv_mass,
        inv_inertia,
        gravity * gravity_factor[tid],
        angular_damping,
        dt,
    )

    body_q_new[tid] = q_new
    body_qd_new[tid] = qd_new


class XPBDIntegrator(warp.sim.XPBDIntegrator):
    """
    Override integrate bodies to make robot not affected by gravity
    """

    def __init__(self, *args, preserve_static_body_poses: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.preserve_static_body_poses = bool(preserve_static_body_poses)

    @override
    def integrate_bodies(
        self,
        model: warp.sim.Model,
        state_in: warp.sim.State,
        state_out: warp.sim.State,
        dt: float,
        angular_damping: float = 0.0,
    ):
        """
        Integrate the rigid bodies of the model.

        Args:
            model (Model): The model to integrate.
            state_in (State): The input state.
            state_out (State): The output state.
            dt (float): The time step (typically in seconds).
            angular_damping (float, optional): The angular damping factor. Defaults to 0.0.
        """
        if model.body_count:
            wp.launch(
                kernel=integrate_bodies,
                dim=model.body_count,
                inputs=[
                    state_in.body_q,
                    state_in.body_qd,
                    state_in.body_f,
                    model.body_com,
                    model.body_mass,
                    model.body_inertia,
                    model.body_inv_mass,
                    model.body_inv_inertia,
                    model.gravity_factor,
                    model.gravity,
                    angular_damping,
                    self.preserve_static_body_poses,
                    dt,
                ],
                outputs=[state_out.body_q, state_out.body_qd],
                device=model.device,
            )


@wp.kernel
def solve_material_tetrahedra_xpbd(
    positions: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    tet_indices: wp.array(dtype=int, ndim=2),
    inverse_rest_matrix: wp.array(dtype=wp.mat33),
    tet_materials: wp.array(dtype=float, ndim=2),
    dt: float,
    compliance_scale: float,
    relaxation: float,
    lambda_energy: wp.array(dtype=float),
    corner_deltas: wp.array(dtype=wp.vec3),
):
    """Project a compressible Neo-Hookean energy constraint for one tet.

    This is the minimal project-local XPBD formulation used by the SUPER B0
    solver gate.  ``k_mu`` and ``k_lambda`` are interpreted as Lamé parameters
    in Pa. The square root of twice the element energy is the XPBD constraint,
    following the arbitrary-energy construction in the XPBD paper. The total
    Lagrange multiplier accumulates across the iterations of one substep.
    """
    tid = wp.tid()
    i = tet_indices[tid, 0]
    j = tet_indices[tid, 1]
    k = tet_indices[tid, 2]
    l = tet_indices[tid, 3]

    x0 = positions[i]
    x1 = positions[j]
    x2 = positions[k]
    x3 = positions[l]
    w0 = inverse_mass[i]
    w1 = inverse_mass[j]
    w2 = inverse_mass[k]
    w3 = inverse_mass[l]

    dm_inverse = inverse_rest_matrix[tid]
    inverse_rest_volume = wp.determinant(dm_inverse) * 6.0
    if inverse_rest_volume <= 0.0:
        return
    rest_volume = 1.0 / inverse_rest_volume

    ds = wp.matrix_from_cols(x1 - x0, x2 - x0, x3 - x0)
    deformation = ds * dm_inverse
    f0 = wp.vec3(deformation[0, 0], deformation[1, 0], deformation[2, 0])
    f1 = wp.vec3(deformation[0, 1], deformation[1, 1], deformation[2, 1])
    f2 = wp.vec3(deformation[0, 2], deformation[1, 2], deformation[2, 2])
    mu = tet_materials[tid, 0]
    lame_lambda = tet_materials[tid, 1]
    determinant = wp.determinant(deformation)
    if mu <= 0.0 or lame_lambda < 0.0 or determinant <= 1.0e-6:
        return

    log_determinant = wp.log(determinant)
    first_invariant = wp.dot(f0, f0) + wp.dot(f1, f1) + wp.dot(f2, f2)
    energy_density = (
        0.5 * mu * (first_invariant - 3.0 - 2.0 * log_determinant)
        + 0.5 * lame_lambda * log_determinant * log_determinant
    )
    total_energy = rest_volume * energy_density
    if total_energy <= 1.0e-12:
        return

    constraint = wp.sqrt(2.0 * total_energy)
    cofactor = wp.matrix_from_cols(
        wp.cross(f1, f2), wp.cross(f2, f0), wp.cross(f0, f1)
    )
    inverse_transpose = cofactor * (1.0 / determinant)
    first_piola = (
        mu * deformation
        + (-mu + lame_lambda * log_determinant) * inverse_transpose
    )
    derivative_f = first_piola * (rest_volume / constraint)
    derivative_x = derivative_f * wp.transpose(dm_inverse)
    grad1 = wp.vec3(derivative_x[0, 0], derivative_x[1, 0], derivative_x[2, 0])
    grad2 = wp.vec3(derivative_x[0, 1], derivative_x[1, 1], derivative_x[2, 1])
    grad3 = wp.vec3(derivative_x[0, 2], derivative_x[1, 2], derivative_x[2, 2])
    grad0 = -grad1 - grad2 - grad3
    weighted_gradient = (
        w0 * wp.dot(grad0, grad0)
        + w1 * wp.dot(grad1, grad1)
        + w2 * wp.dot(grad2, grad2)
        + w3 * wp.dot(grad3, grad3)
    )
    alpha = compliance_scale / (dt * dt)
    old_lambda = lambda_energy[tid]
    delta_lambda = (-constraint - alpha * old_lambda) / (
        weighted_gradient + alpha
    )
    delta_lambda = relaxation * delta_lambda
    lambda_energy[tid] = old_lambda + delta_lambda
    corner_start = tid * 4
    corner_deltas[corner_start] = w0 * delta_lambda * grad0
    corner_deltas[corner_start + 1] = w1 * delta_lambda * grad1
    corner_deltas[corner_start + 2] = w2 * delta_lambda * grad2
    corner_deltas[corner_start + 3] = w3 * delta_lambda * grad3


@wp.kernel
def solve_paper_distance_constraints_xpbd(
    positions: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    edge_indices: wp.array(dtype=int, ndim=2),
    rest_lengths: wp.array(dtype=float),
    particle_stiffness: wp.array(dtype=float),
    dt: float,
    compliance_scale: float,
    relaxation: float,
    constraint_lambda: wp.array(dtype=float),
    corner_deltas: wp.array(dtype=wp.vec3),
):
    """Project the paper's connected-particle distance constraints with XPBD."""
    edge_id = wp.tid()
    i = edge_indices[edge_id, 0]
    j = edge_indices[edge_id, 1]
    difference = positions[i] - positions[j]
    length = wp.length(difference)
    if length <= 1.0e-12:
        return
    stiffness = 0.5 * (
        particle_stiffness[i] + particle_stiffness[j]
    )
    if stiffness <= 0.0:
        return
    w_i = inverse_mass[i]
    w_j = inverse_mass[j]
    alpha = compliance_scale / (stiffness * dt * dt)
    old_lambda = constraint_lambda[edge_id]
    constraint = length - rest_lengths[edge_id]
    delta_lambda = (
        -constraint - alpha * old_lambda
    ) / (w_i + w_j + alpha)
    delta_lambda *= relaxation
    constraint_lambda[edge_id] = old_lambda + delta_lambda
    gradient = difference / length
    corner_start = edge_id * 2
    corner_deltas[corner_start] = w_i * delta_lambda * gradient
    corner_deltas[corner_start + 1] = -w_j * delta_lambda * gradient


@wp.kernel
def solve_paper_volume_constraints_xpbd(
    positions: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    tet_indices: wp.array(dtype=int, ndim=2),
    rest_volumes: wp.array(dtype=float),
    stiffness: float,
    dt: float,
    compliance_scale: float,
    relaxation: float,
    constraint_lambda: wp.array(dtype=float),
    corner_deltas: wp.array(dtype=wp.vec3),
):
    """Project signed tetrahedron-volume constraints with fixed paper stiffness."""
    tet_id = wp.tid()
    i = tet_indices[tet_id, 0]
    j = tet_indices[tet_id, 1]
    k = tet_indices[tet_id, 2]
    l = tet_indices[tet_id, 3]
    x0 = positions[i]
    x1 = positions[j]
    x2 = positions[k]
    x3 = positions[l]
    gradient_1 = wp.cross(x2 - x0, x3 - x0) / 6.0
    gradient_2 = wp.cross(x3 - x0, x1 - x0) / 6.0
    gradient_3 = wp.cross(x1 - x0, x2 - x0) / 6.0
    gradient_0 = -gradient_1 - gradient_2 - gradient_3
    w0 = inverse_mass[i]
    w1 = inverse_mass[j]
    w2 = inverse_mass[k]
    w3 = inverse_mass[l]
    weighted_gradient = (
        w0 * wp.dot(gradient_0, gradient_0)
        + w1 * wp.dot(gradient_1, gradient_1)
        + w2 * wp.dot(gradient_2, gradient_2)
        + w3 * wp.dot(gradient_3, gradient_3)
    )
    if stiffness <= 0.0 or weighted_gradient <= 1.0e-20:
        return
    current_volume = wp.dot(
        wp.cross(x1 - x0, x2 - x0), x3 - x0
    ) / 6.0
    constraint = current_volume - rest_volumes[tet_id]
    alpha = compliance_scale / (stiffness * dt * dt)
    old_lambda = constraint_lambda[tet_id]
    delta_lambda = (
        -constraint - alpha * old_lambda
    ) / (weighted_gradient + alpha)
    delta_lambda *= relaxation
    constraint_lambda[tet_id] = old_lambda + delta_lambda
    corner_start = tet_id * 4
    corner_deltas[corner_start] = w0 * delta_lambda * gradient_0
    corner_deltas[corner_start + 1] = w1 * delta_lambda * gradient_1
    corner_deltas[corner_start + 2] = w2 * delta_lambda * gradient_2
    corner_deltas[corner_start + 3] = w3 * delta_lambda * gradient_3


@wp.kernel
def solve_paper_shape_matching_constraints_xpbd(
    positions: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    tet_indices: wp.array(dtype=int, ndim=2),
    tet_rest_pose: wp.array(dtype=wp.mat33),
    rest_relative_positions: wp.array(dtype=wp.vec3),
    particle_stiffness: wp.array(dtype=float),
    dt: float,
    compliance_scale: float,
    relaxation: float,
    constraint_lambda: wp.array(dtype=float),
    corner_deltas: wp.array(dtype=wp.vec3),
):
    """Project one four-particle shape-matching cluster per tetrahedron.

    The polar goal is held fixed during each local XPBD projection, which is
    the standard PBD shape-matching approximation.  The scalar constraint is
    the L2 norm of the four goal residuals.
    """
    tet_id = wp.tid()
    i = tet_indices[tet_id, 0]
    j = tet_indices[tet_id, 1]
    k = tet_indices[tet_id, 2]
    l = tet_indices[tet_id, 3]
    x0 = positions[i]
    x1 = positions[j]
    x2 = positions[k]
    x3 = positions[l]
    center = 0.25 * (x0 + x1 + x2 + x3)
    corner_start = tet_id * 4
    q0 = rest_relative_positions[corner_start]
    q1 = rest_relative_positions[corner_start + 1]
    q2 = rest_relative_positions[corner_start + 2]
    q3 = rest_relative_positions[corner_start + 3]
    # Polar-decompose the deformation gradient instead of the raw Apq
    # covariance.  The v12 mesh deliberately permits moderately anisotropic
    # tetrahedra; preconditioning by the inverse rest pose makes the identity
    # rest state numerically exact enough that shape matching cannot inject a
    # systematic motion into an otherwise stationary tissue.
    current_pose = wp.matrix_from_cols(
        x1 - x0, x2 - x0, x3 - x0
    )
    deformation = current_pose * tet_rest_pose[tet_id]
    U = wp.mat33f()
    sigma = wp.vec3f()
    V = wp.mat33f()
    wp.svd3(deformation, U, sigma, V)
    rotation = U * wp.transpose(V)
    if wp.determinant(rotation) < 0.0:
        reflection_fix = wp.mat33f(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, -1.0,
        )
        rotation = U * reflection_fix * wp.transpose(V)
    residual_0 = x0 - (center + rotation * q0)
    residual_1 = x1 - (center + rotation * q1)
    residual_2 = x2 - (center + rotation * q2)
    residual_3 = x3 - (center + rotation * q3)
    squared_constraint = (
        wp.dot(residual_0, residual_0)
        + wp.dot(residual_1, residual_1)
        + wp.dot(residual_2, residual_2)
        + wp.dot(residual_3, residual_3)
    )
    squared_rest_extent = (
        wp.dot(q0, q0)
        + wp.dot(q1, q1)
        + wp.dot(q2, q2)
        + wp.dot(q3, q3)
    )
    # A relative 1e-4 positional dead band is below the image/depth
    # resolution, but prevents float32 SVD noise from accumulating over many
    # substeps at the reconstructed rest pose.
    if squared_constraint <= 1.0e-8 * squared_rest_extent:
        return
    constraint = wp.sqrt(squared_constraint)
    gradient_0 = residual_0 / constraint
    gradient_1 = residual_1 / constraint
    gradient_2 = residual_2 / constraint
    gradient_3 = residual_3 / constraint
    w0 = inverse_mass[i]
    w1 = inverse_mass[j]
    w2 = inverse_mass[k]
    w3 = inverse_mass[l]
    weighted_gradient = (
        w0 * wp.dot(gradient_0, gradient_0)
        + w1 * wp.dot(gradient_1, gradient_1)
        + w2 * wp.dot(gradient_2, gradient_2)
        + w3 * wp.dot(gradient_3, gradient_3)
    )
    stiffness = 0.25 * (
        particle_stiffness[i]
        + particle_stiffness[j]
        + particle_stiffness[k]
        + particle_stiffness[l]
    )
    if stiffness <= 0.0 or weighted_gradient <= 1.0e-20:
        return
    alpha = compliance_scale / (stiffness * dt * dt)
    old_lambda = constraint_lambda[tet_id]
    delta_lambda = (
        -constraint - alpha * old_lambda
    ) / (weighted_gradient + alpha)
    delta_lambda *= relaxation
    constraint_lambda[tet_id] = old_lambda + delta_lambda
    corner_deltas[corner_start] = w0 * delta_lambda * gradient_0
    corner_deltas[corner_start + 1] = w1 * delta_lambda * gradient_1
    corner_deltas[corner_start + 2] = w2 * delta_lambda * gradient_2
    corner_deltas[corner_start + 3] = w3 * delta_lambda * gradient_3


@wp.kernel(enable_backward=False)
def reduce_material_tetrahedron_deltas(
    particle_corner_offsets: wp.array(dtype=int),
    particle_corner_ids: wp.array(dtype=int),
    corner_deltas: wp.array(dtype=wp.vec3),
    deltas: wp.array(dtype=wp.vec3),
):
    """Sum incident tetrahedron contributions in a fixed corner-id order."""
    particle_index = wp.tid()
    start = particle_corner_offsets[particle_index]
    end = particle_corner_offsets[particle_index + 1]
    total = wp.vec3()
    for offset in range(start, end):
        total += corner_deltas[particle_corner_ids[offset]]
    deltas[particle_index] = total


@wp.kernel(enable_backward=False)
def reduce_averaged_constraint_deltas(
    particle_corner_offsets: wp.array(dtype=int),
    particle_corner_ids: wp.array(dtype=int),
    corner_deltas: wp.array(dtype=wp.vec3),
    deltas: wp.array(dtype=wp.vec3),
):
    """Average Jacobi constraint proposals incident on one particle."""
    particle_index = wp.tid()
    start = particle_corner_offsets[particle_index]
    end = particle_corner_offsets[particle_index + 1]
    total = wp.vec3()
    for offset in range(start, end):
        total += corner_deltas[particle_corner_ids[offset]]
    count = end - start
    if count > 0:
        total /= float(count)
    deltas[particle_index] = total


@wp.kernel
def apply_material_tetrahedron_deltas(
    positions: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.uint32),
    deltas: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:
        return
    positions[tid] = positions[tid] + deltas[tid]


@wp.kernel
def clamp_particle_shape_deltas(
    deltas: wp.array(dtype=wp.vec3),
    max_correction: float,
):
    """Apply a per-particle contact trust region in-place."""
    tid = wp.tid()
    delta = deltas[tid]
    magnitude = wp.length(delta)
    if max_correction > 0.0 and magnitude > max_correction:
        deltas[tid] = delta * (max_correction / magnitude)


@wp.kernel
def find_safe_particle_shape_delta_scale(
    positions: wp.array(dtype=wp.vec3),
    deltas: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=int, ndim=2),
    inverse_rest_matrix: wp.array(dtype=wp.mat33),
    minimum_volume_ratio: float,
    global_scale: wp.array(dtype=float),
):
    """Line-search a shared contact scale that keeps every tet non-inverted."""
    tet_id = wp.tid()
    i = tet_indices[tet_id, 0]
    j = tet_indices[tet_id, 1]
    k = tet_indices[tet_id, 2]
    l = tet_indices[tet_id, 3]
    dm_inverse = inverse_rest_matrix[tet_id]

    x0 = positions[i]
    x1 = positions[j]
    x2 = positions[k]
    x3 = positions[l]
    d0 = deltas[i]
    d1 = deltas[j]
    d2 = deltas[k]
    d3 = deltas[l]
    current_ds = wp.matrix_from_cols(x1 - x0, x2 - x0, x3 - x0)
    current_ratio = wp.determinant(current_ds * dm_inverse)
    if current_ratio <= minimum_volume_ratio:
        wp.atomic_min(global_scale, 0, 0.0)
        return

    proposed_ds = wp.matrix_from_cols(
        (x1 + d1) - (x0 + d0),
        (x2 + d2) - (x0 + d0),
        (x3 + d3) - (x0 + d0),
    )
    proposed_ratio = wp.determinant(proposed_ds * dm_inverse)
    if proposed_ratio >= minimum_volume_ratio:
        return

    lower = 0.0
    upper = 1.0
    for _ in range(12):
        middle = 0.5 * (lower + upper)
        middle_ds = wp.matrix_from_cols(
            (x1 + middle * d1) - (x0 + middle * d0),
            (x2 + middle * d2) - (x0 + middle * d0),
            (x3 + middle * d3) - (x0 + middle * d0),
        )
        middle_ratio = wp.determinant(middle_ds * dm_inverse)
        if middle_ratio >= minimum_volume_ratio:
            lower = middle
        else:
            upper = middle
    wp.atomic_min(global_scale, 0, lower)


@wp.kernel
def apply_scaled_particle_shape_deltas(
    positions: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.uint32),
    deltas: wp.array(dtype=wp.vec3),
    global_scale: wp.array(dtype=float),
):
    tid = wp.tid()
    if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:
        return
    positions[tid] = positions[tid] + global_scale[0] * deltas[tid]


@wp.kernel
def find_safe_material_step_scale(
    previous_positions: wp.array(dtype=wp.vec3),
    proposed_positions: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=int, ndim=2),
    inverse_rest_matrix: wp.array(dtype=wp.mat33),
    minimum_volume_ratio: float,
    global_scale: wp.array(dtype=float),
):
    """Limit only substeps that would cross the positive-volume boundary."""
    tet_id = wp.tid()
    i = tet_indices[tet_id, 0]
    j = tet_indices[tet_id, 1]
    k = tet_indices[tet_id, 2]
    l = tet_indices[tet_id, 3]
    rest_inverse = inverse_rest_matrix[tet_id]

    p0 = previous_positions[i]
    p1 = previous_positions[j]
    p2 = previous_positions[k]
    p3 = previous_positions[l]
    q0 = proposed_positions[i]
    q1 = proposed_positions[j]
    q2 = proposed_positions[k]
    q3 = proposed_positions[l]

    previous_matrix = wp.matrix_from_cols(p1 - p0, p2 - p0, p3 - p0)
    previous_ratio = wp.determinant(previous_matrix * rest_inverse)
    proposed_matrix = wp.matrix_from_cols(q1 - q0, q2 - q0, q3 - q0)
    proposed_ratio = wp.determinant(proposed_matrix * rest_inverse)
    if proposed_ratio >= minimum_volume_ratio:
        return
    if previous_ratio <= minimum_volume_ratio:
        if proposed_ratio >= previous_ratio:
            return
        wp.atomic_min(global_scale, 0, 0.0)
        return

    lower = 0.0
    upper = 1.0
    for _ in range(16):
        middle = 0.5 * (lower + upper)
        x0 = p0 + middle * (q0 - p0)
        x1 = p1 + middle * (q1 - p1)
        x2 = p2 + middle * (q2 - p2)
        x3 = p3 + middle * (q3 - p3)
        middle_matrix = wp.matrix_from_cols(
            x1 - x0, x2 - x0, x3 - x0
        )
        middle_ratio = wp.determinant(middle_matrix * rest_inverse)
        if middle_ratio >= minimum_volume_ratio:
            lower = middle
        else:
            upper = middle
    wp.atomic_min(global_scale, 0, lower)


@wp.kernel
def apply_material_step_scale(
    previous_positions: wp.array(dtype=wp.vec3),
    current_positions: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.uint32),
    global_scale: wp.array(dtype=float),
):
    tid = wp.tid()
    if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:
        return
    scale = global_scale[0]
    current_positions[tid] = (
        previous_positions[tid]
        + scale * (current_positions[tid] - previous_positions[tid])
    )


@wp.kernel
def mark_unsafe_material_step_particles(
    previous_positions: wp.array(dtype=wp.vec3),
    proposed_positions: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=int, ndim=2),
    inverse_rest_matrix: wp.array(dtype=wp.mat33),
    minimum_volume_ratio: float,
    unsafe_particles: wp.array(dtype=int),
    unsafe_tet_count: wp.array(dtype=int),
):
    """Mark only vertices around elements that would cross the barrier."""
    tet_id = wp.tid()
    i = tet_indices[tet_id, 0]
    j = tet_indices[tet_id, 1]
    k = tet_indices[tet_id, 2]
    l = tet_indices[tet_id, 3]
    rest_inverse = inverse_rest_matrix[tet_id]

    p0 = previous_positions[i]
    p1 = previous_positions[j]
    p2 = previous_positions[k]
    p3 = previous_positions[l]
    q0 = proposed_positions[i]
    q1 = proposed_positions[j]
    q2 = proposed_positions[k]
    q3 = proposed_positions[l]
    previous_matrix = wp.matrix_from_cols(p1 - p0, p2 - p0, p3 - p0)
    proposed_matrix = wp.matrix_from_cols(q1 - q0, q2 - q0, q3 - q0)
    previous_ratio = wp.determinant(previous_matrix * rest_inverse)
    proposed_ratio = wp.determinant(proposed_matrix * rest_inverse)

    unsafe = proposed_ratio < minimum_volume_ratio
    if previous_ratio <= minimum_volume_ratio:
        unsafe = proposed_ratio < previous_ratio
    if not unsafe:
        return
    wp.atomic_max(unsafe_particles, i, 1)
    wp.atomic_max(unsafe_particles, j, 1)
    wp.atomic_max(unsafe_particles, k, 1)
    wp.atomic_max(unsafe_particles, l, 1)
    wp.atomic_add(unsafe_tet_count, 0, 1)


@wp.kernel
def revert_unsafe_material_step_particles(
    previous_positions: wp.array(dtype=wp.vec3),
    current_positions: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.uint32),
    unsafe_particles: wp.array(dtype=int),
):
    tid = wp.tid()
    if (
        unsafe_particles[tid] != 0
        and (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) != 0
    ):
        current_positions[tid] = previous_positions[tid]


@wp.kernel
def accumulate_unsafe_tet_rigid_translations(
    previous_positions: wp.array(dtype=wp.vec3),
    proposed_positions: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=int, ndim=2),
    inverse_rest_matrix: wp.array(dtype=wp.mat33),
    minimum_volume_ratio: float,
    translation_sums: wp.array(dtype=wp.vec3),
    translation_counts: wp.array(dtype=float),
):
    """Replace only a crossing tet's deformation by its mean translation."""
    tet_id = wp.tid()
    i = tet_indices[tet_id, 0]
    j = tet_indices[tet_id, 1]
    k = tet_indices[tet_id, 2]
    l = tet_indices[tet_id, 3]
    rest_inverse = inverse_rest_matrix[tet_id]

    p0 = previous_positions[i]
    p1 = previous_positions[j]
    p2 = previous_positions[k]
    p3 = previous_positions[l]
    q0 = proposed_positions[i]
    q1 = proposed_positions[j]
    q2 = proposed_positions[k]
    q3 = proposed_positions[l]
    previous_matrix = wp.matrix_from_cols(p1 - p0, p2 - p0, p3 - p0)
    proposed_matrix = wp.matrix_from_cols(q1 - q0, q2 - q0, q3 - q0)
    previous_ratio = wp.determinant(previous_matrix * rest_inverse)
    proposed_ratio = wp.determinant(proposed_matrix * rest_inverse)
    unsafe = proposed_ratio < minimum_volume_ratio
    if previous_ratio <= minimum_volume_ratio:
        unsafe = proposed_ratio < previous_ratio
    if not unsafe:
        return

    translation = 0.25 * (
        (q0 - p0) + (q1 - p1) + (q2 - p2) + (q3 - p3)
    )
    wp.atomic_add(translation_sums, i, translation)
    wp.atomic_add(translation_sums, j, translation)
    wp.atomic_add(translation_sums, k, translation)
    wp.atomic_add(translation_sums, l, translation)
    wp.atomic_add(translation_counts, i, 1.0)
    wp.atomic_add(translation_counts, j, 1.0)
    wp.atomic_add(translation_counts, k, 1.0)
    wp.atomic_add(translation_counts, l, 1.0)


@wp.kernel
def apply_unsafe_tet_rigid_translations(
    previous_positions: wp.array(dtype=wp.vec3),
    current_positions: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.uint32),
    translation_sums: wp.array(dtype=wp.vec3),
    translation_counts: wp.array(dtype=float),
):
    particle_id = wp.tid()
    count = translation_counts[particle_id]
    if (
        count > 0.0
        and (particle_flags[particle_id] & PARTICLE_FLAG_ACTIVE) != 0
    ):
        current_positions[particle_id] = (
            previous_positions[particle_id]
            + translation_sums[particle_id] / count
        )


@wp.kernel
def update_material_particle_velocities(
    predicted_positions: wp.array(dtype=wp.vec3),
    current_positions: wp.array(dtype=wp.vec3),
    predicted_velocities: wp.array(dtype=wp.vec3),
    contact_projection_displacements: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.uint32),
    dt: float,
    velocity_damping: float,
    material_projection_velocity_scale: float,
    contact_projection_velocity_scale: float,
    velocities: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:
        return
    # Material recovery and kinematic contact have different velocity roles.
    # Feeding every distance/volume/shape projection back at full strength
    # reintroduces the old contact-material chatter, while attenuating the
    # complete displacement hides the real velocity transferred by a closing
    # jaw.  Track the accepted contact motion independently and damp only the
    # remaining material correction aggressively.
    total_projection_displacement = (
        current_positions[tid] - predicted_positions[tid]
    )
    contact_displacement = contact_projection_displacements[tid]
    material_displacement = (
        total_projection_displacement - contact_displacement
    )
    velocities[tid] = velocity_damping * (
        predicted_velocities[tid]
        + material_projection_velocity_scale
        * material_displacement
        / dt
        + contact_projection_velocity_scale
        * contact_displacement
        / dt
    )


@wp.kernel
def accumulate_projection_displacements(
    previous_positions: wp.array(dtype=wp.vec3),
    current_positions: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.uint32),
    accumulated_displacements: wp.array(dtype=wp.vec3),
):
    particle_id = wp.tid()
    if (particle_flags[particle_id] & PARTICLE_FLAG_ACTIVE) == 0:
        return
    accumulated_displacements[particle_id] += (
        current_positions[particle_id] - previous_positions[particle_id]
    )


@wp.kernel
def stabilize_particle_ground_velocities(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
    ground: wp.array(dtype=float),
    contact_margin: float,
):
    tid = wp.tid()
    if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0 or inverse_mass[tid] == 0.0:
        return
    normal = wp.vec3(ground[0], ground[1], ground[2])
    distance = (
        wp.dot(normal, positions[tid]) + ground[3] - particle_radius[tid]
    )
    if distance <= contact_margin:
        velocity = velocities[tid]
        normal_velocity = wp.dot(normal, velocity)
        # Restitution is disabled for the SUPER support plane. A particle that
        # ended the positional solve on the plane must not turn penetration
        # correction into artificial normal bounce. A genuine lifting force
        # predicts a position outside the contact margin and is not clamped.
        velocities[tid] = velocity - normal_velocity * normal


class MaterialTetrahedronXPBDProjector:
    """Material-aware particle step used by SUPER soft-tissue work.

    ``neo_hookean`` retains the project-local energy constraint. ``paper``
    selects the distance, fixed-volume, and tetrahedral shape-matching
    constraints used by Liang et al.  Contact and local non-inversion repair
    remain shared so changing the constitutive baseline does not silently
    change the tool geometry.
    """

    def __init__(
        self,
        model: warp.sim.Model,
        iterations: int = 10,
        relaxation: float = 0.25,
        compliance_scale: float = 1.0,
        constraint_model: str = "neo_hookean",
        paper_distance_stiffness: float = 0.2,
        paper_volume_stiffness: float = 1.0e10,
        paper_shape_stiffness: float = 0.005,
    ):
        if model.tet_count <= 0:
            raise ValueError("MaterialTetrahedronXPBDProjector requires tetrahedra")
        self.iterations = int(iterations)
        self.relaxation = float(relaxation)
        self.compliance_scale = float(compliance_scale)
        self.constraint_model = str(constraint_model)
        self.paper_volume_stiffness = float(paper_volume_stiffness)
        self.deltas = wp.zeros_like(model.particle_q)
        tet_indices = model.tet_indices.numpy().astype(
            np.int32, copy=False
        )
        flat_particle_ids = tet_indices.reshape(-1)
        flat_corner_ids = np.arange(
            model.tet_count * 4, dtype=np.int32
        )
        order = np.lexsort((flat_corner_ids, flat_particle_ids))
        sorted_particle_ids = flat_particle_ids[order]
        sorted_corner_ids = flat_corner_ids[order]
        particle_corner_counts = np.bincount(
            sorted_particle_ids, minlength=model.particle_count
        )
        particle_corner_offsets = np.empty(
            model.particle_count + 1, dtype=np.int32
        )
        particle_corner_offsets[0] = 0
        particle_corner_offsets[1:] = np.cumsum(
            particle_corner_counts, dtype=np.int64
        ).astype(np.int32)
        self.material_particle_corner_offsets = wp.array(
            particle_corner_offsets, dtype=int, device=model.device
        )
        self.material_particle_corner_ids = wp.array(
            sorted_corner_ids, dtype=int, device=model.device
        )
        self.material_corner_deltas = wp.zeros(
            model.tet_count * 4, dtype=wp.vec3, device=model.device
        )
        self.particle_shape_deltas = wp.zeros_like(model.particle_q)
        self.particle_shape_delta_scale = wp.ones(
            1, dtype=float, device=model.device
        )
        self.discarded_body_deltas = (
            wp.zeros_like(model.body_qd) if model.body_count else None
        )
        self.previous_positions = wp.zeros_like(model.particle_q)
        self.predicted_positions = wp.zeros_like(model.particle_q)
        self.predicted_velocities = wp.zeros_like(model.particle_q)
        self.contact_pre_projection_positions = wp.zeros_like(
            model.particle_q
        )
        self.contact_projection_displacements = wp.zeros_like(
            model.particle_q
        )
        self.post_contact_pre_material_positions = wp.zeros_like(
            model.particle_q
        )
        self.lambda_energy = wp.zeros(model.tet_count, dtype=float, device=model.device)
        rest_positions = model.particle_q.numpy().astype(np.float64, copy=False)
        edge_indices = np.concatenate(
            (
                tet_indices[:, (0, 1)],
                tet_indices[:, (0, 2)],
                tet_indices[:, (0, 3)],
                tet_indices[:, (1, 2)],
                tet_indices[:, (1, 3)],
                tet_indices[:, (2, 3)],
            ),
            axis=0,
        )
        edge_indices.sort(axis=1)
        edge_indices = np.unique(edge_indices, axis=0).astype(np.int32)
        edge_rest_lengths = np.linalg.norm(
            rest_positions[edge_indices[:, 0]]
            - rest_positions[edge_indices[:, 1]],
            axis=1,
        ).astype(np.float32)
        self.paper_edge_indices = wp.array(
            edge_indices, dtype=int, ndim=2, device=model.device
        )
        self.paper_edge_count = int(len(edge_indices))
        self.paper_edge_rest_lengths = wp.array(
            edge_rest_lengths, dtype=float, device=model.device
        )
        self.paper_distance_stiffness = wp.full(
            model.particle_count,
            float(paper_distance_stiffness),
            dtype=float,
            device=model.device,
        )
        self.paper_shape_stiffness = wp.full(
            model.particle_count,
            float(paper_shape_stiffness),
            dtype=float,
            device=model.device,
        )
        rest_tet_points = rest_positions[tet_indices]
        rest_centers = rest_tet_points.mean(axis=1, keepdims=True)
        rest_relative = (rest_tet_points - rest_centers).reshape(-1, 3)
        rest_volumes = np.linalg.det(
            np.stack(
                (
                    rest_tet_points[:, 1] - rest_tet_points[:, 0],
                    rest_tet_points[:, 2] - rest_tet_points[:, 0],
                    rest_tet_points[:, 3] - rest_tet_points[:, 0],
                ),
                axis=-1,
            )
        ) / 6.0
        if np.any(rest_volumes <= 0.0):
            raise ValueError("Paper XPBD requires positively oriented tetrahedra")
        self.paper_rest_relative_positions = wp.array(
            rest_relative.astype(np.float32),
            dtype=wp.vec3,
            device=model.device,
        )
        self.paper_rest_volumes = wp.array(
            rest_volumes.astype(np.float32),
            dtype=float,
            device=model.device,
        )
        self.paper_lambda_distance = wp.zeros(
            len(edge_indices), dtype=float, device=model.device
        )
        self.paper_lambda_volume = wp.zeros(
            model.tet_count, dtype=float, device=model.device
        )
        self.paper_lambda_shape = wp.zeros(
            model.tet_count, dtype=float, device=model.device
        )
        self.paper_edge_corner_deltas = wp.zeros(
            len(edge_indices) * 2, dtype=wp.vec3, device=model.device
        )

        edge_flat_particles = edge_indices.reshape(-1)
        edge_flat_corners = np.arange(
            len(edge_indices) * 2, dtype=np.int32
        )
        edge_order = np.lexsort((edge_flat_corners, edge_flat_particles))
        edge_sorted_particles = edge_flat_particles[edge_order]
        edge_counts = np.bincount(
            edge_sorted_particles, minlength=model.particle_count
        )
        edge_offsets = np.empty(model.particle_count + 1, dtype=np.int32)
        edge_offsets[0] = 0
        edge_offsets[1:] = np.cumsum(
            edge_counts, dtype=np.int64
        ).astype(np.int32)
        self.paper_particle_edge_corner_offsets = wp.array(
            edge_offsets, dtype=int, device=model.device
        )
        self.paper_particle_edge_corner_ids = wp.array(
            edge_flat_corners[edge_order], dtype=int, device=model.device
        )
        self.material_step_scale = wp.ones(
            1, dtype=float, device=model.device
        )
        self.material_unsafe_particles = wp.zeros(
            model.particle_count, dtype=int, device=model.device
        )
        self.material_unsafe_tet_count = wp.zeros(
            1, dtype=int, device=model.device
        )
        self.material_unsafe_translation_sums = wp.zeros_like(
            model.particle_q
        )
        self.material_unsafe_translation_counts = wp.zeros(
            model.particle_count, dtype=float, device=model.device
        )
        self.configure_constraint_model(
            constraint_model,
            paper_distance_stiffness,
            paper_volume_stiffness,
            paper_shape_stiffness,
        )

    def configure_constraint_model(
        self,
        constraint_model: str,
        paper_distance_stiffness: float,
        paper_volume_stiffness: float,
        paper_shape_stiffness: float,
        preserve_spatial_stiffness: bool = False,
    ) -> None:
        if constraint_model not in {"neo_hookean", "paper"}:
            raise ValueError(
                f"Unknown tetrahedral constraint model: {constraint_model}"
            )
        if (
            paper_distance_stiffness <= 0.0
            or paper_volume_stiffness <= 0.0
            or paper_shape_stiffness <= 0.0
        ):
            raise ValueError("Paper XPBD stiffness values must be positive")
        self.constraint_model = constraint_model
        self.paper_volume_stiffness = float(paper_volume_stiffness)
        if not preserve_spatial_stiffness:
            self.paper_distance_stiffness.fill_(
                float(paper_distance_stiffness)
            )
            self.paper_shape_stiffness.fill_(float(paper_shape_stiffness))

    def capture_previous_positions(self, state_in: warp.sim.State) -> None:
        """Save the substep start before Warp XPBD reuses its state buffers."""
        wp.copy(self.previous_positions, state_in.particle_q)

    def _project_paper_material_iteration(
        self,
        model: warp.sim.Model,
        state_out: warp.sim.State,
        dt: float,
        *,
        solve_ground: bool = False,
        ground_relaxation: float = 0.9,
    ) -> None:
        """Run one distance/volume/shape XPBD pass.

        Keeping this as a single reusable pass lets tool contact alternate with
        material recovery.  In particular, the volume constraint can redirect
        a local downward press laterally before another contact proposal is
        generated, instead of leaving the displayed state as a contact-made
        bowl until the next physics substep.
        """
        self.deltas.zero_()
        self.paper_edge_corner_deltas.zero_()
        wp.launch(
            kernel=solve_paper_distance_constraints_xpbd,
            dim=self.paper_edge_count,
            inputs=[
                state_out.particle_q,
                model.particle_inv_mass,
                self.paper_edge_indices,
                self.paper_edge_rest_lengths,
                self.paper_distance_stiffness,
                dt,
                self.compliance_scale,
                self.relaxation,
                self.paper_lambda_distance,
            ],
            outputs=[self.paper_edge_corner_deltas],
            device=model.device,
        )
        wp.launch(
            kernel=reduce_averaged_constraint_deltas,
            dim=model.particle_count,
            inputs=[
                self.paper_particle_edge_corner_offsets,
                self.paper_particle_edge_corner_ids,
                self.paper_edge_corner_deltas,
            ],
            outputs=[self.deltas],
            device=model.device,
        )
        wp.launch(
            kernel=apply_material_tetrahedron_deltas,
            dim=model.particle_count,
            inputs=[
                state_out.particle_q,
                model.particle_flags,
                self.deltas,
            ],
            device=model.device,
        )

        self.deltas.zero_()
        self.material_corner_deltas.zero_()
        wp.launch(
            kernel=solve_paper_volume_constraints_xpbd,
            dim=model.tet_count,
            inputs=[
                state_out.particle_q,
                model.particle_inv_mass,
                model.tet_indices,
                self.paper_rest_volumes,
                self.paper_volume_stiffness,
                dt,
                self.compliance_scale,
                self.relaxation,
                self.paper_lambda_volume,
            ],
            outputs=[self.material_corner_deltas],
            device=model.device,
        )
        wp.launch(
            kernel=reduce_averaged_constraint_deltas,
            dim=model.particle_count,
            inputs=[
                self.material_particle_corner_offsets,
                self.material_particle_corner_ids,
                self.material_corner_deltas,
            ],
            outputs=[self.deltas],
            device=model.device,
        )
        wp.launch(
            kernel=apply_material_tetrahedron_deltas,
            dim=model.particle_count,
            inputs=[
                state_out.particle_q,
                model.particle_flags,
                self.deltas,
            ],
            device=model.device,
        )

        self.deltas.zero_()
        self.material_corner_deltas.zero_()
        wp.launch(
            kernel=solve_paper_shape_matching_constraints_xpbd,
            dim=model.tet_count,
            inputs=[
                state_out.particle_q,
                model.particle_inv_mass,
                model.tet_indices,
                model.tet_poses,
                self.paper_rest_relative_positions,
                self.paper_shape_stiffness,
                dt,
                self.compliance_scale,
                self.relaxation,
                self.paper_lambda_shape,
            ],
            outputs=[self.material_corner_deltas],
            device=model.device,
        )
        wp.launch(
            kernel=reduce_averaged_constraint_deltas,
            dim=model.particle_count,
            inputs=[
                self.material_particle_corner_offsets,
                self.material_particle_corner_ids,
                self.material_corner_deltas,
            ],
            outputs=[self.deltas],
            device=model.device,
        )
        if solve_ground:
            if not model.ground:
                raise ValueError(
                    "solve_ground=True requires a model ground plane"
                )
            wp.launch(
                kernel=solve_particle_ground_contacts,
                dim=model.particle_count,
                inputs=[
                    state_out.particle_q,
                    state_out.particle_qd,
                    model.particle_inv_mass,
                    model.particle_radius,
                    model.particle_flags,
                    model.soft_contact_ke,
                    model.soft_contact_kd,
                    model.soft_contact_kf,
                    model.soft_contact_mu,
                    model.ground_plane,
                    dt,
                    ground_relaxation,
                ],
                outputs=[self.deltas],
                device=model.device,
            )
        wp.launch(
            kernel=apply_material_tetrahedron_deltas,
            dim=model.particle_count,
            inputs=[
                state_out.particle_q,
                model.particle_flags,
                self.deltas,
            ],
            device=model.device,
        )

    def project_particles(
        self,
        model: warp.sim.Model,
        state_out: warp.sim.State,
        dt: float,
        velocity_damping: float = 1.0,
        solve_ground: bool = False,
        ground_relaxation: float = 0.9,
        solve_particle_shapes: bool = False,
        particle_shape_relaxation: float = 0.9,
        particle_shape_max_correction: float = 0.0,
        particle_shape_min_volume_ratio: float = 0.05,
        solve_triangle_skin_contacts: bool = False,
        triangle_skin_contact_projector=None,
        triangle_skin_contact_margin_m: float = 0.0004,
        triangle_skin_query_distance_m: float = 0.010,
        triangle_skin_ccd_velocity_scale: float = 1.0,
        triangle_skin_friction: float = 0.35,
        triangle_skin_relaxation: float = 0.7,
        triangle_skin_max_correction_m: float = 0.0004,
        triangle_skin_top_barrier_max_correction_m: float = 0.0,
        triangle_skin_contact_iterations: int = 1,
        triangle_skin_post_contact_material_iterations: int = 0,
        triangle_skin_final_barrier_max_correction_m: float = 0.0,
        triangle_skin_min_volume_ratio: float = 0.05,
        triangle_skin_contact_time_scale: float = 1.0,
        material_min_volume_ratio: float = 1.0e-4,
        projection_velocity_scale: float = 1.0,
        contact_projection_velocity_scale: float = 1.0,
        final_ground_passes: int = 2,
    ) -> None:
        """Project material, ground, and optional tool-contact constraints."""
        if (
            solve_triangle_skin_contacts
            and triangle_skin_contact_projector is None
        ):
            raise ValueError(
                "Triangle-skin contact is enabled without a configured projector"
            )
        if not 0.0 <= projection_velocity_scale <= 1.0:
            raise ValueError("projection_velocity_scale must lie in [0, 1]")
        if not 0.0 <= contact_projection_velocity_scale <= 1.0:
            raise ValueError(
                "contact_projection_velocity_scale must lie in [0, 1]"
            )
        top_barrier_max_correction_m = (
            triangle_skin_top_barrier_max_correction_m
            if triangle_skin_top_barrier_max_correction_m > 0.0
            else triangle_skin_max_correction_m
        )
        # Snapshot Warp's unconstrained prediction. The final velocity update
        # uses this state to distinguish true integrated motion from geometric
        # constraint correction.
        wp.copy(self.predicted_positions, state_out.particle_q)
        wp.copy(self.predicted_velocities, state_out.particle_qd)
        self.contact_projection_displacements.zero_()
        self.lambda_energy.zero_()
        self.paper_lambda_distance.zero_()
        self.paper_lambda_volume.zero_()
        self.paper_lambda_shape.zero_()
        for _ in range(self.iterations):
            if self.constraint_model == "paper":
                self._project_paper_material_iteration(
                    model,
                    state_out,
                    dt,
                    solve_ground=solve_ground,
                    ground_relaxation=ground_relaxation,
                )
                continue
            else:
                self.deltas.zero_()
                self.material_corner_deltas.zero_()
                wp.launch(
                    kernel=solve_material_tetrahedra_xpbd,
                    dim=model.tet_count,
                    inputs=[
                        state_out.particle_q,
                        model.particle_inv_mass,
                        model.tet_indices,
                        model.tet_poses,
                        model.tet_materials,
                        dt,
                        self.compliance_scale,
                        self.relaxation,
                        self.lambda_energy,
                    ],
                    outputs=[self.material_corner_deltas],
                    device=model.device,
                )
                wp.launch(
                    kernel=reduce_material_tetrahedron_deltas,
                    dim=model.particle_count,
                    inputs=[
                        self.material_particle_corner_offsets,
                        self.material_particle_corner_ids,
                        self.material_corner_deltas,
                    ],
                    outputs=[self.deltas],
                    device=model.device,
                )
            if solve_ground:
                if not model.ground:
                    raise ValueError("solve_ground=True requires a model ground plane")
                wp.launch(
                    kernel=solve_particle_ground_contacts,
                    dim=model.particle_count,
                    inputs=[
                        state_out.particle_q,
                        state_out.particle_qd,
                        model.particle_inv_mass,
                        model.particle_radius,
                        model.particle_flags,
                        model.soft_contact_ke,
                        model.soft_contact_kd,
                        model.soft_contact_kf,
                        model.soft_contact_mu,
                        model.ground_plane,
                        dt,
                        ground_relaxation,
                    ],
                    outputs=[self.deltas],
                    device=model.device,
                )
            wp.launch(
                kernel=apply_material_tetrahedron_deltas,
                dim=model.particle_count,
                inputs=[
                    state_out.particle_q,
                    model.particle_flags,
                    self.deltas,
                ],
                device=model.device,
            )
        if (
            solve_ground
            or solve_particle_shapes
            or solve_triangle_skin_contacts
        ):
            for final_pass_index in range(final_ground_passes):
                if solve_particle_shapes:
                    self.particle_shape_deltas.zero_()
                    self._accumulate_particle_shape_contact_deltas(
                        model,
                        state_out,
                        dt * triangle_skin_contact_time_scale,
                        particle_shape_relaxation,
                    )
                    wp.launch(
                        kernel=clamp_particle_shape_deltas,
                        dim=model.particle_count,
                        inputs=[
                            self.particle_shape_deltas,
                            particle_shape_max_correction,
                        ],
                        device=model.device,
                    )
                    self.particle_shape_delta_scale.fill_(1.0)
                    wp.launch(
                        kernel=find_safe_particle_shape_delta_scale,
                        dim=model.tet_count,
                        inputs=[
                            state_out.particle_q,
                            self.particle_shape_deltas,
                            model.tet_indices,
                            model.tet_poses,
                            particle_shape_min_volume_ratio,
                            self.particle_shape_delta_scale,
                        ],
                        device=model.device,
                    )
                    wp.launch(
                        kernel=apply_scaled_particle_shape_deltas,
                        dim=model.particle_count,
                        inputs=[
                            state_out.particle_q,
                            model.particle_flags,
                            self.particle_shape_deltas,
                            self.particle_shape_delta_scale,
                        ],
                        device=model.device,
                    )
                if (
                    solve_triangle_skin_contacts
                    and final_pass_index == final_ground_passes - 1
                ):
                    for contact_iteration in range(
                        triangle_skin_contact_iterations
                    ):
                        wp.copy(
                            self.contact_pre_projection_positions,
                            state_out.particle_q,
                        )
                        triangle_skin_contact_projector.project(
                            model,
                            state_out,
                            dt,
                            contact_margin_m=triangle_skin_contact_margin_m,
                            query_distance_m=triangle_skin_query_distance_m,
                            ccd_velocity_scale=(
                                triangle_skin_ccd_velocity_scale
                            ),
                            friction_coefficient=triangle_skin_friction,
                            relaxation=triangle_skin_relaxation,
                            maximum_correction_m=(
                                triangle_skin_max_correction_m
                            ),
                            minimum_volume_ratio=(
                                triangle_skin_min_volume_ratio
                            ),
                            top_barrier_maximum_correction_m=(
                                top_barrier_max_correction_m
                            ),
                            enable_top_pressure_shoulder=False,
                        )
                        wp.launch(
                            kernel=accumulate_projection_displacements,
                            dim=model.particle_count,
                            inputs=[
                                self.contact_pre_projection_positions,
                                state_out.particle_q,
                                model.particle_flags,
                                self.contact_projection_displacements,
                            ],
                            device=model.device,
                        )
                        # Snapshot configuration: the first two full contact
                        # passes recover material; the third remains the full
                        # final barrier that produced the preferred visible press.
                        if (
                            self.constraint_model == "paper"
                            and contact_iteration
                            < triangle_skin_contact_iterations - 1
                        ):
                            for _ in range(
                                triangle_skin_post_contact_material_iterations
                            ):
                                wp.copy(
                                    self.post_contact_pre_material_positions,
                                    state_out.particle_q,
                                )
                                self._project_paper_material_iteration(
                                    model,
                                    state_out,
                                    dt,
                                    solve_ground=False,
                                )
                                # The post-contact pass should restore volume,
                                # but distance/shape corrections on a very thin
                                # boundary tet can occasionally compete with
                                # it. Backtrack only this material pass before
                                # it creates a new low-volume element; do not
                                # freeze or revert the whole contact patch.
                                self.material_step_scale.fill_(1.0)
                                wp.launch(
                                    kernel=find_safe_material_step_scale,
                                    dim=model.tet_count,
                                    inputs=[
                                        self.post_contact_pre_material_positions,
                                        state_out.particle_q,
                                        model.tet_indices,
                                        model.tet_poses,
                                        triangle_skin_min_volume_ratio,
                                        self.material_step_scale,
                                    ],
                                    device=model.device,
                                )
                                wp.launch(
                                    kernel=apply_material_step_scale,
                                    dim=model.particle_count,
                                    inputs=[
                                        self.post_contact_pre_material_positions,
                                        state_out.particle_q,
                                        model.particle_flags,
                                        self.material_step_scale,
                                    ],
                                    device=model.device,
                                )
                    # An explicit post-material shoulder is optional. Skip
                    # the pass entirely when the environment disables it so
                    # contact cannot inject a repeated upward displacement.
                    if (
                        triangle_skin_contact_projector
                        .top_pressure_shoulder_entry_count
                        > 0
                    ):
                        triangle_skin_contact_projector.apply_top_pressure_shoulder(
                            model,
                            state_out,
                            maximum_correction_m=(
                                2.0 * triangle_skin_max_correction_m
                            ),
                            minimum_volume_ratio=(
                                triangle_skin_min_volume_ratio
                            ),
                        )
                    if triangle_skin_final_barrier_max_correction_m > 0.0:
                        wp.copy(
                            self.contact_pre_projection_positions,
                            state_out.particle_q,
                        )
                        triangle_skin_contact_projector.project(
                            model,
                            state_out,
                            dt,
                            contact_margin_m=triangle_skin_contact_margin_m,
                            query_distance_m=triangle_skin_query_distance_m,
                            ccd_velocity_scale=(
                                triangle_skin_ccd_velocity_scale
                            ),
                            friction_coefficient=triangle_skin_friction,
                            relaxation=triangle_skin_relaxation,
                            maximum_correction_m=(
                                triangle_skin_final_barrier_max_correction_m
                            ),
                            minimum_volume_ratio=(
                                triangle_skin_min_volume_ratio
                            ),
                            top_barrier_maximum_correction_m=min(
                                top_barrier_max_correction_m,
                                triangle_skin_final_barrier_max_correction_m,
                            ),
                            enable_top_pressure_shoulder=False,
                        )
                        wp.launch(
                            kernel=accumulate_projection_displacements,
                            dim=model.particle_count,
                            inputs=[
                                self.contact_pre_projection_positions,
                                state_out.particle_q,
                                model.particle_flags,
                                self.contact_projection_displacements,
                            ],
                            device=model.device,
                        )
                self.deltas.zero_()
                if solve_ground:
                    wp.launch(
                        kernel=solve_particle_ground_contacts,
                        dim=model.particle_count,
                        inputs=[
                            state_out.particle_q,
                            state_out.particle_qd,
                            model.particle_inv_mass,
                            model.particle_radius,
                            model.particle_flags,
                            model.soft_contact_ke,
                            model.soft_contact_kd,
                            model.soft_contact_kf,
                            model.soft_contact_mu,
                            model.ground_plane,
                            dt,
                            1.0,
                        ],
                        outputs=[self.deltas],
                        device=model.device,
                    )
                wp.launch(
                    kernel=apply_material_tetrahedron_deltas,
                    dim=model.particle_count,
                    inputs=[
                        state_out.particle_q,
                        model.particle_flags,
                        self.deltas,
                    ],
                    device=model.device,
                )
        if material_min_volume_ratio > 0.0:
            # Keep the mean motion of a crossing element, but discard only
            # that element's inversion-producing relative deformation. This
            # is local topology repair, not a support/spread layer.
            for _ in range(4):
                self.material_unsafe_translation_sums.zero_()
                self.material_unsafe_translation_counts.zero_()
                wp.launch(
                    kernel=accumulate_unsafe_tet_rigid_translations,
                    dim=model.tet_count,
                    inputs=[
                        self.previous_positions,
                        state_out.particle_q,
                        model.tet_indices,
                        model.tet_poses,
                        material_min_volume_ratio,
                        self.material_unsafe_translation_sums,
                        self.material_unsafe_translation_counts,
                    ],
                    device=model.device,
                )
                wp.launch(
                    kernel=apply_unsafe_tet_rigid_translations,
                    dim=model.particle_count,
                    inputs=[
                        self.previous_positions,
                        state_out.particle_q,
                        model.particle_flags,
                        self.material_unsafe_translation_sums,
                        self.material_unsafe_translation_counts,
                    ],
                    device=model.device,
                )
            # If an element is still unsafe, revert only the vertices in its
            # one-ring. Repeat because reverting one shared vertex can expose
            # a neighboring element. This prevents one compressed tet from
            # globally freezing every tissue particle.
            for _ in range(8):
                self.material_unsafe_particles.zero_()
                self.material_unsafe_tet_count.zero_()
                wp.launch(
                    kernel=mark_unsafe_material_step_particles,
                    dim=model.tet_count,
                    inputs=[
                        self.previous_positions,
                        state_out.particle_q,
                        model.tet_indices,
                        model.tet_poses,
                        material_min_volume_ratio,
                    ],
                    outputs=[
                        self.material_unsafe_particles,
                        self.material_unsafe_tet_count,
                    ],
                    device=model.device,
                )
                wp.launch(
                    kernel=revert_unsafe_material_step_particles,
                    dim=model.particle_count,
                    inputs=[
                        self.previous_positions,
                        state_out.particle_q,
                        model.particle_flags,
                        self.material_unsafe_particles,
                    ],
                    device=model.device,
                )
            self.material_step_scale.fill_(1.0)
            # Absolute non-inversion fallback after local repairs.  Unlike the
            # removed 1e-4 global material limiter, this only rejects a residual
            # proposal that would cross the true zero-volume boundary.  Normal
            # compression above 1e-8 is unaffected.
            wp.launch(
                kernel=find_safe_material_step_scale,
                dim=model.tet_count,
                inputs=[
                    self.previous_positions,
                    state_out.particle_q,
                    model.tet_indices,
                    model.tet_poses,
                    1.0e-8,
                    self.material_step_scale,
                ],
                device=model.device,
            )
            wp.launch(
                kernel=apply_material_step_scale,
                dim=model.particle_count,
                inputs=[
                    self.previous_positions,
                    state_out.particle_q,
                    model.particle_flags,
                    self.material_step_scale,
                ],
                device=model.device,
            )
        # Contact detection remains strided because the deforming mesh queries
        # are expensive. Once a grasp is active, project its four anchors after
        # the material-quality repair on every substep. The projector performs
        # a line search over only the incident grip tetrahedra, so attachment
        # safety cannot revert or freeze unrelated material nodes.
        if triangle_skin_contact_projector is not None:
            triangle_skin_contact_projector.apply_persistent_grip(
                model, state_out, dt
            )
        wp.launch(
            kernel=update_material_particle_velocities,
            dim=model.particle_count,
            inputs=[
                self.predicted_positions,
                state_out.particle_q,
                self.predicted_velocities,
                self.contact_projection_displacements,
                model.particle_flags,
                dt,
                velocity_damping,
                projection_velocity_scale,
                contact_projection_velocity_scale,
            ],
            outputs=[state_out.particle_qd],
            device=model.device,
        )
        if solve_ground:
            wp.launch(
                kernel=stabilize_particle_ground_velocities,
                dim=model.particle_count,
                inputs=[
                    state_out.particle_q,
                    state_out.particle_qd,
                    model.particle_inv_mass,
                    model.particle_radius,
                    model.particle_flags,
                    model.ground_plane,
                    1.0e-6,
                ],
                device=model.device,
            )

    def _accumulate_particle_shape_contact_deltas(
        self,
        model: warp.sim.Model,
        state: warp.sim.State,
        dt: float,
        relaxation: float,
    ) -> None:
        if not model.body_count or self.discarded_body_deltas is None:
            raise ValueError("Particle-shape contacts require rigid bodies")
        self.discarded_body_deltas.zero_()
        wp.launch(
            kernel=solve_particle_shape_contacts,
            dim=model.soft_contact_max,
            inputs=[
                state.particle_q,
                state.particle_qd,
                model.particle_inv_mass,
                model.particle_radius,
                model.particle_flags,
                state.body_q,
                state.body_qd,
                model.body_com,
                model.body_inv_mass,
                model.body_inv_inertia,
                model.shape_body,
                model.shape_materials,
                model.soft_contact_mu,
                model.particle_adhesion,
                model.soft_contact_count,
                model.soft_contact_particle,
                model.soft_contact_shape,
                model.soft_contact_body_pos,
                model.soft_contact_body_vel,
                model.soft_contact_normal,
                model.soft_contact_max,
                dt,
                relaxation,
            ],
            outputs=[self.particle_shape_deltas, self.discarded_body_deltas],
            device=model.device,
        )

    def simulate_unconstrained_particles(
        self,
        model: warp.sim.Model,
        state_in: warp.sim.State,
        state_out: warp.sim.State,
        dt: float,
        velocity_damping: float = 1.0,
        solve_ground: bool = False,
        ground_relaxation: float = 0.9,
        material_min_volume_ratio: float = 1.0e-4,
        final_ground_passes: int = 2,
    ) -> None:
        self.capture_previous_positions(state_in)
        wp.launch(
            kernel=integrate_particles,
            dim=model.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_in.particle_f,
                model.particle_inv_mass,
                model.particle_flags,
                model.gravity,
                dt,
                model.particle_max_velocity,
            ],
            outputs=[state_out.particle_q, state_out.particle_qd],
            device=model.device,
        )
        self.project_particles(
            model,
            state_out,
            dt,
            velocity_damping=velocity_damping,
            solve_ground=solve_ground,
            ground_relaxation=ground_relaxation,
            material_min_volume_ratio=material_min_volume_ratio,
            final_ground_passes=final_ground_passes,
        )
