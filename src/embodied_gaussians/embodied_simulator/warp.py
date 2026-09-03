# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import warp as wp


@wp.kernel
def update_gaussians_transforms_kernel(
    gaussian_pos_rel_body: wp.array(dtype=wp.vec3f),  # type: ignore
    gaussian_quats_rel_body: wp.array(dtype=wp.vec4f),  # type: ignore (w, x, y, z)
    gaussian_body_ids: wp.array(dtype=wp.int32),  # type: ignore
    body_transform: wp.array(dtype=wp.transformf),  # type: ignore (xyz, qw qx qy qz),
    gaussian_pos: wp.array(dtype=wp.vec3f),  # type: ignore
    gaussian_quats: wp.array(dtype=wp.vec4f),  # type: ignore (w, x, y, z)
):
    tid = wp.tid()
    body_id = gaussian_body_ids[tid]
    if body_id == -1:
        return
    T_WB = body_transform[body_id]
    p_BG = gaussian_pos_rel_body[tid]
    r_BG = gaussian_quats_rel_body[tid]
    q_BG = wp.quatf(r_BG[1], r_BG[2], r_BG[3], r_BG[0])
    T_BG = wp.transformf(p_BG, q_BG)
    T_WG = wp.transform_multiply(T_WB, T_BG)
    gaussian_pos[tid] = wp.transform_get_translation(T_WG)
    q_WG = wp.transform_get_rotation(T_WG)
    gaussian_quats[tid] = wp.vec4(q_WG[3], q_WG[0], q_WG[1], q_WG[2])  # type: ignore


@wp.kernel
def update_soft_gaussians_transforms_kernel(
    soft_gaussian_ids: wp.array(dtype=wp.int32),  # type: ignore
    particle_indices: wp.array(dtype=wp.int32, ndim=2),  # type: ignore
    tet_ids: wp.array(dtype=wp.int32),  # type: ignore
    barycentric_weights: wp.array(dtype=wp.float32, ndim=2),  # type: ignore
    rest_offsets: wp.array(dtype=wp.vec3f),  # type: ignore
    binding_modes: wp.array(dtype=wp.int32),  # type: ignore
    face_particle_indices: wp.array(dtype=wp.int32, ndim=2),  # type: ignore
    rest_face_frames: wp.array(dtype=wp.mat33f),  # type: ignore
    visual_vertex_particle_indices: wp.array(dtype=wp.int32, ndim=2),  # type: ignore
    visual_vertex_weights: wp.array(dtype=wp.float32, ndim=2),  # type: ignore
    visual_vertex_rest_offsets: wp.array(dtype=wp.vec3f, ndim=2),  # type: ignore
    visual_vertex_rest_physical_frames: wp.array(dtype=wp.mat33f, ndim=2),  # type: ignore
    rest_visual_face_frames: wp.array(dtype=wp.mat33f),  # type: ignore
    rest_visual_face_poses: wp.array(dtype=wp.mat33f),  # type: ignore
    rest_quats_wxyz: wp.array(dtype=wp.vec4f),  # type: ignore
    rest_scales: wp.array(dtype=wp.vec3f),  # type: ignore
    particle_q: wp.array(dtype=wp.vec3f),  # type: ignore
    tet_rest_poses: wp.array(dtype=wp.mat33f),  # type: ignore
    gaussian_pos: wp.array(dtype=wp.vec3f),  # type: ignore
    gaussian_quats_wxyz: wp.array(dtype=wp.vec4f),  # type: ignore
    gaussian_scale_log: wp.array(dtype=wp.vec3f),  # type: ignore
):
    """Skin one Gaussian to a tetrahedron or an embedded visual triangle.

    Mode 2 reconstructs the three vertices of a high-resolution visual face
    from physical boundary faces, then places the Gaussian at their exact
    centroid.  Its complete rest covariance is transported by the visual
    triangle deformation, so tangential stretch and shear update both the
    rendered orientation and footprint.
    """
    tid = wp.tid()
    gaussian_id = soft_gaussian_ids[tid]
    i0 = particle_indices[tid, 0]
    i1 = particle_indices[tid, 1]
    i2 = particle_indices[tid, 2]
    i3 = particle_indices[tid, 3]
    x0 = particle_q[i0]
    x1 = particle_q[i1]
    x2 = particle_q[i2]
    x3 = particle_q[i3]

    base = (
        barycentric_weights[tid, 0] * x0
        + barycentric_weights[tid, 1] * x1
        + barycentric_weights[tid, 2] * x2
        + barycentric_weights[tid, 3] * x3
    )
    R = wp.mat33f()
    covariance_rotation = wp.mat33f()
    covariance_scales = wp.vec3f()
    if binding_modes[tid] == 2:
        visual_0 = wp.vec3f()
        visual_1 = wp.vec3f()
        visual_2 = wp.vec3f()
        for visual_corner in range(3):
            support = visual_corner * 3
            p0 = particle_q[visual_vertex_particle_indices[tid, support]]
            p1 = particle_q[visual_vertex_particle_indices[tid, support + 1]]
            p2 = particle_q[visual_vertex_particle_indices[tid, support + 2]]
            physical_edge_x = p1 - p0
            physical_raw_normal = wp.cross(physical_edge_x, p2 - p0)
            if (
                wp.length(physical_edge_x) <= 1.0e-8
                or wp.length(physical_raw_normal) <= 1.0e-12
            ):
                return
            physical_tangent_x = wp.normalize(physical_edge_x)
            physical_normal = wp.normalize(physical_raw_normal)
            physical_tangent_y = wp.normalize(
                wp.cross(physical_normal, physical_tangent_x)
            )
            current_physical_frame = wp.matrix_from_cols(
                physical_tangent_x, physical_tangent_y, physical_normal
            )
            physical_rotation = current_physical_frame * wp.transpose(
                visual_vertex_rest_physical_frames[tid, visual_corner]
            )
            visual_position = (
                visual_vertex_weights[tid, support] * p0
                + visual_vertex_weights[tid, support + 1] * p1
                + visual_vertex_weights[tid, support + 2] * p2
                + physical_rotation
                * visual_vertex_rest_offsets[tid, visual_corner]
            )
            if visual_corner == 0:
                visual_0 = visual_position
            elif visual_corner == 1:
                visual_1 = visual_position
            else:
                visual_2 = visual_position

        base = (visual_0 + visual_1 + visual_2) / 3.0
        visual_edge_x = visual_1 - visual_0
        visual_raw_normal = wp.cross(visual_edge_x, visual_2 - visual_0)
        if (
            wp.length(visual_edge_x) <= 1.0e-8
            or wp.length(visual_raw_normal) <= 1.0e-12
        ):
            return
        visual_tangent_x = wp.normalize(visual_edge_x)
        visual_normal = wp.normalize(visual_raw_normal)
        visual_tangent_y = wp.normalize(
            wp.cross(visual_normal, visual_tangent_x)
        )
        current_visual_frame = wp.matrix_from_cols(
            visual_tangent_x, visual_tangent_y, visual_normal
        )
        R = current_visual_frame * wp.transpose(
            rest_visual_face_frames[tid]
        )
        current_visual_shape = wp.matrix_from_cols(
            visual_edge_x, visual_2 - visual_0, visual_normal
        )
        surface_deformation = (
            current_visual_shape * rest_visual_face_poses[tid]
        )
        rest = rest_quats_wxyz[gaussian_id]
        q_rest = wp.quatf(rest[1], rest[2], rest[3], rest[0])
        deformed_axes = (
            surface_deformation
            * wp.quat_to_matrix(q_rest)
            * wp.diag(rest_scales[gaussian_id])
        )
        covariance_basis = wp.mat33f()
        wp.svd3(
            deformed_axes,
            covariance_rotation,
            covariance_scales,
            covariance_basis,
        )
        if wp.determinant(covariance_rotation) < 0.0:
            reflection_fix = wp.mat33f(
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, -1.0,
            )
            covariance_rotation = covariance_rotation * reflection_fix
    elif binding_modes[tid] == 1:
        f0 = particle_q[face_particle_indices[tid, 0]]
        f1 = particle_q[face_particle_indices[tid, 1]]
        f2 = particle_q[face_particle_indices[tid, 2]]
        edge_x = f1 - f0
        raw_normal = wp.cross(edge_x, f2 - f0)
        if wp.length(edge_x) <= 1.0e-8 or wp.length(raw_normal) <= 1.0e-12:
            # Keep the previous render pose for a degenerate face.
            return
        tangent_x = wp.normalize(edge_x)
        normal = wp.normalize(raw_normal)
        tangent_y = wp.normalize(wp.cross(normal, tangent_x))
        current_face_frame = wp.matrix_from_cols(
            tangent_x, tangent_y, normal
        )
        R = current_face_frame * wp.transpose(rest_face_frames[tid])
    else:
        Ds = wp.matrix_from_cols(x1 - x0, x2 - x0, x3 - x0)
        F = Ds * tet_rest_poses[tet_ids[tid]]
        if wp.determinant(F) <= 1.0e-8:
            # Keep the previous render pose for a degenerate or inverted element.
            return

        U = wp.mat33f()
        sigma = wp.vec3f()
        V = wp.mat33f()
        wp.svd3(F, U, sigma, V)
        R = U * wp.transpose(V)
    gaussian_pos[gaussian_id] = base + R * rest_offsets[tid]

    if binding_modes[tid] == 2:
        q_current = wp.normalize(wp.quat_from_matrix(covariance_rotation))
        gaussian_scale_log[gaussian_id] = wp.vec3f(
            wp.log(wp.max(wp.abs(covariance_scales[0]), 1.0e-12)),
            wp.log(wp.max(wp.abs(covariance_scales[1]), 1.0e-12)),
            wp.log(wp.max(wp.abs(covariance_scales[2]), 1.0e-12)),
        )
    else:
        q_delta = wp.quat_from_matrix(R)
        rest = rest_quats_wxyz[gaussian_id]
        q_rest = wp.quatf(rest[1], rest[2], rest[3], rest[0])
        q_current = wp.normalize(wp.mul(q_delta, q_rest))
    gaussian_quats_wxyz[gaussian_id] = wp.vec4f(
        q_current[3], q_current[0], q_current[1], q_current[2]
    )


@wp.kernel
def scatter_soft_gaussian_forces_kernel(
    soft_gaussian_ids: wp.array(dtype=wp.int32),  # type: ignore
    particle_indices: wp.array(dtype=wp.int32, ndim=2),  # type: ignore
    barycentric_weights: wp.array(dtype=wp.float32, ndim=2),  # type: ignore
    particle_weight_sums: wp.array(dtype=wp.float32),  # type: ignore
    particle_inverse_mass: wp.array(dtype=wp.float32),  # type: ignore
    gaussian_opacities: wp.array(dtype=wp.float32),  # type: ignore
    gaussian_displacements: wp.array(dtype=wp.vec3f),  # type: ignore
    kp: float,
    max_gaussian_force: float,
    particle_forces: wp.array(dtype=wp.vec3f),  # type: ignore
):
    """Scatter one soft Gaussian target force to its four bound particles."""
    tid = wp.tid()
    gaussian_id = soft_gaussian_ids[tid]
    force = (
        gaussian_opacities[gaussian_id]
        * kp
        * gaussian_displacements[gaussian_id]
    )
    force_norm = wp.length(force)
    if max_gaussian_force > 0.0 and force_norm > max_gaussian_force:
        force = force * (max_gaussian_force / force_norm)

    for corner in range(4):
        particle_id = particle_indices[tid, corner]
        support = particle_weight_sums[particle_id]
        if particle_inverse_mass[particle_id] > 0.0 and support > 1.0e-8:
            normalized_weight = barycentric_weights[tid, corner] / support
            wp.atomic_add(
                particle_forces,
                particle_id,
                normalized_weight * force,
            )


@wp.kernel
def seed_soft_particle_force_spread_kernel(
    direct_forces: wp.array(dtype=wp.vec3f),  # type: ignore
    spread_forces: wp.array(dtype=wp.vec3f),  # type: ignore
    spread_weights: wp.array(dtype=wp.float32),  # type: ignore
):
    tid = wp.tid()
    force = direct_forces[tid]
    if wp.dot(force, force) > 1.0e-24:
        spread_forces[tid] = force
        spread_weights[tid] = 1.0
    else:
        spread_forces[tid] = wp.vec3f()
        spread_weights[tid] = 0.0


@wp.kernel
def accumulate_soft_particle_force_spread_kernel(
    tet_indices: wp.array(dtype=wp.int32, ndim=2),  # type: ignore
    inverse_mass: wp.array(dtype=wp.float32),  # type: ignore
    input_forces: wp.array(dtype=wp.vec3f),  # type: ignore
    input_weights: wp.array(dtype=wp.float32),  # type: ignore
    output_forces: wp.array(dtype=wp.vec3f),  # type: ignore
    output_weights: wp.array(dtype=wp.float32),  # type: ignore
):
    """Diffuse one layer as acceleration so unequal nodal masses move together."""
    tet_id = wp.tid()
    i0 = tet_indices[tet_id, 0]
    i1 = tet_indices[tet_id, 1]
    i2 = tet_indices[tet_id, 2]
    i3 = tet_indices[tet_id, 3]
    average_acceleration = wp.vec3f()
    source_count = 0.0
    if input_weights[i0] > 0.0 and inverse_mass[i0] > 0.0:
        average_acceleration += input_forces[i0] * inverse_mass[i0]
        source_count += 1.0
    if input_weights[i1] > 0.0 and inverse_mass[i1] > 0.0:
        average_acceleration += input_forces[i1] * inverse_mass[i1]
        source_count += 1.0
    if input_weights[i2] > 0.0 and inverse_mass[i2] > 0.0:
        average_acceleration += input_forces[i2] * inverse_mass[i2]
        source_count += 1.0
    if input_weights[i3] > 0.0 and inverse_mass[i3] > 0.0:
        average_acceleration += input_forces[i3] * inverse_mass[i3]
        source_count += 1.0
    if source_count <= 0.0:
        return
    average_acceleration /= source_count

    if inverse_mass[i0] > 0.0:
        wp.atomic_add(output_forces, i0, average_acceleration / inverse_mass[i0])
        wp.atomic_add(output_weights, i0, 1.0)
    if inverse_mass[i1] > 0.0:
        wp.atomic_add(output_forces, i1, average_acceleration / inverse_mass[i1])
        wp.atomic_add(output_weights, i1, 1.0)
    if inverse_mass[i2] > 0.0:
        wp.atomic_add(output_forces, i2, average_acceleration / inverse_mass[i2])
        wp.atomic_add(output_weights, i2, 1.0)
    if inverse_mass[i3] > 0.0:
        wp.atomic_add(output_forces, i3, average_acceleration / inverse_mass[i3])
        wp.atomic_add(output_weights, i3, 1.0)


@wp.kernel
def normalize_soft_particle_force_spread_kernel(
    direct_forces: wp.array(dtype=wp.vec3f),  # type: ignore
    spread_forces: wp.array(dtype=wp.vec3f),  # type: ignore
    spread_weights: wp.array(dtype=wp.float32),  # type: ignore
):
    tid = wp.tid()
    direct = direct_forces[tid]
    if wp.dot(direct, direct) > 1.0e-24:
        spread_forces[tid] = direct
        spread_weights[tid] = 1.0
        return
    weight = spread_weights[tid]
    if weight > 0.0:
        spread_forces[tid] /= weight
        spread_weights[tid] = 1.0


@wp.kernel
def copy_soft_particle_force_spread_kernel(
    spread_forces: wp.array(dtype=wp.vec3f),  # type: ignore
    particle_forces: wp.array(dtype=wp.vec3f),  # type: ignore
):
    particle_forces[wp.tid()] = spread_forces[wp.tid()]


@wp.kernel
def clamp_soft_particle_forces_kernel(
    particle_inverse_mass: wp.array(dtype=wp.float32),  # type: ignore
    max_particle_force: float,
    max_particle_acceleration: float,
    particle_forces: wp.array(dtype=wp.vec3f),  # type: ignore
):
    tid = wp.tid()
    inverse_mass = particle_inverse_mass[tid]
    if inverse_mass == 0.0:
        particle_forces[tid] = wp.vec3f(0.0, 0.0, 0.0)
        return
    force = particle_forces[tid]
    force_norm = wp.length(force)
    force_limit = max_particle_force
    if max_particle_acceleration > 0.0:
        acceleration_force_limit = max_particle_acceleration / inverse_mass
        if force_limit <= 0.0 or acceleration_force_limit < force_limit:
            force_limit = acceleration_force_limit
    if force_limit > 0.0 and force_norm > force_limit:
        particle_forces[tid] = force * (force_limit / force_norm)


@wp.kernel
def apply_soft_particle_forces_kernel(
    particle_force_scale: wp.array(dtype=wp.float32),  # type: ignore
    particle_inverse_mass: wp.array(dtype=wp.float32),  # type: ignore
    soft_particle_forces: wp.array(dtype=wp.vec3f),  # type: ignore
    particle_forces: wp.array(dtype=wp.vec3f),  # type: ignore
):
    tid = wp.tid()
    if particle_inverse_mass[tid] > 0.0:
        particle_forces[tid] = (
            particle_forces[tid]
            + particle_force_scale[0] * soft_particle_forces[tid]
        )


@wp.kernel
def update_visual_forces_kernel(
    kp: float,
    old_means: wp.array(dtype=wp.vec3f),  # type: ignore
    old_quats: wp.array(dtype=wp.quatf),  # type: ignore
    old_opacities: wp.array(dtype=wp.float32),  # type: ignore
    new_means: wp.array(dtype=wp.vec3f),  # type: ignore
    new_quats: wp.array(dtype=wp.quatf),  # type: ignore
    body_ids: wp.array(dtype=wp.int32),  # type: ignore
    body_q: wp.array(dtype=wp.transformf),  # type: ignore
    forces: wp.array(dtype=wp.vec3f),  # type: ignore
    moments: wp.array(dtype=wp.vec3f),  # type: ignore
):
    tid = wp.tid()
    body_id = body_ids[tid]
    if body_id == -1:
        return
    opacity = old_opacities[tid]
    displacement = new_means[tid] - old_means[tid]
    q_WO = old_quats[tid]
    q_WN = new_quats[tid]
    q_NW = wp.quat_inverse(q_WN)
    q_NO = wp.mul(q_NW, q_WO)
    axis = wp.vec3()
    angle = wp.float32(0.0)  # type: ignore
    wp.quat_to_axis_angle(q_NO, axis, angle)
    T_WB = body_q[body_id]
    com = wp.transform_get_translation(T_WB)
    force = opacity * kp * displacement
    r = old_means[tid] - com
    moment_from_force = wp.cross(r, force)
    moment = moment_from_force
    forces[tid] = force
    moments[tid] = moment


@wp.kernel
def apply_forces_kernel(
    dt: float,
    total_force: wp.array(dtype=wp.vec3f),  # type: ignore
    total_moment: wp.array(dtype=wp.vec3f),  # type: ignore
    body_ids: wp.array(dtype=wp.int32),  # type: ignore
    gaussian_counts: wp.array(dtype=wp.int32),  # type: ignore
    apply_physics_forces: wp.array(dtype=wp.int32),  # type: ignore
    normalize_by_gaussian_count: int,
    max_force: float,
    max_moment: float,
    body_f: wp.array(dtype=wp.spatial_vectorf),  # type: ignore
):
    tid = wp.tid()
    bid = body_ids[tid]
    if apply_physics_forces[tid] != 0:
        force = total_force[tid]
        moment = total_moment[tid]
        if normalize_by_gaussian_count != 0 and gaussian_counts[tid] > 0:
            divisor = float(gaussian_counts[tid])
            force = force / divisor
            moment = moment / divisor
        force_norm = wp.length(force)
        if max_force > 0.0 and force_norm > max_force:
            force = force * (max_force / force_norm)
        moment_norm = wp.length(moment)
        if max_moment > 0.0 and moment_norm > max_moment:
            moment = moment * (max_moment / moment_norm)
        body_f[bid] = wp.spatial_vector(moment, force)  # type: ignore
