"""CPU gates for bilateral four-point jaw-frame tissue controls u_t."""

from __future__ import annotations

import json

import numpy as np
import warp as wp

from embodied_gaussians.physics_simulator.triangle_skin_contact import (
    advance_persistent_grip_support_generation,
    accumulate_persistent_grip_contact_patch,
    accumulate_persistent_grip_constraint_deltas,
    accumulate_triangle_skin_contact_spread,
    apply_persistent_grip_jaw_deltas,
    apply_triangle_skin_contact_deltas,
    capture_or_release_persistent_grip_particles,
    filter_persistent_grip_jaw_deltas,
    find_safe_persistent_grip_jaw_scale,
    normalize_triangle_skin_contact_spread,
    propagate_persistent_grip_support_particles,
    remove_persistent_grip_direct_from_support,
    select_nearest_persistent_grip_surface_particles,
    update_persistent_grip_state,
)


def main() -> None:
    wp.config.kernel_cache_dir = "/tmp/warp-super-tissue-persistent-grip-test"
    wp.init()
    device = "cpu"
    particle_count = 6
    positions_np = np.asarray(
        [
            [-0.0002, 0.0000, 0.0000],
            [+0.0002, 0.0000, 0.0000],
            [0.0000, +0.0002, 0.0000],
            [0.0000, -0.0002, 0.0000],
            [0.0040, 0.0000, 0.0000],
            [0.0050, 0.0000, 0.0000],
        ],
        dtype=np.float32,
    )
    positions = wp.array(positions_np, dtype=wp.vec3, device=device)
    inverse_mass = wp.ones(particle_count, dtype=float, device=device)
    flags = wp.full(
        particle_count, value=1, dtype=wp.uint32, device=device
    )
    skin_nodes = wp.array(
        np.arange(particle_count, dtype=np.int32), dtype=int, device=device
    )
    # The two jaws may touch opposite sides of the patch without sharing one
    # exact mesh vertex. Their two centroids still define one coherent grasp.
    jaw_a_counts = wp.array([3, 0, 2, 0, 1, 0], dtype=int, device=device)
    jaw_b_counts = wp.array([0, 3, 0, 2, 0, 1], dtype=int, device=device)
    patch_sums = wp.zeros(8, dtype=float, device=device)
    shared_node_count = wp.zeros(1, dtype=int, device=device)
    wp.launch(
        accumulate_persistent_grip_contact_patch,
        particle_count,
        inputs=[
            positions,
            inverse_mass,
            flags,
            jaw_a_counts,
            jaw_b_counts,
        ],
        outputs=[patch_sums, shared_node_count],
        device=device,
    )
    grasp_center = wp.zeros(1, dtype=wp.vec3, device=device)
    patch_separation = wp.zeros(1, dtype=float, device=device)
    selected_ids = wp.full(4, value=-1, dtype=int, device=device)
    selected_count = wp.zeros(1, dtype=int, device=device)
    particle_minimum_ratios = wp.ones(
        particle_count, dtype=float, device=device
    )
    wp.launch(
        select_nearest_persistent_grip_surface_particles,
        1,
        inputs=[
            positions,
            inverse_mass,
            flags,
            skin_nodes,
            particle_count,
            jaw_a_counts,
            jaw_b_counts,
            patch_sums,
            particle_minimum_ratios,
            0.01,
        ],
        outputs=[
            grasp_center,
            patch_separation,
            selected_ids,
            selected_count,
        ],
        device=device,
    )
    selected_ids_np = selected_ids.numpy()
    two_contacted_nodes_are_selected_per_jaw = bool(
        set(selected_ids_np[:2].tolist()) == {0, 2}
        and set(selected_ids_np[2:].tolist()) == {1, 3}
        and int(selected_count.numpy()[0]) == 4
    )
    # A closer node whose incident tetrahedron is already a sliver must not
    # become a grasp anchor and globally stall the four-anchor no-flip search.
    filtered_ratios = wp.array(
        [0.001, 1.0, 1.0, 1.0, 1.0, 1.0],
        dtype=float,
        device=device,
    )
    filtered_ids = wp.full(4, value=-1, dtype=int, device=device)
    filtered_count = wp.zeros(1, dtype=int, device=device)
    wp.launch(
        select_nearest_persistent_grip_surface_particles,
        1,
        inputs=[
            positions,
            inverse_mass,
            flags,
            skin_nodes,
            particle_count,
            jaw_a_counts,
            jaw_b_counts,
            patch_sums,
            filtered_ratios,
            0.01,
        ],
        outputs=[
            grasp_center,
            patch_separation,
            filtered_ids,
            filtered_count,
        ],
        device=device,
    )
    collapsed_nearest_node_is_replaced = bool(
        set(filtered_ids.numpy().tolist()) == {1, 2, 3, 4}
        and int(filtered_count.numpy()[0]) == 4
    )

    contact_counts = wp.array([12, 13], dtype=int, device=device)
    maximum_penetration = wp.array([0.001], dtype=float, device=device)
    state = wp.zeros(3, dtype=int, device=device)
    angle = wp.array([0.05], dtype=float, device=device)
    timestamp = wp.array([18.25], dtype=float, device=device)
    capture_allowed = wp.ones(1, dtype=int, device=device)
    release_requested = wp.zeros(1, dtype=int, device=device)
    capture_angle = wp.zeros(1, dtype=float, device=device)
    capture_timestamp = wp.zeros(1, dtype=float, device=device)
    update_inputs = [
        contact_counts,
        0,
        1,
        selected_count,
        patch_separation,
        maximum_penetration,
        8,
        0.006,
        3,
        0.010,
        0.08,
        angle,
        timestamp,
        capture_allowed,
        release_requested,
    ]
    for _ in range(2):
        wp.launch(
            update_persistent_grip_state,
            1,
            inputs=update_inputs,
            outputs=[state, capture_angle, capture_timestamp],
            device=device,
        )
    two_solves_do_not_capture = bool(
        int(state.numpy()[0]) == 0 and int(state.numpy()[1]) == 2
    )
    wp.launch(
        update_persistent_grip_state,
        1,
        inputs=update_inputs,
        outputs=[state, capture_angle, capture_timestamp],
        device=device,
    )
    third_solve_captures = bool(
        int(state.numpy()[0]) == 1
        and np.isclose(capture_angle.numpy()[0], 0.05)
        and np.isclose(capture_timestamp.numpy()[0], 18.25)
    )

    body_q = wp.array(
        [
            wp.transform(wp.vec3(-0.001, 0.0, 0.0), wp.quat_identity()),
            wp.transform(wp.vec3(+0.001, 0.0, 0.0), wp.quat_identity()),
            wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        ],
        dtype=wp.transform,
        device=device,
    )
    bodies = wp.full(particle_count, value=-1, dtype=int, device=device)
    local = wp.zeros(particle_count, dtype=wp.vec3, device=device)
    direct = wp.zeros(particle_count, dtype=int, device=device)
    weights = wp.zeros(particle_count, dtype=float, device=device)
    levels = wp.full(particle_count, value=-1, dtype=int, device=device)
    wp.launch(
        capture_or_release_persistent_grip_particles,
        particle_count,
        inputs=[
            positions,
            inverse_mass,
            flags,
            body_q,
            0,
            1,
            selected_ids,
            state,
        ],
        outputs=[bodies, local, direct, weights, levels],
        device=device,
    )
    captured_ids = np.flatnonzero(direct.numpy() != 0)
    captured_body_ids = bodies.numpy()
    two_anchors_use_each_jaw_frame = bool(
        np.all(captured_body_ids[selected_ids_np[:2]] == 0)
        and np.all(captured_body_ids[selected_ids_np[2:]] == 1)
    )

    # One nearby outside particle receives a half-weight compliant target from
    # jaw A. A second-hop particle is excluded because the configured support
    # uses exactly one generation.
    support_offsets = wp.array(
        [0, 0, 0, 0, 0, 1, 2], dtype=int, device=device
    )
    support_sources = wp.array([0, 4], dtype=int, device=device)
    support_weights = wp.array([0.5, 0.5], dtype=float, device=device)
    support_generation = wp.zeros(1, dtype=int, device=device)
    wp.launch(
        advance_persistent_grip_support_generation,
        1,
        inputs=[state, 1],
        outputs=[support_generation],
        device=device,
    )
    wp.launch(
        propagate_persistent_grip_support_particles,
        particle_count,
        inputs=[
            positions,
            inverse_mass,
            flags,
            body_q,
            2,
            state,
            support_offsets,
            support_sources,
            support_weights,
            direct,
            support_generation,
            levels,
        ],
        outputs=[bodies, local, weights],
        device=device,
    )
    support_body_ids = bodies.numpy()
    one_local_support_ring_is_captured = bool(
        support_body_ids[4] == 2
        and support_body_ids[5] == -1
        and direct.numpy()[4] == 0
        and np.isclose(weights.numpy()[4], 0.5)
    )

    moved_body_q = wp.array(
        [
            wp.transform(wp.vec3(-0.001, 0.0, 0.002), wp.quat_identity()),
            wp.transform(wp.vec3(+0.001, 0.0, 0.002), wp.quat_identity()),
            wp.transform(wp.vec3(0.0, 0.0, 0.002), wp.quat_identity()),
        ],
        dtype=wp.transform,
        device=device,
    )
    grip_deltas = wp.zeros(particle_count, dtype=wp.vec3, device=device)
    grip_delta_counts = wp.zeros(particle_count, dtype=float, device=device)
    wp.launch(
        accumulate_persistent_grip_constraint_deltas,
        particle_count,
        inputs=[
            positions,
            inverse_mass,
            flags,
            moved_body_q,
            state,
            bodies,
            local,
            weights,
            0.01,
            0.0,
            1.0,
            0.003,
        ],
        outputs=[grip_deltas, grip_delta_counts],
        device=device,
    )
    global_scale = wp.ones(1, dtype=float, device=device)
    wp.launch(
        apply_triangle_skin_contact_deltas,
        particle_count,
        inputs=[
            positions,
            inverse_mass,
            flags,
            grip_deltas,
            grip_delta_counts,
            global_scale,
        ],
        device=device,
    )
    followed = positions.numpy().copy()
    direct_and_soft_support_follow_common_jaw_motion = bool(
        np.array_equal(np.sort(captured_ids), np.arange(4))
        and np.allclose(followed[:4, 2], 0.002, atol=1.0e-7)
        and np.isclose(followed[4, 2], 0.001, atol=1.0e-7)
        and np.isclose(followed[5, 2], 0.0, atol=1.0e-7)
    )

    differential_body_q = wp.array(
        [
            wp.transform(wp.vec3(-0.0015, 0.0, 0.002), wp.quat_identity()),
            wp.transform(wp.vec3(+0.0015, 0.0, 0.002), wp.quat_identity()),
            wp.transform(wp.vec3(0.0, 0.0, 0.002), wp.quat_identity()),
        ],
        dtype=wp.transform,
        device=device,
    )
    grip_deltas.zero_()
    grip_delta_counts.zero_()
    wp.launch(
        accumulate_persistent_grip_constraint_deltas,
        particle_count,
        inputs=[
            positions,
            inverse_mass,
            flags,
            differential_body_q,
            state,
            bodies,
            local,
            weights,
            0.01,
            0.0,
            1.0,
            0.003,
        ],
        outputs=[grip_deltas, grip_delta_counts],
        device=device,
    )
    wp.launch(
        apply_triangle_skin_contact_deltas,
        particle_count,
        inputs=[
            positions,
            inverse_mass,
            flags,
            grip_deltas,
            grip_delta_counts,
            global_scale,
        ],
        device=device,
    )
    differentially_followed = positions.numpy()
    per_jaw_closure_targets_are_distinct = bool(
        np.allclose(
            differentially_followed[selected_ids_np[:2], 0],
            followed[selected_ids_np[:2], 0] - 0.0005,
            atol=1.0e-7,
        )
        and np.allclose(
            differentially_followed[selected_ids_np[2:], 0],
            followed[selected_ids_np[2:], 0] + 0.0005,
            atol=1.0e-7,
        )
    )
    local_support_ignores_q7_closure = bool(
        np.isclose(
            differentially_followed[4, 0], followed[4, 0], atol=1.0e-7
        )
    )

    # A collapsing proposal under jaw A must not reduce the independent,
    # volume-safe jaw B control. The sequential second solve sees A's accepted
    # position, so this also gates the combined result rather than two isolated
    # scale calculations.
    safety_positions_np = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [3.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    safety_positions = wp.array(
        safety_positions_np, dtype=wp.vec3, device=device
    )
    safety_inverse_mass = wp.ones(8, dtype=float, device=device)
    safety_flags = wp.full(8, value=1, dtype=wp.uint32, device=device)
    safety_deltas_np = np.zeros((8, 3), dtype=np.float32)
    safety_deltas_np[3, 2] = -2.0
    safety_deltas_np[7, 2] = +0.1
    safety_deltas = wp.array(
        safety_deltas_np, dtype=wp.vec3, device=device
    )
    safety_counts = wp.array(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        dtype=float,
        device=device,
    )
    safety_bodies = wp.array(
        [-1, -1, -1, 10, -1, -1, -1, 11],
        dtype=int,
        device=device,
    )
    safety_tets_np = np.asarray(
        [[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32
    )
    safety_tets = wp.array(safety_tets_np, dtype=int, device=device)
    safety_rest_inverse = wp.array(
        np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
        dtype=wp.mat33,
        device=device,
    )
    safety_tet_ids = wp.array([0, 1], dtype=int, device=device)
    jaw_scales = wp.ones(2, dtype=float, device=device)
    for jaw_index, jaw_body_id in enumerate((10, 11)):
        wp.launch(
            find_safe_persistent_grip_jaw_scale,
            2,
            inputs=[
                safety_positions,
                safety_deltas,
                safety_bodies,
                jaw_body_id,
                safety_tets,
                safety_rest_inverse,
                safety_tet_ids,
                0.03,
                jaw_index,
                jaw_scales,
            ],
            device=device,
        )
        wp.launch(
            apply_persistent_grip_jaw_deltas,
            8,
            inputs=[
                safety_positions,
                safety_inverse_mass,
                safety_flags,
                safety_deltas,
                safety_counts,
                safety_bodies,
                jaw_body_id,
                jaw_index,
                jaw_scales,
            ],
            device=device,
        )
    safety_result = safety_positions.numpy()
    safety_scales = jaw_scales.numpy()
    safety_ratios = []
    for tet in safety_tets_np:
        points = safety_result[tet]
        safety_ratios.append(
            float(
                np.linalg.det(
                    np.column_stack(
                        (points[1] - points[0], points[2] - points[0], points[3] - points[0])
                    )
                )
            )
        )
    blocked_jaw_does_not_freeze_other_jaw = bool(
        0.0 < safety_scales[0] < 1.0
        and np.isclose(safety_scales[1], 1.0)
        and np.isclose(safety_result[7, 2], 1.1, atol=1.0e-6)
        and min(safety_ratios) >= 0.03 - 1.0e-6
    )

    volume_positions = wp.array(
        safety_positions_np[:4], dtype=wp.vec3, device=device
    )
    volume_inverse_mass = wp.ones(4, dtype=float, device=device)
    volume_flags = wp.full(4, value=1, dtype=wp.uint32, device=device)
    volume_deltas_np = np.zeros((4, 3), dtype=np.float32)
    volume_deltas_np[3, 2] = -2.0
    volume_deltas = wp.array(
        volume_deltas_np, dtype=wp.vec3, device=device
    )
    volume_counts = wp.array(
        [0.0, 0.0, 0.0, 1.0], dtype=float, device=device
    )
    volume_bodies = wp.array(
        [-1, -1, -1, 10], dtype=int, device=device
    )
    volume_tets = wp.array(
        np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        dtype=int,
        device=device,
    )
    volume_rest_inverse = wp.array(
        np.eye(3, dtype=np.float32)[None],
        dtype=wp.mat33,
        device=device,
    )
    volume_tet_ids = wp.array([0], dtype=int, device=device)
    jaw_direct_deltas = wp.zeros(4, dtype=wp.vec3, device=device)
    jaw_direct_weights = wp.zeros(4, dtype=float, device=device)
    wp.launch(
        filter_persistent_grip_jaw_deltas,
        4,
        inputs=[volume_deltas, volume_counts, volume_bodies, 10],
        outputs=[jaw_direct_deltas, jaw_direct_weights],
        device=device,
    )
    volume_support_deltas = wp.zeros(4, dtype=wp.vec3, device=device)
    volume_support_weights = wp.zeros(4, dtype=float, device=device)
    wp.launch(
        accumulate_triangle_skin_contact_spread,
        1,
        inputs=[
            volume_tets,
            volume_tet_ids,
            jaw_direct_deltas,
            jaw_direct_weights,
            volume_inverse_mass,
        ],
        outputs=[volume_support_deltas, volume_support_weights],
        device=device,
    )
    wp.launch(
        normalize_triangle_skin_contact_spread,
        4,
        inputs=[
            jaw_direct_deltas,
            jaw_direct_weights,
            volume_support_deltas,
            volume_support_weights,
        ],
        device=device,
    )
    wp.launch(
        remove_persistent_grip_direct_from_support,
        4,
        inputs=[
            volume_counts,
            volume_support_deltas,
            volume_support_weights,
        ],
        device=device,
    )
    unit_scale = wp.ones(1, dtype=float, device=device)
    wp.launch(
        apply_triangle_skin_contact_deltas,
        4,
        inputs=[
            volume_positions,
            volume_inverse_mass,
            volume_flags,
            volume_support_deltas,
            volume_support_weights,
            unit_scale,
        ],
        device=device,
    )
    volume_jaw_scales = wp.ones(2, dtype=float, device=device)
    wp.launch(
        find_safe_persistent_grip_jaw_scale,
        1,
        inputs=[
            volume_positions,
            volume_deltas,
            volume_bodies,
            10,
            volume_tets,
            volume_rest_inverse,
            volume_tet_ids,
            0.03,
            0,
            volume_jaw_scales,
        ],
        device=device,
    )
    wp.launch(
        apply_persistent_grip_jaw_deltas,
        4,
        inputs=[
            volume_positions,
            volume_inverse_mass,
            volume_flags,
            volume_deltas,
            volume_counts,
            volume_bodies,
            10,
            0,
            volume_jaw_scales,
        ],
        device=device,
    )
    volume_result = volume_positions.numpy()
    volume_matrix = np.column_stack(
        (
            volume_result[1] - volume_result[0],
            volume_result[2] - volume_result[0],
            volume_result[3] - volume_result[0],
        )
    )
    finite_volume_patch_preserves_tet_and_full_control = bool(
        np.isclose(volume_jaw_scales.numpy()[0], 1.0)
        and np.isclose(np.linalg.det(volume_matrix), 1.0, atol=1.0e-6)
        and np.allclose(
            volume_result,
            safety_positions_np[:4] + np.asarray([0.0, 0.0, -2.0]),
            atol=1.0e-6,
        )
    )

    angle.fill_(0.20)
    timestamp.fill_(41.80)
    capture_allowed.zero_()
    release_requested.fill_(1)
    wp.launch(
        update_persistent_grip_state,
        1,
        inputs=update_inputs,
        outputs=[state, capture_angle, capture_timestamp],
        device=device,
    )
    wp.launch(
        capture_or_release_persistent_grip_particles,
        particle_count,
        inputs=[
            positions,
            inverse_mass,
            flags,
            moved_body_q,
            0,
            1,
            selected_ids,
            state,
        ],
        outputs=[bodies, local, direct, weights, levels],
        device=device,
    )
    gates = {
        "bilateral_patches_need_not_share_one_mesh_vertex": bool(
            int(shared_node_count.numpy()[0]) == 0
        ),
        "two_contacted_surface_nodes_are_selected_per_jaw": (
            two_contacted_nodes_are_selected_per_jaw
        ),
        "collapsed_nearest_node_is_replaced_by_healthy_surface_node": (
            collapsed_nearest_node_is_replaced
        ),
        "two_contact_solves_do_not_capture": two_solves_do_not_capture,
        "third_sustained_bilateral_solve_captures": third_solve_captures,
        "two_anchors_are_bound_to_each_jaw_frame": (
            two_anchors_use_each_jaw_frame
        ),
        "four_anchors_and_local_support_follow_common_jaw_motion": (
            direct_and_soft_support_follow_common_jaw_motion
        ),
        "one_generation_excludes_second_support_ring": (
            one_local_support_ring_is_captured
        ),
        "per_jaw_closure_produces_distinct_u_t_targets": (
            per_jaw_closure_targets_are_distinct
        ),
        "local_support_follows_wrist_not_q7_closure": (
            local_support_ignores_q7_closure
        ),
        "blocked_jaw_does_not_freeze_other_volume_safe_jaw": (
            blocked_jaw_does_not_freeze_other_jaw
        ),
        "finite_volume_patch_preserves_tet_and_full_u_t": (
            finite_volume_patch_preserves_tet_and_full_control
        ),
        "opening_releases_and_clears_all_anchors": bool(
            int(state.numpy()[0]) == 0 and bodies.numpy().max() == -1
        ),
    }
    print(
        json.dumps(
            {
                "passed": all(gates.values()),
                "gates": gates,
                "selected_particle_ids": selected_ids_np.tolist(),
                "grasp_center_m": grasp_center.numpy()[0].tolist(),
                "jaw_patch_separation_m": float(
                    patch_separation.numpy()[0]
                ),
            },
            indent=2,
        )
    )
    raise SystemExit(0 if all(gates.values()) else 1)


if __name__ == "__main__":
    main()
