# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
from dataclasses import dataclass

from embodied_gaussians.physics_simulator.builder import ModelBuilder
import torch
import warp as wp
import warp.sim
import open3d as o3d
from embodied_gaussians.scene_builders.domain import Body, SoftBody
from embodied_gaussians.embodied_simulator.gaussians import GaussianModel


@dataclass(frozen=True)
class SoftBodyHandle:
    name: str
    particle_start: int
    particle_end: int
    tet_start: int
    tet_end: int
    collision_skin_face_start: int
    collision_skin_face_end: int
    gaussian_start: int
    gaussian_end: int


class EmbodiedGaussiansBuilder(ModelBuilder):
    def __init__(self, up_vector=(0.0, 0.0, 1.0), gravity=-9.80665):
        super().__init__(up_vector=up_vector, gravity=gravity)

        self.gaussian_means = []
        self.gaussian_quats = []
        self.gaussian_scales = []
        self.gaussian_opacities = []
        self.gaussian_colors = []
        self.gaussian_body_ids = []
        self.soft_gaussian_ids = []
        self.soft_gaussian_particle_indices = []
        self.soft_gaussian_tet_ids = []
        self.soft_gaussian_barycentric_weights = []
        self.soft_gaussian_rest_offsets = []
        self.soft_gaussian_binding_modes = []
        self.soft_gaussian_face_particle_indices = []
        self.soft_gaussian_rest_face_frames = []
        self.soft_gaussian_visual_vertex_particle_indices = []
        self.soft_gaussian_visual_vertex_weights = []
        self.soft_gaussian_visual_vertex_rest_offsets = []
        self.soft_gaussian_visual_vertex_rest_physical_frames = []
        self.soft_gaussian_rest_visual_face_frames = []
        self.soft_gaussian_rest_visual_face_poses = []
        self.soft_collision_skin_faces = []
        self.soft_collision_skin_face_markers = []
        # Physics contact may deliberately use zero-radius particles when the
        # deforming triangle skin owns collision.  Keep display radii separate
        # so the physics debug view remains useful without re-enabling sphere
        # contacts.
        self.soft_particle_visual_radius_indices = []
        self.soft_particle_visual_radii = []
        self.bodies_affected_by_visual_forces = []
        self.soft_body_handles: list[SoftBodyHandle] = []

    def add_builder(
        self,
        builder: "EmbodiedGaussiansBuilder",
        xform=None,
        update_num_env_count=True,
        separate_collision_group=True,
    ):
        body_count = self.body_count
        particle_count = self.particle_count
        tet_count = self.tet_count
        collision_skin_face_count = len(self.soft_collision_skin_faces)
        gaussian_count = self.num_gaussians()
        arrs = [
            "gaussian_means",
            "gaussian_quats",
            "gaussian_scales",
            "gaussian_opacities",
            "gaussian_colors",
        ]
        for arr in arrs:
            getattr(self, arr).extend(getattr(builder, arr))

        new_gaussian_body_ids = [
            b + body_count if b != -1 else -1 for b in builder.gaussian_body_ids
        ]
        new_bodies_affected_by_visual_forces = [
            b + body_count for b in builder.bodies_affected_by_visual_forces
        ]
        self.gaussian_body_ids.extend(new_gaussian_body_ids)
        self.soft_gaussian_ids.extend(
            gaussian_count + int(i) for i in builder.soft_gaussian_ids
        )
        self.soft_gaussian_particle_indices.extend(
            (np.asarray(indices, dtype=np.int32) + particle_count).tolist()
            for indices in builder.soft_gaussian_particle_indices
        )
        self.soft_gaussian_tet_ids.extend(
            tet_count + int(i) for i in builder.soft_gaussian_tet_ids
        )
        self.soft_gaussian_barycentric_weights.extend(
            builder.soft_gaussian_barycentric_weights
        )
        self.soft_gaussian_rest_offsets.extend(builder.soft_gaussian_rest_offsets)
        self.soft_gaussian_binding_modes.extend(builder.soft_gaussian_binding_modes)
        self.soft_gaussian_face_particle_indices.extend(
            np.where(
                np.asarray(indices, dtype=np.int32) >= 0,
                np.asarray(indices, dtype=np.int32) + particle_count,
                -1,
            ).tolist()
            for indices in builder.soft_gaussian_face_particle_indices
        )
        self.soft_gaussian_rest_face_frames.extend(
            builder.soft_gaussian_rest_face_frames
        )
        self.soft_gaussian_visual_vertex_particle_indices.extend(
            (
                np.asarray(indices, dtype=np.int32) + particle_count
            ).tolist()
            for indices in builder.soft_gaussian_visual_vertex_particle_indices
        )
        self.soft_gaussian_visual_vertex_weights.extend(
            builder.soft_gaussian_visual_vertex_weights
        )
        self.soft_gaussian_visual_vertex_rest_offsets.extend(
            builder.soft_gaussian_visual_vertex_rest_offsets
        )
        self.soft_gaussian_visual_vertex_rest_physical_frames.extend(
            builder.soft_gaussian_visual_vertex_rest_physical_frames
        )
        self.soft_gaussian_rest_visual_face_frames.extend(
            builder.soft_gaussian_rest_visual_face_frames
        )
        self.soft_gaussian_rest_visual_face_poses.extend(
            builder.soft_gaussian_rest_visual_face_poses
        )
        self.soft_collision_skin_faces.extend(
            (np.asarray(face, dtype=np.int32) + particle_count).tolist()
            for face in builder.soft_collision_skin_faces
        )
        self.soft_collision_skin_face_markers.extend(
            int(marker) for marker in builder.soft_collision_skin_face_markers
        )
        self.soft_particle_visual_radius_indices.extend(
            particle_count + int(index)
            for index in builder.soft_particle_visual_radius_indices
        )
        self.soft_particle_visual_radii.extend(
            float(radius) for radius in builder.soft_particle_visual_radii
        )
        self.bodies_affected_by_visual_forces.extend(
            new_bodies_affected_by_visual_forces
        )
        self.soft_body_handles.extend(
            SoftBodyHandle(
                name=handle.name,
                particle_start=handle.particle_start + particle_count,
                particle_end=handle.particle_end + particle_count,
                tet_start=handle.tet_start + tet_count,
                tet_end=handle.tet_end + tet_count,
                collision_skin_face_start=(
                    handle.collision_skin_face_start + collision_skin_face_count
                ),
                collision_skin_face_end=(
                    handle.collision_skin_face_end + collision_skin_face_count
                ),
                gaussian_start=handle.gaussian_start + gaussian_count,
                gaussian_end=handle.gaussian_end + gaussian_count,
            )
            for handle in builder.soft_body_handles
        )

        super().add_builder(
            builder, xform, update_num_env_count, separate_collision_group
        )

    def add_soft_body(
        self,
        soft_body: SoftBody,
        young_modulus_pa: float,
        poisson_ratio: float,
        anchor_mode: str = "fixed",
        add_gaussians: bool = False,
        add_collision_skin: bool = True,
    ) -> SoftBodyHandle:
        """Add Warp particles/tetrahedra and optional tetra-skinned Gaussians."""
        if young_modulus_pa <= 0.0:
            raise ValueError("young_modulus_pa must be positive")
        if not (-1.0 < poisson_ratio < 0.5):
            raise ValueError("poisson_ratio must be in (-1, 0.5)")
        if anchor_mode not in {"none", "fixed", "support_candidate"}:
            raise ValueError(f"Unknown soft-body anchor mode: {anchor_mode}")

        mesh = soft_body.tetra_mesh
        positions = np.asarray(mesh.rest_positions, dtype=np.float64)
        tets = np.asarray(mesh.tet_indices, dtype=np.int32)
        masses = np.asarray(mesh.particle_mass, dtype=np.float64).copy()
        radii = np.asarray(mesh.particle_radius, dtype=np.float64)
        visual_radii = (
            radii
            if mesh.particle_visual_radius is None
            else np.asarray(mesh.particle_visual_radius, dtype=np.float64)
        )
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"Expected particle positions Nx3, got {positions.shape}")
        if tets.ndim != 2 or tets.shape[1] != 4:
            raise ValueError(f"Expected tetrahedra Mx4, got {tets.shape}")
        if len(masses) != len(positions) or len(radii) != len(positions):
            raise ValueError("Soft particle mass/radius count does not match positions")
        if len(visual_radii) != len(positions):
            raise ValueError(
                "Soft particle visual-radius count does not match positions"
            )
        if (
            not np.isfinite(positions).all()
            or not np.isfinite(masses).all()
            or not np.isfinite(radii).all()
            or not np.isfinite(visual_radii).all()
        ):
            raise ValueError("Soft particle asset contains non-finite values")
        if np.any(radii < 0.0) or np.any(visual_radii < 0.0):
            raise ValueError("Soft particle radii must be non-negative")
        if tets.min() < 0 or tets.max() >= len(positions):
            raise ValueError("Soft tetrahedron index is outside the particle range")

        if anchor_mode == "fixed":
            anchor_mask = np.asarray(mesh.fixed_mask, dtype=bool)
        elif anchor_mode == "support_candidate":
            anchor_mask = np.asarray(mesh.support_candidate_mask, dtype=bool)
        else:
            anchor_mask = np.zeros(len(positions), dtype=bool)
        if len(anchor_mask) != len(positions):
            raise ValueError("Soft anchor mask count does not match positions")
        masses[anchor_mask] = 0.0

        particle_start = self.particle_count
        for position, mass, radius in zip(positions, masses, radii):
            self.add_particle(
                position,
                (0.0, 0.0, 0.0),
                float(mass),
                radius=float(radius),
            )
        self.soft_particle_visual_radius_indices.extend(
            range(particle_start, particle_start + len(positions))
        )
        self.soft_particle_visual_radii.extend(
            float(radius) for radius in visual_radii
        )

        mu = young_modulus_pa / (2.0 * (1.0 + poisson_ratio))
        lame_lambda = (
            young_modulus_pa
            * poisson_ratio
            / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
        )
        tet_start = self.tet_count
        particle_offset = particle_start
        generated_volumes = []
        for tet in tets:
            volume = self.add_tetrahedron(
                particle_offset + int(tet[0]),
                particle_offset + int(tet[1]),
                particle_offset + int(tet[2]),
                particle_offset + int(tet[3]),
                float(mu),
                float(lame_lambda),
                0.0,
            )
            if volume <= 0.0:
                raise RuntimeError("Soft asset produced an inverted runtime tetrahedron")
            generated_volumes.append(volume)
        generated_volumes = np.asarray(generated_volumes, dtype=np.float64)
        rest_volumes = np.asarray(mesh.rest_tet_volume, dtype=np.float64)
        if not np.allclose(generated_volumes, rest_volumes, rtol=2.0e-4, atol=1.0e-12):
            max_error = float(np.max(np.abs(generated_volumes - rest_volumes)))
            raise RuntimeError(
                f"Runtime tetrahedron volume differs from stage-A asset; max={max_error}"
            )

        collision_skin_face_start = len(self.soft_collision_skin_faces)
        if add_collision_skin and mesh.collision_skin_faces is not None:
            collision_skin_faces = np.asarray(
                mesh.collision_skin_faces, dtype=np.int32
            )
            if (
                collision_skin_faces.ndim != 2
                or collision_skin_faces.shape[1] != 3
            ):
                raise ValueError(
                    "Expected collision skin faces Mx3, got "
                    f"{collision_skin_faces.shape}"
                )
            if (
                collision_skin_faces.size
                and (
                    collision_skin_faces.min() < 0
                    or collision_skin_faces.max() >= len(positions)
                )
            ):
                raise ValueError(
                    "Collision skin face index is outside the soft body"
                )
            if mesh.collision_skin_enabled_faces is None:
                enabled_faces = np.ones(len(collision_skin_faces), dtype=bool)
            else:
                enabled_faces = np.asarray(
                    mesh.collision_skin_enabled_faces, dtype=bool
                )
                if enabled_faces.shape != (len(collision_skin_faces),):
                    raise ValueError(
                        "Collision skin enabled-mask count does not match faces"
                    )
            if mesh.collision_skin_face_markers is None:
                face_markers = np.zeros(
                    len(collision_skin_faces), dtype=np.uint8
                )
            else:
                face_markers = np.asarray(
                    mesh.collision_skin_face_markers, dtype=np.uint8
                )
                if face_markers.shape != (len(collision_skin_faces),):
                    raise ValueError(
                        "Collision skin marker count does not match faces"
                    )
            enabled_skin_faces = collision_skin_faces[enabled_faces]
            self.soft_collision_skin_faces.extend(
                (enabled_skin_faces + particle_start).tolist()
            )
            self.soft_collision_skin_face_markers.extend(
                face_markers[enabled_faces].tolist()
            )
        collision_skin_face_end = len(self.soft_collision_skin_faces)

        gaussian_start = self.num_gaussians()
        if add_gaussians:
            skinning = soft_body.gaussian_skinning
            if skinning is None:
                raise ValueError("add_gaussians=True requires Gaussian skinning data")
            rest_means = np.asarray(skinning.rest_means, dtype=np.float32)
            rest_quats = np.asarray(skinning.rest_quats_wxyz, dtype=np.float32)
            scales = np.asarray(skinning.scales, dtype=np.float32)
            opacities = np.asarray(skinning.opacities, dtype=np.float32)
            colors = np.asarray(skinning.colors_rgb, dtype=np.float32)
            particle_indices = np.asarray(skinning.particle_indices, dtype=np.int32)
            tet_ids = np.asarray(skinning.tet_ids, dtype=np.int32)
            weights = np.asarray(skinning.barycentric_weights, dtype=np.float32)
            rest_offsets = np.asarray(skinning.rest_offset, dtype=np.float32)
            binding_mode = str(skinning.binding_mode)
            gaussian_count = len(rest_means)
            expected_shapes = {
                "rest_quats": (gaussian_count, 4),
                "scales": (gaussian_count, 3),
                "colors": (gaussian_count, 3),
                "particle_indices": (gaussian_count, 4),
                "weights": (gaussian_count, 4),
                "rest_offsets": (gaussian_count, 3),
            }
            arrays = {
                "rest_quats": rest_quats,
                "scales": scales,
                "colors": colors,
                "particle_indices": particle_indices,
                "weights": weights,
                "rest_offsets": rest_offsets,
            }
            if rest_means.shape != (gaussian_count, 3):
                raise ValueError(f"Expected Gaussian means Nx3, got {rest_means.shape}")
            for name, expected in expected_shapes.items():
                if arrays[name].shape != expected:
                    raise ValueError(
                        f"Expected Gaussian {name} shape {expected}, got {arrays[name].shape}"
                    )
            if opacities.shape != (gaussian_count,) or tet_ids.shape != (gaussian_count,):
                raise ValueError("Soft Gaussian opacity/tet-id count mismatch")
            if not all(np.isfinite(a).all() for a in arrays.values()):
                raise ValueError("Soft Gaussian binding contains non-finite values")
            if not np.isfinite(rest_means).all() or not np.isfinite(opacities).all():
                raise ValueError("Soft Gaussian appearance contains non-finite values")
            if particle_indices.min() < 0 or particle_indices.max() >= len(positions):
                raise ValueError("Soft Gaussian particle index is outside the soft body")
            if tet_ids.min() < 0 or tet_ids.max() >= len(tets):
                raise ValueError("Soft Gaussian tetrahedron id is outside the soft body")
            if not np.array_equal(particle_indices, tets[tet_ids]):
                raise ValueError("Soft Gaussian particle indices disagree with tetrahedron ids")
            if not np.allclose(weights.sum(axis=1), 1.0, atol=2.0e-5):
                raise ValueError("Soft Gaussian barycentric weights do not sum to one")

            def face_frames(face_indices: np.ndarray) -> np.ndarray:
                face_points = positions[face_indices]
                tangent_x = face_points[:, 1] - face_points[:, 0]
                edge_norm = np.linalg.norm(tangent_x, axis=1, keepdims=True)
                normal = np.cross(
                    face_points[:, 1] - face_points[:, 0],
                    face_points[:, 2] - face_points[:, 0],
                )
                normal_norm = np.linalg.norm(normal, axis=1, keepdims=True)
                if np.any(edge_norm <= 1.0e-12) or np.any(normal_norm <= 1.0e-12):
                    raise ValueError("Gaussian binding contains a degenerate face")
                tangent_x = tangent_x / edge_norm
                normal = normal / normal_norm
                tangent_y = np.cross(normal, tangent_x)
                tangent_y /= np.maximum(
                    np.linalg.norm(tangent_y, axis=1, keepdims=True), 1.0e-12
                )
                return np.stack((tangent_x, tangent_y, normal), axis=-1).astype(
                    np.float32
                )

            identity_frames = np.tile(
                np.eye(3, dtype=np.float32)[None], (gaussian_count, 1, 1)
            )
            face_particle_indices = np.zeros((gaussian_count, 3), dtype=np.int32)
            rest_face_frames = identity_frames.copy()
            visual_vertex_particle_indices = np.zeros(
                (gaussian_count, 9), dtype=np.int32
            )
            visual_vertex_weights = np.zeros((gaussian_count, 9), dtype=np.float32)
            visual_vertex_rest_offsets = np.zeros(
                (gaussian_count, 3, 3), dtype=np.float32
            )
            visual_vertex_rest_physical_frames = np.tile(
                np.eye(3, dtype=np.float32)[None, None],
                (gaussian_count, 3, 1, 1),
            )
            rest_visual_face_frames = identity_frames.copy()
            rest_visual_face_poses = identity_frames.copy()

            if binding_mode == "surface_face_barycentric":
                if (
                    skinning.face_particle_indices is None
                    or skinning.face_barycentric_weights is None
                ):
                    raise ValueError(
                        "Surface-face Gaussian binding requires face indices and weights"
                    )
                face_particle_indices = np.asarray(
                    skinning.face_particle_indices, dtype=np.int32
                )
                face_weights = np.asarray(
                    skinning.face_barycentric_weights, dtype=np.float32
                )
                if face_particle_indices.shape != (gaussian_count, 3):
                    raise ValueError("Expected Gaussian face particle indices Nx3")
                if face_weights.shape != (gaussian_count, 3):
                    raise ValueError("Expected Gaussian face weights Nx3")
                if (
                    face_particle_indices.min() < 0
                    or face_particle_indices.max() >= len(positions)
                ):
                    raise ValueError("Gaussian face index is outside the soft body")
                if not np.allclose(face_weights.sum(axis=1), 1.0, atol=2.0e-5):
                    raise ValueError("Gaussian face weights do not sum to one")
                face_is_in_tet = np.any(
                    particle_indices[:, :, None]
                    == face_particle_indices[:, None, :],
                    axis=1,
                )
                if not np.all(face_is_in_tet):
                    raise ValueError(
                        "Gaussian face particles are not in the adjacent tetrahedron"
                    )
                rest_face_frames = face_frames(face_particle_indices)
                binding_modes = np.ones(gaussian_count, dtype=np.int32)
            elif binding_mode == "visual_surface_face_centroid":
                required_visual = {
                    "visual_surface_rest_vertices": skinning.visual_surface_rest_vertices,
                    "visual_surface_faces": skinning.visual_surface_faces,
                    "visual_vertex_particle_indices": skinning.visual_vertex_particle_indices,
                    "visual_vertex_barycentric_weights": skinning.visual_vertex_barycentric_weights,
                    "visual_vertex_rest_offsets": skinning.visual_vertex_rest_offsets,
                    "visual_face_ids": skinning.visual_face_ids,
                }
                missing_visual = [
                    name for name, value in required_visual.items() if value is None
                ]
                if missing_visual:
                    raise ValueError(
                        "Visual-surface centroid binding is missing: "
                        + ", ".join(missing_visual)
                    )
                visual_vertices = np.asarray(
                    skinning.visual_surface_rest_vertices, dtype=np.float32
                )
                visual_faces = np.asarray(
                    skinning.visual_surface_faces, dtype=np.int32
                )
                vertex_particle_indices = np.asarray(
                    skinning.visual_vertex_particle_indices, dtype=np.int32
                )
                vertex_weights = np.asarray(
                    skinning.visual_vertex_barycentric_weights, dtype=np.float32
                )
                vertex_offsets = np.asarray(
                    skinning.visual_vertex_rest_offsets, dtype=np.float32
                )
                visual_face_ids = np.asarray(
                    skinning.visual_face_ids, dtype=np.int32
                )
                vertex_count = len(visual_vertices)
                if visual_vertices.ndim != 2 or visual_vertices.shape[1] != 3:
                    raise ValueError("Expected visual surface vertices Vx3")
                if visual_faces.ndim != 2 or visual_faces.shape[1] != 3:
                    raise ValueError("Expected visual surface faces Fx3")
                if visual_faces.min() < 0 or visual_faces.max() >= vertex_count:
                    raise ValueError("Visual surface face index is invalid")
                if vertex_particle_indices.shape != (vertex_count, 3):
                    raise ValueError("Expected visual vertex particle indices Vx3")
                if vertex_particle_indices.min() < 0 or vertex_particle_indices.max() >= len(positions):
                    raise ValueError("Visual vertex physical support is invalid")
                if vertex_weights.shape != (vertex_count, 3):
                    raise ValueError("Expected visual vertex weights Vx3")
                if not np.allclose(vertex_weights.sum(axis=1), 1.0, atol=2.0e-5):
                    raise ValueError("Visual vertex weights do not sum to one")
                if vertex_offsets.shape != (vertex_count, 3):
                    raise ValueError("Expected visual vertex rest offsets Vx3")
                if visual_face_ids.shape != (gaussian_count,):
                    raise ValueError("Expected one visual face id per Gaussian")
                if visual_face_ids.min() < 0 or visual_face_ids.max() >= len(visual_faces):
                    raise ValueError("Gaussian visual face id is invalid")

                visual_face_vertex_ids = visual_faces[visual_face_ids]
                rest_visual_face_points = visual_vertices[visual_face_vertex_ids]
                expected_centroids = rest_visual_face_points.mean(axis=1)
                if not np.allclose(rest_means, expected_centroids, atol=2.0e-8):
                    raise ValueError("Gaussian means are not exact visual-face centroids")
                gathered_particle_indices = vertex_particle_indices[
                    visual_face_vertex_ids
                ]
                gathered_weights = vertex_weights[visual_face_vertex_ids]
                gathered_offsets = vertex_offsets[visual_face_vertex_ids]
                reconstructed_vertices = np.einsum(
                    "nvi,nvij->nvj",
                    gathered_weights,
                    positions[gathered_particle_indices],
                ) + gathered_offsets
                if not np.allclose(
                    reconstructed_vertices, rest_visual_face_points, atol=2.0e-8
                ):
                    raise ValueError("Visual vertex physical embedding does not reconstruct")

                visual_vertex_particle_indices = gathered_particle_indices.reshape(
                    gaussian_count, 9
                )
                visual_vertex_weights = gathered_weights.reshape(gaussian_count, 9)
                visual_vertex_rest_offsets = gathered_offsets
                visual_vertex_rest_physical_frames = face_frames(
                    gathered_particle_indices.reshape(-1, 3)
                ).reshape(gaussian_count, 3, 3, 3)
                tangent_x = rest_visual_face_points[:, 1] - rest_visual_face_points[:, 0]
                normal = np.cross(
                    tangent_x,
                    rest_visual_face_points[:, 2] - rest_visual_face_points[:, 0],
                )
                tangent_x /= np.maximum(
                    np.linalg.norm(tangent_x, axis=1, keepdims=True), 1.0e-12
                )
                normal /= np.maximum(
                    np.linalg.norm(normal, axis=1, keepdims=True), 1.0e-12
                )
                tangent_y = np.cross(normal, tangent_x)
                tangent_y /= np.maximum(
                    np.linalg.norm(tangent_y, axis=1, keepdims=True), 1.0e-12
                )
                rest_visual_face_frames = np.stack(
                    (tangent_x, tangent_y, normal), axis=-1
                ).astype(np.float32)
                # Map the rest triangle basis to the current triangle basis.
                # A unit normal preserves Gaussian thickness while the two
                # in-plane axes follow triangle stretch and shear.
                rest_visual_face_shapes = np.stack(
                    (
                        rest_visual_face_points[:, 1]
                        - rest_visual_face_points[:, 0],
                        rest_visual_face_points[:, 2]
                        - rest_visual_face_points[:, 0],
                        normal,
                    ),
                    axis=-1,
                )
                rest_visual_face_poses = np.linalg.inv(
                    rest_visual_face_shapes
                ).astype(np.float32)
                binding_modes = np.full(gaussian_count, 2, dtype=np.int32)
            elif binding_mode == "tetrahedron":
                binding_modes = np.zeros(gaussian_count, dtype=np.int32)
            else:
                raise ValueError(f"Unknown Gaussian binding mode: {binding_mode}")

            self.gaussian_means.extend(rest_means.tolist())
            self.gaussian_quats.extend(rest_quats.tolist())
            self.gaussian_scales.extend(scales.tolist())
            self.gaussian_opacities.extend(opacities.tolist())
            self.gaussian_colors.extend(colors.tolist())
            self.gaussian_body_ids.extend([-1] * gaussian_count)
            self.soft_gaussian_ids.extend(
                range(gaussian_start, gaussian_start + gaussian_count)
            )
            self.soft_gaussian_particle_indices.extend(
                (particle_indices + particle_start).tolist()
            )
            self.soft_gaussian_tet_ids.extend((tet_ids + tet_start).tolist())
            self.soft_gaussian_barycentric_weights.extend(weights.tolist())
            self.soft_gaussian_rest_offsets.extend(rest_offsets.tolist())
            self.soft_gaussian_binding_modes.extend(binding_modes.tolist())
            self.soft_gaussian_face_particle_indices.extend(
                np.where(
                    face_particle_indices >= 0,
                    face_particle_indices + particle_start,
                    -1,
                ).tolist()
            )
            self.soft_gaussian_rest_face_frames.extend(
                rest_face_frames.tolist()
            )
            self.soft_gaussian_visual_vertex_particle_indices.extend(
                visual_vertex_particle_indices.tolist()
            )
            self.soft_gaussian_visual_vertex_weights.extend(
                visual_vertex_weights.tolist()
            )
            self.soft_gaussian_visual_vertex_rest_offsets.extend(
                visual_vertex_rest_offsets.tolist()
            )
            self.soft_gaussian_visual_vertex_rest_physical_frames.extend(
                visual_vertex_rest_physical_frames.tolist()
            )
            self.soft_gaussian_rest_visual_face_frames.extend(
                rest_visual_face_frames.tolist()
            )
            self.soft_gaussian_rest_visual_face_poses.extend(
                rest_visual_face_poses.tolist()
            )

        handle = SoftBodyHandle(
            name=soft_body.name,
            particle_start=particle_start,
            particle_end=self.particle_count,
            tet_start=tet_start,
            tet_end=self.tet_count,
            collision_skin_face_start=collision_skin_face_start,
            collision_skin_face_end=collision_skin_face_end,
            gaussian_start=gaussian_start,
            gaussian_end=self.num_gaussians(),
        )
        self.soft_body_handles.append(handle)
        return handle

    def add_visual_body(self, body: Body):
        # X_WB = np.asarray(body.X_WB)
        # quat = wp.quat_from_matrix(X_WB[:3, :3])
        assert body.gaussians
        gaussians = body.gaussians
        self.gaussian_means.extend(gaussians.means)
        self.gaussian_quats.extend(gaussians.quats)
        self.gaussian_scales.extend(gaussians.scales)
        self.gaussian_opacities.extend(gaussians.opacities)
        self.gaussian_colors.extend(gaussians.colors)
        self.gaussian_body_ids.extend([-1] * len(gaussians.means))

    def add_rigid_body(
        self,
        body: Body,
        mu: float = 0.0,
        add_gaussians: bool = True,
        density: float | None = None,
        individual_collision_groups: bool = False,
    ):
        X_WB = np.asarray(body.X_WB)
        quat = wp.quat_from_matrix(X_WB[:3, :3])
        trans = X_WB[:3, 3]
        t = wp.transformf(*trans, *quat)
        b = self.add_body(origin=t)  # type: ignore
        self.bodies_affected_by_visual_forces.append(b)
        particles = body.particles
        assert particles
        shape_ids = []
        first_collision_group = max(self.last_collision_group + 1, 1)
        for i in range(len(particles.means)):
            pos = particles.means[i]
            quat = particles.quats[i]
            radius = particles.radii[i]
            if individual_collision_groups:
                # Warp otherwise creates every same-body filter pair, which is
                # quadratic for dense sphere compounds. One shape per group
                # encodes the same no-self-collision rule in linear storage.
                self.body_shapes[b] = []
            shape_id = self.add_shape_sphere(
                body=b,
                radius=radius,
                pos=pos,
                rot=[quat[1], quat[2], quat[3], quat[0]],
                mu=mu,
                density=density,
                collision_group=(
                    first_collision_group + i
                    if individual_collision_groups
                    else -1
                ),
            )
            shape_ids.append(shape_id)
        if individual_collision_groups:
            self.body_shapes[b] = shape_ids

        if add_gaussians:
            gaussians = body.gaussians
            assert gaussians
            self.gaussian_means.extend(gaussians.means)
            self.gaussian_quats.extend(gaussians.quats)
            self.gaussian_scales.extend(gaussians.scales)
            self.gaussian_opacities.extend(gaussians.opacities)
            self.gaussian_colors.extend(gaussians.colors)
            self.gaussian_body_ids.extend([b] * len(gaussians.means))

        return b

    def num_gaussians(self):
        return len(self.gaussian_means)

    def add_renderable_articulation_from_urdf(
        self,
        urdf_path: str,
        initial_joints: np.ndarray | None = None,
        X_WB: np.ndarray = np.eye(4),
        armature: float = 0.1,
        damping: float = 80.0,
        stiffness: float = 400,
        enable_self_collisions: bool = False,
        add_gaussians: bool = True,
        **kwargs,
    ):
        start_shape_idx = len(self.shape_body)
        self.add_articulation_from_urdf(
            urdf_path,
            initial_joints,
            X_WB,
            armature,
            damping,
            stiffness,
            enable_self_collisions,
            **kwargs,
        )
        end_shape_idx = len(self.shape_body)
        if add_gaussians:
            for i in range(start_shape_idx, end_shape_idx):
                mesh: warp.sim.model.Mesh = self.shape_geo_src[i]
                if mesh is None or mesh.vertices is None or len(mesh.vertices) == 0:
                    continue  # skip non-mesh shapes (box, sphere, capsule primitives)
                body_id = self.shape_body[i]
                mesh_open3d = o3d.geometry.TriangleMesh()
                mesh_open3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
                mesh_open3d.triangles = o3d.utility.Vector3iVector(
                    mesh.indices.reshape(-1, 3)
                )
                area = mesh_open3d.get_surface_area()
                points_per_unit_area = 10000
                n_samples = max(1, int(area * points_per_unit_area))
                points: o3d.geometry.PointCloud = (
                    mesh_open3d.sample_points_poisson_disk(n_samples)
                )
                means = np.asarray(points.points)
                num_points = len(points.points)
                area_per_point = 0.005

                self.gaussian_means.extend(means.tolist())
                self.gaussian_quats.extend([[1, 0, 0, 0]] * num_points)
                self.gaussian_scales.extend(
                    [[area_per_point, area_per_point, area_per_point]] * num_points
                )
                self.gaussian_opacities.extend([0.5] * num_points)
                self.gaussian_colors.extend([[0.5, 0.5, 0.5]] * num_points)
                self.gaussian_body_ids.extend([body_id] * num_points)

    def build_gaussian_model(self, device: str = "cuda"):
        gaussian_model = GaussianModel(
            means=torch.tensor(self.gaussian_means, device=device, dtype=torch.float32),
            quats=torch.tensor(self.gaussian_quats, device=device, dtype=torch.float32),
            scales=torch.tensor(
                self.gaussian_scales, device=device, dtype=torch.float32
            ),
            opacities=torch.tensor(
                self.gaussian_opacities, device=device, dtype=torch.float32
            ),
            colors=torch.tensor(
                self.gaussian_colors, device=device, dtype=torch.float32
            ),
            body_ids=torch.tensor(
                self.gaussian_body_ids, device=device, dtype=torch.int32
            ),
            soft_gaussian_ids=torch.tensor(
                self.soft_gaussian_ids, device=device, dtype=torch.int32
            ),
            soft_gaussian_particle_indices=torch.tensor(
                np.asarray(
                    self.soft_gaussian_particle_indices, dtype=np.int32
                ).reshape(-1, 4),
                device=device,
                dtype=torch.int32,
            ),
            soft_gaussian_tet_ids=torch.tensor(
                self.soft_gaussian_tet_ids, device=device, dtype=torch.int32
            ),
            soft_gaussian_barycentric_weights=torch.tensor(
                np.asarray(
                    self.soft_gaussian_barycentric_weights, dtype=np.float32
                ).reshape(-1, 4),
                device=device,
                dtype=torch.float32,
            ),
            soft_gaussian_rest_offsets=torch.tensor(
                np.asarray(self.soft_gaussian_rest_offsets, dtype=np.float32).reshape(
                    -1, 3
                ),
                device=device,
                dtype=torch.float32,
            ),
            soft_gaussian_binding_modes=torch.tensor(
                self.soft_gaussian_binding_modes,
                device=device,
                dtype=torch.int32,
            ),
            soft_gaussian_face_particle_indices=torch.tensor(
                np.asarray(
                    self.soft_gaussian_face_particle_indices, dtype=np.int32
                ).reshape(-1, 3),
                device=device,
                dtype=torch.int32,
            ),
            soft_gaussian_rest_face_frames=torch.tensor(
                np.asarray(
                    self.soft_gaussian_rest_face_frames, dtype=np.float32
                ).reshape(-1, 3, 3),
                device=device,
                dtype=torch.float32,
            ),
            soft_gaussian_visual_vertex_particle_indices=torch.tensor(
                np.asarray(
                    self.soft_gaussian_visual_vertex_particle_indices,
                    dtype=np.int32,
                ).reshape(-1, 9),
                device=device,
                dtype=torch.int32,
            ),
            soft_gaussian_visual_vertex_weights=torch.tensor(
                np.asarray(
                    self.soft_gaussian_visual_vertex_weights, dtype=np.float32
                ).reshape(-1, 9),
                device=device,
                dtype=torch.float32,
            ),
            soft_gaussian_visual_vertex_rest_offsets=torch.tensor(
                np.asarray(
                    self.soft_gaussian_visual_vertex_rest_offsets,
                    dtype=np.float32,
                ).reshape(-1, 3, 3),
                device=device,
                dtype=torch.float32,
            ),
            soft_gaussian_visual_vertex_rest_physical_frames=torch.tensor(
                np.asarray(
                    self.soft_gaussian_visual_vertex_rest_physical_frames,
                    dtype=np.float32,
                ).reshape(-1, 3, 3, 3),
                device=device,
                dtype=torch.float32,
            ),
            soft_gaussian_rest_visual_face_frames=torch.tensor(
                np.asarray(
                    self.soft_gaussian_rest_visual_face_frames, dtype=np.float32
                ).reshape(-1, 3, 3),
                device=device,
                dtype=torch.float32,
            ),
            soft_gaussian_rest_visual_face_poses=torch.tensor(
                np.asarray(
                    self.soft_gaussian_rest_visual_face_poses,
                    dtype=np.float32,
                ).reshape(-1, 3, 3),
                device=device,
                dtype=torch.float32,
            ),
            soft_tet_rest_poses=torch.tensor(
                np.asarray(self.tet_poses, dtype=np.float32).reshape(-1, 3, 3),
                device=device,
                dtype=torch.float32,
            ),
            num_body_gaussians=sum(
                int(body_id >= 0) for body_id in self.gaussian_body_ids
            ),
        )
        return gaussian_model

    def finalize(self, device=None, requires_grad=False):
        if device is None:
            device = str(wp.get_preferred_device())
        model = super().finalize(device, requires_grad)
        model.rigid_contact_torsional_friction = 0.0  # type: ignore
        model.rigid_contact_rolling_friction = 0.0  # type: ignore
        particle_visual_radii = np.asarray(
            self.particle_radius, dtype=np.float32
        ).copy()
        if self.soft_particle_visual_radius_indices:
            visual_indices = np.asarray(
                self.soft_particle_visual_radius_indices, dtype=np.int64
            )
            visual_values = np.asarray(
                self.soft_particle_visual_radii, dtype=np.float32
            )
            if (
                len(visual_indices) != len(visual_values)
                or visual_indices.min() < 0
                or visual_indices.max() >= len(particle_visual_radii)
            ):
                raise RuntimeError("Invalid soft particle display-radius mapping")
            particle_visual_radii[visual_indices] = visual_values
        model.particle_visual_radius = wp.array(  # type: ignore[attr-defined]
            particle_visual_radii,
            dtype=wp.float32,
            device=device,
        )
        self.gaussian_model = self.build_gaussian_model(device)
        self.gaussian_state = self.gaussian_model.state()
        return model
