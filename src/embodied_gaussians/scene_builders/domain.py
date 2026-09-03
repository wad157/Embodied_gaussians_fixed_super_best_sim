# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from pathlib import Path
from typing import Literal
import numpy as np
import torch
from pydantic import BaseModel
from dataclasses import dataclass

@dataclass
class Posed:
    X_WC: np.ndarray # (Blender standard) (4, 4)

    def get_X_WC(self, format: Literal["opencv", "blender"] = "blender") -> np.ndarray:
        """Get the camera to world transform in the specified format"""

        if format == "opencv":
            X_WC = self.X_WC @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0.0, 0.0, 0.0, 1.0]])
            return X_WC

        return self.X_WC


@dataclass
class Image:
    K: np.ndarray # (3, 3)
    image: np.ndarray # (H, W, 3) in uint8
    format: Literal['rgb', 'bgr']

@dataclass
class Depth:
    depth: np.ndarray # (H, W) in float32 [0, 1]
    depth_scale: float

@dataclass
class Masked:
    mask: np.ndarray  # Mask from camera (H, W) where 0 is background and 1 is object and 2 is occlusion

@dataclass
class PosedImage(Posed, Image):
    pass

@dataclass
class PosedImageAndDepth(Posed, Image, Depth):
    pass

@dataclass
class MaskedPosedImageAndDepth(Masked, Posed, Image, Depth):
    pass

def save_posed_images(path: Path, posed_images):
    path = Path(path)
    assert path.suffix == ".npz" 
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, posed_images, allow_pickle=True)

def load_posed_images(path: Path):
    path = Path(path)
    return np.load(path, allow_pickle=True)["arr_0"]


class GaussianLearningRates(BaseModel):
    means: float = 0.001
    opacities: float = 0.001
    colors: float = 0.01
    quats: float = 0.01 
    scales: float = 0.01

class Ground(BaseModel):
    plane: tuple[float, float, float, float] =  (0.0, 0.0, 1.0, 0.0) # (4,) ax + by + cz + d = 0

    def normal(self) -> np.ndarray:
        return self.plane[:3]
    
    def offset(self) -> float:
        return -self.plane[3]

class Gaussians(BaseModel):
    means: list[list[float]] # (n_gaussians, 3)
    quats: list[list[float]] # (n_gaussians, 4) (w, x, y, z)
    scales: list[list[float]]# (n_gaussians, 3)
    opacities: list[float]# (n_gaussians,)
    colors: list[list[float]]# (n_gaussians, 3)

    def __len__(self):
        return len(self.means)
    
    def mask(self, mask: np.ndarray):
        return Gaussians(
            means=np.asarray(self.means)[mask].tolist(),
            quats=np.asarray(self.quats)[mask].tolist(),
            scales=np.asarray(self.scales)[mask].tolist(),
            opacities=np.asarray(self.opacities)[mask].tolist(),
            colors=np.asarray(self.colors)[mask].tolist()
        )

class Particles(BaseModel):
    means: list[list[float]] # (n_gaussians, 3)
    quats: list[list[float]] # (n_gaussians, 4) (w, x, y, z)
    radii: list[float]# (n_gaussians,)
    colors: list[list[float]]# (n_gaussians, 3)

    def __len__(self):
        return len(self.means)
    
    def mask(self, mask: np.ndarray):
        return Particles(
            means=np.asarray(self.means)[mask].tolist(),
            quats=np.asarray(self.quats)[mask].tolist(),
            radii=np.asarray(self.radii)[mask].tolist(),
            colors=np.asarray(self.colors)[mask].tolist()
        )

class Body(BaseModel):
    name: str
    X_WB: list[list[float]]
    gaussians: Gaussians | None = None
    particles: Particles | None = None


@dataclass
class TetraMesh:
    rest_positions: np.ndarray
    tet_indices: np.ndarray
    surface_faces: np.ndarray
    particle_mass: np.ndarray
    particle_radius: np.ndarray
    fixed_mask: np.ndarray
    support_candidate_mask: np.ndarray
    rest_tet_volume: np.ndarray
    surface_face_markers: np.ndarray | None = None
    collision_skin_faces: np.ndarray | None = None
    collision_skin_face_markers: np.ndarray | None = None
    collision_skin_enabled_faces: np.ndarray | None = None
    particle_visual_radius: np.ndarray | None = None
    particle_target_spacing: np.ndarray | None = None
    particle_inward_depth: np.ndarray | None = None
    surface_node_mask: np.ndarray | None = None
    top_node_mask: np.ndarray | None = None
    top_rest_curvature_faces: np.ndarray | None = None
    top_rest_curvature_edges: np.ndarray | None = None
    top_rest_curvature_opposite_vertices: np.ndarray | None = None
    top_rest_dihedral_angle: np.ndarray | None = None


@dataclass
class GaussianSkinning:
    rest_means: np.ndarray
    rest_quats_wxyz: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray
    colors_rgb: np.ndarray
    tet_ids: np.ndarray
    particle_indices: np.ndarray
    barycentric_weights: np.ndarray
    rest_offset: np.ndarray
    binding_mode: str = "tetrahedron"
    face_particle_indices: np.ndarray | None = None
    face_barycentric_weights: np.ndarray | None = None
    visual_surface_rest_vertices: np.ndarray | None = None
    visual_surface_faces: np.ndarray | None = None
    visual_vertex_particle_indices: np.ndarray | None = None
    visual_vertex_barycentric_weights: np.ndarray | None = None
    visual_vertex_rest_offsets: np.ndarray | None = None
    visual_face_ids: np.ndarray | None = None


@dataclass
class SoftBody:
    name: str
    tetra_mesh: TetraMesh
    gaussian_skinning: GaussianSkinning | None = None

    @staticmethod
    def from_npz(path: Path | str, name: str = "soft_body") -> "SoftBody":
        with np.load(path) as loaded:
            def optional(key: str) -> np.ndarray | None:
                return loaded[key].copy() if key in loaded.files else None

            mesh = TetraMesh(
                rest_positions=loaded["rest_positions_table"].copy(),
                tet_indices=loaded["tet_indices"].copy(),
                surface_faces=loaded["surface_faces"].copy(),
                particle_mass=loaded["particle_mass"].copy(),
                particle_radius=loaded["particle_radius"].copy(),
                fixed_mask=loaded["fixed_mask"].copy(),
                support_candidate_mask=loaded["support_candidate_mask"].copy(),
                rest_tet_volume=loaded["rest_tet_volume"].copy(),
                surface_face_markers=optional("surface_face_markers"),
                collision_skin_faces=optional("collision_skin_faces"),
                collision_skin_face_markers=optional(
                    "collision_skin_face_markers"
                ),
                collision_skin_enabled_faces=optional(
                    "collision_skin_enabled_faces"
                ),
                particle_visual_radius=optional("particle_visual_radius"),
                particle_target_spacing=optional("particle_target_spacing"),
                particle_inward_depth=optional("particle_inward_depth"),
                surface_node_mask=optional("surface_node_mask"),
                top_node_mask=optional("top_node_mask"),
                top_rest_curvature_faces=optional(
                    "top_rest_curvature_faces"
                ),
                top_rest_curvature_edges=optional(
                    "top_rest_curvature_edges"
                ),
                top_rest_curvature_opposite_vertices=optional(
                    "top_rest_curvature_opposite_vertices"
                ),
                top_rest_dihedral_angle=optional(
                    "top_rest_dihedral_angle"
                ),
            )
            skinning = GaussianSkinning(
                rest_means=loaded["gaussian_rest_means_table"].copy(),
                rest_quats_wxyz=loaded[
                    "gaussian_rest_quats_table_wxyz"
                ].copy(),
                scales=loaded["gaussian_scales"].copy(),
                opacities=loaded["gaussian_opacities"].copy(),
                colors_rgb=loaded["gaussian_colors_rgb"].copy(),
                tet_ids=loaded["gaussian_tet_ids"].copy(),
                particle_indices=loaded["gaussian_particle_indices"].copy(),
                barycentric_weights=loaded[
                    "gaussian_barycentric_weights"
                ].copy(),
                rest_offset=loaded["gaussian_rest_offset_table"].copy(),
                binding_mode=(
                    str(loaded["gaussian_binding_mode"].item())
                    if "gaussian_binding_mode" in loaded.files
                    else "tetrahedron"
                ),
                face_particle_indices=optional(
                    "gaussian_face_particle_indices"
                ),
                face_barycentric_weights=optional(
                    "gaussian_face_barycentric_weights"
                ),
                visual_surface_rest_vertices=optional(
                    "visual_surface_rest_vertices_table"
                ),
                visual_surface_faces=optional("visual_surface_faces"),
                visual_vertex_particle_indices=optional(
                    "visual_vertex_particle_indices"
                ),
                visual_vertex_barycentric_weights=optional(
                    "visual_vertex_barycentric_weights"
                ),
                visual_vertex_rest_offsets=optional(
                    "visual_vertex_rest_offset_table"
                ),
                visual_face_ids=optional("gaussian_visual_face_ids"),
            )
        return SoftBody(name=name, tetra_mesh=mesh, gaussian_skinning=skinning)


class GaussianActivations:
    quat = torch.nn.functional.normalize
    scale = torch.exp
    opacity = torch.sigmoid
    color = torch.sigmoid

    inv_scale = torch.log
    inv_opacity = torch.logit
    inv_color = torch.logit
