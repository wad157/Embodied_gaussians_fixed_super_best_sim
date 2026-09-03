# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
from pathlib import Path
import marsoom
import marsoom.overlay
import numpy as np
import open3d as o3d
import pyglet
import pyglet.gl as gl
import torch
import warp as wp
from imgui_bundle import imgui
from imgui_bundle import portable_file_dialogs as pfd
from typing_extensions import override
from pyglet.math import Vec3 as PyVec3

from marsoom.cuda import EllipseRenderer, InstancedMeshRenderer, VectorRenderer
from embodied_gaussians.physics_visualizer.simulation_viewer import SimulationViewer

# from marsoom.cuda.ellipse_renderer_2 import EllipseRenderer as EllipseRenderer2
from embodied_gaussians.environments.embodied_environment import (
    EmbodiedGaussiansEnvironment,
)


def _build_arrow_segments(
    positions: torch.Tensor,
    directions: torch.Tensor,
    vector_scale: float,
    head_length_ratio: float = 0.30,
    head_width_ratio: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand vectors into shafts plus four 3-D arrowhead line segments."""
    if positions.shape != directions.shape or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Arrow positions/directions must both have shape (N, 3)")
    norms = torch.linalg.vector_norm(directions, dim=1)
    valid = torch.isfinite(norms) & (norms > 1.0e-12)
    positions = positions[valid]
    directions = directions[valid]
    norms = norms[valid]
    if len(positions) == 0:
        return positions, directions

    unit = directions / norms[:, None]
    reference_z = torch.zeros_like(unit)
    reference_z[:, 2] = 1.0
    reference_x = torch.zeros_like(unit)
    reference_x[:, 0] = 1.0
    use_x = torch.abs((unit * reference_z).sum(dim=1)) > 0.90
    reference = torch.where(use_x[:, None], reference_x, reference_z)
    side_1 = torch.linalg.cross(unit, reference, dim=1)
    side_1 = side_1 / torch.clamp(
        torch.linalg.vector_norm(side_1, dim=1, keepdim=True), min=1.0e-12
    )
    side_2 = torch.linalg.cross(unit, side_1, dim=1)

    # VectorRenderer multiplies every supplied direction by vector_scale.
    # Keep the head directions in the same pre-scale units as the shaft.
    backward = -head_length_ratio * unit
    heads = torch.stack(
        (
            backward + head_width_ratio * side_1,
            backward - head_width_ratio * side_1,
            backward + head_width_ratio * side_2,
            backward - head_width_ratio * side_2,
        ),
        dim=1,
    ) * norms[:, None, None]
    endpoints = positions + float(vector_scale) * directions
    segment_positions = torch.cat(
        (positions, endpoints.repeat_interleave(4, dim=0)), dim=0
    )
    segment_directions = torch.cat(
        (directions, heads.reshape(-1, 3)), dim=0
    )
    return segment_positions.contiguous(), segment_directions.contiguous()


def _set_calibrated_camera_image_geometry(
    camera: marsoom.CameraWireframeWithImage,
    K: np.ndarray,
    width: int,
    height: int,
) -> None:
    """Make the textured camera plane follow the four calibrated pixel rays."""
    wireframe = camera.camera_wireframe
    pixels = np.array(
        [[0.0, 0.0, 1.0], [width, 0.0, 1.0], [width, height, 1.0], [0.0, height, 1.0]],
        dtype=np.float32,
    )
    K_inv = np.linalg.inv(K).astype(np.float32)
    rays = (K_inv @ pixels.T).T
    rays[:, 1:] *= -1.0  # OpenCV camera axes to Blender camera axes.
    rays *= wireframe.z_offset / np.abs(rays[:, 2:3])
    top_left, top_right, bot_right, bot_left = rays.tolist()

    wireframe.top_left = top_left
    wireframe.top_right = top_right
    wireframe.bot_right = bot_right
    wireframe.bot_left = bot_left
    axis_size = abs(top_right[0] - top_left[0]) * 0.1
    wireframe.position = (
        0.0,
        0.0,
        0.0,
        *top_left,
        *top_right,
        *bot_right,
        *bot_left,
        0.0,
        0.0,
        0.0,
        axis_size,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        axis_size,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        axis_size,
    )
    wireframe._update_vertices()
    camera.image_quad.update(
        top_left=top_left,
        top_right=top_right,
        bot_right=bot_right,
        bot_left=bot_left,
    )


@dataclass
class VisualizerSettings:
    draw_gaussian_meshes: bool = False
    draw_gaussian_outlines: bool = False
    draw_physics: bool = True
    draw_gaussian_render: bool = False
    draw_visual_forces_gaussians_outlines: bool = False
    draw_visual_forces_gaussians_meshes: bool = False
    draw_visual_forces_render: bool = False
    draw_visual_forces: bool = False
    draw_cameras: bool = True
    draw_virtual_cameras: bool = True
    gaussian_render_alpha: float = 1.0
    visual_forces_scale: float = 25.0
    visual_forces_pose_scale: float = 25.0
    soft_force_display_gain: float = 10.0
    soft_force_line_width: float = 2.0
    soft_force_max_arrows: int = 1000
    near_plane: float = 0.01
    far_plane: float = 10.0
    wireframe_alpha: float = 0.5
    wireframe_z_offset: float = 0.1


class EmbodiedViewer(SimulationViewer):
    def __init__(self, window, show_origin: bool = True):
        super().__init__(window, show_origin)
        self.ellipse_renderer = EllipseRenderer()
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        mesh.compute_vertex_normals()
        self.mesh_ellipse_renderer = InstancedMeshRenderer(
            np.array(mesh.vertices),
            np.array(mesh.triangles),
            np.array(mesh.vertex_normals),
        )
        self.vector_renderer = VectorRenderer()
        self.last_soft_force_arrow_count = 0
        # gsplat rasterization returns RGB; using GL_BGR swaps red and blue.
        self.gaussian_texture = marsoom.Texture(640, 480, fmt=gl.GL_RGB)
        self.gaussian_overlay = marsoom.Overlay(self.gaussian_texture.id, alpha=1.0)  # type: ignore
        self.batch_cameras = pyglet.graphics.Batch()
        self.batch_virtual_cameras = pyglet.graphics.Batch()
        self.last_selected_camera = 0
        self.camera_go_zoom = 1.0
        self.settings = VisualizerSettings()
        self.env: EmbodiedGaussiansEnvironment | None = None
        self.cameras: dict[str, marsoom.CameraWireframeWithImage] = {}
        self.virtual_cameras: dict[str, marsoom.CameraWireframeWithImage] = {}
        self._side_view_pose: tuple[PyVec3, PyVec3, PyVec3] | None = None
        self.save_dialog: pfd.save_file | None = None

    def set_environment(self, env: EmbodiedGaussiansEnvironment):
        self.env = env
        self.set_simulator(self.env.sim)
        self.gaussian_render_state = self.env.sim.gaussian_state.clone()

    def render_controls(self):
        if self.env is None:
            return

        imgui.begin("Controls", flags=imgui.WindowFlags_.no_collapse)
        s = self.settings

        # Style setup
        imgui.push_style_var(imgui.StyleVar_.frame_padding, (4, 3))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (4, 4))

        # Display Settings
        imgui.text("Display Settings")
        imgui.separator()

        _, s.draw_gaussian_meshes = imgui.checkbox(
            "Gaussian Meshes", s.draw_gaussian_meshes
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip("Display 3D mesh representation of gaussians")

        _, s.draw_gaussian_outlines = imgui.checkbox(
            "Gaussian Outlines", s.draw_gaussian_outlines
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip("Show 2D projections of gaussians")

        _, s.draw_gaussian_render = imgui.checkbox(
            "Gaussian Render", s.draw_gaussian_render
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip("Render full gaussian visualization")

        sim_reconstruction_mode = bool(
            getattr(self.env, "super_sim_reconstruction_mode", False)
        )
        if sim_reconstruction_mode:
            _, s.draw_physics = imgui.checkbox(
                "Show Reconstructed Tissue Physics", s.draw_physics
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Only deformable tissue particles/skin; legacy SUPER "
                    "rigid geometry stays hidden"
                )
        else:
            _, s.draw_physics = imgui.checkbox("Show Physics", s.draw_physics)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Display physics simulation elements")

        _, s.draw_cameras = imgui.checkbox("Show Cameras", s.draw_cameras)
        _, s.draw_virtual_cameras = imgui.checkbox(
            "Show Virtual Cameras", s.draw_virtual_cameras
        )

        imgui.spacing()
        imgui.spacing()

        # Visual Forces
        imgui.text("Visual Forces")
        imgui.separator()

        _, s.draw_visual_forces = imgui.checkbox("Show Forces", s.draw_visual_forces)
        _, s.draw_visual_forces_gaussians_outlines = imgui.checkbox(
            "Force Gaussian Outlines", s.draw_visual_forces_gaussians_outlines
        )
        _, s.draw_visual_forces_gaussians_meshes = imgui.checkbox(
            "Force Gaussian Meshes", s.draw_visual_forces_gaussians_meshes
        )
        _, s.visual_forces_scale = imgui.slider_float(
            "Force Vector Scale", s.visual_forces_scale, 0.0, 100.0
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip("Adjust the scale of force visualization")
        _, s.visual_forces_pose_scale = imgui.slider_float(
            "Force Pose Scale", s.visual_forces_pose_scale, 1.0, 100.0
        )
        _, s.soft_force_display_gain = imgui.slider_float(
            "Soft Force Display Gain", s.soft_force_display_gain, 1.0, 100.0
        )
        _, s.soft_force_line_width = imgui.slider_float(
            "Soft Force Line Width", s.soft_force_line_width, 1.0, 6.0
        )
        imgui.text(f"Soft particle arrows: {self.last_soft_force_arrow_count}")

        imgui.spacing()
        imgui.spacing()

        # View Settings
        imgui.text("View Settings")
        imgui.separator()

        _, s.near_plane = imgui.slider_float("Near Plane", s.near_plane, 0.001, 0.3)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Distance to near clipping plane")

        _, s.far_plane = imgui.slider_float("Far Plane", s.far_plane, 0.2, 30.0)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Distance to far clipping plane")

        # Camera Controls
        frames = self.env.frames
        if frames is not None:
            imgui.spacing()
            imgui.spacing()
            imgui.text("Camera Controls")
            imgui.separator()

            num_cameras = len(frames.names)
            c, s.wireframe_alpha = imgui.slider_float(
                "Wireframe Opacity", s.wireframe_alpha, 0.0, 1.0
            )
            if c:
                for camera in self.cameras.values():
                    camera.alpha = s.wireframe_alpha

            c, s.wireframe_z_offset = imgui.slider_float(
                "Wireframe Offset", s.wireframe_z_offset, 0.0, 1.0
            )
            if c:
                for camera in self.cameras.values():
                    camera.update_z_offset(s.wireframe_z_offset)

            c, self.last_selected_camera = imgui.slider_int(
                "Camera Index", self.last_selected_camera, 0, num_cameras - 1
            )

            imgui.push_style_var(imgui.StyleVar_.frame_padding, (8, 4))
            imgui.push_style_var(imgui.StyleVar_.button_text_align, (0.5, 0.5))

            imgui.push_style_color(imgui.Col_.button, (0.2, 0.5, 0.8, 0.8))
            imgui.push_style_color(imgui.Col_.button_hovered, (0.3, 0.6, 0.9, 1.0))
            imgui.push_style_color(imgui.Col_.button_active, (0.1, 0.4, 0.7, 1.0))
            if imgui.button("Go##goto", (120, 30)):
                self.go_to_camera(self.last_selected_camera)
            imgui.pop_style_color(3)

            imgui.same_line(spacing=10)

            if self._side_view_pose is not None:
                imgui.push_style_color(imgui.Col_.button, (0.25, 0.65, 0.35, 0.8))
                imgui.push_style_color(
                    imgui.Col_.button_hovered, (0.35, 0.75, 0.45, 1.0)
                )
                imgui.push_style_color(
                    imgui.Col_.button_active, (0.15, 0.55, 0.25, 1.0)
                )
                if imgui.button("Side View##side_view", (120, 30)):
                    self.go_to_side_view()
                imgui.pop_style_color(3)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "Show the calibrated physical side view of tool-tissue contact"
                    )
                imgui.same_line(spacing=10)

            imgui.push_style_color(imgui.Col_.button, (0.8, 0.3, 0.3, 0.8))
            imgui.push_style_color(imgui.Col_.button_hovered, (0.9, 0.4, 0.4, 1.0))
            imgui.push_style_color(imgui.Col_.button_active, (0.7, 0.2, 0.2, 1.0))
            if imgui.button("Reset##reset", (120, 30)):
                self.reset_view()
            imgui.pop_style_color(3)

            imgui.pop_style_var(2)

        # Visual Forces Parameters
        imgui.spacing()
        imgui.spacing()
        imgui.text("Visual Forces Parameters")
        imgui.separator()

        vs = self.env.visual_forces_settings
        _, vs.kp = imgui.slider_float("Proportional Gain", vs.kp, 0.0, 1.0)
        _, vs.lr_means = imgui.slider_float("Mean Learning Rate", vs.lr_means, 0.0, 0.1)
        _, vs.lr_quats = imgui.slider_float(
            "Rotation Learning Rate", vs.lr_quats, 0.0, 0.1
        )
        _, vs.lr_color = imgui.slider_float(
            "Color Learning Rate", vs.lr_color, 0.0, 0.1
        )
        _, vs.lr_opacity = imgui.slider_float(
            "Opacity Learning Rate", vs.lr_opacity, 0.0, 0.1
        )
        _, vs.lr_scale = imgui.slider_float(
            "Scale Learning Rate", vs.lr_scale, 0.0, 0.1
        )
        _, vs.iterations = imgui.slider_int("Iteration Count", vs.iterations, 0, 10)

        # Physics Parameters
        imgui.spacing()
        imgui.spacing()
        imgui.text("Physics Parameters")
        imgui.separator()

        ps = self.env.physics_settings
        imgui.text(f"Simulation Rate: {round(1.0 / ps.dt)} Hz")
        _, ps.xpbd_iterations = imgui.slider_int(
            "XPBD Iterations", ps.xpbd_iterations, 1, 100
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip("Number of position-based dynamics iterations")
        _, ps.substeps = imgui.slider_int("Physics Substeps", ps.substeps, 2, 100)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Number of physics simulation steps per frame")

        imgui.pop_style_var(2)

        imgui.spacing()
        imgui.text("Scene Settings")
        imgui.separator()

        imgui.push_style_var(imgui.StyleVar_.frame_padding, (8, 4))
        imgui.push_style_var(imgui.StyleVar_.button_text_align, (0.5, 0.5))

        imgui.push_style_color(imgui.Col_.button, (0.2, 0.5, 0.8, 0.8))
        imgui.push_style_color(imgui.Col_.button_hovered, (0.3, 0.6, 0.9, 1.0))
        imgui.push_style_color(imgui.Col_.button_active, (0.1, 0.4, 0.7, 1.0))

        if imgui.button("Stash State", (120, 30)):
            self.env.stash_state()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Set the state that will be restored when the scene is reset")

        if imgui.button("Reset##reset_scene", (120, 30)):
            self.env.restore_state()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Restore the state that was stashed when the scene was reset")

        if imgui.button("Save Scene", (120, 30)) and self.env is not None:
            self.save_dialog = pfd.save_file("Save Scene")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Save a builder of the current scene to a file")

        imgui.pop_style_var(2)
        imgui.pop_style_color(3)

        if self.save_dialog is not None and self.save_dialog.ready():
            result = self.save_dialog.result()
            if result:
                result = Path(result)
                self.env.save_builder(result)
            self.save_dialog = None

        imgui.end()

    @override
    def reset_camera(self):
        self._camera_pos = PyVec3(0.0, -0.8, 0.4)
        self._camera_front = PyVec3(0.0, 1.0, 0.0)
        self._camera_up = PyVec3(0.0, 0.0, 1.0)
        self._render_new_frame = True
        self.update_view_matrix()
        self.update_projection_matrix()

    def reset_cameras(self):
        for cam in self.cameras.values():
            cam.timestamp = -1.0

    def configure_side_view(
        self,
        *,
        position: tuple[float, float, float],
        front: tuple[float, float, float],
        up: tuple[float, float, float] = (0.0, 0.0, 1.0),
        activate: bool = False,
    ) -> None:
        """Register a named side view without creating another render path."""
        position_values = np.asarray(position, dtype=np.float64)
        front_values = np.asarray(front, dtype=np.float64)
        up_values = np.asarray(up, dtype=np.float64)
        if not (
            position_values.shape == (3,)
            and front_values.shape == (3,)
            and up_values.shape == (3,)
            and np.isfinite(position_values).all()
            and np.isfinite(front_values).all()
            and np.isfinite(up_values).all()
        ):
            raise ValueError("Side-view position/front/up must be finite 3-vectors")
        front_norm = float(np.linalg.norm(front_values))
        up_norm = float(np.linalg.norm(up_values))
        if front_norm <= 1.0e-12 or up_norm <= 1.0e-12:
            raise ValueError("Side-view front and up vectors must be non-zero")
        front_values /= front_norm
        up_values /= up_norm
        if abs(float(np.dot(front_values, up_values))) >= 1.0 - 1.0e-6:
            raise ValueError("Side-view front and up vectors must not be parallel")
        self._side_view_pose = (
            PyVec3(*position_values.tolist()),
            PyVec3(*front_values.tolist()),
            PyVec3(*up_values.tolist()),
        )
        if activate:
            self.go_to_side_view()

    def go_to_side_view(self) -> bool:
        """Switch the shared 3-D viewport to the configured physical side view."""
        if self._side_view_pose is None:
            return False
        position, front, up = self._side_view_pose
        self._camera_pos = PyVec3(position.x, position.y, position.z)
        self._camera_front = PyVec3(front.x, front.y, front.z)
        self._camera_up = PyVec3(up.x, up.y, up.z)
        self._render_new_frame = True
        self.update_view_matrix()
        self.update_projection_matrix()
        return True

    def render_cameras(self):
        assert self.env is not None
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glDisable(gl.GL_DEPTH_TEST)
        frames = self.env.frames
        if frames is None:
            return
        for i, name in enumerate(frames.names):
            if name not in self.cameras:
                self.cameras[name] = marsoom.CameraWireframeWithImage(
                    width=frames.width,
                    height=frames.height,
                    K=frames.Ks_cpu[i].numpy(),
                    batch=self.batch_cameras,
                    alpha=self.settings.wireframe_alpha,
                    texture_fmt=gl.GL_BGR,
                )
                _set_calibrated_camera_image_geometry(
                    self.cameras[name],
                    frames.Ks_cpu[i].numpy(),
                    frames.width,
                    frames.height,
                )
                self.cameras[name].matrix = pyglet.math.Mat4(
                    *frames.X_WCs_cpu[i].T.flatten().numpy().tolist()
                )
                self.cameras[name].timestamp = -1.0
            camera = self.cameras[name]
            if camera.timestamp != frames.timestamps[i]:
                camera.update_image(frames.colors_gpu[i])
                camera.set_texture_id(camera.texture.id)  # sync after potential resize
                camera.timestamp = frames.timestamps[i]
        self.batch_cameras.draw()
        gl.glEnable(gl.GL_DEPTH_TEST)

    def render_virtual_camerawireframes(self):
        assert self.env is not None
        cameras = self.env.virtual_cameras
        if cameras is None:
            return

        gl.glDisable(gl.GL_CULL_FACE)
        gl.glDisable(gl.GL_DEPTH_TEST)
        X_WCs = cameras.X_WC.cpu().numpy()
        self.env.render_virtual_cameras()
        for j in range(self.env.num_envs()):
            for i, name in enumerate(cameras.names):
                camera_key = f"{name}_{j}"
                if camera_key not in self.virtual_cameras:
                    self.virtual_cameras[camera_key] = marsoom.CameraWireframeWithImage(
                        width=cameras.width,
                        height=cameras.height,
                        K=cameras.K_cpu[i].numpy(),
                        batch=self.batch_virtual_cameras,
                        alpha=self.settings.wireframe_alpha,
                        texture_fmt=gl.GL_BGR,
                    )
                    _set_calibrated_camera_image_geometry(
                        self.virtual_cameras[camera_key],
                        cameras.K_cpu[i].numpy(),
                        cameras.width,
                        cameras.height,
                    )
                    self.virtual_cameras[camera_key].timestamp = -1.0
                camera = self.virtual_cameras[camera_key]
                X_WC = X_WCs[j, i]
                t_WC = self.env_xforms_numpy[j][:3]
                X_WC[:3, 3] += t_WC
                camera.matrix = pyglet.math.Mat4(*X_WC.T.flatten().tolist())
                if camera.timestamp != cameras.last_rendered_at:
                    camera.update_image(cameras.rendered_images[j, i])
                    camera.set_texture_id(camera.texture.id)  # sync after potential resize
                    camera.timestamp = cameras.last_rendered_at
        self.batch_virtual_cameras.draw()
        gl.glEnable(gl.GL_DEPTH_TEST)

    def go_to_camera(self, camera_number: int):
        assert self.env is not None
        frames = self.env.frames
        if frames is None:
            return
        if camera_number >= len(frames.names):
            return
        X_WC = frames.X_WCs_cpu[camera_number]
        K = frames.Ks_cpu[camera_number]
        source_width = float(frames.width)
        source_height = float(frames.height)
        viewport_width = float(max(self.screen_width, 1))
        viewport_height = float(max(self.screen_height, 1))
        contain_scale = min(
            viewport_width / source_width,
            viewport_height / source_height,
        )
        scale = contain_scale * float(self.camera_go_zoom)
        fitted_width = source_width * scale
        fitted_height = source_height * scale
        viewport_cx = 0.5 * (viewport_width - fitted_width) + float(K[0, 2]) * scale
        viewport_cy = 0.5 * (viewport_height - fitted_height) + float(K[1, 2]) * scale
        self.go_to_view(
            x_wv=X_WC.numpy(),
            fx=float(K[0, 0]) * scale,
            fy=float(K[1, 1]) * scale,
            cx=viewport_cx,
            cy=viewport_cy,
            h=int(viewport_height),
            w=int(viewport_width),
        )

    def render_visual_forces(self):
        s = self.settings
        if (
            not s.draw_visual_forces_gaussians_outlines
            and not s.draw_visual_forces_gaussians_meshes
            and not s.draw_visual_forces_render
            and not s.draw_visual_forces
        ):
            return

        gl.glEnable(gl.GL_DEPTH_TEST)
        assert self.env is not None
        ss = self.env.sim
        meta = None
        draw_force_gaussians = (
            s.draw_visual_forces_gaussians_outlines
            or s.draw_visual_forces_gaussians_meshes
            or s.draw_visual_forces_render
        )
        if draw_force_gaussians:
            X_CWs = torch.tensor(self.x_vw("opencv")).cuda().unsqueeze(0)
            Ks = torch.tensor(self.K()).cuda().unsqueeze(0)
            _, _, meta = ss.render_visual_forces(
                X_CWs=X_CWs,
                Ks=Ks,
                width=self.screen_width,
                height=self.screen_height,
                background=torch.tensor([1.0, 1.0, 1.0]).cuda().unsqueeze(0),
            )
        involved = ~ss.visual_forces._gaussians_not_involved_in_visual_forces
        if meta is None:
            ids = torch.nonzero(involved, as_tuple=False).flatten()
        else:
            ids = meta.get("gaussian_ids")
            if ids is None:
                ids = torch.arange(
                    ss.visual_forces.means.shape[0],
                    device=ss.visual_forces.means.device,
                )
            ids = ids[involved[ids]]

        with torch.no_grad():
            preview_positions = ss.gaussian_state.means[ids] + (
                s.visual_forces_pose_scale
                * (ss.visual_forces.means[ids] - ss.gaussian_state.means[ids])
            )
            force_colors = torch.zeros_like(ss.gaussian_state.colors[ids])
            force_colors[:, 0] = 1.0
            force_colors[:, 1] = 0.2
            force_colors[:, 2] = 0.02
            if s.draw_visual_forces_gaussians_meshes:
                self.mesh_ellipse_renderer.update(
                    positions=preview_positions,
                    rotations=ss.visual_forces.quats[ids],
                    scaling=ss.gaussian_state.scales[ids],
                    colors=force_colors,
                )
                self.mesh_ellipse_renderer.draw()
            if s.draw_visual_forces_gaussians_outlines:
                assert meta is not None
                self.ellipse_renderer.update(
                    positions=ss.visual_forces.means[ids],
                    colors=force_colors,
                    opacity=ss.gaussian_state.opacities[ids].unsqueeze(1),
                    conics=meta["conics"].reshape(-1, 3),
                )
                self.ellipse_renderer.draw(3.0)

            if s.draw_visual_forces:
                # Preserve the historical rigid-Gaussian force display, but do
                # not send soft Gaussians (body_id=-1) through its zero-valued
                # rigid-force array.
                rigid_ids = ids[ss.gaussian_model.body_ids[ids] >= 0]
                rigid_positions, rigid_directions = _build_arrow_segments(
                    ss.gaussian_state.means[rigid_ids],
                    ss.visual_forces.forces[rigid_ids],
                    vector_scale=s.visual_forces_scale,
                )
                if len(rigid_positions) > 0:
                    self.vector_renderer.update(
                        positions=rigid_positions,
                        directions=rigid_directions,
                    )
                    self.vector_renderer.draw(
                        vector_scale=s.visual_forces_scale,
                        line_width=2.0,
                    )

                # Soft visual forces bypass body_f: they are barycentrically
                # scattered and globally clamped before entering particle_f.
                # Draw that actual post-clamp value at the PBD particle, not
                # the old per-Gaussian rigid-force buffer.
                self.last_soft_force_arrow_count = 0
                scatter = ss.last_soft_force_scatter
                if scatter is not None and ss.model.particle_count > 0:
                    applied_forces = (
                        scatter.particle_forces * scatter.global_scale.reshape(1, 1)
                    )
                    force_norms = torch.linalg.vector_norm(applied_forces, dim=1)
                    dynamic = wp.to_torch(ss.model.particle_inv_mass) > 0.0
                    supported = ss._soft_particle_weight_sums > 0.0
                    candidates = torch.nonzero(
                        dynamic
                        & supported
                        & torch.isfinite(force_norms)
                        & (force_norms > 1.0e-12),
                        as_tuple=False,
                    ).flatten()
                    if len(candidates) > s.soft_force_max_arrows:
                        strongest = torch.topk(
                            force_norms[candidates],
                            k=s.soft_force_max_arrows,
                            sorted=False,
                        ).indices
                        candidates = candidates[strongest]
                    particle_positions = wp.to_torch(ss.state_0.particle_q)[candidates]
                    particle_directions = (
                        applied_forces[candidates] * s.soft_force_display_gain
                    )
                    soft_positions, soft_directions = _build_arrow_segments(
                        particle_positions,
                        particle_directions,
                        vector_scale=s.visual_forces_scale,
                    )
                    if len(soft_positions) > 0:
                        self.vector_renderer.update(
                            positions=soft_positions,
                            directions=soft_directions,
                        )
                        self.vector_renderer.draw(
                            vector_scale=s.visual_forces_scale,
                            line_width=s.soft_force_line_width,
                        )
                        self.last_soft_force_arrow_count = len(candidates)
            else:
                self.last_soft_force_arrow_count = 0

        # if s.draw_gaussian_render:
        #     self.gaussian_texture.copy_from_device(render_colors.squeeze(0))
        #     self.gaussian_overlay.draw()

    def render_gaussians(self):
        s = self.settings
        if (
            not s.draw_gaussian_meshes
            and not s.draw_gaussian_outlines
            and not s.draw_gaussian_render
        ):
            return

        X_CWs = torch.tensor(self.x_vw("opencv")).cuda().unsqueeze(0)
        Ks = torch.tensor(self.K()).cuda().unsqueeze(0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        assert self.env is not None
        sim = self.env.sim

        self._refresh_gaussian_state()
        render_colors, render_alphas, meta = sim.render_gaussians(
            self.gaussian_render_state,
            # sim.gaussian_state,
            X_CWs=X_CWs,
            Ks=Ks,
            width=self.screen_width,
            height=self.screen_height,
            background=torch.tensor([1.0, 1.0, 1.0]).cuda().unsqueeze(0),
            near_plane=s.near_plane,
            far_plane=s.far_plane,
        )

        ids = meta.get("gaussian_ids")
        if ids is None:
            ids = torch.arange(
                sim.gaussian_state.means.shape[0], device=sim.gaussian_state.means.device
            )
        if len(ids) == 0:
            return

        with torch.no_grad():
            sim = self.env.sim
            if s.draw_gaussian_meshes:
                self.mesh_ellipse_renderer.update(
                    positions=sim.gaussian_state.means,
                    rotations=sim.gaussian_state.quats,
                    scaling=sim.gaussian_state.scales,
                    colors=sim.gaussian_state.colors,
                )
                self.mesh_ellipse_renderer.draw()
            if s.draw_gaussian_outlines:
                self.ellipse_renderer.update(
                    positions=sim.gaussian_state.means[ids],
                    colors=sim.gaussian_state.colors[ids],
                    opacity=sim.gaussian_state.opacities[ids].unsqueeze(1),
                    conics=meta["conics"].reshape(-1, 3),
                )
                self.ellipse_renderer.draw(3.0)
            if s.draw_gaussian_render:
                self.gaussian_texture.copy_from_device(render_colors.squeeze(0))
                self.gaussian_overlay.tex_id = self.gaussian_texture.id  # sync after potential resize
                self.gaussian_overlay.draw()

    def render(self):
        if self.env is None:
            return
        sim_reconstruction_mode = bool(
            getattr(self.env, "super_sim_reconstruction_mode", False)
        )
        if self.settings.draw_physics and not sim_reconstruction_mode:
            self.render_meshes()
        self.render_gaussians()
        # Draw the orange deformable skin after the 2-D Gaussian overlay so it
        # remains visible as a physics diagnostic. The soft-only renderer has
        # shape_count=0, hence it cannot contain the old PSM.
        if self.settings.draw_physics and sim_reconstruction_mode:
            self.render_soft_meshes()
        self.render_visual_forces()
        if self.settings.draw_cameras:
            self.render_cameras()
        if self.settings.draw_virtual_cameras:
            self.render_virtual_camerawireframes()
        if self.settings.draw_gaussian_render:
            self.gaussian_overlay.draw()
        gl.glLineWidth(1.0)

    def _refresh_gaussian_state(self):
        assert self.env is not None
        means = self.env.sim.gaussian_state.means.reshape(self.env.num_envs(), -1, 3)
        self.gaussian_render_state.copy(self.env.sim.gaussian_state)
        render_means = self.gaussian_render_state.means.reshape(
            self.env.num_envs(), -1, 3
        )
        wp.launch(
            kernel=transform_gaussian_to_env_state_kernel,
            dim=(self.env.num_envs(), means.shape[1]),
            inputs=[
                self.env_xforms,
                means,
                render_means,
            ],
        )


@wp.kernel
def transform_gaussian_to_env_state_kernel(
    T_WE: wp.array(dtype=wp.transformf),  # type: ignore
    means: wp.array(ndim=2, dtype=wp.vec3f),  # type: ignore
    out_means: wp.array(ndim=2, dtype=wp.vec3f),  # type: ignore
):
    env_id, tid = wp.tid()  # type: ignore
    out_means[env_id, tid] = wp.transform_point(T_WE[env_id], means[env_id, tid])
