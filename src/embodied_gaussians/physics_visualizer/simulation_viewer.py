# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import copy

import marsoom
import marsoom.cuda
import numpy as np
import torch
import warp as wp
import warp.sim.render
from marsoom import guizmo, imgui

from embodied_gaussians.physics_simulator.simulator import Simulator
from embodied_gaussians.utils.utils import GridBuilder
from embodied_gaussians.utils.physics_utils import transform_from_matrix, transform_to_matrix


class SimulationViewer(marsoom.Viewer3D):
    def __init__(self, window, show_origin: bool = True):
        super().__init__(window, show_origin)
        self.simulator = None
        self.enable_manipulate = False
        self.manipulate_operation = guizmo.OPERATION.translate
        self.manipulate_mode = guizmo.MODE.local

    def set_simulator(self, simulator: Simulator):
        self.simulator = simulator
        # The renderer gets a shallow model copy so debug-only geometry cannot
        # alter contact radii or triangle constraints in the live simulation.
        render_model = copy.copy(self.simulator.model)
        render_model.particle_radius = getattr(
            self.simulator.model,
            "particle_visual_radius",
            self.simulator.model.particle_radius,
        )
        collision_skin_faces = np.asarray(
            getattr(self.simulator.builder, "soft_collision_skin_faces", []),
            dtype=np.int32,
        ).reshape(-1, 3)
        if len(collision_skin_faces):
            render_model.tri_indices = wp.array(
                collision_skin_faces,
                dtype=wp.vec3i,
                device=self.simulator.model.device,
            )
            render_model.tri_count = len(collision_skin_faces)
        self.sim_renderer = warp.sim.render.CreateSimRenderer(
            marsoom.cuda.OpenGLRendererWrapper
        )(render_model, 0)
        # A second debug renderer contains only deformable particles/skin.
        # Dataset-reconstruction mode needs this view without reviving any of
        # the legacy SUPER rigid shapes (PSM, table, background bodies).
        soft_render_model = copy.copy(render_model)
        soft_render_model.shape_count = 0
        soft_render_model.ground = False
        self.soft_sim_renderer = warp.sim.render.CreateSimRenderer(
            marsoom.cuda.OpenGLRendererWrapper
        )(soft_render_model, 0)
        self.render_state = self.simulator.model.state()
        self.num_bodies = self.simulator.model.body_count
        self.body_id = 0
        self._create_env_xforms()

    def render_meshes(self):
        if self.simulator is None:
            return
        self._refresh_body_q()
        self.sim_renderer.render(self.render_state)
        self.sim_renderer.draw()

    def render_soft_meshes(self):
        """Render deformable particles/skin while hiding every rigid shape."""
        if self.simulator is None:
            return
        self._refresh_body_q()
        self.soft_sim_renderer.render(self.render_state)
        self.soft_sim_renderer.draw()

    def render_manipulation(self):
        if self.simulator is None:
            return

        self.keyboard()
        if self.enable_manipulate:
            env_id = self.body_id // self.bodies_per_env
            X_WO = transform_to_matrix(
                wp.to_torch(self.render_state.body_q)[self.body_id]
                .detach()
                .cpu()
                .numpy()
            )
            guizmo.set_id(100)
            c, X_WO = self.manipulate(
                X_WO, operation=self.manipulate_operation, mode=self.manipulate_mode
            )
            if c:
                T_WO = wp.transformf(*transform_from_matrix(X_WO).tolist())
                T_EW = wp.transform_inverse(self.env_xforms_numpy[env_id])
                T_EO = T_EW * T_WO
                wp.to_torch(self.simulator.state_0.body_q)[self.body_id] = torch.tensor(
                    T_EO
                ).cuda()

    def keyboard(self):
        if imgui.is_key_pressed(imgui.Key.m):
            self.enable_manipulate = not self.enable_manipulate
        if imgui.is_key_pressed(imgui.Key.g):
            if self.manipulate_operation == guizmo.OPERATION.translate:
                if self.manipulate_mode == guizmo.MODE.world:
                    self.manipulate_mode = guizmo.MODE.local
                else:
                    self.manipulate_mode = guizmo.MODE.world
            else:
                self.manipulate_operation = guizmo.OPERATION.translate
        if imgui.is_key_pressed(imgui.Key.r):
            if self.manipulate_operation == guizmo.OPERATION.rotate:
                if self.manipulate_mode == guizmo.MODE.world:
                    self.manipulate_mode = guizmo.MODE.local
                else:
                    self.manipulate_mode = guizmo.MODE.world
            else:
                self.manipulate_operation = guizmo.OPERATION.rotate
        if imgui.is_key_pressed(imgui.Key.keypad_add):
            self.body_id = (self.body_id + 1) % self.num_bodies
        if imgui.is_key_pressed(imgui.Key.keypad_subtract):
            self.body_id = (self.body_id - 1) % self.num_bodies

    def _create_env_xforms(self):
        num_envs = self.simulator.model.num_envs
        self.env_xforms = []
        grid = iter(GridBuilder())
        for i in range(num_envs):
            xform = wp.transformf(next(grid), wp.quat_identity(float))
            self.env_xforms.append(xform)
        self.env_xforms = wp.array(self.env_xforms, dtype=wp.transformf)
        self.env_xforms_numpy = wp.array(self.env_xforms, dtype=wp.transformf).numpy()
        self.bodies_per_env = self.simulator.model.body_count // num_envs

    def _refresh_body_q(self):
        body_q = self.simulator.state_0.body_q
        wp.launch(
            kernel=transform_to_env_state_kernel,
            dim=(body_q.shape[0],),
            inputs=[
                self.bodies_per_env,
                self.env_xforms,
                body_q,
                self.render_state.body_q,
            ],
        )
        # CreateSimRenderer owns a separate State.  Body transforms were
        # refreshed above, but soft particles previously stayed at the
        # model's rest positions forever.  That made the debug tissue skin and
        # "Physics Particles" look rigid even while state_0 was deforming.
        # SUPER soft tissue currently uses one environment, so its particle
        # coordinates already match the render world's coordinates.
        if self.simulator.model.particle_count > 0:
            wp.copy(
                self.render_state.particle_q,
                self.simulator.state_0.particle_q,
            )


@wp.kernel
def transform_to_env_state_kernel(
    bodies_per_env: int,
    X_WE: wp.array(dtype=wp.transformf),
    body_q: wp.array(dtype=wp.transformf),
    out_body_q: wp.array(dtype=wp.transformf),
):
    tid = wp.tid()
    env_id = tid / bodies_per_env
    out_body_q[tid] = X_WE[env_id] * body_q[tid]
