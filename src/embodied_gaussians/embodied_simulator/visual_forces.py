# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
import torch
import warp as wp

from embodied_gaussians.embodied_simulator.adam import Adam
from embodied_gaussians.embodied_simulator.gaussians import GaussianModel, GaussianState


@dataclass
class VisualForcesSettings:
    # 每个 physics step 内，视觉优化要迭代多少次。
    # 次数越大，视觉对齐通常越强，但代价是更慢。
    iterations: int = 3
    # 下面这些学习率控制“允许视觉误差推动哪些量变化”。
    # 常见做法是只让 means / quats 变化，把颜色、透明度、尺度固定住，
    # 这样得到的是“位姿修正力”，而不是把外观直接拟合到图像里。
    lr_means: float = 0.0015
    lr_quats: float = 0.001
    lr_color: float = 0.000
    lr_opacity: float = 0.0000
    lr_scale: float = 0.0000
    # 把“高斯位姿偏差”转成“物理受力/力矩”时使用的比例系数。
    kp: float = 4.0
    # OfflineCamera currently exposes BGR frames for OpenGL display, while
    # gsplat renders RGB. Dataset adapters can enable this conversion for the
    # photometric loss without changing the camera display path.
    observations_are_bgr: bool = False
    # A visual-force solve starts from the current physical pose every frame.
    # Carrying Adam momentum across these independent solves can create drift.
    reset_optimizer_each_step: bool = False
    # Summed per-Gaussian forces otherwise scale with representation density.
    normalize_forces_by_gaussian_count: bool = False
    # Zero disables the corresponding safety clamp.
    max_force: float = 0.0
    max_moment: float = 0.0
    # Positive values use Smooth L1 instead of unbounded pixel MSE.
    robust_loss_beta: float = 0.0
    # Soft Gaussian residuals can be scattered to tetrahedral particles only
    # after a tissue/tool/visibility pixel weighting map is available.
    enable_soft_particle_forces: bool = False
    require_loss_weights_for_soft: bool = True
    soft_max_gaussian_force: float = 0.0
    soft_max_particle_force: float = 0.0
    soft_max_total_force: float = 0.0
    # A uniform force cap can violently accelerate tiny tetrahedral nodes.
    # Positive values additionally enforce |f_i| / m_i <= this limit.
    soft_max_particle_acceleration: float = 0.0
    soft_force_spread_layers: int = 0


class VisualForces:
    # 这一类维护的是“视觉优化过程中的临时变量”，不是场景真值本身。
    # 典型流程是：
    # 1. 从当前 gaussian_state 拷一份 means / quats 到这里
    # 2. 用渲染误差对这份拷贝做若干步梯度优化
    # 3. 把优化前后差值转成 forces / moments
    # 4. 再把这些力施加回物理系统
    means: torch.Tensor  # (n_gaussians, 3)
    quats: torch.Tensor  # (n_gaussians, 4) (w, x, y, z)
    forces: torch.Tensor  # (n_gaussians, 3)
    moments: torch.Tensor  # (n_gaussians, 3)

    def __init__(
        self,
        gaussian_model: GaussianModel,
        gaussian_state: GaussianState,
        bodies_affected_by_visual_forces: list[int],
    ):
        # 为每个 Gaussian 分配一份“可求梯度的临时位姿副本”和最终输出的力/力矩。
        num_gaussians = gaussian_model.means.shape[0]
        device = gaussian_model.means.device
        self.device = device
        self.means = torch.zeros((num_gaussians, 3), dtype=torch.float32, device=device)
        self.quats = torch.zeros((num_gaussians, 4), dtype=torch.float32, device=device)
        self.forces = torch.zeros(
            (num_gaussians, 3), dtype=torch.float32, device=device
        )
        self.moments = torch.zeros(
            (num_gaussians, 3), dtype=torch.float32, device=device
        )
        self.means.requires_grad = True
        self.quats.requires_grad = True

        # 预计算“每个 body 在 Gaussian 数组中对应的连续段”，
        # 这样后面可以把 per-Gaussian 的力快速聚合成 per-body 的总力。
        self._initialize(gaussian_model.body_ids)
        self._gaussian_body_ids = gaussian_model.body_ids
        self.configure_body_participation(
            gradient_body_ids=bodies_affected_by_visual_forces,
            physics_force_body_ids=bodies_affected_by_visual_forces,
        )

        # 这里不是标准 torch.optim.Adam，而是项目里包过的 Warp 版本 Adam。
        # 它直接吃 Warp/Torch 桥接后的向量数组，便于和后面的 kernel 配合。
        self.optimizer = Adam(
            [
                wp.from_torch(self.means, dtype=wp.vec3),
                wp.from_torch(self.quats, dtype=wp.vec4),
            ],
            lrs=[0.01, 0.01],
        )
        self.gaussian_state = gaussian_state

    def configure_body_participation(
        self,
        gradient_body_ids: list[int],
        physics_force_body_ids: list[int],
    ) -> None:
        gradient_mask = torch.zeros_like(self._gaussian_body_ids, dtype=torch.bool)
        for body_id in gradient_body_ids:
            gradient_mask |= self._gaussian_body_ids == int(body_id)
        self._gaussians_not_involved_in_visual_forces = ~gradient_mask

        physics_mask = torch.zeros_like(self._body_ids, dtype=torch.bool)
        for body_id in physics_force_body_ids:
            physics_mask |= self._body_ids == int(body_id)
        self._apply_physics_forces = physics_mask.to(dtype=torch.int32)

    def configure_gaussian_participation(
        self, gradient_gaussian_ids: torch.Tensor
    ) -> None:
        """Allow pose gradients only for explicitly listed Gaussian indices."""
        ids = gradient_gaussian_ids.to(device=self.device, dtype=torch.long)
        if ids.ndim != 1:
            raise ValueError("gradient_gaussian_ids must be one-dimensional")
        if len(ids) > 0 and (
            int(ids.min().item()) < 0 or int(ids.max().item()) >= len(self.means)
        ):
            raise ValueError("gradient Gaussian id is outside the model")
        self._gaussians_not_involved_in_visual_forces.fill_(True)
        self._gaussians_not_involved_in_visual_forces[ids] = False

    def set_learnings_rates(self, lrs):
        # 每个 step 前根据 VisualForcesSettings 动态更新学习率。
        self.optimizer.lrs = lrs

    def zero_grad(self):
        # 每轮反传前都要清空梯度缓存。
        self.means.grad.zero_()
        self.quats.grad.zero_()

    def step(self):
        # 只允许指定 body 上的 Gaussian 通过视觉误差更新。
        # 其他 Gaussian 即使算出了梯度，也在这里强制清零。
        self.means.grad[self._gaussians_not_involved_in_visual_forces, :] = 0
        self.quats.grad[self._gaussians_not_involved_in_visual_forces, :] = 0
        self.optimizer.step(
            grad=[
                wp.from_torch(self.means.grad, dtype=wp.vec3),
                wp.from_torch(self.quats.grad, dtype=wp.vec4),
            ]
        )

    def _initialize(self, body_ids: torch.Tensor):
        # body_ids 通常已经按 body 分组排列。
        # 这里把它转换成若干连续段：
        # [start_i, end_i) 表示第 i 个 body 对应哪些 Gaussian。
        #
        # 后面 pysegreduce 会利用这些分段，把 per-Gaussian 力聚合成 per-body 力。
        if len(body_ids) == 0:
            return
        starts = body_ids[1:] - body_ids[:-1] != 0
        start_inds = torch.nonzero(starts).squeeze() + 1
        start_inds = start_inds.reshape(-1).tolist()
        start_inds = [0] + start_inds
        end_inds = start_inds[1:] + [len(body_ids)]
        bids = body_ids[start_inds]
        mask = bids != -1
        start_inds = torch.tensor(
            start_inds, device=self.means.device, dtype=torch.int32
        )
        end_inds = torch.tensor(end_inds, device=self.means.device, dtype=torch.int32)

        self._start_inds = start_inds[mask]
        self._end_inds = end_inds[mask]
        self._body_ids = bids[mask]
        self._gaussian_counts = (self._end_inds - self._start_inds).to(
            dtype=torch.int32
        )

        self._num_bodies = len(self._start_inds)

        # 聚合后的总力 / 总力矩，每个 body 一条。
        self._total_forces = torch.zeros(
            (self._num_bodies, 3), device=self.device, dtype=torch.float32
        )
        self._total_moments = torch.zeros(
            (self._num_bodies, 3), device=self.device, dtype=torch.float32
        )
