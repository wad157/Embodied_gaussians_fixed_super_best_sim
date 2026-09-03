# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field, replace
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trio
import warp as wp

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent

# 优先导入当前仓库源码，避免运行到系统里旧安装的 embodied_gaussians。
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(current_dir))

from embodied_environments.super_embodied.super_embodied import (  # noqa: E402
    PSM_ARTICULATION_INDEX,
    PSM_LND_POSE_DRIVER_PATH,
    PSM_POSE_DRIVER_PATHS,
    apply_psm_lnd_pose,
    build_environment,
    expand_psm_q7_to_urdf_order,
    load_mimic_config,
    set_psm_tissue_collisions,
    urdf_actuated_joint_order,
)
from embodied_gaussians import DatasetManager, EmbodiedGaussiansEnvironment  # noqa: E402
from embodied_gaussians.embodied_simulator.visual_force_masks import (  # noqa: E402
    MultiCameraPackedTissueVisualForceWeights,
)
from embodied_gaussians.physics_simulator.visual_tissue_residual_mapping import (  # noqa: E402
    TetrahedralGaussianVisualResidualMapper,
    VisualTissueResidualMappingSettings,
    strict_one_tetrahedron_ring_mask,
)
from embodied_gaussians.physics_simulator.flow_depth_particle_observer import (  # noqa: E402
    FlowDepthObservationSequence,
    FlowDepthParticleRangeBindings,
    FlowDepthStateUpdateSettings,
    compute_flow_depth_particle_state_update,
    fixed_range_centers,
    load_fixed_particle_range_bindings,
    load_flow_depth_observation_sequence,
)
from embodied_gaussians.physics_simulator.online_tissue_stiffness import (  # noqa: E402
    OnlineTissueStiffnessSettings,
    PaperStiffnessCandidate,
    ResidualDrivenPaperStiffnessUpdater,
)
from embodied_gaussians.physics_simulator.stiffness_evaluation import (  # noqa: E402
    StiffnessActionPhaseClassifier,
    StiffnessMetricsRecorder,
    parse_stiffness_evaluation_horizons,
)
from embodied_gaussians.physics_simulator.sim_benchmark import (  # noqa: E402
    SimBenchmarkArtifactWriter,
)
import embodied_gaussians as embodied_gaussians_package  # noqa: E402


LOADED_EMBODIED_GAUSSIANS_SOURCE = Path(
    embodied_gaussians_package.__file__
).resolve().parent
EXPECTED_EMBODIED_GAUSSIANS_SOURCE = (
    repo_root / "src/embodied_gaussians"
).resolve()
# A residual can still be visually acceptable around a tet that was already
# compressed by contact. Mask only particles incident on such a tet instead of
# allowing one global minimum to disable stiffness learning everywhere.
STIFFNESS_UPDATE_LOCAL_MINIMUM_VOLUME_RATIO = 0.03
# Preliminary validation gates.  They are deliberately centralized so a
# frozen replay can calibrate them instead of scattering magic values through
# the runtime loop.
STIFFNESS_VISUAL_GRADIENT_RELATIVE_FLOOR = 1.0e-4
# Keep proposal admission nearly open. Fast jaw motion is still excluded from
# history snapshots below, but it no longer suppresses a new candidate by
# itself; the next-image shadow rollout decides whether that candidate is real.
STIFFNESS_MAXIMUM_JAW_SPEED_RAD_S: float | None = None
STIFFNESS_TRANSITION_COOLDOWN_UPDATES = 1
# Align material-learning admission with the configured grip capture envelope.
STIFFNESS_MAXIMUM_PENETRATION_M = 0.0028
# The cumulative H=5 shadow exposed 20 commits at a 1e-6 floor, but most of
# their apparent gains were only optimizer/render noise and the 211-frame mean
# regressed by 0.0523%. Ten micro-loss is about 0.05% of the observed loss and
# retains the loose contact gate while requiring material evidence above that
# measured noise floor.
STIFFNESS_PREDICTION_ABSOLUTE_MARGIN = 1.0e-5
STIFFNESS_PREDICTION_RELATIVE_MARGIN = 0.0
STIFFNESS_CAMERA_ABSOLUTE_REGRESSION = 1.0e-6
STIFFNESS_CAMERA_RELATIVE_REGRESSION = 5.0e-3
STIFFNESS_MINIMUM_VOLUME_ABSOLUTE_DROP = 0.01
STIFFNESS_MINIMUM_VOLUME_RELATIVE_DROP = 0.02
STIFFNESS_PENETRATION_TOLERANCE_M = 0.0001
STIFFNESS_ANCHOR_ERROR_TOLERANCE_M = 0.00005
STIFFNESS_MAXIMUM_PENDING_ROLLOUT_STEPS = 120
STIFFNESS_MAXIMUM_PREDICTION_HORIZON_FRAMES = 10
# A one-frame gain was too myopic: all 33 loose-gate commits passed H=1, while
# their end-to-end H=3/5/10 means regressed. Retrospective filtering showed H=5
# was the first horizon whose accepted subset stayed positive at available H=10.
STIFFNESS_COMMIT_VALIDATION_HORIZON_FRAMES = 5
STIFFNESS_CANDIDATE_LOG_STEP_SCALES = (0.5, 1.0, 1.5, 2.0)
# Candidate validation replays the last H frames from a saved state. Once a
# candidate is verified, adopt that already-observed fixed-lag state as well as
# its material field; otherwise the visually better branch is discarded and
# only its parameters affect future frames. The bound is far above the measured
# 0.006..0.059 mm branch RMS but below a visibly abrupt tissue jump.
STIFFNESS_ADOPT_VALIDATED_ROLLOUT = True
STIFFNESS_MAXIMUM_ADOPTED_STATE_RMS_M = 0.00025
STIFFNESS_MAXIMUM_ADOPTED_STATE_MAXIMUM_M = 0.001
STIFFNESS_HISTORY_MAXIMUM_SNAPSHOTS = 4
STIFFNESS_HISTORY_RELATIVE_TOLERANCE = 0.10
STIFFNESS_HISTORY_ABSOLUTE_TOLERANCE_M = 1.0e-5
STIFFNESS_HISTORY_MAXIMUM_PARTICLE_SPEED_M_S = 0.02
STIFFNESS_HISTORY_MAXIMUM_JAW_SPEED_RAD_S = 0.10


@dataclass(frozen=True)
class StiffnessToolCommand:
    frame_index: int
    timestep: float
    state_index: int
    phase: str = "idle"


@dataclass
class StiffnessHistorySnapshot:
    rollout_state: object
    accepted_positions: torch.Tensor
    frame_index: int


@dataclass
class FlowDepthSourceSnapshot:
    """Pre-transition state captured after the previous visual correction."""

    rollout_state: object
    positions: np.ndarray
    velocities: torch.Tensor
    physics_iteration_count: int


@dataclass
class WarpGradientTransition:
    """Already observed transition used only for an actual-Warp FD check."""

    source_frame: int
    destination_command: StiffnessToolCommand
    rollout_state: object
    physics_steps: int
    corrected_positions: torch.Tensor
    eligible_mask: torch.Tensor
    material_active_mask: torch.Tensor
    control_exclusion_mask: torch.Tensor
    track_target_positions: torch.Tensor
    track_valid_mask: torch.Tensor


@dataclass
class PendingStiffnessValidation:
    candidate: PaperStiffnessCandidate
    rollout_state: object
    frame_index: int
    grip_active: bool
    history_baseline_rms_m: float
    history_candidate_rms_m: float
    previous_residual: torch.Tensor | None = None
    commands: list[StiffnessToolCommand] = field(default_factory=list)


@dataclass
class CommittedStiffnessEvaluation:
    """One committed candidate awaiting exact multi-horizon evaluation."""

    evaluation_id: int
    candidate: PaperStiffnessCandidate
    rollout_state: object
    start_frame_index: int
    start_phase: str
    commands: list[StiffnessToolCommand]
    pending_horizons: set[int]
    previous_residual: torch.Tensor | None = None


@dataclass(frozen=True)
class ContactGripGuiSettings:
    """Editable contact/grip policy copied from the live runtime."""

    sample_spacing_m: float
    contact_spread_layers: int
    contact_margin_m: float
    query_distance_m: float
    ccd_velocity_scale: float
    contact_relaxation: float
    contact_max_correction_m: float
    top_barrier_max_correction_m: float
    contact_iterations: int
    post_contact_material_iterations: int
    final_barrier_max_correction_m: float
    contact_min_volume_ratio: float
    contact_substep_stride: int
    contact_projection_velocity_scale: float
    material_projection_velocity_scale: float
    particle_velocity_damping_per_second: float
    top_barrier_lateral_tolerance_m: float
    top_barrier_contact_patch_radius_m: float
    top_barrier_clearance_m: float
    jaw_friction_coefficient: float
    jaw_contact_distal_length_m: float
    top_barrier_distal_length_m: float
    top_barrier_tip_allowance_m: float
    persistent_grip_enabled: bool
    grip_minimum_contact_samples_per_jaw: int
    grip_maximum_jaw_patch_separation_m: float
    grip_activation_steps: int
    grip_maximum_capture_penetration_m: float
    grip_minimum_capture_volume_ratio: float
    grip_closed_angle_max_rad: float
    grip_release_angle_min_rad: float
    grip_release_angle_delta_rad: float
    grip_wide_open_angle_rad: float
    grip_angle_motion_epsilon_rad: float
    grip_compliance_m_per_n: float
    grip_relaxation: float
    grip_maximum_correction_m: float
    grip_transfer_layers: int
    grip_minimum_volume_ratio: float
    grip_support_radius_m: float
    grip_support_generations: int

    @classmethod
    def from_runtime(cls, physics, projector) -> ContactGripGuiSettings:
        return cls(
            sample_spacing_m=float(projector.sample_spacing_m),
            contact_spread_layers=int(projector.spread_layers),
            contact_margin_m=float(physics.triangle_skin_contact_margin_m),
            query_distance_m=float(physics.triangle_skin_query_distance_m),
            ccd_velocity_scale=float(
                physics.triangle_skin_ccd_velocity_scale
            ),
            contact_relaxation=float(
                physics.triangle_skin_contact_relaxation
            ),
            contact_max_correction_m=float(
                physics.triangle_skin_contact_max_correction_m
            ),
            top_barrier_max_correction_m=float(
                physics.triangle_skin_top_barrier_max_correction_m
            ),
            contact_iterations=int(
                physics.triangle_skin_contact_iterations
            ),
            post_contact_material_iterations=int(
                physics.triangle_skin_post_contact_material_iterations
            ),
            final_barrier_max_correction_m=float(
                physics.triangle_skin_final_barrier_max_correction_m
            ),
            contact_min_volume_ratio=float(
                physics.triangle_skin_contact_min_volume_ratio
            ),
            contact_substep_stride=int(
                physics.triangle_skin_contact_substep_stride
            ),
            contact_projection_velocity_scale=float(
                physics.contact_projection_velocity_scale
            ),
            material_projection_velocity_scale=float(
                physics.material_projection_velocity_scale
            ),
            particle_velocity_damping_per_second=float(
                physics.particle_velocity_damping_per_second
            ),
            top_barrier_lateral_tolerance_m=float(
                projector.top_barrier_lateral_tolerance_m
            ),
            top_barrier_contact_patch_radius_m=float(
                projector.top_barrier_contact_patch_radius_m
            ),
            top_barrier_clearance_m=float(
                projector.top_barrier_clearance_m
            ),
            jaw_friction_coefficient=float(
                projector.jaw_friction_coefficient
            ),
            jaw_contact_distal_length_m=float(
                projector.jaw_contact_distal_length_m
            ),
            top_barrier_distal_length_m=float(
                projector.top_barrier_distal_length_m
            ),
            top_barrier_tip_allowance_m=float(
                projector.top_barrier_tip_allowance_m
            ),
            persistent_grip_enabled=bool(
                projector.persistent_grip_enabled
            ),
            grip_minimum_contact_samples_per_jaw=int(
                projector.persistent_grip_minimum_contact_samples_per_jaw
            ),
            grip_maximum_jaw_patch_separation_m=float(
                projector.persistent_grip_maximum_jaw_patch_separation_m
            ),
            grip_activation_steps=int(
                projector.persistent_grip_activation_steps
            ),
            grip_maximum_capture_penetration_m=float(
                projector.persistent_grip_maximum_capture_penetration_m
            ),
            grip_minimum_capture_volume_ratio=float(
                projector.persistent_grip_minimum_capture_volume_ratio
            ),
            grip_closed_angle_max_rad=float(
                projector.persistent_grip_closed_angle_max_rad
            ),
            grip_release_angle_min_rad=float(
                projector.persistent_grip_release_angle_min_rad
            ),
            grip_release_angle_delta_rad=float(
                projector.persistent_grip_release_angle_delta_rad
            ),
            grip_wide_open_angle_rad=float(
                projector.persistent_grip_wide_open_angle_rad
            ),
            grip_angle_motion_epsilon_rad=float(
                projector.persistent_grip_angle_motion_epsilon_rad
            ),
            grip_compliance_m_per_n=float(
                projector.persistent_grip_compliance_m_per_n
            ),
            grip_relaxation=float(
                projector.persistent_grip_relaxation
            ),
            grip_maximum_correction_m=float(
                projector.persistent_grip_maximum_correction_m
            ),
            grip_transfer_layers=int(
                projector.persistent_grip_transfer_layers
            ),
            grip_minimum_volume_ratio=float(
                projector.persistent_grip_minimum_volume_ratio
            ),
            grip_support_radius_m=float(
                projector.persistent_grip_support_radius_m
            ),
            grip_support_generations=int(
                projector.persistent_grip_support_generations
            ),
        )

    def validate(self) -> None:
        scalar_values = (
            self.sample_spacing_m,
            self.contact_margin_m,
            self.query_distance_m,
            self.ccd_velocity_scale,
            self.contact_relaxation,
            self.contact_max_correction_m,
            self.top_barrier_max_correction_m,
            self.final_barrier_max_correction_m,
            self.contact_min_volume_ratio,
            self.contact_projection_velocity_scale,
            self.material_projection_velocity_scale,
            self.particle_velocity_damping_per_second,
            self.top_barrier_lateral_tolerance_m,
            self.top_barrier_contact_patch_radius_m,
            self.top_barrier_clearance_m,
            self.jaw_friction_coefficient,
            self.jaw_contact_distal_length_m,
            self.top_barrier_distal_length_m,
            self.top_barrier_tip_allowance_m,
            self.grip_maximum_jaw_patch_separation_m,
            self.grip_maximum_capture_penetration_m,
            self.grip_minimum_capture_volume_ratio,
            self.grip_closed_angle_max_rad,
            self.grip_release_angle_min_rad,
            self.grip_release_angle_delta_rad,
            self.grip_wide_open_angle_rad,
            self.grip_angle_motion_epsilon_rad,
            self.grip_compliance_m_per_n,
            self.grip_relaxation,
            self.grip_maximum_correction_m,
            self.grip_minimum_volume_ratio,
            self.grip_support_radius_m,
        )
        if not all(np.isfinite(value) for value in scalar_values):
            raise ValueError("contact/grip settings must be finite")
        if self.sample_spacing_m <= 0.0:
            raise ValueError("contact sample spacing must be positive")
        if self.contact_spread_layers < 0:
            raise ValueError("contact spread layers cannot be negative")
        if self.contact_margin_m < 0.0:
            raise ValueError("contact margin cannot be negative")
        if self.query_distance_m < self.contact_margin_m:
            raise ValueError("query distance must be at least the contact margin")
        if self.ccd_velocity_scale < 0.0:
            raise ValueError("CCD velocity scale cannot be negative")
        if not 0.0 < self.contact_relaxation <= 1.0:
            raise ValueError("contact relaxation must lie in (0, 1]")
        if min(
            self.contact_max_correction_m,
            self.top_barrier_max_correction_m,
            self.final_barrier_max_correction_m,
        ) < 0.0:
            raise ValueError("contact correction caps cannot be negative")
        if self.contact_iterations < 1:
            raise ValueError("contact iterations must be positive")
        if self.post_contact_material_iterations < 0:
            raise ValueError("post-contact material iterations cannot be negative")
        if self.contact_substep_stride < 1:
            raise ValueError("contact substep stride must be positive")
        if not 0.0 <= self.contact_min_volume_ratio <= 1.0:
            raise ValueError("contact minimum J must lie in [0, 1]")
        if not 0.0 <= self.contact_projection_velocity_scale <= 1.0:
            raise ValueError("contact velocity transfer must lie in [0, 1]")
        if not 0.0 <= self.material_projection_velocity_scale <= 1.0:
            raise ValueError("material velocity transfer must lie in [0, 1]")
        if self.particle_velocity_damping_per_second < 0.0:
            raise ValueError("particle velocity damping cannot be negative")
        if self.top_barrier_lateral_tolerance_m <= 0.0:
            raise ValueError("top barrier lateral tolerance must be positive")
        if min(
            self.top_barrier_contact_patch_radius_m,
            self.top_barrier_clearance_m,
            self.jaw_friction_coefficient,
            self.jaw_contact_distal_length_m,
            self.top_barrier_distal_length_m,
            self.top_barrier_tip_allowance_m,
        ) < 0.0:
            raise ValueError("contact geometry/friction values cannot be negative")
        barrier_distal = (
            self.top_barrier_distal_length_m
            if self.top_barrier_distal_length_m > 0.0
            else self.jaw_contact_distal_length_m
        )
        if barrier_distal > self.jaw_contact_distal_length_m:
            raise ValueError("top barrier length cannot exceed jaw contact length")
        if barrier_distal > 0.0 and self.top_barrier_tip_allowance_m >= barrier_distal:
            raise ValueError("tip entry allowance must be shorter than top barrier")
        if self.grip_minimum_contact_samples_per_jaw < 1:
            raise ValueError("grip samples per jaw must be positive")
        if self.grip_maximum_jaw_patch_separation_m <= 0.0:
            raise ValueError("grip patch separation must be positive")
        if self.grip_activation_steps < 1:
            raise ValueError("grip activation steps must be positive")
        if self.grip_maximum_capture_penetration_m <= 0.0:
            raise ValueError("grip capture penetration must be positive")
        if not 0.0 < self.grip_minimum_capture_volume_ratio <= 1.0:
            raise ValueError("grip capture minimum J must lie in (0, 1]")
        if not (
            self.grip_closed_angle_max_rad
            < self.grip_release_angle_min_rad
            <= self.grip_wide_open_angle_rad
        ):
            raise ValueError("grip angles must satisfy closed < release <= wide-open")
        if self.grip_release_angle_delta_rad <= 0.0:
            raise ValueError("grip release angle delta must be positive")
        if self.grip_angle_motion_epsilon_rad <= 0.0:
            raise ValueError("grip motion epsilon must be positive")
        if self.grip_compliance_m_per_n < 0.0:
            raise ValueError("grip compliance cannot be negative")
        if not 0.0 < self.grip_relaxation <= 1.0:
            raise ValueError("grip relaxation must lie in (0, 1]")
        if self.grip_maximum_correction_m < 0.0:
            raise ValueError("grip correction cap cannot be negative")
        if self.grip_transfer_layers < 1:
            raise ValueError("grip transfer layers must be positive")
        if not 0.0 < self.grip_minimum_volume_ratio <= 1.0:
            raise ValueError("grip minimum J must lie in (0, 1]")
        if self.grip_support_radius_m < 0.0:
            raise ValueError("grip support radius cannot be negative")
        if self.grip_support_generations < 1:
            raise ValueError("grip support generations must be positive")
if (
    LOADED_EMBODIED_GAUSSIANS_SOURCE
    != EXPECTED_EMBODIED_GAUSSIANS_SOURCE
):
    raise RuntimeError(
        "Loaded embodied_gaussians from the wrong checkout: "
        f"{LOADED_EMBODIED_GAUSSIANS_SOURCE}; expected "
        f"{EXPECTED_EMBODIED_GAUSSIANS_SOURCE}"
    )


VISUAL_FORCE_MASK_DIR = (
    repo_root / "data/super/grasp5_native/visual_force_masks_v1"
)
RIGHT_VISUAL_FORCE_MASK_DIR = (
    repo_root / "data/super/grasp5_native/visual_force_masks_right_v1"
)
VISUAL_FORCE_INSTRUMENT_MASKS = (
    repo_root
    / "data/super/psm_visual_calibration/raw_paper_lnd_stereo_dense_contact_v4/"
    "surgicalsam2_multianchor_parts_dense_contact_v6/"
    "stereo_multianchor_part_masks.npz"
)


def stiffness_local_quality_mask(
    mapper: TetrahedralGaussianVisualResidualMapper,
    physical_prediction: torch.Tensor,
    corrected_positions: torch.Tensor,
    minimum_volume_ratio: float,
) -> torch.Tensor:
    """Return particles whose incident tets are valid before and after residual."""

    def volume_ratios(positions: torch.Tensor) -> torch.Tensor:
        points = positions[mapper.tet_indices]
        signed_six_volume = torch.linalg.det(
            torch.stack(
                (
                    points[:, 1] - points[:, 0],
                    points[:, 2] - points[:, 0],
                    points[:, 3] - points[:, 0],
                ),
                dim=-1,
            )
        )
        return signed_six_volume / (6.0 * mapper.rest_volumes)

    initial_ratio = volume_ratios(physical_prediction)
    corrected_ratio = volume_ratios(corrected_positions)
    valid_tets = (
        torch.isfinite(initial_ratio)
        & torch.isfinite(corrected_ratio)
        & (initial_ratio >= minimum_volume_ratio)
        & (corrected_ratio >= minimum_volume_ratio)
    )
    valid_particles = torch.ones_like(mapper.fixed_mask)
    invalid_particle_ids = mapper.tet_indices[~valid_tets].reshape(-1)
    if invalid_particle_ids.numel():
        valid_particles[invalid_particle_ids] = False
    valid_particles[mapper.fixed_mask] = False
    return valid_particles


def stiffness_visual_supervision_mask(
    mapper: TetrahedralGaussianVisualResidualMapper,
    visual_gradient_norm: torch.Tensor,
) -> torch.Tensor:
    """Select nodes with a non-negligible masked RGB data gradient."""
    gradient = visual_gradient_norm.detach().to(
        device=mapper.rest_positions.device, dtype=torch.float32
    )
    if gradient.shape != mapper.fixed_mask.shape:
        raise ValueError("Visual supervision gradient has the wrong shape")
    finite = torch.isfinite(gradient)
    dynamic = finite & ~mapper.fixed_mask
    maximum = gradient[dynamic].max() if bool(dynamic.any().item()) else None
    if maximum is None or float(maximum.item()) <= 0.0:
        return torch.zeros_like(mapper.fixed_mask)
    threshold = max(
        float(maximum.item()) * STIFFNESS_VISUAL_GRADIENT_RELATIVE_FLOOR,
        torch.finfo(gradient.dtype).tiny,
    )
    return dynamic & (gradient >= threshold)


def grip_control_exclusion_mask(
    mapper: TetrahedralGaussianVisualResidualMapper,
    environment: EmbodiedGaussiansEnvironment,
) -> torch.Tensor:
    """Mask direct/support controls without leaking known motion into learning."""
    excluded = torch.zeros_like(mapper.fixed_mask)
    expandable = torch.zeros_like(mapper.fixed_mask)
    sim_control = getattr(
        environment, "super_sim_grasp_control_mask", None
    )
    if sim_control is not None:
        sim_control = sim_control.to(device=excluded.device, dtype=torch.bool)
        if sim_control.shape != excluded.shape:
            raise ValueError("Dataset grasp control mask has the wrong shape")
        if bool(
            getattr(
                environment,
                "super_sim_grasp_control_mask_is_complete",
                False,
            )
        ):
            excluded |= sim_control
        else:
            expandable |= sim_control

    projector = environment.sim.triangle_skin_contact_projector
    if (
        projector is not None
        and bool(projector.persistent_grip_state.numpy()[0])
    ):
        grip_body = wp.to_torch(projector.persistent_grip_particle_body).to(
            device=excluded.device
        )
        expandable |= grip_body >= 0
    if bool(expandable.any().item()):
        excluded |= strict_one_tetrahedron_ring_mask(
            expandable, mapper.tet_indices
        )
    if not bool(excluded.any().item()):
        return excluded
    excluded[mapper.fixed_mask] = True
    return excluded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 SUPER grasp5 离线回放 demo。")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "SUPER 离线数据目录。目录内需要包含 robots.json、cameras.json、"
            "videos/stereo_left.mp4、videos/stereo_right.mp4 和对应 json。"
        ),
    )
    parser.add_argument("--fps", type=int, default=30, help="离线回放速度。默认按视频约 30 FPS 播放。")
    parser.add_argument(
        "--monitor-psm-base-q",
        action="store_true",
        help="打印 PSM 基座 body_q，诊断机械臂基座是否真的在物理仿真中漂移。",
    )
    parser.add_argument(
        "--monitor-tissue-q",
        action="store_true",
        help="打印 tissue 刚体 body_q，诊断组织是否被物理或视觉力拉走。",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=0.5,
        help="body_q 监控打印间隔，单位秒。默认 0.5 秒。",
    )
    parser.add_argument(
        "--visual-feedback-mode",
        choices=("trajectory", "residual", "force", "off"),
        default="residual",
        help=(
            "视觉反馈方式：trajectory=CoTracker二维运动+数据集深度的固定范围"
            "粒子位置/速度更新；residual=论文式RGB物理节点残差实时回写（默认）；"
            "force=旧 Gaussian 位移转粒子力；off=关闭视觉反馈。"
        ),
    )
    parser.add_argument(
        "--flow-depth-bindings",
        type=Path,
        default=None,
        help="trajectory 模式的固定轨迹到物理粒子范围绑定。",
    )
    parser.add_argument(
        "--flow-depth-observations",
        type=Path,
        default=None,
        help="trajectory 模式的因果 CoTracker+数据集深度三维观测。",
    )
    parser.add_argument("--flow-depth-position-gain", type=float, default=0.70)
    parser.add_argument("--flow-depth-velocity-gain", type=float, default=0.20)
    parser.add_argument(
        "--flow-depth-absolute-position-weight", type=float, default=0.85
    )
    parser.add_argument(
        "--flow-depth-solver-regularization", type=float, default=0.01
    )
    parser.add_argument("--flow-depth-solver-iterations", type=int, default=24)
    parser.add_argument("--flow-depth-robust-residual-mm", type=float, default=5.0)
    parser.add_argument(
        "--flow-depth-maximum-position-correction-mm", type=float, default=2.0
    )
    parser.add_argument(
        "--flow-depth-maximum-velocity-correction-m-s", type=float, default=0.06
    )
    parser.add_argument(
        "--flow-depth-initial-alignment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "用首帧RGB估计深度与重建可视表面之间的鲁棒刚体平移初始化组织；"
            "不读取GT深度/轨迹，也不做逐点非刚性形变。"
        ),
    )
    parser.add_argument(
        "--flow-depth-initial-alignment-maximum-mm",
        type=float,
        default=1.0,
        help="首帧RGB深度刚体平移上限，单位mm，默认1.0。",
    )
    parser.add_argument(
        "--visual-force-iterations",
        type=int,
        default=1,
        help=(
            "tissue visual forces 每次更新的图像优化迭代数，默认 1；"
            "PSM 始终不参与。"
        ),
    )
    parser.add_argument(
        "--visual-residual-iterations",
        type=int,
        default=8,
        help="每次实时视觉残差更新的 Adam 迭代数，默认 8。",
    )
    parser.add_argument(
        "--visual-residual-learning-rate-m",
        type=float,
        default=4.0e-5,
        help="实时视觉残差节点学习率，单位 m，默认 4.0e-5。",
    )
    parser.add_argument(
        "--visual-residual-maximum-mm",
        type=float,
        default=None,
        help=(
            "单帧视觉状态增量上限，单位 mm；仿真重建正式默认 0.50 mm。"
        ),
    )
    parser.add_argument(
        "--visual-residual-image-scale",
        type=float,
        default=None,
        help="视觉残差内部 RGB 分辨率比例；仿真重建默认 0.125。",
    )
    parser.add_argument(
        "--visual-force-update-interval",
        "--visual-feedback-update-interval",
        dest="visual_force_update_interval",
        type=int,
        default=3,
        help=(
            "每隔多少个物理步重新计算一次双目视觉力，默认 3；"
            "中间物理步复用最近一次力。"
        ),
    )
    parser.add_argument(
        "--online-stiffness-update",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "仿真重建中由 accepted RGB residual 的局部边应变因果更新 "
            "paper distance/shape，逐帧直接提交且不做分支选择；volume 刚度固定。"
        ),
    )
    parser.add_argument(
        "--sim-grasp-boundary-mode",
        choices=("red_marker_5", "known_grasp_region"),
        default="red_marker_5",
        help=(
            "仿真重建的已知夹持运动边界：red_marker_5=旧五点边界；"
            "known_grasp_region=数据生成器完整8x8 mm夹持核心。"
        ),
    )
    parser.add_argument(
        "--stiffness-log-learning-rate",
        type=float,
        default=0.18,
        help="在线刚度 log-space 学习率，默认使用强档 0.18。",
    )
    parser.add_argument(
        "--stiffness-update-mode",
        choices=(
            "heuristic",
            "particle_residual",
            "differentiable_low_dim",
            "differentiable_global",
            "differentiable_global_relative",
            "differentiable_global_mhe",
            "differentiable_local_relative",
            "differentiable_hierarchical_relative",
            "differentiable_particle_graph_lm",
        ),
        default="heuristic",
        help=(
            "刚度更新器：heuristic=原残差/边应变规则；"
            "particle_residual=逐粒子distance/shape刚度场，邻接图只平滑"
            "残差信号，固定/夹持/无效粒子不更新；"
            "differentiable_low_dim=3至5帧因果distance/volume/shape-PBD"
            "+低维Adam；differentiable_global=保留paper-PBD并仅估计"
            "全局distance刚度与阻尼（夹持耦合固定为1），提交前要求"
            "Warp有限差分梯度方向余弦不低于0.95，并由Warp梯度驱动Adam。"
            "differentiable_global_relative=在同一物理参数上改用Cauchy"
            "邻接轨迹相对形变和无遗忘累计Warp梯度；"
            "differentiable_global_mhe=保留20个已发生转移，使用实际Warp"
            "多重射击有限差分雅可比、可观测性门和LM，只更新全局distance"
            "与阻尼；"
            "differentiable_local_relative=固定全局均值和阻尼，只用12个"
            "平滑局部系数拟合H=1/3/5残差写入前轨迹，并由Warp有限差分"
            "梯度更新，不做分支选择；"
            "differentiable_hierarchical_relative=从统一初值联合更新一个"
            "全局distance均值和零均值平滑区域偏差，阻尼与夹持耦合固定。"
            "differentiable_particle_graph_lm=固定适中初值，每个非夹持"
            "粒子拥有零均值局部distance变量，并联合一个全局均值；使用"
            "观测度加权的图正则对角GN/LM更新和Warp方向有限差分检查，"
            "shape弱耦合、volume固定。"
        ),
    )
    parser.add_argument(
        "--stiffness-strain-signal-weight",
        type=float,
        default=0.80,
        help="旧启发式边应变信号权重；本次保守实验使用0.20。",
    )
    parser.add_argument(
        "--stiffness-autograd-unroll-steps",
        type=int,
        choices=(3, 4, 5),
        default=4,
        help="低维Adam材料梯度所用的已观测因果PBD帧数。",
    )
    parser.add_argument(
        "--stiffness-autograd-region-count",
        type=int,
        default=12,
        help="低维平滑log-distance刚度系数数量，默认12。",
    )
    parser.add_argument(
        "--stiffness-maximum-log-step",
        type=float,
        default=None,
        help=(
            "可选的在线刚度单帧 log 增量上限；仿真重建默认 0.02。"
        ),
    )
    parser.add_argument(
        "--stiffness-signal-ema-decay",
        type=float,
        default=None,
        help=(
            "可选的刚度信号历史 EMA 衰减；仿真重建默认 0.90。"
        ),
    )
    parser.add_argument(
        "--stiffness-spatial-smoothing-iterations",
        type=int,
        default=None,
        help="可选的局部刚度空间平滑轮数；仿真重建默认 3。",
    )
    parser.add_argument(
        "--stiffness-spatial-smoothing-blend",
        type=float,
        default=None,
        help="可选的每轮局部刚度空间平滑混合系数；仿真重建默认 0.35。",
    )
    parser.add_argument(
        "--initial-paper-distance-stiffness",
        type=float,
        default=None,
        help=(
            "可选的全局均匀 paper distance 初值；只改变初始假设，不载入"
            "局部材料真值。默认沿用场景值 0.31。"
        ),
    )
    parser.add_argument(
        "--initial-paper-shape-stiffness",
        type=float,
        default=None,
        help=(
            "可选的全局均匀 paper shape 初值；只改变初始假设，不载入"
            "局部材料真值。默认沿用场景值 0.0058。"
        ),
    )
    parser.add_argument(
        "--stiffness-evaluation-output",
        type=Path,
        default=None,
        help=(
            "可选评测输出目录。指定后，对每个已提交刚度运行 H=1/3/5/10 "
            "材料隔离与端到端开放环，并写 events.jsonl/summary.json；"
            "目录必须没有同名结果文件。默认关闭，正常 GUI 不承担多影子开销。"
        ),
    )
    parser.add_argument(
        "--stiffness-evaluation-horizons",
        type=str,
        default="1,3,5,10",
        help="开放环视频帧 horizon，逗号分隔，默认 1,3,5,10。",
    )
    parser.add_argument(
        "--evaluation-headless",
        action="store_true",
        help=(
            "不启动 GUI，按固定视频帧和每帧物理步数跑可重复评测；"
            "正式消融只需同时指定 --benchmark-output。"
        ),
    )
    parser.add_argument(
        "--evaluation-start-frame",
        type=int,
        default=350,
        help="Headless 评测起始视频帧，默认 350。",
    )
    parser.add_argument(
        "--evaluation-frame-count",
        type=int,
        default=0,
        help="Headless 评测帧数；0 表示从起始帧跑到末尾。",
    )
    parser.add_argument(
        "--evaluation-physics-steps-per-frame",
        type=int,
        default=3,
        help="Headless 评测每个视频帧固定执行的物理步数，默认 3。",
    )
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=None,
        help=(
            "可选的仿真消融产物目录；逐帧导出固定表面点的 3D/双目 2D "
            "预测、材料诊断和组织 Gaussian 渲染。仅在 --evaluation-headless 下使用。"
        ),
    )
    parser.add_argument(
        "--evaluation-label",
        default="unnamed",
        help="写入评估元数据的消融组名称。",
    )
    parser.add_argument(
        "--evaluation-render-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Headless 消融评估是否保存原生分辨率组织 Gaussian PNG，默认开启。",
    )
    parser.add_argument(
        "--evaluation-open-loop-start-frame",
        type=int,
        default=-1,
        help=(
            "从该帧起冻结视觉残差和在线刚度，直接进行开放环未来预测；"
            "-1 表示不切换，正式 80/20 协议使用 240。"
        ),
    )
    parser.add_argument(
        "--evaluation-holdout-stride",
        type=int,
        default=0,
        help=(
            "因果重建测试的周期留出间隔；0 表示关闭。EH-SurGS 7:1 "
            "协议使用 8，并在留出帧禁止视觉残差和刚度更新。"
        ),
    )
    parser.add_argument(
        "--evaluation-holdout-offset",
        type=int,
        default=7,
        help=(
            "周期留出帧偏移；正式因果 7:1 使用 7，即帧 "
            "7,15,23,... 为测试帧，前七帧为训练帧。"
        ),
    )
    parser.add_argument(
        "--evaluation-render-frame-mode",
        choices=("all", "holdout", "future"),
        default="all",
        help=(
            "保存组织渲染的帧集合：all 全部、holdout 仅 7:1 测试帧、"
            "future 仅开放环分界及之后。轨迹状态仍逐帧完整导出。"
        ),
    )
    parser.add_argument(
        "--psm-tissue-contact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "启用器械与组织接触（默认开启）；使用 "
            "--no-psm-tissue-contact 可关闭。"
        ),
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default="stereo_left,stereo_right",
        help="逗号分隔的离线相机名。默认同时启用左右目。",
    )
    parser.add_argument(
        "--camera-go-zoom",
        type=float,
        default=0.9,
        help=(
            "Go To Camera 视角缩放系数。1.0 表示保持纵横比并完整包含相机画面；"
            "默认 0.9，在完整画面外额外保留约 10%% 边距。"
        ),
    )
    parser.add_argument(
        "--psm-pose-driver",
        choices=tuple(PSM_POSE_DRIVER_PATHS),
        default="depth_then_visual",
        help=(
            "PSM逐帧位姿来源（默认 depth_then_visual）："
            "raw_kinematics=原始bag/q7/标定/hand-eye/LND纯机器人运动学；"
            "raw_p420006=相同原始运动学主干加P420006器械CAD；"
            "raw_p420006_stereo_visual=在raw_p420006上使用10对原始双目"
            "人工标注得到的受约束视觉矫正；"
            "raw_p420006_sam2_online=仅首对人工提示、完整1631对双目"
            "SurgicalSAM2在线视觉矫正；"
            "raw_paper_lnd_sam2_online=重新标注首对、采用论文原始LND "
            "CAD/FK并扩展到完整1631对双目的在线视觉矫正；"
            "raw_paper_lnd_first_stereo_static_q5=只在原始首对上按论文"
            "方法联合左右目优化固定SE(3)与q5零位，后续完全使用机器人学；"
            "raw_paper_lnd_first_stereo_se3_fixed_q5=只在原始首对上联合"
            "左右目优化固定XYZ与三维旋转（可修正倾转和自旋），q5及其余"
            "q7逐时刻严格保留原始值，后续完全使用机器人学；"
            "raw_paper_lnd_sam2_multianchor_closedjaw=左右目8组锚点、"
            "黑杆/银色末端分对象SurgicalSAM2，并严格保留原始q7夹爪"
            "闭合状态的论文LND双目矫正；"
            "raw_paper_lnd_sam2_dense_contact_closedjaw=在上一版基础上"
            "增加500、540、700、747、900、1100双目接触段锚点，"
            "并严格保留原始q7夹爪闭合状态；"
            "raw_paper_lnd_sam2_dense_contact_se3_only=沿用接触段加密"
            "人工双目锚点，对全部1631对仅优化时变XYZ与三维旋转，"
            "q1-q7逐时刻严格保留原始值；"
            "raw_paper_lnd_sam2_dense_contact_unbounded_xyz=使用22对双目"
            "锚点、增强人工尖端约束，对XYZ取消硬边界，仍严格保留原始"
            "q1-q7；"
            "strict=现有LND；"
            "registered_lnd=固定几何注册后的LND先验；paper=纯论文图像跟踪；"
            "paper_robust=此前的论文增强版；hybrid=LND先验加论文图像残差；"
            "corrected=深度优化前的完整三部件时序矫正结果；"
            "depth_then_visual=在 corrected 基础上完成多轮双目深度微调和视觉矫正。"
        ),
    )
    parser.add_argument(
        "--psm-visual-mode",
        choices=("tip", "full"),
        default="full",
        help=(
            "PSM可视范围：full=显示长杆、腕部和夹爪（默认）；"
            "tip=只显示腕部和夹爪。组织接触始终只使用两片夹爪。"
        ),
    )
    parser.add_argument(
        "--tissue-mode",
        choices=("paper_pbd", "paper_soft", "adaptive_soft", "rigid_v9"),
        default="paper_pbd",
        help=(
            "组织运行模式：paper_pbd=论文式 distance/volume/shape-matching "
            "XPBD 基线并支持 RGB residual/在线刚度（默认）；paper_soft=旧 "
            "Neo-Hookean XPBD 固定参数回退；两者均保留 Gaussian、视觉力和 "
            "triangle-skin 接触，paper_soft 不启用在线刚度；"
            "adaptive_soft=旧软体实验；rigid_v9=旧刚体组织回退。"
        ),
    )
    parser.add_argument(
        "--calibrated-profile",
        type=Path,
        default=None,
        help=(
            "可选的离线组织标定 profile。默认不加载；新组织先以冻结基础"
            "参数进入 GUI，刚度优化留到后续阶段。"
        ),
    )
    parser.add_argument(
        "--psm-roll-offset-deg",
        type=float,
        default=None,
        help=(
            "手动给 PSM roll 关节增加一个角度偏移，单位 degree。"
            "roll 是器械沿长杆轴线的自旋，用于临时对齐视频中的腕部/夹爪朝向；"
            "raw运动学/P420006系列默认0 deg，其他模式默认-27 deg；只在运行时"
            "生效，不修改输入数据。"
        ),
    )
    parser.add_argument(
        "--psm-camera-translation-mm",
        type=float,
        nargs=3,
        metavar=("RIGHT", "DOWN", "FAR"),
        default=(0.0, 0.0, 0.0),
        help="人工相机坐标平移修正，单位 mm，顺序为右、下、远。",
    )
    parser.add_argument(
        "--psm-world-translation-mm",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help=(
            "固定的 PSM 世界坐标平移修正，单位 mm，顺序为 X、Y、Z；"
            "只从命令行载入，不在 GUI 中实时修改。"
        ),
    )
    return parser.parse_args()


def load_calibrated_tissue_profile(
    environment, profile_path: Path | None
) -> bool:
    """Apply frozen scalar and smooth regional material settings to the GUI env."""
    if profile_path is None:
        return False
    if not profile_path.is_file() or getattr(environment, "super_tissue_mode", "") != "paper_soft":
        return False
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    material = profile["material"]
    E = float(material["young_modulus_pa"]); nu = float(material["poisson_ratio"])
    region_path = profile_path.parent.parent / "stage_f_local_stiffness" / "region_profile.npz"
    region = np.load(region_path)
    weights = np.asarray(region["tet_region_weights"], dtype=np.float32)
    multipliers = profile["regional_stiffness"]
    E_tet = E * (weights @ np.asarray([multipliers["primary_multiplier"], multipliers["transition_multiplier"], multipliers["far_multiplier"]], dtype=np.float32))
    handle = environment.super_tissue_soft_handle
    expected_tets = handle.tet_end - handle.tet_start
    if len(weights) != expected_tets:
        raise ValueError(
            "Calibrated stiffness profile does not match the selected tissue "
            f"asset: profile has {len(weights)} tetrahedra, asset has "
            f"{expected_tets}. Re-optimize it for the new asset first."
        )
    materials = wp.to_torch(environment.sim.model.tet_materials)
    mu = torch.as_tensor(E_tet / (2.0 * (1.0 + nu)), device=materials.device, dtype=materials.dtype)
    lam = torch.as_tensor(E_tet * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), device=materials.device, dtype=materials.dtype)
    materials[handle.tet_start:handle.tet_end, 0] = mu
    materials[handle.tet_start:handle.tet_end, 1] = lam
    materials[handle.tet_start:handle.tet_end, 2] = 0.0
    environment.physics_settings.particle_velocity_damping_per_second = float(material["velocity_damping_per_second"])
    environment.super_tissue_young_modulus_pa = float(E)
    environment.super_tissue_calibrated_profile_path = str(profile_path)
    environment.super_tissue_calibrated_profile = profile
    if hasattr(environment.sim, "_physics_step_cache"):
        delattr(environment.sim, "_physics_step_cache")
    print(f"[example_embodied_super_offline] loaded calibrated tissue profile: {profile_path}; E={E:g} Pa; regional stiffness=ON; visual residual=OFF")
    return True


def resolve_dataset_path(dataset_arg: Path | None) -> Path:
    default_dataset = repo_root / "data/super/grasp5_offline_demo"
    dataset_from_env = os.environ.get("EMBODIED_GAUSSIANS_SUPER_DATASET")
    dataset_path = dataset_arg
    if dataset_path is None and dataset_from_env:
        dataset_path = Path(dataset_from_env).expanduser()
    if dataset_path is None:
        dataset_path = default_dataset
    if not dataset_path.is_absolute():
        dataset_path = (Path.cwd() / dataset_path).resolve()
    return dataset_path


def build_visual_tissue_residual_mapper(
    environment: EmbodiedGaussiansEnvironment,
    *,
    iterations: int,
    learning_rate_m: float,
) -> TetrahedralGaussianVisualResidualMapper:
    """Construct the online particle residual map from runtime mode-2 bindings."""
    simulator = environment.sim
    model = simulator.gaussian_model
    if model.num_soft_gaussians == 0:
        raise ValueError("Visual residual mode requires soft tissue Gaussians")
    binding_modes = model.soft_gaussian_binding_modes.long()
    if not bool(torch.all(binding_modes == 2).item()):
        modes = torch.unique(binding_modes).detach().cpu().tolist()
        raise ValueError(
            "Visual residual mode currently requires triangular-face-centroid "
            f"Gaussian bindings (mode 2); found modes {modes}"
        )
    settings = VisualTissueResidualMappingSettings(
        iterations=iterations,
        learning_rate_m=learning_rate_m,
        image_scale=0.25,
        maximum_residual_m=0.00060,
        minimum_volume_ratio=0.30,
    )
    mapper = (
        TetrahedralGaussianVisualResidualMapper.from_visual_face_centroid_bindings(
            rest_positions=wp.to_torch(simulator.model.particle_q).detach().clone(),
            tet_indices=wp.to_torch(simulator.model.tet_indices)
            .long()
            .reshape(-1, 4),
            fixed_mask=(
                wp.to_torch(simulator.model.particle_inv_mass).detach() == 0.0
            ),
            soft_gaussian_ids=model.soft_gaussian_ids,
            visual_vertex_particle_indices=(
                model.soft_gaussian_visual_vertex_particle_indices
            ),
            visual_vertex_weights=model.soft_gaussian_visual_vertex_weights,
            visual_vertex_rest_offsets=(
                model.soft_gaussian_visual_vertex_rest_offsets
            ),
            visual_vertex_rest_physical_frames=(
                model.soft_gaussian_visual_vertex_rest_physical_frames
            ),
            rest_visual_face_poses=(
                model.soft_gaussian_rest_visual_face_poses
            ),
            rest_gaussian_quats=model.quats[model.soft_gaussian_ids.long()],
            rest_gaussian_scales=model.scales[model.soft_gaussian_ids.long()],
            settings=settings,
        )
    )
    print(
        "[example_embodied_super_offline] realtime visual residual mapper: "
        f"particles={len(mapper.rest_positions)}, tets={len(mapper.tet_indices)}, "
        f"soft_gaussians={len(mapper.soft_gaussian_ids)}, "
        f"iterations={settings.iterations}, lr={settings.learning_rate_m:g}m, "
        f"image_scale={settings.image_scale:g}, "
        f"max_step={settings.maximum_residual_m * 1e3:.3f}mm"
    )
    return mapper


def align_sim_tissue_to_first_flow_depth(
    environment: EmbodiedGaussiansEnvironment,
    *,
    tissue_asset: Path,
    bindings: FlowDepthParticleRangeBindings,
    depth_source: str,
    maximum_translation_m: float,
) -> dict:
    """Rigidly align the reconstructed visual surface to first-frame RGB depth.

    Only one robust translation is applied to every physical particle.  This
    deliberately preserves every edge length, tetrahedral volume, local
    stiffness value, and relative grasp geometry; a noisy per-track warp would
    manufacture strain before material identification has even started.
    """

    if maximum_translation_m <= 0.0:
        raise ValueError("First-depth alignment cap must be positive")
    with np.load(tissue_asset, allow_pickle=False) as loaded:
        visual_vertices = np.asarray(
            loaded["visual_surface_rest_vertices_table"], dtype=np.float64
        )
        visual_faces = np.asarray(loaded["visual_surface_faces"], dtype=np.int32)
    observed = np.asarray(bindings.initial_points_table, dtype=np.float64)
    valid = np.asarray(bindings.track_valid, dtype=bool)
    valid &= np.isfinite(observed).all(axis=1)
    if int(valid.sum()) < 3:
        raise ValueError("First-depth alignment needs at least three valid tracks")

    # Open3D returns exact closest points on triangle interiors/edges rather
    # than nearest vertices.  It is imported only for this opt-in initializer.
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(visual_vertices),
        o3d.utility.Vector3iVector(visual_faces),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    closest = scene.compute_closest_points(
        o3d.core.Tensor(observed[valid].astype(np.float32))
    )["points"].numpy().astype(np.float64)
    residuals = observed[valid] - closest

    component_median = np.median(residuals, axis=0)
    centered_norm = np.linalg.norm(residuals - component_median[None, :], axis=1)
    centered_median = float(np.median(centered_norm))
    centered_mad = float(
        np.median(np.abs(centered_norm - centered_median))
    )
    inlier_limit_m = max(
        0.00025,
        centered_median + 3.0 * 1.4826 * centered_mad,
    )
    inliers = centered_norm <= inlier_limit_m
    if int(inliers.sum()) < 3:
        raise ValueError("Robust first-depth alignment rejected too many tracks")
    translation = np.median(residuals[inliers], axis=0)
    raw_translation_norm = float(np.linalg.norm(translation))
    if raw_translation_norm > maximum_translation_m:
        translation *= maximum_translation_m / raw_translation_norm

    translation_tensor = torch.as_tensor(
        translation,
        device=wp.to_torch(environment.sim.state_0.particle_q).device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        for state in (environment.sim.state_0, environment.sim.state_1):
            wp.to_torch(state.particle_q).add_(translation_tensor)
    environment.sim.update_gaussian_transforms()

    before_norm = np.linalg.norm(residuals[inliers], axis=1)
    after_norm = np.linalg.norm(
        residuals[inliers] - translation[None, :], axis=1
    )
    diagnostics = {
        "enabled": True,
        "method": "robust_rigid_translation_to_reconstructed_visual_surface",
        "depth_source": str(depth_source),
        "uses_ground_truth_depth": False,
        "uses_ground_truth_trajectory": False,
        "valid_track_count": int(valid.sum()),
        "inlier_track_count": int(inliers.sum()),
        "outlier_track_count": int(valid.sum() - inliers.sum()),
        "inlier_limit_mm": float(inlier_limit_m * 1.0e3),
        "translation_mm": (translation * 1.0e3).tolist(),
        "translation_norm_mm": float(np.linalg.norm(translation) * 1.0e3),
        "uncapped_translation_norm_mm": float(raw_translation_norm * 1.0e3),
        "maximum_translation_mm": float(maximum_translation_m * 1.0e3),
        "surface_residual_before_mm": {
            "mean": float(before_norm.mean() * 1.0e3),
            "median": float(np.median(before_norm) * 1.0e3),
            "p95": float(np.quantile(before_norm, 0.95) * 1.0e3),
        },
        "surface_residual_after_mm": {
            "mean": float(after_norm.mean() * 1.0e3),
            "median": float(np.median(after_norm) * 1.0e3),
            "p95": float(np.quantile(after_norm, 0.95) * 1.0e3),
        },
        "physical_invariants": (
            "one translation for all particles; edge lengths, tetrahedral "
            "volumes, local stiffness, and zero initial velocity preserved"
        ),
    }
    environment.super_flow_depth_initial_alignment = diagnostics
    print(
        "[example_embodied_super_offline] first RGB-depth alignment: ON; "
        f"source={depth_source}; tracks={int(inliers.sum())}/{int(valid.sum())}; "
        f"translation_mm={diagnostics['translation_mm']}; "
        f"surface_mean_mm={before_norm.mean() * 1e3:.4f}->"
        f"{after_norm.mean() * 1e3:.4f}; GT depth/trajectory=UNUSED"
    )
    return diagnostics


class SuperPlaybackControls:
    """SUPER 离线回放控制器。

    它做三件事：
    1. 按当前时间戳从 DatasetManager 取视频帧；
    2. 按同一个时间戳从当前 pose source 取 q7，展开成完整 PSM q_full。
    3. 在所选 pose driver 上叠加 GUI 的自旋、夹爪和整体平移增量。

    原 PushT demo 可以直接把 q 塞给 Panda。SUPER 不行，因为 PSM 的完整
    URDF 有 mimic 关节，所以这里多了一步 q7 -> q_full。
    """

    def __init__(
        self,
        environment: EmbodiedGaussiansEnvironment,
        dataset_manager: DatasetManager,
        fps: int,
        monitor_psm_base_q: bool = False,
        monitor_tissue_q: bool = False,
        monitor_interval: float = 0.5,
        psm_roll_offset_deg: float = 0.0,
        psm_camera_translation_mm=(0.0, 0.0, 0.0),
        psm_world_translation_mm=(0.0, 0.0, 0.0),
        visual_force_update_interval: int = 4,
        visual_feedback_mode: str = "residual",
        visual_residual_mapper: TetrahedralGaussianVisualResidualMapper
        | None = None,
        flow_depth_bindings: FlowDepthParticleRangeBindings | None = None,
        flow_depth_observations: FlowDepthObservationSequence | None = None,
        flow_depth_settings: FlowDepthStateUpdateSettings | None = None,
        flow_depth_source: str = "unspecified_precomputed_depth",
        flow_depth_initial_alignment: dict | None = None,
        stiffness_updater: ResidualDrivenPaperStiffnessUpdater | None = None,
        enable_psm_tissue_contact: bool = True,
        stiffness_evaluation_output: Path | None = None,
        stiffness_evaluation_horizons: tuple[int, ...] = (1, 3, 5, 10),
    ):
        self.playing = False
        self.environment = environment
        self.dataset_manager = dataset_manager
        self.fps = fps
        offline_cameras = self.dataset_manager.offline_cameras
        if offline_cameras is None or len(offline_cameras) == 0:
            raise RuntimeError("SUPER exact-timestamp playback requires a camera")
        self.playback_camera_name, playback_camera = next(
            iter(offline_cameras.items())
        )
        self.playback_timestamps = np.asarray(
            playback_camera.timestamps, dtype=np.float64
        )
        if len(self.playback_timestamps) == 0:
            raise RuntimeError("SUPER playback camera has no timestamps")
        if np.any(np.diff(self.playback_timestamps) <= 0.0):
            raise ValueError("SUPER playback camera timestamps must increase")
        self.current_frame_index = 0
        self.current_timestep = float(self.playback_timestamps[0])
        self.first_state = (
            environment.sim.clone_embodied_gaussian_rollout_state()
        )
        self._tissue_rest_positions = wp.to_torch(
            self.first_state.embodied_state.physics_state.particle_q
        ).clone()
        self._tissue_max_displacement_m = 0.0
        self.monitor_psm_base_q = monitor_psm_base_q
        self.monitor_tissue_q = monitor_tissue_q
        self.monitor_interval = monitor_interval
        self.default_psm_roll_offset_deg = float(psm_roll_offset_deg)
        self.psm_roll_offset_deg = self.default_psm_roll_offset_deg
        self.psm_wrist_rod_offset_deg = 0.0
        self.psm_wrist_rod_offset_enabled = bool(
            environment.super_psm_lnd_kinematics.manual_offset_conventions.get(
                "paper_wrist_pitch"
            )
        )
        self.psm_jaw_offset_deg = 0.0
        self._last_base_monitor_time = -float("inf")
        self._last_tissue_monitor_time = -float("inf")
        self._initial_base_q: np.ndarray | None = None
        self._initial_tissue_q: np.ndarray | None = None
        self._warned_missing_base_id = False
        self._warned_missing_tissue_id = False
        self.default_psm_camera_translation_mm = np.asarray(
            psm_camera_translation_mm, dtype=np.float64
        ).copy()
        if self.default_psm_camera_translation_mm.shape != (3,):
            raise ValueError("PSM camera translation must contain three values")
        self.psm_manual_camera_translation_mm = (
            self.default_psm_camera_translation_mm.copy()
        )
        self.psm_world_translation_mm = np.asarray(
            psm_world_translation_mm, dtype=np.float64
        ).copy()
        if self.psm_world_translation_mm.shape != (3,):
            raise ValueError("PSM world translation must contain three values")
        self.default_psm_tissue_contact_enabled = bool(
            enable_psm_tissue_contact
        )
        self.psm_tissue_collisions_enabled = False
        self.visual_force_update_interval = int(visual_force_update_interval)
        if self.visual_force_update_interval < 1:
            raise ValueError("Visual-force update interval must be at least one")
        self._visual_force_step = self.visual_force_update_interval - 1
        self.visual_feedback_mode = str(visual_feedback_mode)
        if self.visual_feedback_mode not in {
            "trajectory",
            "residual",
            "force",
            "off",
        }:
            raise ValueError("Unknown visual feedback mode")
        if (
            self.visual_feedback_mode == "residual"
            and visual_residual_mapper is None
        ):
            raise ValueError("Residual feedback mode requires a residual mapper")
        self.visual_residual_mapper = visual_residual_mapper
        if (flow_depth_bindings is None) != (flow_depth_observations is None):
            raise ValueError(
                "Trajectory mode requires both flow-depth bindings and observations"
            )
        if self.visual_feedback_mode == "trajectory":
            if flow_depth_bindings is None or flow_depth_observations is None:
                raise ValueError("Trajectory mode requires flow-depth assets")
            if visual_residual_mapper is None:
                raise ValueError("Trajectory mode requires the physical safety mapper")
            if len(flow_depth_bindings.track_valid) != (
                flow_depth_observations.track_valid.shape[1]
            ):
                raise ValueError("Flow-depth bindings and observations disagree")
        self.flow_depth_bindings = flow_depth_bindings
        self.flow_depth_observations = flow_depth_observations
        self.flow_depth_settings = flow_depth_settings or FlowDepthStateUpdateSettings()
        self.flow_depth_settings.validate()
        self.flow_depth_source = str(flow_depth_source)
        self.flow_depth_initial_alignment = flow_depth_initial_alignment
        self._flow_depth_source_states: dict[int, np.ndarray] = {}
        self._flow_depth_source_rollouts: dict[
            int, FlowDepthSourceSnapshot
        ] = {}
        if stiffness_updater is None:
            warp_history_length = 3
        elif (
            stiffness_updater.settings.update_mode
            == "differentiable_global_mhe"
        ):
            warp_history_length = int(
                stiffness_updater.settings.observable_window_size
            )
        elif stiffness_updater.settings.update_mode in {
            "differentiable_local_relative",
            "differentiable_hierarchical_relative",
            "differentiable_particle_graph_lm",
        }:
            # Local objectives are evaluated at H=1/3/5.
            warp_history_length = max(
                5, int(stiffness_updater.settings.autograd_unroll_steps)
            )
        else:
            # The global Adam objective is defined strictly on H=1/2/3.
            # ``autograd_unroll_steps=4`` controls the differentiable PBD
            # configuration for backward compatibility; it must not retain a
            # fourth Warp transition and index past the three horizon weights.
            warp_history_length = 3
        self._warp_gradient_transitions: deque[WarpGradientTransition] = deque(
            maxlen=warp_history_length
        )
        self._flow_depth_reference_range_centers: np.ndarray | None = None
        self._last_flow_depth_runtime_frame_index = -1
        self._flow_depth_update_count = 0
        self._defer_flow_depth_update_until_frame_end = False
        self.stiffness_updater = stiffness_updater
        self.stiffness_evaluation_horizons = tuple(
            sorted(set(int(value) for value in stiffness_evaluation_horizons))
        )
        if not self.stiffness_evaluation_horizons or any(
            value <= 0 for value in self.stiffness_evaluation_horizons
        ):
            raise ValueError("Stiffness evaluation horizons must be positive")
        if (
            stiffness_evaluation_output is not None
            and visual_residual_mapper is None
        ):
            raise ValueError(
                "Trajectory evaluation requires a visual residual mapper"
            )
        self._action_phase_classifier = StiffnessActionPhaseClassifier()
        self._current_action_phase = "idle"
        self._active_stiffness_evaluations: list[
            CommittedStiffnessEvaluation
        ] = []
        self._next_stiffness_evaluation_id = 1
        self._stiffness_evaluation_epoch = 0
        evaluation_projector = (
            self.environment.sim.triangle_skin_contact_projector
        )
        self.stiffness_metrics_recorder = (
            StiffnessMetricsRecorder(
                stiffness_evaluation_output,
                metadata={
                    "stiffness_admission_profile": "formal",
                    "dataset": str(self.dataset_manager.path),
                    "horizons": self.stiffness_evaluation_horizons,
                    "protocols": ("material_isolation", "end_to_end"),
                    "visual_feedback_mode": self.visual_feedback_mode,
                    "online_stiffness_update": stiffness_updater is not None,
                    "stiffness_maximum_penetration_m": (
                        STIFFNESS_MAXIMUM_PENETRATION_M
                    ),
                    "stiffness_maximum_jaw_speed_rad_s": (
                        STIFFNESS_MAXIMUM_JAW_SPEED_RAD_S
                    ),
                    "stiffness_transition_cooldown_updates": (
                        STIFFNESS_TRANSITION_COOLDOWN_UPDATES
                    ),
                    "stiffness_prediction_absolute_margin": (
                        STIFFNESS_PREDICTION_ABSOLUTE_MARGIN
                    ),
                    "stiffness_prediction_relative_margin": (
                        STIFFNESS_PREDICTION_RELATIVE_MARGIN
                    ),
                    "stiffness_camera_absolute_regression": (
                        STIFFNESS_CAMERA_ABSOLUTE_REGRESSION
                    ),
                    "stiffness_camera_relative_regression": (
                        STIFFNESS_CAMERA_RELATIVE_REGRESSION
                    ),
                    "stiffness_minimum_volume_absolute_drop": (
                        STIFFNESS_MINIMUM_VOLUME_ABSOLUTE_DROP
                    ),
                    "stiffness_minimum_volume_relative_drop": (
                        STIFFNESS_MINIMUM_VOLUME_RELATIVE_DROP
                    ),
                    "stiffness_penetration_tolerance_m": (
                        STIFFNESS_PENETRATION_TOLERANCE_M
                    ),
                    "stiffness_anchor_error_tolerance_m": (
                        STIFFNESS_ANCHOR_ERROR_TOLERANCE_M
                    ),
                    "stiffness_history_relative_tolerance": (
                        STIFFNESS_HISTORY_RELATIVE_TOLERANCE
                    ),
                    "stiffness_history_absolute_tolerance_m": (
                        STIFFNESS_HISTORY_ABSOLUTE_TOLERANCE_M
                    ),
                    "stiffness_local_minimum_volume_ratio": (
                        STIFFNESS_UPDATE_LOCAL_MINIMUM_VOLUME_RATIO
                    ),
                    "stiffness_commit_validation_horizon_frames": (
                        STIFFNESS_COMMIT_VALIDATION_HORIZON_FRAMES
                    ),
                    "stiffness_validation_visual_objective": (
                        "mean_prediction_loss_frames_1_to_h"
                    ),
                    "stiffness_candidate_log_step_scales": (
                        STIFFNESS_CANDIDATE_LOG_STEP_SCALES
                    ),
                    "stiffness_adopt_validated_rollout": (
                        STIFFNESS_ADOPT_VALIDATED_ROLLOUT
                    ),
                    "stiffness_maximum_adopted_state_rms_m": (
                        STIFFNESS_MAXIMUM_ADOPTED_STATE_RMS_M
                    ),
                    "stiffness_maximum_adopted_state_maximum_m": (
                        STIFFNESS_MAXIMUM_ADOPTED_STATE_MAXIMUM_M
                    ),
                    "stiffness_log_learning_rate": (
                        None
                        if stiffness_updater is None
                        else stiffness_updater.settings.log_learning_rate
                    ),
                    "tip_entry_allowance_m": (
                        None
                        if evaluation_projector is None
                        else evaluation_projector.top_barrier_tip_allowance_m
                    ),
                    "grip_maximum_capture_penetration_m": (
                        None
                        if evaluation_projector is None
                        else (
                            evaluation_projector
                            .persistent_grip_maximum_capture_penetration_m
                        )
                    ),
                    "experiment_mode": (
                        "fixed_pbd"
                        if self.visual_feedback_mode == "off"
                        else (
                            f"{self.visual_feedback_mode}_online_stiffness"
                            if stiffness_updater is not None
                            else f"{self.visual_feedback_mode}_only"
                        )
                    ),
                },
            )
            if stiffness_evaluation_output is not None
            else None
        )
        if self.stiffness_metrics_recorder is not None:
            print(
                "[stiffness evaluation] ON; output="
                f"{self.stiffness_metrics_recorder.output_directory}; "
                f"horizons={self.stiffness_evaluation_horizons}; "
                "protocols=material_isolation,end_to_end"
            )
        self._startup_stiffness_settings = (
            stiffness_updater.settings if stiffness_updater is not None else None
        )
        self._stiffness_gui_draft = self._startup_stiffness_settings
        self._stiffness_gui_message = "Pause playback before applying changes."
        self._pending_stiffness_validation: (
            PendingStiffnessValidation | None
        ) = None
        self._stiffness_history: deque[StiffnessHistorySnapshot] = deque(
            maxlen=STIFFNESS_HISTORY_MAXIMUM_SNAPSHOTS
        )
        self._last_stiffness_gate_timestep: float | None = None
        self._last_stiffness_gate_jaw_angle: float | None = None
        self._stiffness_transition_cooldown = 0
        self._last_stiffness_grip_active = False
        self._last_stiffness_contact_count: int | None = None
        self._last_stiffness_validation_metrics: dict | None = None
        self._previous_visual_residual: torch.Tensor | None = None
        self._visual_residual_solve_count = 0
        self._visual_residual_busy = False
        self._single_visual_residual_requested = False
        self._last_visual_residual_frame_index = -1
        self._last_visual_residual_accepted: bool | None = None
        self._last_trajectory_observation_frame_index = -1
        self._physics_iteration_count = 0
        self._evaluation_open_loop_start_frame: int | None = None
        self._evaluation_open_loop_entered = False
        self._evaluation_holdout_stride: int | None = None
        self._evaluation_holdout_offset = 0
        self._evaluation_holdout_last_cancelled_frame = -1
        # Headless benchmark owns an exact step budget.  Keeping the worker
        # alive is useful because it also performs residual/stiffness updates,
        # but it must not continue integrating while native-resolution PNGs
        # are being written.
        self._evaluation_max_physics_iterations: int | None = None
        self._contact_ui_metrics: dict | None = None
        self._last_contact_ui_refresh_time = -float("inf")
        contact_projector = (
            self.environment.sim.triangle_skin_contact_projector
        )
        self._startup_contact_grip_settings = (
            ContactGripGuiSettings.from_runtime(
                self.environment.physics_settings,
                contact_projector,
            )
            if contact_projector is not None
            else None
        )
        self._contact_grip_gui_draft = (
            self._startup_contact_grip_settings
        )
        self._contact_grip_gui_message = (
            "Pause playback before applying contact/grip changes."
        )

        # mimic_cfg records how 7 active joints derive mimic joints.
        # joint_order records the joint order expected by the simulator.
        self.mimic_cfg = load_mimic_config(
            environment.super_psm_mimic_map_path
        )
        self.joint_order = urdf_actuated_joint_order(
            environment.super_psm_urdf_path
        )
        # PSM is fully driven by offline q, needs re-anchoring after each physics step.
        # _last_q_full stores the most recent q from go_to_timestep,
        # so run_physics can pull PSM back to the correct pose after each step.
        self._last_q_full: torch.Tensor = self.q_full_at(self.current_timestep)
        self._last_state_index = self.state_index_at(self.current_timestep)
        self._sim_reconstruction_mode = bool(
            getattr(environment, "super_sim_reconstruction_mode", False)
        )
        self._sim_grasp_boundary_mode = str(
            getattr(
                environment,
                "super_sim_grasp_boundary_mode",
                "red_marker_5",
            )
        )
        self._sim_grasp_boundary_schema = ""
        self._sim_grasp_boundary_path = ""
        self._sim_grasp_active = False
        self._sim_grasp_particle_ids: torch.Tensor | None = None
        self._sim_grasp_original_inverse_mass: torch.Tensor | None = None
        self._sim_grasp_support_particle_ids: torch.Tensor | None = None
        self._sim_grasp_support_capture_positions: torch.Tensor | None = None
        self._sim_grasp_trajectory_alignment: torch.Tensor | None = None
        self._sim_grasp_capture_positions: torch.Tensor | None = None
        self._sim_grasp_trajectory_origin: torch.Tensor | None = None
        self._sim_grasp_coupling_gain = 1.0
        self._sim_grasp_trajectory_positions: torch.Tensor | None = None
        self._sim_grasp_trajectory_velocities: torch.Tensor | None = None
        self._sim_grasp_target: torch.Tensor | None = None
        self._sim_grasp_target_velocity: torch.Tensor | None = None
        self._sim_grasp_tracking_error_m = 0.0
        self._sim_grasp_projection_correction_m = 0.0
        self._sim_grasp_natural_neighbor_displacement_m = 0.0
        self._sim_psm_q7: np.ndarray | None = None
        self._sim_grasp_schedule: np.ndarray | None = None
        if self._sim_reconstruction_mode:
            self._configure_sim_grasp_boundary()

    def _configure_sim_grasp_boundary(self) -> None:
        """Load the selected prescribed grasp trajectory boundary."""
        dataset_path = Path(self.dataset_manager.path)
        pose_path = dataset_path / "task_inputs" / "psm_link_poses.npz"
        phase_path = dataset_path / "task_inputs" / "phases.json"
        boundary_specs = {
            "red_marker_5": (
                dataset_path / "task_inputs" / "red_marker_boundary.npz",
                "fixedsuperbest.red_marker_boundary.v2",
                "red_marker_center_world",
            ),
            "known_grasp_region": (
                dataset_path
                / "task_inputs"
                / "known_grasp_region_boundary.npz",
                "fixedsuperbest.known_grasp_region_boundary.v1",
                "grasp_region_center_world",
            ),
        }
        if self._sim_grasp_boundary_mode not in boundary_specs:
            raise ValueError(
                "未知仿真夹持边界模式："
                f"{self._sim_grasp_boundary_mode}"
            )
        boundary_path, expected_schema, center_key = boundary_specs[
            self._sim_grasp_boundary_mode
        ]
        tissue_path = Path(self.environment.super_tissue_asset_path)
        for path in (pose_path, phase_path, boundary_path, tissue_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        with np.load(pose_path, allow_pickle=False) as poses:
            pose_timestamps = np.asarray(poses["timestamps"], dtype=np.float64)
            q7 = np.asarray(poses["q7"], dtype=np.float64)
        with np.load(boundary_path, allow_pickle=False) as boundary:
            schema = str(np.asarray(boundary["schema"]).item())
            if schema != expected_schema:
                raise ValueError(
                    f"夹持边界格式错误：期望{expected_schema}，实际{schema}"
                )
            boundary_timestamps = np.asarray(
                boundary["timestamps"], dtype=np.float64
            )
            boundary_grasped = np.asarray(boundary["grasped"], dtype=bool)
            particle_local_ids = np.asarray(
                boundary["reconstruction_particle_ids"], dtype=np.int64
            )
            trajectory_positions = np.asarray(
                boundary["trajectory_positions_world"], dtype=np.float32
            )
            trajectory_velocities = np.asarray(
                boundary["trajectory_velocities_world"], dtype=np.float32
            )
            marker_center = np.asarray(
                boundary[center_key], dtype=np.float32
            )
        phases = json.loads(phase_path.read_text(encoding="utf-8"))
        grasp_schedule = np.asarray(
            [bool(record["grasped"]) for record in phases], dtype=bool
        )
        frame_count = len(self.playback_timestamps)
        if (
            len(pose_timestamps) != frame_count
            or len(grasp_schedule) != frame_count
            or len(boundary_timestamps) != frame_count
            or len(boundary_grasped) != frame_count
            or len(trajectory_positions) != frame_count
            or len(trajectory_velocities) != frame_count
        ):
            raise ValueError("PSM、红色标记边界、grasp phase 与视频帧数不一致")
        if not np.allclose(
            pose_timestamps, self.playback_timestamps, atol=1.0e-6, rtol=0.0
        ) or not np.allclose(
            boundary_timestamps,
            self.playback_timestamps,
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise ValueError("PSM/红色标记边界时间戳与视频时间戳不一致")
        if not np.array_equal(boundary_grasped, grasp_schedule):
            raise ValueError("红色标记边界与 PSM 闭合时序不一致")
        if trajectory_positions.shape != (frame_count, len(particle_local_ids), 3):
            raise ValueError("红色标记位置轨迹形状错误")
        if trajectory_velocities.shape != trajectory_positions.shape:
            raise ValueError("红色标记速度轨迹形状错误")
        grasp_frames = np.flatnonzero(grasp_schedule)
        if len(grasp_frames) == 0:
            raise ValueError("数据集没有有效 grasp 区间")

        with np.load(tissue_path, allow_pickle=False) as tissue:
            top_mask = np.asarray(tissue["top_node_mask"], dtype=bool)
            fixed_mask = np.asarray(tissue["fixed_mask"], dtype=bool)
            local_tets = np.asarray(tissue["tet_indices"], dtype=np.int64)
        handle = self.environment.super_tissue_soft_handle
        if handle is None:
            raise RuntimeError("数据集 grasp 边界需要可变形组织")
        local_count = int(handle.particle_end - handle.particle_start)
        if len(top_mask) != local_count or len(fixed_mask) != local_count:
            raise ValueError("组织 top/fixed mask 与运行时粒子数量不一致")
        if (
            len(particle_local_ids) == 0
            or particle_local_ids.min() < 0
            or particle_local_ids.max() >= local_count
        ):
            raise ValueError("红色标记边界包含越界的重建粒子 ID")
        controlled_top = top_mask[particle_local_ids]
        if (
            self._sim_grasp_boundary_mode == "red_marker_5"
            and not bool(np.all(controlled_top))
        ):
            raise ValueError("红色标记五点边界必须全部位于重建顶面")
        if (
            self._sim_grasp_boundary_mode == "known_grasp_region"
            and (not bool(np.any(controlled_top)) or bool(np.all(controlled_top)))
        ):
            raise ValueError("完整夹持核心必须同时覆盖重建上下表面")
        if bool(np.any(fixed_mask[particle_local_ids])):
            raise ValueError("红色标记边界不能覆盖固定支撑节点")
        rest = self._tissue_rest_positions[
            handle.particle_start : handle.particle_end
        ]
        particle_ids = torch.as_tensor(
            particle_local_ids + int(handle.particle_start),
            device=rest.device,
            dtype=torch.long,
        )

        # Track two complete tetrahedral rings for diagnostics only.  These
        # nodes receive no target position, velocity, force, or GT trajectory;
        # all motion in them must be transmitted naturally by PBD constraints.
        direct_local_mask = np.zeros(local_count, dtype=bool)
        direct_local_mask[particle_local_ids] = True
        two_ring_mask = direct_local_mask.copy()
        for _ in range(2):
            incident_tets = np.any(two_ring_mask[local_tets], axis=1)
            two_ring_mask[np.unique(local_tets[incident_tets])] = True
        support_local_mask = two_ring_mask & ~direct_local_mask & ~fixed_mask
        support_local_ids = np.flatnonzero(support_local_mask).astype(np.int64)
        if len(support_local_ids) == 0:
            raise ValueError("红色标记五点没有可用的两环传递节点")
        support_particle_ids = torch.as_tensor(
            support_local_ids + int(handle.particle_start),
            device=rest.device,
            dtype=torch.long,
        )

        control_mask = torch.zeros(
            len(self._tissue_rest_positions),
            device=rest.device,
            dtype=torch.bool,
        )
        control_mask[particle_ids] = True
        self.environment.super_sim_grasp_control_mask = control_mask
        # The selected Dirichlet core and exactly one incident-tet ring are
        # excluded from RGB/stiffness learning.  The surrounding tissue stays
        # dynamic and receives no ground-truth motion.
        self.environment.super_sim_grasp_control_mask_is_complete = False
        self.environment.super_sim_grasp_boundary_mode = (
            self._sim_grasp_boundary_mode
        )
        self.environment.super_sim_grasp_boundary_schema = schema
        self.environment.super_sim_grasp_boundary_path = str(boundary_path)
        self.environment.super_sim_grasp_boundary_particle_count = int(
            len(particle_ids)
        )
        self._sim_grasp_boundary_schema = schema
        self._sim_grasp_boundary_path = str(boundary_path)
        self._sim_grasp_particle_ids = particle_ids
        particle_inverse_mass = wp.to_torch(
            self.environment.sim.model.particle_inv_mass
        )
        self._sim_grasp_original_inverse_mass = (
            particle_inverse_mass[particle_ids].detach().clone()
        )
        if not bool(
            torch.all(self._sim_grasp_original_inverse_mass > 0.0).item()
        ):
            raise ValueError("红色标记五点必须是可运动的组织粒子")
        self._sim_grasp_support_particle_ids = support_particle_ids
        self._sim_grasp_trajectory_positions = torch.as_tensor(
            trajectory_positions, device=rest.device, dtype=rest.dtype
        )
        self._sim_grasp_trajectory_velocities = torch.as_tensor(
            trajectory_velocities, device=rest.device, dtype=rest.dtype
        )
        self._sim_psm_q7 = q7
        self._sim_grasp_schedule = grasp_schedule
        first_grasp_frame = int(grasp_frames[0])
        print(
            "[sim grasp boundary] ON; source=task_inputs prescribed trajectory "
            "+ PSM grasp phase; "
            f"mode={self._sim_grasp_boundary_mode}; schema={schema}; "
            f"first_grasp_frame={first_grasp_frame}; direct_particles="
            f"{len(particle_ids)}; local_ids={particle_local_ids.tolist()}; "
            f"natural_pbd_neighbor_particles={len(support_particle_ids)}; "
            "diagnostic_neighbor_rings=2; "
            "grasp_center_mm="
            f"{(marker_center * 1e3).round(3).tolist()}; "
            "boundary=prescribed_trajectory_while_closed; controlled patch "
            "is kinematic only while grasped and excluded from visual "
            "residual/stiffness"
        )

    def _evaluation_feedback_allowed(self, frame_index: int | None = None) -> bool:
        frame = self.current_frame_index if frame_index is None else int(frame_index)
        if self._evaluation_frame_is_holdout(frame):
            return False
        cutoff = self._evaluation_open_loop_start_frame
        if cutoff is not None and frame >= cutoff:
            return False
        if (
            self._sim_reconstruction_mode
            and self._sim_grasp_schedule is not None
            and frame != 0
            and not bool(self._sim_grasp_schedule[frame])
        ):
            # Frame zero calibrates the fixed appearance gap.  Afterwards RGB
            # state corrections are causal only while the recorded jaw is
            # actually holding tissue; pre-grasp tool/shadow changes must not
            # be explained as tissue deformation.
            return False
        return True

    def _evaluation_frame_is_holdout(
        self, frame_index: int | None = None
    ) -> bool:
        stride = self._evaluation_holdout_stride
        if stride is None:
            return False
        frame = (
            self.current_frame_index
            if frame_index is None
            else int(frame_index)
        )
        return frame % stride == self._evaluation_holdout_offset

    def _enter_evaluation_holdout_if_needed(self) -> None:
        """Prevent a 7:1 test RGB frame from entering any learned state."""
        if (
            not self._evaluation_frame_is_holdout()
            or self._evaluation_holdout_last_cancelled_frame
            == self.current_frame_index
        ):
            return
        self._evaluation_holdout_last_cancelled_frame = int(
            self.current_frame_index
        )
        if (
            self._pending_stiffness_validation is not None
            and self.stiffness_updater is not None
        ):
            pending = self._pending_stiffness_validation
            metrics = self.stiffness_updater.reject(
                "reconstruction_holdout_frame", pending.candidate
            )
            metrics.update(
                validation_status="cancelled_at_reconstruction_holdout",
                prediction_horizon_frames=(
                    self.current_frame_index - pending.frame_index
                ),
                prediction_rollout_steps=len(pending.commands),
            )
            self._record_terminal_stiffness_validation(
                pending, metrics, "reconstruction_holdout_frame"
            )
            self._pending_stiffness_validation = None
            self._last_stiffness_validation_metrics = metrics
        self._record_incomplete_stiffness_evaluations(
            "reconstruction_holdout_frame"
        )
        self._active_stiffness_evaluations.clear()
        print(
            "[trajectory evaluation] reconstruction holdout frame "
            f"{self.current_frame_index}: RGB residual=OFF; stiffness "
            "update/validation=OFF; state rendered for test only"
        )

    def _enter_evaluation_open_loop_if_needed(self) -> None:
        """Freeze every learned state before writing the first future frame.

        A stiffness proposal created near the end of the assimilation interval
        normally needs several later RGB observations for its validation
        rollout.  Letting that proposal cross the benchmark split would leak
        future RGB into the learned material field even though ordinary visual
        residual updates are disabled.  Reject the unfinished proposal and
        stop diagnostic committed-rollout jobs exactly at the split; already
        committed particle state and stiffness arrays are left untouched.
        """
        cutoff = self._evaluation_open_loop_start_frame
        if (
            cutoff is None
            or self.current_frame_index < cutoff
            or self._evaluation_open_loop_entered
        ):
            return
        self._evaluation_open_loop_entered = True
        if (
            self._pending_stiffness_validation is not None
            and self.stiffness_updater is not None
        ):
            pending = self._pending_stiffness_validation
            metrics = self.stiffness_updater.reject(
                "future_open_loop_freeze", pending.candidate
            )
            metrics.update(
                validation_status="cancelled_at_future_split",
                prediction_horizon_frames=(
                    self.current_frame_index - pending.frame_index
                ),
                prediction_rollout_steps=len(pending.commands),
            )
            self._record_terminal_stiffness_validation(
                pending, metrics, "future_open_loop_freeze"
            )
            self._pending_stiffness_validation = None
            self._last_stiffness_validation_metrics = metrics
        self._record_incomplete_stiffness_evaluations(
            "future_open_loop_freeze"
        )
        self._active_stiffness_evaluations.clear()
        self._stiffness_history.clear()
        self._previous_visual_residual = None
        print(
            "[trajectory evaluation] future open loop entered at frame "
            f"{self.current_frame_index}; RGB feedback=OFF; stiffness "
            "proposals=FROZEN; committed material=HELD"
        )

    def _set_sim_grasp_boundary_kinematic(self, enabled: bool) -> None:
        """Make the five prescribed nodes a true Dirichlet boundary.

        Leaving their dynamic inverse masses enabled makes every PBD edge
        split its correction between the prescribed node and its neighbor.
        The later exact target projection then discards the prescribed half,
        which is precisely the five-point spike seen in the GUI. Zero inverse
        mass during a closed grasp sends the complete constraint correction
        into free neighboring tissue. Release restores the original masses.
        """
        if (
            self._sim_grasp_particle_ids is None
            or self._sim_grasp_original_inverse_mass is None
        ):
            return
        inverse_mass = wp.to_torch(
            self.environment.sim.model.particle_inv_mass
        )
        with torch.no_grad():
            if enabled:
                inverse_mass[self._sim_grasp_particle_ids] = 0.0
            else:
                inverse_mass[self._sim_grasp_particle_ids] = (
                    self._sim_grasp_original_inverse_mass
                )

    def _update_sim_grasp_target(self) -> None:
        if not self._sim_reconstruction_mode:
            return
        assert self._sim_grasp_schedule is not None
        assert self._sim_grasp_particle_ids is not None
        assert self._sim_grasp_trajectory_positions is not None
        assert self._sim_grasp_trajectory_velocities is not None
        scheduled = bool(self._sim_grasp_schedule[self.current_frame_index])
        if not scheduled:
            self._set_sim_grasp_boundary_kinematic(False)
            self._sim_grasp_active = False
            self._sim_grasp_trajectory_alignment = None
            self._sim_grasp_capture_positions = None
            self._sim_grasp_trajectory_origin = None
            self._sim_grasp_support_capture_positions = None
            self._sim_grasp_target = None
            self._sim_grasp_target_velocity = None
            self._sim_grasp_tracking_error_m = 0.0
            self._sim_grasp_projection_correction_m = 0.0
            self._sim_grasp_natural_neighbor_displacement_m = 0.0
            return

        trajectory_position = self._sim_grasp_trajectory_positions[
            self.current_frame_index
        ]
        if (
            not self._sim_grasp_active
            or self._sim_grasp_trajectory_alignment is None
        ):
            positions = wp.to_torch(self.environment.sim.state_0.particle_q)
            captured = positions[self._sim_grasp_particle_ids].detach().clone()
            assert self._sim_grasp_support_particle_ids is not None
            self._sim_grasp_support_capture_positions = positions[
                self._sim_grasp_support_particle_ids
            ].detach().clone()
            # Align only once at closure so reconstruction/GT rest-mesh
            # mismatch cannot create a snap. From this point onward the
            # prescribed displacement is exactly the simulated marker path.
            self._sim_grasp_trajectory_alignment = (
                captured - trajectory_position
            )
            self._sim_grasp_capture_positions = captured
            self._sim_grasp_trajectory_origin = trajectory_position.detach().clone()
            self._set_sim_grasp_boundary_kinematic(True)
            self._sim_grasp_active = True

        assert self._sim_grasp_capture_positions is not None
        assert self._sim_grasp_trajectory_origin is not None
        self._sim_grasp_target = self._sim_grasp_capture_positions + (
            self._sim_grasp_coupling_gain
            * (trajectory_position - self._sim_grasp_trajectory_origin)
        )
        self._sim_grasp_target_velocity = (
            self._sim_grasp_coupling_gain
            * self._sim_grasp_trajectory_velocities[self.current_frame_index]
        )

    def _project_sim_grasp_boundary(self, *, update_gaussians: bool) -> None:
        if not self._sim_grasp_active:
            return
        assert self._sim_grasp_particle_ids is not None
        assert self._sim_grasp_target is not None
        assert self._sim_grasp_target_velocity is not None
        ids = self._sim_grasp_particle_ids
        current_positions = wp.to_torch(
            self.environment.sim.state_0.particle_q
        )[ids]
        self._sim_grasp_projection_correction_m = float(
            torch.linalg.vector_norm(
                self._sim_grasp_target - current_positions, dim=1
            ).max().item()
        )
        with torch.no_grad():
            for state in (
                self.environment.sim.state_0,
                self.environment.sim.state_1,
            ):
                wp.to_torch(state.particle_q)[ids] = self._sim_grasp_target
                wp.to_torch(state.particle_qd)[ids] = (
                    self._sim_grasp_target_velocity
                )
        final_positions = wp.to_torch(
            self.environment.sim.state_0.particle_q
        )[ids]
        self._sim_grasp_tracking_error_m = float(
            torch.linalg.vector_norm(
                self._sim_grasp_target - final_positions, dim=1
            ).max().item()
        )
        if update_gaussians:
            assert self._sim_grasp_support_particle_ids is not None
            assert self._sim_grasp_support_capture_positions is not None
            natural_neighbors = wp.to_torch(
                self.environment.sim.state_0.particle_q
            )[self._sim_grasp_support_particle_ids]
            self._sim_grasp_natural_neighbor_displacement_m = float(
                torch.linalg.vector_norm(
                    natural_neighbors
                    - self._sim_grasp_support_capture_positions,
                    dim=1,
                ).max().item()
            )
            self.environment.sim.update_gaussian_transforms()

    def _sim_grasp_contact_metrics(self) -> dict[str, object]:
        count = (
            0
            if not self._sim_grasp_active
            or self._sim_grasp_particle_ids is None
            else int(len(self._sim_grasp_particle_ids))
        )
        natural_neighbor_count = (
            0
            if not self._sim_grasp_active
            or self._sim_grasp_support_particle_ids is None
            else int(len(self._sim_grasp_support_particle_ids))
        )
        jaw_angle = 0.0
        if self._sim_psm_q7 is not None:
            jaw_angle = float(self._sim_psm_q7[self.current_frame_index, -1])
        return {
            "contact_count": count,
            "maximum_penetration_m": 0.0,
            "persistent_grip_active": self._sim_grasp_active,
            "persistent_grip_particle_count": count,
            "persistent_grip_direct_particle_count": count,
            "persistent_grip_support_particle_count": 0,
            "natural_pbd_neighbor_particle_count": natural_neighbor_count,
            "natural_pbd_neighbor_displacement_maximum_m": (
                self._sim_grasp_natural_neighbor_displacement_m
            ),
            "persistent_grip_anchor_error_maximum_m": (
                self._sim_grasp_tracking_error_m
            ),
            "prescribed_boundary_preprojection_correction_m": (
                self._sim_grasp_projection_correction_m
            ),
            "persistent_grip_q7_angle_rad": jaw_angle,
            "persistent_grip_q7_timestamp_s": float(self.current_timestep),
            "persistent_grip_q7_motion_state": (
                "grasp" if self._sim_grasp_active else "open"
            ),
        }

    def state_index_at(self, timestep: float) -> int:
        timestamps = self.environment.super_psm_lnd_timestamps
        state_index = int(
            np.searchsorted(timestamps, timestep, side="right") - 1
        )
        return max(0, min(state_index, len(timestamps) - 1))

    def commanded_psm_q7_at(self, timestep: float) -> np.ndarray:
        """Return the recorded q7 command plus explicit manual calibration."""
        state_index = self.state_index_at(timestep)
        q7 = np.asarray(
            self.environment.super_psm_q7_states[state_index],
            dtype=np.float64,
        ).copy()
        return q7 + self.manual_psm_joint_offsets()

    def q_full_at(self, timestep: float) -> torch.Tensor:
        # Articulation, visible LND jaws, and collision bodies all receive the
        # recorded q7 directly.  No artificial opening or resistance limit is
        # inserted into the jaw kinematics.
        q7 = self.commanded_psm_q7_at(timestep)
        q_full = expand_psm_q7_to_urdf_order(
            q7,
            mimic_cfg=self.mimic_cfg,
            joint_order=self.joint_order,
        )
        return torch.from_numpy(q_full).float()

    def manual_psm_joint_offsets(self) -> np.ndarray:
        offsets = np.zeros(7, dtype=np.float64)
        offsets[3] = np.deg2rad(self.psm_roll_offset_deg)
        if self.psm_wrist_rod_offset_enabled:
            offsets[4] = np.deg2rad(self.psm_wrist_rod_offset_deg)
        offsets[6] = np.deg2rad(self.psm_jaw_offset_deg)
        return offsets

    def apply_current_psm_pose(self) -> None:
        self._last_q_full = self.q_full_at(self.current_timestep)
        self.environment.set_robot_q(
            PSM_ARTICULATION_INDEX, self._last_q_full
        )
        self.environment.set_robot_desired_q(
            PSM_ARTICULATION_INDEX, self._last_q_full
        )
        apply_psm_lnd_pose(
            self.environment,
            self._last_state_index,
            joint_offsets=self.manual_psm_joint_offsets(),
            translation_offset=self.manual_psm_translation_world(),
        )

    def manual_psm_translation_world(self) -> np.ndarray:
        fixed_world = self.psm_world_translation_mm / 1000.0
        if self.environment.frames is None:
            return fixed_world.copy()
        X_CW = (
            self.environment.frames.X_CWs_opencv_gpu[0]
            .detach()
            .cpu()
            .numpy()
        )
        translation_camera = self.psm_manual_camera_translation_mm / 1000.0
        return fixed_world + X_CW[:3, :3].T @ translation_camera

    def reset(self):
        self._set_sim_grasp_boundary_kinematic(False)
        self._record_incomplete_stiffness_evaluations("reset")
        self._active_stiffness_evaluations.clear()
        self._stiffness_evaluation_epoch += 1
        self._action_phase_classifier.reset()
        self._current_action_phase = "idle"
        set_psm_tissue_collisions(self.environment, False)
        self.psm_tissue_collisions_enabled = False
        self.psm_manual_camera_translation_mm[:] = (
            self.default_psm_camera_translation_mm
        )
        self.current_frame_index = 0
        self.current_timestep = float(self.playback_timestamps[0])
        self.playing = False
        self._last_state_index = self.state_index_at(self.current_timestep)
        self._visual_force_step = self.visual_force_update_interval - 1
        self.environment.sim.copy_embodied_gaussian_rollout_state(
            self.first_state
        )
        self.environment.sim.clear_soft_visual_force_cache()
        self.environment.sim.clear_visual_tissue_residual_metrics()
        if self.stiffness_updater is not None:
            self.stiffness_updater.reset()
            self.environment.physics_settings.particle_velocity_damping_per_second = (
                self.stiffness_updater.global_velocity_damping_per_second
            )
        self._pending_stiffness_validation = None
        self._stiffness_history.clear()
        self._last_stiffness_gate_timestep = None
        self._last_stiffness_gate_jaw_angle = None
        self._stiffness_transition_cooldown = 0
        self._last_stiffness_grip_active = False
        self._last_stiffness_contact_count = None
        self._last_stiffness_validation_metrics = None
        self._previous_visual_residual = None
        self._visual_residual_solve_count = 0
        self._visual_residual_busy = False
        self._single_visual_residual_requested = False
        self._last_visual_residual_frame_index = -1
        self._last_visual_residual_accepted = None
        self._flow_depth_source_states.clear()
        self._flow_depth_source_rollouts.clear()
        self._warp_gradient_transitions.clear()
        self._flow_depth_reference_range_centers = None
        self._last_flow_depth_runtime_frame_index = -1
        self._flow_depth_update_count = 0
        self._last_trajectory_observation_frame_index = -1
        self._physics_iteration_count = 0
        self._evaluation_open_loop_entered = False
        self._evaluation_holdout_last_cancelled_frame = -1
        self._tissue_max_displacement_m = 0.0
        self._sim_grasp_active = False
        self._sim_grasp_support_capture_positions = None
        self._sim_grasp_trajectory_alignment = None
        self._sim_grasp_capture_positions = None
        self._sim_grasp_trajectory_origin = None
        self._sim_grasp_coupling_gain = 1.0
        self._sim_grasp_target = None
        self._sim_grasp_target_velocity = None
        self._sim_grasp_tracking_error_m = 0.0
        self._sim_grasp_projection_correction_m = 0.0
        self._sim_grasp_natural_neighbor_displacement_m = 0.0
        self.environment.sim.eval_ik()
        self.go_to_frame(0)
        self.environment.sim.sync_kinematic_body_interpolation()
        if self.default_psm_tissue_contact_enabled:
            self.psm_tissue_collisions_enabled = set_psm_tissue_collisions(
                self.environment, True
            )
        self._initial_base_q = self.current_psm_base_q()
        self._initial_tissue_q = self.current_tissue_q()
        self._last_base_monitor_time = -float("inf")
        self._last_tissue_monitor_time = -float("inf")
        self.maybe_print_psm_base_q(force=True)
        self.maybe_print_tissue_q(force=True)
        if self.stiffness_metrics_recorder is not None:
            self.stiffness_metrics_recorder.record(
                event="reset",
                frame_index=self.current_frame_index,
                timestamp_s=self.current_timestep,
                phase=self._current_action_phase,
                details={"epoch": self._stiffness_evaluation_epoch},
                force_summary=True,
            )

    def go_to_frame(self, frame_index: int):
        self.current_frame_index = max(
            0, min(int(frame_index), len(self.playback_timestamps) - 1)
        )
        self._enter_evaluation_open_loop_if_needed()
        self.go_to_timestep(
            float(self.playback_timestamps[self.current_frame_index])
        )
        self._enter_evaluation_holdout_if_needed()

    def advance_one_frame(self) -> bool:
        next_frame_index = self.current_frame_index + 1
        if next_frame_index >= len(self.playback_timestamps):
            self.playing = False
            return False
        self.go_to_frame(next_frame_index)
        if self.current_frame_index == len(self.playback_timestamps) - 1:
            self.playing = False
        return True

    def go_to_timestep(self, timestep: float):
        self.current_timestep = timestep
        self._last_state_index = self.state_index_at(timestep)

        # Both the articulation and all LND visual/collision bodies use the
        # same raw q7 angle. apply_psm_pose_driver_joint_offsets() rotates the
        # two jaws about the shared local-z hinge by +/-q7/2.
        self.apply_current_psm_pose()
        self.dataset_manager.update_frames(timestep)
        self._update_sim_grasp_target()

    def psm_base_body_id(self) -> int | None:
        """返回第 0 个环境里的 PSM 基座 body id。

        body_q 是按 body id 索引的。SUPER 场景构建时已经把
        PSM1_psm_base_link 的 id 存到 environment.super_psm_base_body_ids。
        """
        body_ids = getattr(self.environment, "super_psm_base_body_ids", [])
        if not body_ids:
            return None
        return int(body_ids[0])

    def current_psm_base_q(self) -> np.ndarray | None:
        """读取当前 PSM 基座 body_q。

        返回值是 7 维数组：[x, y, z, qx, qy, qz, qw]。前三个数是基座在
        world 坐标系的位置，后四个数是姿态四元数。
        """
        body_id = self.psm_base_body_id()
        if body_id is None:
            return None
        body_q = wp.to_torch(self.environment.sim.state_0.body_q)
        return body_q[body_id].detach().cpu().numpy().copy()

    def tissue_body_id(self) -> int | None:
        """返回第 0 个环境里的 tissue body id。"""
        body_ids = getattr(self.environment, "super_tissue_body_ids", [])
        if body_ids:
            return int(body_ids[0])
        body_id = getattr(self.environment, "super_tissue_body_id", None)
        if body_id is None:
            return None
        return int(body_id)

    def current_tissue_q(self) -> np.ndarray | None:
        """读取当前 tissue 刚体 body_q。

        返回值是 7 维数组：[x, y, z, qx, qy, qz, qw]。
        """
        body_id = self.tissue_body_id()
        if body_id is None:
            return None
        body_q = wp.to_torch(self.environment.sim.state_0.body_q)
        return body_q[body_id].detach().cpu().numpy().copy()

    def format_pose_delta(self, q: np.ndarray, q0: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        p = q[:3]
        quat = q[3:]
        p0 = q0[:3]
        quat0 = q0[3:]
        dpos = float(np.linalg.norm(p - p0))

        quat_norm = np.linalg.norm(quat)
        quat0_norm = np.linalg.norm(quat0)
        if quat_norm > 0.0 and quat0_norm > 0.0:
            quat_dot = float(abs(np.dot(quat / quat_norm, quat0 / quat0_norm)))
            dangle_deg = float(np.degrees(2.0 * np.arccos(np.clip(quat_dot, -1.0, 1.0))))
        else:
            dangle_deg = float("nan")
        return p, quat, dpos, dangle_deg

    def maybe_print_psm_base_q(self, force: bool = False):
        if not self.monitor_psm_base_q:
            return

        current_time = self.environment.time()
        if not force and current_time - self._last_base_monitor_time < self.monitor_interval:
            return
        self._last_base_monitor_time = current_time

        body_id = self.psm_base_body_id()
        q = self.current_psm_base_q()
        if body_id is None or q is None:
            if not self._warned_missing_base_id:
                print("[PSM base monitor] 没找到 PSM1_psm_base_link 的 body id，无法监控基座。")
                self._warned_missing_base_id = True
            return

        if self._initial_base_q is None:
            self._initial_base_q = q.copy()

        p, quat, dpos, dangle_deg = self.format_pose_delta(q, self._initial_base_q)

        print(
            "[PSM base monitor] "
            f"sim_t={current_time:.4f}s body_id={body_id} "
            f"p=({p[0]:+.6f}, {p[1]:+.6f}, {p[2]:+.6f}) "
            f"q_xyzw=({quat[0]:+.6f}, {quat[1]:+.6f}, {quat[2]:+.6f}, {quat[3]:+.6f}) "
            f"dpos_from_reset={dpos:.9f}m dangle_from_reset={dangle_deg:.6f}deg"
        )

    def maybe_print_tissue_q(self, force: bool = False):
        if not self.monitor_tissue_q:
            return

        current_time = self.environment.time()
        if not force and current_time - self._last_tissue_monitor_time < self.monitor_interval:
            return
        self._last_tissue_monitor_time = current_time

        body_id = self.tissue_body_id()
        q = self.current_tissue_q()
        if body_id is None or q is None:
            if not self._warned_missing_tissue_id:
                print("[tissue monitor] 没找到 tissue 的 body id，无法监控组织。")
                self._warned_missing_tissue_id = True
            return

        if self._initial_tissue_q is None:
            self._initial_tissue_q = q.copy()

        p, quat, dpos, dangle_deg = self.format_pose_delta(q, self._initial_tissue_q)

        print(
            "[tissue monitor] "
            f"sim_t={current_time:.4f}s body_id={body_id} "
            f"p=({p[0]:+.6f}, {p[1]:+.6f}, {p[2]:+.6f}) "
            f"q_xyzw=({quat[0]:+.6f}, {quat[1]:+.6f}, {quat[2]:+.6f}, {quat[3]:+.6f}) "
            f"dpos_from_reset={dpos:.9f}m dangle_from_reset={dangle_deg:.6f}deg"
        )

    def _apply_stiffness_gui_draft(self) -> None:
        updater = self.stiffness_updater
        draft = self._stiffness_gui_draft
        if updater is None or draft is None:
            self._stiffness_gui_message = "Online stiffness is not enabled."
            return
        if self.playing:
            self._stiffness_gui_message = "Pause playback, then press Apply again."
            return
        if (
            self._pending_stiffness_validation is not None
            or updater.pending_candidate is not None
        ):
            self._stiffness_gui_message = (
                "A candidate is still pending; keep paused for one physics tick."
            )
            return
        try:
            updater.reconfigure(draft)
        except (RuntimeError, ValueError) as error:
            self._stiffness_gui_message = f"Settings rejected: {error}"
            return
        # History scores and temporal/contact evidence were collected under a
        # different update policy.  Keep verified material and particle state,
        # but do not compare new candidates against stale tuning evidence.
        self._stiffness_history.clear()
        self._last_stiffness_validation_metrics = None
        self._previous_visual_residual = None
        self._stiffness_transition_cooldown = (
            STIFFNESS_TRANSITION_COOLDOWN_UPDATES
        )
        self._stiffness_gui_message = (
            "Applied: verified k preserved/clipped; EMA and history cleared."
        )

    def _apply_contact_grip_gui_draft(self) -> None:
        draft = self._contact_grip_gui_draft
        sim = self.environment.sim
        projector = sim.triangle_skin_contact_projector
        if draft is None or projector is None:
            self._contact_grip_gui_message = "Triangle-skin contact is not configured."
            return
        if self.playing:
            self._contact_grip_gui_message = (
                "Pause playback, then press Apply again."
            )
            return
        if (
            self._pending_stiffness_validation is not None
            or (
                self.stiffness_updater is not None
                and self.stiffness_updater.pending_candidate is not None
            )
        ):
            self._contact_grip_gui_message = (
                "A stiffness candidate is pending; keep paused for one physics tick."
            )
            return
        try:
            draft.validate()
            # Rebuilding is intentional: sample masks, grip transfer tets and
            # support neighborhoods depend on these settings.  Constructing
            # the replacement first also keeps the live projector intact if
            # validation or allocation fails.
            sim.configure_triangle_skin_contacts(
                projector.tool_shape_ids,
                sample_spacing_m=draft.sample_spacing_m,
                spread_layers=draft.contact_spread_layers,
                top_support_lateral_radius_m=(
                    projector.top_support_lateral_radius_m
                ),
                top_support_depth_m=projector.top_support_depth_m,
                top_support_weight_scale=projector.top_support_weight_scale,
                top_pressure_shoulder_lateral_radius_m=(
                    projector.top_pressure_shoulder_lateral_radius_m
                ),
                top_pressure_shoulder_depth_m=(
                    projector.top_pressure_shoulder_depth_m
                ),
                top_pressure_shoulder_upward_scale=(
                    projector.top_pressure_shoulder_upward_scale
                ),
                top_pressure_shoulder_outward_scale=(
                    projector.top_pressure_shoulder_outward_scale
                ),
                top_pressure_shoulder_bias_direction_world=(
                    projector.top_pressure_shoulder_bias_direction_world
                ),
                top_pressure_shoulder_bias_start_m=(
                    projector.top_pressure_shoulder_bias_start_m
                ),
                top_barrier_lateral_tolerance_m=(
                    draft.top_barrier_lateral_tolerance_m
                ),
                top_barrier_contact_patch_radius_m=(
                    draft.top_barrier_contact_patch_radius_m
                ),
                top_barrier_clearance_m=draft.top_barrier_clearance_m,
                top_barrier_shape_ids=projector.top_barrier_shape_ids,
                jaw_contact_shape_ids=projector.jaw_contact_shape_ids,
                jaw_contact_distal_length_m=(
                    draft.jaw_contact_distal_length_m
                ),
                top_barrier_distal_length_m=(
                    draft.top_barrier_distal_length_m
                ),
                top_barrier_tip_allowance_m=(
                    draft.top_barrier_tip_allowance_m
                ),
                jaw_friction_coefficient=draft.jaw_friction_coefficient,
                persistent_grip_enabled=draft.persistent_grip_enabled,
                persistent_grip_minimum_contact_samples_per_jaw=(
                    draft.grip_minimum_contact_samples_per_jaw
                ),
                persistent_grip_nearest_surface_particles=(
                    projector.persistent_grip_nearest_surface_particles
                ),
                persistent_grip_maximum_jaw_patch_separation_m=(
                    draft.grip_maximum_jaw_patch_separation_m
                ),
                persistent_grip_activation_steps=(
                    draft.grip_activation_steps
                ),
                persistent_grip_maximum_capture_penetration_m=(
                    draft.grip_maximum_capture_penetration_m
                ),
                persistent_grip_minimum_capture_volume_ratio=(
                    draft.grip_minimum_capture_volume_ratio
                ),
                persistent_grip_closed_angle_max_rad=(
                    draft.grip_closed_angle_max_rad
                ),
                persistent_grip_release_angle_min_rad=(
                    draft.grip_release_angle_min_rad
                ),
                persistent_grip_release_angle_delta_rad=(
                    draft.grip_release_angle_delta_rad
                ),
                persistent_grip_wide_open_angle_rad=(
                    draft.grip_wide_open_angle_rad
                ),
                persistent_grip_angle_motion_epsilon_rad=(
                    draft.grip_angle_motion_epsilon_rad
                ),
                persistent_grip_compliance_m_per_n=(
                    draft.grip_compliance_m_per_n
                ),
                persistent_grip_relaxation=draft.grip_relaxation,
                persistent_grip_maximum_correction_m=(
                    draft.grip_maximum_correction_m
                ),
                persistent_grip_transfer_layers=(
                    draft.grip_transfer_layers
                ),
                persistent_grip_minimum_volume_ratio=(
                    draft.grip_minimum_volume_ratio
                ),
                persistent_grip_support_radius_m=(
                    draft.grip_support_radius_m
                ),
                persistent_grip_support_generations=(
                    draft.grip_support_generations
                ),
            )
        except (RuntimeError, ValueError) as error:
            self._contact_grip_gui_message = f"Settings rejected: {error}"
            return

        physics = self.environment.physics_settings
        physics.triangle_skin_contact_margin_m = draft.contact_margin_m
        physics.triangle_skin_query_distance_m = draft.query_distance_m
        physics.triangle_skin_ccd_velocity_scale = draft.ccd_velocity_scale
        physics.triangle_skin_contact_relaxation = draft.contact_relaxation
        physics.triangle_skin_contact_max_correction_m = (
            draft.contact_max_correction_m
        )
        physics.triangle_skin_top_barrier_max_correction_m = (
            draft.top_barrier_max_correction_m
        )
        physics.triangle_skin_contact_iterations = draft.contact_iterations
        physics.triangle_skin_post_contact_material_iterations = (
            draft.post_contact_material_iterations
        )
        physics.triangle_skin_final_barrier_max_correction_m = (
            draft.final_barrier_max_correction_m
        )
        physics.triangle_skin_contact_min_volume_ratio = (
            draft.contact_min_volume_ratio
        )
        physics.triangle_skin_contact_substep_stride = (
            draft.contact_substep_stride
        )
        physics.contact_projection_velocity_scale = (
            draft.contact_projection_velocity_scale
        )
        physics.material_projection_velocity_scale = (
            draft.material_projection_velocity_scale
        )
        physics.particle_velocity_damping_per_second = (
            draft.particle_velocity_damping_per_second
        )

        # Contact semantics changed: preserve tissue/material state, but drop
        # the old grip anchors and any learning evidence gathered around them.
        if self.stiffness_updater is not None:
            self.stiffness_updater.invalidate_signal_history()
        self._stiffness_history.clear()
        self._last_stiffness_gate_timestep = None
        self._last_stiffness_gate_jaw_angle = None
        self._last_stiffness_grip_active = False
        self._last_stiffness_contact_count = None
        self._last_stiffness_validation_metrics = None
        self._previous_visual_residual = None
        self._stiffness_transition_cooldown = (
            STIFFNESS_TRANSITION_COOLDOWN_UPDATES
        )
        self._contact_ui_metrics = sim.triangle_skin_contact_metrics()
        self._last_contact_ui_refresh_time = self.environment.time()
        self._contact_grip_gui_message = (
            "Applied: tissue pose kept; contact cache and old grip anchors cleared."
        )

    def _draw_manual_pose_panel(self, imgui) -> None:
        if not imgui.collapsing_header("PSM manual alignment"):
            return
        joint_changed = False
        changed, value = imgui.slider_float(
            "PSM roll offset deg",
            float(self.psm_roll_offset_deg),
            -180.0,
            180.0,
        )
        if changed:
            self.psm_roll_offset_deg = float(value)
            joint_changed = True
        if self.psm_wrist_rod_offset_enabled:
            changed, value = imgui.slider_float(
                "PSM wrist pitch offset deg",
                float(self.psm_wrist_rod_offset_deg),
                -60.0,
                60.0,
            )
            if changed:
                self.psm_wrist_rod_offset_deg = float(value)
                joint_changed = True
        changed, value = imgui.slider_float(
            "PSM jaw offset deg",
            float(self.psm_jaw_offset_deg),
            -30.0,
            30.0,
        )
        if changed:
            self.psm_jaw_offset_deg = float(value)
            joint_changed = True
        if joint_changed:
            self.go_to_timestep(self.current_timestep)

        translation_changed = False
        labels_and_ranges = (
            ("Manual image X mm (+right)", 0, -5.0, 5.0),
            ("Manual image Y mm (+down)", 1, -5.0, 5.0),
            ("Manual camera Z mm (+far)", 2, -30.0, 30.0),
        )
        for label, index, minimum, maximum in labels_and_ranges:
            changed, value = imgui.slider_float(
                label,
                float(self.psm_manual_camera_translation_mm[index]),
                minimum,
                maximum,
            )
            if changed:
                self.psm_manual_camera_translation_mm[index] = value
                translation_changed = True
        if translation_changed:
            self.go_to_timestep(self.current_timestep)
        if imgui.button("Reset manual position"):
            self.psm_manual_camera_translation_mm[:] = (
                self.default_psm_camera_translation_mm
            )
            self.go_to_timestep(self.current_timestep)
        imgui.same_line()
        if imgui.button("Reset all pose offsets"):
            self.psm_roll_offset_deg = self.default_psm_roll_offset_deg
            self.psm_wrist_rod_offset_deg = 0.0
            self.psm_jaw_offset_deg = 0.0
            self.psm_manual_camera_translation_mm[:] = (
                self.default_psm_camera_translation_mm
            )
            self.go_to_timestep(self.current_timestep)
        total_mm = self.manual_psm_translation_world() * 1000.0
        q7 = self.commanded_psm_q7_at(self.current_timestep)
        imgui.text(
            "World XYZ / q7: "
            f"({total_mm[0]:+.3f}, {total_mm[1]:+.3f}, "
            f"{total_mm[2]:+.3f}) mm / {q7[6]:+.3f} rad"
        )

    @staticmethod
    def _stiffness_tooltip(imgui, text: str) -> None:
        if imgui.is_item_hovered():
            imgui.set_tooltip(text)

    def _draw_stiffness_panel(self, imgui) -> None:
        updater = self.stiffness_updater
        if updater is None:
            imgui.text("Online stiffness: OFF")
            return

        settings = updater.settings
        metrics = updater.last_metrics
        if metrics is None:
            imgui.text("Online stiffness: waiting for accepted visual evidence")
            imgui.text(
                "Allowed dist / shape: "
                f"{settings.distance_minimum:.2f}..{settings.distance_maximum:.2f} / "
                f"{settings.shape_minimum:.3f}..{settings.shape_maximum:.3f}"
            )
        else:
            imgui.text(
                "Stiffness: "
                f"{metrics.get('status', 'unknown')} | "
                f"commit {int(metrics.get('update_count', 0))} | "
                f"reject {int(metrics.get('rejected_count', 0))}"
            )
            imgui.text(
                "Distance min / median / max: "
                f"{float(metrics['distance_minimum']):.4f} / "
                f"{float(metrics['distance_median']):.4f} / "
                f"{float(metrics['distance_maximum']):.4f}"
            )
            imgui.text(
                "Shape min / median / max: "
                f"{float(metrics['shape_minimum']):.5f} / "
                f"{float(metrics['shape_median']):.5f} / "
                f"{float(metrics['shape_maximum']):.5f}"
            )
            imgui.text(
                "Evidence active / harden / soften: "
                f"{int(metrics['active_particles'])} / "
                f"{int(metrics['hardening_particles'])} / "
                f"{int(metrics['softening_particles'])}"
            )
            validation = self._last_stiffness_validation_metrics
            if validation is not None and validation.get("rejection_reason"):
                imgui.text(
                    "Latest rejection: "
                    f"{validation['rejection_reason']}"
                )
        recorder = getattr(self, "stiffness_metrics_recorder", None)
        if recorder is not None:
            imgui.text(
                "Open-loop eval: ON | active "
                f"{len(self._active_stiffness_evaluations)} | events "
                f"{recorder.event_count}"
            )

        if imgui.collapsing_header("Online stiffness tuning (pause to apply)"):
            draft = self._stiffness_gui_draft or settings
            changed, value = imgui.slider_float(
                "Distance lower bound",
                float(draft.distance_minimum),
                0.01,
                1.00,
                "%.3f",
            )
            self._stiffness_tooltip(
                imgui, "Lower = softer regions and more local stretch."
            )
            if changed:
                draft = replace(draft, distance_minimum=float(value))
            changed, value = imgui.slider_float(
                "Distance upper bound",
                float(draft.distance_maximum),
                0.10,
                10.00,
                "%.2f",
            )
            self._stiffness_tooltip(
                imgui, "Higher = harder regions; may resist the gripper more."
            )
            if changed:
                draft = replace(draft, distance_maximum=float(value))
            changed, value = imgui.slider_float(
                "Shape lower bound",
                float(draft.shape_minimum),
                0.0001,
                0.030,
                "%.4f",
            )
            self._stiffness_tooltip(
                imgui,
                "Higher raises the softest allowed shape stiffness; "
                "it must not exceed the selected upper bound.",
            )
            if changed:
                draft = replace(draft, shape_minimum=float(value))
            changed, value = imgui.slider_float(
                "Shape upper bound",
                float(draft.shape_maximum),
                0.001,
                0.100,
                "%.4f",
            )
            self._stiffness_tooltip(
                imgui,
                "Above 0.020 is an intentionally wide experimental range; "
                "high values can spread a local impulse farther.",
            )
            if changed:
                draft = replace(draft, shape_maximum=float(value))
            changed, value = imgui.slider_float(
                "Learning rate (log space)",
                float(draft.log_learning_rate),
                0.02,
                0.80,
                "%.3f",
            )
            self._stiffness_tooltip(
                imgui, "Higher = each accepted image changes stiffness faster."
            )
            if changed:
                draft = replace(draft, log_learning_rate=float(value))
            ema_new_weight = 1.0 - draft.signal_ema_decay
            changed, value = imgui.slider_float(
                "EMA new-evidence weight",
                float(ema_new_weight),
                0.05,
                0.60,
                "%.2f",
            )
            self._stiffness_tooltip(
                imgui, "Higher = reacts faster but follows image noise more."
            )
            if changed:
                draft = replace(draft, signal_ema_decay=1.0 - float(value))
            changed, value = imgui.slider_float(
                "Hardening bias",
                float(draft.hardening_bias),
                -0.20,
                0.40,
                "%.2f",
            )
            self._stiffness_tooltip(
                imgui, "Higher = ambiguous evidence is more likely to harden."
            )
            if changed:
                draft = replace(draft, hardening_bias=float(value))
            changed, value = imgui.slider_float(
                "Shape update gain",
                float(draft.shape_update_gain),
                0.50,
                2.50,
                "%.2f",
            )
            self._stiffness_tooltip(
                imgui, "Multiplier applied to the distance log step for shape."
            )
            if changed:
                draft = replace(draft, shape_update_gain=float(value))

            if imgui.collapsing_header("Advanced evidence and smoothing"):
                changed, value = imgui.slider_float(
                    "Maximum log step",
                    float(draft.maximum_log_step),
                    0.03,
                    0.60,
                    "%.3f",
                )
                self._stiffness_tooltip(
                    imgui, "Absolute per-commit safety cap before bounds."
                )
                if changed:
                    draft = replace(draft, maximum_log_step=float(value))
                changed, value = imgui.slider_float(
                    "Minimum residual (mm)",
                    float(draft.minimum_residual_m * 1.0e3),
                    0.005,
                    0.100,
                    "%.3f mm",
                )
                if changed:
                    draft = replace(
                        draft, minimum_residual_m=float(value) * 1.0e-3
                    )
                changed, value = imgui.slider_float(
                    "Residual full scale (mm)",
                    float(draft.residual_full_scale_m * 1.0e3),
                    0.05,
                    0.60,
                    "%.3f mm",
                )
                if changed:
                    draft = replace(
                        draft, residual_full_scale_m=float(value) * 1.0e-3
                    )
                changed, value = imgui.slider_float(
                    "Minimum deformation (mm)",
                    float(draft.minimum_deformation_m * 1.0e3),
                    0.02,
                    0.50,
                    "%.3f mm",
                )
                if changed:
                    draft = replace(
                        draft, minimum_deformation_m=float(value) * 1.0e-3
                    )
                changed, value = imgui.slider_float(
                    "Deformation full scale (mm)",
                    float(draft.deformation_full_scale_m * 1.0e3),
                    0.20,
                    2.00,
                    "%.3f mm",
                )
                if changed:
                    draft = replace(
                        draft, deformation_full_scale_m=float(value) * 1.0e-3
                    )
                changed, value = imgui.slider_int(
                    "Spatial smoothing passes",
                    int(draft.spatial_smoothing_iterations),
                    0,
                    3,
                )
                if changed:
                    draft = replace(
                        draft, spatial_smoothing_iterations=int(value)
                    )
                changed, value = imgui.slider_float(
                    "Neighbor smoothing blend",
                    float(draft.spatial_smoothing_blend),
                    0.0,
                    0.75,
                    "%.2f",
                )
                if changed:
                    draft = replace(
                        draft, spatial_smoothing_blend=float(value)
                    )
                changed, value = imgui.slider_float(
                    "Rejected EMA keep ratio",
                    float(draft.rejected_ema_decay),
                    0.0,
                    0.90,
                    "%.2f",
                )
                if changed:
                    draft = replace(draft, rejected_ema_decay=float(value))

            self._stiffness_gui_draft = draft
            if imgui.button("Apply stiffness settings"):
                self._apply_stiffness_gui_draft()
            imgui.same_line()
            if imgui.button("Restore startup values"):
                self._stiffness_gui_draft = self._startup_stiffness_settings
                self._stiffness_gui_message = (
                    "Startup values loaded as draft; press Apply."
                )
            imgui.text(self._stiffness_gui_message)

        if metrics is not None and imgui.collapsing_header(
            "Stiffness diagnostics"
        ):
            imgui.text(
                "Candidate / commit / reject: "
                f"{int(metrics.get('candidate_count', 0))} / "
                f"{int(metrics.get('update_count', 0))} / "
                f"{int(metrics.get('rejected_count', 0))}"
            )
            imgui.text(
                "Tet valid / masked; visual masked; u_t excluded: "
                f"{int(metrics['quality_valid_particles'])} / "
                f"{int(metrics['quality_masked_particles'])}; "
                f"{int(metrics.get('supervision_masked_particles', 0))}; "
                f"{int(metrics.get('control_excluded_particles', 0))}"
            )
            imgui.text(
                "EMA active / mean |log step|: "
                f"{int(metrics.get('ema_active_particles', 0))} / "
                f"{float(metrics.get('mean_absolute_log_step', 0.0)):.6f}"
            )
            imgui.text(
                "Edge roughness distance / shape: "
                f"{float(metrics.get('distance_edge_roughness', 0.0)):.6f} / "
                f"{float(metrics.get('shape_edge_roughness', 0.0)):.6f}"
            )
            validation = self._last_stiffness_validation_metrics
            if validation is not None:
                imgui.text(
                    "Prediction verified -> candidate / horizon: "
                    f"{float(validation.get('baseline_visual_loss', 0.0)):.6f} -> "
                    f"{float(validation.get('candidate_visual_loss', 0.0)):.6f} / "
                    f"{int(validation.get('prediction_horizon_frames', 0))} frames"
                )

    def _draw_material_panel(self, imgui, paper_mode: bool) -> None:
        tissue_young_modulus_pa = getattr(
            self.environment, "super_tissue_young_modulus_pa", None
        )
        if tissue_young_modulus_pa is None:
            return
        physics = self.environment.physics_settings
        if getattr(
            self.environment, "super_tissue_constraint_model", None
        ) == "paper":
            baseline = self.environment.super_tissue_paper_stiffness
            imgui.text(
                "Reset material dist / volume / shape: "
                f"{baseline['distance']:g} / {baseline['volume']:g} / "
                f"{baseline['shape']:g}"
            )
        else:
            imgui.text(
                "Tissue material / damping: "
                f"E={tissue_young_modulus_pa / 1e3:.2f} kPa / "
                f"{physics.particle_velocity_damping_per_second:.1f}/s"
            )
        imgui.text(
            "Tissue max displacement from Reset: "
            f"{self._tissue_max_displacement_m * 1e3:.2f} mm"
        )
        if paper_mode:
            self._draw_stiffness_panel(imgui)

        if not imgui.collapsing_header("Material and contact setup details"):
            return
        imgui.text(
            "Dynamics gravity / damping: "
            f"{getattr(self.environment, 'super_tissue_gravity_m_s2', 0.0):.1f} m/s^2 / "
            f"{physics.particle_velocity_damping_per_second:.1f}/s"
        )
        imgui.text(
            "Contact stride / iterations / post-material passes: "
            f"{physics.triangle_skin_contact_substep_stride} / "
            f"{physics.triangle_skin_contact_iterations} / "
            f"{physics.triangle_skin_post_contact_material_iterations}"
        )
        imgui.text(
            "Contact correction / final barrier: "
            f"{physics.triangle_skin_contact_max_correction_m * 1e3:.3f} / "
            f"{physics.triangle_skin_final_barrier_max_correction_m * 1e3:.3f} mm"
        )
        imgui.text(
            "Velocity transfer contact / material: "
            f"{physics.contact_projection_velocity_scale:.2f} / "
            f"{physics.material_projection_velocity_scale:.2f}"
        )
        metrics = self._contact_ui_metrics
        if metrics is not None:
            imgui.text(
                "Jaw distal / tip entry / support radius x depth: "
                f"{metrics.get('jaw_contact_distal_length_m', 0.0) * 1e3:.1f} / "
                f"{metrics.get('top_barrier_tip_allowance_m', 0.0) * 1e3:.1f} / "
                f"{metrics.get('top_support_lateral_radius_m', 0.0) * 1e3:.1f} x "
                f"{metrics.get('top_support_depth_m', 0.0) * 1e3:.1f} mm"
            )
        gap = getattr(
            self.environment, "super_psm_tissue_kinematic_guard_gap_m", None
        )
        if gap is not None:
            imgui.text(
                "Kinematic guard gap / last retraction: "
                f"{gap * 1e3:+.3f} / "
                f"{getattr(self.environment, 'super_psm_tissue_kinematic_guard_offset_m', 0.0) * 1e3:.3f} mm"
            )

    def _draw_contact_grip_tuning_panel(self, imgui) -> None:
        draft = self._contact_grip_gui_draft
        if draft is None:
            return
        if not imgui.collapsing_header(
            "Tip entry and grip (pause to apply)"
        ):
            return

        # The hidden top-barrier distal length follows a larger tip allowance
        # up to the jaw-contact distal length.  This gives the visible control
        # useful range while preserving tip < barrier <= jaw geometry.
        maximum_tip_entry_mm = max(
            0.0,
            draft.jaw_contact_distal_length_m * 1.0e3 - 0.01,
        )
        changed, value = imgui.slider_float(
            "Tip entry allowance (mm)",
            float(draft.top_barrier_tip_allowance_m * 1.0e3),
            0.0,
            maximum_tip_entry_mm,
            "%.2f mm",
        )
        self._stiffness_tooltip(
            imgui,
            "Higher lets the distal tip pass farther below the top barrier; "
            "the hidden barrier length expands safely up to the jaw length.",
        )
        if changed:
            tip_allowance_m = float(value) * 1.0e-3
            required_barrier_distal_m = min(
                draft.jaw_contact_distal_length_m,
                tip_allowance_m + 0.00001,
            )
            draft = replace(
                draft,
                top_barrier_tip_allowance_m=tip_allowance_m,
                top_barrier_distal_length_m=max(
                    draft.top_barrier_distal_length_m,
                    required_barrier_distal_m,
                ),
            )

        changed, value = imgui.slider_float(
            "Capture max penetration (mm)",
            float(draft.grip_maximum_capture_penetration_m * 1.0e3),
            0.10,
            10.00,
            "%.2f mm",
        )
        self._stiffness_tooltip(
            imgui,
            "Higher permits deeper overlap when persistent grip is captured.",
        )
        if changed:
            draft = replace(
                draft,
                grip_maximum_capture_penetration_m=(
                    float(value) * 1.0e-3
                ),
            )

        changed, value = imgui.slider_float(
            "Grip support radius (mm)",
            float(draft.grip_support_radius_m * 1.0e3),
            0.0,
            30.0,
            "%.2f mm",
        )
        self._stiffness_tooltip(
            imgui,
            "Higher carries a wider surface neighborhood with the four "
            "direct jaw anchors.",
        )
        if changed:
            draft = replace(
                draft,
                grip_support_radius_m=float(value) * 1.0e-3,
            )

        if imgui.collapsing_header("Particle jump stabilization"):
            changed, value = imgui.slider_float(
                "Grip correction cap (mm/substep)",
                float(draft.grip_maximum_correction_m * 1.0e3),
                0.005,
                2.000,
                "%.3f mm",
            )
            self._stiffness_tooltip(
                imgui,
                "Lower is the first control to try for isolated jumping "
                "particles; higher makes anchors chase the jaw faster.",
            )
            if changed:
                draft = replace(
                    draft,
                    grip_maximum_correction_m=(
                        float(value) * 1.0e-3
                    ),
                )

            changed, value = imgui.slider_float(
                "Grip compliance (m/N)",
                float(draft.grip_compliance_m_per_n),
                0.0,
                2.0,
                "%.3f",
            )
            self._stiffness_tooltip(
                imgui,
                "Higher makes the jaw attachment softer and reduces "
                "snapping; too high can make the grasp lag or slip.",
            )
            if changed:
                draft = replace(
                    draft,
                    grip_compliance_m_per_n=float(value),
                )

            changed, value = imgui.slider_float(
                "Material/grip velocity transfer",
                float(draft.material_projection_velocity_scale),
                0.0,
                1.0,
                "%.3f",
            )
            self._stiffness_tooltip(
                imgui,
                "Lower converts less material and persistent-grip position "
                "correction into next-step velocity.",
            )
            if changed:
                draft = replace(
                    draft,
                    material_projection_velocity_scale=float(value),
                )

            changed, value = imgui.slider_float(
                "Particle velocity damping (/s)",
                float(draft.particle_velocity_damping_per_second),
                0.0,
                100.0,
                "%.1f /s",
            )
            self._stiffness_tooltip(
                imgui,
                "Higher suppresses velocity oscillation globally; too high "
                "looks viscous and slows recovery.",
            )
            if changed:
                draft = replace(
                    draft,
                    particle_velocity_damping_per_second=float(value),
                )

        self._contact_grip_gui_draft = draft
        if imgui.button("Apply entry/grip settings"):
            self._apply_contact_grip_gui_draft()
        imgui.same_line()
        if imgui.button("Restore startup entry/grip"):
            startup = self._startup_contact_grip_settings
            if startup is not None:
                self._contact_grip_gui_draft = replace(
                    draft,
                    top_barrier_tip_allowance_m=(
                        startup.top_barrier_tip_allowance_m
                    ),
                    top_barrier_distal_length_m=(
                        startup.top_barrier_distal_length_m
                    ),
                    grip_maximum_capture_penetration_m=(
                        startup.grip_maximum_capture_penetration_m
                    ),
                    grip_support_radius_m=(
                        startup.grip_support_radius_m
                    ),
                    grip_maximum_correction_m=(
                        startup.grip_maximum_correction_m
                    ),
                    grip_compliance_m_per_n=(
                        startup.grip_compliance_m_per_n
                    ),
                    material_projection_velocity_scale=(
                        startup.material_projection_velocity_scale
                    ),
                    particle_velocity_damping_per_second=(
                        startup.particle_velocity_damping_per_second
                    ),
                )
            self._contact_grip_gui_message = (
                "Startup entry/grip values loaded as draft; press Apply."
            )
        imgui.text(self._contact_grip_gui_message)
    def _draw_contact_runtime_panel(self, imgui) -> None:
        self._draw_contact_grip_tuning_panel(imgui)
        metrics = self._contact_ui_metrics
        if metrics is None:
            imgui.text("Contact runtime: waiting for metrics")
            return
        jaw_shape_ids = metrics["jaw_contact_shape_ids"]
        contact_counts = metrics["contact_count_by_shape"]
        jaw_counts = tuple(contact_counts[index] for index in jaw_shape_ids)
        required = metrics["persistent_grip_minimum_contact_samples_per_jaw"]
        patch_separation_m = metrics["persistent_grip_jaw_patch_separation_m"]
        maximum_separation_m = metrics[
            "persistent_grip_maximum_jaw_patch_separation_m"
        ]
        if metrics["persistent_grip_active"]:
            verdict = "GRASPED"
        elif not metrics["persistent_grip_capture_allowed"]:
            verdict = "WAITING FOR JAW CLOSURE"
        elif min(jaw_counts, default=0) < required:
            verdict = "WAITING FOR TWO-SIDED CONTACT"
        elif patch_separation_m > maximum_separation_m:
            verdict = "CONTACT PATCHES TOO FAR APART"
        else:
            verdict = "CONTACT SUSTAINING"
        imgui.text(
            "Grip: "
            f"{verdict} | state={metrics['persistent_grip_q7_motion_state']} | "
            f"attached={metrics['persistent_grip_particle_count']}"
        )
        imgui.text(
            "Contact left/right | gap | penetration: "
            f"{jaw_counts} | {metrics['minimum_signed_distance_m'] * 1e3:+.3f} | "
            f"{metrics['maximum_penetration_m'] * 1e3:.3f} mm"
        )
        unsafe = int(metrics.get("material_safety_unsafe_tetrahedra", 0))
        local_unsafe = int(metrics.get("contact_local_unsafe_tetrahedra", 0))
        if unsafe or local_unsafe:
            imgui.text(
                "SAFETY WARNING global/local unsafe tets: "
                f"{unsafe} / {local_unsafe}"
            )

        if not imgui.collapsing_header("Contact diagnostics"):
            return
        imgui.text(
            "q7 / capture allowed / active: "
            f"{metrics['persistent_grip_q7_angle_rad']:+.3f} rad / "
            f"{bool(metrics['persistent_grip_capture_allowed'])} / "
            f"{bool(metrics['persistent_grip_active'])}"
        )
        imgui.text(
            "Selected / sustain / patch separation: "
            f"{metrics['persistent_grip_selected_particle_count']}/4 / "
            f"{metrics['persistent_grip_activation_counter']}/"
            f"{metrics['persistent_grip_activation_steps']} / "
            f"{patch_separation_m * 1e3:.2f} mm"
        )
        imgui.text(
            "Contact candidates / barrier contacts / spread layers: "
            f"{metrics['contact_count']} / "
            f"{metrics['top_barrier_contact_count']} / "
            f"{metrics['contact_spread_layers']}"
        )
        jaw_safe = metrics.get("persistent_grip_jaw_safe_scales", (1.0, 1.0))
        imgui.text(
            "Material / direct / jaw A-B safe scales: "
            f"{metrics.get('material_safety_step_scale', 1.0):.4f} / "
            f"{metrics.get('persistent_grip_direct_safe_scale', 1.0):.4f} / "
            f"{jaw_safe[0]:.4f}-{jaw_safe[1]:.4f}"
        )

    def _draw_visual_feedback_panel(self, imgui) -> None:
        imgui.text(
            f"Visual feedback {self.visual_feedback_mode}: "
            f"{'ACTIVE' if self.playing or self._single_visual_residual_requested else 'PAUSED'}"
        )
        if self.visual_feedback_mode == "residual":
            if self._sim_reconstruction_mode:
                imgui.text(
                    "Mask: full visible tissue (exact simulation semantic mask)"
                )
                imgui.text(
                    "Correction policy: every dataset frame once; "
                    f"{self.visual_residual_mapper.settings.iterations} Adam steps"
                )
                if not self.playing:
                    if imgui.button("Solve Visual Residual Once"):
                        self._single_visual_residual_requested = True
                        self._previous_visual_residual = None
            metrics = self.environment.sim.last_visual_tissue_residual_metrics
            if metrics is None:
                imgui.text("Residual: waiting for first solve")
                return
            imgui.text(
                "Residual accepted / loss reduction / max: "
                f"{'YES' if metrics['accepted'] else 'NO'} / "
                f"{float(metrics['visual_loss_reduction_fraction']) * 100.0:+.2f}% / "
                f"{float(metrics['maximum_residual_m']) * 1e3:.3f} mm"
            )
            if not metrics["accepted"]:
                imgui.text(f"Residual rejection: {metrics['rejection_reason']}")
            if not imgui.collapsing_header("Visual residual diagnostics"):
                return
            imgui.text(
                "Solve time / count / RMS: "
                f"{float(metrics.get('solve_elapsed_s', 0.0)) * 1e3:.1f} ms / "
                f"{int(metrics.get('solve_count', 0))} / "
                f"{float(metrics['rms_residual_m']) * 1e3:.3f} mm"
            )
            imgui.text(
                "Visual loss initial -> exact final: "
                f"{float(metrics['initial_visual_loss']):.6f} -> "
                f"{float(metrics.get('exact_final_visual_loss', metrics['final_visual_loss'])):.6f}"
            )
            for camera_index, values in enumerate(
                zip(
                    metrics.get("initial_camera_visual_losses", ()),
                    metrics.get("exact_final_camera_visual_losses", ()),
                    metrics.get("camera_active_pixel_counts", ()),
                    metrics.get("camera_mask_coverage_fractions", ()),
                )
            ):
                initial, final, active_pixels, coverage = values
                imgui.text(
                    f"Camera {camera_index}: {float(initial):.6f} -> "
                    f"{float(final):.6f}; px={int(active_pixels)}; "
                    f"mask={float(coverage) * 100.0:.1f}%"
                )
            imgui.text(
                "Minimum J initial -> final / safety frozen / grip excluded / backtracks: "
                f"{float(metrics['initial_minimum_volume_ratio']):.3f} -> "
                f"{float(metrics['minimum_volume_ratio']):.3f} / "
                f"{int(metrics['locally_frozen_particles'])} / "
                f"{int(metrics.get('dynamically_excluded_particles', 0))} / "
                f"{int(metrics['backtrack_count'])}"
            )
        elif self.visual_feedback_mode == "force":
            metrics = self.environment.sim.last_soft_visual_force_metrics
            if metrics is None:
                imgui.text("Visual force: waiting for first solve")
                return
            imgui.text(
                "Visual force applied / requested / active particles: "
                f"{metrics['applied_force_budget_n']:.5f} / "
                f"{metrics['force_budget_before_n']:.5f} N / "
                f"{metrics['active_particles']}"
            )
            if imgui.collapsing_header("Visual force diagnostics"):
                imgui.text(
                    "Target max / acceleration max: "
                    f"{metrics['target_delta_max_m'] * 1e3:.3f} mm / "
                    f"{metrics.get('maximum_particle_acceleration_m_s2', 0.0):.3f} m/s^2"
                )

    def draw(self):
        # marsoom/pyglet 需要真实显示环境；放到 draw 里导入，可以让
        # `python examples/example_embodied_super_offline.py --help` 这类非 GUI
        # 操作在无显示环境里也能正常运行。
        from marsoom import imgui

        imgui.set_next_window_size_constraints((560, 360), (900, 700))
        imgui.begin("SUPER Playback")
        imgui.text(
            f"Frame: {self.current_frame_index + 1}/{len(self.playback_timestamps)}"
        )
        imgui.text(
            f"{self.playback_camera_name} timestamp: {self.current_timestep:.6f}s"
        )
        imgui.text(f"Playing: {self.playing}")
        _, self.fps = imgui.slider_int("Playback FPS", self.fps, 1, 120)
        sim_reconstruction_mode = bool(
            getattr(
                self.environment,
                "super_sim_reconstruction_mode",
                False,
            )
        )
        if not sim_reconstruction_mode:
            self._draw_manual_pose_panel(imgui)
        paper_mode = getattr(
            self.environment, "super_tissue_mode", None
        ) in {"paper_pbd", "paper_soft"}
        if sim_reconstruction_mode:
            imgui.text(
                "Simulation reconstruction: old SUPER PSM permanently hidden"
            )
            imgui.text(
                "Instrument: official dataset PSM distal links (recorded poses)"
            )
            grasp_count = (
                0
                if self._sim_grasp_particle_ids is None
                else len(self._sim_grasp_particle_ids)
            )
            support_count = (
                0
                if self._sim_grasp_support_particle_ids is None
                else len(self._sim_grasp_support_particle_ids)
            )
            imgui.text(
                "Red-marker trajectory boundary: "
                f"{'ACTIVE' if self._sim_grasp_active else 'open'} | "
                f"direct/natural-neighbor nodes {grasp_count}/{support_count} | "
                "final error / pre-projection correction "
                f"{self._sim_grasp_tracking_error_m * 1e3:.6f} / "
                f"{self._sim_grasp_projection_correction_m * 1e3:.3f} mm"
            )
            imgui.text(
                "Closed jaw: only five nodes are prescribed; both neighbor rings "
                "are free PBD + RGB residual + online stiffness nodes"
            )
        else:
            collision_changed, collision_enabled = imgui.checkbox(
                "PSM-tissue collision",
                self.psm_tissue_collisions_enabled,
            )
            if collision_changed:
                self.psm_tissue_collisions_enabled = set_psm_tissue_collisions(
                    self.environment, collision_enabled
                )
        current_sim_time = self.environment.time()
        if (
            current_sim_time - self._last_contact_ui_refresh_time >= 0.25
        ):
            if (
                self.environment.sim.triangle_skin_contact_projector
                is not None
            ):
                self._contact_ui_metrics = (
                    self.environment.sim.triangle_skin_contact_metrics()
                )
            else:
                self._contact_ui_metrics = None
            soft_handle = getattr(
                self.environment, "super_tissue_soft_handle", None
            )
            if soft_handle is not None:
                current_positions = wp.to_torch(
                    self.environment.sim.state_0.particle_q
                )[soft_handle.particle_start : soft_handle.particle_end]
                rest_positions = self._tissue_rest_positions[
                    soft_handle.particle_start : soft_handle.particle_end
                ]
                self._tissue_max_displacement_m = float(
                    torch.linalg.vector_norm(
                        current_positions - rest_positions, dim=1
                    )
                    .max()
                    .item()
                )
            self._last_contact_ui_refresh_time = current_sim_time
        if not sim_reconstruction_mode:
            imgui.text(
                "Contact status: "
                f"{'ON' if self.psm_tissue_collisions_enabled else 'OFF'}"
            )
        self._draw_material_panel(imgui, paper_mode)
        if not sim_reconstruction_mode:
            self._draw_contact_runtime_panel(imgui)
        self._draw_visual_feedback_panel(imgui)
        if self._visual_residual_busy:
            imgui.text("Visual residual: solving current stereo frame...")
        if imgui.button("Play"):
            if self.current_frame_index >= len(self.playback_timestamps) - 1:
                self.go_to_frame(0)
            self._visual_force_step = self.visual_force_update_interval - 1
            self._previous_visual_residual = None
            self.playing = True
        imgui.same_line()
        if imgui.button("Pause"):
            self.playing = False
        imgui.same_line()
        if imgui.button("Reset"):
            self.reset()
        imgui.end()

    def _current_tool_height_m(self) -> float | None:
        body_ids_by_environment = getattr(
            self.environment, "super_psm_lnd_body_ids", None
        )
        if not body_ids_by_environment:
            return None
        body_ids = tuple(int(value) for value in body_ids_by_environment[0])
        if not body_ids:
            return None
        body_q = wp.to_torch(self.environment.sim.state_0.body_q)
        selected = body_q[
            torch.as_tensor(body_ids, device=body_q.device, dtype=torch.long)
        ]
        return float(selected[:, 2].mean().item())

    def _update_action_phase(self, contact_metrics: dict | None = None) -> str:
        if contact_metrics is None:
            contact_metrics = (
                self.environment.sim.triangle_skin_contact_metrics()
            )
        self._current_action_phase = self._action_phase_classifier.classify(
            contact_metrics,
            self._current_tool_height_m(),
        )
        return self._current_action_phase

    @staticmethod
    def _selected_contact_metrics(
        contact_metrics: dict | None,
    ) -> dict[str, object]:
        if contact_metrics is None:
            return {}
        keys = (
            "contact_count",
            "top_barrier_contact_count",
            "minimum_signed_distance_m",
            "maximum_penetration_m",
            "material_safety_step_scale",
            "material_safety_unsafe_tetrahedra",
            "contact_local_unsafe_tetrahedra",
            "persistent_grip_active",
            "persistent_grip_activation_counter",
            "persistent_grip_particle_count",
            "persistent_grip_direct_particle_count",
            "persistent_grip_support_particle_count",
            "persistent_grip_anchor_error_rms_m",
            "persistent_grip_anchor_error_maximum_m",
            "prescribed_boundary_preprojection_correction_m",
            "natural_pbd_neighbor_particle_count",
            "natural_pbd_neighbor_displacement_maximum_m",
            "persistent_grip_q7_angle_rad",
            "persistent_grip_q7_motion_state",
            "persistent_grip_inversion_safe_scale",
            "persistent_grip_direct_safe_scale",
        )
        return {
            key: contact_metrics[key]
            for key in keys
            if key in contact_metrics
        }

    def _current_physical_evaluation_metrics(
        self,
        contact_metrics: dict | None = None,
        visual_metrics: dict | None = None,
    ) -> dict[str, object]:
        physical: dict[str, object] = {}
        mapper = self.visual_residual_mapper
        if mapper is not None:
            positions = wp.to_torch(
                self.environment.sim.state_0.particle_q
            )
            physical.update(mapper.physical_quality_metrics(positions))
        physical.update(self._selected_contact_metrics(contact_metrics))
        if visual_metrics is not None:
            for key in (
                "dynamically_excluded_particles",
                "locally_frozen_particles",
                "local_volume_projection_passes",
                "backtrack_count",
                "initial_minimum_volume_ratio",
                "minimum_volume_ratio",
                "newly_inverted_tetrahedra",
            ):
                if key in visual_metrics:
                    physical[f"residual_{key}"] = visual_metrics[key]
        return physical

    def _current_material_evaluation_metrics(
        self,
        stiffness_metrics: dict | None = None,
    ) -> dict[str, object]:
        updater = self.stiffness_updater
        if updater is None:
            return {}
        metrics = dict(stiffness_metrics or updater.last_metrics or {})
        settings = updater.settings
        metrics.update(
            configured_distance_minimum=settings.distance_minimum,
            configured_distance_maximum=settings.distance_maximum,
            configured_shape_minimum=settings.shape_minimum,
            configured_shape_maximum=settings.shape_maximum,
            configured_log_learning_rate=settings.log_learning_rate,
            configured_maximum_log_step=settings.maximum_log_step,
            configured_ema_new_weight=1.0 - settings.signal_ema_decay,
            configured_spatial_smoothing_iterations=(
                settings.spatial_smoothing_iterations
            ),
            configured_spatial_smoothing_blend=(
                settings.spatial_smoothing_blend
            ),
            global_velocity_damping_per_second=float(
                updater.global_velocity_damping_per_second
            ),
            global_coupling_gain=float(updater.global_coupling_gain),
        )
        return metrics

    def _record_visual_stiffness_metrics(
        self,
        *,
        result,
        visual_metrics: dict,
        stiffness_metrics: dict | None,
        contact_metrics: dict | None,
        gate_paused: bool,
        gate_reason: str,
    ) -> None:
        recorder = self.stiffness_metrics_recorder
        if recorder is None:
            return
        image = {
            "accepted": bool(visual_metrics["accepted"]),
            "rejection_reason": visual_metrics["rejection_reason"],
            "left_right_loss_before": visual_metrics[
                "initial_camera_visual_losses"
            ],
            "left_right_loss_after": visual_metrics[
                "exact_final_camera_visual_losses"
            ],
            "mean_loss_before": visual_metrics["initial_visual_loss"],
            "mean_loss_after": visual_metrics["exact_final_visual_loss"],
            "loss_reduction_fraction": visual_metrics[
                "visual_loss_reduction_fraction"
            ],
            "residual_maximum_m": result.maximum_residual_m,
            "residual_rms_m": result.rms_residual_m,
            "active_pixel_counts": visual_metrics[
                "camera_active_pixel_counts"
            ],
            "mask_coverage_fractions": visual_metrics[
                "camera_mask_coverage_fractions"
            ],
            "solve_elapsed_s": visual_metrics.get("solve_elapsed_s", 0.0),
        }
        recorder.record(
            event="visual_update",
            frame_index=self.current_frame_index,
            timestamp_s=self.current_timestep,
            phase=self._current_action_phase,
            image=image,
            physical=self._current_physical_evaluation_metrics(
                contact_metrics, visual_metrics
            ),
            material=self._current_material_evaluation_metrics(
                stiffness_metrics
            ),
            details={
                "epoch": self._stiffness_evaluation_epoch,
                "stiffness_gate_paused": gate_paused,
                "stiffness_gate_reason": gate_reason,
            },
        )

    def _record_trajectory_observation(
        self,
        *,
        alignment,
        contact_metrics: dict | None,
    ) -> None:
        """Record the same pre-feedback image loss for every A/B/C mode.

        This observation is evaluated after the physics step and before the
        current frame's residual is applied.  It is therefore a fair
        prediction error for fixed PBD, residual-only and online-stiffness
        runs; post-residual fitting loss remains in the separate visual_update
        event.
        """
        recorder = self.stiffness_metrics_recorder
        if recorder is None:
            return
        recorder.record(
            event="trajectory_observation",
            frame_index=self.current_frame_index,
            timestamp_s=self.current_timestep,
            phase=self._current_action_phase,
            image={
                "prediction_loss": alignment.loss,
                "left_right_prediction_losses": alignment.camera_losses,
                "camera_weight_sums": alignment.camera_weight_sums,
                "active_pixel_counts": (
                    alignment.camera_active_pixel_counts
                ),
                "mask_coverage_fractions": (
                    alignment.camera_mask_coverage_fractions
                ),
            },
            physical=self._current_physical_evaluation_metrics(
                contact_metrics
            ),
            material=self._current_material_evaluation_metrics(),
            details={
                "epoch": self._stiffness_evaluation_epoch,
                "pre_feedback": True,
            },
        )

    def _record_incomplete_stiffness_evaluations(self, reason: str) -> None:
        recorder = getattr(self, "stiffness_metrics_recorder", None)
        if recorder is None:
            return
        for evaluation in getattr(
            self, "_active_stiffness_evaluations", ()
        ):
            recorder.record(
                event="open_loop_incomplete",
                frame_index=self.current_frame_index,
                timestamp_s=self.current_timestep,
                phase=self._current_action_phase,
                prediction={
                    "evaluation_id": evaluation.evaluation_id,
                    "start_frame_index": evaluation.start_frame_index,
                    "remaining_horizons": sorted(
                        evaluation.pending_horizons
                    ),
                },
                details={"reason": reason},
                force_summary=True,
            )

    def close(self) -> None:
        self._set_sim_grasp_boundary_kinematic(False)
        self._record_incomplete_stiffness_evaluations("runtime_closed")
        self._active_stiffness_evaluations.clear()
        if self.stiffness_metrics_recorder is not None:
            self.stiffness_metrics_recorder.close()

    def _current_stiffness_tool_command(self) -> StiffnessToolCommand:
        return StiffnessToolCommand(
            frame_index=int(self.current_frame_index),
            timestep=float(self.current_timestep),
            state_index=int(self._last_state_index),
            phase=self._current_action_phase,
        )

    def _apply_stiffness_tool_command(
        self, command: StiffnessToolCommand
    ) -> None:
        self.current_frame_index = int(command.frame_index)
        self.current_timestep = float(command.timestep)
        self._last_state_index = int(command.state_index)
        self.apply_current_psm_pose()
        if self._sim_reconstruction_mode:
            self._update_sim_grasp_target()

    def _stiffness_global_gate(
        self, contact_metrics: dict | None
    ) -> tuple[bool, str, float, bool]:
        if contact_metrics is None:
            return True, "missing_contact_metrics", float("inf"), False
        jaw_angle = float(
            contact_metrics.get("persistent_grip_q7_angle_rad", 0.0)
        )
        timestamp = float(
            contact_metrics.get("persistent_grip_q7_timestamp_s", 0.0)
        )
        jaw_speed = 0.0
        if (
            self._last_stiffness_gate_timestep is not None
            and timestamp > self._last_stiffness_gate_timestep
            and self._last_stiffness_gate_jaw_angle is not None
        ):
            jaw_speed = abs(
                jaw_angle - self._last_stiffness_gate_jaw_angle
            ) / (timestamp - self._last_stiffness_gate_timestep)
        grip_active = bool(
            contact_metrics.get("persistent_grip_active", False)
        )
        transition_reasons: list[str] = []
        if grip_active != self._last_stiffness_grip_active:
            transition_reasons.append("capture_or_release")
        if (
            STIFFNESS_MAXIMUM_JAW_SPEED_RAD_S is not None
            and jaw_speed > STIFFNESS_MAXIMUM_JAW_SPEED_RAD_S
        ):
            transition_reasons.append("rapid_q7")
        if (
            float(contact_metrics.get("maximum_penetration_m", 0.0))
            > STIFFNESS_MAXIMUM_PENETRATION_M
        ):
            transition_reasons.append("excessive_penetration")
        contact_count = int(contact_metrics.get("contact_count", 0))
        previous_contact_count = self._last_stiffness_contact_count
        if (
            previous_contact_count is not None
            and min(contact_count, previous_contact_count) > 0
            and abs(contact_count - previous_contact_count) >= 8
            and max(contact_count, previous_contact_count)
            > 2 * min(contact_count, previous_contact_count)
        ):
            transition_reasons.append("contact_patch_switch")
        if transition_reasons:
            self._stiffness_transition_cooldown = max(
                self._stiffness_transition_cooldown,
                STIFFNESS_TRANSITION_COOLDOWN_UPDATES,
            )
            if (
                self.stiffness_updater is not None
                and (
                    "capture_or_release" in transition_reasons
                    or "contact_patch_switch" in transition_reasons
                    or "excessive_penetration" in transition_reasons
                )
            ):
                self.stiffness_updater.invalidate_signal_history()
                self._warp_gradient_transitions.clear()
        paused = self._stiffness_transition_cooldown > 0
        reason = (
            "+".join(transition_reasons)
            if transition_reasons
            else ("transition_cooldown" if paused else "")
        )
        if self._stiffness_transition_cooldown > 0:
            self._stiffness_transition_cooldown -= 1
        self._last_stiffness_gate_timestep = timestamp
        self._last_stiffness_gate_jaw_angle = jaw_angle
        self._last_stiffness_grip_active = grip_active
        self._last_stiffness_contact_count = contact_count
        return paused, reason, jaw_speed, grip_active

    def _mass_weighted_particle_rms(
        self, reference: torch.Tensor, current: torch.Tensor
    ) -> float:
        inverse_mass = wp.to_torch(
            self.environment.sim.model.particle_inv_mass
        ).detach()
        reference = reference.to(device=inverse_mass.device)
        current = current.to(device=inverse_mass.device)
        dynamic = inverse_mass > 0.0
        if not bool(dynamic.any().item()):
            return 0.0
        mass = 1.0 / inverse_mass[dynamic]
        squared = torch.sum((current[dynamic] - reference[dynamic]) ** 2, dim=1)
        return float(torch.sqrt(torch.sum(mass * squared) / mass.sum()).item())

    def _run_history_relaxation(
        self,
        snapshot: StiffnessHistorySnapshot,
        candidate: PaperStiffnessCandidate | None,
        verified_distance: torch.Tensor,
        verified_shape: torch.Tensor,
    ) -> float:
        sim = self.environment.sim
        sim.copy_embodied_gaussian_rollout_state(snapshot.rollout_state)
        particle_qd = wp.to_torch(sim.state_0.particle_qd)
        with torch.no_grad():
            particle_qd.zero_()
        assert self.stiffness_updater is not None
        # A historical snapshot carries the stiffness that was verified when
        # it was captured.  Both sides of today's regression test must instead
        # start from today's verified material; otherwise the comparison mixes
        # material age with the candidate's effect.
        self.stiffness_updater.restore_verified_stiffness(
            verified_distance, verified_shape
        )
        if candidate is not None:
            self.stiffness_updater.install_candidate_for_rollout(candidate)
        projector = sim.triangle_skin_contact_projector
        if projector is not None:
            projector.freeze_persistent_grip_state_machine()
        sim.sync_kinematic_body_interpolation()
        self.environment.step(compute_visual_forces=False)
        if projector is not None:
            projector.freeze_persistent_grip_state_machine()
        current = wp.to_torch(sim.state_0.particle_q).detach()
        return self._mass_weighted_particle_rms(
            snapshot.accepted_positions, current
        )

    def _evaluate_stiffness_history(
        self, candidate: PaperStiffnessCandidate
    ) -> tuple[float, float, bool]:
        if not self._stiffness_history:
            return 0.0, 0.0, True
        sim = self.environment.sim
        live = sim.clone_embodied_gaussian_rollout_state()
        assert self.stiffness_updater is not None
        verified_distance = (
            self.stiffness_updater.distance_stiffness.detach().clone()
        )
        verified_shape = (
            self.stiffness_updater.shape_stiffness.detach().clone()
        )
        baseline_values: list[float] = []
        candidate_values: list[float] = []
        try:
            for snapshot in self._stiffness_history:
                baseline_values.append(
                    self._run_history_relaxation(
                        snapshot,
                        None,
                        verified_distance,
                        verified_shape,
                    )
                )
                candidate_values.append(
                    self._run_history_relaxation(
                        snapshot,
                        candidate,
                        verified_distance,
                        verified_shape,
                    )
                )
        finally:
            sim.copy_embodied_gaussian_rollout_state(live)
            sim.update_gaussian_transforms()
        baseline = float(np.mean(baseline_values))
        candidate_loss = float(np.mean(candidate_values))
        allowed = (
            baseline * (1.0 + STIFFNESS_HISTORY_RELATIVE_TOLERANCE)
            + STIFFNESS_HISTORY_ABSOLUTE_TOLERANCE_M
        )
        return baseline, candidate_loss, candidate_loss <= allowed

    def _maybe_store_stiffness_history(
        self,
        *,
        accepted_positions: torch.Tensor,
        jaw_speed_rad_s: float,
        gate_paused: bool,
    ) -> None:
        if gate_paused or jaw_speed_rad_s > STIFFNESS_HISTORY_MAXIMUM_JAW_SPEED_RAD_S:
            return
        speeds = torch.linalg.vector_norm(
            wp.to_torch(self.environment.sim.state_0.particle_qd), dim=1
        )
        if float(speeds.max().item()) > STIFFNESS_HISTORY_MAXIMUM_PARTICLE_SPEED_M_S:
            return
        self._stiffness_history.append(
            StiffnessHistorySnapshot(
                rollout_state=(
                    self.environment.sim.clone_embodied_gaussian_rollout_state()
                ),
                accepted_positions=accepted_positions.detach().clone(),
                frame_index=int(self.current_frame_index),
            )
        )

    def _run_stiffness_rollout_shadow(
        self,
        *,
        rollout_state: object,
        commands: list[StiffnessToolCommand],
        candidate: PaperStiffnessCandidate,
        use_candidate: bool,
        freeze_grip_state_machine: bool,
        measure_open_loop_residual: bool = False,
        replay_visual_residuals: bool = False,
        initial_previous_residual: torch.Tensor | None = None,
        replay_after_frame_index: int | None = None,
    ) -> dict:
        assert self.stiffness_updater is not None
        assert self.visual_residual_mapper is not None
        sim = self.environment.sim
        sim.copy_embodied_gaussian_rollout_state(rollout_state)
        if use_candidate:
            self.stiffness_updater.install_candidate_for_rollout(
                candidate
            )
        projector = sim.triangle_skin_contact_projector
        target_frame_index = (
            commands[-1].frame_index if commands else self.current_frame_index
        )
        previous_residual = (
            None
            if initial_previous_residual is None
            else initial_previous_residual.detach().clone()
        )
        # The experiment's primary score is the prediction loss at every
        # observed frame, not only the final frame of a validation horizon.
        # Keep the pre-residual losses along a replayed branch so admission can
        # minimize the same cumulative quantity that the A/B report measures.
        trajectory_visual_losses: list[float] = []
        trajectory_camera_losses: list[tuple[float, ...]] = []
        for command_index, command in enumerate(commands):
            self._apply_stiffness_tool_command(command)
            if projector is not None and freeze_grip_state_machine:
                projector.freeze_persistent_grip_state_machine()
            if self._sim_reconstruction_mode:
                self._project_sim_grasp_boundary(update_gaussians=False)
            self.environment.step(compute_visual_forces=False)
            if self._sim_reconstruction_mode:
                self._project_sim_grasp_boundary(update_gaussians=True)
            self.apply_current_psm_pose()
            if projector is not None and freeze_grip_state_machine:
                projector.freeze_persistent_grip_state_machine()
            next_frame_index = (
                commands[command_index + 1].frame_index
                if command_index + 1 < len(commands)
                else None
            )
            frame_complete = next_frame_index != command.frame_index
            if (
                replay_visual_residuals
                and frame_complete
                and command.frame_index < target_frame_index
                and (
                    replay_after_frame_index is None
                    or command.frame_index > replay_after_frame_index
                )
            ):
                self.dataset_manager.update_frames(command.timestep)
                sim.update_gaussian_transforms()
                control_exclusion_mask = grip_control_exclusion_mask(
                    self.visual_residual_mapper,
                    self.environment,
                )
                residual = sim.solve_visual_tissue_residual(
                    self.visual_residual_mapper,
                    self.environment.frames,
                    previous_residual=previous_residual,
                    dynamic_exclusion_mask=control_exclusion_mask,
                    observations_are_bgr=True,
                )
                trajectory_visual_losses.append(
                    float(residual.initial_visual_loss)
                )
                trajectory_camera_losses.append(
                    tuple(residual.initial_camera_visual_losses)
                )
                accepted = sim.apply_visual_tissue_residual(
                    residual,
                    mapper=self.visual_residual_mapper,
                    frames=self.environment.frames,
                    observations_are_bgr=True,
                )
                previous_residual = (
                    residual.residual.detach().clone() if accepted else None
                )
        if commands:
            self.dataset_manager.update_frames(commands[-1].timestep)
        sim.update_gaussian_transforms()
        assert self.environment.frames is not None
        alignment = sim.evaluate_visual_tissue_alignment(
            self.visual_residual_mapper,
            self.environment.frames,
            observations_are_bgr=True,
        )
        if replay_visual_residuals:
            trajectory_visual_losses.append(float(alignment.loss))
            trajectory_camera_losses.append(tuple(alignment.camera_losses))
        trajectory_visual_loss = (
            float(np.mean(trajectory_visual_losses))
            if trajectory_visual_losses
            else float(alignment.loss)
        )
        trajectory_camera_loss = (
            tuple(
                float(np.mean(values))
                for values in zip(*trajectory_camera_losses)
            )
            if trajectory_camera_losses
            else tuple(alignment.camera_losses)
        )
        physical = self.visual_residual_mapper.physical_quality_metrics(
            wp.to_torch(sim.state_0.particle_q)
        )
        contact = (
            self._sim_grasp_contact_metrics()
            if self._sim_reconstruction_mode
            else sim.triangle_skin_contact_metrics() or {}
        )
        open_loop_residual = None
        if measure_open_loop_residual:
            control_exclusion_mask = grip_control_exclusion_mask(
                self.visual_residual_mapper,
                self.environment,
            )
            open_loop_residual = sim.solve_visual_tissue_residual(
                self.visual_residual_mapper,
                self.environment.frames,
                previous_residual=None,
                dynamic_exclusion_mask=control_exclusion_mask,
                observations_are_bgr=True,
            )
        particle_positions = (
            wp.to_torch(sim.state_0.particle_q).detach().clone()
        )
        final_rollout_state = sim.clone_embodied_gaussian_rollout_state()
        return {
            "visual_loss": alignment.loss,
            "camera_losses": alignment.camera_losses,
            "trajectory_visual_loss": trajectory_visual_loss,
            "trajectory_camera_losses": trajectory_camera_loss,
            "camera_weight_sums": alignment.camera_weight_sums,
            "camera_active_pixel_counts": (
                alignment.camera_active_pixel_counts
            ),
            "camera_mask_coverage_fractions": (
                alignment.camera_mask_coverage_fractions
            ),
            **physical,
            "maximum_penetration_m": float(
                contact.get("maximum_penetration_m", 0.0)
            ),
            "anchor_error_rms_m": float(
                contact.get("persistent_grip_anchor_error_rms_m", 0.0)
            ),
            "anchor_error_maximum_m": float(
                contact.get(
                    "persistent_grip_anchor_error_maximum_m", 0.0
                )
            ),
            "grip_active": bool(
                contact.get("persistent_grip_active", False)
            ),
            "open_loop_residual_rms_m": (
                None
                if open_loop_residual is None
                else open_loop_residual.rms_residual_m
            ),
            "open_loop_residual_maximum_m": (
                None
                if open_loop_residual is None
                else open_loop_residual.maximum_residual_m
            ),
            "particle_positions": particle_positions,
            "rollout_state": final_rollout_state,
            "rollout_previous_residual": (
                None
                if previous_residual is None
                else previous_residual.detach().clone()
            ),
        }

    def _run_stiffness_prediction_shadow(
        self,
        pending: PendingStiffnessValidation,
        *,
        use_candidate: bool,
    ) -> dict:
        result = self._run_stiffness_rollout_shadow(
            rollout_state=pending.rollout_state,
            commands=pending.commands,
            candidate=pending.candidate,
            use_candidate=use_candidate,
            # Pre-commit evidence must include the actual capture/release
            # state machine. Freezing it made H=1 look better than the live
            # contact-coupled trajectory.
            freeze_grip_state_machine=False,
            replay_visual_residuals=True,
            initial_previous_residual=pending.previous_residual,
            # The proposal frame's residual has already been applied to the
            # saved rollout. Replaying it would count the same correction
            # twice and make the shadow trajectory unlike the live one.
            replay_after_frame_index=pending.frame_index,
        )
        # Admission is based on the mean pre-residual prediction loss over
        # frames 1..H. Endpoint values remain available for diagnostics.
        result["endpoint_visual_loss"] = result["visual_loss"]
        result["endpoint_camera_losses"] = result["camera_losses"]
        result["visual_loss"] = result["trajectory_visual_loss"]
        result["camera_losses"] = result["trajectory_camera_losses"]
        return result

    @staticmethod
    def _stiffness_field_summary(
        values: torch.Tensor | None,
    ) -> dict[str, float]:
        if values is None or values.numel() == 0:
            return {}
        values = values.detach().float()
        return {
            "minimum": float(values.min().item()),
            "median": float(values.median().item()),
            "maximum": float(values.max().item()),
            "mean": float(values.mean().item()),
        }

    def _start_committed_stiffness_evaluation(
        self,
        pending: PendingStiffnessValidation,
    ) -> None:
        recorder = self.stiffness_metrics_recorder
        if recorder is None:
            return
        evaluation = CommittedStiffnessEvaluation(
            evaluation_id=self._next_stiffness_evaluation_id,
            candidate=pending.candidate,
            rollout_state=pending.rollout_state,
            start_frame_index=pending.frame_index,
            start_phase=(
                pending.commands[0].phase
                if pending.commands
                else self._current_action_phase
            ),
            commands=list(pending.commands),
            pending_horizons=set(self.stiffness_evaluation_horizons),
            previous_residual=(
                None
                if pending.previous_residual is None
                else pending.previous_residual.detach().clone()
            ),
        )
        self._next_stiffness_evaluation_id += 1
        self._active_stiffness_evaluations.append(evaluation)
        recorder.record(
            event="open_loop_started",
            frame_index=self.current_frame_index,
            timestamp_s=self.current_timestep,
            phase=self._current_action_phase,
            material={
                "evaluation_id": evaluation.evaluation_id,
                "candidate_metrics": pending.candidate.metrics,
            },
            prediction={
                "start_frame_index": pending.frame_index,
                "horizons": sorted(evaluation.pending_horizons),
                "protocols": ("material_isolation", "end_to_end"),
            },
            details={"epoch": self._stiffness_evaluation_epoch},
            force_summary=True,
        )

    @staticmethod
    def _serializable_shadow_metrics(result: dict) -> dict:
        return {
            key: value
            for key, value in result.items()
            if key
            not in {
                "particle_positions",
                "rollout_state",
                "rollout_previous_residual",
            }
        }

    def _evaluation_phase(
        self,
        evaluation: CommittedStiffnessEvaluation,
        commands: list[StiffnessToolCommand],
    ) -> str:
        if commands:
            return commands[-1].phase
        return evaluation.start_phase

    def _record_unavailable_stiffness_horizon(
        self,
        evaluation: CommittedStiffnessEvaluation,
        horizon: int,
        reason: str,
    ) -> None:
        recorder = self.stiffness_metrics_recorder
        if recorder is None:
            return
        recorder.record(
            event="open_loop_unavailable",
            frame_index=self.current_frame_index,
            timestamp_s=self.current_timestep,
            phase=self._current_action_phase,
            prediction={
                "evaluation_id": evaluation.evaluation_id,
                "start_frame_index": evaluation.start_frame_index,
                "horizon_frames": horizon,
            },
            details={"reason": reason},
            force_summary=True,
        )

    def _evaluate_committed_stiffness_horizon(
        self,
        evaluation: CommittedStiffnessEvaluation,
        horizon: int,
    ) -> None:
        recorder = self.stiffness_metrics_recorder
        if recorder is None:
            return
        target_frame_index = evaluation.start_frame_index + horizon
        if target_frame_index >= len(self.playback_timestamps):
            self._record_unavailable_stiffness_horizon(
                evaluation, horizon, "target_after_dataset_end"
            )
            return
        commands = [
            command
            for command in evaluation.commands
            if command.frame_index <= target_frame_index
        ]
        if not commands:
            self._record_unavailable_stiffness_horizon(
                evaluation, horizon, "no_recorded_tool_commands"
            )
            return

        sim = self.environment.sim
        live = sim.clone_embodied_gaussian_rollout_state()
        controller_state = (
            self.current_frame_index,
            self.current_timestep,
            self._last_state_index,
            self._last_q_full.detach().clone(),
        )
        observation_timestep = self.current_timestep
        target_timestep = float(
            self.playback_timestamps[target_frame_index]
        )
        reference_positions = wp.to_torch(
            evaluation.rollout_state.embodied_state.physics_state.particle_q
        ).detach()
        old_distance = (
            evaluation.rollout_state.auxiliary_state
            .paper_distance_stiffness
        )
        old_shape = (
            evaluation.rollout_state.auxiliary_state
            .paper_shape_stiffness
        )
        material_summary = {
            "evaluation_id": evaluation.evaluation_id,
            "old_distance": self._stiffness_field_summary(old_distance),
            "new_distance": self._stiffness_field_summary(
                evaluation.candidate.distance_stiffness
            ),
            "old_shape": self._stiffness_field_summary(old_shape),
            "new_shape": self._stiffness_field_summary(
                evaluation.candidate.shape_stiffness
            ),
        }
        if old_distance is not None:
            new_distance = evaluation.candidate.distance_stiffness.to(
                device=old_distance.device, dtype=old_distance.dtype
            )
            material_summary["distance_change_rms"] = float(
                torch.sqrt(
                    torch.mean((new_distance - old_distance) ** 2)
                ).item()
            )
        if old_shape is not None:
            new_shape = evaluation.candidate.shape_stiffness.to(
                device=old_shape.device, dtype=old_shape.dtype
            )
            material_summary["shape_change_rms"] = float(
                torch.sqrt(
                    torch.mean((new_shape - old_shape) ** 2)
                ).item()
            )
        phase = self._evaluation_phase(evaluation, commands)
        try:
            self.dataset_manager.update_frames(target_timestep)
            for protocol, freeze_grip in (
                ("material_isolation", True),
                ("end_to_end", False),
            ):
                baseline = self._run_stiffness_rollout_shadow(
                    rollout_state=evaluation.rollout_state,
                    commands=commands,
                    candidate=evaluation.candidate,
                    use_candidate=False,
                    freeze_grip_state_machine=freeze_grip,
                    measure_open_loop_residual=True,
                    replay_visual_residuals=not freeze_grip,
                    initial_previous_residual=evaluation.previous_residual,
                    replay_after_frame_index=evaluation.start_frame_index,
                )
                candidate = self._run_stiffness_rollout_shadow(
                    rollout_state=evaluation.rollout_state,
                    commands=commands,
                    candidate=evaluation.candidate,
                    use_candidate=True,
                    freeze_grip_state_machine=freeze_grip,
                    measure_open_loop_residual=True,
                    replay_visual_residuals=not freeze_grip,
                    initial_previous_residual=evaluation.previous_residual,
                    replay_after_frame_index=evaluation.start_frame_index,
                )
                baseline_positions = baseline["particle_positions"]
                candidate_positions = candidate["particle_positions"]
                prediction = {
                    "evaluation_id": evaluation.evaluation_id,
                    "protocol": protocol,
                    "start_frame_index": evaluation.start_frame_index,
                    "target_frame_index": target_frame_index,
                    "horizon_frames": horizon,
                    "rollout_steps": len(commands),
                    "baseline_gap": baseline["visual_loss"],
                    "candidate_gap": candidate["visual_loss"],
                    "gap_improvement": (
                        baseline["visual_loss"]
                        - candidate["visual_loss"]
                    ),
                    "baseline_camera_gaps": baseline["camera_losses"],
                    "candidate_camera_gaps": candidate["camera_losses"],
                    "baseline_displacement_rms_m": (
                        self._mass_weighted_particle_rms(
                            reference_positions, baseline_positions
                        )
                    ),
                    "candidate_displacement_rms_m": (
                        self._mass_weighted_particle_rms(
                            reference_positions, candidate_positions
                        )
                    ),
                    "baseline_candidate_particle_rms_m": (
                        self._mass_weighted_particle_rms(
                            baseline_positions, candidate_positions
                        )
                    ),
                    "baseline_open_loop_residual_rms_m": baseline[
                        "open_loop_residual_rms_m"
                    ],
                    "candidate_open_loop_residual_rms_m": candidate[
                        "open_loop_residual_rms_m"
                    ],
                }
                recorder.record(
                    event="open_loop_prediction",
                    frame_index=target_frame_index,
                    timestamp_s=target_timestep,
                    phase=phase,
                    image={
                        "baseline_gap": baseline["visual_loss"],
                        "candidate_gap": candidate["visual_loss"],
                        "baseline_camera_gaps": baseline[
                            "camera_losses"
                        ],
                        "candidate_camera_gaps": candidate[
                            "camera_losses"
                        ],
                    },
                    physical={
                        "baseline": self._serializable_shadow_metrics(
                            baseline
                        ),
                        "candidate": self._serializable_shadow_metrics(
                            candidate
                        ),
                    },
                    material=material_summary,
                    prediction=prediction,
                    details={"epoch": self._stiffness_evaluation_epoch},
                    force_summary=True,
                )
        finally:
            sim.copy_embodied_gaussian_rollout_state(live)
            (
                self.current_frame_index,
                self.current_timestep,
                self._last_state_index,
                self._last_q_full,
            ) = controller_state
            self.dataset_manager.update_frames(observation_timestep)
            sim.update_gaussian_transforms()

    def _advance_committed_stiffness_evaluations(self) -> None:
        if self.stiffness_metrics_recorder is None:
            return
        completed: list[CommittedStiffnessEvaluation] = []
        for evaluation in list(self._active_stiffness_evaluations):
            for horizon in sorted(evaluation.pending_horizons):
                target = evaluation.start_frame_index + horizon
                if target >= len(self.playback_timestamps):
                    self._evaluate_committed_stiffness_horizon(
                        evaluation, horizon
                    )
                    evaluation.pending_horizons.remove(horizon)
                elif target <= self.current_frame_index:
                    self._evaluate_committed_stiffness_horizon(
                        evaluation, horizon
                    )
                    evaluation.pending_horizons.remove(horizon)
            if not evaluation.pending_horizons:
                completed.append(evaluation)
        for evaluation in completed:
            self._active_stiffness_evaluations.remove(evaluation)
            self.stiffness_metrics_recorder.record(
                event="open_loop_completed",
                frame_index=self.current_frame_index,
                timestamp_s=self.current_timestep,
                phase=self._current_action_phase,
                prediction={"evaluation_id": evaluation.evaluation_id},
                force_summary=True,
            )

    def _record_terminal_stiffness_validation(
        self,
        pending: PendingStiffnessValidation,
        metrics: dict,
        reason: str,
    ) -> None:
        recorder = self.stiffness_metrics_recorder
        if recorder is None:
            return
        recorder.record(
            event="stiffness_validation",
            frame_index=self.current_frame_index,
            timestamp_s=self.current_timestep,
            phase=self._current_action_phase,
            physical=self._current_physical_evaluation_metrics(),
            material=self._current_material_evaluation_metrics(metrics),
            prediction={
                "start_frame_index": pending.frame_index,
                "horizon_frames": (
                    self.current_frame_index - pending.frame_index
                ),
                "rollout_steps": len(pending.commands),
                "status": metrics.get("validation_status", metrics["status"]),
            },
            details={"reason": reason},
            force_summary=True,
        )

    def _scaled_stiffness_candidate(
        self,
        candidate: PaperStiffnessCandidate,
        scale: float,
    ) -> PaperStiffnessCandidate:
        """Line-search one bounded step without changing verified material."""
        if scale <= 0.0:
            raise ValueError("Stiffness candidate scale must be positive")
        if abs(scale - 1.0) <= 1.0e-12:
            return candidate
        assert self.stiffness_updater is not None
        updater = self.stiffness_updater
        settings = updater.settings
        log_step = torch.clamp(
            candidate.log_step * float(scale),
            min=-settings.maximum_log_step,
            max=settings.maximum_log_step,
        )
        with torch.no_grad():
            distance = torch.clamp(
                updater.distance_stiffness.detach() * torch.exp(log_step),
                settings.distance_minimum,
                settings.distance_maximum,
            )
            shape = torch.clamp(
                updater.shape_stiffness.detach()
                * torch.exp(log_step * settings.shape_update_gain),
                settings.shape_minimum,
                settings.shape_maximum,
            )
        metrics = dict(candidate.metrics)
        metrics.update(
            line_search_scale=float(scale),
            maximum_log_step=float(log_step.abs().max().item()),
            mean_absolute_log_step=float(log_step.abs().mean().item()),
            distance_minimum=float(distance.min().item()),
            distance_median=float(distance.median().item()),
            distance_maximum=float(distance.max().item()),
            shape_minimum=float(shape.min().item()),
            shape_median=float(shape.median().item()),
            shape_maximum=float(shape.max().item()),
        )
        return PaperStiffnessCandidate(
            distance_stiffness=distance.detach().clone(),
            shape_stiffness=shape.detach().clone(),
            signal_ema=candidate.signal_ema.detach().clone(),
            log_step=log_step.detach().clone(),
            eligible_mask=candidate.eligible_mask.detach().clone(),
            source_mask=candidate.source_mask.detach().clone(),
            metrics=metrics,
        )

    @staticmethod
    def _stiffness_shadow_rejection_reasons(
        baseline: dict,
        candidate: dict,
        *,
        history_safe: bool = True,
    ) -> tuple[list[str], float, float]:
        improvement = baseline["visual_loss"] - candidate["visual_loss"]
        required_improvement = max(
            STIFFNESS_PREDICTION_ABSOLUTE_MARGIN,
            STIFFNESS_PREDICTION_RELATIVE_MARGIN * baseline["visual_loss"],
        )
        camera_safe = all(
            candidate_loss
            <= baseline_loss
            + max(
                STIFFNESS_CAMERA_ABSOLUTE_REGRESSION,
                STIFFNESS_CAMERA_RELATIVE_REGRESSION * baseline_loss,
            )
            for baseline_loss, candidate_loss in zip(
                baseline["camera_losses"], candidate["camera_losses"]
            )
        )
        allowed_minimum_volume = max(
            baseline["minimum_volume_ratio"]
            - STIFFNESS_MINIMUM_VOLUME_ABSOLUTE_DROP,
            baseline["minimum_volume_ratio"]
            * (1.0 - STIFFNESS_MINIMUM_VOLUME_RELATIVE_DROP),
        )
        volume_safe = (
            candidate["inverted_tetrahedra"]
            <= baseline["inverted_tetrahedra"]
            and candidate["minimum_volume_ratio"]
            >= allowed_minimum_volume
        )
        penetration_safe = (
            candidate["maximum_penetration_m"]
            <= baseline["maximum_penetration_m"]
            + STIFFNESS_PENETRATION_TOLERANCE_M
        )
        anchor_safe = (
            candidate["anchor_error_rms_m"]
            <= baseline["anchor_error_rms_m"]
            + STIFFNESS_ANCHOR_ERROR_TOLERANCE_M
            and candidate["anchor_error_maximum_m"]
            <= baseline["anchor_error_maximum_m"]
            + STIFFNESS_ANCHOR_ERROR_TOLERANCE_M
        )
        reasons = []
        if improvement < required_improvement:
            reasons.append("prediction_gap_not_improved")
        if not camera_safe:
            reasons.append("camera_regression")
        if not volume_safe:
            reasons.append("volume_quality_regression")
        if not penetration_safe:
            reasons.append("penetration_regression")
        if not anchor_safe:
            reasons.append("grip_anchor_regression")
        if not history_safe:
            reasons.append("history_regression")
        return reasons, improvement, required_improvement

    def _adopt_validated_stiffness_rollout(self, candidate: dict) -> dict:
        """Install a bounded, already-observed fixed-lag candidate state."""
        sim = self.environment.sim
        current_positions = wp.to_torch(sim.state_0.particle_q).detach()
        candidate_positions = candidate["particle_positions"].to(
            device=current_positions.device,
            dtype=current_positions.dtype,
        )
        displacement = torch.linalg.vector_norm(
            candidate_positions - current_positions, dim=1
        )
        state_rms_m = float(
            torch.sqrt(torch.mean(displacement * displacement)).item()
        )
        state_maximum_m = float(displacement.max().item())
        finite = bool(torch.isfinite(displacement).all().item())
        allowed = bool(
            STIFFNESS_ADOPT_VALIDATED_ROLLOUT
            and finite
            and state_rms_m <= STIFFNESS_MAXIMUM_ADOPTED_STATE_RMS_M
            and state_maximum_m
            <= STIFFNESS_MAXIMUM_ADOPTED_STATE_MAXIMUM_M
        )
        if allowed:
            sim.copy_embodied_gaussian_rollout_state(
                candidate["rollout_state"]
            )
            sim.update_gaussian_transforms()
            rollout_previous_residual = candidate[
                "rollout_previous_residual"
            ]
            self._previous_visual_residual = (
                None
                if rollout_previous_residual is None
                else rollout_previous_residual.detach().clone()
            )
        return {
            "validated_rollout_adopted": allowed,
            "validated_rollout_state_rms_m": state_rms_m,
            "validated_rollout_state_maximum_m": state_maximum_m,
            "validated_rollout_state_finite": finite,
            "maximum_adopted_state_rms_m": (
                STIFFNESS_MAXIMUM_ADOPTED_STATE_RMS_M
            ),
            "maximum_adopted_state_maximum_m": (
                STIFFNESS_MAXIMUM_ADOPTED_STATE_MAXIMUM_M
            ),
        }

    def _validate_pending_stiffness(
        self,
        *,
        gate_paused: bool,
        gate_reason: str,
        grip_active: bool,
    ) -> dict | None:
        pending = self._pending_stiffness_validation
        updater = self.stiffness_updater
        if pending is None or updater is None:
            return None
        horizon_frames = self.current_frame_index - pending.frame_index
        if (
            len(pending.commands) > STIFFNESS_MAXIMUM_PENDING_ROLLOUT_STEPS
            or horizon_frames
            > STIFFNESS_MAXIMUM_PREDICTION_HORIZON_FRAMES
        ):
            metrics = updater.reject(
                "validation_rollout_too_long", pending.candidate
            )
            metrics.update(
                validation_status="expired",
                prediction_horizon_frames=horizon_frames,
                prediction_rollout_steps=len(pending.commands),
            )
            self._record_terminal_stiffness_validation(
                pending, metrics, "validation_rollout_too_long"
            )
            self._pending_stiffness_validation = None
            self._last_stiffness_validation_metrics = metrics
            return metrics
        if self.current_frame_index <= pending.frame_index:
            return None
        if gate_paused or grip_active != pending.grip_active:
            reason = gate_reason or "grip_state_changed"
            metrics = updater.reject(reason, pending.candidate)
            metrics.update(
                validation_status="rejected_before_rollout",
                prediction_horizon_frames=(
                    self.current_frame_index - pending.frame_index
                ),
            )
            self._record_terminal_stiffness_validation(
                pending, metrics, reason
            )
            self._pending_stiffness_validation = None
            self._last_stiffness_validation_metrics = metrics
            return metrics
        if horizon_frames < STIFFNESS_COMMIT_VALIDATION_HORIZON_FRAMES:
            metrics = dict(pending.candidate.metrics)
            metrics.update(
                status="pending_horizon",
                validation_status="pending_horizon",
                prediction_horizon_frames=horizon_frames,
                prediction_rollout_steps=len(pending.commands),
                required_prediction_horizon_frames=(
                    STIFFNESS_COMMIT_VALIDATION_HORIZON_FRAMES
                ),
            )
            self._last_stiffness_validation_metrics = metrics
            return metrics
        live = self.environment.sim.clone_embodied_gaussian_rollout_state()
        controller_state = (
            self.current_frame_index,
            self.current_timestep,
            self._last_state_index,
            self._last_q_full.detach().clone(),
        )
        base_candidate = pending.candidate
        line_search_records: list[dict] = []
        try:
            baseline = self._run_stiffness_prediction_shadow(
                pending, use_candidate=False
            )
            for scale in STIFFNESS_CANDIDATE_LOG_STEP_SCALES:
                variant = self._scaled_stiffness_candidate(
                    base_candidate, scale
                )
                pending.candidate = variant
                shadow = self._run_stiffness_prediction_shadow(
                    pending, use_candidate=True
                )
                variant_reasons, variant_improvement, required_improvement = (
                    self._stiffness_shadow_rejection_reasons(
                        baseline, shadow
                    )
                )
                line_search_records.append(
                    {
                        "scale": float(scale),
                        "candidate": variant,
                        "shadow": shadow,
                        "reasons": variant_reasons,
                        "improvement": variant_improvement,
                        "required_improvement": required_improvement,
                    }
                )
        finally:
            pending.candidate = base_candidate
            self.environment.sim.copy_embodied_gaussian_rollout_state(live)
            (
                self.current_frame_index,
                self.current_timestep,
                self._last_state_index,
                self._last_q_full,
            ) = controller_state
            self.dataset_manager.update_frames(self.current_timestep)
            self.environment.sim.update_gaussian_transforms()

        # Prefer the visually best physically safe step, then check its
        # historical relaxation. If history rejects it, fall back to the next
        # best safe scale rather than discarding the entire material direction.
        selected_record = None
        for record in sorted(
            (
                record
                for record in line_search_records
                if not record["reasons"]
            ),
            key=lambda record: record["shadow"]["visual_loss"],
        ):
            history_baseline, history_candidate, history_safe = (
                self._evaluate_stiffness_history(record["candidate"])
            )
            record["history_baseline_rms_m"] = history_baseline
            record["history_candidate_rms_m"] = history_candidate
            if history_safe:
                selected_record = record
                break
            record["reasons"] = [*record["reasons"], "history_regression"]

        if selected_record is None:
            selected_record = next(
                record
                for record in line_search_records
                if abs(record["scale"] - 1.0) <= 1.0e-12
            )
            history_baseline = pending.history_baseline_rms_m
            history_candidate = pending.history_candidate_rms_m
            history_safe = (
                history_candidate
                <= history_baseline
                * (1.0 + STIFFNESS_HISTORY_RELATIVE_TOLERANCE)
                + STIFFNESS_HISTORY_ABSOLUTE_TOLERANCE_M
            )
            if not history_safe and "history_regression" not in selected_record["reasons"]:
                selected_record["reasons"].append("history_regression")
        else:
            history_baseline = selected_record["history_baseline_rms_m"]
            history_candidate = selected_record["history_candidate_rms_m"]

        pending.candidate = selected_record["candidate"]
        updater.pending_candidate = pending.candidate
        pending.history_baseline_rms_m = float(history_baseline)
        pending.history_candidate_rms_m = float(history_candidate)
        pending.candidate.metrics["line_search_scale"] = float(
            selected_record["scale"]
        )
        candidate = selected_record["shadow"]
        improvement = float(selected_record["improvement"])
        required_improvement = float(
            selected_record["required_improvement"]
        )
        reasons = list(selected_record["reasons"])
        if reasons:
            metrics = updater.reject("+".join(reasons), pending.candidate)
        else:
            metrics = updater.commit(pending.candidate)
        rollout_adoption_metrics = {
            "validated_rollout_adopted": False,
            "validated_rollout_state_rms_m": 0.0,
            "validated_rollout_state_maximum_m": 0.0,
            "validated_rollout_state_finite": True,
            "maximum_adopted_state_rms_m": (
                STIFFNESS_MAXIMUM_ADOPTED_STATE_RMS_M
            ),
            "maximum_adopted_state_maximum_m": (
                STIFFNESS_MAXIMUM_ADOPTED_STATE_MAXIMUM_M
            ),
        }
        if metrics["status"] == "committed":
            rollout_adoption_metrics = (
                self._adopt_validated_stiffness_rollout(candidate)
            )
        metrics.update(
            validation_status=metrics["status"],
            prediction_horizon_frames=(
                self.current_frame_index - pending.frame_index
            ),
            prediction_rollout_steps=len(pending.commands),
            baseline_visual_loss=baseline["visual_loss"],
            candidate_visual_loss=candidate["visual_loss"],
            validation_visual_objective="mean_prediction_loss_frames_1_to_h",
            baseline_endpoint_visual_loss=baseline[
                "endpoint_visual_loss"
            ],
            candidate_endpoint_visual_loss=candidate[
                "endpoint_visual_loss"
            ],
            prediction_improvement=improvement,
            required_prediction_improvement=required_improvement,
            baseline_camera_losses=baseline["camera_losses"],
            candidate_camera_losses=candidate["camera_losses"],
            baseline_minimum_volume_ratio=baseline["minimum_volume_ratio"],
            candidate_minimum_volume_ratio=candidate["minimum_volume_ratio"],
            baseline_maximum_penetration_m=baseline["maximum_penetration_m"],
            candidate_maximum_penetration_m=candidate["maximum_penetration_m"],
            baseline_anchor_error_rms_m=baseline["anchor_error_rms_m"],
            candidate_anchor_error_rms_m=candidate["anchor_error_rms_m"],
            baseline_anchor_error_maximum_m=(
                baseline["anchor_error_maximum_m"]
            ),
            candidate_anchor_error_maximum_m=(
                candidate["anchor_error_maximum_m"]
            ),
            baseline_distance_loss=baseline["distance_loss"],
            candidate_distance_loss=candidate["distance_loss"],
            baseline_volume_loss=baseline["volume_loss"],
            candidate_volume_loss=candidate["volume_loss"],
            baseline_shape_loss=baseline["shape_loss"],
            candidate_shape_loss=candidate["shape_loss"],
            history_baseline_rms_m=history_baseline,
            history_candidate_rms_m=history_candidate,
            line_search_scale=selected_record["scale"],
            line_search_trial_count=len(line_search_records),
            **rollout_adoption_metrics,
        )
        if metrics["status"] == "committed":
            self._start_committed_stiffness_evaluation(pending)
        if self.stiffness_metrics_recorder is not None:
            self.stiffness_metrics_recorder.record(
                event="stiffness_validation",
                frame_index=self.current_frame_index,
                timestamp_s=self.current_timestep,
                phase=self._current_action_phase,
                physical={
                    "baseline": self._serializable_shadow_metrics(baseline),
                    "candidate": self._serializable_shadow_metrics(candidate),
                },
                material=self._current_material_evaluation_metrics(metrics),
                prediction={
                    "horizon_frames": metrics[
                        "prediction_horizon_frames"
                    ],
                    "baseline_gap": baseline["visual_loss"],
                    "candidate_gap": candidate["visual_loss"],
                    "gap_improvement": improvement,
                    "required_improvement": required_improvement,
                    "status": metrics["status"],
                },
                details={
                    "rejection_reasons": reasons,
                    "line_search_trials": [
                        {
                            "scale": record["scale"],
                            "gap_improvement": record["improvement"],
                            "rejection_reasons": record["reasons"],
                        }
                        for record in line_search_records
                    ],
                },
                force_summary=True,
            )
        self._pending_stiffness_validation = None
        self._last_stiffness_validation_metrics = metrics
        return metrics

    def capture_flow_depth_source_state(self, frame_index: int) -> None:
        """缓存该图像时刻视觉校正后的 q+，供下一因果观测计算创新量。"""

        sequence = self.flow_depth_observations
        if sequence is None:
            return
        frame_index = int(frame_index)
        if not np.any(sequence.current_source_frames == frame_index):
            return
        positions = (
            wp.to_torch(self.environment.sim.state_0.particle_q)
            .detach()
            .cpu()
            .numpy()
            .copy()
        )
        self._flow_depth_source_states[frame_index] = positions
        self._flow_depth_source_rollouts[frame_index] = FlowDepthSourceSnapshot(
            rollout_state=(
                self.environment.sim.clone_embodied_gaussian_rollout_state()
            ),
            positions=positions,
            velocities=(
                wp.to_torch(self.environment.sim.state_0.particle_qd)
                .detach()
                .clone()
            ),
            physics_iteration_count=int(self._physics_iteration_count),
        )
        if self._flow_depth_reference_range_centers is None:
            assert self.flow_depth_bindings is not None
            self._flow_depth_reference_range_centers = fixed_range_centers(
                positions, self.flow_depth_bindings
            )
        live_sources = set(
            int(value)
            for value in sequence.current_source_frames[
                sequence.next_source_frames > frame_index
            ]
        )
        self._flow_depth_source_states = {
            source: state
            for source, state in self._flow_depth_source_states.items()
            if source in live_sources or source == frame_index
        }
        self._flow_depth_source_rollouts = {
            source: state
            for source, state in self._flow_depth_source_rollouts.items()
            if source in live_sources or source == frame_index
        }

    def _append_warp_gradient_transition(
        self,
        *,
        source_frame: int,
        source_snapshot: FlowDepthSourceSnapshot,
        corrected_positions: torch.Tensor,
        candidate: PaperStiffnessCandidate,
        control_exclusion_mask: torch.Tensor,
        track_target_positions: torch.Tensor,
        track_valid_mask: torch.Tensor,
    ) -> None:
        history = self._warp_gradient_transitions
        preserve_discontinuous_window = (
            self.stiffness_updater is not None
            and self.stiffness_updater.settings.update_mode
            == "differentiable_global_mhe"
        )
        if (
            history
            and source_frame != history[-1].destination_command.frame_index
            and not preserve_discontinuous_window
        ):
            history.clear()
        history.append(
            WarpGradientTransition(
                source_frame=int(source_frame),
                destination_command=self._current_stiffness_tool_command(),
                rollout_state=source_snapshot.rollout_state,
                physics_steps=max(
                    1,
                    int(self._physics_iteration_count)
                    - int(source_snapshot.physics_iteration_count),
                ),
                corrected_positions=corrected_positions.detach().clone(),
                eligible_mask=candidate.eligible_mask.detach().clone(),
                material_active_mask=candidate.source_mask.detach().clone(),
                control_exclusion_mask=(
                    control_exclusion_mask.detach().clone()
                ),
                track_target_positions=track_target_positions.detach().clone(),
                track_valid_mask=track_valid_mask.detach().clone(),
            )
        )

    def _warp_global_parameter_objective_components(
        self,
        log_coefficients: torch.Tensor,
    ) -> dict[str, float]:
        """Replay Warp once and separate combined, H1, and long losses.

        The legacy name is retained for compatibility.  In local-relative
        mode ``log_coefficients`` are the regional zero-mean coefficients and
        global damping/coupling remain fixed.
        """
        updater = self.stiffness_updater
        assert updater is not None
        settings = updater.settings
        coefficients = log_coefficients.detach().to(
            device=updater.rest_positions.device, dtype=torch.float32
        )
        local_mode = settings.update_mode in {
            "differentiable_local_relative",
            "differentiable_hierarchical_relative",
            "differentiable_particle_graph_lm",
        }
        if local_mode:
            distance = updater.local_distance_from_coefficients(coefficients)
            shape = updater.local_shape_from_coefficients(coefficients)
            damping = float(updater.initial_velocity_damping_per_second)
            coupling = float(updater.initial_coupling_gain)
        else:
            distance = torch.clamp(
                updater.initial_distance * torch.exp(coefficients[0]),
                settings.distance_minimum,
                settings.distance_maximum,
            )
            damping = float(
                torch.clamp(
                    torch.as_tensor(
                        updater.initial_velocity_damping_per_second,
                        dtype=coefficients.dtype,
                        device=coefficients.device,
                    )
                    * torch.exp(coefficients[1]),
                    settings.global_damping_minimum_per_second,
                    settings.global_damping_maximum_per_second,
                ).item()
            )
            coupling = (
                float(
                    torch.clamp(
                        torch.as_tensor(
                            updater.initial_coupling_gain,
                            dtype=coefficients.dtype,
                            device=coefficients.device,
                        )
                        * torch.exp(coefficients[2]),
                        settings.global_coupling_minimum,
                        settings.global_coupling_maximum,
                    ).item()
                )
                if settings.global_optimize_coupling
                else float(updater.initial_coupling_gain)
            )
        position_losses: list[tuple[int, float, torch.Tensor]] = []
        relative_track_losses: list[tuple[int, float, torch.Tensor]] = []
        strain_losses: list[tuple[int, float, torch.Tensor]] = []
        source, target = updater.edges[:, 0], updater.edges[:, 1]
        sim = self.environment.sim
        transitions = list(self._warp_gradient_transitions)
        robust_relative_mode = settings.update_mode in {
            "differentiable_global_relative",
            "differentiable_global_mhe",
            "differentiable_local_relative",
            "differentiable_hierarchical_relative",
            "differentiable_particle_graph_lm",
        }
        if not transitions:
            return {
                "combined": float("inf"),
                "short": float("inf"),
                "long": float("inf"),
            }
        # Replay once from the earliest saved, visually corrected state.  Do
        # not restore each intermediate corrected state: the resulting gaps
        # at horizons 1/2/3 are the causal signal relevant to future rollout.
        sim.copy_embodied_gaussian_rollout_state(transitions[0].rollout_state)
        # Restoring a rollout state also restores its saved material arrays.
        # Install the finite-difference probe *after* that restore; otherwise
        # every plus/minus replay silently runs at the same verified stiffness.
        with torch.no_grad():
            updater.distance_stiffness.copy_(distance)
            updater.shape_stiffness.copy_(
                shape if local_mode else updater.initial_shape
            )
        self.environment.physics_settings.particle_velocity_damping_per_second = (
            damping
        )
        self._sim_grasp_coupling_gain = coupling
        for horizon_index, transition in enumerate(transitions, start=1):
            self._apply_stiffness_tool_command(transition.destination_command)
            for _ in range(transition.physics_steps):
                if self._sim_reconstruction_mode:
                    self._project_sim_grasp_boundary(update_gaussians=False)
                self.environment.step(compute_visual_forces=False)
                if self._sim_reconstruction_mode:
                    self._project_sim_grasp_boundary(update_gaussians=False)
            if local_mode:
                local_horizon_index = {1: 0, 3: 1, 5: 2}.get(horizon_index)
                if local_horizon_index is None:
                    continue
                horizon_weight = settings.local_horizon_weights[
                    local_horizon_index
                ]
            else:
                horizon_weight = settings.global_horizon_weights[
                    horizon_index - 1
                ]
            positions = wp.to_torch(sim.state_0.particle_q).detach().clone()
            active = (
                transition.eligible_mask & transition.material_active_mask
            )
            predicted_track_positions = updater.fixed_track_centers(positions)
            support_coverage = torch.sum(
                updater.track_particle_weights
                * active[updater.track_particle_ids].to(torch.float32),
                dim=1,
            )
            track_active = transition.track_valid_mask & (support_coverage >= 0.50)
            if robust_relative_mode:
                (
                    node_loss,
                    relative_track_loss,
                    _relative_active,
                    _active_pair_count,
                ) = updater.robust_relative_track_losses(
                    predicted_track_positions,
                    transition.track_target_positions,
                    track_active,
                )
            else:
                safe_error = torch.where(
                    track_active[:, None],
                    predicted_track_positions
                    - transition.track_target_positions,
                    torch.zeros_like(predicted_track_positions),
                )
                scaled_error = safe_error / settings.residual_full_scale_m
                node_loss = torch.sqrt(
                    torch.sum(scaled_error.square(), dim=1) + 1.0e-4
                ) - 0.01
                relative_track_loss = node_loss.sum() * 0.0
            if bool(track_active.any().item()):
                distribution_loss, *_components = (
                    updater.global_position_distribution_loss(
                        node_loss,
                        track_active,
                        updater.track_region_basis,
                    )
                )
                if robust_relative_mode:
                    relative_weight = settings.global_relative_track_weight
                    distribution_loss = (
                        (1.0 - relative_weight) * distribution_loss
                        + relative_weight * relative_track_loss
                    )
                    relative_track_losses.append(
                        (horizon_index, horizon_weight, relative_track_loss)
                    )
                position_losses.append(
                    (horizon_index, horizon_weight, distribution_loss)
                )
            if updater.edges.numel():
                edge_active = (
                    (active[source] | active[target])
                    & ~transition.control_exclusion_mask[source]
                    & ~transition.control_exclusion_mask[target]
                )
                current_length = torch.linalg.vector_norm(
                    positions[target] - positions[source], dim=1
                )
                target_length = torch.linalg.vector_norm(
                    transition.corrected_positions[target]
                    - transition.corrected_positions[source],
                    dim=1,
                )
                strain_error = (
                    (current_length - target_length)
                    / updater.rest_edge_lengths
                ) / settings.edge_strain_full_scale
                robust_strain = torch.sqrt(
                    strain_error.square() + 1.0e-4
                ) - 0.01
                if bool(edge_active.any().item()):
                    strain_losses.append(
                        (
                            horizon_index,
                            horizon_weight,
                            robust_strain[edge_active].mean(),
                        )
                    )
        if not position_losses and not strain_losses:
            return {
                "combined": float("inf"),
                "short": float("inf"),
                "long": float("inf"),
            }
        blend = settings.strain_signal_weight
        if local_mode:
            prior, _coefficient_prior, _spatial_prior = (
                updater.local_regularization_loss(coefficients)
            )
        else:
            prior_weights = torch.tensor(
                (1.0, 1.0, 0.25 if settings.global_optimize_coupling else 0.0),
                dtype=coefficients.dtype,
                device=coefficients.device,
            )
            prior = settings.global_parameter_prior_weight * torch.mean(
                prior_weights * coefficients.square()
            )

        def selected_loss(
            horizons: set[int],
            *,
            include_prior: bool,
        ) -> torch.Tensor:
            selected_position = [
                (weight, loss)
                for horizon, weight, loss in position_losses
                if horizon in horizons
            ]
            selected_strain = [
                (weight, loss)
                for horizon, weight, loss in strain_losses
                if horizon in horizons
            ]
            position_loss = (
                sum(weight * loss for weight, loss in selected_position)
                / sum(weight for weight, _loss in selected_position)
                if selected_position
                else torch.zeros((), device=coefficients.device)
            )
            strain_loss = (
                sum(weight * loss for weight, loss in selected_strain)
                / sum(weight for weight, _loss in selected_strain)
                if selected_strain
                else torch.zeros((), device=coefficients.device)
            )
            data = (1.0 - blend) * position_loss + blend * strain_loss
            return data + (prior if include_prior else prior * 0.0)

        available_horizons = {
            horizon for horizon, _weight, _loss in position_losses + strain_losses
        }
        short_horizons = {1}
        long_horizons = available_horizons - short_horizons
        return {
            "combined": float(
                selected_loss(available_horizons, include_prior=True).item()
            ),
            # H1 is a pure reconstruction constraint.  The regularizer belongs
            # to the H3/H5 identification objective and must not make a step
            # appear short-safe merely because it moves toward the prior.
            "short": float(
                selected_loss(short_horizons, include_prior=False).item()
            ),
            "long": float(
                selected_loss(long_horizons, include_prior=True).item()
            ),
        }

    def _warp_global_parameter_objective(
        self,
        log_coefficients: torch.Tensor,
    ) -> float:
        """Compatibility scalar for the former combined H1/H3/H5 loss."""
        return self._warp_global_parameter_objective_components(
            log_coefficients
        )["combined"]

    def _warp_global_parameter_finite_difference_gradient(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return combined, H1, and long Warp central-difference gradients."""
        updater = self.stiffness_updater
        assert updater is not None
        local_mode = updater.settings.update_mode in {
            "differentiable_local_relative",
            "differentiable_hierarchical_relative",
            "differentiable_particle_graph_lm",
        }
        required_transitions = 5 if local_mode else 3
        center_template = (
            updater.low_dimensional_log_coefficients
            if local_mode
            else updater.global_log_coefficients
        )
        if len(self._warp_gradient_transitions) < required_transitions:
            missing = torch.full_like(center_template, float("nan"))
            return missing, missing.clone(), missing.clone()
        sim = self.environment.sim
        live = sim.clone_embodied_gaussian_rollout_state()
        live_command = self._current_stiffness_tool_command()
        live_damping = float(
            self.environment.physics_settings.particle_velocity_damping_per_second
        )
        live_coupling = float(self._sim_grasp_coupling_gain)
        live_distance = updater.distance_stiffness.detach().clone()
        live_shape = updater.shape_stiffness.detach().clone()
        epsilon = (
            updater.settings.local_finite_difference_log_epsilon
            if local_mode
            else updater.settings.finite_difference_log_epsilon
        )
        center = center_template.detach().clone()
        if local_mode:
            center = updater.normalize_local_coefficients(center)
        elif not updater.settings.global_optimize_coupling:
            center[2] = 0.0
        gradient = torch.zeros_like(center)
        short_gradient = torch.zeros_like(center)
        long_gradient = torch.zeros_like(center)
        try:
            parameter_count = (
                len(center)
                if local_mode
                else (3 if updater.settings.global_optimize_coupling else 2)
            )
            for parameter_index in range(parameter_count):
                parameter_epsilon = (
                    updater.settings.finite_difference_log_epsilon
                    if (
                        updater.settings.update_mode
                        == "differentiable_hierarchical_relative"
                        and parameter_index == 0
                    )
                    else epsilon
                )
                plus = center.clone()
                minus = center.clone()
                plus[parameter_index] += parameter_epsilon
                minus[parameter_index] -= parameter_epsilon
                if local_mode:
                    plus = updater.normalize_local_coefficients(plus)
                    minus = updater.normalize_local_coefficients(minus)
                plus_losses = self._warp_global_parameter_objective_components(
                    plus
                )
                minus_losses = self._warp_global_parameter_objective_components(
                    minus
                )
                gradient[parameter_index] = (
                    plus_losses["combined"] - minus_losses["combined"]
                ) / (2.0 * parameter_epsilon)
                short_gradient[parameter_index] = (
                    plus_losses["short"] - minus_losses["short"]
                ) / (2.0 * parameter_epsilon)
                long_gradient[parameter_index] = (
                    plus_losses["long"] - minus_losses["long"]
                ) / (2.0 * parameter_epsilon)
            if local_mode:
                gradient = updater.project_local_gradient(gradient)
                short_gradient = updater.project_local_gradient(short_gradient)
                long_gradient = updater.project_local_gradient(long_gradient)
        finally:
            sim.copy_embodied_gaussian_rollout_state(live)
            with torch.no_grad():
                updater.distance_stiffness.copy_(live_distance)
                updater.shape_stiffness.copy_(live_shape)
            self.environment.physics_settings.particle_velocity_damping_per_second = (
                live_damping
            )
            self._sim_grasp_coupling_gain = live_coupling
            self._apply_stiffness_tool_command(live_command)
            sim.update_gaussian_transforms()
        return gradient, short_gradient, long_gradient

    def _warp_particle_graph_directional_derivatives(
        self,
        autograd_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Check a large particle Jacobian with deterministic Warp directions."""
        updater = self.stiffness_updater
        assert updater is not None
        if updater.settings.update_mode != "differentiable_particle_graph_lm":
            raise RuntimeError("Particle directional check has the wrong mode")
        center = updater.normalize_local_coefficients(
            updater.low_dimensional_log_coefficients.detach().clone()
        )
        gradient = updater.project_local_gradient(
            autograd_gradient.detach().to(
                device=center.device, dtype=center.dtype
            )
        )
        if gradient.shape != center.shape:
            raise ValueError("Particle-graph gradient has the wrong shape")
        if len(self._warp_gradient_transitions) < 5:
            missing = torch.full(
                (updater.settings.graph_directional_probe_count,),
                float("nan"),
                dtype=center.dtype,
                device=center.device,
            )
            return missing, missing.clone()

        directions: list[torch.Tensor] = []
        primary = gradient / gradient.abs().max().clamp_min(1.0e-12)
        directions.append(primary)
        indices = torch.arange(
            len(center) - 1, dtype=center.dtype, device=center.device
        )
        frequencies = (0.017, 0.031, 0.047, 0.071)
        for probe_index in range(
            1, updater.settings.graph_directional_probe_count
        ):
            direction = torch.zeros_like(center)
            direction[0] = 1.0 if probe_index % 2 else -1.0
            frequency = frequencies[(probe_index - 1) % len(frequencies)]
            direction[1:] = torch.sin(
                indices * frequency + float(probe_index) * 0.73
            )
            direction = updater.project_local_gradient(direction)
            primary_energy = torch.dot(primary, primary).clamp_min(1.0e-12)
            direction = direction - (
                torch.dot(direction, primary) / primary_energy
            ) * primary
            direction = updater.project_local_gradient(direction)
            direction = direction / direction.abs().max().clamp_min(1.0e-12)
            directions.append(direction)

        sim = self.environment.sim
        live = sim.clone_embodied_gaussian_rollout_state()
        live_command = self._current_stiffness_tool_command()
        live_damping = float(
            self.environment.physics_settings.particle_velocity_damping_per_second
        )
        live_coupling = float(self._sim_grasp_coupling_gain)
        live_distance = updater.distance_stiffness.detach().clone()
        live_shape = updater.shape_stiffness.detach().clone()
        epsilon = updater.settings.graph_directional_log_epsilon
        predicted = torch.empty(
            len(directions), dtype=center.dtype, device=center.device
        )
        measured = torch.empty_like(predicted)
        try:
            for probe_index, direction in enumerate(directions):
                plus = updater.normalize_local_coefficients(
                    center + epsilon * direction
                )
                minus = updater.normalize_local_coefficients(
                    center - epsilon * direction
                )
                plus_loss = self._warp_global_parameter_objective_components(
                    plus
                )["combined"]
                minus_loss = self._warp_global_parameter_objective_components(
                    minus
                )["combined"]
                predicted[probe_index] = torch.dot(gradient, direction)
                measured[probe_index] = (plus_loss - minus_loss) / (
                    2.0 * epsilon
                )
        finally:
            sim.copy_embodied_gaussian_rollout_state(live)
            with torch.no_grad():
                updater.distance_stiffness.copy_(live_distance)
                updater.shape_stiffness.copy_(live_shape)
            self.environment.physics_settings.particle_velocity_damping_per_second = (
                live_damping
            )
            self._sim_grasp_coupling_gain = live_coupling
            self._apply_stiffness_tool_command(live_command)
            sim.update_gaussian_transforms()
        return predicted, measured

    def _warp_global_mhe_residual_vector(
        self,
        log_coefficients: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int, int]:
        """Return equal-weight causal H1 and long-rollout Warp residuals.

        A periodic reconstruction holdout splits the window into independent
        contiguous segments.  Each segment starts from the state that was
        actually available online and is then rolled forward without visual
        state resets.  In parallel, every already observed transition is
        replayed once from its own source snapshot.  The H1 and long vectors
        are normalized independently and assigned one half of the objective
        each.  No future frame, evaluation GT, or parameter branch is used.
        """
        updater = self.stiffness_updater
        assert updater is not None
        settings = updater.settings
        coefficients = log_coefficients.detach().to(
            device=updater.rest_positions.device, dtype=torch.float32
        )
        if coefficients.shape != (3,):
            raise ValueError("Global MHE expects three stored log coefficients")
        distance = torch.clamp(
            updater.initial_distance * torch.exp(coefficients[0]),
            settings.distance_minimum,
            settings.distance_maximum,
        )
        damping = float(
            torch.clamp(
                torch.as_tensor(
                    updater.initial_velocity_damping_per_second,
                    dtype=coefficients.dtype,
                    device=coefficients.device,
                )
                * torch.exp(coefficients[1]),
                settings.global_damping_minimum_per_second,
                settings.global_damping_maximum_per_second,
            ).item()
        )
        transitions = list(self._warp_gradient_transitions)[
            -settings.observable_window_size :
        ]
        if not transitions:
            return torch.empty(0, device=coefficients.device), 0, 0, 0
        segments: list[list[WarpGradientTransition]] = []
        for transition in transitions:
            if (
                segments
                and transition.source_frame
                == segments[-1][-1].destination_command.frame_index
            ):
                segments[-1].append(transition)
            else:
                segments.append([transition])
        transition_weights = [
            min(
                settings.observable_horizon_weight_maximum,
                1.0 + settings.observable_horizon_weight_growth * local_horizon,
            )
            for segment in segments
            for local_horizon in range(1, len(segment) + 1)
        ]
        total_horizon_weight = max(sum(transition_weights), 1.0e-12)
        objective_weight_sum = (
            settings.observable_short_horizon_weight
            + settings.observable_long_horizon_weight
        )
        short_objective_weight = (
            settings.observable_short_horizon_weight / objective_weight_sum
        )
        long_objective_weight = (
            settings.observable_long_horizon_weight / objective_weight_sum
        )

        sim = self.environment.sim
        with torch.no_grad():
            updater.distance_stiffness.copy_(distance)
            updater.shape_stiffness.copy_(updater.initial_shape)
        self.environment.physics_settings.particle_velocity_damping_per_second = (
            damping
        )
        self._sim_grasp_coupling_gain = float(updater.initial_coupling_gain)
        short_residual_blocks: list[torch.Tensor] = []
        long_residual_blocks: list[torch.Tensor] = []
        used_long_transition_count = 0
        used_short_transition_count = 0
        source, target = updater.edges[:, 0], updater.edges[:, 1]
        track_edges = updater.track_neighbor_edges
        relative_weight = float(settings.global_relative_track_weight)
        strain_weight = float(settings.strain_signal_weight)
        position_weight = 1.0 - strain_weight

        def restore_probe_material(transition: WarpGradientTransition) -> None:
            sim.copy_embodied_gaussian_rollout_state(transition.rollout_state)
            # Rollout snapshots contain the verified material.  The current
            # finite-difference probe must therefore be applied after every
            # restore or the distance-stiffness Jacobian becomes exactly zero.
            with torch.no_grad():
                updater.distance_stiffness.copy_(distance)
                updater.shape_stiffness.copy_(updater.initial_shape)

        def advance_transition(transition: WarpGradientTransition) -> torch.Tensor:
            self._apply_stiffness_tool_command(transition.destination_command)
            for _ in range(transition.physics_steps):
                if self._sim_reconstruction_mode:
                    self._project_sim_grasp_boundary(update_gaussians=False)
                self.environment.step(compute_visual_forces=False)
                if self._sim_reconstruction_mode:
                    self._project_sim_grasp_boundary(update_gaussians=False)
            return wp.to_torch(sim.state_0.particle_q).detach().clone()

        def residual_blocks_for_transition(
            positions: torch.Tensor,
            transition: WarpGradientTransition,
            normalized_objective_weight: float,
        ) -> tuple[list[torch.Tensor], bool]:
            blocks: list[torch.Tensor] = []
            active = transition.eligible_mask & transition.material_active_mask
            contributed = False
            if (
                updater.track_particle_ids is not None
                and updater.track_particle_weights is not None
            ):
                predicted_tracks = updater.fixed_track_centers(positions)
                support_coverage = torch.sum(
                    updater.track_particle_weights
                    * active[updater.track_particle_ids].to(torch.float32),
                    dim=1,
                )
                track_active = (
                    transition.track_valid_mask
                    & (support_coverage >= 0.50)
                    & torch.isfinite(
                        transition.track_target_positions
                    ).all(dim=1)
                )
                safe_error = torch.where(
                    track_active[:, None],
                    predicted_tracks - transition.track_target_positions,
                    torch.zeros_like(predicted_tracks),
                )
                absolute = safe_error[track_active].reshape(-1) / (
                    settings.residual_full_scale_m
                )
                if absolute.numel() and position_weight > 0.0:
                    block_weight = math.sqrt(
                        normalized_objective_weight
                        * position_weight
                        * (1.0 - relative_weight)
                        / int(absolute.numel())
                    )
                    blocks.append(absolute * block_weight)
                    contributed = True
                if track_edges.numel() and position_weight > 0.0:
                    track_source = track_edges[:, 0]
                    track_target = track_edges[:, 1]
                    pair_active = (
                        track_active[track_source] & track_active[track_target]
                    )
                    relative = (
                        safe_error[track_source] - safe_error[track_target]
                    )[pair_active].reshape(-1) / (
                        settings.residual_full_scale_m * math.sqrt(2.0)
                    )
                    if relative.numel():
                        block_weight = math.sqrt(
                            normalized_objective_weight
                            * position_weight
                            * relative_weight
                            / int(relative.numel())
                        )
                        blocks.append(relative * block_weight)
                        contributed = True
            if updater.edges.numel() and strain_weight > 0.0:
                edge_active = (
                    (active[source] | active[target])
                    & ~transition.control_exclusion_mask[source]
                    & ~transition.control_exclusion_mask[target]
                )
                edge_ids = torch.nonzero(edge_active, as_tuple=False).flatten()
                maximum_edges = settings.observable_maximum_edge_residuals
                if edge_ids.numel() > maximum_edges:
                    sample = torch.linspace(
                        0,
                        int(edge_ids.numel()) - 1,
                        steps=maximum_edges,
                        device=edge_ids.device,
                    ).round().to(torch.long)
                    edge_ids = edge_ids[sample]
                if edge_ids.numel():
                    current_length = torch.linalg.vector_norm(
                        positions[target[edge_ids]] - positions[source[edge_ids]],
                        dim=1,
                    )
                    target_length = torch.linalg.vector_norm(
                        transition.corrected_positions[target[edge_ids]]
                        - transition.corrected_positions[source[edge_ids]],
                        dim=1,
                    )
                    strain = (
                        (current_length - target_length)
                        / updater.rest_edge_lengths[edge_ids]
                    ) / settings.edge_strain_full_scale
                    block_weight = math.sqrt(
                        normalized_objective_weight
                        * strain_weight
                        / int(strain.numel())
                    )
                    blocks.append(strain * block_weight)
                    contributed = True
            return blocks, contributed

        # H1 reconstruction term: every accepted transition starts from the
        # state that was actually available immediately before that frame.
        normalized_short_weight = short_objective_weight / len(transitions)
        for transition in transitions:
            restore_probe_material(transition)
            positions = advance_transition(transition)
            blocks, contributed = residual_blocks_for_transition(
                positions,
                transition,
                normalized_short_weight,
            )
            short_residual_blocks.extend(blocks)
            used_short_transition_count += int(contributed)

        # Long term: contiguous causal segments remain open loop so the same
        # fixed objective retains the v2 future-prediction capability.
        for segment in segments:
            restore_probe_material(segment[0])
            for local_horizon, transition in enumerate(segment, start=1):
                positions = advance_transition(transition)
                horizon_weight = min(
                    settings.observable_horizon_weight_maximum,
                    1.0
                    + settings.observable_horizon_weight_growth * local_horizon,
                )
                normalized_horizon_weight = (
                    long_objective_weight
                    * horizon_weight
                    / total_horizon_weight
                )
                blocks, contributed = residual_blocks_for_transition(
                    positions,
                    transition,
                    normalized_horizon_weight,
                )
                long_residual_blocks.extend(blocks)
                used_long_transition_count += int(contributed)
        residual_blocks = short_residual_blocks + long_residual_blocks
        if not residual_blocks:
            return torch.empty(0, device=coefficients.device), 0, 0, 0
        short_residual_count = sum(
            int(block.numel()) for block in short_residual_blocks
        )
        return (
            torch.cat(residual_blocks),
            used_long_transition_count,
            short_residual_count,
            used_short_transition_count,
        )

    def _warp_global_mhe_lm_proposal(
        self,
    ) -> tuple[
        torch.Tensor,
        bool,
        str,
        dict[str, float | int | str],
    ]:
        """Build and gate one actual-Warp moving-horizon LM proposal."""
        updater = self.stiffness_updater
        assert updater is not None
        settings = updater.settings
        transitions = list(self._warp_gradient_transitions)
        center = updater.global_log_coefficients.detach().clone()
        center[2] = 0.0
        zero_step = torch.zeros_like(center)
        if len(transitions) < settings.observable_minimum_transitions:
            return zero_step, False, "observable_window_warming_up", {
                "observable_status": "warming_up",
                "observable_allowed": 0,
                "observable_transition_count": len(transitions),
                "observable_minimum_transitions": int(
                    settings.observable_minimum_transitions
                ),
            }
        latest_frame = transitions[-1].destination_command.frame_index
        if latest_frame % settings.observable_update_interval_frames != 0:
            return zero_step, False, "observable_update_interval", {
                "observable_status": "update_interval",
                "observable_allowed": 0,
                "observable_transition_count": len(transitions),
                "observable_latest_frame": int(latest_frame),
                "observable_update_interval_frames": int(
                    settings.observable_update_interval_frames
                ),
            }

        segments = 1 + sum(
            int(current.source_frame != previous.destination_command.frame_index)
            for previous, current in zip(transitions, transitions[1:])
        )
        sim = self.environment.sim
        live = sim.clone_embodied_gaussian_rollout_state()
        live_command = self._current_stiffness_tool_command()
        live_damping = float(
            self.environment.physics_settings.particle_velocity_damping_per_second
        )
        live_coupling = float(self._sim_grasp_coupling_gain)
        live_distance = updater.distance_stiffness.detach().clone()
        live_shape = updater.shape_stiffness.detach().clone()
        epsilon = float(settings.finite_difference_log_epsilon)
        try:
            (
                center_residual,
                used_transitions,
                short_residual_count,
                used_short_transitions,
            ) = (
                self._warp_global_mhe_residual_vector(center)
            )
            jacobian_columns: list[torch.Tensor] = []
            wide_columns: list[torch.Tensor] = []
            for parameter_index in range(2):
                plus = center.clone()
                minus = center.clone()
                wide_plus = center.clone()
                wide_minus = center.clone()
                plus[parameter_index] += epsilon
                minus[parameter_index] -= epsilon
                wide_plus[parameter_index] += 2.0 * epsilon
                wide_minus[parameter_index] -= 2.0 * epsilon
                plus_residual, _, _, _ = self._warp_global_mhe_residual_vector(plus)
                minus_residual, _, _, _ = self._warp_global_mhe_residual_vector(minus)
                wide_plus_residual, _, _, _ = self._warp_global_mhe_residual_vector(
                    wide_plus
                )
                wide_minus_residual, _, _, _ = self._warp_global_mhe_residual_vector(
                    wide_minus
                )
                expected = center_residual.shape
                if any(
                    value.shape != expected
                    for value in (
                        plus_residual,
                        minus_residual,
                        wide_plus_residual,
                        wide_minus_residual,
                    )
                ):
                    raise RuntimeError(
                        "Observable Warp residual changed dimension across FD probes"
                    )
                jacobian_columns.append(
                    (plus_residual - minus_residual) / (2.0 * epsilon)
                )
                wide_columns.append(
                    (wide_plus_residual - wide_minus_residual) / (4.0 * epsilon)
                )
            jacobian = torch.stack(jacobian_columns, dim=1)
            wide_jacobian = torch.stack(wide_columns, dim=1)
            step, allowed, reason, metrics = updater.observable_lm_step(
                center_coefficients=center,
                residual=center_residual,
                jacobian=jacobian,
                wide_jacobian=wide_jacobian,
                short_residual_count=short_residual_count,
            )
            metrics.update(
                observable_transition_count=len(transitions),
                observable_used_transition_count=int(used_transitions),
                observable_used_short_transition_count=int(
                    used_short_transitions
                ),
                observable_short_residual_count=int(short_residual_count),
                observable_short_horizon_weight=float(
                    settings.observable_short_horizon_weight
                ),
                observable_long_horizon_weight=float(
                    settings.observable_long_horizon_weight
                ),
                observable_segment_count=int(segments),
                observable_window_size=int(settings.observable_window_size),
                observable_latest_frame=int(latest_frame),
                observable_fd_log_epsilon=epsilon,
                observable_policy=(
                    "causal_long_primary_multishooting_warp_fd_lm;"
                    "h1_fd_uncertainty_hessian_constraint;distance+damping;"
                    "shape+volume+grasp_fixed;no_future;no_branch"
                ),
            )
            return step, allowed, reason, metrics
        finally:
            sim.copy_embodied_gaussian_rollout_state(live)
            with torch.no_grad():
                updater.distance_stiffness.copy_(live_distance)
                updater.shape_stiffness.copy_(live_shape)
            self.environment.physics_settings.particle_velocity_damping_per_second = (
                live_damping
            )
            self._sim_grasp_coupling_gain = live_coupling
            self._apply_stiffness_tool_command(live_command)
            sim.update_gaussian_transforms()

    def _process_flow_depth_trajectory_frame(self) -> dict | None:
        """每个视频帧至多提交一次 CoTracker+数据集深度的 q/qd 更新。"""

        if self.visual_feedback_mode != "trajectory":
            return None
        frame_index = int(self.current_frame_index)
        if frame_index <= self._last_flow_depth_runtime_frame_index:
            return None
        self._last_flow_depth_runtime_frame_index = frame_index
        assert self.flow_depth_bindings is not None
        assert self.flow_depth_observations is not None
        assert self.visual_residual_mapper is not None
        pair_index = self.flow_depth_observations.pair_index_for_next_frame(
            frame_index
        )
        if pair_index is None:
            return None
        source_frame = int(
            self.flow_depth_observations.current_source_frames[pair_index]
        )
        if not (
            self._evaluation_feedback_allowed(source_frame)
            and self._evaluation_feedback_allowed(frame_index)
        ):
            return {"status": "withheld_by_causal_protocol"}
        current_positions = self._flow_depth_source_states.get(source_frame)
        source_snapshot = self._flow_depth_source_rollouts.get(source_frame)
        if current_positions is None or source_snapshot is None:
            return {"status": "missing_source_state"}

        sim = self.environment.sim
        predicted_positions = wp.to_torch(sim.state_0.particle_q).detach().clone()
        predicted_velocities = wp.to_torch(sim.state_0.particle_qd).detach().clone()
        inverse_masses = (
            wp.to_torch(sim.model.particle_inv_mass)
            .detach()
            .cpu()
            .numpy()
            .copy()
        )
        control_exclusion_mask = grip_control_exclusion_mask(
            self.visual_residual_mapper, self.environment
        )
        observation_dt_s = float(
            self.playback_timestamps[frame_index]
            - self.playback_timestamps[source_frame]
        )
        track_observation = self.flow_depth_observations.observation(pair_index)
        update = compute_flow_depth_particle_state_update(
            bindings=self.flow_depth_bindings,
            observation=track_observation,
            current_positions=current_positions,
            predicted_positions=predicted_positions.cpu().numpy(),
            predicted_velocities=predicted_velocities.cpu().numpy(),
            particle_inverse_masses=inverse_masses,
            observation_dt_s=observation_dt_s,
            reference_range_centers=self._flow_depth_reference_range_centers,
            dynamic_exclusion_mask=control_exclusion_mask.detach().cpu().numpy(),
            settings=self.flow_depth_settings,
        )
        accepted = sim.apply_flow_depth_particle_state_update(
            update,
            mapper=self.visual_residual_mapper,
            maximum_penetration_m=None,
            maximum_backtracks=0,
            maximum_local_inversion_projection_passes=16,
            enforce_inversion_gate=False,
            enforce_low_volume_gate=False,
            enforce_anchor_gate=False,
            enforce_penetration_gate=False,
        )
        metrics = dict(sim.last_flow_depth_particle_update_metrics)
        self._flow_depth_update_count += 1
        metrics.update(
            status="accepted" if accepted else "rejected",
            update_count=self._flow_depth_update_count,
            source_frame=source_frame,
            destination_frame=frame_index,
            depth_source=self.flow_depth_source,
            rgb_motion_source="CoTracker3",
        )

        stiffness_metrics = None
        if accepted and self.stiffness_updater is not None:
            corrected_positions = wp.to_torch(sim.state_0.particle_q).detach().clone()
            accepted_residual = corrected_positions - predicted_positions
            quality_valid_mask = stiffness_local_quality_mask(
                self.visual_residual_mapper,
                predicted_positions,
                corrected_positions,
                STIFFNESS_UPDATE_LOCAL_MINIMUM_VOLUME_RATIO,
            )
            confidence_support = torch.as_tensor(
                update.particle_confidence_support,
                device=corrected_positions.device,
                dtype=corrected_positions.dtype,
            )
            supervision_valid_mask = (
                (confidence_support > 0.0)
                & (torch.linalg.vector_norm(accepted_residual, dim=1) > 0.0)
            )
            if self._flow_depth_reference_range_centers is None:
                raise RuntimeError("Missing fixed CoTracker reference centres")
            track_target_positions = torch.as_tensor(
                self._flow_depth_reference_range_centers
                + (
                    track_observation.next_points_table
                    - self.flow_depth_bindings.initial_points_table
                ),
                device=corrected_positions.device,
                dtype=corrected_positions.dtype,
            )
            track_valid_mask = torch.as_tensor(
                update.track_valid,
                device=corrected_positions.device,
                dtype=torch.bool,
            )
            contact_metrics = (
                self._sim_grasp_contact_metrics()
                if self._sim_reconstruction_mode
                else self.environment.sim.triangle_skin_contact_metrics()
            ) or {}
            (
                stiffness_gate_paused,
                stiffness_gate_reason,
                _stiffness_jaw_speed,
                stiffness_grip_active,
            ) = self._stiffness_global_gate(contact_metrics)
            direct_control_mask = torch.zeros_like(control_exclusion_mask)
            configured_direct_control = getattr(
                self.environment, "super_sim_grasp_control_mask", None
            )
            if stiffness_grip_active and configured_direct_control is not None:
                direct_control_mask = configured_direct_control.detach().to(
                    device=control_exclusion_mask.device, dtype=torch.bool
                )
            if not stiffness_gate_paused:
                projector = self.environment.sim.material_projector
                coupling_base_positions = None
                coupling_displacements = None
                if (
                    self.stiffness_updater.settings.update_mode
                    in {
                        "differentiable_global",
                        "differentiable_global_relative",
                        "differentiable_global_mhe",
                        "differentiable_local_relative",
                        "differentiable_hierarchical_relative",
                        "differentiable_particle_graph_lm",
                    }
                    and self._sim_grasp_active
                    and self._sim_grasp_particle_ids is not None
                    and self._sim_grasp_capture_positions is not None
                    and self._sim_grasp_trajectory_origin is not None
                    and self._sim_grasp_trajectory_positions is not None
                ):
                    coupling_base_positions = corrected_positions.detach().clone()
                    coupling_displacements = torch.zeros_like(corrected_positions)
                    ids = self._sim_grasp_particle_ids
                    coupling_base_positions[ids] = self._sim_grasp_capture_positions
                    coupling_displacements[ids] = (
                        self._sim_grasp_trajectory_positions[frame_index]
                        - self._sim_grasp_trajectory_origin
                    )
                candidate = self.stiffness_updater.propose(
                    physical_prediction=predicted_positions,
                    accepted_residual=accepted_residual,
                    quality_valid_mask=quality_valid_mask,
                    supervision_valid_mask=supervision_valid_mask,
                    control_exclusion_mask=control_exclusion_mask,
                    control_frozen_mask=direct_control_mask,
                    control_coupling_base_positions=coupling_base_positions,
                    control_coupling_displacements=coupling_displacements,
                    track_target_positions=track_target_positions,
                    track_valid_mask=track_valid_mask,
                    rollout_start_positions=torch.as_tensor(
                        source_snapshot.positions,
                        device=predicted_positions.device,
                        dtype=predicted_positions.dtype,
                    ),
                    physical_velocities=source_snapshot.velocities,
                    frame_index=frame_index,
                )
                if (
                    self.stiffness_updater.settings.update_mode
                    in {
                        "differentiable_global",
                        "differentiable_global_relative",
                        "differentiable_global_mhe",
                        "differentiable_local_relative",
                        "differentiable_hierarchical_relative",
                        "differentiable_particle_graph_lm",
                    }
                ):
                    self._append_warp_gradient_transition(
                        source_frame=source_frame,
                        source_snapshot=source_snapshot,
                        corrected_positions=corrected_positions,
                        candidate=candidate,
                        control_exclusion_mask=control_exclusion_mask,
                        track_target_positions=track_target_positions,
                        track_valid_mask=track_valid_mask,
                    )
                update_mode = self.stiffness_updater.settings.update_mode
                if update_mode == "differentiable_global_mhe":
                    (
                        mhe_step,
                        mhe_allowed,
                        mhe_reason,
                        mhe_diagnostics,
                    ) = self._warp_global_mhe_lm_proposal()
                    candidate.metrics.update(mhe_diagnostics)
                    if not mhe_allowed:
                        stiffness_metrics = self.stiffness_updater.reject(
                            mhe_reason, candidate
                        )
                    else:
                        self.stiffness_updater.replace_candidate_step_with_observable_lm(
                            candidate,
                            mhe_step,
                            mhe_diagnostics,
                        )
                        candidate_step_maximum = (
                            self.stiffness_updater.candidate_parameter_step_maximum(
                                candidate
                            )
                        )
                        if candidate_step_maximum <= 1.0e-12:
                            stiffness_metrics = self.stiffness_updater.reject(
                                "observable_lm_zero_parameter_step", candidate
                            )
                        else:
                            stiffness_metrics = self.stiffness_updater.commit(
                                candidate
                            )
                            self.environment.physics_settings.particle_velocity_damping_per_second = float(
                                self.stiffness_updater.global_velocity_damping_per_second
                            )
                            self._sim_grasp_coupling_gain = float(
                                self.stiffness_updater.global_coupling_gain
                            )
                            stiffness_metrics.update(
                                validation_status="causal_observable_lm_commit",
                                update_policy=(
                                    "global_paper_distance+damping;"
                                    "shape+volume+coupling_fixed;"
                                    "20transition_long_primary_multishooting;"
                                    "h1_fd_uncertainty_hessian_constraint;"
                                    "warp_fd_eps_and_2eps;"
                                    "svd_observability_gate;robust_lm;"
                                    "no_future;no_branch_selection"
                                ),
                            )
                else:
                    candidate_step_maximum = (
                        self.stiffness_updater.candidate_parameter_step_maximum(
                            candidate
                        )
                    )
                    if candidate_step_maximum == 0.0:
                        stiffness_metrics = self.stiffness_updater.reject(
                            (
                                "no_active_local_zero_mean_parameter_step"
                                if update_mode in {
                                    "differentiable_local_relative",
                                    "differentiable_hierarchical_relative",
                                    "differentiable_particle_graph_lm",
                                }
                                else "no_active_distance_damping_parameter_step"
                            )
                            if update_mode
                            in {
                                "differentiable_global",
                                "differentiable_global_relative",
                                "differentiable_local_relative",
                                "differentiable_hierarchical_relative",
                                "differentiable_particle_graph_lm",
                            }
                            else "no_active_material_signal",
                            candidate,
                        )
                    elif update_mode == "differentiable_particle_graph_lm":
                        autograd_gradient = candidate.autograd_parameter_gradient
                        if autograd_gradient is None:
                            stiffness_metrics = self.stiffness_updater.reject(
                                "missing_particle_graph_autograd_gradient",
                                candidate,
                            )
                        else:
                            (
                                predicted_directional,
                                warp_directional,
                            ) = self._warp_particle_graph_directional_derivatives(
                                autograd_gradient
                            )
                            gradient_allowed, _gradient_cosine = (
                                self.stiffness_updater.apply_particle_graph_directional_gate(
                                    candidate,
                                    predicted_directional,
                                    warp_directional,
                                )
                            )
                            if not gradient_allowed:
                                stiffness_metrics = self.stiffness_updater.reject(
                                    "particle_graph_directional_cosine_below_"
                                    f"{self.stiffness_updater.settings.graph_directional_cosine_minimum:.2f}",
                                    candidate,
                                )
                            else:
                                self.stiffness_updater.replace_candidate_step_with_particle_graph_lm(
                                    candidate
                                )
                                stiffness_metrics = self.stiffness_updater.commit(
                                    candidate
                                )
                                stiffness_metrics.update(
                                    validation_status=(
                                        "causal_particle_graph_lm_commit"
                                    ),
                                    update_policy=(
                                        "fixed_initial_distance_0.20;"
                                        "1global+one_local_log_distance_per_particle;"
                                        "weak_tied_shape+fixed_volume+damping+coupling;"
                                        "203_cotracker_cauchy_relative_h1_h3_h5;"
                                        "observation_weighted_graph_diagonal_gn_lm;"
                                        "deterministic_warp_directional_fd_cosine>=0.90;"
                                        "no_future_or_gt;no_parameter_selection"
                                    ),
                                )
                    elif update_mode in {
                        "differentiable_global",
                        "differentiable_global_relative",
                        "differentiable_local_relative",
                        "differentiable_hierarchical_relative",
                    }:
                        (
                            warp_gradient,
                            warp_short_gradient,
                            warp_long_gradient,
                        ) = self._warp_global_parameter_finite_difference_gradient()
                        gradient_allowed, _gradient_cosine = (
                            self.stiffness_updater.apply_gradient_consistency_gate(
                                candidate, warp_gradient
                            )
                        )
                        if not gradient_allowed:
                            stiffness_metrics = self.stiffness_updater.reject(
                                "autograd_warp_gradient_cosine_below_"
                                f"{self.stiffness_updater.settings.gradient_cosine_minimum:.2f}",
                                candidate,
                            )
                        else:
                            hierarchical_mode = (
                                update_mode
                                == "differentiable_hierarchical_relative"
                            )
                            selected_gradient = (
                                warp_long_gradient
                                if hierarchical_mode
                                else warp_gradient
                            )
                            if hierarchical_mode:
                                short_long_denominator = (
                                    torch.linalg.vector_norm(warp_short_gradient)
                                    * torch.linalg.vector_norm(warp_long_gradient)
                                )
                                short_long_cosine = (
                                    float(
                                        torch.dot(
                                            warp_short_gradient,
                                            warp_long_gradient,
                                        ).item()
                                    )
                                    / float(short_long_denominator.item())
                                    if float(short_long_denominator.item())
                                    > 1.0e-12
                                    else float("nan")
                                )
                            self.stiffness_updater.replace_candidate_step_with_warp_gradient(
                                candidate,
                                selected_gradient,
                                short_horizon_gradient=(
                                    warp_short_gradient
                                    if hierarchical_mode
                                    else None
                                ),
                            )
                            if hierarchical_mode:
                                candidate.metrics.update(
                                    warp_fd_short_long_gradient_cosine=float(
                                        short_long_cosine
                                    )
                                )
                            stiffness_metrics = self.stiffness_updater.commit(
                                candidate
                            )
                            self.environment.physics_settings.particle_velocity_damping_per_second = float(
                                self.stiffness_updater.global_velocity_damping_per_second
                            )
                            self._sim_grasp_coupling_gain = float(
                                self.stiffness_updater.global_coupling_gain
                            )
                            stiffness_metrics.update(
                                validation_status="causal_gradient_checked_commit",
                                update_policy=(
                                    "joint_global_mean+zero_mean_regional_paper_distance;"
                                    "damping+coupling_fixed;"
                                    "203_cotracker_cauchy_relative_h3_h5;"
                                    "warp_gradient_adam_h1_safe_projection;"
                                    "autograd_cosine>=0.95;"
                                    "no_future_or_gt;no_live_state_branch"
                                    if hierarchical_mode
                                    else
                                    "warp_gradient_adam;autograd_cosine>=0.95;"
                                    "no_future_or_gt;no_live_state_branch"
                                ),
                            )
                    else:
                        stiffness_metrics = self.stiffness_updater.commit(candidate)
                        stiffness_metrics.update(
                            validation_status="causal_flow_depth_commit",
                            update_policy=(
                                "online_flow_depth_innovation_plus_edge_strain;"
                                "no_branch_selection"
                            ),
                        )
            else:
                stiffness_metrics = {
                    "status": "paused",
                    "pause_reason": stiffness_gate_reason,
                }
        metrics["stiffness_status"] = str(
            (stiffness_metrics or {}).get("status", "off")
        )
        if (
            self._flow_depth_update_count <= 3
            or self._flow_depth_update_count % 30 == 1
            or not accepted
        ):
            print(
                "[sim flow-depth trajectory] "
                f"frame={source_frame}->{frame_index}, accepted={accepted}, "
                f"tracks={int(update.track_valid.sum())}, "
                f"particles={int(metrics['updated_particles'])}, "
                "max="
                f"{float(metrics['maximum_position_correction_m']) * 1e3:.3f}mm, "
                f"stiffness={metrics['stiffness_status']}"
            )
        return metrics

    async def run_physics(self):
        dt = self.environment.dt()
        while True:
            if (
                self._evaluation_max_physics_iterations is not None
                and self._physics_iteration_count
                >= self._evaluation_max_physics_iterations
            ):
                await trio.sleep(dt)
                continue
            # XPBD eval_ik restores articulation FK, so reapply the strict
            # timestamp-aligned LND pose after every physics step.
            terminal_evaluation_step = bool(
                self._active_stiffness_evaluations
                and not self.playing
                and self.current_frame_index
                == len(self.playback_timestamps) - 1
            )
            if (
                self._active_stiffness_evaluations
                and not self.playing
                and not terminal_evaluation_step
            ):
                self._record_incomplete_stiffness_evaluations(
                    "playback_paused"
                )
                self._active_stiffness_evaluations.clear()
            command = None
            if (self.playing or terminal_evaluation_step) and (
                self._pending_stiffness_validation is not None
                or self._active_stiffness_evaluations
            ):
                command = self._current_stiffness_tool_command()
            if self._pending_stiffness_validation is not None:
                if not self.playing:
                    assert self.stiffness_updater is not None
                    pending = self._pending_stiffness_validation
                    metrics = self.stiffness_updater.reject(
                        "playback_paused_before_validation",
                        pending.candidate,
                    )
                    metrics.update(validation_status="cancelled")
                    self._record_terminal_stiffness_validation(
                        pending, metrics, "playback_paused_before_validation"
                    )
                    self._pending_stiffness_validation = None
                    self._last_stiffness_validation_metrics = metrics
                else:
                    assert command is not None
                    self._pending_stiffness_validation.commands.append(command)
            if command is not None:
                for evaluation in self._active_stiffness_evaluations:
                    evaluation.commands.append(command)
            if (
                bool(
                    getattr(
                        self.environment,
                        "super_sim_reconstruction_mode",
                        False,
                    )
                )
                and not self.playing
                and not terminal_evaluation_step
                and self._pending_stiffness_validation is None
                and not self._active_stiffness_evaluations
                and not self._single_visual_residual_requested
            ):
                # This mode has zero gravity and intentionally disables the
                # legacy PSM contact.  Idle stepping cannot change the tissue,
                # but can starve ImGui and make its buttons appear frozen.
                await trio.sleep(dt)
                continue
            if self._sim_reconstruction_mode:
                # Only a five-node patch centered on the red visual marker
                # follows its prescribed simulation trajectory while the jaw is
                # closed. There is no inferred tool-contact patch or external
                # spring; every remaining tissue node is determined by PBD,
                # RGB residual, and the online local-stiffness field.
                self._project_sim_grasp_boundary(update_gaussians=False)
            self.environment.step(compute_visual_forces=False)
            if self._sim_reconstruction_mode:
                self._project_sim_grasp_boundary(update_gaussians=True)
            self._physics_iteration_count += 1
            self.apply_current_psm_pose()
            evaluation_contact_metrics = None
            if self.stiffness_metrics_recorder is not None:
                evaluation_contact_metrics = (
                    self._sim_grasp_contact_metrics()
                    if self._sim_reconstruction_mode
                    else self.environment.sim.triangle_skin_contact_metrics()
                )
                self._update_action_phase(evaluation_contact_metrics)
            if (
                self.playing
                and self.environment.frames is not None
                and self.visual_residual_mapper is not None
                and self.current_frame_index
                > self._last_trajectory_observation_frame_index
            ):
                if self.stiffness_metrics_recorder is not None:
                    self.environment.sim.update_gaussian_transforms()
                    alignment = (
                        self.environment.sim.evaluate_visual_tissue_alignment(
                            self.visual_residual_mapper,
                            self.environment.frames,
                            observations_are_bgr=True,
                        )
                    )
                    self._record_trajectory_observation(
                        alignment=alignment,
                        contact_metrics=evaluation_contact_metrics,
                    )
                self._last_trajectory_observation_frame_index = int(
                    self.current_frame_index
                )
            if terminal_evaluation_step:
                self._advance_committed_stiffness_evaluations()
                self._record_incomplete_stiffness_evaluations(
                    "dataset_finished"
                )
                self._active_stiffness_evaluations.clear()
            self._visual_force_step += 1
            if (
                self.playing
                and self.environment.frames is not None
                and self.visual_feedback_mode == "trajectory"
                and not self._defer_flow_depth_update_until_frame_end
            ):
                self._process_flow_depth_trajectory_frame()
            visual_feedback_enabled = (
                (self.playing or self._single_visual_residual_requested)
                and self.environment.frames is not None
                and self.visual_feedback_mode != "off"
                and self._evaluation_feedback_allowed()
            )
            update_visual_feedback = (
                self._single_visual_residual_requested
                or self._visual_force_step % self.visual_force_update_interval == 0
            )
            if (
                visual_feedback_enabled
                and update_visual_feedback
                and self.visual_feedback_mode == "residual"
            ):
                assert self.visual_residual_mapper is not None
                solve_start = time.perf_counter()
                contact_metrics = evaluation_contact_metrics or (
                    self._sim_grasp_contact_metrics()
                    if self._sim_reconstruction_mode
                    else self.environment.sim.triangle_skin_contact_metrics()
                )
                (
                    stiffness_gate_paused,
                    stiffness_gate_reason,
                    stiffness_jaw_speed,
                    stiffness_grip_active,
                ) = self._stiffness_global_gate(contact_metrics)
                stiffness_metrics = self._validate_pending_stiffness(
                    gate_paused=stiffness_gate_paused,
                    gate_reason=stiffness_gate_reason,
                    grip_active=stiffness_grip_active,
                )
                self._advance_committed_stiffness_evaluations()
                # One image timestamp may span several physics iterations.
                # Re-solving the same observation would repeatedly learn from
                # stale evidence and can immediately recreate an expired
                # candidate.  Material learning advances only with images.
                if (
                    self.current_frame_index
                    <= self._last_visual_residual_frame_index
                    and not self._single_visual_residual_requested
                ):
                    await trio.sleep(dt)
                    continue
                control_exclusion_mask = grip_control_exclusion_mask(
                    self.visual_residual_mapper,
                    self.environment,
                )
                direct_control_mask = torch.zeros_like(
                    control_exclusion_mask
                )
                configured_direct_control = getattr(
                    self.environment,
                    "super_sim_grasp_control_mask",
                    None,
                )
                if stiffness_grip_active and configured_direct_control is not None:
                    direct_control_mask = configured_direct_control.detach().to(
                        device=control_exclusion_mask.device,
                        dtype=torch.bool,
                    )
                    if direct_control_mask.shape != control_exclusion_mask.shape:
                        raise ValueError(
                            "Dataset direct grasp mask has the wrong shape"
                        )
                self._visual_residual_busy = True
                # Let the GUI display the busy state before the synchronous
                # CUDA optimization begins.
                await trio.sleep(0)
                try:
                    result = self.environment.sim.solve_visual_tissue_residual(
                        self.visual_residual_mapper,
                        self.environment.frames,
                        previous_residual=self._previous_visual_residual,
                        dynamic_exclusion_mask=control_exclusion_mask,
                        observations_are_bgr=True,
                    )
                finally:
                    self._visual_residual_busy = False
                pre_visual_velocity = (
                    wp.to_torch(self.environment.sim.state_0.particle_qd)
                    .detach()
                    .clone()
                )
                accepted = self.environment.sim.apply_visual_tissue_residual(
                    result,
                    mapper=self.visual_residual_mapper,
                    frames=self.environment.frames,
                    observations_are_bgr=True,
                )
                self._last_visual_residual_frame_index = int(
                    self.current_frame_index
                )
                self._single_visual_residual_requested = False
                if not accepted and self.stiffness_updater is not None:
                    self.stiffness_updater.invalidate_signal_history()
                if accepted and self.stiffness_updater is not None:
                    physical_prediction = (
                        result.corrected_positions - result.residual
                    )
                    quality_valid_mask = stiffness_local_quality_mask(
                        self.visual_residual_mapper,
                        physical_prediction,
                        result.corrected_positions,
                        STIFFNESS_UPDATE_LOCAL_MINIMUM_VOLUME_RATIO,
                    )
                    supervision_valid_mask = (
                        stiffness_visual_supervision_mask(
                            self.visual_residual_mapper,
                            result.visual_gradient_norm,
                        )
                    )
                    if (
                        not stiffness_gate_paused
                        and self._pending_stiffness_validation is None
                    ):
                        candidate = self.stiffness_updater.propose(
                            physical_prediction=physical_prediction,
                            accepted_residual=result.residual,
                            quality_valid_mask=quality_valid_mask,
                            supervision_valid_mask=(
                                supervision_valid_mask
                            ),
                            control_exclusion_mask=(
                                control_exclusion_mask
                            ),
                            control_frozen_mask=direct_control_mask,
                            rollout_start_positions=(
                                wp.to_torch(
                                    self.environment.sim.material_projector
                                    .predicted_positions
                                )
                                .detach()
                                .clone()
                                if self.environment.sim.material_projector
                                is not None
                                else physical_prediction
                            ),
                            physical_velocities=(
                                pre_visual_velocity
                            ),
                            frame_index=int(self.current_frame_index),
                        )
                        candidate_step_maximum = (
                            self.stiffness_updater.candidate_parameter_step_maximum(
                                candidate
                            )
                        )
                        if candidate_step_maximum == 0.0:
                            stiffness_metrics = self.stiffness_updater.reject(
                                "no_active_distance_damping_parameter_step"
                                if self.stiffness_updater.settings.update_mode
                                in {
                                    "differentiable_global",
                                    "differentiable_global_relative",
                                    "differentiable_global_mhe",
                                    "differentiable_local_relative",
                                    "differentiable_hierarchical_relative",
                                    "differentiable_particle_graph_lm",
                                }
                                else "no_active_material_signal",
                                candidate,
                            )
                        elif (
                            self.stiffness_updater.settings.update_mode
                            in {
                                "differentiable_global",
                                "differentiable_global_relative",
                                "differentiable_global_mhe",
                                "differentiable_local_relative",
                                "differentiable_hierarchical_relative",
                                "differentiable_particle_graph_lm",
                            }
                        ):
                            # This path has no saved pre-transition Warp state,
                            # so it cannot satisfy the mandatory FD direction
                            # check. Never silently commit an unchecked global
                            # parameter update from the interactive RGB solver.
                            stiffness_metrics = self.stiffness_updater.reject(
                                "warp_fd_gate_requires_trajectory_mode",
                                candidate,
                            )
                        elif self._sim_reconstruction_mode:
                            # Synthetic RGB benchmark uses a strictly causal
                            # online update: accepted frame t strain evidence
                            # changes only the material used by later physics
                            # steps.  No H-frame shadow branch, line search, or
                            # future observation is needed, so a continuous
                            # surgical stream is never paused for selection.
                            stiffness_metrics = self.stiffness_updater.commit(
                                candidate
                            )
                            stiffness_metrics.update(
                                validation_status="causal_rgb_commit",
                                update_policy=(
                                    "online_edge_strain_plus_visual_residual;"
                                    "no_branch_selection"
                                ),
                            )
                        else:
                            (
                                history_baseline,
                                history_candidate,
                                history_safe,
                            ) = self._evaluate_stiffness_history(candidate)
                            candidate.metrics.update(
                                history_baseline_rms_m=history_baseline,
                                history_candidate_rms_m=history_candidate,
                            )
                            if not history_safe:
                                stiffness_metrics = (
                                    self.stiffness_updater.reject(
                                        "history_regression", candidate
                                    )
                                )
                            else:
                                self._pending_stiffness_validation = (
                                    PendingStiffnessValidation(
                                        candidate=candidate,
                                        rollout_state=(
                                            self.environment.sim
                                            .clone_embodied_gaussian_rollout_state()
                                        ),
                                        frame_index=int(
                                            self.current_frame_index
                                        ),
                                        grip_active=stiffness_grip_active,
                                        history_baseline_rms_m=(
                                            history_baseline
                                        ),
                                        history_candidate_rms_m=(
                                            history_candidate
                                        ),
                                        previous_residual=(
                                            result.residual.detach().clone()
                                        ),
                                    )
                                )
                                stiffness_metrics = candidate.metrics
                    self._maybe_store_stiffness_history(
                        accepted_positions=result.corrected_positions,
                        jaw_speed_rad_s=stiffness_jaw_speed,
                        gate_paused=stiffness_gate_paused,
                    )
                self._visual_residual_solve_count += 1
                metrics = (
                    self.environment.sim.last_visual_tissue_residual_metrics
                )
                assert metrics is not None
                metrics["frame_index"] = int(self.current_frame_index)
                metrics["solve_elapsed_s"] = time.perf_counter() - solve_start
                metrics["solve_count"] = self._visual_residual_solve_count
                self._previous_visual_residual = (
                    result.residual.detach().clone() if accepted else None
                )
                self._record_visual_stiffness_metrics(
                    result=result,
                    visual_metrics=metrics,
                    stiffness_metrics=stiffness_metrics,
                    contact_metrics=contact_metrics,
                    gate_paused=stiffness_gate_paused,
                    gate_reason=stiffness_gate_reason,
                )
                acceptance_changed = (
                    self._last_visual_residual_accepted is None
                    or accepted != self._last_visual_residual_accepted
                )
                self._last_visual_residual_accepted = accepted
                if acceptance_changed or self._visual_residual_solve_count % 30 == 0:
                    stiffness_summary = (
                        "stiffness_med="
                        f"{stiffness_metrics['distance_median']:.3f}/"
                        f"{stiffness_metrics['shape_median']:.4f}, "
                        if stiffness_metrics is not None
                        else ""
                    )
                    print(
                        "[visual residual realtime] "
                        f"accepted={accepted}, "
                        f"loss={result.initial_visual_loss:.6f}->"
                        f"{float(metrics['exact_final_visual_loss']):.6f}, "
                        f"max={result.maximum_residual_m * 1e3:.3f}mm, "
                        "min_volume="
                        f"{result.initial_minimum_volume_ratio:.3f}->"
                        f"{result.minimum_volume_ratio:.3f}, "
                        "grip_excluded="
                        f"{result.dynamically_excluded_particles}, "
                        f"local_frozen={result.locally_frozen_particles}, "
                        f"backtracks={result.backtrack_count}, "
                        "stiffness_quality_masked="
                        f"{int(stiffness_metrics['quality_masked_particles']) if stiffness_metrics is not None else 0}, "
                        "stiffness_status="
                        f"{str(stiffness_metrics.get('status', 'paused')) if stiffness_metrics is not None else ('paused' if stiffness_gate_paused else 'off')}, "
                        f"{stiffness_summary}"
                        f"elapsed={metrics['solve_elapsed_s'] * 1e3:.1f}ms"
                    )
            elif (
                visual_feedback_enabled
                and update_visual_feedback
                and self.visual_feedback_mode == "force"
            ):
                self.environment.sim.compute_visual_forces(
                    self.environment.visual_forces_settings,
                    self.environment.frames,
                    self.environment.physics_settings.dt
                    / self.environment.physics_settings.substeps,
                )
            elif visual_feedback_enabled and self.visual_feedback_mode == "force":
                self.environment.sim.reapply_last_soft_visual_forces()
            self.maybe_print_psm_base_q()
            self.maybe_print_tissue_q()
            await trio.sleep(dt)

    async def run(self):
        async with trio.open_nursery() as nursery:
            nursery.start_soon(self.run_physics)
            next_frame_deadline: float | None = None
            while True:
                if self.playing:
                    if (
                        self.visual_feedback_mode == "residual"
                        and self._evaluation_feedback_allowed()
                        and self.current_frame_index
                        > self._last_visual_residual_frame_index
                    ):
                        # Do not advance past an observation before its visual
                        # correction (and consequent stiffness proposal) has
                        # run.  With CUDA solve time near the requested frame
                        # period, deadline catch-up otherwise lets video frames
                        # outrun the estimator and silently reduces updates.
                        await trio.sleep(1.0 / 240.0)
                        continue
                    self.capture_flow_depth_source_state(
                        self.current_frame_index
                    )
                    now = trio.current_time()
                    if next_frame_deadline is None:
                        next_frame_deadline = now
                    self.advance_one_frame()
                    next_frame_deadline += 1.0 / max(self.fps, 1)
                    # Count frame decoding/GUI work inside the requested frame
                    # period. The old loop slept a full period after the work,
                    # which made playback slower than the FPS slider even when
                    # the machine had enough capacity.
                    await trio.sleep_until(
                        max(next_frame_deadline, trio.current_time())
                    )
                else:
                    next_frame_deadline = None
                    await trio.sleep(1.0 / 60.0)


async def run_headless_trajectory_evaluation(
    playback_controls: SuperPlaybackControls,
    *,
    start_frame: int,
    frame_count: int,
    physics_steps_per_frame: int,
    benchmark_output: Path | None = None,
    evaluation_label: str = "unnamed",
    render_images: bool = True,
    open_loop_start_frame: int | None = None,
    holdout_stride: int | None = None,
    holdout_offset: int = 0,
    render_frame_mode: str = "all",
) -> None:
    """Run an exact frame/physics-step schedule without constructing a GUI."""
    total_frames = len(playback_controls.playback_timestamps)
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError(
            f"Evaluation start frame {start_frame} is outside 0..{total_frames - 1}"
        )
    if frame_count < 0:
        raise ValueError("Evaluation frame count cannot be negative")
    if physics_steps_per_frame < 1:
        raise ValueError("Evaluation physics steps per frame must be positive")
    if (
        playback_controls.visual_feedback_mode == "residual"
        and physics_steps_per_frame
        < playback_controls.visual_force_update_interval
    ):
        raise ValueError(
            "Headless residual evaluation needs at least one complete visual "
            "update interval per video frame: physics_steps_per_frame="
            f"{physics_steps_per_frame} < update_interval="
            f"{playback_controls.visual_force_update_interval}"
        )
    end_frame_exclusive = (
        total_frames
        if frame_count == 0
        else min(total_frames, start_frame + frame_count)
    )
    if open_loop_start_frame is not None and not (
        start_frame <= open_loop_start_frame <= end_frame_exclusive
    ):
        raise ValueError(
            "Open-loop start frame must lie inside the evaluated frame interval"
        )
    if holdout_stride is not None:
        if holdout_stride < 2:
            raise ValueError("Reconstruction holdout stride must be at least 2")
        if not 0 <= holdout_offset < holdout_stride:
            raise ValueError("Reconstruction holdout offset is outside its stride")
        if open_loop_start_frame is not None:
            raise ValueError(
                "7:1 reconstruction holdout and 80/20 future split are "
                "separate capabilities and cannot share one run"
            )
    if render_frame_mode == "holdout" and holdout_stride is None:
        raise ValueError("Holdout-only rendering requires a holdout stride")
    if render_frame_mode == "future" and open_loop_start_frame is None:
        raise ValueError("Future-only rendering requires an open-loop split")
    playback_controls._evaluation_open_loop_start_frame = open_loop_start_frame
    playback_controls._evaluation_open_loop_entered = False
    playback_controls._evaluation_holdout_stride = holdout_stride
    playback_controls._evaluation_holdout_offset = int(holdout_offset)
    playback_controls._evaluation_holdout_last_cancelled_frame = -1
    playback_controls._evaluation_max_physics_iterations = (
        playback_controls._physics_iteration_count
    )
    artifact_writer = (
        SimBenchmarkArtifactWriter(
            playback_controls,
            benchmark_output,
            start_frame=start_frame,
            end_frame_exclusive=end_frame_exclusive,
            label=evaluation_label,
            render_images=render_images,
            open_loop_start_frame=open_loop_start_frame,
            physics_steps_per_frame=physics_steps_per_frame,
            holdout_stride=holdout_stride,
            holdout_offset=holdout_offset,
            render_frame_mode=render_frame_mode,
        )
        if benchmark_output is not None
        else None
    )
    playback_controls.go_to_frame(start_frame)
    playback_controls.environment.sim.sync_kinematic_body_interpolation()
    playback_controls.playing = True
    playback_controls._defer_flow_depth_update_until_frame_end = True
    print(
        "[trajectory evaluation] headless schedule: "
        f"frames={start_frame}..{end_frame_exclusive - 1}, "
        f"physics_steps_per_frame={physics_steps_per_frame}, "
        f"mode={playback_controls.visual_feedback_mode}, "
        "online_stiffness="
        f"{playback_controls.stiffness_updater is not None}, "
        f"open_loop_start={open_loop_start_frame}, "
        f"holdout_stride={holdout_stride}, "
        f"holdout_offset={holdout_offset}, "
        f"render_frame_mode={render_frame_mode}, "
        f"artifacts={benchmark_output}"
    )
    dt = max(float(playback_controls.environment.dt()), 1.0e-4)
    async with trio.open_nursery() as nursery:
        nursery.start_soon(playback_controls.run_physics)
        for frame_index in range(start_frame, end_frame_exclusive):
            if playback_controls.current_frame_index != frame_index:
                playback_controls.go_to_frame(frame_index)
            target_physics_iterations = (
                playback_controls._physics_iteration_count
                + physics_steps_per_frame
            )
            playback_controls._evaluation_max_physics_iterations = (
                target_physics_iterations
            )
            with trio.fail_after(900.0):
                while (
                    playback_controls._physics_iteration_count
                    < target_physics_iterations
                    or playback_controls._last_trajectory_observation_frame_index
                    < frame_index
                    or (
                        playback_controls.visual_feedback_mode == "residual"
                        and playback_controls._evaluation_feedback_allowed()
                        and playback_controls._last_visual_residual_frame_index
                        < frame_index
                    )
                ):
                    await trio.sleep(dt)
            if playback_controls.visual_feedback_mode == "trajectory":
                playback_controls._process_flow_depth_trajectory_frame()
            if artifact_writer is not None:
                artifact_writer.write_frame(frame_index)
            playback_controls.capture_flow_depth_source_state(frame_index)
        playback_controls.playing = False
        playback_controls._evaluation_max_physics_iterations = (
            playback_controls._physics_iteration_count
        )
        # Give run_physics one checkpoint to cancel a pending candidate and
        # persist incomplete open-loop horizons before shutdown.
        await trio.sleep(dt * 1.1)
        nursery.cancel_scope.cancel()
    if artifact_writer is not None:
        artifact_writer.close()
    print(
        "[trajectory evaluation] complete: "
        f"observed_frames={start_frame}..{end_frame_exclusive - 1}, "
        f"physics_iterations={playback_controls._physics_iteration_count}, "
        f"flow_depth_updates={playback_controls._flow_depth_update_count}"
    )


async def main(
    dataset_path: Path,
    fps: int,
    monitor_psm_base_q: bool,
    monitor_tissue_q: bool,
    monitor_interval: float,
    visual_feedback_mode: str,
    flow_depth_bindings_path: Path | None,
    flow_depth_observations_path: Path | None,
    flow_depth_position_gain: float,
    flow_depth_velocity_gain: float,
    flow_depth_absolute_position_weight: float,
    flow_depth_solver_regularization: float,
    flow_depth_solver_iterations: int,
    flow_depth_robust_residual_mm: float,
    flow_depth_maximum_position_correction_mm: float,
    flow_depth_maximum_velocity_correction_m_s: float,
    flow_depth_initial_alignment: bool,
    flow_depth_initial_alignment_maximum_mm: float,
    visual_force_iterations: int,
    visual_residual_iterations: int,
    visual_residual_learning_rate_m: float,
    visual_residual_maximum_mm: float | None,
    visual_residual_image_scale: float | None,
    visual_force_update_interval: int,
    online_stiffness_update: bool,
    sim_grasp_boundary_mode: str,
    stiffness_log_learning_rate: float,
    stiffness_update_mode: str,
    stiffness_strain_signal_weight: float,
    stiffness_autograd_unroll_steps: int,
    stiffness_autograd_region_count: int,
    stiffness_maximum_log_step: float | None,
    stiffness_signal_ema_decay: float | None,
    stiffness_spatial_smoothing_iterations: int | None,
    stiffness_spatial_smoothing_blend: float | None,
    initial_paper_distance_stiffness: float | None,
    initial_paper_shape_stiffness: float | None,
    stiffness_evaluation_output: Path | None,
    stiffness_evaluation_horizons: tuple[int, ...],
    evaluation_headless: bool,
    evaluation_start_frame: int,
    evaluation_frame_count: int,
    evaluation_physics_steps_per_frame: int,
    benchmark_output: Path | None,
    evaluation_label: str,
    evaluation_render_images: bool,
    evaluation_open_loop_start_frame: int | None,
    evaluation_holdout_stride: int | None,
    evaluation_holdout_offset: int,
    evaluation_render_frame_mode: str,
    camera_names: list[str],
    camera_go_zoom: float,
    psm_roll_offset_deg: float,
    psm_camera_translation_mm: tuple[float, float, float],
    psm_world_translation_mm: tuple[float, float, float],
    psm_pose_driver: str,
    psm_visual_mode: str,
    tissue_mode: str,
    psm_tissue_contact: bool,
    calibrated_profile: Path | None,
):
    pose_driver_path = PSM_POSE_DRIVER_PATHS[psm_pose_driver]
    dataset_tissue_asset = (
        dataset_path / "gui_assets" / "tissue_fixedsuperbest.npz"
    )
    sim_reconstruction_mode = (
        tissue_mode == "paper_pbd" and dataset_tissue_asset.is_file()
    )
    effective_psm_tissue_contact = (
        psm_tissue_contact and not sim_reconstruction_mode
    )
    environment = build_environment(
        psm_pose_driver_path=pose_driver_path,
        psm_visual_tip_only=psm_visual_mode == "tip",
        tissue_mode=tissue_mode,
        tissue_asset_path_override=(
            dataset_tissue_asset if sim_reconstruction_mode else None
        ),
        psm_visual_enabled=not sim_reconstruction_mode,
        include_scene_background=not sim_reconstruction_mode,
    )
    environment.super_sim_reconstruction_mode = sim_reconstruction_mode
    environment.super_sim_grasp_boundary_mode = sim_grasp_boundary_mode
    initial_material_override = (
        initial_paper_distance_stiffness is not None
        or initial_paper_shape_stiffness is not None
    )
    if initial_material_override:
        if tissue_mode != "paper_pbd":
            raise ValueError(
                "Initial paper stiffness overrides require --tissue-mode paper_pbd"
            )
        distance = (
            float(initial_paper_distance_stiffness)
            if initial_paper_distance_stiffness is not None
            else float(environment.physics_settings.paper_distance_stiffness)
        )
        shape = (
            float(initial_paper_shape_stiffness)
            if initial_paper_shape_stiffness is not None
            else float(environment.physics_settings.paper_shape_stiffness)
        )
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError("Initial paper distance stiffness must be positive")
        if not np.isfinite(shape) or shape <= 0.0:
            raise ValueError("Initial paper shape stiffness must be positive")
        physics_settings = environment.physics_settings
        physics_settings.paper_distance_stiffness = distance
        physics_settings.paper_shape_stiffness = shape
        material_projector = environment.sim.material_projector
        if material_projector is None:
            raise ValueError("Initial paper stiffness requires a material projector")
        material_projector.configure_constraint_model(
            physics_settings.tetrahedral_constraint_model,
            distance,
            physics_settings.paper_volume_stiffness,
            shape,
            preserve_spatial_stiffness=False,
        )
        print(
            "[example_embodied_super_offline] uniform initial material hypothesis: "
            f"distance={distance:g}, shape={shape:g}; regional GT=UNUSED"
        )
    if sim_reconstruction_mode:
        # The generated sheet contains very thin tetrahedra.  The legacy 0.01
        # material floor allowed a physics step to leave J~=0.02 before the
        # next visual correction.  Do not let ordinary PBD relaxation create
        # nearly collapsed elements that the image optimizer then has to
        # repair.  This changes only this copied simulation project.
        environment.physics_settings.material_min_volume_ratio = max(
            float(environment.physics_settings.material_min_volume_ratio),
            0.15,
        )
        print(
            "[example_embodied_super_offline] 仿真重建模式=ON；"
            f"组织资产={dataset_tissue_asset}；旧 SUPER 背景/PSM 高斯=OFF；"
            "旧 PSM 几何接触=OFF（避免混用不一致的 link 坐标）；"
            "material_min_J=0.15"
        )
    load_calibrated_tissue_profile(environment, calibrated_profile)
    print(
        "[example_embodied_super_offline] psm_pose_driver="
        f"{psm_pose_driver}: {pose_driver_path}; visual_mode={psm_visual_mode}; "
        f"tissue_mode={tissue_mode}"
    )
    print(
        "[example_embodied_super_offline] frozen manual correction="
        f"roll={psm_roll_offset_deg:+.3f} deg, "
        f"camera_translation_mm={list(psm_camera_translation_mm)}, "
        f"world_translation_mm={list(psm_world_translation_mm)}, "
        "jaw_mode=raw_q7_full_closure_with_3_to_5_particle_gate"
    )
    print(
        "[example_embodied_super_offline] embodied_gaussians_source="
        f"{LOADED_EMBODIED_GAUSSIANS_SOURCE}"
    )
    environment.visual_forces_settings.iterations = (
        visual_force_iterations if visual_feedback_mode == "force" else 0
    )
    visual_settings = environment.visual_forces_settings
    environment.super_tissue_residual_mapping_enabled = (
        visual_feedback_mode in {"residual", "trajectory"}
    )
    print(
        "[example_embodied_super_offline] visual_feedback_mode="
        f"{visual_feedback_mode}; force_iterations={visual_force_iterations}; "
        f"residual_iterations={visual_residual_iterations}; "
        f"update_interval={visual_force_update_interval}; PSM feedback disabled"
    )
    print(
        "[example_embodied_super_offline] tissue_visual_force_limits="
        f"lr_means={visual_settings.lr_means}, kp={visual_settings.kp}, "
        f"normalized={visual_settings.normalize_forces_by_gaussian_count}, "
        f"rigid_max_force={visual_settings.max_force}N, "
        f"soft_max_total_force={visual_settings.soft_max_total_force}N, "
        "soft_max_particle_acceleration="
        f"{visual_settings.soft_max_particle_acceleration}m/s^2, "
        f"soft_spread_layers={visual_settings.soft_force_spread_layers}, "
        f"max_moment={visual_settings.max_moment}Nm"
    )

    dataset_manager = DatasetManager(dataset_path)
    if hasattr(dataset_manager, 'keep_only_cameras'):
        dataset_manager.keep_only_cameras(camera_names)
    sim_mask_root = dataset_path / "gui_assets" / "visual_force_masks"
    sim_instrument_masks = dataset_path / "gui_assets" / "instrument_masks.npz"
    if sim_reconstruction_mode:
        camera_mask_assets = {
            camera: sim_mask_root / camera for camera in camera_names
        }
        missing_mask_assets = [
            str(path)
            for path in camera_mask_assets.values()
            if not (path / "report.json").is_file()
        ]
        if missing_mask_assets or not sim_instrument_masks.is_file():
            raise FileNotFoundError(
                "仿真 GUI mask 资产不完整；请先运行 "
                "scripts/prepare_sim_gui_masks.py。缺少："
                f"{missing_mask_assets} / {sim_instrument_masks}"
            )
        # The simulated tissue mask is already an exact visible-surface
        # semantic mask: occluded tool pixels are absent from it. Do not apply
        # the legacy SUPER tool-local SDF, which unnecessarily discarded much
        # of the non-contact tissue that should supervise residual/stiffness.
        instrument_mask_asset = None
    else:
        camera_mask_assets = {
            "stereo_left": VISUAL_FORCE_MASK_DIR,
            "stereo_right": RIGHT_VISUAL_FORCE_MASK_DIR,
        }
        instrument_mask_asset = VISUAL_FORCE_INSTRUMENT_MASKS
    visual_force_weights = MultiCameraPackedTissueVisualForceWeights(
        camera_mask_assets,
        erosion_radius_px=7,
        highlight_weight=0.10,
        instrument_mask_asset=instrument_mask_asset,
        tool_near_radius_px=120.0,
        tool_far_radius_px=360.0,
        tool_falloff_power=2.0,
        tissue_edge_zero_px=24.0,
        tissue_edge_full_px=64.0,
        tool_occlusion_radius_px=6.0,
        image_border_zero_px=48.0,
        image_border_full_px=96.0,
        posterior_full_reach_px=140.0,
        posterior_zero_reach_px=280.0,
    )
    if sim_reconstruction_mode:
        print(
            "[example_embodied_super_offline] visual residual mask="
            "exact full visible-tissue semantic mask; erosion=7px; "
            "legacy tool-local/posterior falloff=OFF"
        )
    else:
        print(
            "[example_embodied_super_offline] visual-force pixel field="
            "distal/jaw SDF near/full=120px, far/zero=360px; "
            "far-transition power=2; tissue edge zero/full=24px/64px; "
            "tool occlusion=6px; image border zero/full=48px/96px; "
            "posterior(-image-y) full/zero reach=140px/280px; "
            "q7/cache/Gaussian gates unchanged"
        )
    dataset_manager.set_visual_force_weight_provider(visual_force_weights)
    dataset_manager.update_frames(0.0)
    print(f"[example_embodied_super_offline] enabled_cameras={getattr(dataset_manager.frames, 'names', 'N/A')}")
    environment.frames = dataset_manager.frames

    flow_depth_bindings = None
    flow_depth_observations = None
    flow_depth_settings = None
    flow_depth_source = "not_applicable"
    flow_depth_initial_alignment_diagnostics = None
    if visual_feedback_mode == "trajectory":
        if flow_depth_bindings_path is None or flow_depth_observations_path is None:
            raise ValueError(
                "trajectory 模式必须同时指定 --flow-depth-bindings 和 "
                "--flow-depth-observations"
            )
        flow_depth_bindings = load_fixed_particle_range_bindings(
            flow_depth_bindings_path
        )
        with np.load(flow_depth_observations_path, allow_pickle=False) as loaded:
            flow_depth_source = (
                str(loaded["depth_source"].item())
                if "depth_source" in loaded.files
                else "unspecified_precomputed_depth"
            )
        flow_depth_observations = load_flow_depth_observation_sequence(
            flow_depth_observations_path
        )
        flow_depth_settings = FlowDepthStateUpdateSettings(
            position_gain=flow_depth_position_gain,
            velocity_gain=flow_depth_velocity_gain,
            absolute_position_weight=flow_depth_absolute_position_weight,
            solver_regularization=flow_depth_solver_regularization,
            solver_iterations=flow_depth_solver_iterations,
            robust_residual_scale_m=flow_depth_robust_residual_mm * 1.0e-3,
            maximum_position_correction_m=(
                flow_depth_maximum_position_correction_mm * 1.0e-3
            ),
            maximum_velocity_correction_m_s=(
                flow_depth_maximum_velocity_correction_m_s
            ),
        )
        flow_depth_settings.validate()
        print(
            "[example_embodied_super_offline] sim flow-depth trajectory: ON; "
            f"pairs={len(flow_depth_observations.current_source_frames)}, "
            f"tracks={int(flow_depth_bindings.track_valid.sum())}/"
            f"{len(flow_depth_bindings.track_valid)}, depth={flow_depth_source}, "
            f"position_gain={flow_depth_position_gain:g}, "
            f"velocity_gain={flow_depth_velocity_gain:g}, "
            f"absolute_weight={flow_depth_absolute_position_weight:g}, "
            "Gaussian=existing_triangle_binding"
        )
        if flow_depth_initial_alignment:
            flow_depth_initial_alignment_diagnostics = (
                align_sim_tissue_to_first_flow_depth(
                    environment,
                    tissue_asset=dataset_tissue_asset,
                    bindings=flow_depth_bindings,
                    depth_source=flow_depth_source,
                    maximum_translation_m=(
                        flow_depth_initial_alignment_maximum_mm * 1.0e-3
                    ),
                )
            )
    elif flow_depth_initial_alignment:
        raise ValueError(
            "--flow-depth-initial-alignment 只适用于 trajectory 模式"
        )

    visual_residual_mapper = None
    stiffness_updater = None
    if (
        visual_feedback_mode in {"residual", "trajectory"}
        or stiffness_evaluation_output is not None
        or benchmark_output is not None
    ):
        visual_residual_mapper = build_visual_tissue_residual_mapper(
            environment,
            iterations=visual_residual_iterations,
            learning_rate_m=visual_residual_learning_rate_m,
        )
    if visual_feedback_mode in {"residual", "trajectory"}:
        environment.sim.clear_soft_visual_force_cache()
        if online_stiffness_update:
            if tissue_mode != "paper_pbd":
                raise ValueError(
                    "Online paper stiffness update requires --tissue-mode paper_pbd"
                )
            material_projector = environment.sim.material_projector
            if material_projector is None:
                raise ValueError(
                    "Online stiffness update requires the material projector"
                )
            # The projector is constructed before the scene-specific paper
            # settings are installed.  Seed its per-particle arrays from the
            # actual simulation baseline before enabling preservation;
            # otherwise constructor defaults would be frozen instead of the
            # configured 0.31/0.0058 values printed by the GUI.
            physics_settings = environment.physics_settings
            if not dataset_manager.cameras:
                raise ValueError(
                    "Causal stiffness rollout requires camera timestamps"
                )
            stiffness_timestamps = np.asarray(
                dataset_manager.cameras[0].timestamps, dtype=np.float64
            )
            if len(stiffness_timestamps) < 2:
                raise ValueError(
                    "Causal stiffness rollout requires at least two frames"
                )
            stiffness_frame_dt = float(
                np.median(np.diff(stiffness_timestamps))
            )
            if not np.isfinite(stiffness_frame_dt) or stiffness_frame_dt <= 0.0:
                raise ValueError("Invalid camera frame interval")
            material_projector.configure_constraint_model(
                physics_settings.tetrahedral_constraint_model,
                physics_settings.paper_distance_stiffness,
                physics_settings.paper_volume_stiffness,
                physics_settings.paper_shape_stiffness,
                preserve_spatial_stiffness=False,
            )
            online_settings = OnlineTissueStiffnessSettings(
                log_learning_rate=stiffness_log_learning_rate,
                maximum_log_step=(
                    0.10
                    if stiffness_maximum_log_step is None
                    else stiffness_maximum_log_step
                ),
                signal_ema_decay=(
                    0.60
                    if stiffness_signal_ema_decay is None
                    else stiffness_signal_ema_decay
                ),
                spatial_smoothing_iterations=(
                    2
                    if stiffness_spatial_smoothing_iterations is None
                    else stiffness_spatial_smoothing_iterations
                ),
                spatial_smoothing_blend=(
                    0.30
                    if stiffness_spatial_smoothing_blend is None
                    else stiffness_spatial_smoothing_blend
                ),
                hardening_bias=0.0,
                strain_signal_weight=stiffness_strain_signal_weight,
                shape_maximum=0.020,
                update_mode=stiffness_update_mode,
                autograd_unroll_steps=stiffness_autograd_unroll_steps,
                autograd_region_count=stiffness_autograd_region_count,
                autograd_frame_dt=stiffness_frame_dt,
                # Match the authoritative Warp projector exactly for gradient
                # direction checks.  The former 6x2 Torch surrogate represented
                # only half a physics frame and one quarter of each material
                # projection, which was especially wrong for local fields.
                autograd_material_iterations=(
                    physics_settings.material_iterations
                ),
                autograd_integration_substeps=max(
                    1,
                    int(
                        round(stiffness_frame_dt / physics_settings.dt)
                    )
                    * physics_settings.substeps,
                ),
                autograd_control_projection_interval_substeps=(
                    physics_settings.substeps
                ),
                autograd_volume_stiffness=(
                    physics_settings.paper_volume_stiffness
                ),
                autograd_velocity_damping_per_second=(
                    physics_settings.particle_velocity_damping_per_second
                ),
                autograd_projection_velocity_scale=(
                    physics_settings.material_projection_velocity_scale
                ),
                autograd_pbd_dt=(
                    physics_settings.dt / physics_settings.substeps
                ),
                autograd_pbd_compliance_scale=(
                    physics_settings.material_compliance_scale
                ),
                autograd_pbd_relaxation=(
                    physics_settings.material_relaxation
                ),
            )
            stiffness_updater = ResidualDrivenPaperStiffnessUpdater(
                rest_positions=visual_residual_mapper.rest_positions,
                fixed_mask=visual_residual_mapper.fixed_mask,
                edges=visual_residual_mapper.edges,
                distance_stiffness=wp.to_torch(
                    material_projector.paper_distance_stiffness
                ),
                shape_stiffness=wp.to_torch(
                    material_projector.paper_shape_stiffness
                ),
                inverse_mass=wp.to_torch(
                    environment.sim.model.particle_inv_mass
                ),
                tet_indices=wp.to_torch(
                    environment.sim.model.tet_indices
                ),
                track_particle_ids=(
                    None
                    if flow_depth_bindings is None
                    else torch.as_tensor(
                        flow_depth_bindings.particle_ids,
                        device=visual_residual_mapper.rest_positions.device,
                        dtype=torch.long,
                    )
                ),
                track_particle_weights=(
                    None
                    if flow_depth_bindings is None
                    else torch.as_tensor(
                        flow_depth_bindings.particle_weights,
                        device=visual_residual_mapper.rest_positions.device,
                        dtype=torch.float32,
                    )
                ),
                track_valid_mask=(
                    None
                    if flow_depth_bindings is None
                    else torch.as_tensor(
                        flow_depth_bindings.track_valid,
                        device=visual_residual_mapper.rest_positions.device,
                        dtype=torch.bool,
                    )
                ),
                initial_velocity_damping_per_second=(
                    physics_settings.particle_velocity_damping_per_second
                ),
                initial_coupling_gain=1.0,
                settings=online_settings,
            )
            environment.physics_settings.preserve_spatial_paper_stiffness = True
            environment.super_tissue_stiffness_optimization_enabled = True
            environment.super_tissue_fixed_material_parameters = False
            autograd_parameterization = (
                (
                    f"1global+{len(visual_residual_mapper.rest_positions)}particle/"
                    "zero_mean_graph_distance+weak_tied_shape+fixed_volume/"
                    "observation_weighted_diagonal_gn_lm+warp_directional_fd"
                )
                if online_settings.update_mode
                == "differentiable_particle_graph_lm"
                else
                (
                    "1global+"
                    f"{online_settings.autograd_region_count}regional_coeff/"
                    "zero_mean_smooth_distance+fixed_damping/"
                    "cauchy_relative_h3_h5+warp_adam+h1_safe_projection"
                )
                if online_settings.update_mode
                == "differentiable_hierarchical_relative"
                else
                (
                    f"{online_settings.autograd_region_count}coeff/"
                    "local_zero_mean_distance+fixed_damping/"
                    "cauchy_relative_h1_h3_h5+warp_adam"
                )
                if online_settings.update_mode
                == "differentiable_local_relative"
                else
                (
                    "global_paper_distance+damping;grasp_coupling_fixed_1;"
                    + (
                        "cauchy_relative+20transition_long_primary+"
                        "h1_uncertainty_constrained_warp_fd_lm"
                        if online_settings.update_mode
                        == "differentiable_global_mhe"
                        else (
                            "cauchy_relative+cumulative_warp"
                            if online_settings.update_mode
                            == "differentiable_global_relative"
                            else "track_mean+warp_adam"
                        )
                    )
                )
                if online_settings.update_mode
                in {
                    "differentiable_global",
                    "differentiable_global_relative",
                    "differentiable_global_mhe",
                }
                else (
                    "particlewise_distance+shape/graph_smoothed_rgb_evidence"
                    if online_settings.update_mode == "particle_residual"
                    else
                    f"{online_settings.autograd_region_count}coeff/"
                    "distance+volume+shape"
                )
            )
            print(
                "[example_embodied_super_offline] online paper stiffness: ON; "
                f"log_lr={online_settings.log_learning_rate:g}, "
                f"max_log_step={online_settings.maximum_log_step:g}, "
                "distance_bounds=0.10..2.00, shape_bounds=0.003..0.020, "
                f"ema_history={online_settings.signal_ema_decay:g}, "
                f"{'flow-depth' if visual_feedback_mode == 'trajectory' else 'RGB'} "
                "edge-strain signal="
                f"{online_settings.strain_signal_weight:g}, volume=FIXED, "
                f"spatial_smoothing={online_settings.spatial_smoothing_iterations}, "
                f"update_mode={online_settings.update_mode}, "
                "autograd="
                f"{online_settings.autograd_unroll_steps}frame-causal/"
                f"{online_settings.autograd_integration_substeps}substep/"
                f"{autograd_parameterization}, edge_loss=0.2, "
                "Young_modulus=UNUSED, "
                "policy=delayed-causal/no-branch, "
                "local_tet_quality_gate=0.03"
            )

    playback_controls = SuperPlaybackControls(
        environment,
        dataset_manager,
        fps,
        monitor_psm_base_q=monitor_psm_base_q,
        monitor_tissue_q=monitor_tissue_q,
        monitor_interval=monitor_interval,
        psm_roll_offset_deg=psm_roll_offset_deg,
        psm_camera_translation_mm=psm_camera_translation_mm,
        psm_world_translation_mm=psm_world_translation_mm,
        visual_force_update_interval=visual_force_update_interval,
        visual_feedback_mode=visual_feedback_mode,
        visual_residual_mapper=visual_residual_mapper,
        flow_depth_bindings=flow_depth_bindings,
        flow_depth_observations=flow_depth_observations,
        flow_depth_settings=flow_depth_settings,
        flow_depth_source=flow_depth_source,
        flow_depth_initial_alignment=(
            flow_depth_initial_alignment_diagnostics
        ),
        stiffness_updater=stiffness_updater,
        enable_psm_tissue_contact=effective_psm_tissue_contact,
        stiffness_evaluation_output=stiffness_evaluation_output,
        stiffness_evaluation_horizons=stiffness_evaluation_horizons,
    )
    playback_controls.reset()

    if evaluation_headless:
        if benchmark_output is None and stiffness_evaluation_output is None:
            raise ValueError(
                "--evaluation-headless requires --benchmark-output or "
                "--stiffness-evaluation-output"
            )
        try:
            await run_headless_trajectory_evaluation(
                playback_controls,
                start_frame=evaluation_start_frame,
                frame_count=evaluation_frame_count,
                physics_steps_per_frame=(
                    evaluation_physics_steps_per_frame
                ),
                benchmark_output=benchmark_output,
                evaluation_label=evaluation_label,
                render_images=evaluation_render_images,
                open_loop_start_frame=evaluation_open_loop_start_frame,
                holdout_stride=evaluation_holdout_stride,
                holdout_offset=evaluation_holdout_offset,
                render_frame_mode=evaluation_render_frame_mode,
            )
        finally:
            playback_controls.close()
        return

    from embodied_gaussians.vis import EmbodiedGUI

    visualizer = EmbodiedGUI()
    visualizer.set_environment(environment)
    visualizer.viewer_3d.camera_go_zoom = camera_go_zoom
    if sim_reconstruction_mode:
        # The legacy PSM remains only as an internal controller dependency.
        # Its URDF frames do not match the ORBIT-Surgical PSM in the video, so
        # The Physics toggle uses a soft-only renderer; old rigid PSM/table
        # geometry cannot enter it. The dataset PSM is drawn separately below.
        visualizer.viewer_3d.settings.draw_physics = True
        visualizer.viewer_3d.settings.draw_gaussian_render = True
        visualizer.viewer_3d.settings.draw_cameras = False
        visualizer.viewer_3d.settings.draw_virtual_cameras = False
        from embodied_gaussians.embodied_visualizer.dataset_psm_tip import (
            DatasetPSMTipRenderer,
        )

        dataset_psm_renderer = DatasetPSMTipRenderer(
            dataset_path / "gui_assets" / "official_psm_tip_meshes_v2.npz",
            dataset_path / "task_inputs" / "psm_link_poses.npz",
            lambda: playback_controls.current_frame_index,
        )
        visualizer.callbacks_3d.append(dataset_psm_renderer.draw)
    # SUPER visual-force previews should show the actual target/force scale.
    # The generic viewer defaults (25x pose and 10x soft-force gain) make a
    # millimetre target look as if Gaussians have flown centimetres away.
    visualizer.viewer_3d.settings.visual_forces_pose_scale = 1.0
    visualizer.viewer_3d.settings.visual_forces_scale = 8.0
    # Display-only gain: make the post-clamp particle arrows readable without
    # changing the visual force sent to the physics solver.
    visualizer.viewer_3d.settings.soft_force_display_gain = 60.0
    visualizer.viewer_3d.settings.soft_force_line_width = 2.0
    visualizer.viewer_3d.settings.draw_visual_forces_gaussians_meshes = False
    visualizer.viewer_3d.settings.draw_visual_forces_gaussians_outlines = False
    visualizer.viewer_3d.settings.draw_visual_forces_render = False
    # The SUPER command explicitly opting into visual forces should show the
    # actual post-clamp particle arrows without another expensive gsplat pass.
    visualizer.viewer_3d.settings.draw_visual_forces = (
        visual_feedback_mode == "force" and visual_force_iterations > 0
    )
    # 默认自由视角改成侧视角：从世界 -X 方向看向手术区域，
    # 方便一进 demo 就检查 tissue 高度、z=0 ground plane、相机视锥和 PSM 的相对位置。
    # 这只影响启动后的 3D viewer 初始视角；点击 Go To Camera 仍然会切到真实离线相机视角。
    visualizer.viewer_3d.configure_side_view(
        position=(-0.18, -0.02, 0.053),
        front=(0.995, 0.0, -0.100),
        up=(0.0, 0.0, 1.0),
        activate=True,
    )
    visualizer.callbacks_render.append(playback_controls.draw)

    try:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(playback_controls.run)
            await visualizer.run()
            nursery.cancel_scope.cancel()
    finally:
        playback_controls.close()


if __name__ == "__main__":
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset)
    print(f"[example_embodied_super_offline] 使用数据集目录: {dataset_path}")
    camera_names = [name.strip() for name in args.cameras.split(",") if name.strip()]
    stiffness_evaluation_horizons = parse_stiffness_evaluation_horizons(
        args.stiffness_evaluation_horizons
    )
    psm_roll_offset_deg = args.psm_roll_offset_deg
    if psm_roll_offset_deg is None:
        psm_roll_offset_deg = (
            0.0
            if args.psm_pose_driver
            in {
                "raw_kinematics",
                "raw_p420006",
                "raw_p420006_stereo_visual",
                "raw_p420006_sam2_online",
                "raw_paper_lnd_sam2_online",
                "raw_paper_lnd_first_stereo_static_q5",
                "raw_paper_lnd_first_stereo_se3_fixed_q5",
                "raw_paper_lnd_sam2_multianchor_closedjaw",
                "raw_paper_lnd_sam2_dense_contact_closedjaw",
                "raw_paper_lnd_sam2_dense_contact_se3_only",
                "raw_paper_lnd_sam2_dense_contact_unbounded_xyz",
            }
            else -27.0
        )
    wp.config.quiet = True
    wp.init()
    trio.run(
        main,
        dataset_path,
        args.fps,
        args.monitor_psm_base_q,
        args.monitor_tissue_q,
        args.monitor_interval,
        args.visual_feedback_mode,
        args.flow_depth_bindings,
        args.flow_depth_observations,
        args.flow_depth_position_gain,
        args.flow_depth_velocity_gain,
        args.flow_depth_absolute_position_weight,
        args.flow_depth_solver_regularization,
        args.flow_depth_solver_iterations,
        args.flow_depth_robust_residual_mm,
        args.flow_depth_maximum_position_correction_mm,
        args.flow_depth_maximum_velocity_correction_m_s,
        args.flow_depth_initial_alignment,
        args.flow_depth_initial_alignment_maximum_mm,
        args.visual_force_iterations,
        args.visual_residual_iterations,
        args.visual_residual_learning_rate_m,
        args.visual_residual_maximum_mm,
        args.visual_residual_image_scale,
        args.visual_force_update_interval,
        args.online_stiffness_update,
        args.sim_grasp_boundary_mode,
        args.stiffness_log_learning_rate,
        args.stiffness_update_mode,
        args.stiffness_strain_signal_weight,
        args.stiffness_autograd_unroll_steps,
        args.stiffness_autograd_region_count,
        args.stiffness_maximum_log_step,
        args.stiffness_signal_ema_decay,
        args.stiffness_spatial_smoothing_iterations,
        args.stiffness_spatial_smoothing_blend,
        args.initial_paper_distance_stiffness,
        args.initial_paper_shape_stiffness,
        args.stiffness_evaluation_output,
        stiffness_evaluation_horizons,
        args.evaluation_headless,
        args.evaluation_start_frame,
        args.evaluation_frame_count,
        args.evaluation_physics_steps_per_frame,
        args.benchmark_output,
        args.evaluation_label,
        args.evaluation_render_images,
        (
            None
            if args.evaluation_open_loop_start_frame < 0
            else args.evaluation_open_loop_start_frame
        ),
        (
            None
            if args.evaluation_holdout_stride == 0
            else args.evaluation_holdout_stride
        ),
        args.evaluation_holdout_offset,
        args.evaluation_render_frame_mode,
        camera_names,
        args.camera_go_zoom,
        psm_roll_offset_deg,
        tuple(args.psm_camera_translation_mm),
        tuple(args.psm_world_translation_mm),
        args.psm_pose_driver,
        args.psm_visual_mode,
        args.tissue_mode,
        args.psm_tissue_contact,
        args.calibrated_profile,
    )
