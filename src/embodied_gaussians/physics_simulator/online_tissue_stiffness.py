"""Bounded online paper-PBD stiffness adaptation from accepted visual residuals."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OnlineTissueStiffnessSettings:
    """Aggressive but bounded adaptation used by the SUPER realtime GUI.

    The signed signal compares the physical deformation with the accepted
    visual correction.  A correction back toward rest means the prediction
    over-deformed and therefore hardens the local distance/shape constraints;
    a correction farther from rest softens them.  This is the realtime
    closed-loop approximation; it does not differentiate through the full
    multi-frame XPBD rollout used by the much slower paper optimizer.
    """

    log_learning_rate: float = 0.18
    maximum_log_step: float = 0.18
    residual_full_scale_m: float = 0.00020
    deformation_full_scale_m: float = 0.00075
    minimum_residual_m: float = 0.00002
    minimum_deformation_m: float = 0.00010
    hardening_bias: float = 0.15
    # Keep temporal smoothing, but give a newly verified visual signal enough
    # weight to be visible within a few accepted updates.  This implements
    # e_t = 0.70 e_{t-1} + 0.30 signal_t.
    signal_ema_decay: float = 0.70
    rejected_ema_decay: float = 0.50
    spatial_smoothing_iterations: int = 1
    spatial_smoothing_blend: float = 0.35
    shape_update_gain: float = 1.50
    # RGB residual first corrects the observed surface position.  Comparing
    # edge strain before/after that accepted correction removes free rigid
    # translation and gives a causal local compliance signal without a shadow
    # rollout or a depth input.
    strain_signal_weight: float = 0.80
    minimum_edge_strain: float = 0.001
    edge_strain_full_scale: float = 0.025
    # Distance stiffness carries most of the regional stretch contrast.  Keep
    # its per-commit log step unchanged, but allow verified regions to separate
    # over repeated, independently validated observations.
    distance_minimum: float = 0.10
    distance_maximum: float = 2.00
    shape_minimum: float = 0.003
    # Liang et al. optimize shape stiffness in [0, 0.02]. Keep a conservative
    # positive floor below the current 0.004 scene baseline, but do not let
    # online feedback harden beyond the paper's upper bound.
    shape_maximum: float = 0.020
    # ``heuristic`` preserves the original residual/edge-strain rule.
    # ``particle_residual`` is the explicit production name for that same
    # particle-wise field update: every eligible particle owns an independent
    # distance/shape value, while graph smoothing acts only on its evidence.
    # ``differentiable_low_dim`` differentiates causally through the latest
    # 3..5 already observed frames.  Every temporal step contains distance,
    # volume, and tetrahedral shape-matching projections.  Warp remains the
    # authoritative forward simulator; the Torch window supplies the material
    # Jacobian used by the online Adam update.
    update_mode: str = "heuristic"
    autograd_unroll_steps: int = 4
    autograd_region_count: int = 12
    autograd_adam_beta2: float = 0.999
    autograd_adam_epsilon: float = 1.0e-8
    autograd_prior_weight: float = 0.02
    autograd_maximum_log_offset: float = 0.35
    # Distance is the identifiable stretch parameter in the first trial.
    # Shape remains fixed; jointly adapting two fields from the same RGB
    # residual recreates the ambiguity observed in the previous experiment.
    autograd_shape_update_gain: float = 0.0
    autograd_frame_dt: float = 1.0 / 30.0
    autograd_material_iterations: int = 2
    autograd_integration_substeps: int = 6
    autograd_control_projection_interval_substeps: int = 6
    autograd_volume_stiffness: float = 1.0e10
    autograd_velocity_damping_per_second: float = 10.0
    autograd_projection_velocity_scale: float = 0.14
    # Retained for compatibility with older saved configurations.  The new
    # causal rollout uses ``autograd_frame_dt`` and actual frame-index gaps.
    autograd_pbd_dt: float = 1.0 / 1800.0
    autograd_pbd_compliance_scale: float = 1.0
    autograd_pbd_relaxation: float = 0.25
    # Global paper-PBD system identification keeps the existing paper
    # distance-stiffness representation (not Young's modulus).  It estimates
    # only a scene-wide distance multiplier, velocity damping, and prescribed
    # grasp coupling.  Shape and volume remain fixed.
    global_damping_minimum_per_second: float = 2.0
    global_damping_maximum_per_second: float = 30.0
    global_coupling_minimum: float = 0.70
    global_coupling_maximum: float = 1.30
    # The one-step objective drove distance and damping to opposite log bounds
    # (-0.35/+0.35).  A dimensionless prior of 0.40 is large enough to make
    # those globally confounded parameters identifiable while coupling keeps
    # a lighter prior and remains free to explain the prescribed grasp motion.
    global_parameter_prior_weight: float = 0.40
    # A pointwise mean improved the future mean error, but slightly regressed
    # the median and allowed one sparsely sampled boundary region to dominate
    # the maximum error.  Keep most weight on the original objective, while
    # adding a stable median-band proxy and spatially balanced tail terms.  The
    # loss uses only already accepted visual observations, never evaluation GT.
    global_median_band_weight: float = 0.00
    global_region_balance_weight: float = 0.10
    global_tail_region_weight: float = 0.05
    global_tail_region_fraction: float = 0.25
    # One fixed causal objective is used in both 7:1 reconstruction and 80/20
    # future prediction.  The 1.5:2:3 compromise restores short-horizon
    # influence without giving up the longest observed rollout gap.  This is
    # not switched by evaluation protocol and is never selected from GT.
    global_horizon_weights: tuple[float, float, float] = (1.5, 2.0, 3.0)
    # The robust-relative variant estimates material from deformation rather
    # than from a few absolute track positions.  Neighbour differences cancel
    # common camera/depth translation, while a Cauchy penalty prevents the
    # sparse multi-millimetre FoundationStereo outliers observed in the
    # benchmark from steering a persistent material parameter.
    global_relative_track_weight: float = 0.65
    global_cauchy_scale: float = 2.5
    # SUPER's useful material behavior is local: most of the tissue keeps the
    # reset stiffness while a small, spatially smooth field explains persistent
    # deformation error.  The local-relative mode uses the same deterministic
    # region basis, removes its particle-space weighted mean exactly, and only
    # attempts an update after a complete H=1/3/5 causal window.  This is a
    # single continuous optimizer, not candidate/branch selection.
    local_horizon_weights: tuple[float, float, float] = (1.5, 2.0, 3.0)
    local_update_interval_frames: int = 5
    local_spatial_prior_weight: float = 0.05
    # Particle-graph material identification stores one zero-mean log-distance
    # offset per physical particle plus one scene-wide mean. A diagonal
    # Gauss--Newton/LM approximation is coupled through the tetrahedral edge
    # graph, while an observation score increases damping away from RGB tracks.
    # Shape follows the same field only weakly; volume stays fixed.
    graph_lm_damping: float = 0.20
    graph_lm_spatial_hessian_weight: float = 0.25
    graph_lm_cg_iterations: int = 8
    graph_observability_floor: float = 0.08
    graph_shape_log_coupling: float = 0.15
    graph_directional_probe_count: int = 2
    graph_directional_log_epsilon: float = 0.01
    graph_directional_cosine_minimum: float = 0.90
    graph_update_interval_frames: int = 10
    # The hierarchical optimizer follows H3/H5 for future deformation, but
    # its actual Adam step must remain a descent direction for the already
    # observed H1 reconstruction loss.  A small fixed margin avoids merely
    # sliding along the H1 tangent plane.  This is a causal gradient projection,
    # not a candidate rollout, rollback, or evaluation-protocol branch.
    hierarchical_short_horizon_descent_margin: float = 0.05
    # Causal moving-horizon material identification.  Unlike the H=1/2/3
    # Adam modes above, this mode keeps several already observed transition
    # segments (periodic 7:1 holdouts are allowed), differentiates the actual
    # Warp replay by finite differences, and solves one damped two-parameter
    # Gauss--Newton step.  The singular-value gate prevents the familiar
    # distance-stiffness/damping trade-off from being committed when RGB
    # motion does not distinguish the two effects.
    observable_window_size: int = 20
    observable_minimum_transitions: int = 8
    observable_update_interval_frames: int = 5
    observable_minimum_singular_ratio: float = 0.02
    observable_minimum_jacobian_norm: float = 1.0e-4
    observable_lm_damping: float = 0.05
    observable_parameter_prior_weight: float = 0.05
    observable_fd_scale_cosine_minimum: float = 0.95
    observable_horizon_weight_growth: float = 0.50
    observable_horizon_weight_maximum: float = 3.0
    observable_maximum_edge_residuals: int = 512
    # Independently normalize the one-step reconstruction objective and the
    # contiguous long-rollout objective before combining them.  Equal weight
    # is a fixed method definition used by every evaluation protocol, not a
    # reconstruction/future switch or a GT-selected branch.
    observable_short_horizon_weight: float = 0.50
    observable_long_horizon_weight: float = 0.50
    # The prescribed grasp trajectory is a known kinematic boundary, not an
    # unknown material parameter.  Keep its gain exactly one and identify only
    # paper distance stiffness plus velocity damping.
    global_optimize_coupling: bool = False
    # Reject marginally aligned late updates while retaining causal material
    # evidence that is useful for the following one-step prediction.
    gradient_cosine_minimum: float = 0.95
    finite_difference_log_epsilon: float = 0.005
    local_finite_difference_log_epsilon: float = 0.05

    def validate(self) -> None:
        if self.log_learning_rate <= 0.0 or self.maximum_log_step <= 0.0:
            raise ValueError("Online stiffness step sizes must be positive")
        if min(
            self.residual_full_scale_m,
            self.deformation_full_scale_m,
            self.minimum_residual_m,
            self.minimum_deformation_m,
        ) <= 0.0:
            raise ValueError("Online stiffness metric scales must be positive")
        if self.minimum_residual_m > self.residual_full_scale_m:
            raise ValueError(
                "Minimum residual cannot exceed residual full scale"
            )
        if self.minimum_deformation_m > self.deformation_full_scale_m:
            raise ValueError(
                "Minimum deformation cannot exceed deformation full scale"
            )
        if not -1.0 <= self.hardening_bias <= 1.0:
            raise ValueError("Online stiffness hardening bias must lie in [-1,1]")
        if not 0.0 <= self.signal_ema_decay < 1.0:
            raise ValueError("Online stiffness EMA decay must lie in [0,1)")
        if not 0.0 <= self.rejected_ema_decay <= 1.0:
            raise ValueError("Rejected stiffness EMA decay must lie in [0,1]")
        if self.spatial_smoothing_iterations < 0:
            raise ValueError("Stiffness smoothing iterations cannot be negative")
        if not 0.0 <= self.spatial_smoothing_blend <= 1.0:
            raise ValueError("Stiffness smoothing blend must lie in [0,1]")
        if self.shape_update_gain <= 0.0:
            raise ValueError("Shape stiffness update gain must be positive")
        if not 0.0 <= self.strain_signal_weight <= 1.0:
            raise ValueError("Strain signal weight must lie in [0,1]")
        if not 0.0 < self.minimum_edge_strain <= self.edge_strain_full_scale:
            raise ValueError("Edge-strain scales are invalid")
        if not 0.0 < self.distance_minimum <= self.distance_maximum:
            raise ValueError("Distance stiffness bounds are invalid")
        if not 0.0 < self.shape_minimum <= self.shape_maximum:
            raise ValueError("Shape stiffness bounds are invalid")
        if self.update_mode not in {
            "heuristic",
            "particle_residual",
            "differentiable_low_dim",
            "differentiable_global",
            "differentiable_global_relative",
            "differentiable_global_mhe",
            "differentiable_local_relative",
            "differentiable_hierarchical_relative",
            "differentiable_particle_graph_lm",
        }:
            raise ValueError("Unknown online stiffness update mode")
        if not 3 <= self.autograd_unroll_steps <= 5:
            raise ValueError("Differentiable PBD rollout must use 3..5 steps")
        if self.autograd_region_count < 2:
            raise ValueError("Low-dimensional stiffness needs at least 2 regions")
        if not 0.0 < self.autograd_adam_beta2 < 1.0:
            raise ValueError("Adam beta2 must lie in (0,1)")
        if self.autograd_adam_epsilon <= 0.0:
            raise ValueError("Adam epsilon must be positive")
        if self.autograd_prior_weight < 0.0:
            raise ValueError("Autograd stiffness prior must be non-negative")
        if self.autograd_maximum_log_offset <= 0.0:
            raise ValueError("Autograd log-offset bound must be positive")
        if self.autograd_shape_update_gain < 0.0:
            raise ValueError("Autograd shape gain must be non-negative")
        if self.autograd_frame_dt <= 0.0:
            raise ValueError("Autograd frame dt must be positive")
        if self.autograd_material_iterations < 1:
            raise ValueError("Autograd material iterations must be positive")
        if self.autograd_integration_substeps < 1:
            raise ValueError("Autograd integration substeps must be positive")
        if self.autograd_control_projection_interval_substeps < 1:
            raise ValueError("Control projection interval must be positive")
        if self.autograd_volume_stiffness <= 0.0:
            raise ValueError("Autograd volume stiffness must be positive")
        if self.autograd_velocity_damping_per_second < 0.0:
            raise ValueError("Autograd velocity damping cannot be negative")
        if not 0.0 <= self.autograd_projection_velocity_scale <= 1.0:
            raise ValueError(
                "Autograd projection velocity scale must lie in [0,1]"
            )
        if self.autograd_pbd_dt <= 0.0:
            raise ValueError("Autograd PBD dt must be positive")
        if self.autograd_pbd_compliance_scale <= 0.0:
            raise ValueError("Autograd PBD compliance scale must be positive")
        if not 0.0 < self.autograd_pbd_relaxation <= 1.0:
            raise ValueError("Autograd PBD relaxation must lie in (0,1]")
        if not (
            0.0
            < self.global_damping_minimum_per_second
            <= self.global_damping_maximum_per_second
        ):
            raise ValueError("Global damping bounds are invalid")
        if not 0.0 < self.global_coupling_minimum <= self.global_coupling_maximum:
            raise ValueError("Global coupling bounds are invalid")
        if self.global_parameter_prior_weight < 0.0:
            raise ValueError("Global parameter prior must be non-negative")
        distribution_weights = (
            self.global_median_band_weight,
            self.global_region_balance_weight,
            self.global_tail_region_weight,
        )
        if any(weight < 0.0 for weight in distribution_weights):
            raise ValueError("Global distribution weights must be non-negative")
        if sum(distribution_weights) > 1.0:
            raise ValueError("Global distribution weights cannot exceed one")
        if not 0.0 < self.global_tail_region_fraction <= 1.0:
            raise ValueError("Global tail-region fraction must lie in (0,1]")
        if (
            len(self.global_horizon_weights) != 3
            or any(weight <= 0.0 for weight in self.global_horizon_weights)
        ):
            raise ValueError("Global H1/H2/H3 weights must be three positive values")
        if not 0.0 <= self.global_relative_track_weight <= 1.0:
            raise ValueError("Relative-track weight must lie in [0,1]")
        if self.global_cauchy_scale <= 0.0:
            raise ValueError("Global Cauchy scale must be positive")
        if (
            len(self.local_horizon_weights) != 3
            or any(weight <= 0.0 for weight in self.local_horizon_weights)
        ):
            raise ValueError("Local H1/H3/H5 weights must be three positive values")
        if self.local_update_interval_frames < 1:
            raise ValueError("Local update interval must be positive")
        if self.local_spatial_prior_weight < 0.0:
            raise ValueError("Local spatial prior must be non-negative")
        if self.graph_lm_damping <= 0.0:
            raise ValueError("Particle-graph LM damping must be positive")
        if self.graph_lm_spatial_hessian_weight < 0.0:
            raise ValueError("Particle-graph spatial Hessian weight is invalid")
        if self.graph_lm_cg_iterations < 1:
            raise ValueError("Particle-graph CG iterations must be positive")
        if not 0.0 < self.graph_observability_floor <= 1.0:
            raise ValueError("Particle observability floor must lie in (0,1]")
        if not 0.0 <= self.graph_shape_log_coupling <= 1.0:
            raise ValueError("Particle-graph shape coupling must lie in [0,1]")
        if self.graph_directional_probe_count < 2:
            raise ValueError("Particle-graph gradient check needs at least 2 probes")
        if self.graph_directional_log_epsilon <= 0.0:
            raise ValueError("Particle-graph directional epsilon must be positive")
        if not -1.0 <= self.graph_directional_cosine_minimum <= 1.0:
            raise ValueError("Particle-graph directional cosine is invalid")
        if self.graph_update_interval_frames < 1:
            raise ValueError("Particle-graph update interval must be positive")
        if self.observable_window_size < 3:
            raise ValueError("Observable window must contain at least 3 transitions")
        if not 3 <= self.observable_minimum_transitions <= self.observable_window_size:
            raise ValueError("Observable warm-up must fit inside its window")
        if self.observable_update_interval_frames < 1:
            raise ValueError("Observable update interval must be positive")
        if not 0.0 < self.observable_minimum_singular_ratio <= 1.0:
            raise ValueError("Observable singular-value ratio must lie in (0,1]")
        if self.observable_minimum_jacobian_norm <= 0.0:
            raise ValueError("Observable Jacobian norm floor must be positive")
        if self.observable_lm_damping <= 0.0:
            raise ValueError("Observable LM damping must be positive")
        if self.observable_parameter_prior_weight < 0.0:
            raise ValueError("Observable parameter prior must be non-negative")
        if not -1.0 <= self.observable_fd_scale_cosine_minimum <= 1.0:
            raise ValueError("Observable FD-scale cosine must lie in [-1,1]")
        if self.observable_horizon_weight_growth < 0.0:
            raise ValueError("Observable horizon growth cannot be negative")
        if self.observable_horizon_weight_maximum < 1.0:
            raise ValueError("Observable horizon weight maximum must be at least one")
        if self.observable_maximum_edge_residuals < 1:
            raise ValueError("Observable edge residual budget must be positive")
        if (
            self.observable_short_horizon_weight < 0.0
            or self.observable_long_horizon_weight < 0.0
            or self.observable_short_horizon_weight
            + self.observable_long_horizon_weight
            <= 0.0
        ):
            raise ValueError("Observable short/long weights are invalid")
        if not -1.0 <= self.gradient_cosine_minimum <= 1.0:
            raise ValueError("Gradient cosine threshold must lie in [-1,1]")
        if self.finite_difference_log_epsilon <= 0.0:
            raise ValueError("Finite-difference epsilon must be positive")
        if self.local_finite_difference_log_epsilon <= 0.0:
            raise ValueError("Local finite-difference epsilon must be positive")
        if not 0.0 <= self.hierarchical_short_horizon_descent_margin < 1.0:
            raise ValueError(
                "Hierarchical short-horizon descent margin must lie in [0,1)"
            )


@dataclass
class PaperStiffnessCandidate:
    """Tentative material state that has not yet reached the XPBD projector."""

    distance_stiffness: torch.Tensor
    shape_stiffness: torch.Tensor
    signal_ema: torch.Tensor
    log_step: torch.Tensor
    eligible_mask: torch.Tensor
    source_mask: torch.Tensor
    metrics: dict[str, float | int | str]
    low_dimensional_log_coefficients: torch.Tensor | None = None
    adam_first_moment: torch.Tensor | None = None
    adam_second_moment: torch.Tensor | None = None
    adam_step: int | None = None
    global_log_coefficients: torch.Tensor | None = None
    global_adam_first_moment: torch.Tensor | None = None
    global_adam_second_moment: torch.Tensor | None = None
    global_adam_step: int | None = None
    global_velocity_damping_per_second: float | None = None
    global_coupling_gain: float | None = None
    autograd_parameter_gradient: torch.Tensor | None = None


@dataclass(frozen=True)
class _CausalMaterialObservation:
    """One accepted, past-or-current RGB state used by causal TBPTT."""

    frame_index: int
    rollout_start_positions: torch.Tensor
    corrected_positions: torch.Tensor
    # Velocity belongs to ``rollout_start_positions`` and is captured before
    # the RGB/depth correction.  Never feed the corrected state velocity into
    # material identification.
    physical_velocities: torch.Tensor
    control_coupling_base_positions: torch.Tensor
    control_coupling_displacements: torch.Tensor
    eligible_mask: torch.Tensor
    material_active_mask: torch.Tensor
    control_exclusion_mask: torch.Tensor
    control_frozen_mask: torch.Tensor
    track_target_positions: torch.Tensor | None
    track_valid_mask: torch.Tensor | None


class ResidualDrivenPaperStiffnessUpdater:
    """Update per-particle paper distance/shape stiffness in log space."""

    def __init__(
        self,
        *,
        rest_positions: torch.Tensor,
        fixed_mask: torch.Tensor,
        edges: torch.Tensor,
        distance_stiffness: torch.Tensor,
        shape_stiffness: torch.Tensor,
        inverse_mass: torch.Tensor | None = None,
        tet_indices: torch.Tensor | None = None,
        track_particle_ids: torch.Tensor | None = None,
        track_particle_weights: torch.Tensor | None = None,
        track_valid_mask: torch.Tensor | None = None,
        initial_velocity_damping_per_second: float | None = None,
        initial_coupling_gain: float = 1.0,
        settings: OnlineTissueStiffnessSettings | None = None,
    ) -> None:
        self.settings = settings or OnlineTissueStiffnessSettings()
        self.settings.validate()
        self.rest_positions = rest_positions.detach().to(dtype=torch.float32)
        self.fixed_mask = fixed_mask.detach().to(
            device=self.rest_positions.device, dtype=torch.bool
        )
        self.edges = edges.detach().to(
            device=self.rest_positions.device, dtype=torch.long
        )
        self.distance_stiffness = distance_stiffness
        self.shape_stiffness = shape_stiffness
        particle_count = len(self.rest_positions)
        expected = (particle_count,)
        if self.fixed_mask.shape != expected:
            raise ValueError("Fixed mask does not match stiffness particles")
        if self.distance_stiffness.shape != expected:
            raise ValueError("Distance stiffness does not match particles")
        if self.shape_stiffness.shape != expected:
            raise ValueError("Shape stiffness does not match particles")
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError("Stiffness graph edges must have shape [E,2]")
        if self.edges.numel() and (
            int(self.edges.min()) < 0 or int(self.edges.max()) >= particle_count
        ):
            raise ValueError("Stiffness graph edge is out of range")
        self.rest_edge_lengths = torch.linalg.vector_norm(
            self.rest_positions[self.edges[:, 1]]
            - self.rest_positions[self.edges[:, 0]],
            dim=1,
        ).clamp_min(1.0e-8)
        self.initial_distance = self.distance_stiffness.detach().clone()
        self.initial_shape = self.shape_stiffness.detach().clone()
        self.initial_velocity_damping_per_second = float(
            self.settings.autograd_velocity_damping_per_second
            if initial_velocity_damping_per_second is None
            else initial_velocity_damping_per_second
        )
        self.initial_coupling_gain = float(initial_coupling_gain)
        if self.initial_velocity_damping_per_second <= 0.0:
            raise ValueError("Initial velocity damping must be positive")
        if self.initial_coupling_gain <= 0.0:
            raise ValueError("Initial grasp coupling gain must be positive")
        if inverse_mass is None:
            inverse_mass = (~self.fixed_mask).to(dtype=torch.float32)
        self.inverse_mass = inverse_mass.detach().to(
            device=self.rest_positions.device, dtype=torch.float32
        )
        if self.inverse_mass.shape != expected:
            raise ValueError("Inverse mass does not match stiffness particles")
        self.inverse_mass = torch.where(
            self.fixed_mask,
            torch.zeros_like(self.inverse_mass),
            self.inverse_mass.clamp_min(0.0),
        )
        if tet_indices is None:
            tet_indices = torch.empty(
                (0, 4), dtype=torch.long, device=self.rest_positions.device
            )
        self.tet_indices = tet_indices.detach().to(
            device=self.rest_positions.device, dtype=torch.long
        )
        if self.tet_indices.ndim != 2 or self.tet_indices.shape[1] != 4:
            raise ValueError("Stiffness tetrahedra must have shape [T,4]")
        if self.tet_indices.numel() and (
            int(self.tet_indices.min()) < 0
            or int(self.tet_indices.max()) >= particle_count
        ):
            raise ValueError("Stiffness tetrahedron is out of range")
        if (track_particle_ids is None) != (track_particle_weights is None):
            raise ValueError("Track ids and weights must be supplied together")
        if track_particle_ids is None:
            self.track_particle_ids = None
            self.track_particle_weights = None
            self.track_valid_mask = None
        else:
            ids = track_particle_ids.detach().to(
                device=self.rest_positions.device, dtype=torch.long
            )
            weights = track_particle_weights.detach().to(
                device=self.rest_positions.device, dtype=torch.float32
            )
            if ids.ndim != 2 or weights.shape != ids.shape:
                raise ValueError("Track bindings must be matching 2-D tensors")
            valid_slots = ids >= 0
            if bool((ids[valid_slots] >= particle_count).any().item()):
                raise ValueError("Track binding references a missing particle")
            if bool((weights[~valid_slots] != 0.0).any().item()):
                raise ValueError("Padded track weights must be zero")
            binding_valid = valid_slots.any(dim=1)
            if track_valid_mask is not None:
                supplied_valid = track_valid_mask.detach().to(
                    device=self.rest_positions.device, dtype=torch.bool
                )
                if supplied_valid.shape != binding_valid.shape:
                    raise ValueError("Track-valid mask has the wrong shape")
                binding_valid &= supplied_valid
            weight_sums = weights.sum(dim=1)
            if bool(
                (
                    binding_valid
                    & ~torch.isclose(
                        weight_sums,
                        torch.ones_like(weight_sums),
                        atol=2.0e-6,
                        rtol=0.0,
                    )
                ).any().item()
            ):
                raise ValueError("Valid track weights must sum to one")
            self.track_particle_ids = ids.clamp_min(0)
            self.track_particle_weights = weights
            self.track_valid_mask = binding_valid
        self.track_neighbor_edges = self._build_track_neighbor_edges(
            neighbor_count=4
        )
        self.edge_incidence_count = torch.zeros(
            particle_count,
            dtype=torch.float32,
            device=self.rest_positions.device,
        )
        if self.edges.numel():
            edge_ones = torch.ones(
                len(self.edges),
                dtype=torch.float32,
                device=self.rest_positions.device,
            )
            self.edge_incidence_count.index_add_(0, self.edges[:, 0], edge_ones)
            self.edge_incidence_count.index_add_(0, self.edges[:, 1], edge_ones)
        self.tet_incidence_count = torch.zeros(
            particle_count,
            dtype=torch.float32,
            device=self.rest_positions.device,
        )
        if self.tet_indices.numel():
            flat_tet_ids = self.tet_indices.reshape(-1)
            self.tet_incidence_count.index_add_(
                0,
                flat_tet_ids,
                torch.ones_like(flat_tet_ids, dtype=torch.float32),
            )
            rest_tets = self.rest_positions[self.tet_indices]
            self.rest_tet_centers = rest_tets.mean(dim=1, keepdim=True)
            self.rest_tet_relative = rest_tets - self.rest_tet_centers
            rest_pose = torch.stack(
                (
                    rest_tets[:, 1] - rest_tets[:, 0],
                    rest_tets[:, 2] - rest_tets[:, 0],
                    rest_tets[:, 3] - rest_tets[:, 0],
                ),
                dim=-1,
            )
            rest_determinant = torch.linalg.det(rest_pose)
            if bool((rest_determinant <= 0.0).any().item()):
                raise ValueError(
                    "Differentiable paper PBD needs positive tetrahedra"
                )
            self.rest_tet_inverse_pose = torch.linalg.inv(rest_pose)
            self.rest_tet_volumes = rest_determinant / 6.0
            self.rest_tet_squared_extent = torch.sum(
                self.rest_tet_relative.square(), dim=(1, 2)
            )
        else:
            self.rest_tet_centers = torch.empty(
                (0, 1, 3),
                dtype=torch.float32,
                device=self.rest_positions.device,
            )
            self.rest_tet_relative = torch.empty(
                (0, 4, 3),
                dtype=torch.float32,
                device=self.rest_positions.device,
            )
            self.rest_tet_inverse_pose = torch.empty(
                (0, 3, 3),
                dtype=torch.float32,
                device=self.rest_positions.device,
            )
            self.rest_tet_volumes = torch.empty(
                (0,), dtype=torch.float32, device=self.rest_positions.device
            )
            self.rest_tet_squared_extent = torch.empty(
                (0,), dtype=torch.float32, device=self.rest_positions.device
            )
        self.low_dimensional_basis = self._build_low_dimensional_basis(
            self.settings.autograd_region_count
        )
        self._rebuild_track_region_basis()
        # The known grasp/fixed patch is never allowed to acquire a learned
        # material offset.  Dynamic control exclusions are accumulated as they
        # become causally known; they are not inferred from evaluation points.
        self.local_parameter_exclusion_mask = self.fixed_mask.clone()
        self.particle_observability = self._build_particle_observability()
        self._last_local_relative_attempt_frame: int | None = None
        coefficient_count = self.low_dimensional_basis.shape[1]
        if self.settings.update_mode == "differentiable_particle_graph_lm":
            # One global mean followed by one local coefficient per particle.
            coefficient_count = particle_count + 1
        elif self.settings.update_mode == "differentiable_hierarchical_relative":
            # One global log-distance followed by zero-mean regional terms.
            coefficient_count += 1
        self.low_dimensional_log_coefficients = torch.zeros(
            coefficient_count,
            dtype=torch.float32,
            device=self.rest_positions.device,
        )
        self.adam_first_moment = torch.zeros_like(
            self.low_dimensional_log_coefficients
        )
        self.adam_second_moment = torch.zeros_like(
            self.low_dimensional_log_coefficients
        )
        self.adam_step = 0
        # [paper distance stiffness, damping, grasp coupling], all optimized
        # as log multipliers around the scene's existing paper-PBD values.
        self.global_log_coefficients = torch.zeros(
            3, dtype=torch.float32, device=self.rest_positions.device
        )
        self.global_adam_first_moment = torch.zeros_like(
            self.global_log_coefficients
        )
        self.global_adam_second_moment = torch.zeros_like(
            self.global_log_coefficients
        )
        self.global_adam_step = 0
        self.global_velocity_damping_per_second = (
            self.initial_velocity_damping_per_second
        )
        self.global_coupling_gain = self.initial_coupling_gain
        self.signal_ema = torch.zeros(
            particle_count,
            dtype=torch.float32,
            device=self.rest_positions.device,
        )
        self.update_count = 0
        self.candidate_count = 0
        self.rejected_count = 0
        self.pending_candidate: PaperStiffnessCandidate | None = None
        self.last_metrics: dict[str, float | int | str] | None = None
        self.last_proposal_diagnostics: dict[str, torch.Tensor] | None = None
        self.last_proposal_candidate_count = 0
        self.causal_material_history: list[_CausalMaterialObservation] = []
        self._next_implicit_frame_index = 0

    def _build_particle_observability(self) -> torch.Tensor:
        """Return a deterministic per-particle RGB support score in [floor,1]."""
        count = len(self.rest_positions)
        score = torch.zeros(
            count, dtype=torch.float32, device=self.rest_positions.device
        )
        if (
            self.track_particle_ids is not None
            and self.track_particle_weights is not None
            and self.track_valid_mask is not None
            and bool(self.track_valid_mask.any().item())
        ):
            ids = self.track_particle_ids[self.track_valid_mask].reshape(-1)
            weights = self.track_particle_weights[self.track_valid_mask].reshape(-1)
            score.index_add_(0, ids, weights)
            if self.edges.numel():
                source, target = self.edges[:, 0], self.edges[:, 1]
                degree = self.edge_incidence_count.clamp_min(1.0)
                for _ in range(max(3, self.settings.spatial_smoothing_iterations)):
                    neighbor_sum = torch.zeros_like(score)
                    neighbor_sum.index_add_(0, source, score[target])
                    neighbor_sum.index_add_(0, target, score[source])
                    score = torch.lerp(score, neighbor_sum / degree, 0.50)
            score = score / score.max().clamp_min(1.0e-12)
            floor = self.settings.graph_observability_floor
            score = floor + (1.0 - floor) * score
        else:
            score.fill_(1.0)
        score[self.fixed_mask] = 0.0
        return score

    def _build_track_neighbor_edges(self, neighbor_count: int) -> torch.Tensor:
        """Build an immutable local graph over the RGB material tracks.

        The graph is computed only from the canonical particle binding.  It
        therefore contains no evaluation points, material labels, or future
        observations.  Undirected four-neighbour edges are sufficient to
        measure local deformation while keeping the loss inexpensive.
        """
        device = self.rest_positions.device
        if (
            self.track_particle_ids is None
            or self.track_particle_weights is None
            or self.track_valid_mask is None
        ):
            return torch.empty((0, 2), dtype=torch.long, device=device)
        valid_ids = torch.nonzero(
            self.track_valid_mask, as_tuple=False
        ).flatten()
        if valid_ids.numel() < 2:
            return torch.empty((0, 2), dtype=torch.long, device=device)
        centers = self.fixed_track_centers(self.rest_positions)
        valid_centers = centers[valid_ids]
        distance = torch.cdist(valid_centers, valid_centers)
        distance.fill_diagonal_(float("inf"))
        count = min(int(neighbor_count), int(valid_ids.numel()) - 1)
        neighbors = torch.topk(
            distance, k=count, dim=1, largest=False
        ).indices
        source = valid_ids[:, None].expand_as(neighbors).reshape(-1)
        target = valid_ids[neighbors.reshape(-1)]
        edges = torch.stack(
            (torch.minimum(source, target), torch.maximum(source, target)),
            dim=1,
        )
        return torch.unique(edges, dim=0)

    def reset(self) -> None:
        with torch.no_grad():
            self.distance_stiffness.copy_(self.initial_distance)
            self.shape_stiffness.copy_(self.initial_shape)
            self.signal_ema.zero_()
            self.low_dimensional_log_coefficients.zero_()
            self.adam_first_moment.zero_()
            self.adam_second_moment.zero_()
            self.global_log_coefficients.zero_()
            self.global_adam_first_moment.zero_()
            self.global_adam_second_moment.zero_()
            self.local_parameter_exclusion_mask.copy_(self.fixed_mask)
        self.adam_step = 0
        self.global_adam_step = 0
        self.global_velocity_damping_per_second = (
            self.initial_velocity_damping_per_second
        )
        self.global_coupling_gain = self.initial_coupling_gain
        self._last_local_relative_attempt_frame = None
        self.update_count = 0
        self.candidate_count = 0
        self.rejected_count = 0
        self.pending_candidate = None
        self.last_metrics = None
        self.last_proposal_diagnostics = None
        self.last_proposal_candidate_count = 0
        self.causal_material_history.clear()
        self._next_implicit_frame_index = 0

    def _build_low_dimensional_basis(self, requested_count: int) -> torch.Tensor:
        """Build deterministic, graph-smoothed partition-of-unity weights."""
        particle_count = len(self.rest_positions)
        count = min(int(requested_count), particle_count)
        coordinates = self.rest_positions
        center = coordinates.mean(dim=0, keepdim=True)
        scale = coordinates.std(dim=0, unbiased=False).clamp_min(1.0e-6)
        normalized = (coordinates - center) / scale
        dynamic_ids = torch.nonzero(~self.fixed_mask, as_tuple=False).flatten()
        candidates = dynamic_ids if dynamic_ids.numel() else torch.arange(
            particle_count, device=coordinates.device
        )
        centroid = normalized[candidates].mean(dim=0, keepdim=True)
        first_local = torch.argmax(
            torch.sum((normalized[candidates] - centroid) ** 2, dim=1)
        )
        seeds = [int(candidates[first_local].item())]
        minimum_distance = torch.sum(
            (normalized - normalized[seeds[0]]) ** 2, dim=1
        )
        for _ in range(1, count):
            scores = minimum_distance[candidates]
            next_id = int(candidates[torch.argmax(scores)].item())
            seeds.append(next_id)
            distance = torch.sum(
                (normalized - normalized[next_id]) ** 2, dim=1
            )
            minimum_distance = torch.minimum(minimum_distance, distance)
        seed_tensor = torch.tensor(
            seeds, dtype=torch.long, device=coordinates.device
        )
        distances = torch.cdist(normalized, normalized[seed_tensor])
        assignments = torch.argmin(distances, dim=1)
        basis = torch.nn.functional.one_hot(
            assignments, num_classes=count
        ).to(dtype=torch.float32)
        if self.edges.numel():
            source, target = self.edges[:, 0], self.edges[:, 1]
            counts = self.edge_incidence_count.clamp_min(1.0).unsqueeze(1)
            # Use the requested spatial-smoothing count for the parameter
            # basis itself.  Every resulting particle field is smooth even
            # though only a handful of coefficients are optimized.
            for _ in range(self.settings.spatial_smoothing_iterations):
                sums = torch.zeros_like(basis)
                sums.index_add_(0, source, basis[target])
                sums.index_add_(0, target, basis[source])
                neighbor = sums / counts
                basis = torch.lerp(
                    basis,
                    neighbor,
                    self.settings.spatial_smoothing_blend,
                )
                basis = basis / basis.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        return basis

    def _rebuild_track_region_basis(self) -> None:
        """Map the deterministic particle regions to fixed CoTracker tracks."""
        if self.track_particle_ids is None or self.track_particle_weights is None:
            self.track_region_basis = None
            return
        particle_regions = self.low_dimensional_basis[self.track_particle_ids]
        self.track_region_basis = torch.sum(
            particle_regions * self.track_particle_weights[..., None], dim=1
        )

    def local_zero_mean_log_field(
        self,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        """Map regional coefficients to a smooth, mean-preserving log field.

        The region basis is a partition of unity.  Centering both coefficient
        space and the resulting particle field removes the global stiffness
        degree of freedom exactly.  Fixed and known grasp-control particles
        remain at their reset material, preventing a small controlled patch
        from steering or receiving the learned field.
        """
        if coefficients.shape != self.low_dimensional_log_coefficients.shape:
            raise ValueError("Local stiffness coefficients have the wrong shape")
        global_local_mode = self.settings.update_mode in {
            "differentiable_hierarchical_relative",
            "differentiable_particle_graph_lm",
        }
        regional_coefficients = coefficients[1:] if global_local_mode else coefficients
        if self.settings.update_mode == "differentiable_particle_graph_lm":
            if regional_coefficients.shape != self.initial_distance.shape:
                raise ValueError("Particle-graph field must contain one value per particle")
            raw = regional_coefficients
        else:
            centered_coefficients = regional_coefficients - regional_coefficients.mean()
            raw = self.low_dimensional_basis @ centered_coefficients
        parameter_mask = ~self.local_parameter_exclusion_mask
        weights = self.inverse_mass.clamp_min(0.0)
        weights = torch.where(parameter_mask, weights, torch.zeros_like(weights))
        denominator = weights.sum()
        if float(denominator.detach().item()) <= 0.0:
            return torch.zeros_like(raw)
        weighted_mean = torch.sum(raw * weights) / denominator
        return torch.where(
            parameter_mask,
            raw - weighted_mean,
            torch.zeros_like(raw),
        )

    def local_distance_from_coefficients(
        self,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        """Return the absolute local distance field around the reset material."""
        local_field = self.local_zero_mean_log_field(coefficients)
        global_offset = (
            coefficients[0]
            if self.settings.update_mode in {
                "differentiable_hierarchical_relative",
                "differentiable_particle_graph_lm",
            }
            else coefficients.new_zeros(())
        )
        log_field = torch.where(
            ~self.local_parameter_exclusion_mask,
            local_field + global_offset,
            torch.zeros_like(local_field),
        )
        return torch.clamp(
            self.initial_distance * torch.exp(log_field),
            self.settings.distance_minimum,
            self.settings.distance_maximum,
        )

    def local_shape_from_coefficients(
        self, coefficients: torch.Tensor
    ) -> torch.Tensor:
        """Weakly tie shape stiffness to the particle graph distance field."""
        if self.settings.update_mode != "differentiable_particle_graph_lm":
            return self.initial_shape
        local_field = self.local_zero_mean_log_field(coefficients)
        log_field = torch.where(
            ~self.local_parameter_exclusion_mask,
            coefficients[0] + local_field,
            torch.zeros_like(local_field),
        )
        return torch.clamp(
            self.initial_shape
            * torch.exp(self.settings.graph_shape_log_coupling * log_field),
            self.settings.shape_minimum,
            self.settings.shape_maximum,
        )

    def normalize_local_coefficients(
        self, coefficients: torch.Tensor
    ) -> torch.Tensor:
        """Remove only the regional gauge while preserving the global axis."""
        if coefficients.shape != self.low_dimensional_log_coefficients.shape:
            raise ValueError("Local stiffness coefficients have the wrong shape")
        result = coefficients.clone()
        if self.settings.update_mode == "differentiable_particle_graph_lm":
            active = ~self.local_parameter_exclusion_mask
            weights = torch.where(
                active,
                self.inverse_mass.clamp_min(0.0),
                torch.zeros_like(self.inverse_mass),
            )
            local = result[1:]
            weighted_mean = torch.sum(local * weights) / weights.sum().clamp_min(
                1.0e-12
            )
            result[1:] = torch.where(
                active, local - weighted_mean, torch.zeros_like(local)
            )
        elif self.settings.update_mode == "differentiable_hierarchical_relative":
            result[1:] = result[1:] - result[1:].mean()
        else:
            result = result - result.mean()
        return result

    def project_local_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        """Project regional derivatives without erasing global stiffness."""
        return self.normalize_local_coefficients(gradient)

    def local_regularization_loss(
        self,
        coefficients: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return coefficient and graph-smoothness priors for local material."""
        global_local_mode = self.settings.update_mode in {
            "differentiable_hierarchical_relative",
            "differentiable_particle_graph_lm",
        }
        regional_coefficients = coefficients[1:] if global_local_mode else coefficients
        centered = regional_coefficients - regional_coefficients.mean()
        coefficient_prior = self.settings.autograd_prior_weight * torch.mean(
            centered.square()
        )
        if global_local_mode:
            coefficient_prior = coefficient_prior + (
                self.settings.global_parameter_prior_weight
                * coefficients[0].square()
            )
        field = self.local_zero_mean_log_field(coefficients)
        if self.edges.numel():
            source, target = self.edges[:, 0], self.edges[:, 1]
            active_edge = (
                ~self.local_parameter_exclusion_mask[source]
                & ~self.local_parameter_exclusion_mask[target]
            )
            spatial = (
                torch.mean((field[target] - field[source])[active_edge].square())
                if bool(active_edge.any().item())
                else field.sum() * 0.0
            )
        else:
            spatial = field.sum() * 0.0
        spatial_prior = self.settings.local_spatial_prior_weight * spatial
        return coefficient_prior + spatial_prior, coefficient_prior, spatial_prior

    def fixed_track_centers(self, positions: torch.Tensor) -> torch.Tensor:
        """Evaluate all immutable CoTracker material bindings as ``A q``."""
        if self.track_particle_ids is None or self.track_particle_weights is None:
            raise RuntimeError("No CoTracker material bindings are configured")
        if positions.shape != self.rest_positions.shape:
            raise ValueError("Track-center positions have the wrong shape")
        supports = positions[self.track_particle_ids]
        return torch.sum(supports * self.track_particle_weights[..., None], dim=1)

    @staticmethod
    def _stable_vector_norm(
        vectors: torch.Tensor, *, dim: int = -1, minimum: float = 1.0e-9
    ) -> torch.Tensor:
        """Vector norm whose backward is finite for an exactly zero vector.

        Clamping ``torch.linalg.vector_norm(v)`` after the square root does
        not protect its backward: at ``v == 0`` SqrtBackward can still form
        ``0 / 0`` before ClampBackward masks the result.  Clamp the squared
        norm first so both the forward value and its local derivative are
        well-defined.  Away from the tiny floor this is the ordinary L2 norm.
        """
        squared = torch.sum(vectors.square(), dim=dim)
        return torch.sqrt(squared.clamp_min(float(minimum) ** 2))

    def _project_distance_constraints(
        self,
        positions: torch.Tensor,
        particle_stiffness: torch.Tensor,
        inverse_mass: torch.Tensor,
        dt: float,
        constraint_lambda: torch.Tensor,
        frozen_mask: torch.Tensor,
        frozen_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.edges.numel():
            return positions, constraint_lambda
        settings = self.settings
        source, target = self.edges[:, 0], self.edges[:, 1]
        difference = positions[source] - positions[target]
        length = self._stable_vector_norm(difference, dim=1)
        gradient = difference / length[:, None]
        edge_stiffness = 0.5 * (
            particle_stiffness[source] + particle_stiffness[target]
        ).clamp_min(1.0e-8)
        alpha = settings.autograd_pbd_compliance_scale / (
            edge_stiffness * dt * dt
        )
        delta_lambda = (
            -(length - self.rest_edge_lengths) - alpha * constraint_lambda
        ) / (inverse_mass[source] + inverse_mass[target] + alpha).clamp_min(
            1.0e-12
        )
        delta_lambda = delta_lambda * settings.autograd_pbd_relaxation
        constraint_lambda = constraint_lambda + delta_lambda
        corner_source = (
            inverse_mass[source] * delta_lambda
        )[:, None] * gradient
        corner_target = -(
            inverse_mass[target] * delta_lambda
        )[:, None] * gradient
        deltas = torch.zeros_like(positions)
        deltas.index_add_(0, source, corner_source)
        deltas.index_add_(0, target, corner_target)
        positions = positions + deltas / self.edge_incidence_count.clamp_min(
            1.0
        )[:, None]
        positions = torch.where(
            frozen_mask[:, None], frozen_positions, positions
        )
        return positions, constraint_lambda

    def _reduce_tet_corner_deltas(
        self, corner_deltas: torch.Tensor
    ) -> torch.Tensor:
        deltas = torch.zeros_like(self.rest_positions)
        deltas.index_add_(
            0,
            self.tet_indices.reshape(-1),
            corner_deltas.reshape(-1, 3),
        )
        return deltas / self.tet_incidence_count.clamp_min(1.0)[:, None]

    def _project_volume_constraints(
        self,
        positions: torch.Tensor,
        inverse_mass: torch.Tensor,
        dt: float,
        constraint_lambda: torch.Tensor,
        frozen_mask: torch.Tensor,
        frozen_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.tet_indices.numel():
            return positions, constraint_lambda
        settings = self.settings
        tets = self.tet_indices
        x = positions[tets]
        x0, x1, x2, x3 = x.unbind(dim=1)
        gradient_1 = torch.linalg.cross(x2 - x0, x3 - x0) / 6.0
        gradient_2 = torch.linalg.cross(x3 - x0, x1 - x0) / 6.0
        gradient_3 = torch.linalg.cross(x1 - x0, x2 - x0) / 6.0
        gradient_0 = -gradient_1 - gradient_2 - gradient_3
        gradients = torch.stack(
            (gradient_0, gradient_1, gradient_2, gradient_3), dim=1
        )
        weights = inverse_mass[tets]
        weighted_gradient = torch.sum(
            weights * torch.sum(gradients.square(), dim=2), dim=1
        )
        current_volume = torch.sum(
            torch.linalg.cross(x1 - x0, x2 - x0) * (x3 - x0), dim=1
        ) / 6.0
        alpha = settings.autograd_pbd_compliance_scale / (
            settings.autograd_volume_stiffness * dt * dt
        )
        delta_lambda = (
            -(current_volume - self.rest_tet_volumes)
            - alpha * constraint_lambda
        ) / (weighted_gradient + alpha).clamp_min(1.0e-20)
        delta_lambda = delta_lambda * settings.autograd_pbd_relaxation
        constraint_lambda = constraint_lambda + delta_lambda
        corner_deltas = (
            weights[:, :, None] * delta_lambda[:, None, None] * gradients
        )
        positions = positions + self._reduce_tet_corner_deltas(corner_deltas)
        positions = torch.where(
            frozen_mask[:, None], frozen_positions, positions
        )
        return positions, constraint_lambda

    def _project_shape_constraints(
        self,
        positions: torch.Tensor,
        particle_stiffness: torch.Tensor,
        inverse_mass: torch.Tensor,
        dt: float,
        constraint_lambda: torch.Tensor,
        frozen_mask: torch.Tensor,
        frozen_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.tet_indices.numel():
            return positions, constraint_lambda
        settings = self.settings
        tets = self.tet_indices
        x = positions[tets]
        center = x.mean(dim=1, keepdim=True)
        current_pose = torch.stack(
            (x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]),
            dim=-1,
        )
        # Warp's paper projector treats the polar goal as fixed during each
        # local XPBD projection.  Detaching only the SVD mirrors that local
        # approximation while retaining gradients through all residuals.
        with torch.no_grad():
            deformation = (
                current_pose.detach() @ self.rest_tet_inverse_pose
            )
            left, _singular, right_h = torch.linalg.svd(deformation)
            reflection = torch.ones(
                (len(tets), 3),
                dtype=positions.dtype,
                device=positions.device,
            )
            raw_rotation = left @ right_h
            reflection[:, 2] = torch.where(
                torch.linalg.det(raw_rotation) < 0.0,
                -torch.ones_like(reflection[:, 2]),
                reflection[:, 2],
            )
            rotation = left @ torch.diag_embed(reflection) @ right_h
        goal = center + torch.einsum(
            "tij,tkj->tki", rotation, self.rest_tet_relative
        )
        residual = x - goal
        squared_constraint = torch.sum(residual.square(), dim=(1, 2))
        active = squared_constraint > 1.0e-8 * self.rest_tet_squared_extent
        constraint = torch.sqrt(squared_constraint.clamp_min(1.0e-20))
        gradients = residual / constraint[:, None, None]
        gradients = torch.where(
            active[:, None, None], gradients, torch.zeros_like(gradients)
        )
        constraint = torch.where(active, constraint, torch.zeros_like(constraint))
        weights = inverse_mass[tets]
        weighted_gradient = torch.sum(
            weights * torch.sum(gradients.square(), dim=2), dim=1
        )
        stiffness = particle_stiffness[tets].mean(dim=1).clamp_min(1.0e-8)
        alpha = settings.autograd_pbd_compliance_scale / (
            stiffness * dt * dt
        )
        delta_lambda = (
            -constraint - alpha * constraint_lambda
        ) / (weighted_gradient + alpha).clamp_min(1.0e-20)
        delta_lambda = torch.where(
            active,
            delta_lambda * settings.autograd_pbd_relaxation,
            torch.zeros_like(delta_lambda),
        )
        constraint_lambda = constraint_lambda + delta_lambda
        corner_deltas = (
            weights[:, :, None] * delta_lambda[:, None, None] * gradients
        )
        positions = positions + self._reduce_tet_corner_deltas(corner_deltas)
        positions = torch.where(
            frozen_mask[:, None], frozen_positions, positions
        )
        return positions, constraint_lambda

    def _differentiable_full_pbd_step(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        distance_stiffness: torch.Tensor,
        shape_stiffness: torch.Tensor,
        frozen_mask: torch.Tensor,
        frozen_positions: torch.Tensor,
        dt: float,
        velocity_damping_per_second: torch.Tensor | float | None = None,
        control_mask: torch.Tensor | None = None,
        control_positions: torch.Tensor | None = None,
        control_velocities: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One complete causal step: integrate, then distance/volume/shape."""
        inverse_mass = torch.where(
            frozen_mask, torch.zeros_like(self.inverse_mass), self.inverse_mass
        )
        damping_rate = (
            self.settings.autograd_velocity_damping_per_second
            if velocity_damping_per_second is None
            else velocity_damping_per_second
        )
        substep_dt = dt / float(self.settings.autograd_integration_substeps)
        if isinstance(damping_rate, torch.Tensor):
            damping = torch.exp(-damping_rate * substep_dt)
        else:
            damping = math.exp(-float(damping_rate) * substep_dt)
        if control_mask is not None:
            if (
                control_positions is None
                or control_velocities is None
                or control_mask.shape != self.fixed_mask.shape
                or control_positions.shape != positions.shape
                or control_velocities.shape != velocities.shape
            ):
                raise ValueError("Differentiable control boundary is incomplete")
            positions = torch.where(
                control_mask[:, None], control_positions, positions
            )
            velocities = torch.where(
                control_mask[:, None], control_velocities, velocities
            )
        for substep_index in range(self.settings.autograd_integration_substeps):
            previous_positions = positions
            predicted = positions + velocities * substep_dt
            predicted = torch.where(
                frozen_mask[:, None], frozen_positions, predicted
            )
            projected = predicted
            distance_lambda = torch.zeros(
                len(self.edges), dtype=positions.dtype, device=positions.device
            )
            volume_lambda = torch.zeros(
                len(self.tet_indices),
                dtype=positions.dtype,
                device=positions.device,
            )
            shape_lambda = torch.zeros_like(volume_lambda)
            for _ in range(self.settings.autograd_material_iterations):
                projected, distance_lambda = self._project_distance_constraints(
                    projected,
                    distance_stiffness,
                    inverse_mass,
                    substep_dt,
                    distance_lambda,
                    frozen_mask,
                    frozen_positions,
                )
                projected, volume_lambda = self._project_volume_constraints(
                    projected,
                    inverse_mass,
                    substep_dt,
                    volume_lambda,
                    frozen_mask,
                    frozen_positions,
                )
                projected, shape_lambda = self._project_shape_constraints(
                    projected,
                    shape_stiffness,
                    inverse_mass,
                    substep_dt,
                    shape_lambda,
                    frozen_mask,
                    frozen_positions,
                )
            velocities = damping * (
                velocities
                + self.settings.autograd_projection_velocity_scale
                * (projected - predicted)
                / substep_dt
            )
            frozen_velocity = (
                frozen_positions - previous_positions
            ) / substep_dt
            velocities = torch.where(
                frozen_mask[:, None], frozen_velocity, velocities
            )
            positions = projected
            # The authoritative Warp path applies the known grasp boundary
            # before and after each full physics step, not inside every one of
            # its twelve material substeps.  Mirror that cadence so local
            # stiffness gradients do not depend on an artificially rigid jaw
            # patch in the Torch direction check.
            if (
                control_mask is not None
                and (substep_index + 1)
                % self.settings.autograd_control_projection_interval_substeps
                == 0
            ):
                positions = torch.where(
                    control_mask[:, None], control_positions, positions
                )
                velocities = torch.where(
                    control_mask[:, None], control_velocities, velocities
                )
        return positions, velocities

    def _append_causal_observation(
        self,
        *,
        frame_index: int | None,
        rollout_start_positions: torch.Tensor,
        corrected_positions: torch.Tensor,
        physical_velocities: torch.Tensor | None,
        eligible: torch.Tensor,
        material_active: torch.Tensor,
        control_excluded: torch.Tensor,
        control_frozen: torch.Tensor,
        control_coupling_base_positions: torch.Tensor | None = None,
        control_coupling_displacements: torch.Tensor | None = None,
        track_target_positions: torch.Tensor | None = None,
        track_valid_mask: torch.Tensor | None = None,
    ) -> None:
        if frame_index is None:
            frame_index = self._next_implicit_frame_index
        frame_index = int(frame_index)
        self._next_implicit_frame_index = frame_index + 1
        history = self.causal_material_history
        preserve_discontinuous_window = (
            self.settings.update_mode == "differentiable_global_mhe"
        )
        if (
            history
            and frame_index != history[-1].frame_index + 1
            and not preserve_discontinuous_window
        ):
            history.clear()
        start = rollout_start_positions.detach().to(
            device=self.rest_positions.device, dtype=torch.float32
        )
        if start.shape != self.rest_positions.shape:
            raise ValueError("Causal rollout start positions have the wrong shape")
        if physical_velocities is None:
            if history and frame_index == history[-1].frame_index + 1:
                dt = self.settings.autograd_frame_dt
                physical_velocities = (
                    start - history[-1].rollout_start_positions
                ) / dt
            else:
                physical_velocities = torch.zeros_like(corrected_positions)
        velocity = physical_velocities.detach().to(
            device=self.rest_positions.device, dtype=torch.float32
        )
        if velocity.shape != self.rest_positions.shape:
            raise ValueError("Causal stiffness velocity has the wrong shape")
        coupling_base = (
            start
            if control_coupling_base_positions is None
            else control_coupling_base_positions.detach().to(
                device=self.rest_positions.device, dtype=torch.float32
            )
        )
        coupling_displacement = (
            corrected_positions.detach() - start
            if control_coupling_displacements is None
            else control_coupling_displacements.detach().to(
                device=self.rest_positions.device, dtype=torch.float32
            )
        )
        if coupling_base.shape != self.rest_positions.shape:
            raise ValueError("Control-coupling base positions have wrong shape")
        if coupling_displacement.shape != self.rest_positions.shape:
            raise ValueError("Control-coupling displacements have wrong shape")
        stored_track_targets = None
        stored_track_valid = None
        if track_target_positions is not None or track_valid_mask is not None:
            if (
                track_target_positions is None
                or track_valid_mask is None
                or self.track_valid_mask is None
            ):
                raise ValueError("Track targets, validity and bindings are required")
            stored_track_targets = track_target_positions.detach().to(
                device=self.rest_positions.device, dtype=torch.float32
            )
            stored_track_valid = track_valid_mask.detach().to(
                device=self.rest_positions.device, dtype=torch.bool
            )
            track_count = len(self.track_valid_mask)
            if stored_track_targets.shape != (track_count, 3):
                raise ValueError("Track targets have the wrong shape")
            if stored_track_valid.shape != (track_count,):
                raise ValueError("Track validity has the wrong shape")
            stored_track_valid &= self.track_valid_mask
        history.append(
            _CausalMaterialObservation(
                frame_index=frame_index,
                rollout_start_positions=start.clone(),
                corrected_positions=corrected_positions.detach().clone(),
                physical_velocities=velocity.clone(),
                control_coupling_base_positions=coupling_base.clone(),
                control_coupling_displacements=coupling_displacement.clone(),
                eligible_mask=eligible.detach().clone(),
                material_active_mask=material_active.detach().clone(),
                control_exclusion_mask=control_excluded.detach().clone(),
                control_frozen_mask=control_frozen.detach().clone(),
                track_target_positions=(
                    None
                    if stored_track_targets is None
                    else stored_track_targets.clone()
                ),
                track_valid_mask=(
                    None if stored_track_valid is None else stored_track_valid.clone()
                ),
            )
        )
        maximum_observations = (
            self.settings.observable_window_size
            if preserve_discontinuous_window
            else self.settings.autograd_unroll_steps + 1
        )
        if len(history) > maximum_observations:
            del history[:-maximum_observations]

    def _propose_differentiable_low_dimensional(
        self,
        *,
        physical_prediction: torch.Tensor,
        accepted_residual: torch.Tensor,
        rollout_start_positions: torch.Tensor,
        physical_velocities: torch.Tensor | None,
        frame_index: int | None,
        eligible: torch.Tensor,
        material_active: torch.Tensor,
        control_excluded: torch.Tensor,
        control_frozen: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        dict[str, float | int | str],
    ]:
        """Differentiate 3..5 already-observed full PBD transitions."""
        settings = self.settings
        active = eligible & material_active
        coefficient = self.low_dimensional_log_coefficients.detach()
        start = rollout_start_positions.detach().to(
            device=self.rest_positions.device, dtype=torch.float32
        )
        if start.shape != physical_prediction.shape:
            raise ValueError("Differentiable PBD start positions have wrong shape")
        corrected_positions = physical_prediction + accepted_residual
        self._append_causal_observation(
            frame_index=frame_index,
            rollout_start_positions=start,
            corrected_positions=corrected_positions,
            physical_velocities=physical_velocities,
            eligible=eligible,
            material_active=material_active,
            control_excluded=control_excluded,
            control_frozen=control_frozen,
        )
        # Preserve the established low-dimensional warm-up protocol (four
        # observations yield three optimized transitions), but each selected
        # transition now starts from its real supplied rollout state.
        temporal_steps = len(self.causal_material_history) - 1
        base_distance = self.distance_stiffness.detach().clone()
        base_shape = self.shape_stiffness.detach().clone()
        zero_particle = torch.zeros_like(base_distance)
        zero_coefficient = torch.zeros_like(coefficient)
        if temporal_steps < 3:
            metrics: dict[str, float | int | str] = {
                "update_mode": settings.update_mode,
                "autograd_unroll_steps": settings.autograd_unroll_steps,
                "autograd_temporal_steps": temporal_steps,
                "autograd_material_iterations": settings.autograd_material_iterations,
                "autograd_integration_substeps": settings.autograd_integration_substeps,
                "autograd_region_count": len(coefficient),
                "autograd_active_regions": 0,
                "autograd_objective": 0.0,
                "autograd_data_loss": 0.0,
                "autograd_position_loss": 0.0,
                "autograd_edge_strain_loss": 0.0,
                "autograd_edge_strain_weight": float(
                    settings.strain_signal_weight
                ),
                "autograd_prior_loss": 0.0,
                "autograd_gradient_norm": 0.0,
                "autograd_coefficient_step_maximum": 0.0,
                "autograd_coefficient_minimum": float(coefficient.min().item()),
                "autograd_coefficient_median": float(coefficient.median().item()),
                "autograd_coefficient_maximum": float(coefficient.max().item()),
                "autograd_shape_frozen": 1,
                "autograd_history_status": "warming_up",
            }
            return (
                base_distance,
                base_shape,
                zero_particle,
                zero_particle,
                coefficient.clone(),
                self.adam_first_moment.clone(),
                self.adam_second_moment.clone(),
                self.adam_step,
                metrics,
            )
        variable = coefficient.clone().requires_grad_(True)
        coefficient_delta = variable - coefficient
        particle_delta = self.low_dimensional_basis @ coefficient_delta
        # Strictly preserve the verified stiffness of fixed, controlled,
        # quality-invalid, and unsupervised particles.  The previous code
        # masked only the exported log_step after globally changing the field.
        particle_delta = torch.where(eligible, particle_delta, zero_particle)
        distance = torch.clamp(
            base_distance * torch.exp(particle_delta),
            settings.distance_minimum,
            settings.distance_maximum,
        )
        observations = self.causal_material_history
        position_losses: list[torch.Tensor] = []
        strain_losses: list[torch.Tensor] = []
        accumulated_active = torch.zeros_like(base_distance)
        source, target = self.edges[:, 0], self.edges[:, 1]
        for observation in observations[-temporal_steps:]:
            dt = settings.autograd_frame_dt
            frozen = self.fixed_mask | observation.control_frozen_mask
            rollout_positions, _rollout_velocities = (
                self._differentiable_full_pbd_step(
                    observation.rollout_start_positions,
                    observation.physical_velocities,
                    distance,
                    base_shape,
                    frozen,
                    observation.corrected_positions,
                    dt,
                )
            )
            target_positions = observation.corrected_positions
            step_active = (
                observation.eligible_mask
                & observation.material_active_mask
            )
            accumulated_active = accumulated_active + step_active.to(
                dtype=torch.float32
            )
            scaled_error = (
                rollout_positions - target_positions
            ) / settings.residual_full_scale_m
            node_loss = torch.sqrt(
                torch.sum(scaled_error.square(), dim=1) + 1.0e-4
            ) - 0.01
            if bool(step_active.any().item()):
                position_losses.append(node_loss[step_active].mean())
            if self.edges.numel():
                edge_active = (
                    (step_active[source] | step_active[target])
                    & ~observation.control_exclusion_mask[source]
                    & ~observation.control_exclusion_mask[target]
                )
                rollout_length = self._stable_vector_norm(
                    rollout_positions[target] - rollout_positions[source], dim=1
                )
                target_length = self._stable_vector_norm(
                    target_positions[target] - target_positions[source], dim=1
                )
                strain_error = (
                    (rollout_length - target_length) / self.rest_edge_lengths
                ) / settings.edge_strain_full_scale
                robust_strain = torch.sqrt(
                    strain_error.square() + 1.0e-4
                ) - 0.01
                if bool(edge_active.any().item()):
                    strain_losses.append(robust_strain[edge_active].mean())
        position_loss = (
            torch.stack(position_losses).mean()
            if position_losses
            else variable.sum() * 0.0
        )
        edge_strain_loss = (
            torch.stack(strain_losses).mean()
            if strain_losses
            else variable.sum() * 0.0
        )
        blend = settings.strain_signal_weight
        data_loss = (1.0 - blend) * position_loss + blend * edge_strain_loss
        region_weight = self.low_dimensional_basis.T @ accumulated_active
        active_regions = region_weight > 1.0e-6
        prior_loss = settings.autograd_prior_weight * torch.mean(variable * variable)
        objective = data_loss + prior_loss
        (gradient,) = torch.autograd.grad(objective, variable)
        finite_gradient = torch.isfinite(gradient)
        gradient = torch.where(finite_gradient, gradient, torch.zeros_like(gradient))

        next_step = self.adam_step + 1
        beta1 = settings.signal_ema_decay
        beta2 = settings.autograd_adam_beta2
        first = beta1 * self.adam_first_moment + (1.0 - beta1) * gradient
        second = beta2 * self.adam_second_moment + (1.0 - beta2) * gradient.square()
        first_hat = first / (1.0 - beta1**next_step)
        second_hat = second / (1.0 - beta2**next_step)
        coefficient_step = -settings.log_learning_rate * first_hat / (
            torch.sqrt(second_hat) + settings.autograd_adam_epsilon
        )
        coefficient_step = torch.clamp(
            coefficient_step,
            min=-settings.maximum_log_step,
            max=settings.maximum_log_step,
        )
        coefficient_step = torch.where(
            active_regions & finite_gradient,
            coefficient_step,
            torch.zeros_like(coefficient_step),
        )
        candidate_coefficient = torch.clamp(
            coefficient + coefficient_step,
            min=-settings.autograd_maximum_log_offset,
            max=settings.autograd_maximum_log_offset,
        )
        effective_coefficient_step = candidate_coefficient - coefficient
        particle_log_step = self.low_dimensional_basis @ effective_coefficient_step
        particle_log_step = torch.where(
            eligible, particle_log_step, zero_particle
        )
        candidate_distance = torch.clamp(
            base_distance * torch.exp(particle_log_step),
            settings.distance_minimum,
            settings.distance_maximum,
        )
        if settings.autograd_shape_update_gain > 0.0:
            candidate_shape = torch.clamp(
                base_shape
                * torch.exp(
                    particle_log_step * settings.autograd_shape_update_gain
                ),
                settings.shape_minimum,
                settings.shape_maximum,
            )
        else:
            candidate_shape = base_shape
        normalized_descent = -gradient / gradient.abs().max().clamp_min(1.0e-12)
        particle_gradient_signal = self.low_dimensional_basis @ normalized_descent
        metrics: dict[str, float | int | str] = {
            "update_mode": settings.update_mode,
            "autograd_unroll_steps": settings.autograd_unroll_steps,
            "autograd_temporal_steps": temporal_steps,
            "autograd_material_iterations": settings.autograd_material_iterations,
            "autograd_integration_substeps": settings.autograd_integration_substeps,
            "autograd_region_count": len(coefficient),
            "autograd_active_regions": int(torch.count_nonzero(active_regions).item()),
            "autograd_objective": float(objective.detach().item()),
            "autograd_data_loss": float(data_loss.detach().item()),
            "autograd_position_loss": float(position_loss.detach().item()),
            "autograd_edge_strain_loss": float(
                edge_strain_loss.detach().item()
            ),
            "autograd_edge_strain_weight": float(blend),
            "autograd_prior_loss": float(prior_loss.detach().item()),
            "autograd_gradient_norm": float(torch.linalg.vector_norm(gradient).item()),
            "autograd_coefficient_step_maximum": float(
                effective_coefficient_step.abs().max().item()
            ),
            "autograd_coefficient_minimum": float(candidate_coefficient.min().item()),
            "autograd_coefficient_median": float(candidate_coefficient.median().item()),
            "autograd_coefficient_maximum": float(candidate_coefficient.max().item()),
            "autograd_shape_frozen": int(settings.autograd_shape_update_gain == 0.0),
            "autograd_history_status": "ready",
        }
        return (
            candidate_distance,
            candidate_shape,
            particle_log_step,
            particle_gradient_signal,
            candidate_coefficient.detach(),
            first.detach(),
            second.detach(),
            next_step,
            metrics,
        )

    def _propose_differentiable_local_relative(
        self,
        *,
        physical_prediction: torch.Tensor,
        accepted_residual: torch.Tensor,
        rollout_start_positions: torch.Tensor,
        physical_velocities: torch.Tensor | None,
        frame_index: int | None,
        eligible: torch.Tensor,
        material_active: torch.Tensor,
        control_excluded: torch.Tensor,
        control_frozen: torch.Tensor,
        control_coupling_base_positions: torch.Tensor | None,
        control_coupling_displacements: torch.Tensor | None,
        track_target_positions: torch.Tensor | None,
        track_valid_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        dict[str, float | int | str],
    ]:
        """Fit a smooth local zero-mean stiffness field from past RGB tracks.

        A single open-loop trajectory starts at the oldest already observed
        source state.  Losses are evaluated at H=1/3/5 before any intermediate
        visual residual is written back.  The state correction path remains
        live, while this delayed material path only changes later PBD steps.
        """
        settings = self.settings
        if settings.autograd_unroll_steps != 5:
            raise ValueError("Local-relative stiffness requires a five-frame window")
        with torch.no_grad():
            self.local_parameter_exclusion_mask |= (
                control_excluded | control_frozen | self.fixed_mask
            )
        start = rollout_start_positions.detach().to(
            device=self.rest_positions.device, dtype=torch.float32
        )
        if start.shape != physical_prediction.shape:
            raise ValueError("Local-relative PBD start positions have wrong shape")
        corrected_positions = physical_prediction + accepted_residual
        self._append_causal_observation(
            frame_index=frame_index,
            rollout_start_positions=start,
            corrected_positions=corrected_positions,
            physical_velocities=physical_velocities,
            eligible=eligible,
            material_active=material_active,
            control_excluded=control_excluded,
            control_frozen=control_frozen,
            control_coupling_base_positions=control_coupling_base_positions,
            control_coupling_displacements=control_coupling_displacements,
            track_target_positions=track_target_positions,
            track_valid_mask=track_valid_mask,
        )
        observations = self.causal_material_history[-5:]
        temporal_steps = len(observations)
        coefficient = self.low_dimensional_log_coefficients.detach()
        base_distance = self.distance_stiffness.detach().clone()
        base_shape = self.shape_stiffness.detach().clone()
        zero_particle = torch.zeros_like(base_distance)
        zero_coefficient = torch.zeros_like(coefficient)
        current_frame = int(
            observations[-1].frame_index
            if observations
            else (0 if frame_index is None else frame_index)
        )
        update_interval = (
            settings.graph_update_interval_frames
            if settings.update_mode == "differentiable_particle_graph_lm"
            else settings.local_update_interval_frames
        )
        cadence_ready = (
            self._last_local_relative_attempt_frame is None
            or current_frame - self._last_local_relative_attempt_frame
            >= update_interval
        )
        if temporal_steps < 5 or not cadence_ready:
            history_status = (
                "warming_up_h1_h3_h5"
                if temporal_steps < 5
                else "waiting_for_persistent_interval"
            )
            metrics: dict[str, float | int | str] = {
                "update_mode": settings.update_mode,
                "autograd_unroll_steps": settings.autograd_unroll_steps,
                "autograd_temporal_steps": temporal_steps,
                "autograd_region_count": len(coefficient),
                "autograd_objective": 0.0,
                "autograd_data_loss": 0.0,
                "autograd_position_loss": 0.0,
                "autograd_edge_strain_loss": 0.0,
                "autograd_prior_loss": 0.0,
                "local_spatial_prior_loss": 0.0,
                "autograd_gradient_norm": 0.0,
                "autograd_history_status": history_status,
                "gradient_consistency_status": history_status,
                "local_rollout_objective": (
                    "203track_cauchy_relative_pre_residual_h1_h3_h5"
                ),
                "local_global_mean_frozen": int(
                    settings.update_mode
                    not in {
                        "differentiable_hierarchical_relative",
                        "differentiable_particle_graph_lm",
                    }
                ),
                "local_velocity_damping_frozen": 1,
                "local_update_interval_frames": (
                    update_interval
                ),
            }
            return (
                base_distance,
                base_shape,
                zero_particle,
                zero_particle,
                coefficient.clone(),
                self.adam_first_moment.clone(),
                self.adam_second_moment.clone(),
                self.adam_step,
                zero_coefficient,
                metrics,
            )

        self._last_local_relative_attempt_frame = current_frame
        variable = self.normalize_local_coefficients(coefficient).requires_grad_(
            True
        )
        distance = self.local_distance_from_coefficients(variable)
        shape = self.local_shape_from_coefficients(variable)
        position_losses: list[tuple[float, torch.Tensor]] = []
        relative_losses: list[tuple[float, torch.Tensor]] = []
        strain_losses: list[tuple[float, torch.Tensor]] = []
        horizon_map = {
            1: float(settings.local_horizon_weights[0]),
            3: float(settings.local_horizon_weights[1]),
            5: float(settings.local_horizon_weights[2]),
        }
        rollout_positions = observations[0].rollout_start_positions
        rollout_velocities = observations[0].physical_velocities
        source, target = self.edges[:, 0], self.edges[:, 1]
        active_any = False
        track_loss_count = 0
        relative_pair_count = 0
        active_horizons: list[int] = []
        for horizon, observation in enumerate(observations, start=1):
            controlled_target = (
                observation.control_coupling_base_positions
                + observation.control_coupling_displacements
            )
            control_velocity = (
                controlled_target - rollout_positions
            ) / settings.autograd_frame_dt
            frozen_positions = observation.corrected_positions
            frozen = self.fixed_mask
            rollout_positions, rollout_velocities = self._differentiable_full_pbd_step(
                rollout_positions,
                rollout_velocities,
                distance,
                shape,
                frozen,
                frozen_positions,
                settings.autograd_frame_dt,
                velocity_damping_per_second=(
                    self.initial_velocity_damping_per_second
                ),
                control_mask=observation.control_frozen_mask,
                control_positions=controlled_target,
                control_velocities=control_velocity,
            )
            if horizon not in horizon_map:
                continue
            horizon_weight = horizon_map[horizon]
            step_active = (
                observation.eligible_mask
                & observation.material_active_mask
                & ~self.local_parameter_exclusion_mask
            )
            if (
                observation.track_target_positions is None
                or observation.track_valid_mask is None
                or self.track_region_basis is None
            ):
                continue
            predicted_tracks = self.fixed_track_centers(rollout_positions)
            support_coverage = torch.sum(
                self.track_particle_weights
                * step_active[self.track_particle_ids].to(torch.float32),
                dim=1,
            )
            track_active = observation.track_valid_mask & (support_coverage >= 0.50)
            (
                node_loss,
                relative_loss,
                _relative_active,
                pair_count,
            ) = self.robust_relative_track_losses(
                predicted_tracks,
                observation.track_target_positions,
                track_active,
            )
            if bool(track_active.any().item()):
                distribution_loss, *_components = self.global_position_distribution_loss(
                    node_loss,
                    track_active,
                    self.track_region_basis,
                )
                relative_weight = settings.global_relative_track_weight
                combined_position = (
                    (1.0 - relative_weight) * distribution_loss
                    + relative_weight * relative_loss
                )
                position_losses.append((horizon_weight, combined_position))
                relative_losses.append((horizon_weight, relative_loss))
                active_horizons.append(horizon)
                active_any = True
                track_loss_count += int(track_active.sum().item())
                relative_pair_count += int(pair_count)
            if self.edges.numel():
                edge_active = (
                    (step_active[source] | step_active[target])
                    & ~observation.control_exclusion_mask[source]
                    & ~observation.control_exclusion_mask[target]
                    & ~self.local_parameter_exclusion_mask[source]
                    & ~self.local_parameter_exclusion_mask[target]
                )
                if bool(edge_active.any().item()):
                    rollout_length = self._stable_vector_norm(
                        rollout_positions[target] - rollout_positions[source], dim=1
                    )
                    target_length = self._stable_vector_norm(
                        observation.corrected_positions[target]
                        - observation.corrected_positions[source],
                        dim=1,
                    )
                    strain_error = (
                        (rollout_length - target_length) / self.rest_edge_lengths
                    ) / settings.edge_strain_full_scale
                    robust_strain = torch.sqrt(
                        strain_error.square() + 1.0e-4
                    ) - 0.01
                    strain_losses.append(
                        (horizon_weight, robust_strain[edge_active].mean())
                    )

        def weighted_average(
            values: list[tuple[float, torch.Tensor]],
        ) -> torch.Tensor:
            return (
                sum(weight * value for weight, value in values)
                / sum(weight for weight, _value in values)
                if values
                else variable.sum() * 0.0
            )

        position_loss = weighted_average(position_losses)
        relative_loss = weighted_average(relative_losses)
        edge_strain_loss = weighted_average(strain_losses)
        blend = settings.strain_signal_weight
        data_loss = (1.0 - blend) * position_loss + blend * edge_strain_loss
        prior_loss, coefficient_prior, spatial_prior = (
            self.local_regularization_loss(variable)
        )
        objective = data_loss + prior_loss
        (raw_gradient,) = torch.autograd.grad(objective, variable)
        finite_gradient = torch.isfinite(raw_gradient)
        gradient = torch.where(
            finite_gradient, raw_gradient, torch.zeros_like(raw_gradient)
        )
        gradient = self.project_local_gradient(gradient)

        next_step = self.adam_step + 1
        beta1 = settings.signal_ema_decay
        beta2 = settings.autograd_adam_beta2
        first = beta1 * self.adam_first_moment + (1.0 - beta1) * gradient
        second = beta2 * self.adam_second_moment + (1.0 - beta2) * gradient.square()
        first_hat = first / (1.0 - beta1**next_step)
        second_hat = second / (1.0 - beta2**next_step)
        coefficient_step = -settings.log_learning_rate * first_hat / (
            torch.sqrt(second_hat) + settings.autograd_adam_epsilon
        )
        coefficient_step = torch.clamp(
            coefficient_step,
            min=-settings.maximum_log_step,
            max=settings.maximum_log_step,
        )
        coefficient_step = torch.where(
            finite_gradient
            & torch.full_like(finite_gradient, active_any, dtype=torch.bool),
            coefficient_step,
            torch.zeros_like(coefficient_step),
        )
        coefficient_step = self.project_local_gradient(coefficient_step)
        candidate_coefficient = self.normalize_local_coefficients(
            coefficient + coefficient_step
        )
        candidate_coefficient = torch.clamp(
            candidate_coefficient,
            min=-settings.autograd_maximum_log_offset,
            max=settings.autograd_maximum_log_offset,
        )
        candidate_coefficient = self.normalize_local_coefficients(
            candidate_coefficient
        )
        effective_step = candidate_coefficient - coefficient
        candidate_distance = self.local_distance_from_coefficients(
            candidate_coefficient
        )
        candidate_shape = self.local_shape_from_coefficients(
            candidate_coefficient
        )
        particle_log_step = torch.log(
            candidate_distance / base_distance.clamp_min(1.0e-12)
        )
        descent = -gradient / gradient.abs().max().clamp_min(1.0e-12)
        particle_signal = self.local_zero_mean_log_field(descent)
        log_field = self.local_zero_mean_log_field(candidate_coefficient)
        dynamic_weights = torch.where(
            ~self.local_parameter_exclusion_mask,
            self.inverse_mass.clamp_min(0.0),
            torch.zeros_like(self.inverse_mass),
        )
        weighted_log_mean = torch.sum(log_field * dynamic_weights) / (
            dynamic_weights.sum().clamp_min(1.0e-12)
        )
        metrics = {
            "update_mode": settings.update_mode,
            "autograd_unroll_steps": settings.autograd_unroll_steps,
            "autograd_temporal_steps": temporal_steps,
            "autograd_material_iterations": settings.autograd_material_iterations,
            "autograd_integration_substeps": settings.autograd_integration_substeps,
            "autograd_region_count": len(coefficient),
            "autograd_objective": float(objective.detach().item()),
            "autograd_data_loss": float(data_loss.detach().item()),
            "autograd_position_loss": float(position_loss.detach().item()),
            "local_relative_track_loss": float(relative_loss.detach().item()),
            "local_relative_track_pair_count": int(relative_pair_count),
            "local_track_loss_count": int(track_loss_count),
            "autograd_edge_strain_loss": float(edge_strain_loss.detach().item()),
            "autograd_edge_strain_weight": float(blend),
            "autograd_prior_loss": float(prior_loss.detach().item()),
            "local_coefficient_prior_loss": float(coefficient_prior.detach().item()),
            "local_spatial_prior_loss": float(spatial_prior.detach().item()),
            "local_spatial_prior_weight": float(settings.local_spatial_prior_weight),
            "autograd_gradient_norm": float(torch.linalg.vector_norm(gradient).item()),
            "autograd_coefficient_step_maximum": float(effective_step.abs().max().item()),
            "autograd_coefficient_minimum": float(candidate_coefficient.min().item()),
            "autograd_coefficient_median": float(candidate_coefficient.median().item()),
            "autograd_coefficient_maximum": float(candidate_coefficient.max().item()),
            "local_particle_log_minimum": float(log_field.min().item()),
            "local_particle_log_median": float(log_field.median().item()),
            "local_particle_log_maximum": float(log_field.max().item()),
            "local_weighted_particle_log_mean": float(weighted_log_mean.item()),
            "local_horizon_weight_h1": float(settings.local_horizon_weights[0]),
            "local_horizon_weight_h3": float(settings.local_horizon_weights[1]),
            "local_horizon_weight_h5": float(settings.local_horizon_weights[2]),
            "local_active_horizons": ",".join(str(value) for value in active_horizons),
            "local_rollout_objective": (
                "203track_cauchy_relative_pre_residual_h1_h3_h5"
            ),
            "local_global_mean_frozen": int(
                settings.update_mode
                not in {
                    "differentiable_hierarchical_relative",
                    "differentiable_particle_graph_lm",
                }
            ),
            "hierarchical_global_log_distance": float(
                candidate_coefficient[0].item()
                if settings.update_mode in {
                    "differentiable_hierarchical_relative",
                    "differentiable_particle_graph_lm",
                }
                else 0.0
            ),
            "local_velocity_damping_frozen": 1,
            "local_update_interval_frames": update_interval,
            "local_parameter_excluded_particles": int(
                self.local_parameter_exclusion_mask.sum().item()
            ),
            "autograd_shape_frozen": int(
                settings.update_mode != "differentiable_particle_graph_lm"
            ),
            "graph_shape_log_coupling": float(
                settings.graph_shape_log_coupling
                if settings.update_mode == "differentiable_particle_graph_lm"
                else 0.0
            ),
            "autograd_history_status": "ready",
            "gradient_consistency_status": "awaiting_warp_finite_difference",
        }
        return (
            candidate_distance.detach(),
            candidate_shape.detach(),
            particle_log_step.detach(),
            particle_signal.detach(),
            candidate_coefficient.detach(),
            first.detach(),
            second.detach(),
            next_step,
            gradient.detach(),
            metrics,
        )

    def _propose_differentiable_global(
        self,
        *,
        physical_prediction: torch.Tensor,
        accepted_residual: torch.Tensor,
        rollout_start_positions: torch.Tensor,
        physical_velocities: torch.Tensor | None,
        frame_index: int | None,
        eligible: torch.Tensor,
        material_active: torch.Tensor,
        control_excluded: torch.Tensor,
        control_frozen: torch.Tensor,
        control_coupling_base_positions: torch.Tensor | None,
        control_coupling_displacements: torch.Tensor | None,
        track_target_positions: torch.Tensor | None,
        track_valid_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        float,
        float,
        torch.Tensor,
        dict[str, float | int | str],
    ]:
        """Identify global paper stiffness, damping and grasp coupling.

        The parameters remain the project's native paper-PBD quantities.  In
        particular, coefficient 0 scales ``paper_distance_stiffness`` and is
        not converted to, or reported as, a Young's modulus.
        """
        settings = self.settings
        start = rollout_start_positions.detach().to(
            device=self.rest_positions.device, dtype=torch.float32
        )
        if start.shape != physical_prediction.shape:
            raise ValueError("Differentiable PBD start positions have wrong shape")
        corrected_positions = physical_prediction + accepted_residual
        self._append_causal_observation(
            frame_index=frame_index,
            rollout_start_positions=start,
            corrected_positions=corrected_positions,
            physical_velocities=physical_velocities,
            eligible=eligible,
            material_active=material_active,
            control_excluded=control_excluded,
            control_frozen=control_frozen,
            control_coupling_base_positions=control_coupling_base_positions,
            control_coupling_displacements=control_coupling_displacements,
            track_target_positions=track_target_positions,
            track_valid_mask=track_valid_mask,
        )
        coefficient = self.global_log_coefficients.detach()
        zero_particle = torch.zeros_like(self.distance_stiffness)
        if settings.update_mode == "differentiable_global_mhe":
            # The authoritative Warp multi-shooting replay builds and gates
            # the actual LM proposal after this observation is paired with its
            # saved rollout state.  Do not spend a second, mismatched Torch
            # H=3 rollout merely to manufacture a provisional Adam step.
            metrics: dict[str, float | int | str] = {
                "update_mode": settings.update_mode,
                "autograd_temporal_steps": len(self.causal_material_history),
                "autograd_history_status": "awaiting_warp_mhe",
                "gradient_consistency_status": "warp_fd_eps_and_2eps_pending",
                "global_distance_scale": float(torch.exp(coefficient[0]).item()),
                "global_velocity_damping_per_second": float(
                    self.global_velocity_damping_per_second
                ),
                "global_coupling_gain": float(self.initial_coupling_gain),
                "global_coupling_fixed": 1,
                "global_rollout_objective": (
                    "causal_multishooting_track_relative+edge_strain_lm"
                ),
                "autograd_shape_frozen": 1,
            }
            return (
                self.distance_stiffness.detach().clone(),
                self.shape_stiffness.detach().clone(),
                zero_particle,
                zero_particle,
                coefficient.clone(),
                self.global_adam_first_moment.clone(),
                self.global_adam_second_moment.clone(),
                self.global_adam_step,
                float(self.global_velocity_damping_per_second),
                float(self.initial_coupling_gain),
                torch.zeros_like(coefficient),
                metrics,
            )
        # Use the shortest permitted causal window for online operation.  A
        # three-transition Warp FD check is substantially cheaper and avoids
        # delaying a continuous surgical stream while still satisfying the
        # requested 3..5-step system-identification window.
        observations = self.causal_material_history[-3:]
        temporal_steps = len(observations)

        def parameter_values(
            value: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            distance = torch.clamp(
                self.initial_distance * torch.exp(value[0]),
                settings.distance_minimum,
                settings.distance_maximum,
            )
            damping = torch.clamp(
                torch.as_tensor(
                    self.initial_velocity_damping_per_second,
                    dtype=value.dtype,
                    device=value.device,
                )
                * torch.exp(value[1]),
                settings.global_damping_minimum_per_second,
                settings.global_damping_maximum_per_second,
            )
            if settings.global_optimize_coupling:
                coupling = torch.clamp(
                    torch.as_tensor(
                        self.initial_coupling_gain,
                        dtype=value.dtype,
                        device=value.device,
                    )
                    * torch.exp(value[2]),
                    settings.global_coupling_minimum,
                    settings.global_coupling_maximum,
                )
            else:
                coupling = value[2] * 0.0 + float(self.initial_coupling_gain)
            return distance, damping, coupling

        if temporal_steps < 3:
            distance, damping, coupling = parameter_values(coefficient)
            metrics: dict[str, float | int | str] = {
                "update_mode": settings.update_mode,
                "autograd_unroll_steps": settings.autograd_unroll_steps,
                "autograd_temporal_steps": temporal_steps,
                "autograd_material_iterations": settings.autograd_material_iterations,
                "autograd_integration_substeps": settings.autograd_integration_substeps,
                "autograd_objective": 0.0,
                "autograd_data_loss": 0.0,
                "autograd_position_loss": 0.0,
                "autograd_edge_strain_loss": 0.0,
                "autograd_edge_strain_weight": float(settings.strain_signal_weight),
                "autograd_prior_loss": 0.0,
                "autograd_gradient_norm": 0.0,
                "autograd_history_status": "warming_up",
                "gradient_consistency_status": "warming_up",
                "global_distance_scale": float(torch.exp(coefficient[0]).item()),
                "global_velocity_damping_per_second": float(damping.item()),
                "global_coupling_gain": float(coupling.item()),
                "global_parameter_prior_weight": float(
                    settings.global_parameter_prior_weight
                ),
                "global_rollout_objective": "203track_mean_h1_h2_h3_weighted_1p5_2_3",
                "autograd_shape_frozen": 1,
            }
            return (
                distance.detach(),
                self.shape_stiffness.detach().clone(),
                zero_particle,
                zero_particle,
                coefficient.clone(),
                self.global_adam_first_moment.clone(),
                self.global_adam_second_moment.clone(),
                self.global_adam_step,
                float(damping.item()),
                float(coupling.item()),
                torch.zeros_like(coefficient),
                metrics,
            )

        variable = coefficient.clone()
        if not settings.global_optimize_coupling:
            variable[2] = 0.0
        variable.requires_grad_(True)
        distance, damping, coupling = parameter_values(variable)
        position_losses: list[tuple[float, torch.Tensor]] = []
        point_mean_losses: list[tuple[float, torch.Tensor]] = []
        median_band_losses: list[tuple[float, torch.Tensor]] = []
        region_mean_losses: list[tuple[float, torch.Tensor]] = []
        tail_region_losses: list[tuple[float, torch.Tensor]] = []
        relative_track_losses: list[tuple[float, torch.Tensor]] = []
        strain_losses: list[tuple[float, torch.Tensor]] = []
        source, target = self.edges[:, 0], self.edges[:, 1]
        active_any = False
        track_loss_count = 0
        relative_track_pair_count = 0
        robust_relative_mode = settings.update_mode in {
            "differentiable_global_relative",
            "differentiable_global_mhe",
        }
        # One causal three-frame open-loop rollout.  The previous version
        # restarted from the visually corrected state at every frame, which
        # optimized one-step assimilation but did not identify parameters that
        # survive the 20% open-loop benchmark.  Later gaps carry larger weight.
        rollout_positions = observations[0].rollout_start_positions
        rollout_velocities = observations[0].physical_velocities
        for horizon_index, observation in enumerate(observations, start=1):
            horizon_weight = settings.global_horizon_weights[horizon_index - 1]
            step_active = (
                observation.eligible_mask & observation.material_active_mask
            )
            controlled_target = (
                observation.control_coupling_base_positions
                + coupling * observation.control_coupling_displacements
            )
            control_velocity = (
                controlled_target - rollout_positions
            ) / settings.autograd_frame_dt
            frozen_positions = observation.corrected_positions
            frozen = self.fixed_mask
            rollout_positions, rollout_velocities = (
                self._differentiable_full_pbd_step(
                    rollout_positions,
                    rollout_velocities,
                    distance,
                    self.initial_shape,
                    frozen,
                    frozen_positions,
                    settings.autograd_frame_dt,
                    velocity_damping_per_second=damping,
                    control_mask=observation.control_frozen_mask,
                    control_positions=controlled_target,
                    control_velocities=control_velocity,
                )
            )
            if (
                observation.track_target_positions is not None
                and observation.track_valid_mask is not None
                and self.track_region_basis is not None
            ):
                predicted_track_positions = self.fixed_track_centers(
                    rollout_positions
                )
                support_coverage = torch.sum(
                    self.track_particle_weights
                    * step_active[self.track_particle_ids].to(torch.float32),
                    dim=1,
                )
                loss_active = (
                    observation.track_valid_mask & (support_coverage >= 0.50)
                )
                if robust_relative_mode:
                    (
                        node_loss,
                        relative_track_loss,
                        _relative_active,
                        active_pair_count,
                    ) = self.robust_relative_track_losses(
                        predicted_track_positions,
                        observation.track_target_positions,
                        loss_active,
                    )
                    relative_track_pair_count += active_pair_count
                else:
                    # Foundation depth legitimately leaves some CoTracker
                    # targets non-finite. Mask before sqrt so an inactive NaN
                    # branch cannot reach SqrtBackward.
                    track_error = torch.where(
                        loss_active[:, None],
                        predicted_track_positions
                        - observation.track_target_positions,
                        torch.zeros_like(predicted_track_positions),
                    )
                    scaled_error = track_error / settings.residual_full_scale_m
                    node_loss = torch.sqrt(
                        torch.sum(scaled_error.square(), dim=1) + 1.0e-4
                    ) - 0.01
                    relative_track_loss = node_loss.sum() * 0.0
                loss_region_basis = self.track_region_basis
            else:
                scaled_error = (
                    rollout_positions - observation.corrected_positions
                ) / settings.residual_full_scale_m
                node_loss = torch.sqrt(
                    torch.sum(scaled_error.square(), dim=1) + 1.0e-4
                ) - 0.01
                loss_active = step_active
                loss_region_basis = None
                relative_track_loss = node_loss.sum() * 0.0
            active_any = active_any or bool(loss_active.any().item())
            if observation.track_target_positions is not None:
                track_loss_count += int(loss_active.sum().item())
            if bool(loss_active.any().item()):
                (
                    distribution_loss,
                    point_mean_loss,
                    median_band_loss,
                    region_mean_loss,
                    tail_region_loss,
                ) = self.global_position_distribution_loss(
                    node_loss, loss_active, loss_region_basis
                )
                if robust_relative_mode:
                    relative_weight = settings.global_relative_track_weight
                    distribution_loss = (
                        (1.0 - relative_weight) * distribution_loss
                        + relative_weight * relative_track_loss
                    )
                    relative_track_losses.append(
                        (horizon_weight, relative_track_loss)
                    )
                position_losses.append((horizon_weight, distribution_loss))
                point_mean_losses.append((horizon_weight, point_mean_loss))
                median_band_losses.append((horizon_weight, median_band_loss))
                region_mean_losses.append((horizon_weight, region_mean_loss))
                tail_region_losses.append((horizon_weight, tail_region_loss))
            if self.edges.numel():
                edge_active = (
                    (step_active[source] | step_active[target])
                    & ~observation.control_exclusion_mask[source]
                    & ~observation.control_exclusion_mask[target]
                )
                rollout_length = self._stable_vector_norm(
                    rollout_positions[target] - rollout_positions[source], dim=1
                )
                target_length = self._stable_vector_norm(
                    observation.corrected_positions[target]
                    - observation.corrected_positions[source],
                    dim=1,
                )
                strain_error = (
                    (rollout_length - target_length) / self.rest_edge_lengths
                ) / settings.edge_strain_full_scale
                robust_strain = torch.sqrt(
                    strain_error.square() + 1.0e-4
                ) - 0.01
                if bool(edge_active.any().item()):
                    strain_losses.append(
                        (horizon_weight, robust_strain[edge_active].mean())
                    )
        position_loss = (
            sum(weight * loss for weight, loss in position_losses)
            / sum(weight for weight, _loss in position_losses)
            if position_losses
            else variable.sum() * 0.0
        )
        def temporal_average(
            values: list[tuple[float, torch.Tensor]],
        ) -> torch.Tensor:
            return (
                sum(weight * loss for weight, loss in values)
                / sum(weight for weight, _loss in values)
                if values
                else variable.sum() * 0.0
            )

        point_mean_loss = temporal_average(point_mean_losses)
        median_band_loss = temporal_average(median_band_losses)
        region_mean_loss = temporal_average(region_mean_losses)
        tail_region_loss = temporal_average(tail_region_losses)
        relative_track_loss = temporal_average(relative_track_losses)
        edge_strain_loss = (
            sum(weight * loss for weight, loss in strain_losses)
            / sum(weight for weight, _loss in strain_losses)
            if strain_losses
            else variable.sum() * 0.0
        )
        blend = settings.strain_signal_weight
        data_loss = (1.0 - blend) * position_loss + blend * edge_strain_loss
        prior_weights = torch.tensor(
            (1.0, 1.0, 0.25 if settings.global_optimize_coupling else 0.0),
            dtype=variable.dtype,
            device=variable.device,
        )
        prior_loss = settings.global_parameter_prior_weight * torch.mean(
            prior_weights * variable.square()
        )
        objective = data_loss + prior_loss
        (raw_gradient,) = torch.autograd.grad(objective, variable)
        finite_gradient = torch.isfinite(raw_gradient)
        gradient = torch.where(
            finite_gradient, raw_gradient, torch.zeros_like(raw_gradient)
        )

        next_step = self.global_adam_step + 1
        beta1 = settings.signal_ema_decay
        beta2 = settings.autograd_adam_beta2
        first = (
            beta1 * self.global_adam_first_moment
            + (1.0 - beta1) * gradient
        )
        second = (
            beta2 * self.global_adam_second_moment
            + (1.0 - beta2) * gradient.square()
        )
        first_hat = first / (1.0 - beta1**next_step)
        second_hat = second / (1.0 - beta2**next_step)
        coefficient_step = -settings.log_learning_rate * first_hat / (
            torch.sqrt(second_hat) + settings.autograd_adam_epsilon
        )
        coefficient_step = torch.clamp(
            coefficient_step,
            min=-settings.maximum_log_step,
            max=settings.maximum_log_step,
        )
        coefficient_step = torch.where(
            finite_gradient
            & torch.full_like(finite_gradient, active_any, dtype=torch.bool),
            coefficient_step,
            torch.zeros_like(coefficient_step),
        )
        candidate_coefficient = torch.clamp(
            coefficient + coefficient_step,
            min=-settings.autograd_maximum_log_offset,
            max=settings.autograd_maximum_log_offset,
        )
        if not settings.global_optimize_coupling:
            candidate_coefficient[2] = 0.0
        effective_step = candidate_coefficient - coefficient
        candidate_distance, candidate_damping, candidate_coupling = (
            parameter_values(candidate_coefficient)
        )
        particle_log_step = torch.ones_like(
            self.distance_stiffness
        ) * effective_step[0]
        particle_signal = torch.ones_like(self.distance_stiffness) * (
            -gradient[0] / gradient.abs().max().clamp_min(1.0e-12)
        )
        metrics = {
            "update_mode": settings.update_mode,
            "autograd_unroll_steps": settings.autograd_unroll_steps,
            "autograd_temporal_steps": temporal_steps,
            "autograd_material_iterations": settings.autograd_material_iterations,
            "autograd_integration_substeps": settings.autograd_integration_substeps,
            "autograd_objective": float(objective.detach().item()),
            "autograd_data_loss": float(data_loss.detach().item()),
            "autograd_position_loss": float(position_loss.detach().item()),
            "global_position_point_mean_loss": float(
                point_mean_loss.detach().item()
            ),
            "global_position_median_band_loss": float(
                median_band_loss.detach().item()
            ),
            "global_position_region_mean_loss": float(
                region_mean_loss.detach().item()
            ),
            "global_position_tail_region_loss": float(
                tail_region_loss.detach().item()
            ),
            "global_relative_track_loss": float(
                relative_track_loss.detach().item()
            ),
            "global_relative_track_weight": float(
                settings.global_relative_track_weight
                if robust_relative_mode
                else 0.0
            ),
            "global_relative_track_pair_count": int(
                relative_track_pair_count
            ),
            "global_cauchy_scale": float(settings.global_cauchy_scale),
            "global_position_point_mean_weight": float(
                1.0
                - settings.global_median_band_weight
                - settings.global_region_balance_weight
                - settings.global_tail_region_weight
            ),
            "global_position_median_band_weight": float(
                settings.global_median_band_weight
            ),
            "global_position_region_balance_weight": float(
                settings.global_region_balance_weight
            ),
            "global_position_tail_region_weight": float(
                settings.global_tail_region_weight
            ),
            "global_tail_region_fraction": float(
                settings.global_tail_region_fraction
            ),
            "global_horizon_weight_h1": float(settings.global_horizon_weights[0]),
            "global_horizon_weight_h2": float(settings.global_horizon_weights[1]),
            "global_horizon_weight_h3": float(settings.global_horizon_weights[2]),
            "autograd_edge_strain_loss": float(edge_strain_loss.detach().item()),
            "autograd_edge_strain_weight": float(blend),
            "autograd_prior_loss": float(prior_loss.detach().item()),
            "autograd_gradient_norm": float(torch.linalg.vector_norm(gradient).item()),
            "autograd_raw_gradient_log_distance": float(raw_gradient[0].item()),
            "autograd_raw_gradient_log_damping": float(raw_gradient[1].item()),
            "autograd_raw_gradient_log_coupling": float(raw_gradient[2].item()),
            "autograd_gradient_finite_count": int(finite_gradient.sum().item()),
            "autograd_gradient_nonfinite_count": int(
                (~finite_gradient).sum().item()
            ),
            "autograd_gradient_log_distance": float(gradient[0].item()),
            "autograd_gradient_log_damping": float(gradient[1].item()),
            "autograd_gradient_log_coupling": float(gradient[2].item()),
            "autograd_coefficient_step_maximum": float(effective_step.abs().max().item()),
            "global_log_distance": float(candidate_coefficient[0].item()),
            "global_log_damping": float(candidate_coefficient[1].item()),
            "global_log_coupling": float(candidate_coefficient[2].item()),
            "global_distance_scale": float(torch.exp(candidate_coefficient[0]).item()),
            "global_velocity_damping_per_second": float(candidate_damping.item()),
            "global_coupling_gain": float(candidate_coupling.item()),
            "global_parameter_prior_weight": float(
                settings.global_parameter_prior_weight
            ),
            "global_rollout_objective": (
                "203track_cauchy_relative_h1_h2_h3_cumulative_warp"
                if robust_relative_mode
                else "203track_mean_h1_h2_h3_weighted_1p5_2_3"
            ),
            "global_track_loss_count": int(track_loss_count),
            "global_coupling_fixed": int(not settings.global_optimize_coupling),
            "autograd_shape_frozen": 1,
            "autograd_history_status": "ready",
            "gradient_consistency_status": "awaiting_warp_finite_difference",
        }
        return (
            candidate_distance.detach(),
            self.shape_stiffness.detach().clone(),
            particle_log_step.detach(),
            particle_signal.detach(),
            candidate_coefficient.detach(),
            first.detach(),
            second.detach(),
            next_step,
            float(candidate_damping.detach().item()),
            float(candidate_coupling.detach().item()),
            gradient.detach(),
            metrics,
        )

    def robust_relative_track_losses(
        self,
        predicted_positions: torch.Tensor,
        target_positions: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Return Cauchy absolute and neighbour-relative RGB track losses.

        Invalid FoundationStereo targets are removed before any subtraction,
        so the inactive branch cannot inject NaNs into autograd.  Relative
        edges use the fixed canonical track graph and hence suppress common
        camera/depth translation without accessing privileged simulation GT.
        """
        if predicted_positions.shape != target_positions.shape:
            raise ValueError("Predicted and target track positions disagree")
        if (
            predicted_positions.ndim != 2
            or predicted_positions.shape[1] != 3
            or active_mask.shape != predicted_positions.shape[:1]
        ):
            raise ValueError("Robust track loss expects [N,3] positions")
        settings = self.settings
        safe_error = torch.where(
            active_mask[:, None],
            predicted_positions - target_positions,
            torch.zeros_like(predicted_positions),
        )
        scaled = safe_error / settings.residual_full_scale_m
        cauchy_squared = settings.global_cauchy_scale**2
        absolute_node_loss = torch.log1p(
            torch.sum(scaled.square(), dim=1) / cauchy_squared
        )
        edges = self.track_neighbor_edges
        if not edges.numel():
            zero = absolute_node_loss.sum() * 0.0
            return absolute_node_loss, zero, torch.empty(
                0, dtype=torch.bool, device=active_mask.device
            ), 0
        source, target = edges[:, 0], edges[:, 1]
        pair_active = active_mask[source] & active_mask[target]
        relative_error = torch.where(
            pair_active[:, None],
            safe_error[source] - safe_error[target],
            torch.zeros_like(safe_error[source]),
        )
        # Independent endpoint noise increases pair variance by two.
        relative_scaled = relative_error / (
            settings.residual_full_scale_m * math.sqrt(2.0)
        )
        relative_edge_loss = torch.log1p(
            torch.sum(relative_scaled.square(), dim=1) / cauchy_squared
        )
        relative_loss = (
            relative_edge_loss[pair_active].mean()
            if bool(pair_active.any().item())
            else absolute_node_loss.sum() * 0.0
        )
        return (
            absolute_node_loss,
            relative_loss,
            pair_active,
            int(pair_active.sum().item()),
        )

    def global_position_distribution_loss(
        self,
        node_loss: torch.Tensor,
        active_mask: torch.Tensor,
        region_basis: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return a causal point/median/region/tail position objective.

        The median proxy averages the central 30 percent after sorting rather
        than differentiating through a single order statistic.  Region terms
        use the deterministic farthest-point partition already built for the
        PBD material field, so sparse tissue edges receive the same influence
        as densely sampled interior regions.
        """
        if node_loss.ndim != 1 or active_mask.shape != node_loss.shape:
            raise ValueError("Global position loss expects matching 1-D inputs")
        active_loss = node_loss[active_mask]
        if not active_loss.numel():
            zero = node_loss.sum() * 0.0
            return zero, zero, zero, zero, zero

        settings = self.settings
        point_mean = active_loss.mean()
        sorted_loss = torch.sort(active_loss).values
        count = int(sorted_loss.numel())
        median_start = min(count - 1, int(math.floor(0.35 * count)))
        median_end = max(
            median_start + 1,
            min(count, int(math.ceil(0.65 * count))),
        )
        median_band = sorted_loss[median_start:median_end].mean()
        selected_basis = (
            self.low_dimensional_basis if region_basis is None else region_basis
        )
        if selected_basis.ndim != 2 or selected_basis.shape[0] != len(node_loss):
            raise ValueError("Position-loss region basis has the wrong shape")
        active_basis = selected_basis[active_mask]
        region_denominator = active_basis.sum(dim=0)
        valid_region = region_denominator > 1.0e-6
        region_values = (
            active_basis.T @ active_loss
        ) / region_denominator.clamp_min(1.0e-6)
        region_values = region_values[valid_region]
        if region_values.numel():
            region_mean = region_values.mean()
            tail_count = max(
                1,
                int(
                    math.ceil(
                        settings.global_tail_region_fraction
                        * int(region_values.numel())
                    )
                ),
            )
            tail_region = torch.topk(
                region_values, k=min(tail_count, int(region_values.numel()))
            ).values.mean()
        else:
            region_mean = point_mean
            tail_region = point_mean

        point_weight = (
            1.0
            - settings.global_median_band_weight
            - settings.global_region_balance_weight
            - settings.global_tail_region_weight
        )
        total = (
            point_weight * point_mean
            + settings.global_median_band_weight * median_band
            + settings.global_region_balance_weight * region_mean
            + settings.global_tail_region_weight * tail_region
        )
        return total, point_mean, median_band, region_mean, tail_region

    def candidate_parameter_step_maximum(
        self,
        candidate: PaperStiffnessCandidate,
    ) -> float:
        """Return the complete active-parameter step used for validity.

        ``candidate.log_step`` is a particle-space export of only the distance
        coefficient in global mode.  It therefore cannot decide whether a
        distance+damping proposal is empty: a damping-only Adam proposal is a
        valid candidate.  Coupling is included only when it is actually being
        optimized; the current benchmark fixes it to one.
        """
        if candidate is not self.pending_candidate:
            raise RuntimeError("Cannot validate an unknown stiffness candidate")
        if self.settings.update_mode in {
            "differentiable_global",
            "differentiable_global_relative",
            "differentiable_global_mhe",
        }:
            coefficients = candidate.global_log_coefficients
            if coefficients is None:
                raise RuntimeError("Global candidate has no parameter coefficients")
            active_parameter_count = (
                3 if self.settings.global_optimize_coupling else 2
            )
            step = (
                coefficients[:active_parameter_count]
                - self.global_log_coefficients[:active_parameter_count]
            )
            maximum = float(step.abs().max().item())
            candidate.metrics.update(
                candidate_validity_step_source=(
                    "global_distance+damping+coupling"
                    if self.settings.global_optimize_coupling
                    else "global_distance+damping"
                ),
                candidate_parameter_step_maximum=maximum,
                candidate_distance_log_step=float(step[0].item()),
                candidate_damping_log_step=float(step[1].item()),
            )
            if self.settings.global_optimize_coupling:
                candidate.metrics["candidate_coupling_log_step"] = float(
                    step[2].item()
                )
            return maximum
        if self.settings.update_mode in {
            "differentiable_local_relative",
            "differentiable_hierarchical_relative",
            "differentiable_particle_graph_lm",
        }:
            coefficients = candidate.low_dimensional_log_coefficients
            if coefficients is None:
                raise RuntimeError("Local candidate has no regional coefficients")
            step = coefficients - self.low_dimensional_log_coefficients
            maximum = float(step.abs().max().item())
            candidate.metrics.update(
                candidate_validity_step_source="local_zero_mean_coefficients",
                candidate_parameter_step_maximum=maximum,
                candidate_local_coefficient_step_norm=float(
                    torch.linalg.vector_norm(step).item()
                ),
            )
            return maximum
        maximum = float(candidate.log_step.abs().max().item())
        candidate.metrics.update(
            candidate_validity_step_source="particle_distance_log_step",
            candidate_parameter_step_maximum=maximum,
        )
        return maximum

    def apply_gradient_consistency_gate(
        self,
        candidate: PaperStiffnessCandidate,
        warp_finite_difference_gradient: torch.Tensor,
    ) -> tuple[bool, float]:
        """Compare Torch autograd and authoritative Warp FD directions."""
        if candidate is not self.pending_candidate:
            raise RuntimeError("Cannot gate an unknown stiffness candidate")
        autograd_gradient = candidate.autograd_parameter_gradient
        if autograd_gradient is None:
            raise RuntimeError("Candidate has no global autograd gradient")
        warp_gradient = warp_finite_difference_gradient.detach().to(
            device=autograd_gradient.device, dtype=autograd_gradient.dtype
        )
        if warp_gradient.shape != autograd_gradient.shape:
            raise ValueError("Warp finite-difference gradient has wrong shape")
        finite = bool(
            torch.isfinite(autograd_gradient).all().item()
            and torch.isfinite(warp_gradient).all().item()
        )
        denominator = (
            torch.linalg.vector_norm(autograd_gradient)
            * torch.linalg.vector_norm(warp_gradient)
        )
        cosine = (
            float(torch.dot(autograd_gradient, warp_gradient).item())
            / float(denominator.item())
            if finite and float(denominator.item()) > 1.0e-12
            else -1.0
        )
        allowed = finite and cosine >= self.settings.gradient_cosine_minimum
        common_metrics: dict[str, float | int | str] = {
            "gradient_direction_cosine": float(cosine),
            "gradient_cosine_minimum": float(
                self.settings.gradient_cosine_minimum
            ),
            "gradient_consistency_status": "passed" if allowed else "failed",
            "warp_fd_gradient_norm": float(
                torch.linalg.vector_norm(warp_gradient).item()
            ),
        }
        if self.settings.update_mode in {
            "differentiable_local_relative",
            "differentiable_hierarchical_relative",
            "differentiable_particle_graph_lm",
        }:
            common_metrics.update(
                warp_fd_local_gradient_minimum=float(warp_gradient.min().item()),
                warp_fd_local_gradient_median=float(warp_gradient.median().item()),
                warp_fd_local_gradient_maximum=float(warp_gradient.max().item()),
                warp_fd_local_gradient_sum=float(warp_gradient.sum().item()),
            )
        else:
            common_metrics.update(
                warp_fd_gradient_log_distance=float(warp_gradient[0].item()),
                warp_fd_gradient_log_damping=float(warp_gradient[1].item()),
                warp_fd_gradient_log_coupling=float(warp_gradient[2].item()),
            )
        candidate.metrics.update(common_metrics)
        return allowed, cosine

    def apply_particle_graph_directional_gate(
        self,
        candidate: PaperStiffnessCandidate,
        autograd_directional_derivatives: torch.Tensor,
        warp_directional_derivatives: torch.Tensor,
    ) -> tuple[bool, float]:
        """Gate a large particle field with a few deterministic Warp probes.

        Computing a full Warp finite-difference gradient for more than two
        thousand material variables would require two rollouts per particle.
        Instead, Warp checks several deterministic directions, including the
        proposed autograd direction. This verifies the local Jacobian without
        using a parameter branch, future frame, or evaluation ground truth.
        """
        if candidate is not self.pending_candidate:
            raise RuntimeError("Cannot gate an unknown stiffness candidate")
        if self.settings.update_mode != "differentiable_particle_graph_lm":
            raise RuntimeError("Directional gate requires particle-graph mode")
        predicted = autograd_directional_derivatives.detach().to(
            device=self.rest_positions.device, dtype=torch.float32
        ).reshape(-1)
        measured = warp_directional_derivatives.detach().to(
            device=predicted.device, dtype=predicted.dtype
        ).reshape(-1)
        finite = bool(
            predicted.shape == measured.shape
            and predicted.numel() >= 2
            and torch.isfinite(predicted).all().item()
            and torch.isfinite(measured).all().item()
        )
        denominator = (
            torch.linalg.vector_norm(predicted)
            * torch.linalg.vector_norm(measured)
        )
        cosine = (
            float(torch.dot(predicted, measured).item())
            / float(denominator.item())
            if finite and float(denominator.item()) > 1.0e-12
            else -1.0
        )
        sign_agreement = (
            float(
                torch.mean(
                    (torch.sign(predicted) == torch.sign(measured)).to(torch.float32)
                ).item()
            )
            if finite
            else 0.0
        )
        threshold = self.settings.graph_directional_cosine_minimum
        allowed = finite and cosine >= threshold
        candidate.metrics.update(
            gradient_direction_cosine=float(cosine),
            gradient_cosine_minimum=float(threshold),
            gradient_consistency_status="passed" if allowed else "failed",
            gradient_check_kind="deterministic_warp_directional_fd",
            graph_directional_probe_count=int(predicted.numel()),
            graph_directional_sign_agreement=sign_agreement,
            graph_autograd_directional_norm=float(
                torch.linalg.vector_norm(predicted).item()
            ),
            graph_warp_directional_norm=float(
                torch.linalg.vector_norm(measured).item()
            ),
        )
        return allowed, cosine

    def _particle_graph_lm_direction(
        self,
        gradient: torch.Tensor,
        second_moment_hat: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Solve a damped graph-preconditioned diagonal GN system by CG."""
        settings = self.settings
        gradient = self.project_local_gradient(gradient)
        active = ~self.local_parameter_exclusion_mask
        step = torch.zeros_like(gradient)

        global_scale = gradient[0].abs().clamp_min(1.0e-8)
        global_curvature = (
            torch.sqrt(second_moment_hat[0]).abs() / global_scale
            + settings.graph_lm_damping
        )
        step[0] = -(gradient[0] / global_scale) / global_curvature

        local_gradient = gradient[1:]
        active_values = local_gradient[active]
        local_scale = (
            torch.sqrt(torch.mean(active_values.square())).clamp_min(1.0e-8)
            if active_values.numel()
            else local_gradient.new_tensor(1.0)
        )
        rhs = -(local_gradient / local_scale)
        rhs = torch.where(active, rhs, torch.zeros_like(rhs))
        curvature = torch.sqrt(second_moment_hat[1:]).abs() / local_scale
        observability = self.particle_observability.clamp_min(
            settings.graph_observability_floor
        )
        diagonal = curvature + settings.graph_lm_damping / observability
        diagonal = torch.where(active, diagonal, torch.ones_like(diagonal))

        source, target = self.edges[:, 0], self.edges[:, 1]
        if self.edges.numel():
            edge_active = active[source] & active[target]
            graph_source = source[edge_active]
            graph_target = target[edge_active]
            degree = torch.zeros_like(local_gradient)
            edge_ones = torch.ones_like(graph_source, dtype=local_gradient.dtype)
            degree.index_add_(0, graph_source, edge_ones)
            degree.index_add_(0, graph_target, edge_ones)
            inverse_sqrt_degree = torch.rsqrt(degree.clamp_min(1.0))
        else:
            graph_source = source
            graph_target = target
            degree = torch.zeros_like(local_gradient)
            inverse_sqrt_degree = torch.ones_like(local_gradient)

        def matrix_vector(vector: torch.Tensor) -> torch.Tensor:
            result = diagonal * vector
            if graph_source.numel():
                normalized = vector * inverse_sqrt_degree
                neighbor_sum = torch.zeros_like(vector)
                neighbor_sum.index_add_(
                    0, graph_source, normalized[graph_target]
                )
                neighbor_sum.index_add_(
                    0, graph_target, normalized[graph_source]
                )
                laplacian = vector - inverse_sqrt_degree * neighbor_sum
                laplacian = torch.where(
                    degree > 0.0, laplacian, torch.zeros_like(laplacian)
                )
                result = result + (
                    settings.graph_lm_spatial_hessian_weight * laplacian
                )
            return torch.where(active, result, vector)

        solution = torch.zeros_like(rhs)
        residual = rhs.clone()
        direction = residual.clone()
        squared_residual = torch.dot(residual, residual)
        for _ in range(settings.graph_lm_cg_iterations):
            product = matrix_vector(direction)
            denominator = torch.dot(direction, product).clamp_min(1.0e-12)
            alpha = squared_residual / denominator
            solution = solution + alpha * direction
            next_residual = residual - alpha * product
            next_squared = torch.dot(next_residual, next_residual)
            if float(next_squared.detach().item()) <= 1.0e-12:
                residual = next_residual
                squared_residual = next_squared
                break
            beta = next_squared / squared_residual.clamp_min(1.0e-12)
            direction = next_residual + beta * direction
            residual = next_residual
            squared_residual = next_squared
        step[1:] = torch.where(active, solution, torch.zeros_like(solution))
        step = settings.log_learning_rate * self.project_local_gradient(step)
        maximum = step.abs().max()
        if float(maximum.item()) > settings.maximum_log_step:
            step = step * (settings.maximum_log_step / maximum)
        return step, float(torch.sqrt(squared_residual).item())

    def replace_candidate_step_with_particle_graph_lm(
        self,
        candidate: PaperStiffnessCandidate,
    ) -> None:
        """Install a graph-regularized per-particle step after Warp validation."""
        if candidate is not self.pending_candidate:
            raise RuntimeError("Cannot rewrite an unknown stiffness candidate")
        if self.settings.update_mode != "differentiable_particle_graph_lm":
            raise RuntimeError("Particle-graph LM replacement has wrong mode")
        if candidate.metrics.get("gradient_consistency_status") != "passed":
            raise RuntimeError("Particle-graph step requires a passed Warp check")
        gradient = candidate.autograd_parameter_gradient
        if gradient is None or not bool(torch.isfinite(gradient).all().item()):
            raise ValueError("Particle-graph autograd gradient is invalid")
        gradient = self.project_local_gradient(gradient)
        settings = self.settings
        next_step = self.adam_step + 1
        beta1 = settings.signal_ema_decay
        beta2 = settings.autograd_adam_beta2
        first = beta1 * self.adam_first_moment + (1.0 - beta1) * gradient
        second = beta2 * self.adam_second_moment + (1.0 - beta2) * gradient.square()
        first_hat = first / (1.0 - beta1**next_step)
        second_hat = second / (1.0 - beta2**next_step)
        coefficient_step, cg_residual = self._particle_graph_lm_direction(
            first_hat, second_hat
        )
        coefficient = self.low_dimensional_log_coefficients.detach().clone()
        candidate_coefficient = self.normalize_local_coefficients(
            coefficient + coefficient_step
        )
        candidate_coefficient = torch.clamp(
            candidate_coefficient,
            min=-settings.autograd_maximum_log_offset,
            max=settings.autograd_maximum_log_offset,
        )
        candidate_coefficient = self.normalize_local_coefficients(
            candidate_coefficient
        )
        effective_step = candidate_coefficient - coefficient
        distance = self.local_distance_from_coefficients(candidate_coefficient)
        shape = self.local_shape_from_coefficients(candidate_coefficient)
        particle_step = torch.log(
            distance / self.distance_stiffness.detach().clamp_min(1.0e-12)
        )
        signal = self.local_zero_mean_log_field(
            -gradient / gradient.abs().max().clamp_min(1.0e-12)
        )
        field = self.local_zero_mean_log_field(candidate_coefficient)
        with torch.no_grad():
            candidate.distance_stiffness.copy_(distance)
            candidate.shape_stiffness.copy_(shape)
            candidate.log_step.copy_(particle_step)
            candidate.signal_ema.copy_(signal)
            candidate.signal_ema[self.local_parameter_exclusion_mask] = 0.0
        candidate.low_dimensional_log_coefficients = candidate_coefficient.clone()
        candidate.adam_first_moment = first.clone()
        candidate.adam_second_moment = second.clone()
        candidate.adam_step = next_step
        candidate.metrics.update(
            optimizer_gradient_source=(
                "torch_autograd+warp_directional_check+diagonal_graph_lm"
            ),
            graph_lm_cg_residual=cg_residual,
            graph_lm_damping=float(settings.graph_lm_damping),
            graph_lm_spatial_hessian_weight=float(
                settings.graph_lm_spatial_hessian_weight
            ),
            graph_lm_step_maximum=float(effective_step.abs().max().item()),
            graph_global_log_distance=float(candidate_coefficient[0].item()),
            graph_particle_log_minimum=float(field.min().item()),
            graph_particle_log_median=float(field.median().item()),
            graph_particle_log_maximum=float(field.max().item()),
            graph_distance_minimum=float(distance.min().item()),
            graph_distance_median=float(distance.median().item()),
            graph_distance_maximum=float(distance.max().item()),
            graph_shape_minimum=float(shape.min().item()),
            graph_shape_median=float(shape.median().item()),
            graph_shape_maximum=float(shape.max().item()),
            graph_observability_minimum=float(
                self.particle_observability.min().item()
            ),
            graph_observability_median=float(
                self.particle_observability.median().item()
            ),
            graph_observability_maximum=float(
                self.particle_observability.max().item()
            ),
        )

    def observable_lm_step(
        self,
        *,
        center_coefficients: torch.Tensor,
        residual: torch.Tensor,
        jacobian: torch.Tensor,
        wide_jacobian: torch.Tensor,
        short_residual_count: int | None = None,
    ) -> tuple[
        torch.Tensor,
        bool,
        str,
        dict[str, float | int | str],
    ]:
        """Solve one observable, robust moving-horizon LM material step.

        ``jacobian`` and ``wide_jacobian`` are central finite differences of
        exactly the same causal Warp residual vector at epsilon and 2*epsilon.
        The second scale is a numerical derivative check, not a parameter
        candidate or an evaluation branch.  Only distance stiffness and
        velocity damping are active; prescribed grasp coupling remains one.
        """
        settings = self.settings
        center = center_coefficients.detach().to(
            device=self.global_log_coefficients.device, dtype=torch.float32
        )
        data_residual = residual.detach().to(
            device=center.device, dtype=center.dtype
        ).reshape(-1)
        data_jacobian = jacobian.detach().to(
            device=center.device, dtype=center.dtype
        )
        data_wide_jacobian = wide_jacobian.detach().to(
            device=center.device, dtype=center.dtype
        )
        if center.shape != (3,):
            raise ValueError("Observable MHE center must contain 3 log parameters")
        expected_shape = (len(data_residual), 2)
        if data_jacobian.shape != expected_shape:
            raise ValueError("Observable MHE Jacobian has the wrong shape")
        if data_wide_jacobian.shape != expected_shape:
            raise ValueError("Observable wide Jacobian has the wrong shape")
        short_count = (
            len(data_residual)
            if short_residual_count is None
            else int(short_residual_count)
        )
        if not 0 < short_count < len(data_residual):
            raise ValueError("Observable H1 residual slice is invalid")

        zero_step = torch.zeros_like(center)
        finite = bool(
            torch.isfinite(data_residual).all().item()
            and torch.isfinite(data_jacobian).all().item()
            and torch.isfinite(data_wide_jacobian).all().item()
        )
        if not finite or data_residual.numel() < 2:
            metrics: dict[str, float | int | str] = {
                "observable_status": "nonfinite_or_empty",
                "observable_residual_count": int(data_residual.numel()),
                "observable_allowed": 0,
            }
            return zero_step, False, "observable_nonfinite_or_empty", metrics

        # The residual builder assigns equal total budget to H1 and long
        # rollout.  Undo those outer weights here: v4 uses the long objective
        # at its full v2 scale for parameter estimation, while H1 supplies an
        # uncertainty-aware causal trust constraint rather than another tuned
        # scalar loss weight.
        objective_weight_sum = (
            settings.observable_short_horizon_weight
            + settings.observable_long_horizon_weight
        )
        short_outer_weight = (
            settings.observable_short_horizon_weight / objective_weight_sum
        )
        long_outer_weight = (
            settings.observable_long_horizon_weight / objective_weight_sum
        )
        short_scale = math.sqrt(short_outer_weight)
        long_scale = math.sqrt(long_outer_weight)
        short_residual = data_residual[:short_count] / short_scale
        short_jacobian = data_jacobian[:short_count] / short_scale
        short_wide_jacobian = data_wide_jacobian[:short_count] / short_scale
        long_residual = data_residual[short_count:] / long_scale
        long_jacobian = data_jacobian[short_count:] / long_scale
        long_wide_jacobian = data_wide_jacobian[short_count:] / long_scale

        # Cauchy IRLS weights are frozen at the verified center.  H1 and long
        # residuals are weighted independently after undoing the outer budget.
        cauchy = float(settings.global_cauchy_scale)
        short_robust_weight = torch.rsqrt(
            1.0 + (short_residual / cauchy).square()
        )
        long_robust_weight = torch.rsqrt(
            1.0 + (long_residual / cauchy).square()
        )
        weighted_short_residual = short_robust_weight * short_residual
        weighted_short_jacobian = (
            short_robust_weight[:, None] * short_jacobian
        )
        weighted_short_wide_jacobian = (
            short_robust_weight[:, None] * short_wide_jacobian
        )
        weighted_long_residual = long_robust_weight * long_residual
        weighted_long_jacobian = long_robust_weight[:, None] * long_jacobian
        weighted_long_wide_jacobian = (
            long_robust_weight[:, None] * long_wide_jacobian
        )

        singular_values = torch.linalg.svdvals(weighted_long_jacobian)
        largest = singular_values[0]
        smallest = singular_values[-1]
        singular_ratio = float(
            (smallest / largest.clamp_min(1.0e-12)).item()
        )
        jacobian_norm = float(
            torch.linalg.vector_norm(weighted_long_jacobian).item()
        )
        long_gradient = (
            weighted_long_jacobian.T @ weighted_long_residual
        )
        long_wide_gradient = (
            weighted_long_wide_jacobian.T @ weighted_long_residual
        )
        short_gradient = (
            weighted_short_jacobian.T @ weighted_short_residual
        )
        short_wide_gradient = (
            weighted_short_wide_jacobian.T @ weighted_short_residual
        )
        gradient_denominator = (
            torch.linalg.vector_norm(long_gradient)
            * torch.linalg.vector_norm(long_wide_gradient)
        )
        fd_scale_cosine = (
            float(torch.dot(long_gradient, long_wide_gradient).item())
            / float(gradient_denominator.item())
            if float(gradient_denominator.item()) > 1.0e-12
            else -1.0
        )
        observable = (
            jacobian_norm >= settings.observable_minimum_jacobian_norm
            and singular_ratio >= settings.observable_minimum_singular_ratio
        )
        derivative_consistent = (
            fd_scale_cosine >= settings.observable_fd_scale_cosine_minimum
        )
        allowed = observable and derivative_consistent

        hessian = weighted_long_jacobian.T @ weighted_long_jacobian
        regularized_gradient = long_gradient.clone()
        active_center = center[:2]
        prior_weight = float(settings.observable_parameter_prior_weight)
        if prior_weight > 0.0:
            hessian = hessian + prior_weight * torch.eye(
                2, dtype=center.dtype, device=center.device
            )
            regularized_gradient = (
                regularized_gradient + prior_weight * active_center
            )
        diagonal = torch.diagonal(hessian).clamp_min(1.0e-8)
        damped_hessian = hessian + settings.observable_lm_damping * torch.diag(
            diagonal
        )
        active_step = (
            torch.linalg.solve(damped_hessian, -regularized_gradient)
            if allowed
            else torch.zeros(2, dtype=center.dtype, device=center.device)
        )
        short_directional_before = float(
            torch.dot(short_gradient, active_step).item()
        )
        # Epsilon-vs-2epsilon disagreement is the causal numerical uncertainty
        # of the H1 directional derivative.  Only a predicted H1 increase that
        # exceeds this data-derived tolerance constrains the long-horizon step.
        short_directional_uncertainty = float(
            (
                torch.linalg.vector_norm(
                    short_gradient - short_wide_gradient
                )
                * torch.linalg.vector_norm(active_step)
            ).item()
        )
        short_descent_projected = False
        if (
            allowed
            and short_directional_before > short_directional_uncertainty
        ):
            # Hessian-metric projection is the minimum change to the primary
            # long-rollout LM solution under g_H1^T step <= uncertainty.
            inverse_hessian_short_gradient = torch.linalg.solve(
                damped_hessian, short_gradient
            )
            constraint_denominator = torch.dot(
                short_gradient, inverse_hessian_short_gradient
            )
            if float(constraint_denominator.item()) > 1.0e-12:
                active_step = active_step - (
                    (short_directional_before - short_directional_uncertainty)
                    / constraint_denominator
                ) * inverse_hessian_short_gradient
                short_descent_projected = True
        if allowed:
            maximum = active_step.abs().max().clamp_min(1.0e-12)
            scale = torch.clamp(
                torch.as_tensor(
                    settings.maximum_log_step,
                    dtype=active_step.dtype,
                    device=active_step.device,
                )
                / maximum,
                max=1.0,
            )
            active_step = active_step * scale
        short_directional_after = float(
            torch.dot(short_gradient, active_step).item()
        )
        long_directional_after = float(
            torch.dot(long_gradient, active_step).item()
        )
        step = zero_step.clone()
        if allowed:
            step[:2] = active_step
        predicted_reduction = float(
            (
                -torch.dot(regularized_gradient, active_step)
                - 0.5 * torch.dot(active_step, hessian @ active_step)
            ).item()
        )
        if not observable:
            reason = "observable_rank_or_sensitivity_below_gate"
            status = "unobservable"
        elif not derivative_consistent:
            reason = "observable_fd_scale_inconsistent"
            status = "fd_scale_inconsistent"
        else:
            reason = "observable_lm_ready"
            status = "ready"
        metrics = {
            "observable_status": status,
            "observable_allowed": int(allowed),
            "observable_residual_count": int(data_residual.numel()),
            "observable_jacobian_norm": jacobian_norm,
            "observable_singular_value_maximum": float(largest.item()),
            "observable_singular_value_minimum": float(smallest.item()),
            "observable_singular_value_ratio": singular_ratio,
            "observable_singular_ratio_minimum": float(
                settings.observable_minimum_singular_ratio
            ),
            "observable_fd_scale_cosine": fd_scale_cosine,
            "observable_fd_scale_cosine_minimum": float(
                settings.observable_fd_scale_cosine_minimum
            ),
            "observable_gradient_log_distance": float(
                regularized_gradient[0].item()
            ),
            "observable_gradient_log_damping": float(
                regularized_gradient[1].item()
            ),
            "observable_short_residual_count": int(short_count),
            "observable_short_gradient_log_distance": float(
                short_gradient[0].item()
            ),
            "observable_short_gradient_log_damping": float(
                short_gradient[1].item()
            ),
            "observable_short_directional_before": short_directional_before,
            "observable_short_directional_after": short_directional_after,
            "observable_short_directional_uncertainty": (
                short_directional_uncertainty
            ),
            "observable_long_directional_after": long_directional_after,
            "observable_short_descent_projected": int(short_descent_projected),
            "observable_lm_damping": float(settings.observable_lm_damping),
            "observable_parameter_prior_weight": prior_weight,
            "observable_predicted_reduction": predicted_reduction,
            "observable_log_distance_step": float(step[0].item()),
            "observable_log_damping_step": float(step[1].item()),
        }
        return step, allowed, reason, metrics

    def replace_candidate_step_with_observable_lm(
        self,
        candidate: PaperStiffnessCandidate,
        parameter_step: torch.Tensor,
        diagnostics: dict[str, float | int | str],
    ) -> None:
        """Install an already gated Warp moving-horizon LM proposal."""
        if candidate is not self.pending_candidate:
            raise RuntimeError("Cannot rewrite an unknown stiffness candidate")
        if self.settings.update_mode != "differentiable_global_mhe":
            raise RuntimeError("Observable LM step requires global MHE mode")
        if int(diagnostics.get("observable_allowed", 0)) != 1:
            raise RuntimeError("Cannot install a rejected observable LM step")
        step = parameter_step.detach().to(
            device=self.global_log_coefficients.device,
            dtype=self.global_log_coefficients.dtype,
        ).clone()
        if step.shape != (3,) or not bool(torch.isfinite(step).all().item()):
            raise ValueError("Observable LM parameter step is invalid")
        step[2] = 0.0
        settings = self.settings
        coefficient = self.global_log_coefficients.detach().clone()
        candidate_coefficient = torch.clamp(
            coefficient + step,
            min=-settings.autograd_maximum_log_offset,
            max=settings.autograd_maximum_log_offset,
        )
        candidate_coefficient[2] = 0.0
        effective_step = candidate_coefficient - coefficient
        distance = torch.clamp(
            self.initial_distance * torch.exp(candidate_coefficient[0]),
            settings.distance_minimum,
            settings.distance_maximum,
        )
        damping = torch.clamp(
            torch.as_tensor(
                self.initial_velocity_damping_per_second,
                device=candidate_coefficient.device,
                dtype=candidate_coefficient.dtype,
            )
            * torch.exp(candidate_coefficient[1]),
            settings.global_damping_minimum_per_second,
            settings.global_damping_maximum_per_second,
        )
        with torch.no_grad():
            candidate.distance_stiffness.copy_(distance)
            candidate.shape_stiffness.copy_(self.initial_shape)
            candidate.log_step.fill_(float(effective_step[0].item()))
            candidate.log_step[~candidate.eligible_mask] = 0.0
            signal = -effective_step[0] / effective_step.abs().max().clamp_min(
                1.0e-12
            )
            candidate.signal_ema.fill_(float(signal.item()))
            candidate.signal_ema[~candidate.eligible_mask] = 0.0
        candidate.global_log_coefficients = candidate_coefficient.clone()
        candidate.global_adam_first_moment = (
            self.global_adam_first_moment.detach().clone()
        )
        candidate.global_adam_second_moment = (
            self.global_adam_second_moment.detach().clone()
        )
        candidate.global_adam_step = self.global_adam_step + 1
        candidate.global_velocity_damping_per_second = float(damping.item())
        candidate.global_coupling_gain = float(self.initial_coupling_gain)
        candidate.metrics.update(
            diagnostics,
            optimizer_gradient_source="warp_multishooting_observable_lm",
            global_log_distance=float(candidate_coefficient[0].item()),
            global_log_damping=float(candidate_coefficient[1].item()),
            global_log_coupling=0.0,
            global_distance_scale=float(torch.exp(candidate_coefficient[0]).item()),
            global_velocity_damping_per_second=float(damping.item()),
            global_coupling_gain=float(self.initial_coupling_gain),
            global_coupling_fixed=1,
            warp_lm_step_maximum=float(effective_step.abs().max().item()),
        )

    def _replace_local_candidate_step_with_warp_gradient(
        self,
        candidate: PaperStiffnessCandidate,
        warp_finite_difference_gradient: torch.Tensor,
        short_horizon_gradient: torch.Tensor | None = None,
    ) -> None:
        """Use Warp long-horizon Adam with an optional causal H1 constraint."""
        settings = self.settings
        gradient = warp_finite_difference_gradient.detach().to(
            device=self.low_dimensional_log_coefficients.device,
            dtype=self.low_dimensional_log_coefficients.dtype,
        ).clone()
        if gradient.shape != self.low_dimensional_log_coefficients.shape:
            raise ValueError("Warp local gradient has the wrong shape")
        if not bool(torch.isfinite(gradient).all().item()):
            raise ValueError("Warp local gradient must be finite")
        gradient = self.project_local_gradient(gradient)
        next_step = self.adam_step + 1
        beta1 = settings.signal_ema_decay
        beta2 = settings.autograd_adam_beta2
        first = beta1 * self.adam_first_moment + (1.0 - beta1) * gradient
        second = beta2 * self.adam_second_moment + (
            1.0 - beta2
        ) * gradient.square()
        first_hat = first / (1.0 - beta1**next_step)
        second_hat = second / (1.0 - beta2**next_step)
        coefficient_step = -settings.log_learning_rate * first_hat / (
            torch.sqrt(second_hat) + settings.autograd_adam_epsilon
        )
        coefficient_step = torch.clamp(
            coefficient_step,
            min=-settings.maximum_log_step,
            max=settings.maximum_log_step,
        )
        coefficient_step = self.project_local_gradient(coefficient_step)
        short_constraint_applied = 0
        short_constraint_pre_dot = float("nan")
        short_constraint_target_dot = float("nan")
        short_constraint_post_dot = float("nan")
        short_constraint_projection_norm = 0.0
        short_gradient_norm = float("nan")
        short_long_gradient_cosine = float("nan")
        if short_horizon_gradient is not None:
            short_gradient = short_horizon_gradient.detach().to(
                device=coefficient_step.device,
                dtype=coefficient_step.dtype,
            ).clone()
            if short_gradient.shape != coefficient_step.shape:
                raise ValueError("Warp H1 gradient has the wrong shape")
            if not bool(torch.isfinite(short_gradient).all().item()):
                raise ValueError("Warp H1 gradient must be finite")
            short_gradient = self.project_local_gradient(short_gradient)
            short_norm = torch.linalg.vector_norm(short_gradient)
            long_norm = torch.linalg.vector_norm(gradient)
            step_norm = torch.linalg.vector_norm(coefficient_step)
            short_gradient_norm = float(short_norm.item())
            gradient_denominator = short_norm * long_norm
            short_long_gradient_cosine = (
                float(torch.dot(short_gradient, gradient).item())
                / float(gradient_denominator.item())
                if float(gradient_denominator.item()) > 1.0e-12
                else float("nan")
            )
            if float(short_norm.item()) > 1.0e-12 and float(step_norm.item()) > 0.0:
                pre_dot = torch.dot(short_gradient, coefficient_step)
                target_dot = (
                    -settings.hierarchical_short_horizon_descent_margin
                    * short_norm
                    * step_norm
                )
                short_constraint_pre_dot = float(pre_dot.item())
                short_constraint_target_dot = float(target_dot.item())
                if float(pre_dot.item()) > float(target_dot.item()):
                    correction = (
                        (pre_dot - target_dot)
                        / short_gradient.square().sum().clamp_min(1.0e-12)
                    ) * short_gradient
                    coefficient_step = coefficient_step - correction
                    coefficient_step = self.project_local_gradient(
                        coefficient_step
                    )
                    short_constraint_projection_norm = float(
                        torch.linalg.vector_norm(correction).item()
                    )
                    short_constraint_applied = 1
                # A uniform rescale preserves the H1 descent half-space while
                # enforcing the same maximum parameter step as the old Adam.
                maximum = coefficient_step.abs().max()
                if float(maximum.item()) > settings.maximum_log_step:
                    coefficient_step = coefficient_step * (
                        settings.maximum_log_step / maximum
                    )
                short_constraint_target_dot = float(
                    (
                        -settings.hierarchical_short_horizon_descent_margin
                        * short_norm
                        * torch.linalg.vector_norm(coefficient_step)
                    ).item()
                )
                short_constraint_post_dot = float(
                    torch.dot(short_gradient, coefficient_step).item()
                )
        coefficient = self.low_dimensional_log_coefficients.detach().clone()
        candidate_coefficient = self.normalize_local_coefficients(
            coefficient + coefficient_step
        )
        candidate_coefficient = torch.clamp(
            candidate_coefficient,
            min=-settings.autograd_maximum_log_offset,
            max=settings.autograd_maximum_log_offset,
        )
        candidate_coefficient = self.normalize_local_coefficients(
            candidate_coefficient
        )
        effective_step = candidate_coefficient - coefficient
        distance = self.local_distance_from_coefficients(candidate_coefficient)
        particle_step = torch.log(
            distance / self.distance_stiffness.detach().clamp_min(1.0e-12)
        )
        descent = -gradient / gradient.abs().max().clamp_min(1.0e-12)
        particle_signal = self.local_zero_mean_log_field(descent)
        log_field = self.local_zero_mean_log_field(candidate_coefficient)
        dynamic_weights = torch.where(
            ~self.local_parameter_exclusion_mask,
            self.inverse_mass.clamp_min(0.0),
            torch.zeros_like(self.inverse_mass),
        )
        weighted_log_mean = torch.sum(log_field * dynamic_weights) / (
            dynamic_weights.sum().clamp_min(1.0e-12)
        )
        with torch.no_grad():
            candidate.distance_stiffness.copy_(distance)
            candidate.shape_stiffness.copy_(self.initial_shape)
            candidate.log_step.copy_(particle_step)
            candidate.signal_ema.copy_(particle_signal)
            candidate.signal_ema[self.local_parameter_exclusion_mask] = 0.0
        candidate.low_dimensional_log_coefficients = (
            candidate_coefficient.detach().clone()
        )
        candidate.adam_first_moment = first.detach().clone()
        candidate.adam_second_moment = second.detach().clone()
        candidate.adam_step = next_step
        candidate.metrics.update(
            optimizer_gradient_source="warp_finite_difference_local_h1_h3_h5",
            warp_adam_step_maximum=float(effective_step.abs().max().item()),
            warp_fd_long_horizon_gradient_norm=float(
                torch.linalg.vector_norm(gradient).item()
            ),
            warp_fd_short_horizon_gradient_norm=short_gradient_norm,
            warp_fd_short_long_gradient_cosine=short_long_gradient_cosine,
            short_constraint_pre_directional_derivative=(
                short_constraint_pre_dot
            ),
            short_constraint_target_directional_derivative=(
                short_constraint_target_dot
            ),
            short_constraint_post_directional_derivative=(
                short_constraint_post_dot
            ),
            short_constraint_projection_applied=short_constraint_applied,
            short_constraint_projection_norm=(
                short_constraint_projection_norm
            ),
            short_horizon_descent_margin=float(
                settings.hierarchical_short_horizon_descent_margin
            ),
            local_coefficient_minimum=float(candidate_coefficient.min().item()),
            local_coefficient_median=float(candidate_coefficient.median().item()),
            local_coefficient_maximum=float(candidate_coefficient.max().item()),
            local_particle_log_minimum=float(log_field.min().item()),
            local_particle_log_median=float(log_field.median().item()),
            local_particle_log_maximum=float(log_field.max().item()),
            local_weighted_particle_log_mean=float(weighted_log_mean.item()),
            local_distance_minimum=float(distance.min().item()),
            local_distance_median=float(distance.median().item()),
            local_distance_maximum=float(distance.max().item()),
            local_velocity_damping_per_second=float(
                self.initial_velocity_damping_per_second
            ),
            local_global_mean_frozen=int(
                settings.update_mode
                not in {
                    "differentiable_hierarchical_relative",
                    "differentiable_particle_graph_lm",
                }
            ),
            hierarchical_global_log_distance=float(
                candidate_coefficient[0].item()
                if settings.update_mode in {
                    "differentiable_hierarchical_relative",
                    "differentiable_particle_graph_lm",
                }
                else 0.0
            ),
            local_velocity_damping_frozen=1,
        )

    def replace_candidate_step_with_warp_gradient(
        self,
        candidate: PaperStiffnessCandidate,
        warp_finite_difference_gradient: torch.Tensor,
        *,
        short_horizon_gradient: torch.Tensor | None = None,
    ) -> None:
        """Rebuild the pending Adam step from the authoritative Warp gradient.

        Torch autograd remains a direction-consistency gate.  Once that gate
        passes, neither its gradient magnitude nor its Adam proposal is used
        for the committed material state.
        """
        if candidate is not self.pending_candidate:
            raise RuntimeError("Cannot rewrite an unknown stiffness candidate")
        if candidate.metrics.get("gradient_consistency_status") != "passed":
            raise RuntimeError("Warp gradient may update only a passed candidate")
        if self.settings.update_mode in {
            "differentiable_local_relative",
            "differentiable_hierarchical_relative",
        }:
            self._replace_local_candidate_step_with_warp_gradient(
                candidate,
                warp_finite_difference_gradient,
                short_horizon_gradient=short_horizon_gradient,
            )
            return
        if short_horizon_gradient is not None:
            raise ValueError(
                "H1-safe projection is implemented only for local/hierarchical mode"
            )
        settings = self.settings
        gradient = warp_finite_difference_gradient.detach().to(
            device=self.global_log_coefficients.device,
            dtype=self.global_log_coefficients.dtype,
        ).clone()
        if gradient.shape != self.global_log_coefficients.shape:
            raise ValueError("Warp gradient has the wrong shape")
        if not settings.global_optimize_coupling:
            gradient[2] = 0.0
        if not bool(torch.isfinite(gradient).all().item()):
            raise ValueError("Warp gradient must be finite")

        next_step = self.global_adam_step + 1
        cumulative_mode = (
            settings.update_mode == "differentiable_global_relative"
        )
        if cumulative_mode:
            active_parameter_count = (
                3 if settings.global_optimize_coupling else 2
            )
            active_norm = torch.linalg.vector_norm(
                gradient[:active_parameter_count]
            ).clamp_min(1.0e-12)
            normalized_gradient = gradient / active_norm
            if not settings.global_optimize_coupling:
                normalized_gradient[2] = 0.0
            # The material is constant throughout an episode.  Retain every
            # verified causal direction instead of allowing the final three
            # high-motion frames to overwrite earlier evidence as Adam EMA
            # did in the previous experiment.
            first = self.global_adam_first_moment + normalized_gradient
            second = (
                self.global_adam_second_moment
                + normalized_gradient.square()
            )
            first_hat = first / float(next_step)
            second_hat = second / float(next_step)
        else:
            beta1 = settings.signal_ema_decay
            beta2 = settings.autograd_adam_beta2
            first = (
                beta1 * self.global_adam_first_moment
                + (1.0 - beta1) * gradient
            )
            second = (
                beta2 * self.global_adam_second_moment
                + (1.0 - beta2) * gradient.square()
            )
            first_hat = first / (1.0 - beta1**next_step)
            second_hat = second / (1.0 - beta2**next_step)
        coefficient_step = -settings.log_learning_rate * first_hat / (
            torch.sqrt(second_hat) + settings.autograd_adam_epsilon
        )
        coefficient_step = torch.clamp(
            coefficient_step,
            min=-settings.maximum_log_step,
            max=settings.maximum_log_step,
        )
        if not settings.global_optimize_coupling:
            coefficient_step[2] = 0.0
        coefficient = self.global_log_coefficients.detach().clone()
        candidate_coefficient = torch.clamp(
            coefficient + coefficient_step,
            min=-settings.autograd_maximum_log_offset,
            max=settings.autograd_maximum_log_offset,
        )
        if not settings.global_optimize_coupling:
            candidate_coefficient[2] = 0.0
        effective_step = candidate_coefficient - coefficient
        distance = torch.clamp(
            self.initial_distance * torch.exp(candidate_coefficient[0]),
            settings.distance_minimum,
            settings.distance_maximum,
        )
        damping = torch.clamp(
            torch.as_tensor(
                self.initial_velocity_damping_per_second,
                device=candidate_coefficient.device,
                dtype=candidate_coefficient.dtype,
            )
            * torch.exp(candidate_coefficient[1]),
            settings.global_damping_minimum_per_second,
            settings.global_damping_maximum_per_second,
        )
        coupling = (
            torch.clamp(
                torch.as_tensor(
                    self.initial_coupling_gain,
                    device=candidate_coefficient.device,
                    dtype=candidate_coefficient.dtype,
                )
                * torch.exp(candidate_coefficient[2]),
                settings.global_coupling_minimum,
                settings.global_coupling_maximum,
            )
            if settings.global_optimize_coupling
            else torch.as_tensor(
                self.initial_coupling_gain,
                device=candidate_coefficient.device,
                dtype=candidate_coefficient.dtype,
            )
        )
        with torch.no_grad():
            candidate.distance_stiffness.copy_(distance)
            candidate.shape_stiffness.copy_(self.initial_shape)
            candidate.log_step.fill_(float(effective_step[0].item()))
            candidate.log_step[~candidate.eligible_mask] = 0.0
            signal = -gradient[0] / gradient.abs().max().clamp_min(1.0e-12)
            candidate.signal_ema.fill_(float(signal.item()))
            candidate.signal_ema[~candidate.eligible_mask] = 0.0
        candidate.global_log_coefficients = candidate_coefficient.detach().clone()
        candidate.global_adam_first_moment = first.detach().clone()
        candidate.global_adam_second_moment = second.detach().clone()
        candidate.global_adam_step = next_step
        candidate.global_velocity_damping_per_second = float(damping.item())
        candidate.global_coupling_gain = float(coupling.item())
        candidate.metrics.update(
            optimizer_gradient_source=(
                "warp_finite_difference_cumulative_normalized"
                if cumulative_mode
                else "warp_finite_difference"
            ),
            global_log_distance=float(candidate_coefficient[0].item()),
            global_log_damping=float(candidate_coefficient[1].item()),
            global_log_coupling=float(candidate_coefficient[2].item()),
            global_distance_scale=float(torch.exp(candidate_coefficient[0]).item()),
            global_velocity_damping_per_second=float(damping.item()),
            global_coupling_gain=float(coupling.item()),
            global_coupling_fixed=int(not settings.global_optimize_coupling),
            warp_adam_step_maximum=float(effective_step.abs().max().item()),
            warp_gradient_accumulator_count=(
                int(next_step) if cumulative_mode else 0
            ),
            warp_gradient_consensus_norm=(
                float(torch.linalg.vector_norm(first_hat).item())
                if cumulative_mode
                else 0.0
            ),
        )

    def reconfigure(self, settings: OnlineTissueStiffnessSettings) -> None:
        """Install runtime tuning without mixing old EMA into the new policy.

        The caller must finish or reject an outstanding candidate first.  The
        verified material field is preserved, except that values outside newly
        selected absolute bounds are clipped.  Reset baselines are unchanged.
        """
        settings.validate()
        if self.pending_candidate is not None:
            raise RuntimeError(
                "Cannot reconfigure online stiffness with a pending candidate"
            )
        previous_update_mode = self.settings.update_mode
        rebuild_basis = (
            settings.autograd_region_count
            != self.settings.autograd_region_count
            or settings.spatial_smoothing_iterations
            != self.settings.spatial_smoothing_iterations
            or settings.spatial_smoothing_blend
            != self.settings.spatial_smoothing_blend
            or (settings.update_mode == "differentiable_hierarchical_relative")
            != (self.settings.update_mode == "differentiable_hierarchical_relative")
            or (settings.update_mode == "differentiable_particle_graph_lm")
            != (self.settings.update_mode == "differentiable_particle_graph_lm")
        )
        with torch.no_grad():
            self.distance_stiffness.clamp_(
                settings.distance_minimum,
                settings.distance_maximum,
            )
            self.shape_stiffness.clamp_(
                settings.shape_minimum,
                settings.shape_maximum,
            )
            self.signal_ema.zero_()
        self.settings = settings
        self.particle_observability = self._build_particle_observability()
        if rebuild_basis:
            self.low_dimensional_basis = self._build_low_dimensional_basis(
                settings.autograd_region_count
            )
            self._rebuild_track_region_basis()
            coefficient_count = self.low_dimensional_basis.shape[1]
            if settings.update_mode == "differentiable_particle_graph_lm":
                coefficient_count = len(self.rest_positions) + 1
            elif settings.update_mode == "differentiable_hierarchical_relative":
                coefficient_count += 1
            self.low_dimensional_log_coefficients = torch.zeros(
                coefficient_count,
                dtype=torch.float32,
                device=self.rest_positions.device,
            )
            self.adam_first_moment = torch.zeros_like(
                self.low_dimensional_log_coefficients
            )
            self.adam_second_moment = torch.zeros_like(
                self.low_dimensional_log_coefficients
            )
            self.adam_step = 0
        if settings.update_mode != previous_update_mode:
            with torch.no_grad():
                self.global_log_coefficients.zero_()
                self.global_adam_first_moment.zero_()
                self.global_adam_second_moment.zero_()
            self.global_adam_step = 0
            self.global_velocity_damping_per_second = (
                self.initial_velocity_damping_per_second
            )
            self.global_coupling_gain = self.initial_coupling_gain
        self.last_metrics = None
        self.last_proposal_diagnostics = None
        self.last_proposal_candidate_count = self.candidate_count
        self.causal_material_history.clear()

    def invalidate_signal_history(
        self, mask: torch.Tensor | None = None
    ) -> None:
        """Clear EMA evidence invalidated by a control/contact transition."""
        with torch.no_grad():
            if mask is None:
                self.signal_ema.zero_()
                self.adam_first_moment.zero_()
                self.adam_second_moment.zero_()
                self.global_adam_first_moment.zero_()
                self.global_adam_second_moment.zero_()
                self.adam_step = 0
                self.global_adam_step = 0
                self.causal_material_history.clear()
                return
            invalid = self._validated_mask(
                mask, default=False, label="EMA invalidation"
            )
            self.signal_ema[invalid] = 0.0
            if bool(invalid.any().item()):
                dominant_region = torch.argmax(
                    self.low_dimensional_basis, dim=1
                )
                invalid_regions = torch.zeros_like(
                    self.adam_first_moment, dtype=torch.bool
                )
                invalid_regions[dominant_region[invalid]] = True
                self.adam_first_moment[invalid_regions] = 0.0
                self.adam_second_moment[invalid_regions] = 0.0

    def _smooth(self, signal: torch.Tensor) -> torch.Tensor:
        if self.settings.spatial_smoothing_iterations == 0:
            return signal
        source = self.edges[:, 0]
        target = self.edges[:, 1]
        ones = torch.ones_like(source, dtype=signal.dtype)
        counts = torch.zeros_like(signal)
        counts.index_add_(0, source, ones)
        counts.index_add_(0, target, ones)
        current = signal
        for _ in range(self.settings.spatial_smoothing_iterations):
            sums = torch.zeros_like(current)
            sums.index_add_(0, source, current[target])
            sums.index_add_(0, target, current[source])
            neighbor_average = sums / counts.clamp_min(1.0)
            current = torch.lerp(
                current,
                neighbor_average,
                self.settings.spatial_smoothing_blend,
            )
        return current

    def _edge_strain_signal(
        self,
        prediction: torch.Tensor,
        corrected: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return node-local log-stiffness evidence from RGB-corrected strain.

        Positive evidence means the physical prediction strained an edge more
        than the RGB-corrected state and therefore hardens it; negative
        evidence softens it.  Edge lengths make the update invariant to the
        free sheet's global translation and rotation.
        """
        if not self.edges.numel():
            zeros = torch.zeros(
                len(self.rest_positions),
                dtype=prediction.dtype,
                device=prediction.device,
            )
            return zeros, zeros
        source, target = self.edges[:, 0], self.edges[:, 1]
        predicted_lengths = torch.linalg.vector_norm(
            prediction[target] - prediction[source], dim=1
        )
        corrected_lengths = torch.linalg.vector_norm(
            corrected[target] - corrected[source], dim=1
        )
        predicted_strain = (
            predicted_lengths - self.rest_edge_lengths
        ) / self.rest_edge_lengths
        corrected_strain = (
            corrected_lengths - self.rest_edge_lengths
        ) / self.rest_edge_lengths
        evidence_scale = torch.maximum(
            predicted_strain.abs(), corrected_strain.abs()
        )
        # Do not infer material from strain created solely by the image
        # correction when the physical model predicted a rigid edge.  Such a
        # frame identifies missing state, not yet a stiffness derivative.
        active = predicted_strain.abs() >= self.settings.minimum_edge_strain
        denominator = torch.maximum(
            evidence_scale,
            torch.full_like(evidence_scale, self.settings.minimum_edge_strain),
        )
        edge_signal = torch.where(
            active,
            (predicted_strain.abs() - corrected_strain.abs()) / denominator,
            torch.zeros_like(evidence_scale),
        ).clamp(-1.0, 1.0)
        confidence = torch.where(
            active,
            torch.clamp(
                evidence_scale / self.settings.edge_strain_full_scale,
                max=1.0,
            ),
            torch.zeros_like(evidence_scale),
        )
        node_sum = torch.zeros(
            len(self.rest_positions), dtype=prediction.dtype, device=prediction.device
        )
        node_weight = torch.zeros_like(node_sum)
        weighted = edge_signal * confidence
        node_sum.index_add_(0, source, weighted)
        node_sum.index_add_(0, target, weighted)
        node_weight.index_add_(0, source, confidence)
        node_weight.index_add_(0, target, confidence)
        return node_sum / node_weight.clamp_min(1.0e-8), node_weight

    def _validated_mask(
        self,
        mask: torch.Tensor | None,
        *,
        default: bool,
        label: str,
    ) -> torch.Tensor:
        if mask is None:
            return torch.full_like(self.fixed_mask, default)
        result = mask.detach().to(
            device=self.rest_positions.device, dtype=torch.bool
        )
        if result.shape != self.fixed_mask.shape:
            raise ValueError(f"Stiffness {label} mask has the wrong shape")
        return result

    def propose(
        self,
        *,
        physical_prediction: torch.Tensor,
        accepted_residual: torch.Tensor,
        quality_valid_mask: torch.Tensor | None = None,
        supervision_valid_mask: torch.Tensor | None = None,
        control_exclusion_mask: torch.Tensor | None = None,
        control_frozen_mask: torch.Tensor | None = None,
        control_coupling_base_positions: torch.Tensor | None = None,
        control_coupling_displacements: torch.Tensor | None = None,
        track_target_positions: torch.Tensor | None = None,
        track_valid_mask: torch.Tensor | None = None,
        rollout_start_positions: torch.Tensor | None = None,
        physical_velocities: torch.Tensor | None = None,
        frame_index: int | None = None,
        globally_paused: bool = False,
    ) -> PaperStiffnessCandidate:
        """Create a bounded candidate without mutating verified stiffness.

        ``control_exclusion_mask`` is a hard target mask as well as a source
        mask: spatial smoothing and EMA history cannot leak an update back into
        a direct/support ``u_t`` neighborhood.
        """
        if self.pending_candidate is not None:
            raise RuntimeError(
                "A stiffness candidate is already awaiting verification"
            )
        settings = self.settings
        prediction = physical_prediction.detach().to(
            device=self.rest_positions.device, dtype=torch.float32
        )
        residual = accepted_residual.detach().to(
            device=self.rest_positions.device, dtype=torch.float32
        )
        if prediction.shape != self.rest_positions.shape:
            raise ValueError("Stiffness update prediction has the wrong shape")
        if residual.shape != self.rest_positions.shape:
            raise ValueError("Stiffness update residual has the wrong shape")
        quality_valid = self._validated_mask(
            quality_valid_mask,
            default=True,
            label="quality-valid",
        )
        supervision_valid = self._validated_mask(
            supervision_valid_mask,
            default=True,
            label="supervision-valid",
        )
        control_excluded = self._validated_mask(
            control_exclusion_mask,
            default=False,
            label="control-exclusion",
        )
        control_frozen = (
            control_excluded.clone()
            if control_frozen_mask is None
            else self._validated_mask(
                control_frozen_mask,
                default=False,
                label="control-frozen",
            )
        )
        if bool((control_frozen & ~control_excluded).any().item()):
            raise ValueError(
                "Direct frozen control must be contained in exclusion mask"
            )
        eligible = (
            quality_valid
            & supervision_valid
            & ~control_excluded
            & ~self.fixed_mask
        )
        if globally_paused:
            eligible.zero_()
        # A bad tetrahedron or direct positional-control region invalidates
        # old evidence immediately, even if the new material candidate is
        # later rejected.  Otherwise stale EMA can reappear as soon as the
        # node becomes eligible again.
        with torch.no_grad():
            self.signal_ema[
                ~quality_valid | control_excluded | self.fixed_mask
            ] = 0.0
        deformation = prediction - self.rest_positions
        corrected = prediction + residual
        deformation_norm = torch.linalg.vector_norm(deformation, dim=1)
        residual_norm = torch.linalg.vector_norm(residual, dim=1)
        active = (
            (residual_norm >= settings.minimum_residual_m)
            & (deformation_norm >= settings.minimum_deformation_m)
            & eligible
        )
        cosine = -torch.sum(deformation * residual, dim=1) / (
            deformation_norm * residual_norm
        ).clamp_min(1.0e-12)
        signed_direction = torch.clamp(
            cosine + settings.hardening_bias, min=-1.0, max=1.0
        )
        residual_strength = torch.clamp(
            residual_norm / settings.residual_full_scale_m, max=1.0
        )
        deformation_strength = torch.clamp(
            deformation_norm / settings.deformation_full_scale_m, max=1.0
        )
        vector_signal = torch.where(
            active,
            signed_direction * residual_strength * deformation_strength,
            torch.zeros_like(signed_direction),
        )
        strain_signal, strain_weight = self._edge_strain_signal(
            prediction, corrected
        )
        strain_active = (
            (strain_weight > 0.0)
            & (residual_norm >= settings.minimum_residual_m)
            & eligible
        )
        blend = settings.strain_signal_weight
        signal = torch.where(
            strain_active,
            blend * strain_signal + (1.0 - blend) * vector_signal,
            vector_signal,
        )
        material_active = active | strain_active
        low_dimensional_log_coefficients = None
        adam_first_moment = None
        adam_second_moment = None
        adam_step = None
        global_log_coefficients = None
        global_adam_first_moment = None
        global_adam_second_moment = None
        global_adam_step = None
        global_velocity_damping_per_second = None
        global_coupling_gain = None
        autograd_parameter_gradient = None
        mode_metrics: dict[str, float | int | str] = {
            "update_mode": settings.update_mode
        }
        if settings.update_mode in {
            "differentiable_low_dim",
            "differentiable_global",
            "differentiable_global_relative",
            "differentiable_global_mhe",
            "differentiable_local_relative",
            "differentiable_hierarchical_relative",
            "differentiable_particle_graph_lm",
        }:
            rollout_start = (
                prediction
                if rollout_start_positions is None
                else rollout_start_positions
            )
            if settings.update_mode in {
                "differentiable_global",
                "differentiable_global_relative",
                "differentiable_global_mhe",
            }:
                (
                    candidate_distance,
                    candidate_shape,
                    log_step,
                    gradient_signal,
                    global_log_coefficients,
                    global_adam_first_moment,
                    global_adam_second_moment,
                    global_adam_step,
                    global_velocity_damping_per_second,
                    global_coupling_gain,
                    autograd_parameter_gradient,
                    mode_metrics,
                ) = self._propose_differentiable_global(
                    physical_prediction=prediction,
                    accepted_residual=residual,
                    rollout_start_positions=rollout_start,
                    physical_velocities=physical_velocities,
                    frame_index=frame_index,
                    eligible=eligible,
                    material_active=material_active,
                    control_excluded=control_excluded,
                    control_frozen=control_frozen,
                    control_coupling_base_positions=(
                        control_coupling_base_positions
                    ),
                    control_coupling_displacements=(
                        control_coupling_displacements
                    ),
                    track_target_positions=track_target_positions,
                    track_valid_mask=track_valid_mask,
                )
            elif settings.update_mode in {
                "differentiable_local_relative",
                "differentiable_hierarchical_relative",
                "differentiable_particle_graph_lm",
            }:
                (
                    candidate_distance,
                    candidate_shape,
                    log_step,
                    gradient_signal,
                    low_dimensional_log_coefficients,
                    adam_first_moment,
                    adam_second_moment,
                    adam_step,
                    autograd_parameter_gradient,
                    mode_metrics,
                ) = self._propose_differentiable_local_relative(
                    physical_prediction=prediction,
                    accepted_residual=residual,
                    rollout_start_positions=rollout_start,
                    physical_velocities=physical_velocities,
                    frame_index=frame_index,
                    eligible=eligible,
                    material_active=material_active,
                    control_excluded=control_excluded,
                    control_frozen=control_frozen,
                    control_coupling_base_positions=(
                        control_coupling_base_positions
                    ),
                    control_coupling_displacements=(
                        control_coupling_displacements
                    ),
                    track_target_positions=track_target_positions,
                    track_valid_mask=track_valid_mask,
                )
            else:
                (
                    candidate_distance,
                    candidate_shape,
                    log_step,
                    gradient_signal,
                    low_dimensional_log_coefficients,
                    adam_first_moment,
                    adam_second_moment,
                    adam_step,
                    mode_metrics,
                ) = self._propose_differentiable_low_dimensional(
                    physical_prediction=prediction,
                    accepted_residual=residual,
                    rollout_start_positions=rollout_start,
                    physical_velocities=physical_velocities,
                    frame_index=frame_index,
                    eligible=eligible,
                    material_active=material_active,
                    control_excluded=control_excluded,
                    control_frozen=control_frozen,
                )
            # These exported fields retain a common schema across modes.  In
            # differentiable mode they describe the low-dimensional Adam
            # descent direction rather than the rejected local heuristic.
            blended_signal = gradient_signal
            signal = gradient_signal
            signal[~eligible] = 0.0
            candidate_ema = signal.detach().clone()
            candidate_ema[
                ~quality_valid | control_excluded | self.fixed_mask
            ] = 0.0
            log_step[~eligible] = 0.0
        else:
            blended_signal = signal
            signal = self._smooth(blended_signal)
            signal[~eligible] = 0.0
            candidate_ema = self.signal_ema * settings.signal_ema_decay
            candidate_ema = candidate_ema.add(
                signal, alpha=1.0 - settings.signal_ema_decay
            )
            # Repeat the hard mask after mixing so neither the current signal nor
            # neighbor smoothing can repopulate invalidated candidate history.
            candidate_ema[
                ~quality_valid | control_excluded | self.fixed_mask
            ] = 0.0
            if globally_paused:
                candidate_ema.zero_()
            log_step = torch.clamp(
                settings.log_learning_rate * candidate_ema,
                min=-settings.maximum_log_step,
                max=settings.maximum_log_step,
            )
            log_step[~eligible] = 0.0
            with torch.no_grad():
                candidate_distance = torch.clamp(
                    self.distance_stiffness.detach() * torch.exp(log_step),
                    settings.distance_minimum,
                    settings.distance_maximum,
                )
                candidate_shape = torch.clamp(
                    self.shape_stiffness.detach()
                    * torch.exp(log_step * settings.shape_update_gain),
                    settings.shape_minimum,
                    settings.shape_maximum,
                )
        self.candidate_count += 1
        self.last_proposal_candidate_count = self.candidate_count
        self.last_proposal_diagnostics = {
            "residual_norm_m": residual_norm.detach().clone(),
            "deformation_norm_m": deformation_norm.detach().clone(),
            "vector_signal": vector_signal.detach().clone(),
            "strain_signal": strain_signal.detach().clone(),
            "strain_confidence": strain_weight.detach().clone(),
            "blended_signal_before_smoothing": blended_signal.detach().clone(),
            "smoothed_signal": signal.detach().clone(),
            "candidate_ema": candidate_ema.detach().clone(),
            "log_step": log_step.detach().clone(),
            "quality_valid_mask": quality_valid.detach().clone(),
            "supervision_valid_mask": supervision_valid.detach().clone(),
            "control_exclusion_mask": control_excluded.detach().clone(),
            "control_frozen_mask": control_frozen.detach().clone(),
            "eligible_mask": eligible.detach().clone(),
            "vector_active_mask": active.detach().clone(),
            "strain_active_mask": strain_active.detach().clone(),
            "material_active_mask": material_active.detach().clone(),
        }
        active_count = int(torch.count_nonzero(material_active).item())
        hardening_count = int(torch.count_nonzero(log_step > 0.0).item())
        softening_count = int(torch.count_nonzero(log_step < 0.0).item())
        quality_valid_count = int(
            torch.count_nonzero(quality_valid & ~self.fixed_mask).item()
        )
        quality_masked_count = int(
            torch.count_nonzero(~quality_valid & ~self.fixed_mask).item()
        )
        supervision_masked_count = int(
            torch.count_nonzero(~supervision_valid & ~self.fixed_mask).item()
        )
        control_excluded_count = int(
            torch.count_nonzero(control_excluded & ~self.fixed_mask).item()
        )
        ema_active_count = int(
            torch.count_nonzero(candidate_ema.abs() > 1.0e-8).item()
        )
        if self.edges.numel():
            distance_roughness = float(
                torch.mean(
                    torch.abs(
                        candidate_distance[self.edges[:, 1]]
                        - candidate_distance[self.edges[:, 0]]
                    )
                ).item()
            )
            shape_roughness = float(
                torch.mean(
                    torch.abs(
                        candidate_shape[self.edges[:, 1]]
                        - candidate_shape[self.edges[:, 0]]
                    )
                ).item()
            )
        else:
            distance_roughness = 0.0
            shape_roughness = 0.0
        metrics: dict[str, float | int | str] = {
            "status": "candidate",
            "candidate_count": self.candidate_count,
            "update_count": self.update_count,
            "rejected_count": self.rejected_count,
            "active_particles": active_count,
            "hardening_particles": hardening_count,
            "softening_particles": softening_count,
            "quality_valid_particles": quality_valid_count,
            "quality_masked_particles": quality_masked_count,
            "supervision_masked_particles": supervision_masked_count,
            "control_excluded_particles": control_excluded_count,
            "ema_active_particles": ema_active_count,
            "globally_paused": int(globally_paused),
            "maximum_log_step": float(log_step.abs().max().item()),
            "mean_absolute_log_step": float(log_step.abs().mean().item()),
            "distance_edge_roughness": distance_roughness,
            "shape_edge_roughness": shape_roughness,
            "distance_minimum": float(candidate_distance.min().item()),
            "distance_median": float(candidate_distance.median().item()),
            "distance_maximum": float(candidate_distance.max().item()),
            "shape_minimum": float(candidate_shape.min().item()),
            "shape_median": float(candidate_shape.median().item()),
            "shape_maximum": float(candidate_shape.max().item()),
            **mode_metrics,
        }
        candidate = PaperStiffnessCandidate(
            distance_stiffness=candidate_distance.detach().clone(),
            shape_stiffness=candidate_shape.detach().clone(),
            signal_ema=candidate_ema.detach().clone(),
            log_step=log_step.detach().clone(),
            eligible_mask=eligible.detach().clone(),
            source_mask=material_active.detach().clone(),
            metrics=metrics,
            low_dimensional_log_coefficients=(
                None
                if low_dimensional_log_coefficients is None
                else low_dimensional_log_coefficients.detach().clone()
            ),
            adam_first_moment=(
                None
                if adam_first_moment is None
                else adam_first_moment.detach().clone()
            ),
            adam_second_moment=(
                None
                if adam_second_moment is None
                else adam_second_moment.detach().clone()
            ),
            adam_step=adam_step,
            global_log_coefficients=(
                None
                if global_log_coefficients is None
                else global_log_coefficients.detach().clone()
            ),
            global_adam_first_moment=(
                None
                if global_adam_first_moment is None
                else global_adam_first_moment.detach().clone()
            ),
            global_adam_second_moment=(
                None
                if global_adam_second_moment is None
                else global_adam_second_moment.detach().clone()
            ),
            global_adam_step=global_adam_step,
            global_velocity_damping_per_second=(
                global_velocity_damping_per_second
            ),
            global_coupling_gain=global_coupling_gain,
            autograd_parameter_gradient=(
                None
                if autograd_parameter_gradient is None
                else autograd_parameter_gradient.detach().clone()
            ),
        )
        self.pending_candidate = candidate
        self.last_metrics = metrics
        return candidate

    def install_candidate_for_rollout(
        self, candidate: PaperStiffnessCandidate | None = None
    ) -> None:
        """Temporarily expose a candidate to XPBD for an isolated rollout."""
        candidate = candidate or self.pending_candidate
        if candidate is None:
            raise RuntimeError("No stiffness candidate is available")
        with torch.no_grad():
            self.distance_stiffness.copy_(candidate.distance_stiffness)
            self.shape_stiffness.copy_(candidate.shape_stiffness)

    def restore_verified_stiffness(
        self,
        distance: torch.Tensor,
        shape: torch.Tensor,
    ) -> None:
        """Restore explicitly saved verified arrays after a shadow rollout."""
        with torch.no_grad():
            self.distance_stiffness.copy_(distance)
            self.shape_stiffness.copy_(shape)

    def commit(
        self, candidate: PaperStiffnessCandidate | None = None
    ) -> dict[str, float | int | str]:
        candidate = candidate or self.pending_candidate
        if candidate is None or candidate is not self.pending_candidate:
            raise RuntimeError("Cannot commit an unknown stiffness candidate")
        self.install_candidate_for_rollout(candidate)
        with torch.no_grad():
            self.signal_ema.copy_(candidate.signal_ema)
            if candidate.low_dimensional_log_coefficients is not None:
                self.low_dimensional_log_coefficients.copy_(
                    candidate.low_dimensional_log_coefficients
                )
            if candidate.adam_first_moment is not None:
                self.adam_first_moment.copy_(candidate.adam_first_moment)
            if candidate.adam_second_moment is not None:
                self.adam_second_moment.copy_(candidate.adam_second_moment)
            if candidate.global_log_coefficients is not None:
                self.global_log_coefficients.copy_(
                    candidate.global_log_coefficients
                )
            if candidate.global_adam_first_moment is not None:
                self.global_adam_first_moment.copy_(
                    candidate.global_adam_first_moment
                )
            if candidate.global_adam_second_moment is not None:
                self.global_adam_second_moment.copy_(
                    candidate.global_adam_second_moment
                )
        if candidate.adam_step is not None:
            self.adam_step = int(candidate.adam_step)
        if candidate.global_adam_step is not None:
            self.global_adam_step = int(candidate.global_adam_step)
        if candidate.global_velocity_damping_per_second is not None:
            self.global_velocity_damping_per_second = float(
                candidate.global_velocity_damping_per_second
            )
        if candidate.global_coupling_gain is not None:
            self.global_coupling_gain = float(candidate.global_coupling_gain)
        self.update_count += 1
        metrics = dict(candidate.metrics)
        metrics.update(
            status="committed",
            update_count=self.update_count,
            rejected_count=self.rejected_count,
        )
        self.pending_candidate = None
        self.last_metrics = metrics
        return metrics

    def reject(
        self,
        reason: str,
        candidate: PaperStiffnessCandidate | None = None,
    ) -> dict[str, float | int | str]:
        candidate = candidate or self.pending_candidate
        if candidate is None or candidate is not self.pending_candidate:
            raise RuntimeError("Cannot reject an unknown stiffness candidate")
        with torch.no_grad():
            self.signal_ema.mul_(self.settings.rejected_ema_decay)
        self.rejected_count += 1
        metrics = dict(candidate.metrics)
        metrics.update(
            status="rejected",
            rejection_reason=str(reason),
            update_count=self.update_count,
            rejected_count=self.rejected_count,
        )
        self.pending_candidate = None
        self.last_metrics = metrics
        return metrics

    def update(
        self,
        *,
        physical_prediction: torch.Tensor,
        accepted_residual: torch.Tensor,
        quality_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, float | int | str]:
        """Legacy immediate update retained for isolated unit tests only."""
        candidate = self.propose(
            physical_prediction=physical_prediction,
            accepted_residual=accepted_residual,
            quality_valid_mask=quality_valid_mask,
        )
        return self.commit(candidate)
