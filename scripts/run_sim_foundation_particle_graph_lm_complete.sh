#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# A/B/C all use the same fixed, moderate paper-PBD initialization supplied by
# the common evaluator: distance=0.20 and shape=0.004. No initial-parameter
# fitting, sweep, GT stiffness, or evaluation-metric selection is performed.
export SIM_METHOD_C="pbd_cotracker_foundation_depth_particle_graph_lm"
export SIM_STIFFNESS_UPDATE_MODE="differentiable_particle_graph_lm"
export SIM_C_LABEL="C-GraphLM：B + 全局均值/逐粒子图正则刚度在线更新"
export SIM_STIFFNESS_AUTOGRAD_UNROLL_STEPS=5
export SIM_STIFFNESS_AUTOGRAD_REGION_COUNT=12
# Particle-graph defaults use a fixed fast protocol: two deterministic Warp
# directions, one material update every ten frames, and eight graph-CG steps.

bash "$ROOT_DIR/scripts/run_sim_foundation_observable_mhe_probe.sh" \
    "${1:-$ROOT_DIR/outputs/sim_foundation_particle_graph_lm_complete_v1}"
