#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SIM_METHOD_C="pbd_cotracker_foundation_depth_hierarchical_stiffness"
export SIM_STIFFNESS_UPDATE_MODE="differentiable_hierarchical_relative"
export SIM_C_LABEL="C-H：B + 全局均值/零均值区域刚度联合在线更新"
export SIM_STIFFNESS_AUTOGRAD_UNROLL_STEPS=5
export SIM_STIFFNESS_AUTOGRAD_REGION_COUNT=6

bash "$ROOT_DIR/scripts/run_sim_foundation_observable_mhe_probe.sh" \
    "${1:-$ROOT_DIR/outputs/sim_foundation_hierarchical_stiffness_complete_v1}"
