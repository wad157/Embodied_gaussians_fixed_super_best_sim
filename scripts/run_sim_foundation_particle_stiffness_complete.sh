#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SIM_METHOD_C="pbd_cotracker_foundation_depth_particle_stiffness"
export SIM_STIFFNESS_UPDATE_MODE="particle_residual"
export SIM_C_LABEL="C-P：B + 逐粒子RGB残差/边应变刚度更新"

bash "$ROOT_DIR/scripts/run_sim_foundation_observable_mhe_probe.sh" \
    "${1:-$ROOT_DIR/outputs/sim_foundation_particle_stiffness_complete_v1}"
