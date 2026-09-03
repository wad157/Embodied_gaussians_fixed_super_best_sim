#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${1:-$ROOT_DIR/outputs/sim_known_grasp_autograd4_conservative_$(date +%Y%m%d_%H%M%S)}"

export STIFFNESS_UPDATE_MODE=differentiable_low_dim
export STIFFNESS_LOG_LR=0.03
export STIFFNESS_LOG_CAP=0.02
export STIFFNESS_EMA_DECAY=0.90
export STIFFNESS_SMOOTHING_ITERATIONS=3
export STIFFNESS_SMOOTHING_BLEND=0.35
export STIFFNESS_STRAIN_WEIGHT=0.20
export STIFFNESS_AUTOGRAD_STEPS=4
export STIFFNESS_AUTOGRAD_REGIONS=12

exec bash "$ROOT_DIR/scripts/run_sim_known_grasp_full_evaluation.sh" "$OUTPUT_ROOT"
