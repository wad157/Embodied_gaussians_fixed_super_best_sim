#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 不做首帧深度对齐。A/B沿用同一数据、同一协议下已完成的正式基线；
# C只改变刚度更新：以H3/H5作为长期目标，并把Adam参数步投影到H1下降半空间。
export SIM_METHOD_C="pbd_cotracker_foundation_depth_hierarchical_stiffness_h1_safe"
export SIM_STIFFNESS_UPDATE_MODE="differentiable_hierarchical_relative"
export SIM_C_LABEL="C-H1：B + H1短期约束的H3/H5全局/区域刚度更新"
export SIM_STIFFNESS_AUTOGRAD_UNROLL_STEPS=5
export SIM_STIFFNESS_AUTOGRAD_REGION_COUNT=6

bash "$ROOT_DIR/scripts/run_sim_foundation_observable_mhe_probe.sh" \
    "${1:-$ROOT_DIR/outputs/sim_foundation_h1_safe_hierarchical_complete_v1}"
