#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="$ROOT_DIR/data/sim/tissue_retraction_closeup_inplane_full_v1"
ASSETS="$ROOT_DIR/outputs/sim_inplane_cotracker3_foundation_depth_v1"
DEPTH="$DATASET/estimated_depth/foundation_stereo_rgb_v2_rig_aware"
EVALUATION="${1:-$ROOT_DIR/outputs/sim_inplane_foundation_h1_safe_complete_v1}"

# 旧平面内双目相机具有相反的视差方向和更强的会聚角；alpha=0.85只属于
# 该数据集的相机标定预处理。夹起拉升任务继续使用自己的alpha=0资产。
SIM_DATASET="$DATASET" \
SIM_ESTIMATED_DEPTH_ROOT="$DEPTH" \
SIM_DEPTH_SOURCE_LABEL="foundation_stereo_rgb_rig_aware_v2" \
SIM_RECTIFICATION_ALPHA="0.85" \
    bash "$ROOT_DIR/scripts/prepare_sim_foundation_dataset_assets.sh" \
    "$ASSETS"

SIM_DATASET="$DATASET" \
SIM_ESTIMATED_DEPTH_ROOT="$DEPTH" \
FLOW_DEPTH_ASSET_ROOT="$ASSETS/flow_depth_assets" \
    bash "$ROOT_DIR/scripts/run_sim_foundation_h1_safe_dataset_complete.sh" \
    "$EVALUATION"

echo "平面内拉扯独立测评完成：$EVALUATION/comparison_complete.md"
