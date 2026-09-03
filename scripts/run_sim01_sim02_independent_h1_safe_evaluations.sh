#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIM01="$ROOT_DIR/data/sim/tissue_retraction_free_support_front_v2"
SIM02="$ROOT_DIR/data/sim/tissue_retraction_free_support_side_v2"
SIM01_ASSETS="$ROOT_DIR/outputs/sim01_planar_x_cotracker_foundation_v1"
SIM02_ASSETS="$ROOT_DIR/outputs/sim02_planar_y_cotracker_foundation_v1"
SIM01_DEPTH="$SIM01/estimated_depth/foundation_stereo_rgb_rig_aware_v1"
SIM02_DEPTH="$SIM02/estimated_depth/foundation_stereo_rgb_rig_aware_v1"
SIM01_EVAL="$ROOT_DIR/outputs/sim01_planar_x_h1_safe_complete_v1"
SIM02_EVAL="$ROOT_DIR/outputs/sim02_planar_y_h1_safe_complete_v1"
SIM03_EVAL="$ROOT_DIR/outputs/sim_foundation_h1_safe_hierarchical_complete_v1"

SIM_DATASET="$SIM01" \
SIM_ESTIMATED_DEPTH_ROOT="$SIM01_DEPTH" \
SIM_GPU_ID=0 \
SIM_RECTIFICATION_ALPHA=0.85 \
SIM_DEPTH_SOURCE_LABEL="sim01_foundation_stereo_rgb_rig_aware_v1" \
    bash "$ROOT_DIR/scripts/prepare_sim_numbered_dataset_assets.sh" \
    "$SIM01_ASSETS" &
sim01_prepare_pid=$!
SIM_DATASET="$SIM02" \
SIM_ESTIMATED_DEPTH_ROOT="$SIM02_DEPTH" \
SIM_GPU_ID=1 \
SIM_RECTIFICATION_ALPHA=0.85 \
SIM_DEPTH_SOURCE_LABEL="sim02_foundation_stereo_rgb_rig_aware_v1" \
    bash "$ROOT_DIR/scripts/prepare_sim_numbered_dataset_assets.sh" \
    "$SIM02_ASSETS" &
sim02_prepare_pid=$!

set +e
wait "$sim01_prepare_pid"
sim01_prepare_status=$?
wait "$sim02_prepare_pid"
sim02_prepare_status=$?
set -e
if [[ "$sim01_prepare_status" -ne 0 || "$sim02_prepare_status" -ne 0 ]]; then
    echo "ERROR: 资产生成失败：SIM-01=$sim01_prepare_status SIM-02=$sim02_prepare_status" >&2
    exit 3
fi

SIM_DATASET="$SIM01" \
SIM_ESTIMATED_DEPTH_ROOT="$SIM01_DEPTH" \
FLOW_DEPTH_ASSET_ROOT="$SIM01_ASSETS/flow_depth_assets" \
    bash "$ROOT_DIR/scripts/run_sim_foundation_h1_safe_dataset_complete.sh" \
    "$SIM01_EVAL"

SIM_DATASET="$SIM02" \
SIM_ESTIMATED_DEPTH_ROOT="$SIM02_DEPTH" \
FLOW_DEPTH_ASSET_ROOT="$SIM02_ASSETS/flow_depth_assets" \
    bash "$ROOT_DIR/scripts/run_sim_foundation_h1_safe_dataset_complete.sh" \
    "$SIM02_EVAL"

"${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python" \
    "$ROOT_DIR/scripts/summarize_sim_h1_safe_three_datasets.py" \
    --sim01 "$SIM01_EVAL/comparison_complete.json" \
    --sim02 "$SIM02_EVAL/comparison_complete.json" \
    --sim03 "$SIM03_EVAL/comparison_complete.json" \
    --output "$ROOT_DIR/outputs/sim01_sim02_sim03_h1_safe_summary_v1.md"
echo "三套独立测评完成：$ROOT_DIR/outputs/sim01_sim02_sim03_h1_safe_summary_v1.md"
