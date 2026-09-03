#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
RUN_ROOT="${1:-$ROOT_DIR/outputs/sim01_sim02_particle_graph_lm_complete_v1}"

SIM01="$ROOT_DIR/data/sim/tissue_retraction_free_support_front_v2"
SIM02="$ROOT_DIR/data/sim/tissue_retraction_free_support_side_v2"
SIM01_ASSETS="$ROOT_DIR/outputs/sim01_planar_x_cotracker_foundation_v1/flow_depth_assets"
SIM02_ASSETS="$ROOT_DIR/outputs/sim02_planar_y_cotracker_foundation_v1/flow_depth_assets"
SIM01_DEPTH="$SIM01/estimated_depth/foundation_stereo_rgb_rig_aware_v1"
SIM02_DEPTH="$SIM02/estimated_depth/foundation_stereo_rgb_rig_aware_v1"
SIM01_BASELINE="$ROOT_DIR/outputs/sim_three_datasets_previous_c_complete_v2/sim01"
SIM02_BASELINE="$ROOT_DIR/outputs/sim_three_datasets_previous_c_complete_v2/sim02"
SIM03_RESULT="$ROOT_DIR/outputs/sim_foundation_particle_graph_lm_complete_v2_fast/comparison_complete.json"

if [[ $# -gt 1 ]]; then
    echo "用法：bash scripts/run_sim01_sim02_particle_graph_lm_complete.sh [新输出目录]" >&2
    exit 2
fi
if [[ -e "$RUN_ROOT" ]]; then
    echo "ERROR: 拒绝覆盖已有输出：$RUN_ROOT" >&2
    exit 1
fi
mkdir -p "$RUN_ROOT/logs"
printf 'status=running\nalgorithm=particle_graph_lm_v1_frozen\n' >"$RUN_ROOT/status.txt"

run_dataset() {
    local dataset_id="$1"
    local dataset="$2"
    local assets="$3"
    local depth="$4"
    local baseline="$5"
    local output="$RUN_ROOT/$dataset_id"

    SIM_DATASET="$dataset" \
    SIM_ESTIMATED_DEPTH_ROOT="$depth" \
    FLOW_DEPTH_ASSET_ROOT="$assets" \
    SIM_FOUNDATION_BASELINE_ROOT="$baseline" \
        bash "$ROOT_DIR/scripts/run_sim_foundation_particle_graph_lm_complete.sh" \
        "$output" >"$RUN_ROOT/logs/${dataset_id}.log" 2>&1
}

# Each dataset internally uses GPU0 for reconstruction and GPU1 for future.
# Run the two datasets sequentially so they never share observations or GPUs.
run_dataset sim01 "$SIM01" "$SIM01_ASSETS" "$SIM01_DEPTH" "$SIM01_BASELINE"
run_dataset sim02 "$SIM02" "$SIM02_ASSETS" "$SIM02_DEPTH" "$SIM02_BASELINE"

"$PYTHON" "$ROOT_DIR/scripts/summarize_sim_particle_graph_three_datasets.py" \
    --sim01 "$RUN_ROOT/sim01/comparison_complete.json" \
    --sim02 "$RUN_ROOT/sim02/comparison_complete.json" \
    --sim03 "$SIM03_RESULT" \
    --output "$RUN_ROOT/comparison_all.md"

printf 'status=complete\nalgorithm=particle_graph_lm_v1_frozen\n' \
    >"$RUN_ROOT/status.txt"
find "$RUN_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"$RUN_ROOT/SHA256SUMS"
echo "数据集1/2测评完成：$RUN_ROOT/comparison_all.md"

