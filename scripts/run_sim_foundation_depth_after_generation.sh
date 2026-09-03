#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${SIM_DATASET:-$ROOT_DIR/data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm}"
DEPTH_ROOT="${SIM_ESTIMATED_DEPTH_ROOT:-$DATASET/estimated_depth/foundation_stereo_rgb_v1}"
SUMMARY="$DEPTH_ROOT/depth_generation_summary.json"
OUTPUT_ROOT="${1:-$ROOT_DIR/outputs/sim_foundation_depth_complete_v1}"
WAIT_SECONDS="${FOUNDATION_DEPTH_WAIT_SECONDS:-10800}"
deadline=$((SECONDS + WAIT_SECONDS))

echo "等待FoundationStereo深度生成完成：$SUMMARY"
while [[ ! -f "$SUMMARY" ]]; do
    if (( SECONDS >= deadline )); then
        echo "ERROR: 等待深度生成超时（${WAIT_SECONDS}s）" >&2
        exit 4
    fi
    sleep 30
done

for camera in stereo_left stereo_right; do
    count=$(find "$DEPTH_ROOT/$camera" -maxdepth 1 -type f -name '*-depth.npy' | wc -l)
    if [[ "$count" -ne 360 ]]; then
        echo "ERROR: $camera 深度帧数为$count，期望360" >&2
        exit 5
    fi
done
echo "深度完整性通过，启动完整测评：$OUTPUT_ROOT"
exec bash "$ROOT_DIR/scripts/run_sim_foundation_depth_complete_evaluation.sh" "$OUTPUT_ROOT"
