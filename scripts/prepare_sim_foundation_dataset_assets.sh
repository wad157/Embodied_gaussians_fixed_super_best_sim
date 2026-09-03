#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
DATASET="${SIM_DATASET:-$ROOT_DIR/data/sim/tissue_retraction_closeup_inplane_full_v1}"
ASSET_RUN_ROOT="${1:-$ROOT_DIR/outputs/sim_inplane_cotracker3_foundation_depth_v1}"
DEPTH_ROOT="${SIM_ESTIMATED_DEPTH_ROOT:-$DATASET/estimated_depth/foundation_stereo_rgb_v1}"
DEPTH_SOURCE_LABEL="${SIM_DEPTH_SOURCE_LABEL:-foundation_stereo_rgb_rig_aware_v2}"
RECTIFICATION_ALPHA="${SIM_RECTIFICATION_ALPHA:-0.0}"
TRACK_ROOT="$ASSET_RUN_ROOT/tracks"
FLOW_ROOT="$ASSET_RUN_ROOT/flow_depth_assets"
FRAME_COUNT="$($PYTHON -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["frames"]))' "$DATASET/episode.json")"

if [[ $# -gt 1 ]]; then
    echo "用法：SIM_DATASET=<数据集> bash scripts/prepare_sim_foundation_dataset_assets.sh [资产输出目录]" >&2
    exit 2
fi
for required in \
    "$DATASET/videos/stereo_left.mp4" \
    "$DATASET/ground_truth/masks/tissue/stereo_left/000000.png" \
    "$DATASET/gui_assets/visual_force_masks/stereo_left/tissue_masks_packbits.npy" \
    "$DATASET/gui_assets/tissue_fixedsuperbest.npz"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: 缺少资产生成输入：$required" >&2
        exit 1
    fi
done

mkdir -p "$ASSET_RUN_ROOT/logs"
depth_status=0
track_status=0
if [[ ! -f "$DEPTH_ROOT/depth_generation_summary.json" ]]; then
    "$PYTHON" "$ROOT_DIR/scripts/generate_sim_rgb_stereo_depth.py" \
        --dataset "$DATASET" \
        --output-dir "$DEPTH_ROOT" \
        --frames all \
        --device cuda:0 \
        --rectification-alpha "$RECTIFICATION_ALPHA" \
        >"$ASSET_RUN_ROOT/logs/foundation_depth_gpu0.log" 2>&1 &
    depth_pid=$!
else
    depth_pid=""
fi
if [[ ! -f "$TRACK_ROOT/tracks.npz" ]]; then
    "$PYTHON" "$ROOT_DIR/scripts/track_sim_tissue_cotracker3.py" \
        --video "$DATASET/videos/stereo_left.mp4" \
        --tissue-mask "$DATASET/ground_truth/masks/tissue/stereo_left/000000.png" \
        --dynamic-tissue-masks "$DATASET/gui_assets/visual_force_masks/stereo_left/tissue_masks_packbits.npy" \
        --output-dir "$TRACK_ROOT" \
        --device cuda:1 \
        --grid-size 24 \
        --frame-stride 1 \
        --resize-width 512 \
        >"$ASSET_RUN_ROOT/logs/cotracker_gpu1.log" 2>&1 &
    track_pid=$!
else
    track_pid=""
fi

set +e
if [[ -n "$depth_pid" ]]; then
    wait "$depth_pid"
    depth_status=$?
fi
if [[ -n "$track_pid" ]]; then
    wait "$track_pid"
    track_status=$?
fi
set -e
printf 'depth_status=%s\ntrack_status=%s\n' "$depth_status" "$track_status" \
    >"$ASSET_RUN_ROOT/preparation_status.txt"
if [[ "$depth_status" -ne 0 || "$track_status" -ne 0 ]]; then
    exit 3
fi

for camera in stereo_left stereo_right; do
    count=$(find "$DEPTH_ROOT/$camera" -maxdepth 1 -type f -name '*-depth.npy' | wc -l)
    if [[ "$count" -ne "$FRAME_COUNT" ]]; then
        echo "ERROR: $camera RGB深度帧数为$count，期望$FRAME_COUNT" >&2
        exit 4
    fi
done
if [[ ! -f "$FLOW_ROOT/report.json" ]]; then
    "$PYTHON" "$ROOT_DIR/scripts/prepare_sim_cotracker_gt_depth_assets.py" \
        --dataset "$DATASET" \
        --tracks "$TRACK_ROOT/tracks.npz" \
        --output-dir "$FLOW_ROOT" \
        --camera stereo_left \
        --depth-dir "$DEPTH_ROOT/stereo_left" \
        --depth-filename-pattern '{frame:06d}-depth.npy' \
        --depth-source-label "$DEPTH_SOURCE_LABEL"
fi
for required in \
    "$FLOW_ROOT/bindings.npz" \
    "$FLOW_ROOT/observations.npz" \
    "$FLOW_ROOT/report.json" \
    "$DEPTH_ROOT/depth_generation_summary.json"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: 迁移资产不完整：$required" >&2
        exit 5
    fi
done
echo "完成：$ASSET_RUN_ROOT"
