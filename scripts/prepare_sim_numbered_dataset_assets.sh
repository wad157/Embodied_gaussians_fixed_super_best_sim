#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
DATASET="${SIM_DATASET:?必须设置SIM_DATASET}"
ASSET_ROOT="${1:?必须给出独立资产目录}"
DEPTH_ROOT="${SIM_ESTIMATED_DEPTH_ROOT:?必须设置SIM_ESTIMATED_DEPTH_ROOT}"
GPU_ID="${SIM_GPU_ID:?必须设置SIM_GPU_ID}"
RECTIFICATION_ALPHA="${SIM_RECTIFICATION_ALPHA:-0.85}"
DEPTH_SOURCE_LABEL="${SIM_DEPTH_SOURCE_LABEL:-foundation_stereo_rgb_rig_aware_v1}"
TRACK_ROOT="$ASSET_ROOT/tracks"
FLOW_ROOT="$ASSET_ROOT/flow_depth_assets"
FRAME_COUNT="$($PYTHON -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["frames"]))' "$DATASET/episode.json")"

if [[ $# -ne 1 ]]; then
    echo "用法：设置SIM_DATASET/SIM_ESTIMATED_DEPTH_ROOT/SIM_GPU_ID后，bash scripts/prepare_sim_numbered_dataset_assets.sh <独立资产目录>" >&2
    exit 2
fi
for required in \
    "$DATASET/videos/stereo_left.mp4" \
    "$DATASET/ground_truth/masks/tissue/stereo_left/000000.png" \
    "$DATASET/gui_assets/visual_force_masks/stereo_left/tissue_masks_packbits.npy" \
    "$DATASET/gui_assets/tissue_fixedsuperbest.npz" \
    "$DATASET/task_inputs/known_grasp_region_boundary.npz" \
    "$DATASET/evaluation/evaluation_points_30_non_grasp.json"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: 数据集独立资产缺失：$required" >&2
        exit 1
    fi
done
mkdir -p "$ASSET_ROOT/logs"

if [[ ! -f "$TRACK_ROOT/tracks.npz" ]]; then
    "$PYTHON" "$ROOT_DIR/scripts/track_sim_tissue_cotracker3.py" \
        --video "$DATASET/videos/stereo_left.mp4" \
        --tissue-mask "$DATASET/ground_truth/masks/tissue/stereo_left/000000.png" \
        --dynamic-tissue-masks "$DATASET/gui_assets/visual_force_masks/stereo_left/tissue_masks_packbits.npy" \
        --output-dir "$TRACK_ROOT" \
        --device "cuda:$GPU_ID" \
        --grid-size 24 \
        --frame-stride 1 \
        --resize-width 512 \
        >"$ASSET_ROOT/logs/cotracker_gpu${GPU_ID}.log" 2>&1
fi

if [[ ! -f "$DEPTH_ROOT/depth_generation_summary.json" ]]; then
    "$PYTHON" "$ROOT_DIR/scripts/generate_sim_rgb_stereo_depth.py" \
        --dataset "$DATASET" \
        --output-dir "$DEPTH_ROOT" \
        --frames all \
        --device "cuda:$GPU_ID" \
        --rectification-alpha "$RECTIFICATION_ALPHA" \
        >"$ASSET_ROOT/logs/foundation_depth_gpu${GPU_ID}.log" 2>&1
fi

for camera in stereo_left stereo_right; do
    count=$(find "$DEPTH_ROOT/$camera" -maxdepth 1 -type f -name '*-depth.npy' | wc -l)
    if [[ "$count" -ne "$FRAME_COUNT" ]]; then
        echo "ERROR: $camera深度帧数为$count，期望$FRAME_COUNT" >&2
        exit 3
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
    "$TRACK_ROOT/tracks.npz" \
    "$DEPTH_ROOT/depth_generation_summary.json" \
    "$FLOW_ROOT/bindings.npz" \
    "$FLOW_ROOT/observations.npz" \
    "$FLOW_ROOT/report.json"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: 独立观测资产不完整：$required" >&2
        exit 4
    fi
done
printf 'status=complete\ndataset=%s\nframes=%s\ngpu=%s\n' \
    "$DATASET" "$FRAME_COUNT" "$GPU_ID" >"$ASSET_ROOT/status.txt"
echo "独立资产完成：$ASSET_ROOT"
