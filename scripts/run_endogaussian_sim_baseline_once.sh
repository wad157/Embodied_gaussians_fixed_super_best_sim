#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EG_PY="$ROOT_DIR/scripts/run_endogaussian_baseline_python.sh"
EVAL_PY="${EVAL_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
GPU_ID="${SIM_GPU_ID:-0}"
CONFIG="$ROOT_DIR/baselines/endogaussian_sim/configs/sim_unified.py"

if [[ $# -ne 4 ]]; then
    echo "用法：bash scripts/run_endogaussian_sim_baseline_once.sh <sim01|sim02|sim03> <repeat_01|repeat_02|repeat_03> <seed> <输出目录>" >&2
    exit 2
fi
DATASET_KEY="$1"
REPEAT_ID="$2"
SEED="$3"
OUTPUT_ROOT="$(realpath -m "$4")"
case "$DATASET_KEY" in
    sim01) DATASET="$ROOT_DIR/data/sim/tissue_retraction_free_support_front_v2"; FRAMES=300; DATASET_INDEX=1 ;;
    sim02) DATASET="$ROOT_DIR/data/sim/tissue_retraction_free_support_side_v2"; FRAMES=300; DATASET_INDEX=2 ;;
    sim03) DATASET="$ROOT_DIR/data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm"; FRAMES=360; DATASET_INDEX=3 ;;
    *) echo "ERROR: 未知数据集：$DATASET_KEY" >&2; exit 2 ;;
esac
if [[ ! "$SEED" =~ ^[0-9]+$ ]] || [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: seed 与 SIM_GPU_ID 必须是非负整数" >&2
    exit 2
fi
PORT="${ENDOGAUSSIAN_PORT:-$((6200 + SEED * 10 + DATASET_INDEX))}"
if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1024 || PORT > 65535 )); then
    echo "ERROR: ENDOGAUSSIAN_PORT 必须在 1024..65535" >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "ERROR: 拒绝覆盖已有 baseline 输出：$OUTPUT_ROOT" >&2
    exit 1
fi
for required in "$EG_PY" "$EVAL_PY" "$CONFIG" "$DATASET/episode.json"; do
    if [[ ! -e "$required" ]]; then
        echo "ERROR: 缺少输入：$required" >&2
        exit 1
    fi
done

FUTURE_START=$((FRAMES * 4 / 5))
MANIFEST="$DATASET/evaluation/evaluation_points_30_non_grasp.json"
BOUNDARY="$DATASET/task_inputs/known_grasp_region_boundary.npz"
MODEL_PATH="$OUTPUT_ROOT/model"
ARTIFACTS="$OUTPUT_ROOT/artifacts"
LOG="$OUTPUT_ROOT/run.log"
mkdir -p "$OUTPUT_ROOT"
printf '%s\n' \
    'status=running' \
    "dataset=$DATASET_KEY" \
    "repeat=$REPEAT_ID" \
    "seed=$SEED" \
    "gpu=$GPU_ID" \
    "port=$PORT" \
    'method=endogaussian_pinned_som_query_anchored_displacement_v2' \
    'protocol=offline_prefix_reconstruction_and_zero_shot_future_extrapolation' \
    >"$OUTPUT_ROOT/status.txt"

"$EVAL_PY" "$ROOT_DIR/scripts/audit_endogaussian_sim_protocol.py" \
    --dataset-key "$DATASET_KEY" \
    --dataset "$DATASET" \
    --output "$OUTPUT_ROOT/protocol_audit.json" \
    >"$LOG" 2>&1

CUDA_VISIBLE_DEVICES="$GPU_ID" bash "$EG_PY" \
    "$ROOT_DIR/baselines/endogaussian_sim/train_sim.py" \
    --dataset-key "$DATASET_KEY" \
    --seed "$SEED" \
    --source_path "$DATASET" \
    --model_path "$MODEL_PATH" \
    --configs "$CONFIG" \
    --port "$PORT" \
    >>"$LOG" 2>&1

CUDA_VISIBLE_DEVICES="$GPU_ID" bash "$EG_PY" \
    "$ROOT_DIR/baselines/endogaussian_sim/export_sim.py" \
    --dataset-key "$DATASET_KEY" \
    --source_path "$DATASET" \
    --model_path "$MODEL_PATH" \
    --output-dir "$ARTIFACTS" \
    --configs "$CONFIG" \
    >>"$LOG" 2>&1

for CAPABILITY in reconstruction_7to1 future_80to20; do
    METRICS="$OUTPUT_ROOT/metrics/$CAPABILITY"
    mkdir -p "$METRICS"
    if [[ "$CAPABILITY" == reconstruction_7to1 ]]; then
        SELECTION=(--frame-start 0 --frame-end-exclusive "$FUTURE_START" --frame-stride 8 --frame-offset 7)
    else
        SELECTION=(--frame-start "$FUTURE_START" --frame-end-exclusive "$FRAMES")
    fi
    "$EVAL_PY" "$ROOT_DIR/scripts/evaluate_sim_trajectory_metrics.py" \
        --reference-dataset "$DATASET" \
        --prediction "$ARTIFACTS/predicted_trajectories.npz" \
        --controlled-boundary "$BOUNDARY" \
        --evaluation-node-manifest "$MANIFEST" \
        "${SELECTION[@]}" \
        --output "$METRICS/trajectory_metrics.json" \
        >>"$LOG" 2>&1
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$EVAL_PY" \
        "$ROOT_DIR/scripts/evaluate_sim_rendering_metrics.py" \
        --reference-dataset "$DATASET" \
        --prediction-dir "$ARTIFACTS" \
        "${SELECTION[@]}" \
        --device cuda:0 \
        --output "$METRICS/render_metrics.json" \
        >>"$LOG" 2>&1
done

printf '%s\n' \
    'status=complete' \
    "dataset=$DATASET_KEY" \
    "repeat=$REPEAT_ID" \
    "seed=$SEED" \
    "gpu=$GPU_ID" \
    "port=$PORT" \
    'method=endogaussian_pinned_som_query_anchored_displacement_v2' \
    'protocol=offline_prefix_reconstruction_and_zero_shot_future_extrapolation' \
    >"$OUTPUT_ROOT/status.txt"
find "$OUTPUT_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"$OUTPUT_ROOT/SHA256SUMS"
echo "完成：$OUTPUT_ROOT"
