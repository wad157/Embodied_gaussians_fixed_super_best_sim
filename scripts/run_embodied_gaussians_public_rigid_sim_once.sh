#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EG_PY="${EG_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
GPU_ID="${SIM_GPU_ID:-0}"

if [[ $# -ne 4 ]]; then
    echo "用法：bash scripts/run_embodied_gaussians_public_rigid_sim_once.sh <sim01|sim02|sim03> <repeat_01|repeat_02|repeat_03> <seed> <输出目录>" >&2
    exit 2
fi
DATASET_KEY="$1"
REPEAT_ID="$2"
SEED="$3"
OUTPUT_ROOT="$(realpath -m "$4")"
case "$DATASET_KEY" in
    sim01) DATASET="$ROOT_DIR/data/sim/tissue_retraction_free_support_front_v2"; FRAMES=300 ;;
    sim02) DATASET="$ROOT_DIR/data/sim/tissue_retraction_free_support_side_v2"; FRAMES=300 ;;
    sim03) DATASET="$ROOT_DIR/data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm"; FRAMES=360 ;;
    *) echo "ERROR: 未知数据集：$DATASET_KEY" >&2; exit 2 ;;
esac
if [[ ! "$SEED" =~ ^[0-9]+$ ]] || [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: seed 与 SIM_GPU_ID 必须是非负整数" >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "ERROR: 拒绝覆盖已有 baseline 输出：$OUTPUT_ROOT" >&2
    exit 1
fi

export PYTHONNOUSERSITE=1
export TMPDIR="${TMPDIR:-$ROOT_DIR/.tmp/embodied_gaussians_baseline}"
mkdir -p "$TMPDIR" "$OUTPUT_ROOT/initialization"
BODY="$OUTPUT_ROOT/initialization/body.json"
ARTIFACTS="$OUTPUT_ROOT/artifacts"
LOG="$OUTPUT_ROOT/run.log"
FUTURE_START=$((FRAMES * 4 / 5))
MANIFEST="$DATASET/evaluation/evaluation_points_30_non_grasp.json"
BOUNDARY="$DATASET/task_inputs/known_grasp_region_boundary.npz"

printf '%s\n' \
    'status=running' \
    "dataset=$DATASET_KEY" \
    "repeat=$REPEAT_ID" \
    "seed=$SEED" \
    "gpu=$GPU_ID" \
    'actuation=psm-fk-collision-only' \
    'method=embodied_gaussians_public_rigid_only' \
    >"$OUTPUT_ROOT/status.txt"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$EG_PY" \
    "$ROOT_DIR/baselines/embodied_gaussians_sim/initialize.py" \
    --dataset-key "$DATASET_KEY" --dataset "$DATASET" --seed "$SEED" --output "$BODY" \
    >"$LOG" 2>&1

"$EG_PY" "$ROOT_DIR/scripts/audit_embodied_gaussians_sim_protocol.py" \
    --dataset-key "$DATASET_KEY" --dataset "$DATASET" --body "$BODY" \
    --variant public-rigid --actuation psm-fk-collision-only \
    --output "$OUTPUT_ROOT/protocol_audit.json" >>"$LOG" 2>&1

CUDA_VISIBLE_DEVICES="$GPU_ID" "$EG_PY" \
    "$ROOT_DIR/baselines/embodied_gaussians_sim/run_public_rigid.py" \
    --dataset-key "$DATASET_KEY" --dataset "$DATASET" --body "$BODY" \
    --output-dir "$ARTIFACTS" >>"$LOG" 2>&1

CUDA_VISIBLE_DEVICES="$GPU_ID" "$EG_PY" \
    "$ROOT_DIR/baselines/embodied_gaussians_sim/export_tracks.py" \
    --dataset-key "$DATASET_KEY" --dataset "$DATASET" --body "$BODY" \
    --rollout-dir "$ARTIFACTS" --output "$ARTIFACTS/predicted_trajectories.npz" \
    >>"$LOG" 2>&1

for CAPABILITY in reconstruction_7to1 future_80to20; do
    METRICS="$OUTPUT_ROOT/metrics/$CAPABILITY"
    mkdir -p "$METRICS"
    if [[ "$CAPABILITY" == reconstruction_7to1 ]]; then
        SELECTION=(--frame-start 0 --frame-end-exclusive "$FUTURE_START" --frame-stride 8 --frame-offset 7)
    else
        SELECTION=(--frame-start "$FUTURE_START" --frame-end-exclusive "$FRAMES")
    fi
    "$EG_PY" "$ROOT_DIR/scripts/evaluate_sim_trajectory_metrics.py" \
        --reference-dataset "$DATASET" --prediction "$ARTIFACTS/predicted_trajectories.npz" \
        --controlled-boundary "$BOUNDARY" --evaluation-node-manifest "$MANIFEST" \
        "${SELECTION[@]}" --output "$METRICS/trajectory_metrics.json" >>"$LOG" 2>&1
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$EG_PY" "$ROOT_DIR/scripts/evaluate_sim_rendering_metrics.py" \
        --reference-dataset "$DATASET" --prediction-dir "$ARTIFACTS" \
        "${SELECTION[@]}" --device cuda:0 \
        --output "$METRICS/render_metrics.json" >>"$LOG" 2>&1
done

printf '%s\n' \
    'status=complete' \
    "dataset=$DATASET_KEY" \
    "repeat=$REPEAT_ID" \
    "seed=$SEED" \
    "gpu=$GPU_ID" \
    'actuation=psm-fk-collision-only' \
    'method=embodied_gaussians_public_rigid_only' \
    >"$OUTPUT_ROOT/status.txt"
find "$OUTPUT_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"$OUTPUT_ROOT/SHA256SUMS"
echo "完成：$OUTPUT_ROOT"
