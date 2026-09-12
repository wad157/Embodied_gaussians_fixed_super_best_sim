#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EG_PY="$ROOT_DIR/scripts/run_endogaussian_baseline_python.sh"
EVAL_PY="${EVAL_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
GPU_ID="${SIM_GPU_ID:-0}"
CONFIG="$ROOT_DIR/baselines/endogaussian_sim/configs/sim_unified.py"

if [[ $# -ne 6 ]]; then
    echo "用法：bash scripts/reexport_endogaussian_sim_baseline_once.sh <sim01|sim02|sim03> <repeat_id> <seed> <旧运行目录> <新运行目录> <说明>" >&2
    exit 2
fi
DATASET_KEY="$1"
REPEAT_ID="$2"
SEED="$3"
OLD_RUN="$(realpath "$4")"
NEW_RUN="$(realpath -m "$5")"
NOTE="$6"
case "$DATASET_KEY" in
    sim01) DATASET="$ROOT_DIR/data/sim/tissue_retraction_free_support_front_v2"; FRAMES=300 ;;
    sim02) DATASET="$ROOT_DIR/data/sim/tissue_retraction_free_support_side_v2"; FRAMES=300 ;;
    sim03) DATASET="$ROOT_DIR/data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm"; FRAMES=360 ;;
    *) echo "ERROR: 未知数据集：$DATASET_KEY" >&2; exit 2 ;;
esac
if [[ -e "$NEW_RUN" ]]; then
    echo "ERROR: 拒绝覆盖新运行目录：$NEW_RUN" >&2
    exit 1
fi
for expected in \
    "status=complete" \
    "dataset=$DATASET_KEY" \
    "repeat=$REPEAT_ID" \
    "seed=$SEED"; do
    if ! grep -qx "$expected" "$OLD_RUN/status.txt"; then
        echo "ERROR: 旧运行身份或完成状态不匹配：$expected" >&2
        exit 1
    fi
done
for required in \
    "$OLD_RUN/model/point_cloud/iteration_3000/point_cloud.ply" \
    "$OLD_RUN/artifacts/provenance.json" \
    "$DATASET/episode.json"; do
    if [[ ! -e "$required" ]]; then
        echo "ERROR: 缺少复用输入：$required" >&2
        exit 1
    fi
done

FUTURE_START=$((FRAMES * 4 / 5))
MANIFEST="$DATASET/evaluation/evaluation_points_30_non_grasp.json"
BOUNDARY="$DATASET/task_inputs/known_grasp_region_boundary.npz"
MODEL_PATH="$NEW_RUN/model"
ARTIFACTS="$NEW_RUN/artifacts"
LOG="$NEW_RUN/run.log"
mkdir -p "$NEW_RUN"
printf '%s\n' \
    'status=running' \
    "dataset=$DATASET_KEY" \
    "repeat=$REPEAT_ID" \
    "seed=$SEED" \
    "gpu=$GPU_ID" \
    'method=endogaussian_pinned_som_query_anchored_displacement_v2' \
    "training_checkpoint_reused_from=$OLD_RUN" \
    "note=$NOTE" \
    >"$NEW_RUN/status.txt"

"$EVAL_PY" "$ROOT_DIR/scripts/audit_endogaussian_sim_protocol.py" \
    --dataset-key "$DATASET_KEY" \
    --dataset "$DATASET" \
    --output "$NEW_RUN/protocol_audit.json" \
    >"$LOG" 2>&1

# The decoder change cannot affect the frozen checkpoint or RGB/alpha renders.
# Hard links keep this corrected run self-contained without duplicating large files.
cp -al "$OLD_RUN/model" "$MODEL_PATH"
CUDA_VISIBLE_DEVICES="$GPU_ID" bash "$EG_PY" \
    "$ROOT_DIR/baselines/endogaussian_sim/export_sim.py" \
    --dataset-key "$DATASET_KEY" \
    --source_path "$DATASET" \
    --model_path "$MODEL_PATH" \
    --output-dir "$ARTIFACTS" \
    --reuse-renders-from "$OLD_RUN/artifacts" \
    --configs "$CONFIG" \
    >>"$LOG" 2>&1

for CAPABILITY in reconstruction_7to1 future_80to20; do
    METRICS="$NEW_RUN/metrics/$CAPABILITY"
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
    'method=endogaussian_pinned_som_query_anchored_displacement_v2' \
    "training_checkpoint_reused_from=$OLD_RUN" \
    "note=$NOTE" \
    >"$NEW_RUN/status.txt"
find "$NEW_RUN" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"$NEW_RUN/SHA256SUMS"
echo "完成：$NEW_RUN"
