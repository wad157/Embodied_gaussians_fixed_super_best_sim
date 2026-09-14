#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHYSTWIN_PY="$ROOT_DIR/scripts/run_phystwin_baseline_python.sh"
EVAL_PY="${EVAL_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
GPU_ID="${SIM_GPU_ID:-0}"

if [[ $# -ne 4 ]]; then
    echo "用法：bash scripts/run_phystwin_sim_baseline_once.sh <sim01|sim02|sim03> <repeat_01|repeat_02|repeat_03> <seed> <输出目录>" >&2
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
for required in "$PHYSTWIN_PY" "$EVAL_PY" "$DATASET/episode.json"; do
    if [[ ! -e "$required" ]]; then
        echo "ERROR: 缺少输入：$required" >&2
        exit 1
    fi
done

FUTURE_START=$((FRAMES * 4 / 5))
MANIFEST="$DATASET/evaluation/evaluation_points_30_non_grasp.json"
BOUNDARY="$DATASET/task_inputs/known_grasp_region_boundary.npz"
PREPROCESS="$OUTPUT_ROOT/preprocess"
PHYSICS="$OUTPUT_ROOT/physics"
APPEARANCE="$OUTPUT_ROOT/appearance"
ARTIFACTS="$OUTPUT_ROOT/artifacts"
LOG="$OUTPUT_ROOT/run.log"
mkdir -p "$OUTPUT_ROOT"
printf '%s\n' \
    'status=running' \
    "dataset=$DATASET_KEY" \
    "repeat=$REPEAT_ID" \
    "seed=$SEED" \
    "gpu=$GPU_ID" \
    'method=phystwin_native_particles_upstream_lbs_v1' \
    'protocol=offline_prefix_reconstruction_and_known_control_future_rollout' \
    >"$OUTPUT_ROOT/status.txt"

"$EVAL_PY" "$ROOT_DIR/scripts/audit_phystwin_sim_protocol.py" \
    --dataset-key "$DATASET_KEY" \
    --dataset "$DATASET" \
    --output "$OUTPUT_ROOT/protocol_audit.json" \
    >"$LOG" 2>&1

CUDA_VISIBLE_DEVICES="$GPU_ID" bash "$PHYSTWIN_PY" \
    "$ROOT_DIR/baselines/phystwin_sim/prepare_sim.py" \
    --dataset-key "$DATASET_KEY" \
    --dataset "$DATASET" \
    --output-dir "$PREPROCESS" \
    --seed "$SEED" \
    --device cuda:0 \
    >>"$LOG" 2>&1

CUDA_VISIBLE_DEVICES="$GPU_ID" bash "$PHYSTWIN_PY" \
    "$ROOT_DIR/baselines/phystwin_sim/train_appearance.py" \
    --dataset-key "$DATASET_KEY" \
    --dataset "$DATASET" \
    --preprocess-dir "$PREPROCESS" \
    --output-dir "$APPEARANCE" \
    --seed "$SEED" \
    --device cuda:0 \
    --iterations 1000 \
    --downsample 2 \
    >>"$LOG" 2>&1

CUDA_VISIBLE_DEVICES="$GPU_ID" bash "$PHYSTWIN_PY" \
    "$ROOT_DIR/baselines/phystwin_sim/run_physics.py" \
    --dataset-key "$DATASET_KEY" \
    --preprocess-dir "$PREPROCESS" \
    --output-dir "$PHYSICS" \
    --seed "$SEED" \
    --device cuda:0 \
    --cma-iterations 20 \
    --adam-iterations 200 \
    --checkpoint-interval 20 \
    --substeps 667 \
    >>"$LOG" 2>&1

CUDA_VISIBLE_DEVICES="$GPU_ID" bash "$PHYSTWIN_PY" \
    "$ROOT_DIR/baselines/phystwin_sim/export_sim.py" \
    --dataset-key "$DATASET_KEY" \
    --dataset "$DATASET" \
    --physics-dir "$PHYSICS" \
    --appearance-dir "$APPEARANCE" \
    --output-dir "$ARTIFACTS" \
    --device cuda:0 \
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
    'method=phystwin_native_particles_upstream_lbs_v1' \
    'protocol=offline_prefix_reconstruction_and_known_control_future_rollout' \
    >"$OUTPUT_ROOT/status.txt"
find "$OUTPUT_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"$OUTPUT_ROOT/SHA256SUMS"
echo "完成：$OUTPUT_ROOT"
