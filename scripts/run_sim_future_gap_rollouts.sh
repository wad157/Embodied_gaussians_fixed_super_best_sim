#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="$ROOT_DIR/outputs/two_free_support_ablation_v2/visualizations"
ROLLOUT_ROOT="$OUTPUT_ROOT/future_gap_rollouts"
MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"
INITIAL_DISTANCE=0.20
INITIAL_SHAPE=0.004
HORIZON=10

mkdir -p "$ROLLOUT_ROOT" "$MPLCONFIGDIR"
export MPLCONFIGDIR

run_rollout() {
    local task="$1"
    local dataset_name="$2"
    local frame_a="$3"
    local method="$4"
    local stiffness="$5"
    local dataset="$ROOT_DIR/data/sim/$dataset_name"
    local group="$ROLLOUT_ROOT/$task/a_$(printf '%03d' "$frame_a")/$method"
    local artifacts="$group/artifacts"
    local target_frame=$((frame_a + HORIZON))
    local frame_count=$((target_frame + 1))
    local open_loop_start=$((frame_a + 1))

    if [[ -f "$artifacts/predicted_trajectories.npz" && -f "$artifacts/artifact_metadata.json" ]]; then
        echo "[future-gap] 已完成，跳过：$task a=$frame_a $method"
        return
    fi
    if [[ -e "$group" ]]; then
        echo "ERROR: 拒绝覆盖不完整的 rollout：$group" >&2
        exit 1
    fi
    mkdir -p "$group"
    local stiffness_flag="--no-online-stiffness-update"
    if [[ "$stiffness" == "on" ]]; then
        stiffness_flag="--online-stiffness-update"
    fi
    echo "[future-gap] 开始：$task a=$frame_a b=$HORIZON $method；从帧 $open_loop_start 禁止未来 RGB"
    SIM_DATASET="$dataset" \
        bash "$ROOT_DIR/scripts/run_sim_reconstruction_headless.sh" \
        --evaluation-start-frame 0 \
        --evaluation-frame-count "$frame_count" \
        --evaluation-physics-steps-per-frame 3 \
        --evaluation-open-loop-start-frame "$open_loop_start" \
        --evaluation-label "future_gap_${task}_a${frame_a}_${method}" \
        --benchmark-output "$artifacts" \
        --no-evaluation-render-images \
        --visual-feedback-mode residual \
        "$stiffness_flag" \
        --stiffness-log-learning-rate 0.30 \
        --stiffness-maximum-log-step 0.10 \
        --stiffness-signal-ema-decay 0.60 \
        --stiffness-spatial-smoothing-iterations 2 \
        --stiffness-spatial-smoothing-blend 0.30 \
        --initial-paper-distance-stiffness "$INITIAL_DISTANCE" \
        --initial-paper-shape-stiffness "$INITIAL_SHAPE"
}

for task_spec in \
    "front_pull:tissue_retraction_free_support_front_v2" \
    "side_pull:tissue_retraction_free_support_side_v2"; do
    IFS=: read -r task dataset_name <<<"$task_spec"
    for frame_a in 170 220; do
        run_rollout "$task" "$dataset_name" "$frame_a" \
            pbd_visual_residual off
        run_rollout "$task" "$dataset_name" "$frame_a" \
            pbd_visual_residual_stiffness on
    done
done

"/Media_HDD/jwshan/conda_envs/eg_codex/bin/python" \
    "$ROOT_DIR/scripts/visualize_sim_future_gaps.py" \
    --data-root "$ROOT_DIR/data/sim" \
    --evaluation-root "$ROOT_DIR/outputs/two_free_support_ablation_v2" \
    --output "$OUTPUT_ROOT"

echo "[future-gap] 所有 rollout 和图片已完成：$OUTPUT_ROOT"
