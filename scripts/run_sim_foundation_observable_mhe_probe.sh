#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${SIM_DATASET:-$ROOT_DIR/data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm}"
ENV_ROOT="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
PYTHON="$ENV_ROOT/bin/python"
ASSET_ROOT="${FLOW_DEPTH_ASSET_ROOT:-$ROOT_DIR/outputs/sim_cotracker3_foundation_depth_v1/flow_depth_assets}"
BASELINE_ROOT="${SIM_FOUNDATION_BASELINE_ROOT:-$ROOT_DIR/outputs/sim_foundation_depth_complete_v1}"
OUTPUT_ROOT="${1:-$ROOT_DIR/outputs/sim_foundation_observable_mhe_probe_v1}"
DEPTH_ROOT="${SIM_ESTIMATED_DEPTH_ROOT:-$DATASET/estimated_depth/foundation_stereo_rgb_v1}"
DEPTH_SUMMARY="$DEPTH_ROOT/depth_generation_summary.json"
OBSERVATION_REPORT="$ASSET_ROOT/report.json"
NODE_MANIFEST="${SIM_EVALUATION_NODE_MANIFEST:-$DATASET/evaluation/evaluation_points_30_non_grasp.json}"
BOUNDARY="${SIM_GRASP_BOUNDARY:-$DATASET/task_inputs/known_grasp_region_boundary.npz}"
METHOD_B="pbd_cotracker_foundation_depth"
METHOD_C="${SIM_METHOD_C:-pbd_cotracker_foundation_depth_global_mhe}"
STIFFNESS_UPDATE_MODE="${SIM_STIFFNESS_UPDATE_MODE:-differentiable_global_mhe}"
C_LABEL="${SIM_C_LABEL:-C-MHE-v4：B + 长期主目标/H1不确定性约束LM}"
STIFFNESS_AUTOGRAD_UNROLL_STEPS="${SIM_STIFFNESS_AUTOGRAD_UNROLL_STEPS:-4}"
STIFFNESS_AUTOGRAD_REGION_COUNT="${SIM_STIFFNESS_AUTOGRAD_REGION_COUNT:-12}"
FRAME_COUNT="$($PYTHON -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["frames"]))' "$DATASET/episode.json")"
FUTURE_START=$((FRAME_COUNT * 4 / 5))

if [[ $# -gt 1 ]]; then
    echo "用法：bash scripts/run_sim_foundation_observable_mhe_probe.sh [新输出目录]" >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "ERROR: 拒绝覆盖已有输出：$OUTPUT_ROOT" >&2
    exit 1
fi
if [[ "$FRAME_COUNT" -lt 10 || "$FUTURE_START" -le 0 ]]; then
    echo "ERROR: 数据集帧数不足以执行7:1与80/20正式协议" >&2
    exit 1
fi
for required in \
    "$PYTHON" \
    "$ASSET_ROOT/bindings.npz" \
    "$ASSET_ROOT/observations.npz" \
    "$OBSERVATION_REPORT" \
    "$DEPTH_SUMMARY" \
    "$NODE_MANIFEST" \
    "$BOUNDARY"; do
    if [[ ! -e "$required" ]]; then
        echo "ERROR: 缺少正式测评输入：$required" >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_ROOT/logs"
for capability in reconstruction_7to1 future_80to20; do
    for method in pbd "$METHOD_B"; do
        target="$OUTPUT_ROOT/$capability/$method"
        mkdir -p "$target"
        for metric_file in trajectory_metrics.json render_metrics.json; do
            source="$BASELINE_ROOT/$capability/$method/$metric_file"
            if [[ ! -f "$source" ]]; then
                echo "ERROR: 缺少A/B基线指标：$source" >&2
                exit 1
            fi
            cp "$source" "$target/$metric_file"
        done
    done
done
export SIM_DATASET="$DATASET"

run_c() {
    local gpu_id="$1"
    local capability="$2"
    local group="$OUTPUT_ROOT/$capability/$METHOD_C"
    local schedule_args=()
    local metric_args=()
    if [[ "$capability" == "reconstruction_7to1" ]]; then
        schedule_args=(
            --evaluation-holdout-stride 8
            --evaluation-holdout-offset 7
            --evaluation-render-frame-mode holdout
        )
        metric_args=(--frame-stride 8 --frame-offset 7)
    else
        schedule_args=(
            --evaluation-open-loop-start-frame "$FUTURE_START"
            --evaluation-render-frame-mode future
        )
        metric_args=(--frame-start "$FUTURE_START")
    fi
    mkdir -p "$group"
    CUDA_VISIBLE_DEVICES="$gpu_id" \
        TORCH_EXTENSIONS_DIR="/Media_HDD/jwshan/tmp/torch_extensions_foundation_mhe_gpu${gpu_id}" \
        bash "$ROOT_DIR/scripts/run_sim_reconstruction_headless.sh" \
            --evaluation-start-frame 0 \
            --evaluation-frame-count 0 \
            --evaluation-physics-steps-per-frame 2 \
            --evaluation-label "${capability}_${METHOD_C}_probe" \
            --benchmark-output "$group/artifacts" \
            "${schedule_args[@]}" \
            --visual-feedback-mode trajectory \
            --flow-depth-bindings "$ASSET_ROOT/bindings.npz" \
            --flow-depth-observations "$ASSET_ROOT/observations.npz" \
            --flow-depth-position-gain 0.70 \
            --flow-depth-velocity-gain 0.20 \
            --flow-depth-absolute-position-weight 0.85 \
            --flow-depth-solver-regularization 0.01 \
            --flow-depth-solver-iterations 24 \
            --flow-depth-robust-residual-mm 5.0 \
            --flow-depth-maximum-position-correction-mm 2.0 \
            --flow-depth-maximum-velocity-correction-m-s 0.06 \
            --online-stiffness-update \
            --sim-grasp-boundary-mode known_grasp_region \
            --stiffness-update-mode "$STIFFNESS_UPDATE_MODE" \
            --stiffness-log-learning-rate 0.03 \
            --stiffness-maximum-log-step 0.02 \
            --stiffness-signal-ema-decay 0.90 \
            --stiffness-spatial-smoothing-iterations 3 \
            --stiffness-spatial-smoothing-blend 0.35 \
            --stiffness-strain-signal-weight 0.20 \
            --stiffness-autograd-unroll-steps "$STIFFNESS_AUTOGRAD_UNROLL_STEPS" \
            --stiffness-autograd-region-count "$STIFFNESS_AUTOGRAD_REGION_COUNT" \
            --initial-paper-distance-stiffness 0.20 \
            --initial-paper-shape-stiffness 0.004

    "$PYTHON" "$ROOT_DIR/scripts/evaluate_sim_trajectory_metrics.py" \
        --reference-dataset "$DATASET" \
        --prediction "$group/artifacts/predicted_trajectories.npz" \
        --controlled-boundary "$BOUNDARY" \
        --evaluation-node-manifest "$NODE_MANIFEST" \
        "${metric_args[@]}" \
        --output "$group/trajectory_metrics.json"

    "$PYTHON" "$ROOT_DIR/scripts/evaluate_sim_rendering_metrics.py" \
        --reference-dataset "$DATASET" \
        --prediction-dir "$group/artifacts/renders" \
        "${metric_args[@]}" \
        --device cuda:0 \
        --output "$group/render_metrics.json"
}

set +e
run_c 0 reconstruction_7to1 >"$OUTPUT_ROOT/logs/reconstruction_gpu0.log" 2>&1 &
reconstruction_pid=$!
run_c 1 future_80to20 >"$OUTPUT_ROOT/logs/future_gpu1.log" 2>&1 &
future_pid=$!
wait "$reconstruction_pid"
reconstruction_status=$?
wait "$future_pid"
future_status=$?
set -e
printf 'reconstruction_status=%s\nfuture_status=%s\n' \
    "$reconstruction_status" "$future_status" >"$OUTPUT_ROOT/status.txt"
if [[ "$reconstruction_status" -ne 0 || "$future_status" -ne 0 ]]; then
    exit 3
fi

"$PYTHON" "$ROOT_DIR/scripts/summarize_sim_relative_stiffness_probe.py" \
    --evaluation-root "$OUTPUT_ROOT" \
    --method-c "$METHOD_C" \
    --label-c "$C_LABEL"
if [[ "$STIFFNESS_UPDATE_MODE" == "differentiable_global_mhe" ]]; then
    "$PYTHON" "$ROOT_DIR/scripts/summarize_sim_observable_mhe_diagnostics.py" \
        --evaluation-root "$OUTPUT_ROOT" \
        --method "$METHOD_C"
elif [[ "$STIFFNESS_UPDATE_MODE" == "particle_residual" ]]; then
    "$PYTHON" "$ROOT_DIR/scripts/summarize_sim_particle_stiffness_diagnostics.py" \
        --evaluation-root "$OUTPUT_ROOT" \
        --method "$METHOD_C"
elif [[ "$STIFFNESS_UPDATE_MODE" == "differentiable_particle_graph_lm" ]]; then
    "$PYTHON" "$ROOT_DIR/scripts/summarize_sim_particle_stiffness_diagnostics.py" \
        --evaluation-root "$OUTPUT_ROOT" \
        --method "$METHOD_C"
elif [[ "$STIFFNESS_UPDATE_MODE" == "differentiable_hierarchical_relative" ]]; then
    "$PYTHON" "$ROOT_DIR/scripts/summarize_sim_hierarchical_stiffness_diagnostics.py" \
        --evaluation-root "$OUTPUT_ROOT" \
        --method "$METHOD_C"
fi
"$PYTHON" "$ROOT_DIR/scripts/summarize_sim_foundation_depth_evaluation.py" \
    --evaluation-root "$OUTPUT_ROOT" \
    --depth-summary "$DEPTH_SUMMARY" \
    --observation-report "$OBSERVATION_REPORT" \
    --method-c "$METHOD_C" \
    --label-c "$C_LABEL"
find "$OUTPUT_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"$OUTPUT_ROOT/SHA256SUMS"
echo "完成：$OUTPUT_ROOT/comparison_complete.md"
