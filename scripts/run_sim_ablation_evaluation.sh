#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${SIM_DATASET:-$ROOT_DIR/data/sim/tissue_retraction_closeup_inplane_full_v1}"
PYTHON="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
DEFAULT_OUTPUT="$ROOT_DIR/outputs/sim_ablation_$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${1:-${SIM_ABLATION_OUTPUT_ROOT:-$DEFAULT_OUTPUT}}"
FRAME_COUNT="$($PYTHON -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["frames"]))' "$DATASET/episode.json")"
if [[ ! "$FRAME_COUNT" =~ ^[0-9]+$ ]] || (( FRAME_COUNT < 5 )); then
    echo "ERROR: 数据集帧数非法：$FRAME_COUNT" >&2
    exit 1
fi
SPLIT_FRAME="${SIM_SPLIT_FRAME_INDEX:-$((FRAME_COUNT * 4 / 5))}"
EVALUATION_NODE_MANIFEST="${SIM_EVALUATION_NODE_MANIFEST:-}"
# 正式消融禁止根据当前数据集选择初始材料。三组都从项目预先约定的
# 全局均匀软组织先验开始；只有第三组能通过 RGB 残差在线更新局部刚度。
INITIAL_DISTANCE=0.20
INITIAL_SHAPE=0.004

if [[ $# -gt 1 ]]; then
    echo "用法：bash scripts/run_sim_ablation_evaluation.sh [新输出目录]" >&2
    exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: 找不到 eg_codex Python：$PYTHON" >&2
    exit 1
fi
for required in \
    "$DATASET/episode.json" \
    "$DATASET/ground_truth/trajectories_3d.npz" \
    "$DATASET/ground_truth/trajectories_2d/stereo_left.npz" \
    "$DATASET/ground_truth/trajectories_2d/stereo_right.npz" \
    "$DATASET/gui_assets/tissue_fixedsuperbest.npz"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: 缺少正式评估输入：$required" >&2
        exit 1
    fi
done
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "ERROR: 拒绝覆盖已有输出：$OUTPUT_ROOT" >&2
    exit 1
fi
if [[ -n "$EVALUATION_NODE_MANIFEST" && ! -f "$EVALUATION_NODE_MANIFEST" ]]; then
    echo "ERROR: 固定评估点清单不存在：$EVALUATION_NODE_MANIFEST" >&2
    exit 1
fi
export SIM_DATASET="$DATASET"
mkdir -p "$OUTPUT_ROOT"
echo "[三组评估] 固定预设初值：distance=$INITIAL_DISTANCE shape=$INITIAL_SHAPE；参数选择=OFF；80/20 分界=$SPLIT_FRAME/$FRAME_COUNT"
if [[ -n "$EVALUATION_NODE_MANIFEST" ]]; then
    echo "[三组评估] 固定评估点清单：$EVALUATION_NODE_MANIFEST"
fi

run_variant() {
    local key="$1"
    local feedback="$2"
    local stiffness="$3"
    local group="$OUTPUT_ROOT/$key"
    mkdir -p "$group"
    echo "[三组评估] 开始 $key；visual=$feedback；stiffness=$stiffness"

    local stiffness_flag="--no-online-stiffness-update"
    if [[ "$stiffness" == "on" ]]; then
        stiffness_flag="--online-stiffness-update"
    fi
    bash "$ROOT_DIR/scripts/run_sim_reconstruction_headless.sh" \
        --evaluation-start-frame 0 \
        --evaluation-frame-count 0 \
        --evaluation-physics-steps-per-frame 3 \
        --evaluation-open-loop-start-frame "$SPLIT_FRAME" \
        --evaluation-label "$key" \
        --benchmark-output "$group/artifacts" \
        --evaluation-render-images \
        --visual-feedback-mode "$feedback" \
        "$stiffness_flag" \
        --stiffness-log-learning-rate 0.30 \
        --stiffness-maximum-log-step 0.10 \
        --stiffness-signal-ema-decay 0.60 \
        --stiffness-spatial-smoothing-iterations 2 \
        --stiffness-spatial-smoothing-blend 0.30 \
        --initial-paper-distance-stiffness "$INITIAL_DISTANCE" \
        --initial-paper-shape-stiffness "$INITIAL_SHAPE"

    local node_manifest_args=()
    if [[ -n "$EVALUATION_NODE_MANIFEST" ]]; then
        node_manifest_args=(--evaluation-node-manifest "$EVALUATION_NODE_MANIFEST")
    fi
    "$PYTHON" "$ROOT_DIR/scripts/evaluate_sim_trajectory_metrics.py" \
        --reference-dataset "$DATASET" \
        --prediction "$group/artifacts/predicted_trajectories.npz" \
        --split-frame-index "$SPLIT_FRAME" \
        "${node_manifest_args[@]}" \
        --output "$group/trajectory_metrics.json"

    "$PYTHON" "$ROOT_DIR/scripts/evaluate_sim_rendering_metrics.py" \
        --reference-dataset "$DATASET" \
        --prediction-dir "$group/artifacts/renders" \
        --split-frame-index "$SPLIT_FRAME" \
        --device cuda \
        --output "$group/render_metrics.json"
}

run_variant "pbd" "off" "off"
run_variant "pbd_visual_residual" "residual" "off"
run_variant "pbd_visual_residual_stiffness" "residual" "on"

"$PYTHON" "$ROOT_DIR/scripts/summarize_sim_ablation_evaluation.py" \
    --evaluation-root "$OUTPUT_ROOT"

"$PYTHON" "$ROOT_DIR/scripts/visualize_sim_reconstruction_diagnostics.py" \
    --evaluation-root "$OUTPUT_ROOT"

echo "[三组评估] 完成：$OUTPUT_ROOT/comparison.md"
