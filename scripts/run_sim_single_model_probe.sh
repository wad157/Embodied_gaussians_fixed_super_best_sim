#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${SIM_DATASET:-$ROOT_DIR/data/sim/tissue_retraction_sufia_smooth_300frames_10s_v5}"
ENV_ROOT="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
PYTHON="$ENV_ROOT/bin/python"

if [[ $# -ne 8 ]]; then
    echo "用法：$0 OUTPUT DIST SHAPE LOG_LR LOG_CAP EMA SMOOTH_ITERS SMOOTH_BLEND" >&2
    exit 2
fi

OUTPUT_ROOT="$1"
DISTANCE="$2"
SHAPE="$3"
LOG_LR="$4"
LOG_CAP="$5"
EMA_DECAY="$6"
SMOOTH_ITERS="$7"
SMOOTH_BLEND="$8"
VISUAL_MODE="${VISUAL_MODE:-residual}"
VISUAL_ITERS="${VISUAL_ITERS:-12}"
VISUAL_LR_M="${VISUAL_LR_M:-0.00006}"
VISUAL_MAX_MM="${VISUAL_MAX_MM:-1.00}"
VISUAL_IMAGE_SCALE="${VISUAL_IMAGE_SCALE:-0.125}"
ONLINE_STIFFNESS="${ONLINE_STIFFNESS:-1}"

if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "ERROR: 拒绝覆盖已有输出：$OUTPUT_ROOT" >&2
    exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: 找不到 eg_codex Python：$PYTHON" >&2
    exit 1
fi

export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/examples:${PYTHONPATH:-}"
export CONDA_PREFIX="$ENV_ROOT"
export CUDA_HOME="$ENV_ROOT"
export CUDA_PATH="$ENV_ROOT"
export PATH="$ENV_ROOT/usr/bin:$ENV_ROOT/bin:$PATH"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/Media_HDD/jwshan/tmp/torch_extensions}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-2}"
mkdir -p "$TORCH_EXTENSIONS_DIR"

STIFFNESS_FLAG=(--online-stiffness-update)
if [[ "$ONLINE_STIFFNESS" == "0" ]]; then
    STIFFNESS_FLAG=(--no-online-stiffness-update)
fi

"$PYTHON" "$ROOT_DIR/examples/example_embodied_super_offline.py" \
    --dataset "$DATASET" \
    --fps 20 \
    --tissue-mode paper_pbd \
    --visual-feedback-mode "$VISUAL_MODE" \
    --visual-residual-iterations "$VISUAL_ITERS" \
    --visual-residual-learning-rate-m "$VISUAL_LR_M" \
    --visual-residual-maximum-mm "$VISUAL_MAX_MM" \
    --visual-residual-image-scale "$VISUAL_IMAGE_SCALE" \
    --visual-feedback-update-interval 1 \
    "${STIFFNESS_FLAG[@]}" \
    --stiffness-log-learning-rate "$LOG_LR" \
    --stiffness-maximum-log-step "$LOG_CAP" \
    --stiffness-signal-ema-decay "$EMA_DECAY" \
    --stiffness-spatial-smoothing-iterations "$SMOOTH_ITERS" \
    --stiffness-spatial-smoothing-blend "$SMOOTH_BLEND" \
    --initial-paper-distance-stiffness "$DISTANCE" \
    --initial-paper-shape-stiffness "$SHAPE" \
    --no-psm-tissue-contact \
    --cameras stereo_left,stereo_right \
    --psm-pose-driver raw_paper_lnd_sam2_dense_contact_unbounded_xyz \
    --psm-visual-mode full \
    --psm-roll-offset-deg 0 \
    --psm-camera-translation-mm 0 0 0 \
    --psm-world-translation-mm 0 0 0 \
    --evaluation-headless \
    --evaluation-start-frame 0 \
    --evaluation-frame-count 0 \
    --evaluation-physics-steps-per-frame 2 \
    --evaluation-open-loop-start-frame 240 \
    --evaluation-label single_model_parameter_probe \
    --benchmark-output "$OUTPUT_ROOT/artifacts" \
    --no-evaluation-render-images

"$PYTHON" "$ROOT_DIR/scripts/evaluate_sim_trajectory_metrics.py" \
    --reference-dataset "$DATASET" \
    --prediction "$OUTPUT_ROOT/artifacts/predicted_trajectories.npz" \
    --split-frame-index 240 \
    --output "$OUTPUT_ROOT/trajectory_metrics.json"

echo "[单实例参数标定] 完成：$OUTPUT_ROOT"
