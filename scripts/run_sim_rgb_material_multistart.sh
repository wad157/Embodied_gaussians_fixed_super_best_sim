#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${SIM_DATASET:-$ROOT_DIR/data/sim/tissue_retraction_closeup_inplane_full_v1}"
PYTHON="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
DEFAULT_OUTPUT="$ROOT_DIR/outputs/sim_rgb_material_multistart_$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${1:-${SIM_RGB_MULTISTART_OUTPUT_ROOT:-$DEFAULT_OUTPUT}}"
TRAIN_VALIDATION_SPLIT_FRAME=220
FORMAL_FUTURE_SPLIT_FRAME=240

if [[ $# -gt 1 ]]; then
    echo "用法：bash scripts/run_sim_rgb_material_multistart.sh [新输出目录]" >&2
    exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: 找不到 eg_codex Python：$PYTHON" >&2
    exit 1
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "ERROR: 拒绝覆盖已有输出：$OUTPUT_ROOT" >&2
    exit 1
fi
mkdir -p "$OUTPUT_ROOT"

export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/examples:${PYTHONPATH:-}"
export CONDA_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/usr/bin:$CONDA_PREFIX/bin:$PATH"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/Media_HDD/jwshan/tmp/torch_extensions}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-2}"
export SIM_DATASET="$DATASET"
mkdir -p "$TORCH_EXTENSIONS_DIR"

run_candidate() {
    local name="$1"
    local distance="$2"
    local shape="$3"
    local candidate_root="$OUTPUT_ROOT/$name"
    mkdir -p "$candidate_root"
    echo "[RGB多初值] $name: distance=$distance shape=$shape"
    bash "$ROOT_DIR/scripts/run_sim_reconstruction_headless.sh" \
        --evaluation-start-frame 0 \
        --evaluation-frame-count "$FORMAL_FUTURE_SPLIT_FRAME" \
        --evaluation-physics-steps-per-frame 3 \
        --evaluation-open-loop-start-frame "$TRAIN_VALIDATION_SPLIT_FRAME" \
        --evaluation-label "rgb_multistart_$name" \
        --benchmark-output "$candidate_root/artifacts" \
        --no-evaluation-render-images \
        --initial-paper-distance-stiffness "$distance" \
        --initial-paper-shape-stiffness "$shape"
}

# All hypotheses are global and uniform. They contain no ten-region label or
# simulator material truth; the same online local updater starts from each one.
run_candidate soft 0.18 0.0035
run_candidate nominal 0.31 0.0058
run_candidate firm 0.48 0.0085

"$PYTHON" "$ROOT_DIR/scripts/select_sim_rgb_material_hypothesis.py" \
    --evaluation-root "$OUTPUT_ROOT" \
    --split-frame-index "$FORMAL_FUTURE_SPLIT_FRAME" \
    --history-frames 20 \
    --snapshot-count 4

echo "[RGB多初值] 完成：$OUTPUT_ROOT/rgb_material_selection.md"
