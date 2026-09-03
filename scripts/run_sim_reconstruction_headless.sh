#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${SIM_DATASET:-$ROOT_DIR/data/sim/tissue_retraction_closeup_inplane_full_v1}"
ENV_ROOT="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
PYTHON="$ENV_ROOT/bin/python"
TORCH_LIB="$ENV_ROOT/lib/python3.11/site-packages/torch/lib"

for required in \
    "$DATASET/episode.json" \
    "$DATASET/gui_assets/tissue_fixedsuperbest.npz" \
    "$DATASET/gui_assets/instrument_masks.npz" \
    "$DATASET/gui_assets/official_psm_tip_meshes_v2.npz" \
    "$DATASET/task_inputs/psm_link_poses.npz" \
    "$DATASET/task_inputs/phases.json" \
    "$DATASET/task_inputs/red_marker_boundary.npz"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: 缺少 headless 重建输入：$required" >&2
        exit 1
    fi
done
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: 找不到 eg_codex Python：$PYTHON" >&2
    exit 1
fi

export CONDA_PREFIX="$ENV_ROOT"
export CUDA_HOME="$ENV_ROOT"
export CUDA_PATH="$ENV_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="$ENV_ROOT/usr/bin:$ENV_ROOT/bin:$PATH"
export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/examples:$ENV_ROOT/usr/lib/python3/dist-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$TORCH_LIB:$ENV_ROOT/usr/lib/x86_64-linux-gnu:$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$ENV_ROOT/targets/x86_64-linux/include:$ENV_ROOT/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$ENV_ROOT/targets/x86_64-linux/include:$ENV_ROOT/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$ENV_ROOT/targets/x86_64-linux/lib:$ENV_ROOT/lib:${LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/Media_HDD/jwshan/tmp/torch_extensions}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-2}"
mkdir -p "$TORCH_EXTENSIONS_DIR"

cd "$ROOT_DIR"
exec "$PYTHON" examples/example_embodied_super_offline.py \
    --dataset "$DATASET" \
    --fps 30 \
    --tissue-mode paper_pbd \
    --visual-feedback-mode residual \
    --visual-residual-iterations 8 \
    --visual-residual-learning-rate-m 0.00004 \
    --visual-feedback-update-interval 2 \
    --online-stiffness-update \
    --stiffness-log-learning-rate 0.18 \
    --no-psm-tissue-contact \
    --cameras stereo_left,stereo_right \
    --psm-pose-driver raw_paper_lnd_sam2_dense_contact_unbounded_xyz \
    --psm-visual-mode full \
    --psm-roll-offset-deg 0 \
    --psm-camera-translation-mm 0 0 0 \
    --psm-world-translation-mm 0 0 0 \
    --evaluation-headless \
    "$@"
