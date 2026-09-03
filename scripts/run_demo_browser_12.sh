#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"
bash scripts/start_display_browser.sh

export ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
export DISPLAY_NUM="${DISPLAY_NUM:-12}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/Media_HDD/jwshan/tmp/torch_extensions}"
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export CPATH="$ENV_PREFIX/targets/x86_64-linux/include:$ENV_PREFIX/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$ENV_PREFIX/targets/x86_64-linux/include:$ENV_PREFIX/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$ENV_PREFIX/targets/x86_64-linux/lib:$ENV_PREFIX/lib:${LIBRARY_PATH:-}"
exec bash scripts/run_demo_on_display.sh \
    --visual-force-iterations 1 \
    --cameras stereo_left,stereo_right \
    --camera-go-zoom 0.9 \
    --psm-pose-driver depth_then_visual \
    --tissue-mode paper_pbd \
    --psm-tissue-contact \
    "$@"
