#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_OPTIMIZE="${PYTHON_OPTIMIZE:-/Media_HDD/jwshan/conda_envs/online_dvrk/bin/python}"
PYTHON_GUI="${PYTHON_GUI:-/Media_HDD/jwshan/conda_envs/eg_codex/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
STAGE="${STAGE:-all}"
OVERWRITE="${OVERWRITE:-0}"

ARGS=()
if [[ "$OVERWRITE" == "1" ]]; then
    ARGS=(--overwrite)
fi

run_optimize() {
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-first-stereo}" \
        "$PYTHON_OPTIMIZE" \
        "$ROOT_DIR/scripts/optimize_super_paper_lnd_first_stereo_static.py" \
        --cuda-device "$CUDA_DEVICE" \
        "${ARGS[@]}"
}

run_gui() {
    "$PYTHON_GUI" \
        "$ROOT_DIR/scripts/build_super_paper_lnd_first_stereo_static_gui.py" \
        "${ARGS[@]}"
}

case "$STAGE" in
    all)
        run_optimize
        run_gui
        ;;
    optimize)
        run_optimize
        ;;
    gui)
        run_gui
        ;;
    *)
        echo "ERROR: STAGE must be all, optimize, or gui" >&2
        exit 2
        ;;
esac
