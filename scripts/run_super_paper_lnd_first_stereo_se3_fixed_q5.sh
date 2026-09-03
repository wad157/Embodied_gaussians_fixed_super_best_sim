#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_OPTIMIZE="${PYTHON_OPTIMIZE:-/Media_HDD/jwshan/conda_envs/online_dvrk/bin/python}"
PYTHON_GUI="${PYTHON_GUI:-/Media_HDD/jwshan/conda_envs/eg_codex/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
STAGE="${STAGE:-all}"
OVERWRITE="${OVERWRITE:-0}"
CALIBRATION_ROOT="$ROOT_DIR/data/super/psm_visual_calibration/raw_paper_lnd_first_stereo_se3_fixed_q5_v1"

ARGS=()
if [[ "$OVERWRITE" == "1" ]]; then
    ARGS=(--overwrite)
fi

run_optimize() {
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-first-stereo-se3-fixed-q5}" \
        "$PYTHON_OPTIMIZE" \
        "$ROOT_DIR/scripts/optimize_super_paper_lnd_first_stereo_se3_fixed_q5.py" \
        --cuda-device "$CUDA_DEVICE" \
        "${ARGS[@]}"
}

run_gui() {
    "$PYTHON_GUI" \
        "$ROOT_DIR/scripts/build_super_paper_lnd_first_stereo_static_gui.py" \
        --calibration-root "$CALIBRATION_ROOT" \
        --output-dir "$CALIBRATION_ROOT/gui_v1" \
        --pose-version raw_paper_lnd_first_stereo_se3_fixed_q5 \
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
