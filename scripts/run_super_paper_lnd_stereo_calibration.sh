#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/Media_HDD/jwshan/conda_envs/eg_codex/bin/python}"
PAPER_REPO="${PAPER_REPO:-/Media_HDD/jwshan/wad/online_dvrk_tracking}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
STAGE="${STAGE:-all}"
OVERWRITE="${OVERWRITE:-0}"

VISUAL_ROOT="$ROOT_DIR/data/super/psm_visual_calibration/raw_paper_lnd_stereo_v1"
SAM2_ROOT="$VISUAL_ROOT/surgicalsam2_stereo_sequence_v1"
OPTIMIZATION_ROOT="$SAM2_ROOT/online_stereo_cma_v1"
GUI_ROOT="$OPTIMIZATION_ROOT/gui_v1"
OVERWRITE_ARGS=()
if [[ "$OVERWRITE" == "1" ]]; then
    OVERWRITE_ARGS=(--overwrite)
fi

run_segmentation() {
    "$PYTHON_BIN" \
        "$ROOT_DIR/scripts/track_super_p420006_stereo_surgicalsam2.py" \
        --instrument paper_lnd \
        --visual-root "$VISUAL_ROOT" \
        --paper-repo "$PAPER_REPO" \
        --output-dir "$SAM2_ROOT" \
        --anchor-policy first \
        --cuda-device "$CUDA_DEVICE" \
        "${OVERWRITE_ARGS[@]}"
}

run_optimization() {
    "$PYTHON_BIN" \
        "$ROOT_DIR/scripts/optimize_super_p420006_stereo_sam2_online.py" \
        --geometry paper_lnd \
        --visual-root "$VISUAL_ROOT" \
        --masks "$SAM2_ROOT/stereo_surgicalsam2_masks.npz" \
        --sam2-report "$SAM2_ROOT/report.json" \
        --first-annotation "$VISUAL_ROOT/annotations/keyframe_00.json" \
        --paper-repo "$PAPER_REPO" \
        --mesh-dir "$PAPER_REPO/urdfs/dVRK/meshes" \
        --output-dir "$OPTIMIZATION_ROOT" \
        --cuda-device "$CUDA_DEVICE" \
        "${OVERWRITE_ARGS[@]}"
}

run_gui_build() {
    "$PYTHON_BIN" \
        "$ROOT_DIR/scripts/build_super_paper_lnd_sam2_online_gui.py" \
        --visual-root "$VISUAL_ROOT" \
        --optimization-root "$OPTIMIZATION_ROOT" \
        --paper-repo "$PAPER_REPO" \
        --output-dir "$GUI_ROOT" \
        "${OVERWRITE_ARGS[@]}"
}

case "$STAGE" in
    all)
        run_segmentation
        run_optimization
        run_gui_build
        ;;
    segment)
        run_segmentation
        ;;
    optimize)
        run_optimization
        ;;
    gui)
        run_gui_build
        ;;
    *)
        echo "ERROR: STAGE must be all, segment, optimize, or gui" >&2
        exit 2
        ;;
esac
