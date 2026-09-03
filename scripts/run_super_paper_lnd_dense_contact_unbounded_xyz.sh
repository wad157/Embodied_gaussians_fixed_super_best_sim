#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_OPTIMIZE="${PYTHON_OPTIMIZE:-/Media_HDD/jwshan/conda_envs/online_dvrk/bin/python}"
PYTHON_GUI="${PYTHON_GUI:-/Media_HDD/jwshan/conda_envs/eg_codex/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
STAGE="${STAGE:-all}"
OVERWRITE="${OVERWRITE:-0}"

VISUAL_ROOT="$ROOT_DIR/data/super/psm_visual_calibration/raw_paper_lnd_stereo_dense_contact_v4"
SAM2_ROOT="$VISUAL_ROOT/surgicalsam2_multianchor_parts_dense_contact_v6"
OPTIMIZATION_ROOT="$SAM2_ROOT/online_stereo_cma_se3_unbounded_dense_contact_v4"
GUI_ROOT="$OPTIMIZATION_ROOT/gui_unbounded_xyz_v1"
RAW_ROOT="$ROOT_DIR/data/super/psm_raw_kinematics_v1（纯机器人学版本）"
PAPER_ROOT="/Media_HDD/jwshan/wad/online_dvrk_tracking"

ARGS=()
if [[ "$OVERWRITE" == "1" ]]; then
    ARGS=(--overwrite)
fi

run_optimize() {
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-dense-se3-unbounded}" \
        "$PYTHON_OPTIMIZE" \
        "$ROOT_DIR/scripts/optimize_super_p420006_stereo_sam2_online.py" \
        --observation-mode multianchor_parts \
        --geometry paper_lnd \
        --visual-root "$VISUAL_ROOT" \
        --masks "$SAM2_ROOT/stereo_multianchor_part_masks.npz" \
        --sam2-report "$SAM2_ROOT/report.json" \
        --anchor-state "$VISUAL_ROOT/annotations/live_sam2_dense_contact_v6/annotation_state.json" \
        --paper-repo "$PAPER_ROOT" \
        --mesh-dir "$PAPER_ROOT/urdfs/dVRK/meshes" \
        --output-dir "$OPTIMIZATION_ROOT" \
        --freeze-all-joint-corrections \
        --unbounded-translation \
        --global-translation-scale-mm 5 \
        --local-translation-scale-mm 5 \
        --global-prior-weight 0.04 \
        --local-prior-weight 0.2 \
        --manual-anchor-tip-weight-multiplier 3 \
        --cuda-device "$CUDA_DEVICE" \
        "${ARGS[@]}"
}

run_gui() {
    "$PYTHON_GUI" \
        "$ROOT_DIR/scripts/build_super_paper_lnd_sam2_online_gui.py" \
        --raw-root "$RAW_ROOT" \
        --visual-root "$VISUAL_ROOT" \
        --optimization-root "$OPTIMIZATION_ROOT" \
        --paper-repo "$PAPER_ROOT" \
        --output-dir "$GUI_ROOT" \
        --pose-version raw_paper_lnd_sam2_dense_contact_unbounded_xyz \
        --require-frozen-all-joint-corrections \
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
