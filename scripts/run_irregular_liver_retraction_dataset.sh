#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-full}"
PREVIEW_FRAME="${2:-270}"
TRAJECTORY_PATH="$ROOT_DIR/data/sim_precompute/tissue_retraction_irregular_liver_closeup_v1.npz"
DATASET_PATH="$ROOT_DIR/data/sim/tissue_retraction_irregular_liver_closeup_v1"
PREVIEW_VERSION="${PREVIEW_VERSION:-v2}"
PREVIEW_PATH="$ROOT_DIR/outputs/irregular_liver_retraction_preview_${PREVIEW_VERSION}_frame_${PREVIEW_FRAME}"
EG_CODEX_PYTHON="/Media_HDD/jwshan/conda_envs/eg_codex/bin/python"

if [[ "$MODE" != "preview" && "$MODE" != "full" ]]; then
    echo "用法：$0 [preview|full] [preview_frame]" >&2
    exit 2
fi

if [[ ! -f "$TRAJECTORY_PATH" ]]; then
    PYTHONPATH="$ROOT_DIR/src" "$EG_CODEX_PYTHON" \
        "$ROOT_DIR/scripts/precompute_sufia_tissue_trajectory.py" \
        --output "$TRAJECTORY_PATH" \
        --frames 360 \
        --fps 30 \
        --physics-hz 120 \
        --device cuda \
        --geometry-profile irregular_liver_lobe \
        --regional-seed 240610788 \
        --regional-youngs-min-pa 850 \
        --regional-youngs-max-pa 1000 \
        --poissons-ratio 0.45 \
        --material-iterations 28 \
        --grasp-coupling-frequency-hz 12 \
        --support-mode free \
        --grasp-offset-mm 30 -3 \
        --lift-displacement-mm -3 0 1.2 \
        --pull-displacement-mm -12 4 2.5
fi

COMMON_ARGS=(
    --frames 360
    --fps 30
    --physics-hz 120
    --regional-seed 240610788
    --regional-youngs-min-pa 850
    --regional-youngs-max-pa 1000
    --poissons-ratio 0.45
    --task-variant front_pull
    --task-camera-eye 0.022 -0.062 0.175
    --task-camera-target 0.003 0.000 0.055
    --tissue-uv-detail-scale 0.65
    --tissue-texture-contrast 3.00
    --tissue-texture-bias 0.00 0.00 0.00
    --psm-ready-q7 0.285 0.020 0.116 0 0 0 0.96
    --psm-contact-q7 0.215 0.020 0.151 0 0 0 0.96
    --psm-lift-q7 0.195 0.020 0.148 0 0 0 0.14
    --psm-pull-q7 0.135 -0.007 0.1435 0 0 0 0.14
)

if [[ "$MODE" == "preview" ]]; then
    "$ROOT_DIR/scripts/run_generate_sim_dataset.sh" \
        --output "$PREVIEW_PATH" \
        --scene-style sufia_viewpoint_train \
        --material-provenance synthetic_seeded_ten_region_benchmark_v2 \
        --width 1536 \
        --height 1536 \
        --renderer PathTracing \
        --samples-per-pixel 16 \
        --no-canonical-scan \
        --preview-frame-index "$PREVIEW_FRAME" \
        "${COMMON_ARGS[@]}" \
        --tissue-trajectory "$TRAJECTORY_PATH"
    exit 0
fi

TISSUE_TRAJECTORY_PATH="$TRAJECTORY_PATH" \
    "$ROOT_DIR/scripts/run_complete_sufia_viewpoint_dataset.sh" \
    "$DATASET_PATH" \
    "${COMMON_ARGS[@]}"
