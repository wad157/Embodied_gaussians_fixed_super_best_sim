#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${SIM_DATASET:-$ROOT_DIR/data/sim/tissue_retraction_closeup_inplane_full_v1}"

if [[ ! -f "$DATASET/gui_assets/tissue_fixedsuperbest.npz" ]]; then
    echo "ERROR: 缺少重建组织资产：$DATASET/gui_assets/tissue_fixedsuperbest.npz" >&2
    exit 1
fi
if [[ ! -f "$DATASET/gui_assets/instrument_masks.npz" ]]; then
    echo "ERROR: 缺少 GUI mask 资产：$DATASET/gui_assets/instrument_masks.npz" >&2
    exit 1
fi
if [[ ! -f "$DATASET/gui_assets/official_psm_tip_meshes_v2.npz" ]]; then
    echo "ERROR: 缺少官方 PSM 末端 GUI 网格：$DATASET/gui_assets/official_psm_tip_meshes_v2.npz" >&2
    exit 1
fi
if [[ ! -f "$DATASET/task_inputs/psm_link_poses.npz" ]]; then
    echo "ERROR: 缺少任务输入 PSM link 位姿：$DATASET/task_inputs/psm_link_poses.npz" >&2
    exit 1
fi
if [[ ! -f "$DATASET/task_inputs/phases.json" ]]; then
    echo "ERROR: 缺少任务输入抓取时序：$DATASET/task_inputs/phases.json" >&2
    exit 1
fi
if [[ ! -f "$DATASET/task_inputs/red_marker_boundary.npz" ]]; then
    echo "ERROR: 缺少红色标记轨迹边界：$DATASET/task_inputs/red_marker_boundary.npz" >&2
    echo "请先运行 scripts/prepare_sim_red_marker_boundary.py。" >&2
    exit 1
fi

exec bash "$ROOT_DIR/scripts/run_demo_thinlinc.sh" \
    --dataset "$DATASET" \
    --fps 20 \
    --tissue-mode paper_pbd \
    --visual-feedback-mode residual \
    --visual-residual-iterations 8 \
    --visual-residual-learning-rate-m 0.00004 \
    --visual-feedback-update-interval 3 \
    --online-stiffness-update \
    --stiffness-log-learning-rate 0.18 \
    --no-psm-tissue-contact \
    "$@"
