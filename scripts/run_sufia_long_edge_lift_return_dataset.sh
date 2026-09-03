#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-full}"
PREVIEW_FRAME="${2:-130}"
PREVIEW_VERSION="${PREVIEW_VERSION:-sidegrasp_v12_horizontal_plane_lift30mm}"
EG_CODEX_PYTHON="/Media_HDD/jwshan/conda_envs/eg_codex/bin/python"
TRAJECTORY_PATH="$ROOT_DIR/data/sim_precompute/tissue_long_edge_lift_return_sufia_v2_lift30mm.npz"
DATASET_PATH="$ROOT_DIR/data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm"
PREVIEW_PATH="$ROOT_DIR/outputs/tissue_long_edge_lift_return_preview_${PREVIEW_VERSION}_frame_${PREVIEW_FRAME}"
PSM_ASSET_SOURCE="$ROOT_DIR/data/sim/tissue_retraction_closeup_inplane_full_v1/gui_assets"

if [[ "$MODE" != "preview" && "$MODE" != "side_preview" && "$MODE" != "full" ]]; then
    echo "用法：$0 [preview|side_preview|full] [preview_frame]" >&2
    exit 2
fi

if [[ ! -f "$TRAJECTORY_PATH" ]]; then
    WARP_CACHE_PATH="$ROOT_DIR/.cache/warp" \
    PYTHONPATH="$ROOT_DIR/src" \
        "$EG_CODEX_PYTHON" "$ROOT_DIR/scripts/precompute_sufia_tissue_trajectory.py" \
        --output "$TRAJECTORY_PATH" \
        --frames 360 \
        --fps 30 \
        --physics-hz 120 \
        --device cuda \
        --geometry-profile rectangular_tissue_strip \
        --motion-profile edge_lift_return \
        --regional-seed 240610788 \
        --regional-youngs-min-pa 850 \
        --regional-youngs-max-pa 1050 \
        --poissons-ratio 0.45 \
        --material-iterations 28 \
        --grasp-coupling-frequency-hz 12 \
        --support-mode free \
        --grasp-offset-mm 0 -24 \
        --lift-displacement-mm 0 0 30 \
        --pull-displacement-mm 0 0 0
fi

COMMON_ARGS=(
    --frames 360
    --fps 30
    --physics-hz 120
    --regional-seed 240610788
    # 固定随机种子保证可复现；10 个区域的真值刚度在 850--1050 Pa 内随机采样。
    --regional-youngs-min-pa 850
    --regional-youngs-max-pa 1050
    --poissons-ratio 0.45
    --task-variant edge_lift_return
    --motion-profile edge_lift_return
    --grasp-offset-mm 0 -24
    # 65 度侧向斜上方观察；令图像水平轴与组织长轴 x 对齐，避免组织平面斜置。
    --task-camera-eye 0.0030 0.0583 0.1801
    --task-camera-target 0.003 0.0 0.055
    --tissue-uv-detail-scale 0.68
    --tissue-texture-contrast 1.75
    --tissue-texture-bias -0.632 -0.363 -0.208
    # 侧向进入：远端沿 +y 指向组织；夹爪接触点连线 94% 以上沿世界 z。
    --psm-ready-q7 0.014 0.3160 0.1700 1.663196 0 -1.45 0.96
    --psm-contact-q7 0.013 0.2025 0.1655 1.663196 0 -1.45 0.96
    # 官方 PSM 网格校准：30 mm 峰值处夹爪中点到组织夹持中心约 0.21 mm。
    --psm-lift-q7 0.01625 0.2505 0.1355 1.663196 0 -1.45 0.14
    --psm-pull-q7 0.013 0.2025 0.1655 1.663196 0 -1.45 0.14
)

if [[ "$MODE" == "preview" ]]; then
    "$ROOT_DIR/scripts/run_generate_sim_dataset.sh" \
        --output "$PREVIEW_PATH" \
        --scene-style sufia_viewpoint_train \
        --material-provenance synthetic_seeded_ten_region_long_edge_lift30mm_benchmark_v2 \
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

if [[ "$MODE" == "side_preview" ]]; then
    "$ROOT_DIR/scripts/run_generate_sim_dataset.sh" \
        --output "$PREVIEW_PATH" \
        --scene-style sufia_viewpoint_train \
        --material-provenance synthetic_seeded_ten_region_long_edge_lift30mm_benchmark_v2 \
        --width 1024 \
        --height 1024 \
        --renderer PathTracing \
        --samples-per-pixel 16 \
        --no-canonical-scan \
        --preview-frame-index "$PREVIEW_FRAME" \
        "${COMMON_ARGS[@]}" \
        --task-camera-eye 0.112 -0.045 0.063 \
        --task-camera-target 0.0029 -0.0239 0.0516 \
        --tissue-trajectory "$TRAJECTORY_PATH"
    exit 0
fi

TISSUE_TRAJECTORY_PATH="$TRAJECTORY_PATH" \
    "$ROOT_DIR/scripts/run_complete_sufia_viewpoint_dataset.sh" \
    "$DATASET_PATH" \
    --material-provenance synthetic_seeded_ten_region_long_edge_lift30mm_benchmark_v2 \
    "${COMMON_ARGS[@]}"

"$EG_CODEX_PYTHON" "$ROOT_DIR/scripts/prepare_sim_tissue_gui_asset.py" \
    --dataset "$DATASET_PATH" \
    --support-mode free
"$EG_CODEX_PYTHON" "$ROOT_DIR/scripts/prepare_sim_gui_masks.py" \
    --dataset "$DATASET_PATH"
cp "$PSM_ASSET_SOURCE/official_psm_tip_meshes_v2.npz" \
    "$DATASET_PATH/gui_assets/official_psm_tip_meshes_v2.npz"
cp "$PSM_ASSET_SOURCE/official_psm_tip_meshes_v2.json" \
    "$DATASET_PATH/gui_assets/official_psm_tip_meshes_v2.json"
"$EG_CODEX_PYTHON" "$ROOT_DIR/scripts/prepare_sim_red_marker_boundary.py" \
    --dataset "$DATASET_PATH"

echo "[完成] 长边夹持—抬升—放回数据集：$DATASET_PATH"
