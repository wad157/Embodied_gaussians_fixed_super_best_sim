#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EG_CODEX="${EG_CODEX_PYTHON:-/Media_HDD/jwshan/conda_envs/eg_codex/bin/python}"
DATA_ROOT="$ROOT_DIR/data/sim"
PRECOMPUTE_ROOT="$ROOT_DIR/data/sim_precompute"
FRONT_DATASET="$DATA_ROOT/tissue_retraction_free_support_front_v2"
SIDE_DATASET="$DATA_ROOT/tissue_retraction_free_support_side_v2"
EVALUATION_ROOT="$ROOT_DIR/outputs/two_free_support_ablation_v2"
PSM_ASSET_SOURCE="$DATA_ROOT/tissue_retraction_closeup_inplane_full_v1/gui_assets"
RESUME_AFTER_FRONT_RENDER="${RESUME_AFTER_FRONT_RENDER:-0}"
RESUME_AFTER_FRONT_GUI="${RESUME_AFTER_FRONT_GUI:-0}"
RESUME_AFTER_SIDE_PRECOMPUTE="${RESUME_AFTER_SIDE_PRECOMPUTE:-0}"

paths_to_create=("$SIDE_DATASET")
if [[ "$RESUME_AFTER_FRONT_RENDER" != "1" && "$RESUME_AFTER_FRONT_GUI" != "1" && "$RESUME_AFTER_SIDE_PRECOMPUTE" != "1" ]]; then
    paths_to_create+=("$FRONT_DATASET" "$EVALUATION_ROOT")
fi
for path in "${paths_to_create[@]}"; do
    if [[ -e "$path" ]]; then
        echo "错误：拒绝覆盖已有路径：$path" >&2
        exit 1
    fi
done
for path in \
    "$PSM_ASSET_SOURCE/official_psm_tip_meshes_v2.npz" \
    "$PSM_ASSET_SOURCE/official_psm_tip_meshes_v2.json"; do
    if [[ ! -f "$path" ]]; then
        echo "错误：缺少可复用的官方 PSM 末端视觉资产：$path" >&2
        exit 1
    fi
done

mkdir -p "$PRECOMPUTE_ROOT" "$ROOT_DIR/.cache/warp" "$EVALUATION_ROOT"
FRONT_TRAJECTORY="$PRECOMPUTE_ROOT/tissue_retraction_free_support_front_v2_ten_regions.npz"
SIDE_TRAJECTORY="$PRECOMPUTE_ROOT/tissue_retraction_free_support_side_v2_ten_regions.npz"
if [[ "$RESUME_AFTER_FRONT_RENDER" != "1" && "$RESUME_AFTER_FRONT_GUI" != "1" && "$RESUME_AFTER_SIDE_PRECOMPUTE" != "1" && -e "$FRONT_TRAJECTORY" ]]; then
    echo "错误：拒绝覆盖已有正面预计算轨迹。" >&2
    exit 1
fi
if [[ "$RESUME_AFTER_SIDE_PRECOMPUTE" != "1" && -e "$SIDE_TRAJECTORY" ]]; then
    echo "错误：拒绝覆盖已有预计算轨迹。" >&2
    exit 1
fi
if [[ "$RESUME_AFTER_FRONT_RENDER" == "1" || "$RESUME_AFTER_FRONT_GUI" == "1" || "$RESUME_AFTER_SIDE_PRECOMPUTE" == "1" ]]; then
    for path in "$FRONT_DATASET/episode.json" "$FRONT_TRAJECTORY"; do
        if [[ ! -f "$path" ]]; then
            echo "错误：断点恢复缺少正面数据：$path" >&2
            exit 1
        fi
    done
    if find "$EVALUATION_ROOT" -mindepth 1 -print -quit | grep -q .; then
        echo "错误：评估目录并非空目录，拒绝不明确地恢复：$EVALUATION_ROOT" >&2
        exit 1
    fi
fi
if [[ "$RESUME_AFTER_FRONT_GUI" == "1" || "$RESUME_AFTER_SIDE_PRECOMPUTE" == "1" ]]; then
    for path in \
        "$FRONT_DATASET/gui_assets/tissue_fixedsuperbest.npz" \
        "$FRONT_DATASET/gui_assets/instrument_masks.npz" \
        "$FRONT_DATASET/task_inputs/red_marker_boundary.npz"; do
        if [[ ! -f "$path" ]]; then
            echo "错误：GUI 后断点恢复缺少资产：$path" >&2
            exit 1
        fi
    done
fi
if [[ "$RESUME_AFTER_SIDE_PRECOMPUTE" == "1" && ! -f "$SIDE_TRAJECTORY" ]]; then
    echo "错误：侧向轨迹后断点恢复缺少预计算轨迹：$SIDE_TRAJECTORY" >&2
    exit 1
fi

precompute() {
    local output="$1"
    shift
    WARP_CACHE_PATH="$ROOT_DIR/.cache/warp" PYTHONPATH="$ROOT_DIR/src" \
        "$EG_CODEX" "$ROOT_DIR/scripts/precompute_sufia_tissue_trajectory.py" \
        --output "$output" \
        --frames 300 \
        --fps 30 \
        --physics-hz 120 \
        --regional-seed 240610788 \
        --regional-youngs-min-pa 780 \
        --regional-youngs-max-pa 1020 \
        --poissons-ratio 0.45 \
        --material-iterations 24 \
        --grasp-coupling-frequency-hz 12 \
        --support-mode free \
        "$@"
}

render_complete() {
    local trajectory="$1"
    local dataset="$2"
    local variant="$3"
    shift 3
    TISSUE_TRAJECTORY_PATH="$trajectory" \
        "$ROOT_DIR/scripts/run_complete_sufia_viewpoint_dataset.sh" "$dataset" \
        --task-variant "$variant" \
        --regional-seed 240610788 \
        --regional-youngs-min-pa 780 \
        --regional-youngs-max-pa 1020 \
        --material-provenance synthetic_seeded_ten_region_free_support_rgb_benchmark_v2 \
        --task-camera-eye 0.029 -0.0728 0.1932 \
        --task-camera-target 0.0025 0.0 0.059 \
        --tissue-uv-detail-scale 0.68 \
        --tissue-texture-contrast 1.75 \
        --tissue-texture-bias -0.632 -0.363 -0.208 \
        "$@"
}

prepare_gui_assets() {
    local dataset="$1"
    "$EG_CODEX" "$ROOT_DIR/scripts/prepare_sim_tissue_gui_asset.py" \
        --dataset "$dataset" \
        --support-mode free
    "$EG_CODEX" "$ROOT_DIR/scripts/prepare_sim_gui_masks.py" \
        --dataset "$dataset"
    cp "$PSM_ASSET_SOURCE/official_psm_tip_meshes_v2.npz" \
        "$dataset/gui_assets/official_psm_tip_meshes_v2.npz"
    cp "$PSM_ASSET_SOURCE/official_psm_tip_meshes_v2.json" \
        "$dataset/gui_assets/official_psm_tip_meshes_v2.json"
    "$EG_CODEX" "$ROOT_DIR/scripts/prepare_sim_red_marker_boundary.py" \
        --dataset "$dataset"
}

if [[ "$RESUME_AFTER_SIDE_PRECOMPUTE" == "1" ]]; then
    echo "[断点恢复] 正面数据/GUI 与侧向组织真值已完成，重新渲染已修正 PSM 位姿的侧向任务"
elif [[ "$RESUME_AFTER_FRONT_GUI" == "1" ]]; then
    echo "[断点恢复] 正面数据与 GUI/五点边界已完成，直接继续侧面任务"
elif [[ "$RESUME_AFTER_FRONT_RENDER" == "1" ]]; then
    echo "[断点恢复] 保留首次失败报告并按修正后的遮挡门禁重新验证正面数据"
    if [[ -f "$FRONT_DATASET/validation_report.json" ]]; then
        if [[ -e "$FRONT_DATASET/validation_report.initial_failed.json" ]]; then
            echo "错误：首次失败报告备份已存在。" >&2
            exit 1
        fi
        mv "$FRONT_DATASET/validation_report.json" \
            "$FRONT_DATASET/validation_report.initial_failed.json"
    fi
    "$EG_CODEX" "$ROOT_DIR/scripts/validate_sim_dataset.py" \
        --dataset "$FRONT_DATASET"
else
    echo "[1/8] 正面牵拉：生成无硬固定十区域组织真值"
    precompute "$FRONT_TRAJECTORY" \
        --grasp-offset-mm 22 0 \
        --lift-displacement-mm -5 0 0.99 \
        --pull-displacement-mm -14 0 2.76

    echo "[2/8] 正面牵拉：渲染完整双目 RGB/GT 数据集"
    render_complete "$FRONT_TRAJECTORY" "$FRONT_DATASET" front_pull \
        --grasp-offset-mm 22 0 \
        --psm-ready-q7 0.2335 0 0.1158 0 0 0 0.96 \
        --psm-contact-q7 0.1560 0 0.1510 0 0 0 0.96 \
        --psm-lift-q7 0.1228 0 0.1482 0 0 0 0.14 \
        --psm-pull-q7 0.0631 0 0.1431 0 0 0 0.14
fi

echo "[3/8] 正面牵拉：接入首帧全视角建模、GUI mask、红标记边界"
if [[ "$RESUME_AFTER_FRONT_GUI" != "1" && "$RESUME_AFTER_SIDE_PRECOMPUTE" != "1" ]]; then
    prepare_gui_assets "$FRONT_DATASET"
fi

echo "[4/8] 侧面牵拉：生成同材质无硬固定十区域组织真值"
if [[ "$RESUME_AFTER_SIDE_PRECOMPUTE" != "1" ]]; then
    precompute "$SIDE_TRAJECTORY" \
        --grasp-offset-mm 0 -22 \
        --lift-displacement-mm 0 5 0.99 \
        --pull-displacement-mm 0 14 2.76
fi

echo "[5/8] 侧面牵拉：渲染完整双目 RGB/GT 数据集"
render_complete "$SIDE_TRAJECTORY" "$SIDE_DATASET" side_pull \
    --grasp-offset-mm 0 -22 \
    --psm-ready-q7 0 0.2335 0.1158 0 0 0 0.96 \
    --psm-contact-q7 0 0.1560 0.1510 0 0 0 0.96 \
    --psm-lift-q7 0 0.1228 0.1482 0 0 0 0.14 \
    --psm-pull-q7 0 0.0631 0.1431 0 0 0 0.14

echo "[6/8] 侧面牵拉：接入首帧全视角建模、GUI mask、红标记边界"
prepare_gui_assets "$SIDE_DATASET"

echo "[7/8] 正面牵拉：固定初值三组 RGB-only 80/20 评估"
SIM_DATASET="$FRONT_DATASET" \
    "$ROOT_DIR/scripts/run_sim_ablation_evaluation.sh" \
    "$EVALUATION_ROOT/front_pull"

echo "[8/8] 侧面牵拉：固定初值三组 RGB-only 80/20 评估"
SIM_DATASET="$SIDE_DATASET" \
    "$ROOT_DIR/scripts/run_sim_ablation_evaluation.sh" \
    "$EVALUATION_ROOT/side_pull"

"$EG_CODEX" "$ROOT_DIR/scripts/summarize_dual_sim_evaluation.py" \
    --front "$EVALUATION_ROOT/front_pull/comparison.json" \
    --side "$EVALUATION_ROOT/side_pull/comparison.json" \
    --output "$EVALUATION_ROOT/两个任务总表.md"

echo "[全部完成] $EVALUATION_ROOT/两个任务总表.md"
