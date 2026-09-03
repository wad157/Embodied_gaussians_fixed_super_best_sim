#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/data/sim/tissue_retraction_sufia_ten_regions_v1}"
if [[ $# -gt 0 ]]; then
    shift
fi

echo "[完整数据集 1/5] 十区域连续组织预计算与 Isaac 双目渲染"
if [[ -n "${TISSUE_TRAJECTORY_PATH:-}" ]]; then
    TRAJECTORY_PATH="$(readlink -f "$TISSUE_TRAJECTORY_PATH")"
    if [[ ! -f "$TRAJECTORY_PATH" ]]; then
        echo "错误：指定的预计算组织轨迹不存在：$TRAJECTORY_PATH" >&2
        exit 1
    fi
    echo "[完整数据集] 复用已确认的组织轨迹：$TRAJECTORY_PATH"
    "$ROOT_DIR/scripts/run_generate_sim_dataset.sh" \
        --output "$OUTPUT_DIR" \
        --scene-style sufia_viewpoint_train \
        --material-provenance synthetic_seeded_ten_region_benchmark_v1 \
        --width 1536 \
        --height 1536 \
        --renderer PathTracing \
        --frames 300 \
        --samples-per-pixel 16 \
        "$@" \
        --tissue-trajectory "$TRAJECTORY_PATH"
else
    "$ROOT_DIR/scripts/run_generate_sufia_viewpoint_dataset.sh" "$OUTPUT_DIR" \
        --frames 300 \
        --samples-per-pixel 16 \
        "$@"
fi

echo "[完整数据集 2/5] 导出组织与 PSM 的 3D/双目 2D 真值轨迹"
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
    "$ROOT_DIR/scripts/export_sim_ground_truth_trajectories.py" \
    --dataset "$OUTPUT_DIR"

echo "[完整数据集 3/5] 编码 GUI 兼容视频与无损 RGB 视频"
"$ROOT_DIR/scripts/encode_sim_dataset_videos.sh" "$OUTPUT_DIR"

echo "[完整数据集 4/5] 写入渲染评估协议"
mkdir -p "$OUTPUT_DIR/evaluation"
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
protocol = {
    "schema": "fixedsuperbest.render_evaluation_protocol.v1",
    "reference_rgb": "rgb/<camera>/%06d.png",
    "cameras": ["stereo_left", "stereo_right"],
    "metrics": ["PSNR", "SSIM", "LPIPS AlexNet v0.1"],
    "regions": ["full frame", "ground_truth tissue mask"],
    "evaluator": "../../scripts/evaluate_sim_rendering_metrics.py",
    "example": (
        "python scripts/evaluate_sim_rendering_metrics.py "
        f"--reference-dataset {root} --prediction-dir <重建渲染目录> "
        f"--output {root}/evaluation/reconstruction_render_metrics.json"
    ),
    "note_zh": "当前只有仿真参考图；重建完成后才能产生非恒等的 PSNR/SSIM/LPIPS 数值。",
}
(root / "evaluation/rendering_metrics_protocol.json").write_text(
    json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
trajectory_protocol = {
    "schema": "fixedsuperbest.trajectory_evaluation_protocol.v1",
    "reference_3d": "ground_truth/trajectories_3d.npz",
    "reference_2d": "ground_truth/trajectories_2d/<camera>.npz",
    "evaluation_nodes": "trajectories_3d.npz/tissue_evaluation_node_ids（固定 240 点）",
    "metrics_3d": ["mean/RMSE/median/p95/max mm", "fraction <= 1/2/5/10 mm"],
    "metrics_2d": ["mean/RMSE/median/p95/max pixel", "PCK <= 1/3/5/10 pixel"],
    "alignment": "按 node id 精确匹配，不做位姿、尺度或时间对齐",
    "evaluator": "../../scripts/evaluate_sim_trajectory_metrics.py",
    "example": (
        "python scripts/evaluate_sim_trajectory_metrics.py "
        f"--reference-dataset {root} --prediction <重建轨迹.npz> "
        f"--output {root}/evaluation/reconstruction_trajectory_metrics.json"
    ),
}
(root / "evaluation/trajectory_metrics_protocol.json").write_text(
    json.dumps(trajectory_protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY

echo "[完整数据集 5/5] 运行完整性与数值质量门禁"
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
    "$ROOT_DIR/scripts/validate_sim_dataset.py" \
    --dataset "$OUTPUT_DIR"

echo "[完成] 可用于重建评估的数据集：$OUTPUT_DIR"
