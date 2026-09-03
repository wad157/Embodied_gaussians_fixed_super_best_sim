#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/data/sim/tissue_retraction_sufia_viewpoint_v1}"
EG_CODEX_PYTHON="${EG_CODEX_PYTHON:-/Media_HDD/jwshan/conda_envs/eg_codex/bin/python}"
if [[ $# -gt 0 ]]; then
    shift
fi

if [[ ! -x "$EG_CODEX_PYTHON" ]]; then
    echo "错误：未找到 eg_codex Python：$EG_CODEX_PYTHON" >&2
    exit 1
fi

PRECOMPUTE_DIR="$ROOT_DIR/data/sim_precompute"
TRAJECTORY_PATH="$PRECOMPUTE_DIR/$(basename "$OUTPUT_DIR")_ten_regions.npz"
WARP_CACHE_DIR="$ROOT_DIR/.cache/warp"
mkdir -p "$PRECOMPUTE_DIR" "$WARP_CACHE_DIR"

echo "[1/2] 在 eg_codex 中预计算连续组织的 10 个随机刚度区域……"
WARP_CACHE_PATH="$WARP_CACHE_DIR" PYTHONPATH="$ROOT_DIR/src" \
    "$EG_CODEX_PYTHON" "$ROOT_DIR/scripts/precompute_sufia_tissue_trajectory.py" \
    --output "$TRAJECTORY_PATH" \
    "$@"

echo "[2/2] 在 eg_sim / Isaac Sim 中渲染官方 PSM、腹部资产和组织轨迹……"

exec "$ROOT_DIR/scripts/run_generate_sim_dataset.sh" \
    --output "$OUTPUT_DIR" \
    --scene-style sufia_viewpoint_train \
    --material-provenance synthetic_seeded_ten_region_benchmark_v1 \
    --width 1536 \
    --height 1536 \
    --renderer PathTracing \
    --samples-per-pixel 64 \
    "$@" \
    --tissue-trajectory "$TRAJECTORY_PATH"
