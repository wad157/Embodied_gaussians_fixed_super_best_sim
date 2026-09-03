#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
RUN_ROOT="${1:-$ROOT_DIR/outputs/sim_three_datasets_global_only_foundation_complete_v3}"
DATASET="$ROOT_DIR/data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm"
DEPTH_ROOT="$DATASET/estimated_depth/foundation_stereo_rgb_v1"
ASSET_ROOT="$ROOT_DIR/outputs/sim_cotracker3_foundation_depth_v1/flow_depth_assets"
METHOD_C="pbd_cotracker_foundation_depth_global_only"
GROUP="$RUN_ROOT/sim03/future_80to20/$METHOD_C"

for required in \
    "$GROUP/artifacts/predicted_trajectories.npz" \
    "$GROUP/trajectory_metrics.json" \
    "$DEPTH_ROOT/depth_generation_summary.json" \
    "$ASSET_ROOT/report.json"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: 缺少续跑输入：$required" >&2
        exit 1
    fi
done

CUDA_VISIBLE_DEVICES=1 "$PYTHON" \
    "$ROOT_DIR/scripts/evaluate_sim_rendering_metrics.py" \
    --reference-dataset "$DATASET" \
    --prediction-dir "$GROUP/artifacts/renders" \
    --frame-start 288 \
    --device cuda:0 \
    --output "$GROUP/render_metrics.json"

"$PYTHON" "$ROOT_DIR/scripts/summarize_sim_foundation_depth_evaluation.py" \
    --evaluation-root "$RUN_ROOT/sim03" \
    --depth-summary "$DEPTH_ROOT/depth_generation_summary.json" \
    --observation-report "$ASSET_ROOT/report.json" \
    --method-b "pbd_cotracker_foundation_depth" \
    --label-b "B：PBD + CoTracker轨迹校正 + FoundationStereo RGB深度" \
    --method-c "$METHOD_C" \
    --label-c "C：B + 全局distance刚度与全局阻尼更新（无区域/逐粒子刚度）"

"$PYTHON" "$ROOT_DIR/scripts/summarize_sim_h1_safe_three_datasets.py" \
    --sim01 "$RUN_ROOT/sim01/comparison_complete.json" \
    --sim02 "$RUN_ROOT/sim02/comparison_complete.json" \
    --sim03 "$RUN_ROOT/sim03/comparison_complete.json" \
    --method-c "$METHOD_C" \
    --output "$RUN_ROOT/comparison_all.md"

printf 'reconstruction_status=0\nfuture_status=0\n' >"$RUN_ROOT/sim03/status.txt"
printf 'status=complete\nalgorithm=differentiable_global_distance_and_damping\ndepth=foundation_stereo_per_dataset\n' \
    >"$RUN_ROOT/status.txt"
find "$RUN_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"$RUN_ROOT/SHA256SUMS"
echo "补全完成：$RUN_ROOT/comparison_all.md"
