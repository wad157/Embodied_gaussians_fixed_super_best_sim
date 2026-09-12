#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_PY="${EVAL_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
GPU_ID="${SIM_GPU_ID:-0}"

if [[ $# -ne 4 ]]; then
    echo "用法：bash scripts/finalize_endogaussian_sim_baseline_existing.sh <sim01|sim02|sim03> <repeat_id> <seed> <输出目录>" >&2
    exit 2
fi

DATASET_KEY="$1"
REPEAT_ID="$2"
SEED="$3"
OUTPUT_ROOT="$(realpath -m "$4")"
case "$DATASET_KEY" in
    sim01) DATASET="$ROOT_DIR/data/sim/tissue_retraction_free_support_front_v2"; FRAMES=300 ;;
    sim02) DATASET="$ROOT_DIR/data/sim/tissue_retraction_free_support_side_v2"; FRAMES=300 ;;
    sim03) DATASET="$ROOT_DIR/data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm"; FRAMES=360 ;;
    *) echo "ERROR: 未知数据集：$DATASET_KEY" >&2; exit 2 ;;
esac

ARTIFACTS="$OUTPUT_ROOT/artifacts"
LOG="$OUTPUT_ROOT/run.log"
MANIFEST="$DATASET/evaluation/evaluation_points_30_non_grasp.json"
BOUNDARY="$DATASET/task_inputs/known_grasp_region_boundary.npz"
FUTURE_START=$((FRAMES * 4 / 5))

for required in \
    "$OUTPUT_ROOT/model/point_cloud/iteration_3000/point_cloud.ply" \
    "$ARTIFACTS/predicted_trajectories.npz" \
    "$ARTIFACTS/provenance.json" \
    "$MANIFEST" \
    "$BOUNDARY"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: 缺少完整训练/导出文件：$required" >&2
        exit 1
    fi
done

"$EVAL_PY" - "$ARTIFACTS/provenance.json" "$DATASET_KEY" "$OUTPUT_ROOT/model" "$FRAMES" <<'PY'
import json
import sys
from pathlib import Path

provenance = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
dataset_key, expected_model, frames = sys.argv[2], Path(sys.argv[3]).resolve(), int(sys.argv[4])
if provenance.get("dataset_key") != dataset_key:
    raise SystemExit("ERROR: provenance dataset_key 不匹配")
if Path(provenance.get("model_path", "")).resolve() != expected_model:
    raise SystemExit("ERROR: provenance model_path 不匹配")
if provenance.get("iteration") != 3000:
    raise SystemExit("ERROR: provenance checkpoint 不是 iteration_3000")
if provenance.get("trajectory_decoder_version") != "som_query_anchored_displacement_v2":
    raise SystemExit("ERROR: trajectory decoder 版本错误")
if provenance.get("tracks", {}).get("query_valid_count") != 30:
    raise SystemExit("ERROR: 查询点不是 30/30 有效")
expected_renders = len(range(7, frames * 4 // 5, 8)) + frames - frames * 4 // 5
for camera in ("stereo_left", "stereo_right"):
    rgb = list((Path(sys.argv[1]).parent / "rgb" / camera).glob("*.png"))
    alpha = list((Path(sys.argv[1]).parent / "alpha" / camera).glob("*.png"))
    if len(rgb) != expected_renders or len(alpha) != expected_renders:
        raise SystemExit("ERROR: {} 渲染不完整：rgb={} alpha={} expected={}".format(camera, len(rgb), len(alpha), expected_renders))
PY

if [[ -d "$OUTPUT_ROOT/metrics" ]]; then
    suffix="$(date +%Y%m%d_%H%M%S)"
    mv "$OUTPUT_ROOT/metrics" "$OUTPUT_ROOT/metrics.interrupted_$suffix"
fi

for CAPABILITY in reconstruction_7to1 future_80to20; do
    METRICS="$OUTPUT_ROOT/metrics/$CAPABILITY"
    mkdir -p "$METRICS"
    if [[ "$CAPABILITY" == reconstruction_7to1 ]]; then
        SELECTION=(--frame-start 0 --frame-end-exclusive "$FUTURE_START" --frame-stride 8 --frame-offset 7)
    else
        SELECTION=(--frame-start "$FUTURE_START" --frame-end-exclusive "$FRAMES")
    fi
    "$EVAL_PY" "$ROOT_DIR/scripts/evaluate_sim_trajectory_metrics.py" \
        --reference-dataset "$DATASET" \
        --prediction "$ARTIFACTS/predicted_trajectories.npz" \
        --controlled-boundary "$BOUNDARY" \
        --evaluation-node-manifest "$MANIFEST" \
        "${SELECTION[@]}" \
        --output "$METRICS/trajectory_metrics.json" \
        >>"$LOG" 2>&1
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$EVAL_PY" \
        "$ROOT_DIR/scripts/evaluate_sim_rendering_metrics.py" \
        --reference-dataset "$DATASET" \
        --prediction-dir "$ARTIFACTS" \
        "${SELECTION[@]}" \
        --device cuda:0 \
        --output "$METRICS/render_metrics.json" \
        >>"$LOG" 2>&1
done

PORT="$(sed -n 's/^port=//p' "$OUTPUT_ROOT/status.txt" | head -1)"
printf '%s\n' \
    'status=complete' \
    "dataset=$DATASET_KEY" \
    "repeat=$REPEAT_ID" \
    "seed=$SEED" \
    "gpu=$GPU_ID" \
    "port=${PORT:-unknown}" \
    'method=endogaussian_pinned_som_query_anchored_displacement_v2' \
    'protocol=offline_prefix_reconstruction_and_zero_shot_future_extrapolation' \
    >"$OUTPUT_ROOT/status.txt"

find "$OUTPUT_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"$OUTPUT_ROOT/SHA256SUMS"
echo "完成续算：$OUTPUT_ROOT"
