#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_PY="${EVAL_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
RUN_ROOT="${1:-$ROOT_DIR/outputs/endogaussian_sim_unified_three_repeats_v1}"
GPU_ID="${SIM_GPU_ID:-0}"

if [[ $# -gt 1 ]]; then
    echo "用法：bash scripts/run_endogaussian_sim_baseline_three_repeats.sh [输出目录]" >&2
    exit 2
fi
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SIM_GPU_ID 必须是非负整数" >&2
    exit 2
fi
if [[ -e "$RUN_ROOT/comparison_mean_std.json" ]]; then
    echo "ERROR: baseline 三次评测已汇总，拒绝覆盖：$RUN_ROOT" >&2
    exit 1
fi
mkdir -p "$RUN_ROOT/logs"
printf '%s\n' \
    'status=running' \
    'method=endogaussian_pinned_som_query_anchored_displacement_v2' \
    'datasets=sim01,sim02,sim03' \
    'repeat_count=3' \
    'seeds=0,1,2' \
    'aggregation=mean_and_sample_std_no_best_selection' \
    "gpu=$GPU_ID" \
    >"$RUN_ROOT/status.txt"

for repeat_number in 1 2 3; do
    repeat_id="$(printf 'repeat_%02d' "$repeat_number")"
    seed=$((repeat_number - 1))
    for dataset_key in sim01 sim02 sim03; do
        run="$RUN_ROOT/runs/$repeat_id/$dataset_key"
        if [[ -f "$run/status.txt" ]] && grep -q '^status=complete$' "$run/status.txt"; then
            echo "skip_complete=$repeat_id/$dataset_key" >>"$RUN_ROOT/logs/progress.log"
            continue
        fi
        if [[ -e "$run" ]]; then
            echo "ERROR: 检测到不完整输出，拒绝覆盖：$run" >&2
            exit 1
        fi
        echo "start=$repeat_id/$dataset_key" >>"$RUN_ROOT/logs/progress.log"
        SIM_GPU_ID="$GPU_ID" bash "$ROOT_DIR/scripts/run_endogaussian_sim_baseline_once.sh" \
            "$dataset_key" "$repeat_id" "$seed" "$run"
        echo "complete=$repeat_id/$dataset_key" >>"$RUN_ROOT/logs/progress.log"
    done
done

"$EVAL_PY" "$ROOT_DIR/scripts/summarize_endogaussian_sim_baseline.py" --root "$RUN_ROOT"
printf '%s\n' \
    'status=complete' \
    'method=endogaussian_pinned_som_query_anchored_displacement_v2' \
    'datasets=sim01,sim02,sim03' \
    'repeat_count=3' \
    'seeds=0,1,2' \
    'aggregation=mean_and_sample_std_no_best_selection' \
    "gpu=$GPU_ID" \
    >"$RUN_ROOT/status.txt"
find "$RUN_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"$RUN_ROOT/SHA256SUMS"
echo "完成：$RUN_ROOT/comparison_mean_std.md"
