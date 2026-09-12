#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_PY="${EVAL_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
RUN_ROOT="$(realpath -m "${1:-$ROOT_DIR/outputs/eh_surgs_sim_unified_three_repeats_v1}")"
SIM03_SEED0="$(realpath -e "${2:-$ROOT_DIR/outputs/eh_surgs_sim03_once_v1/repeat_01/sim03}")"

if [[ $# -gt 2 ]]; then
    echo "用法：bash scripts/run_eh_surgs_sim_three_repeats_two_gpus.sh [输出目录] [已有SIM03 seed0目录]" >&2
    exit 2
fi
if [[ ! -f "$SIM03_SEED0/status.txt" ]] || ! grep -q '^status=complete$' "$SIM03_SEED0/status.txt"; then
    echo "ERROR: 已有 SIM03 seed0 不是完整运行：$SIM03_SEED0" >&2
    exit 1
fi
if [[ -e "$RUN_ROOT/comparison_mean_std.json" ]]; then
    echo "ERROR: 已存在汇总，拒绝覆盖：$RUN_ROOT" >&2
    exit 1
fi
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/runs/repeat_01"
if [[ ! -e "$RUN_ROOT/runs/repeat_01/sim03" ]]; then
    ln -s "$SIM03_SEED0" "$RUN_ROOT/runs/repeat_01/sim03"
fi
printf '%s\n' \
    'status=running' \
    'method=eh_surgs_pinned_som_query_anchored_displacement_v2' \
    'datasets=sim01,sim02,sim03' \
    'repeat_count=3' \
    'seeds=0,1,2' \
    'aggregation=mean_and_sample_std_no_best_selection' \
    'sim03_repeat_01=reused_completed_seed0_without_rerun' \
    'gpus=0,1' \
    >"$RUN_ROOT/status.txt"

run_one() {
    local dataset_key="$1" repeat_id="$2" seed="$3" gpu="$4"
    local run="$RUN_ROOT/runs/$repeat_id/$dataset_key"
    if [[ -f "$run/status.txt" ]] && grep -q '^status=complete$' "$run/status.txt"; then
        return
    fi
    if [[ -e "$run" ]]; then
        echo "ERROR: 检测到不完整输出，拒绝覆盖：$run" >&2
        return 1
    fi
    printf 'start=%s/%s gpu=%s\n' "$repeat_id" "$dataset_key" "$gpu" >>"$RUN_ROOT/logs/progress.log"
    SIM_GPU_ID="$gpu" bash "$ROOT_DIR/scripts/run_eh_surgs_sim_baseline_once.sh" \
        "$dataset_key" "$repeat_id" "$seed" "$run"
    printf 'complete=%s/%s gpu=%s\n' "$repeat_id" "$dataset_key" "$gpu" >>"$RUN_ROOT/logs/progress.log"
}

gpu_zero() {
    run_one sim01 repeat_01 0 0
    run_one sim02 repeat_01 0 0
    run_one sim03 repeat_02 1 0
    run_one sim01 repeat_03 2 0
}

gpu_one() {
    run_one sim01 repeat_02 1 1
    run_one sim02 repeat_02 1 1
    run_one sim03 repeat_03 2 1
    run_one sim02 repeat_03 2 1
}

gpu_zero & pid_zero=$!
gpu_one & pid_one=$!
batch_status=0
wait "$pid_zero" || batch_status=1
wait "$pid_one" || batch_status=1
if (( batch_status != 0 )); then
    echo "ERROR: 至少一个 GPU worker 失败；保留所有输出并拒绝汇总" >&2
    exit 1
fi

"$EVAL_PY" "$ROOT_DIR/scripts/summarize_eh_surgs_sim_baseline.py" --root "$RUN_ROOT"
printf '%s\n' \
    'status=complete' \
    'method=eh_surgs_pinned_som_query_anchored_displacement_v2' \
    'datasets=sim01,sim02,sim03' \
    'repeat_count=3' \
    'seeds=0,1,2' \
    'aggregation=mean_and_sample_std_no_best_selection' \
    'sim03_repeat_01=reused_completed_seed0_without_rerun' \
    'gpus=0,1' \
    >"$RUN_ROOT/status.txt"
find "$RUN_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"$RUN_ROOT/SHA256SUMS"
echo "完成：$RUN_ROOT/comparison_mean_std.md"
