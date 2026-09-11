#!/usr/bin/env bash
set -euo pipefail

# 新协议三次复测：SIM01/02 的 A/B/C 各新跑三次；SIM03 的 A 新跑三次，
# B/C 复用已完成的一次并各新跑两次。全部在同一GPU上串行执行。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
RUN_ROOT="${1:-$ROOT_DIR/outputs/sim_three_datasets_alltracker_unified_protocol_three_repeats_v1}"
SIM03_EXISTING_ROOT="${SIM03_UNIFIED_EXISTING_ROOT:-$ROOT_DIR/outputs/sim03_alltracker_unified_80_20_7to1_once_v2}"
ASSET_BASE="${SIM_ALLTRACKER_ASSET_BASE:-$ROOT_DIR/outputs/sim_three_datasets_alltracker_medium_h3w4_complete_v1/assets}"
GPU_ID="${SIM_GPU_ID:-0}"

if [[ $# -gt 1 ]]; then
    echo "用法：bash scripts/run_sim_three_datasets_unified_protocol_three_repeats.sh [输出目录]" >&2
    exit 2
fi
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SIM_GPU_ID 必须是非负整数：$GPU_ID" >&2
    exit 2
fi
if [[ -f "$RUN_ROOT/status.txt" ]] && grep -q '^status=complete$' "$RUN_ROOT/status.txt"; then
    echo "ERROR: 三次复测已完成，拒绝覆盖：$RUN_ROOT" >&2
    exit 1
fi
for path in \
    "$PYTHON" \
    "$SIM03_EXISTING_ROOT/comparison_unified.json" \
    "$ASSET_BASE/sim01/flow_depth_assets/report.json" \
    "$ASSET_BASE/sim02/flow_depth_assets/report.json" \
    "$ASSET_BASE/sim03/flow_depth_assets/report.json"; do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: 缺少批量评测输入：$path" >&2
        exit 1
    fi
done

mkdir -p "$RUN_ROOT/logs"
printf '%s\n' \
    'status=running' \
    "pid=$$" \
    'tracker=AllTracker' \
    'datasets=sim01,sim02,sim03' \
    'repeat_count_per_dataset_method=3' \
    'sim01_new_rollouts=A3,B3,C3' \
    'sim02_new_rollouts=A3,B3,C3' \
    'sim03_new_rollouts=A3,B2,C2' \
    "sim03_BC_repeat_01_source=$SIM03_EXISTING_ROOT" \
    'protocol=single_rollout_first80_7to1_holdout_last20_open_loop' \
    'execution=continuous_serial' \
    "gpu=$GPU_ID" >"$RUN_ROOT/status.txt"

run_one() {
    local repeat_id="$1"
    local dataset_id="$2"
    local method_id="$3"
    local evaluation_root="$RUN_ROOT/runs/$repeat_id/$dataset_id"
    local method_name
    case "$method_id" in
        A) method_name=pbd ;;
        B) method_name=pbd_alltracker_foundation_depth ;;
        C) method_name=pbd_alltracker_foundation_depth_global_only_h3w4 ;;
    esac
    local marker="$evaluation_root/status_${method_name}.txt"
    if [[ -f "$marker" ]] && grep -q '^status=complete$' "$marker"; then
        printf '%s\n' "skip_complete=$repeat_id/$dataset_id/$method_id" >>"$RUN_ROOT/logs/progress.log"
        return
    fi
    if [[ -e "$evaluation_root/methods/$method_name" ]]; then
        echo "ERROR: 检测到不完整输出，停止以免覆盖：$evaluation_root/methods/$method_name" >&2
        exit 1
    fi
    printf '%s\n' "start=$repeat_id/$dataset_id/$method_id" >>"$RUN_ROOT/logs/progress.log"
    SIM_GPU_ID="$GPU_ID" \
    SIM_REPEAT_ID="$repeat_id" \
    SIM_ALLTRACKER_ASSET_BASE="$ASSET_BASE" \
        bash "$ROOT_DIR/scripts/run_sim_alltracker_unified_method_once.sh" \
        "$dataset_id" "$method_id" "$evaluation_root"
    printf '%s\n' "complete=$repeat_id/$dataset_id/$method_id" >>"$RUN_ROOT/logs/progress.log"
}

# SIM01和SIM02：A/B/C各三次，全部使用新协议重新运行。
for dataset_id in sim01 sim02; do
    for repeat_number in 1 2 3; do
        repeat_id="$(printf 'repeat_%02d' "$repeat_number")"
        for method_id in A B C; do
            run_one "$repeat_id" "$dataset_id" "$method_id"
        done
    done
done

# SIM03：A跑三次；B/C第1次直接使用已经完成的新协议结果，只新增第2、3次。
for repeat_number in 1 2 3; do
    repeat_id="$(printf 'repeat_%02d' "$repeat_number")"
    run_one "$repeat_id" sim03 A
    if [[ "$repeat_number" -ge 2 ]]; then
        run_one "$repeat_id" sim03 B
        run_one "$repeat_id" sim03 C
    fi
done

"$PYTHON" "$ROOT_DIR/scripts/summarize_sim_three_datasets_unified_protocol_three_repeats.py" \
    --root "$RUN_ROOT" \
    --sim03-existing-root "$SIM03_EXISTING_ROOT" \
    --asset-base "$ASSET_BASE"

printf '%s\n' \
    'status=complete' \
    'tracker=AllTracker' \
    'datasets=sim01,sim02,sim03' \
    'repeat_count_per_dataset_method=3' \
    'sim01_new_rollouts=A3,B3,C3' \
    'sim02_new_rollouts=A3,B3,C3' \
    'sim03_new_rollouts=A3,B2,C2' \
    "sim03_BC_repeat_01_source=$SIM03_EXISTING_ROOT" \
    'protocol=single_rollout_first80_7to1_holdout_last20_open_loop' \
    'execution=continuous_serial' \
    "gpu=$GPU_ID" >"$RUN_ROOT/status.txt"
find "$RUN_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"$RUN_ROOT/SHA256SUMS"
echo "完成：$RUN_ROOT/comparison_mean_std.md"
