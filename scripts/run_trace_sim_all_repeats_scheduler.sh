#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="$(realpath -m "${1:-$ROOT_DIR/outputs/trace_sim_all_v1}")"
EXISTING_SIM03="$(realpath -m "${TRACE_SIM03_REPEAT01:-$ROOT_DIR/outputs/trace_sim03_rgb_only_full_k_v1/repeat_01}")"
EXISTING_SIM03_PID="${TRACE_SIM03_REPEAT01_PID:-328614}"
RUNNER="$ROOT_DIR/scripts/run_trace_sim_baseline_once.sh"
SUMMARY_PY="${EVAL_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
MAX_ATTEMPTS="${TRACE_MAX_ATTEMPTS:-3}"

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/runs/repeat_01" "$OUTPUT_ROOT/runs/repeat_02" "$OUTPUT_ROOT/runs/repeat_03"
if [[ ! -e "$OUTPUT_ROOT/runs/repeat_01/sim03" && ! -L "$OUTPUT_ROOT/runs/repeat_01/sim03" ]]; then
    ln -s "$EXISTING_SIM03" "$OUTPUT_ROOT/runs/repeat_01/sim03"
fi

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }

is_complete() {
    local output="$1"
    [[ -f "$output/status.txt" ]] && grep -q '^status=complete$' "$output/status.txt"
}

archive_incomplete() {
    local output="$1"
    local attempt="$2"
    if [[ -e "$output" || -L "$output" ]]; then
        local archived="${output}.failed_attempt_${attempt}_$(date '+%Y%m%d_%H%M%S')"
        mv "$output" "$archived"
        echo "[$(timestamp)] archived incomplete output: $archived"
    fi
}

run_one() {
    local gpu="$1"
    local dataset="$2"
    local repeat="$3"
    local seed="$4"
    local output="$5"
    local attempt=1
    if is_complete "$output"; then
        echo "[$(timestamp)] skip complete $dataset/$repeat"
        return 0
    fi
    while (( attempt <= MAX_ATTEMPTS )); do
        archive_incomplete "$output" "$attempt"
        echo "[$(timestamp)] start gpu=$gpu $dataset/$repeat seed=$seed attempt=$attempt"
        if SIM_GPU_ID="$gpu" bash "$RUNNER" "$dataset" "$repeat" "$seed" "$output"; then
            if is_complete "$output"; then
                echo "[$(timestamp)] complete gpu=$gpu $dataset/$repeat"
                return 0
            fi
        fi
        echo "[$(timestamp)] failed gpu=$gpu $dataset/$repeat attempt=$attempt"
        attempt=$((attempt + 1))
        sleep 15
    done
    echo "[$(timestamp)] exhausted retries gpu=$gpu $dataset/$repeat" >&2
    return 1
}

run_queue() {
    local gpu="$1"
    shift
    local failed=0
    while (( $# >= 4 )); do
        run_one "$gpu" "$1" "$2" "$3" "$4" || failed=1
        shift 4
    done
    return "$failed"
}

echo "status=running" >"$OUTPUT_ROOT/scheduler_status.txt"
echo "started=$(timestamp)" >>"$OUTPUT_ROOT/scheduler_status.txt"
echo "existing_sim03_pid=$EXISTING_SIM03_PID" >>"$OUTPUT_ROOT/scheduler_status.txt"

run_queue 0 \
    sim01 repeat_01 0 "$OUTPUT_ROOT/runs/repeat_01/sim01" \
    sim01 repeat_03 2 "$OUTPUT_ROOT/runs/repeat_03/sim01" \
    >"$OUTPUT_ROOT/logs/gpu0_queue_a.log" 2>&1 &
QUEUE_A=$!

run_queue 0 \
    sim02 repeat_01 0 "$OUTPUT_ROOT/runs/repeat_01/sim02" \
    sim02 repeat_03 2 "$OUTPUT_ROOT/runs/repeat_03/sim02" \
    >"$OUTPUT_ROOT/logs/gpu0_queue_b.log" 2>&1 &
QUEUE_B=$!

run_queue 1 \
    sim01 repeat_02 1 "$OUTPUT_ROOT/runs/repeat_02/sim01" \
    sim03 repeat_03 2 "$OUTPUT_ROOT/runs/repeat_03/sim03" \
    >"$OUTPUT_ROOT/logs/gpu1_queue_a.log" 2>&1 &
QUEUE_C=$!

(
    if [[ "$EXISTING_SIM03_PID" =~ ^[0-9]+$ ]] \
        && (( EXISTING_SIM03_PID > 1 )) \
        && kill -0 "$EXISTING_SIM03_PID" 2>/dev/null; then
        echo "[$(timestamp)] waiting for existing sim03/repeat_01 pid=$EXISTING_SIM03_PID"
        tail --pid="$EXISTING_SIM03_PID" -f /dev/null
    fi
    run_one 1 sim03 repeat_01 0 "$EXISTING_SIM03"
    run_queue 1 \
        sim02 repeat_02 1 "$OUTPUT_ROOT/runs/repeat_02/sim02" \
        sim03 repeat_02 1 "$OUTPUT_ROOT/runs/repeat_02/sim03"
) >"$OUTPUT_ROOT/logs/gpu1_queue_b.log" 2>&1 &
QUEUE_D=$!

printf 'queue_pids=%s,%s,%s,%s\n' "$QUEUE_A" "$QUEUE_B" "$QUEUE_C" "$QUEUE_D" >>"$OUTPUT_ROOT/scheduler_status.txt"

FAILED=0
wait "$QUEUE_A" || FAILED=1
wait "$QUEUE_B" || FAILED=1
wait "$QUEUE_C" || FAILED=1
wait "$QUEUE_D" || FAILED=1

if (( FAILED == 0 )); then
    if "$SUMMARY_PY" "$ROOT_DIR/scripts/summarize_trace_sim_baseline.py" --root "$OUTPUT_ROOT" \
        >"$OUTPUT_ROOT/logs/summary.log" 2>&1; then
        printf 'status=complete\nfinished=%s\n' "$(timestamp)" >"$OUTPUT_ROOT/scheduler_status.txt"
        exit 0
    fi
fi
printf 'status=failed\nfinished=%s\n' "$(timestamp)" >"$OUTPUT_ROOT/scheduler_status.txt"
exit 1
