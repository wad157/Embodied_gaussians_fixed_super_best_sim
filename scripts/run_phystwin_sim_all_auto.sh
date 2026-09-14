#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${PHYSTWIN_SIM_OUTPUT_ROOT:-$ROOT_DIR/outputs/phystwin_sim_joint_v1}"
PHYSTWIN_PY="$ROOT_DIR/scripts/run_phystwin_baseline_python.sh"
EVAL_PY="${EVAL_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
AUTO_ROOT="$OUTPUT_ROOT/_automation"
MASTER_LOG="$AUTO_ROOT/automation.log"
mkdir -p "$AUTO_ROOT"

exec 9>"$AUTO_ROOT/automation.lock"
if ! flock -n 9; then
    echo "ERROR: PhysTwin 自动队列已经在运行" >&2
    exit 1
fi

dataset_info() {
    case "$1" in
        sim01) printf '%s %s %s\n' "$ROOT_DIR/data/sim/tissue_retraction_free_support_front_v2" 300 240 ;;
        sim02) printf '%s %s %s\n' "$ROOT_DIR/data/sim/tissue_retraction_free_support_side_v2" 300 240 ;;
        sim03) printf '%s %s %s\n' "$ROOT_DIR/data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm" 360 288 ;;
        *) echo "ERROR: unknown dataset $1" >&2; return 2 ;;
    esac
}

is_physics_running() {
    local key="$1" repeat="$2"
    pgrep -af '[r]un_physics.py' | grep -Fq "$key/$repeat/physics_formal"
}

cma_is_complete() {
    local path="$1"
    "$EVAL_PY" -c \
        'import json,sys; h=json.load(open(sys.argv[1])); raise SystemExit(0 if len(h)==220 and all(float(x["loss"])<float("inf") for x in h) else 1)' \
        "$path"
}

archive_incomplete() {
    local path="$1"
    if [[ -e "$path" ]]; then
        local archived="${path}.interrupted.$(date +%Y%m%d_%H%M%S)"
        mv "$path" "$archived"
        printf '[%s] archived incomplete output %s -> %s\n' \
            "$(date --iso-8601=seconds)" "$path" "$archived" >>"$MASTER_LOG"
    fi
}

run_one() {
    local gpu="$1" key="$2" repeat="$3" seed="$4"
    local dataset frames split
    read -r dataset frames split < <(dataset_info "$key")
    local run="$OUTPUT_ROOT/$key/$repeat"
    local preprocess="$run/preprocess_formal"
    if [[ "$key" == sim01 && "$repeat" == repeat_01 && -f "$run/preprocess_v2/metadata.json" ]]; then
        preprocess="$run/preprocess_v2"
    fi
    local appearance="$run/appearance_formal"
    local physics="$run/physics_formal"
    local artifacts="$run/artifacts_formal"
    local metrics="$run/metrics_formal"
    local log="$run/auto_run.log"
    local manifest="$dataset/evaluation/evaluation_points_30_non_grasp.json"
    local boundary="$dataset/task_inputs/known_grasp_region_boundary.npz"
    mkdir -p "$run"
    printf '[%s] gpu=%s start %s %s seed=%s\n' \
        "$(date --iso-8601=seconds)" "$gpu" "$key" "$repeat" "$seed" >>"$MASTER_LOG"

    if [[ ! -f "$run/protocol_audit.json" ]]; then
        "$EVAL_PY" "$ROOT_DIR/scripts/audit_phystwin_sim_protocol.py" \
            --dataset-key "$key" --dataset "$dataset" \
            --output "$run/protocol_audit.json" >>"$log" 2>&1
    fi

    if [[ ! -f "$preprocess/metadata.json" ]]; then
        archive_incomplete "$preprocess"
        CUDA_VISIBLE_DEVICES="$gpu" bash "$PHYSTWIN_PY" \
            "$ROOT_DIR/baselines/phystwin_sim/prepare_sim.py" \
            --dataset-key "$key" --dataset "$dataset" \
            --output-dir "$preprocess" --seed "$seed" --device cuda:0 \
            >>"$log" 2>&1
    fi

    if [[ ! -f "$appearance/metadata.json" ]]; then
        archive_incomplete "$appearance"
        CUDA_VISIBLE_DEVICES="$gpu" bash "$PHYSTWIN_PY" \
            "$ROOT_DIR/baselines/phystwin_sim/train_appearance.py" \
            --dataset-key "$key" --dataset "$dataset" \
            --preprocess-dir "$preprocess" --output-dir "$appearance" \
            --seed "$seed" --device cuda:0 --iterations 1000 --downsample 2 \
            >>"$log" 2>&1
    fi

    while is_physics_running "$key" "$repeat"; do
        sleep 30
    done
    if [[ ! -f "$physics/metadata.json" ]]; then
        if [[ -f "$physics/cma_history.json" && -f "$physics/optimal_params.pkl" ]] \
            && cma_is_complete "$physics/cma_history.json"; then
            CUDA_VISIBLE_DEVICES="$gpu" bash "$PHYSTWIN_PY" \
                "$ROOT_DIR/baselines/phystwin_sim/run_physics.py" \
                --dataset-key "$key" --preprocess-dir "$preprocess" \
                --output-dir "$physics" --seed "$seed" --device cuda:0 \
                --cma-iterations 20 --adam-iterations 200 \
                --checkpoint-interval 20 --substeps 667 --resume-after-cma \
                >>"$log" 2>&1
        else
            archive_incomplete "$physics"
            CUDA_VISIBLE_DEVICES="$gpu" bash "$PHYSTWIN_PY" \
                "$ROOT_DIR/baselines/phystwin_sim/run_physics.py" \
                --dataset-key "$key" --preprocess-dir "$preprocess" \
                --output-dir "$physics" --seed "$seed" --device cuda:0 \
                --cma-iterations 20 --adam-iterations 200 \
                --checkpoint-interval 20 --substeps 667 \
                >>"$log" 2>&1
        fi
    fi

    if [[ ! -f "$artifacts/provenance.json" ]]; then
        archive_incomplete "$artifacts"
        CUDA_VISIBLE_DEVICES="$gpu" bash "$PHYSTWIN_PY" \
            "$ROOT_DIR/baselines/phystwin_sim/export_sim.py" \
            --dataset-key "$key" --dataset "$dataset" \
            --physics-dir "$physics" --appearance-dir "$appearance" \
            --output-dir "$artifacts" --device cuda:0 >>"$log" 2>&1
    fi

    for capability in reconstruction_7to1 future_80to20; do
        local metric_root="$metrics/$capability"
        local selection=()
        if [[ "$capability" == reconstruction_7to1 ]]; then
            selection=(--frame-start 0 --frame-end-exclusive "$split" --frame-stride 8 --frame-offset 7)
        else
            selection=(--frame-start "$split" --frame-end-exclusive "$frames")
        fi
        mkdir -p "$metric_root"
        if [[ ! -f "$metric_root/trajectory_metrics.json" ]]; then
            "$EVAL_PY" "$ROOT_DIR/scripts/evaluate_sim_trajectory_metrics.py" \
                --reference-dataset "$dataset" \
                --prediction "$artifacts/predicted_trajectories.npz" \
                --controlled-boundary "$boundary" \
                --evaluation-node-manifest "$manifest" \
                "${selection[@]}" \
                --output "$metric_root/trajectory_metrics.json" >>"$log" 2>&1
        fi
        if [[ ! -f "$metric_root/render_metrics.json" ]]; then
            CUDA_VISIBLE_DEVICES="$gpu" "$EVAL_PY" \
                "$ROOT_DIR/scripts/evaluate_sim_rendering_metrics.py" \
                --reference-dataset "$dataset" --prediction-dir "$artifacts" \
                "${selection[@]}" --device cuda:0 \
                --output "$metric_root/render_metrics.json" >>"$log" 2>&1
        fi
    done

    printf '%s\n' \
        'status=complete' "dataset=$key" "repeat=$repeat" "seed=$seed" \
        "gpu=$gpu" 'method=phystwin_native_particles_upstream_lbs_v1' \
        'protocol=offline_prefix_reconstruction_and_known_control_future_rollout' \
        >"$run/status.txt"
    find "$run" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
        >"$run/SHA256SUMS"
    printf '[%s] gpu=%s complete %s %s seed=%s\n' \
        "$(date --iso-8601=seconds)" "$gpu" "$key" "$repeat" "$seed" >>"$MASTER_LOG"
}

worker_zero() {
    run_one 0 sim01 repeat_01 0
    run_one 0 sim03 repeat_01 0
    run_one 0 sim02 repeat_02 1
    run_one 0 sim01 repeat_03 2
    run_one 0 sim02 repeat_03 2
}

worker_one() {
    run_one 1 sim02 repeat_01 0
    run_one 1 sim01 repeat_02 1
    run_one 1 sim03 repeat_02 1
    run_one 1 sim03 repeat_03 2
}

printf 'status=running\nstarted=%s\n' "$(date --iso-8601=seconds)" >"$AUTO_ROOT/status.txt"
worker_zero &
worker_zero_pid=$!
worker_one &
worker_one_pid=$!
failure=0
wait "$worker_zero_pid" || failure=1
wait "$worker_one_pid" || failure=1
if (( failure != 0 )); then
    printf 'status=failed\nfinished=%s\n' "$(date --iso-8601=seconds)" >"$AUTO_ROOT/status.txt"
    exit 1
fi

"$EVAL_PY" "$ROOT_DIR/scripts/summarize_phystwin_sim_baseline.py" \
    --root "$OUTPUT_ROOT" >>"$MASTER_LOG" 2>&1
printf 'status=complete\nfinished=%s\n' "$(date --iso-8601=seconds)" >"$AUTO_ROOT/status.txt"
printf '[%s] all PhysTwin SIM runs complete\n' \
    "$(date --iso-8601=seconds)" >>"$MASTER_LOG"
