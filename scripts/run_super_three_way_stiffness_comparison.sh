#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${THREE_WAY_OUTPUT:-${ROOT_DIR}/outputs/stiffness_three_way_${RUN_STAMP}}"
START_FRAME="${THREE_WAY_START_FRAME:-350}"
FRAME_COUNT="${THREE_WAY_FRAME_COUNT:-211}"
PHYSICS_STEPS="${THREE_WAY_PHYSICS_STEPS_PER_FRAME:-3}"

COMMON_ARGS=(
    --tissue-mode paper_pbd
    --evaluation-headless
    --evaluation-start-frame "${START_FRAME}"
    --evaluation-frame-count "${FRAME_COUNT}"
    --evaluation-physics-steps-per-frame "${PHYSICS_STEPS}"
    --visual-feedback-update-interval 3
)

printf 'Three-way evaluation output: %s\n' "${OUTPUT_ROOT}"
printf 'Frame schedule: start=%s count=%s physics_steps=%s\n' \
    "${START_FRAME}" "${FRAME_COUNT}" "${PHYSICS_STEPS}"

PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/examples" python \
    "${ROOT_DIR}/examples/example_embodied_super_offline.py" \
    "${COMMON_ARGS[@]}" \
    --visual-feedback-mode off \
    --no-online-stiffness-update \
    --stiffness-evaluation-output "${OUTPUT_ROOT}/fixed_pbd"

PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/examples" python \
    "${ROOT_DIR}/examples/example_embodied_super_offline.py" \
    "${COMMON_ARGS[@]}" \
    --visual-feedback-mode residual \
    --no-online-stiffness-update \
    --stiffness-evaluation-output "${OUTPUT_ROOT}/residual_only"

PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/examples" python \
    "${ROOT_DIR}/examples/example_embodied_super_offline.py" \
    "${COMMON_ARGS[@]}" \
    --visual-feedback-mode residual \
    --online-stiffness-update \
    --stiffness-evaluation-output "${OUTPUT_ROOT}/residual_online_stiffness"

python "${ROOT_DIR}/scripts/summarize_super_three_way_comparison.py" \
    "${OUTPUT_ROOT}/fixed_pbd" \
    "${OUTPUT_ROOT}/residual_only" \
    "${OUTPUT_ROOT}/residual_online_stiffness" \
    --output "${OUTPUT_ROOT}"

printf 'Comparison JSON: %s\n' "${OUTPUT_ROOT}/comparison.json"
printf 'Comparison Markdown: %s\n' "${OUTPUT_ROOT}/comparison.md"
