#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${STIFFNESS_EVALUATION_OUTPUT:-${ROOT_DIR}/outputs/stiffness_evaluation_${RUN_STAMP}}"

printf 'Stiffness evaluation output: %s\n' "$RESULT_DIR"
exec bash "${ROOT_DIR}/scripts/run_demo_browser_12.sh" \
    --tissue-mode paper_pbd \
    --visual-feedback-mode residual \
    --online-stiffness-update \
    --visual-feedback-update-interval 3 \
    --stiffness-evaluation-horizons 1,3,5,10 \
    --stiffness-evaluation-output "$RESULT_DIR" \
    "$@"
