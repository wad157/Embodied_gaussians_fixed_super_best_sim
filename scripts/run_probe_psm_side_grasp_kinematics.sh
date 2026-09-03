#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_sim}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/Media_HDD/jwshan/wad/sufia_sim/IsaacLab}"
ISAACSIM_SITE="$ENV_PREFIX/lib/python3.10/site-packages/isaacsim"
GPU_FOUNDATION_DEPS="$ISAACSIM_SITE/extscache/omni.gpu_foundation/bin/deps"
TORCH_LIB="$ENV_PREFIX/lib/python3.10/site-packages/torch/lib"

export CONDA_PREFIX="$ENV_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export ISAACLAB_PATH="$ISAACLAB_ROOT"
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/.cache/isaac_sim}"
export TMPDIR="${TMPDIR:-$ROOT_DIR/.tmp/isaac_sim}"
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
unset PYTHONPATH
export LD_LIBRARY_PATH="$GPU_FOUNDATION_DEPS:$TORCH_LIB"

exec "$CONDA_PREFIX/bin/python" \
    "$ROOT_DIR/scripts/probe_psm_side_grasp_kinematics.py" \
    --output "$ROOT_DIR/outputs/psm_side_grasp_kinematic_candidates.json" "$@"
