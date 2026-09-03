#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_sim}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/Media_HDD/jwshan/wad/sufia_sim/IsaacLab}"
ISAACSIM_SITE="$ENV_PREFIX/lib/python3.10/site-packages/isaacsim"
GPU_FOUNDATION_DEPS="$ISAACSIM_SITE/extscache/omni.gpu_foundation/bin/deps"
TORCH_LIB="$ENV_PREFIX/lib/python3.10/site-packages/torch/lib"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    echo "错误：未找到 eg_sim Python：$ENV_PREFIX/bin/python" >&2
    exit 1
fi
if [[ ! -f "$ROOT_DIR/scripts/generate_sim_tissue_retraction_dataset.py" ]]; then
    echo "错误：数据集生成器不存在。" >&2
    exit 1
fi
if [[ ! -f "$ISAACLAB_ROOT/source/apps/isaaclab.python.headless.rendering.kit" ]]; then
    echo "错误：未找到 Isaac Lab 渲染 experience：$ISAACLAB_ROOT" >&2
    exit 1
fi
if [[ ! -f "$GPU_FOUNDATION_DEPS/libslang-glslang.so" ]]; then
    echo "错误：Isaac Sim Slang 运行库不存在：$GPU_FOUNDATION_DEPS" >&2
    exit 1
fi

export CONDA_PREFIX="$ENV_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export ISAACLAB_PATH="$ISAACLAB_ROOT"
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/.cache/isaac_sim}"
# 系统 /tmp 空间很紧张；Isaac Sim 长序列渲染统一使用项目盘，避免中途被终止。
export TMPDIR="$ROOT_DIR/.tmp/isaac_sim"
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"

# 允许从 eg_codex shell 调用，但不混入它的 Python/C++ 运行库。
unset PYTHONPATH
export LD_LIBRARY_PATH="$GPU_FOUNDATION_DEPS:$TORCH_LIB"

exec "$CONDA_PREFIX/bin/python" \
    "$ROOT_DIR/scripts/generate_sim_tissue_retraction_dataset.py" "$@"
