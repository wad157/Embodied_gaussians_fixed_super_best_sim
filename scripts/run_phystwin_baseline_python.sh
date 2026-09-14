#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${PHYSTWIN_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/phystwin_sim}"
UPSTREAM_ROOT="${PHYSTWIN_ROOT:-$ROOT_DIR/baselines/PhysTwin}"
EXTENSION_ROOT="${PHYSTWIN_EXTENSION_ROOT:-$ROOT_DIR/.cache/phystwin_torch_extensions_v2}"

if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
    echo "ERROR: 找不到 PhysTwin Python：$ENV_ROOT/bin/python" >&2
    exit 1
fi
if [[ ! -d "$UPSTREAM_ROOT/.git" ]]; then
    echo "ERROR: 找不到 PhysTwin checkout：$UPSTREAM_ROOT" >&2
    exit 1
fi

export CUDA_HOME="$ENV_ROOT"
export CUDA_PATH="$ENV_ROOT"
export CC="${CC:-/usr/bin/gcc-11}"
export CXX="${CXX:-/usr/bin/g++-11}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++-11}"
export CPATH="$ENV_ROOT/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$ENV_ROOT/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export PATH="$ENV_ROOT/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_EXTENSIONS_DIR="$EXTENSION_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/phystwin_mpl}"
export PYTHONPATH="$ROOT_DIR/baselines/phystwin_sim:$UPSTREAM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$EXTENSION_ROOT" "$MPLCONFIGDIR"

exec "$ENV_ROOT/bin/python" "$@"
