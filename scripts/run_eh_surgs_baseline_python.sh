#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${EH_SURGS_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eh_surgs_baseline}"
UPSTREAM_ROOT="${EH_SURGS_ROOT:-$ROOT_DIR/baselines/EH-SurGS}"
ADAPTER_ROOT="$ROOT_DIR/baselines/eh_surgs_sim"
CUDA_TOOLKIT="${EH_SURGS_CUDA_ROOT:-/Media_HDD/jwshan/conda_envs/EndoGaussian}"

if [[ $# -lt 1 ]]; then
    echo "用法：bash scripts/run_eh_surgs_baseline_python.sh <python脚本> [参数...]" >&2
    exit 2
fi
if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
    echo "ERROR: 找不到 EH-SurGS 独立环境：$ENV_ROOT" >&2
    exit 1
fi

unset CC CXX CPP CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CPATH CPLUS_INCLUDE_PATH LIBRARY_PATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDA_HOME="$CUDA_TOOLKIT"
export CUDA_PATH="$CUDA_TOOLKIT"
export PATH="$CUDA_TOOLKIT/bin:$ENV_ROOT/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ADAPTER_ROOT:$UPSTREAM_ROOT"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"

exec "$ENV_ROOT/bin/python" "$@"
