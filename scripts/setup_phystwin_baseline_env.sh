#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ENV="${PHYSTWIN_SOURCE_ENV:-/Media_HDD/jwshan/conda_envs/eg_codex}"
ENV_ROOT="${PHYSTWIN_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/phystwin_sim}"
UPSTREAM_ROOT="${PHYSTWIN_ROOT:-$ROOT_DIR/baselines/PhysTwin}"
UPSTREAM_URL="https://github.com/jianghanxiao/phystwin.git"
UPSTREAM_COMMIT="81c718790a37e5e0102eb77af2c6edd34a9db25f"
HEADLESS_PATCH="$ROOT_DIR/baselines/phystwin_sim/patches/headless_imports.patch"
EXPECTED_PATCH_SHA256="4ad745d0baf8410672b95817f529bb4d9ac0cfbfe45bc6cb1626d0d08df693b8"

if [[ ! -x "$SOURCE_ENV/bin/python" ]]; then
    echo "ERROR: 找不到用于克隆的已有 baseline 环境：$SOURCE_ENV" >&2
    exit 1
fi
if [[ ! -d "$UPSTREAM_ROOT/.git" ]]; then
    if [[ -e "$UPSTREAM_ROOT" ]]; then
        echo "ERROR: 上游路径已存在但不是 Git checkout：$UPSTREAM_ROOT" >&2
        exit 1
    fi
    git clone --recursive "$UPSTREAM_URL" "$UPSTREAM_ROOT"
fi
if [[ "$(git -C "$UPSTREAM_ROOT" rev-parse HEAD)" != "$UPSTREAM_COMMIT" ]]; then
    git -C "$UPSTREAM_ROOT" fetch origin
    git -C "$UPSTREAM_ROOT" checkout --detach "$UPSTREAM_COMMIT"
    git -C "$UPSTREAM_ROOT" submodule update --init --recursive
fi

actual_patch_sha256="$(git -C "$UPSTREAM_ROOT" diff -- qqtt/__init__.py qqtt/utils/__init__.py | sha256sum | awk '{print $1}')"
if [[ -z "$(git -C "$UPSTREAM_ROOT" diff -- qqtt/__init__.py qqtt/utils/__init__.py)" ]]; then
    git -C "$UPSTREAM_ROOT" apply "$HEADLESS_PATCH"
    actual_patch_sha256="$(git -C "$UPSTREAM_ROOT" diff -- qqtt/__init__.py qqtt/utils/__init__.py | sha256sum | awk '{print $1}')"
fi
if [[ "$actual_patch_sha256" != "$EXPECTED_PATCH_SHA256" ]]; then
    echo "ERROR: PhysTwin 仅允许固定 headless import 补丁：$actual_patch_sha256" >&2
    exit 1
fi
if [[ -n "$(git -C "$UPSTREAM_ROOT" diff --name-only -- . ':(exclude)qqtt/__init__.py' ':(exclude)qqtt/utils/__init__.py')" ]]; then
    echo "ERROR: PhysTwin 上游存在未授权的已跟踪源码修改" >&2
    exit 1
fi

if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
    conda create --prefix "$ENV_ROOT" --clone "$SOURCE_ENV" --yes
fi
"$ENV_ROOT/bin/pip" install -r "$ROOT_DIR/baselines/phystwin_sim/requirements.lock.txt"

CUDA_VISIBLE_DEVICES="${PHYSTWIN_GPU_ID:-0}" \
    bash "$ROOT_DIR/scripts/run_phystwin_baseline_python.sh" -c \
    'import cma, gsplat, open3d, torch, warp; from qqtt.model.diff_simulator import SpringMassSystemWarp; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print("warp", warp.__version__, "gsplat", gsplat.__version__)'
