#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ENV="${EH_SURGS_SOURCE_ENV:-/Media_HDD/jwshan/conda_envs/endogaussian_baseline}"
ENV_ROOT="${EH_SURGS_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eh_surgs_baseline}"
UPSTREAM_ROOT="${EH_SURGS_ROOT:-$ROOT_DIR/baselines/EH-SurGS}"
CUDA_TOOLKIT="${EH_SURGS_CUDA_ROOT:-/Media_HDD/jwshan/conda_envs/EndoGaussian}"
UPSTREAM_URL="https://github.com/IRMVLab/EH-SurGS.git"
UPSTREAM_COMMIT="73fa04e6f5c21cc1685f728eccb1332e81ce620c"
CAMERA_PATCH="$ROOT_DIR/baselines/eh_surgs_sim/patches/train_camera_intrinsics.patch"
CAMERA_PATCH_SHA256="f75e5d151e7f885024e6c75ccc3a1457ebc86389d8277d433cae58175bc310d8"

if [[ ! -x "$SOURCE_ENV/bin/python" ]]; then
    echo "ERROR: 找不到用于离线克隆的 Python 3.7 环境：$SOURCE_ENV" >&2
    exit 1
fi
if [[ ! -d "$UPSTREAM_ROOT/.git" ]]; then
    if [[ -e "$UPSTREAM_ROOT" ]]; then
        echo "ERROR: 上游路径已存在但不是 Git checkout：$UPSTREAM_ROOT" >&2
        exit 1
    fi
    git clone "$UPSTREAM_URL" "$UPSTREAM_ROOT"
    git -C "$UPSTREAM_ROOT" checkout --detach "$UPSTREAM_COMMIT"
fi
if [[ "$(git -C "$UPSTREAM_ROOT" rev-parse HEAD)" != "$UPSTREAM_COMMIT" ]]; then
    echo "ERROR: EH-SurGS checkout 不是固定提交 $UPSTREAM_COMMIT" >&2
    exit 1
fi
if git -C "$UPSTREAM_ROOT" diff --quiet -- train.py; then
    git -C "$UPSTREAM_ROOT" apply "$CAMERA_PATCH"
fi
actual_patch_sha256="$(git -C "$UPSTREAM_ROOT" diff -- train.py | sha256sum | awk '{print $1}')"
if [[ "$actual_patch_sha256" != "$CAMERA_PATCH_SHA256" ]]; then
    echo "ERROR: EH-SurGS 相机补丁不匹配：$actual_patch_sha256" >&2
    exit 1
fi
if [[ ! -x "$CUDA_TOOLKIT/bin/nvcc" ]]; then
    echo "ERROR: 找不到一致的 CUDA 11.8 编译工具链：$CUDA_TOOLKIT" >&2
    exit 1
fi
if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
    conda create --prefix "$ENV_ROOT" --clone "$SOURCE_ENV" --yes
fi

BUILD_ROOT="$(mktemp -d /tmp/eh_surgs_extensions.XXXXXX)"
trap 'rm -rf "$BUILD_ROOT"' EXIT
cp -a "$UPSTREAM_ROOT/submodules/simple-knn" "$BUILD_ROOT/simple-knn"
cp -a "$UPSTREAM_ROOT/submodules/depth-diff-gaussian-rasterization" \
    "$BUILD_ROOT/depth-diff-gaussian-rasterization"
find "$BUILD_ROOT" -type d -name build -prune -exec rm -rf {} +
find "$BUILD_ROOT" -type d -name '*.egg-info' -prune -exec rm -rf {} +
find "$BUILD_ROOT" -type f -name '*.so' -delete

clean_build_env=(
    env
    -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS -u CPP
    -u CPATH -u CPLUS_INCLUDE_PATH -u LIBRARY_PATH -u PYTHONPATH
    CC=/usr/bin/gcc-11
    CXX=/usr/bin/g++-11
    CUDA_HOME="$CUDA_TOOLKIT"
    CUDA_PATH="$CUDA_TOOLKIT"
    MAX_JOBS="${MAX_JOBS:-8}"
    PATH="$CUDA_TOOLKIT/bin:$ENV_ROOT/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    PYTHONNOUSERSITE=1
)

"${clean_build_env[@]}" "$ENV_ROOT/bin/pip" uninstall -y \
    diff-gaussian-rasterization simple-knn || true
"${clean_build_env[@]}" "$ENV_ROOT/bin/pip" install --no-deps --no-build-isolation \
    "$BUILD_ROOT/simple-knn"
"${clean_build_env[@]}" "$ENV_ROOT/bin/pip" install --no-deps --no-build-isolation \
    "$BUILD_ROOT/depth-diff-gaussian-rasterization"

CUDA_VISIBLE_DEVICES="${EH_SURGS_GPU_ID:-0}" \
    PYTHONNOUSERSITE=1 \
    LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}" \
    "$ENV_ROOT/bin/python" -c \
    'import torch; from simple_knn._C import distCUDA2; import diff_gaussian_rasterization as d; x=torch.randn(5000,3,device="cuda"); y=distCUDA2(x); assert y.shape==(5000,); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print(d.__file__); print("distCUDA2", y.shape, int(torch.cuda.memory_allocated()))'
