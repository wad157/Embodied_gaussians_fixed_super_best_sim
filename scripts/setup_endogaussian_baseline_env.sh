#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${ENDOGAUSSIAN_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/endogaussian_baseline}"
UPSTREAM_ROOT="${ENDOGAUSSIAN_ROOT:-$ROOT_DIR/baselines/EndoGaussian}"
UPSTREAM_COMMIT="8d12793838a1595b299df0696c8149c07329e980"
GLM_COMMIT="6f14f4792a0cde5d0cf2c910506724d61cb95834"

if [[ ! -d "$UPSTREAM_ROOT/.git" ]]; then
    git clone https://github.com/CUHK-AIM-Group/EndoGaussian.git "$UPSTREAM_ROOT"
fi
git -C "$UPSTREAM_ROOT" fetch origin master
git -C "$UPSTREAM_ROOT" checkout --detach "$UPSTREAM_COMMIT"

GLM_ROOT="$UPSTREAM_ROOT/submodules/depth-diff-gaussian-rasterization/third_party/glm"
if [[ ! -d "$GLM_ROOT/.git" ]]; then
    git clone https://github.com/g-truc/glm.git "$GLM_ROOT"
fi
git -C "$GLM_ROOT" checkout --detach "$GLM_COMMIT"
printf '%s\n' '/submodules/depth-diff-gaussian-rasterization/third_party/glm/' \
    >>"$UPSTREAM_ROOT/.git/info/exclude"

if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
    conda create --prefix "$ENV_ROOT" --yes python=3.7 pip
fi
"$ENV_ROOT/bin/pip" install \
    torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
"$ENV_ROOT/bin/pip" install \
    mmcv==1.6.0 matplotlib==3.5.3 lpips==0.1.4 plyfile==0.9 \
    imageio-ffmpeg==0.5.1 imageio==2.31.2 open3d==0.17.0 \
    torchmetrics==0.11.4 opencv-python ninja
conda install --prefix "$ENV_ROOT" --yes -c nvidia cuda-nvcc=11.7

clean_build_env=(
    env
    -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS -u CPP
    -u CPATH -u CPLUS_INCLUDE_PATH -u LIBRARY_PATH
    CC=/usr/bin/gcc-11
    CXX=/usr/bin/g++-11
    CUDA_HOME="$ENV_ROOT"
    MAX_JOBS="${MAX_JOBS:-8}"
    PATH="$ENV_ROOT/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
"${clean_build_env[@]}" "$ENV_ROOT/bin/pip" uninstall -y \
    diff-gaussian-rasterization simple-knn || true
"${clean_build_env[@]}" "$ENV_ROOT/bin/pip" install --no-build-isolation \
    "$UPSTREAM_ROOT/submodules/simple-knn"
"${clean_build_env[@]}" "$ENV_ROOT/bin/pip" install --no-build-isolation \
    "$UPSTREAM_ROOT/submodules/depth-diff-gaussian-rasterization"
git -C "$UPSTREAM_ROOT" checkout -- \
    submodules/simple-knn/simple_knn/_C.cpython-37m-x86_64-linux-gnu.so \
    submodules/depth-diff-gaussian-rasterization/diff_gaussian_rasterization/_C.cpython-37m-x86_64-linux-gnu.so

bash "$ROOT_DIR/scripts/run_endogaussian_baseline_python.sh" -c \
    'import torch,diff_gaussian_rasterization,simple_knn; assert torch.__version__.split("+")[0] == "1.13.1"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print(diff_gaussian_rasterization.__file__)'
