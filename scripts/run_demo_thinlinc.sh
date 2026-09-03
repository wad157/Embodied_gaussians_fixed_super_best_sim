#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
VGL_ROOT="${VGL_ROOT:-$ROOT_DIR/tools/virtualgl/vendor/virtualgl-3.1.4}"
BUNDLED_NVIDIA_GL_ROOT="$ROOT_DIR/tools/virtualgl/vendor/nvidia-535.104.12"
SYSTEM_NVIDIA_GL_ROOT="${SYSTEM_NVIDIA_GL_ROOT:-/usr/lib/x86_64-linux-gnu}"
SYSTEM_NVIDIA_EGL_VENDOR_JSON="${SYSTEM_NVIDIA_EGL_VENDOR_JSON:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
NVIDIA_GL_ROOT="${NVIDIA_GL_ROOT:-}"
NVIDIA_GL_VERSION="${NVIDIA_GL_VERSION:-}"
NVIDIA_EGL_VENDOR_JSON="${NVIDIA_EGL_VENDOR_JSON:-}"
NVIDIA_RUNTIME_SOURCE=""
VGL_DEVICE="${VGL_DEVICE:-egl0}"
USE_VIRTUALGL="${USE_VIRTUALGL:-1}"
VGL_BIN="$VGL_ROOT/opt/VirtualGL/bin"
VGL_LIB="$VGL_ROOT/usr/lib"
TORCH_LIB="$ENV_PREFIX/lib/python3.11/site-packages/torch/lib"

detect_nvidia_kernel_version() {
    [[ -r /proc/driver/nvidia/version ]] || return 1
    awk '
        NR == 1 {
            for (i = 1; i <= NF; ++i) {
                if ($i ~ /^[0-9]+\.[0-9]+\.[0-9]+$/) {
                    print $i
                    exit
                }
            }
        }
    ' /proc/driver/nvidia/version
}

select_nvidia_runtime() {
    local kernel_version required_library
    kernel_version="$(detect_nvidia_kernel_version || true)"
    if [[ -z "$kernel_version" ]]; then
        echo "ERROR: Cannot detect the loaded NVIDIA kernel-module version." >&2
        echo "Check /proc/driver/nvidia/version and ask the host administrator to load the driver." >&2
        exit 1
    fi

    if [[ -z "$NVIDIA_GL_VERSION" ]]; then
        NVIDIA_GL_VERSION="$kernel_version"
    fi
    if [[ "$NVIDIA_GL_VERSION" != "$kernel_version" ]]; then
        echo "ERROR: NVIDIA OpenGL runtime does not match the loaded kernel module." >&2
        echo "Requested userspace: $NVIDIA_GL_VERSION" >&2
        echo "Loaded kernel:      $kernel_version" >&2
        exit 1
    fi

    if [[ -z "$NVIDIA_GL_ROOT" ]]; then
        if [[ -f "$SYSTEM_NVIDIA_GL_ROOT/libEGL_nvidia.so.$NVIDIA_GL_VERSION" ]]; then
            NVIDIA_GL_ROOT="$SYSTEM_NVIDIA_GL_ROOT"
            NVIDIA_RUNTIME_SOURCE="system GLVND"
        elif [[ -f "$BUNDLED_NVIDIA_GL_ROOT/libEGL_nvidia.so.$NVIDIA_GL_VERSION" ]]; then
            NVIDIA_GL_ROOT="$BUNDLED_NVIDIA_GL_ROOT"
            NVIDIA_RUNTIME_SOURCE="repository bundle"
        else
            echo "ERROR: No NVIDIA OpenGL runtime matches kernel $NVIDIA_GL_VERSION." >&2
            echo "Checked system: $SYSTEM_NVIDIA_GL_ROOT" >&2
            echo "Checked bundle: $BUNDLED_NVIDIA_GL_ROOT" >&2
            echo "Install matching NVIDIA userspace libraries or set NVIDIA_GL_ROOT explicitly." >&2
            exit 1
        fi
    else
        NVIDIA_RUNTIME_SOURCE="explicit NVIDIA_GL_ROOT"
    fi

    if [[ -z "$NVIDIA_EGL_VENDOR_JSON" ]]; then
        if [[ -f "$NVIDIA_GL_ROOT/10_nvidia.json" ]]; then
            NVIDIA_EGL_VENDOR_JSON="$NVIDIA_GL_ROOT/10_nvidia.json"
        else
            NVIDIA_EGL_VENDOR_JSON="$SYSTEM_NVIDIA_EGL_VENDOR_JSON"
        fi
    fi

    for required_library in \
        "libEGL_nvidia.so.$NVIDIA_GL_VERSION" \
        "libGLX_nvidia.so.$NVIDIA_GL_VERSION" \
        "libnvidia-eglcore.so.$NVIDIA_GL_VERSION" \
        "libnvidia-glcore.so.$NVIDIA_GL_VERSION"; do
        if [[ ! -f "$NVIDIA_GL_ROOT/$required_library" ]]; then
            echo "ERROR: Matching NVIDIA library is missing: $NVIDIA_GL_ROOT/$required_library" >&2
            exit 1
        fi
    done
    if [[ ! -f "$NVIDIA_EGL_VENDOR_JSON" ]]; then
        echo "ERROR: NVIDIA EGL vendor manifest is missing: $NVIDIA_EGL_VENDOR_JSON" >&2
        exit 1
    fi
}

detect_thinlinc_display() {
    local socket display vendor
    for socket in /tmp/.X11-unix/X*; do
        [[ -S "$socket" && -O "$socket" ]] || continue
        display=":${socket##*/X}"
        vendor="$(
            xdpyinfo -display "$display" 2>/dev/null \
                | sed -n 's/^vendor string:[[:space:]]*//p' \
                | head -n 1
        )"
        if [[ "$vendor" == *"ThinLinc"* ]]; then
            printf '%s\n' "$display"
            return 0
        fi
    done
    return 1
}

if [[ "${1:-}" == "--check-nvidia-runtime" ]]; then
    select_nvidia_runtime
    echo "NVIDIA kernel/OpenGL version: $NVIDIA_GL_VERSION"
    echo "NVIDIA OpenGL source:         $NVIDIA_RUNTIME_SOURCE"
    echo "NVIDIA OpenGL library root:   $NVIDIA_GL_ROOT"
    echo "NVIDIA EGL vendor manifest:   $NVIDIA_EGL_VENDOR_JSON"
    exit 0
fi

if [[ -z "${DISPLAY:-}" ]]; then
    DISPLAY="$(detect_thinlinc_display || true)"
    if [[ -z "$DISPLAY" ]]; then
        echo "ERROR: DISPLAY is empty and no ThinLinc display was detected." >&2
        exit 1
    fi
    export DISPLAY
    echo "[run_demo_thinlinc] detected ThinLinc DISPLAY=$DISPLAY"
fi

if [[ "$USE_VIRTUALGL" != "0" ]]; then
    select_nvidia_runtime
fi

if [[ "$USE_VIRTUALGL" != "0" && -z "${VGL_ISACTIVE:-}" ]]; then
    if [[ ! -x "$VGL_BIN/vglrun" ]]; then
        echo "ERROR: Local VirtualGL is missing: $VGL_BIN/vglrun" >&2
        echo "See $ROOT_DIR/tools/virtualgl/README.md" >&2
        exit 1
    fi

    export __EGL_VENDOR_LIBRARY_FILENAMES="$NVIDIA_EGL_VENDOR_JSON"
    export LD_LIBRARY_PATH="$NVIDIA_GL_ROOT:${LD_LIBRARY_PATH:-}"
    echo "[run_demo_thinlinc] NVIDIA OpenGL $NVIDIA_GL_VERSION from $NVIDIA_RUNTIME_SOURCE"
    echo "[run_demo_thinlinc] VirtualGL $VGL_DEVICE -> DISPLAY=$DISPLAY"
    exec "$VGL_BIN/vglrun" \
        -ld "$VGL_LIB" \
        -d "$VGL_DEVICE" \
        bash "$SCRIPT_PATH" "$@"
fi

if [[ "${1:-}" == "--check-virtualgl" ]]; then
    if [[ -z "${VGL_ISACTIVE:-}" ]]; then
        echo "ERROR: VirtualGL is not active." >&2
        exit 1
    fi
    exec "$VGL_BIN/glxinfo" -B
fi

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    echo "ERROR: Python not found: $ENV_PREFIX/bin/python" >&2
    exit 1
fi

export CONDA_PREFIX="$ENV_PREFIX"
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/usr/bin:$CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/examples:$CONDA_PREFIX/usr/lib/python3/dist-packages:${PYTHONPATH:-}"
if [[ -n "${VGL_ISACTIVE:-}" ]]; then
    export LD_LIBRARY_PATH="$VGL_LIB:$NVIDIA_GL_ROOT:$TORCH_LIB:$CONDA_PREFIX/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
else
    export LD_LIBRARY_PATH="$TORCH_LIB:$CONDA_PREFIX/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:${LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/Media_HDD/jwshan/tmp/torch_extensions}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-2}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
cd "$ROOT_DIR"

exec "$CONDA_PREFIX/bin/python" examples/example_embodied_super_offline.py \
    --fps 30 \
    --visual-feedback-mode residual \
    --visual-force-iterations 1 \
    --visual-residual-iterations 8 \
    --visual-residual-learning-rate-m 0.000040 \
    --visual-force-update-interval 3 \
    --online-stiffness-update \
    --stiffness-log-learning-rate 0.18 \
    --cameras stereo_left,stereo_right \
    --camera-go-zoom 0.9 \
    --psm-pose-driver raw_paper_lnd_sam2_dense_contact_unbounded_xyz \
    --tissue-mode paper_pbd \
    --psm-tissue-contact \
    --psm-visual-mode full \
    --psm-roll-offset-deg 0 \
    --psm-camera-translation-mm 0 0 0 \
    --psm-world-translation-mm 0 0 0 \
    "$@"
