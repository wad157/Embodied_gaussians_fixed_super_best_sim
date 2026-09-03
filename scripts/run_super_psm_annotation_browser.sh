#!/usr/bin/env bash
set -euo pipefail

# Launch the paper's first-frame annotator on the SUPER grasp5 left video.
# The annotation files are kept inside this repository, while the paper source
# and its isolated Python environment remain external and read-only.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_REPO="${PAPER_REPO:-/Media_HDD/jwshan/wad/online_dvrk_tracking}"
TRACK_ENV="${TRACK_ENV:-/Media_HDD/jwshan/conda_envs/online_dvrk}"
DISPLAY_STACK_SCRIPT="${DISPLAY_STACK_SCRIPT:-/home/jwshan/hsieh_codes/wad/embodied_gaussians_fixed_super_fin/scripts/start_display_browser.sh}"
DISPLAY_NUM="${DISPLAY_NUM:-12}"
TRACK_ROOT="${TRACK_ROOT:-$ROOT_DIR/data/super/psm_tracking}"
VIDEO_DIR="$TRACK_ROOT/online_videos/grasp5"
SOURCE_VIDEO="$ROOT_DIR/data/super/grasp5_offline_demo/videos/stereo_left.mp4"

for required in \
    "$PAPER_REPO/scripts/video_annotator.py" \
    "$PAPER_REPO/SurgicalSAM2/checkpoints/sam2.1_hiera_s_endo18.pth" \
    "$TRACK_ENV/bin/python" \
    "$DISPLAY_STACK_SCRIPT" \
    "$SOURCE_VIDEO"; do
    if [[ ! -e "$required" ]]; then
        echo "Missing required path: $required" >&2
        exit 1
    fi
done

mkdir -p "$VIDEO_DIR"
ln -sfn "$SOURCE_VIDEO" "$VIDEO_DIR/video.mp4"

# The source video is 1920x1080. A same-sized virtual screen keeps all OpenCV
# instructions and the ENTER-to-save control visible in noVNC.
GEOMETRY="${GEOMETRY:-1920x1080x24}" \
DISPLAY_NUM="$DISPLAY_NUM" \
bash "$DISPLAY_STACK_SCRIPT"

export DISPLAY=":$DISPLAY_NUM"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.2}"
export PATH="$TRACK_ENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$TRACK_ENV/lib:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/Media_HDD/jwshan/tmp/torch_extensions_online_dvrk}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-2}"

cd "$PAPER_REPO"
exec "$TRACK_ENV/bin/python" scripts/video_annotator.py \
    --video_path "$TRACK_ROOT/online_videos" \
    --video_name video.mp4 \
    --idx grasp5 \
    --machine_label PSM1 \
    --downsample_factor 2 \
    --sam2_checkpoint "$PAPER_REPO/SurgicalSAM2/checkpoints/sam2.1_hiera_s_endo18.pth" \
    --model_cfg configs/sam2.1/sam2.1_hiera_s.yaml
