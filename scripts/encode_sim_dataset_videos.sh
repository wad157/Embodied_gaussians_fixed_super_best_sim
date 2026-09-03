#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "用法：$0 数据集目录" >&2
    exit 2
fi

DATASET_DIR="$(readlink -f "$1")"
FFMPEG_BIN="${FFMPEG_BIN:-/Media_HDD/jwshan/conda_envs/eg_codex/bin/ffmpeg}"
if [[ ! -f "$DATASET_DIR/episode.json" || ! -d "$DATASET_DIR/rgb" ]]; then
    echo "错误：不是本项目生成的仿真数据集：$DATASET_DIR" >&2
    exit 1
fi
if [[ ! -x "$FFMPEG_BIN" ]]; then
    echo "错误：未找到 ffmpeg：$FFMPEG_BIN" >&2
    exit 1
fi

FPS="$(sed -n 's/.*"fps": \([0-9][0-9]*\).*/\1/p' "$DATASET_DIR/episode.json" | head -n 1)"
if [[ -z "$FPS" ]]; then
    echo "错误：无法从 episode.json 读取 fps。" >&2
    exit 1
fi

for CAMERA in stereo_left stereo_right; do
    INPUT_PATTERN="$DATASET_DIR/rgb/$CAMERA/%06d.png"
    OUTPUT_VIDEO="$DATASET_DIR/videos/$CAMERA.mp4"
    LOSSLESS_VIDEO="$DATASET_DIR/videos/${CAMERA}_lossless_rgb.mp4"
    if [[ ! -f "$DATASET_DIR/rgb/$CAMERA/000000.png" ]]; then
        echo "错误：$CAMERA 缺少第 0 帧。" >&2
        exit 1
    fi
    if [[ -e "$OUTPUT_VIDEO" || -e "$LOSSLESS_VIDEO" ]]; then
        echo "错误：拒绝覆盖已有视频：$OUTPUT_VIDEO 或 $LOSSLESS_VIDEO" >&2
        exit 1
    fi
    "$FFMPEG_BIN" -hide_banner -loglevel warning \
        -framerate "$FPS" -start_number 0 -i "$INPUT_PATTERN" \
        -an -c:v libx264rgb -crf 0 -preset medium -pix_fmt rgb24 "$LOSSLESS_VIDEO"
    # GUI/ThinLinc 版使用兼容性最好的 H.264 High + yuv420p，并把 moov atom
    # 前置以支持边生成边加载。无损 RGB MP4 仍作为定量归档；评估始终以 PNG 为准。
    "$FFMPEG_BIN" -hide_banner -loglevel warning \
        -framerate "$FPS" -start_number 0 -i "$INPUT_PATTERN" \
        -an -c:v libx264 -crf 8 -preset medium -profile:v high \
        -pix_fmt yuv420p -movflags +faststart "$OUTPUT_VIDEO"
    echo "[视频] GUI 兼容版：$OUTPUT_VIDEO"
    echo "[视频] 无损 RGB 归档：$LOSSLESS_VIDEO"
done
