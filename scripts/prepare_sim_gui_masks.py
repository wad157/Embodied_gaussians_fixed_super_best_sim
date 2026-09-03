#!/usr/bin/env python3
"""把仿真逐帧 PNG 真值 mask 转为现有 GUI 可直接读取的压缩格式。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


CAMERAS = ("stereo_left", "stereo_right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    return parser.parse_args()


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0


def ordered_paths(root: Path, expected_count: int) -> list[Path]:
    paths = sorted(root.glob("*.png"))
    expected_names = [f"{index:06d}.png" for index in range(expected_count)]
    if [path.name for path in paths] != expected_names:
        raise ValueError(f"mask 序列不连续：{root}")
    return paths


def pack_camera_masks(
    source: Path,
    output: Path,
    timestamps: np.ndarray,
    resolution_wh: tuple[int, int],
) -> np.ndarray:
    width, height = resolution_wh
    paths = ordered_paths(source, len(timestamps))
    packed = np.empty((len(paths), height, (width + 7) // 8), dtype=np.uint8)
    foreground_pixels = np.empty(len(paths), dtype=np.int64)
    for index, path in enumerate(paths):
        mask = load_mask(path)
        if mask.shape != (height, width):
            raise ValueError(f"{path} 分辨率 {mask.shape} != {(height, width)}")
        packed[index] = np.packbits(mask, axis=1, bitorder="big")
        foreground_pixels[index] = int(mask.sum())
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "timestamps.npy", timestamps)
    np.save(output / "tissue_masks_packbits.npy", packed)
    report = {
        "schema": "fixedsuperbest.sim_png_tissue_masks.v1",
        "source": str(source),
        "resolution_wh": list(resolution_wh),
        "frames": int(len(paths)),
        "foreground_pixels_min": int(foreground_pixels.min()),
        "foreground_pixels_max": int(foreground_pixels.max()),
        "passed": bool(np.all(foreground_pixels > 0)),
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not report["passed"]:
        raise RuntimeError(f"组织 mask 门禁失败：{source}")
    return packed


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    gui_root = dataset / "gui_assets"
    if not (gui_root / "tissue_fixedsuperbest.npz").is_file():
        raise FileNotFoundError("请先生成 tissue_fixedsuperbest.npz")
    output_root = gui_root / "visual_force_masks"
    instrument_output = gui_root / "instrument_masks.npz"
    if output_root.exists() or instrument_output.exists():
        raise FileExistsError("拒绝覆盖已有 GUI mask 资产")

    metadata = {
        camera: json.loads(
            (dataset / "videos" / f"{camera}.json").read_text(encoding="utf-8")
        )
        for camera in CAMERAS
    }
    reference = metadata[CAMERAS[0]]
    resolution_wh = tuple(int(value) for value in reference["resolution"])
    timestamps = np.asarray(reference["timestamps"], dtype=np.float64)
    for camera in CAMERAS[1:]:
        current = metadata[camera]
        if tuple(current["resolution"]) != resolution_wh:
            raise ValueError("双目视频分辨率不一致")
        if not np.array_equal(
            np.asarray(current["timestamps"], dtype=np.float64), timestamps
        ):
            raise ValueError("双目时间戳不一致")

    tissue_outputs: dict[str, str] = {}
    psm_packed: dict[str, np.ndarray] = {}
    for camera in CAMERAS:
        camera_output = output_root / camera
        pack_camera_masks(
            dataset / "ground_truth" / "masks" / "tissue" / camera,
            camera_output,
            timestamps,
            resolution_wh,
        )
        tissue_outputs[camera] = str(camera_output.relative_to(dataset))
        psm_paths = ordered_paths(
            dataset / "ground_truth" / "masks" / "psm" / camera,
            len(timestamps),
        )
        width, height = resolution_wh
        packed = np.empty(
            (len(psm_paths), (height * width + 7) // 8), dtype=np.uint8
        )
        for index, path in enumerate(psm_paths):
            mask = load_mask(path)
            if mask.shape != (height, width):
                raise ValueError(f"{path} 分辨率错误：{mask.shape}")
            packed[index] = np.packbits(mask.reshape(-1), bitorder="big")
        psm_packed[camera] = packed

    np.savez_compressed(
        instrument_output,
        mask_shape=np.asarray((resolution_wh[1], resolution_wh[0]), dtype=np.int32),
        bitorder=np.asarray("big"),
        quality_valid=np.ones(len(timestamps), dtype=bool),
        left_masks_packbits=psm_packed["stereo_left"],
        right_masks_packbits=psm_packed["stereo_right"],
        left_distal_masks_packbits=psm_packed["stereo_left"],
        right_distal_masks_packbits=psm_packed["stereo_right"],
        stereo_left_index=np.arange(len(timestamps), dtype=np.int64),
        stereo_right_index=np.arange(len(timestamps), dtype=np.int64),
    )
    report = {
        "schema": "fixedsuperbest.sim_gui_masks.v1",
        "说明": "仿真真值 mask 仅用于无分割误差的视觉残差评估。",
        "frames": int(len(timestamps)),
        "resolution_wh": list(resolution_wh),
        "tissue_assets": tissue_outputs,
        "instrument_asset": str(instrument_output.relative_to(dataset)),
        "passed": True,
    }
    (gui_root / "visual_force_masks_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
