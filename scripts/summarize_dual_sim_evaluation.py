#!/usr/bin/env python3
"""把正面/侧面两套三组消融结果合并成一张中文总表。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", type=Path, required=True)
    parser.add_argument("--side", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for task, path in (("正面牵拉", args.front), ("侧面牵拉", args.side)):
        report = json.loads(path.read_text(encoding="utf-8"))
        for row in report["rows"]:
            if row["segment"] in {
                "assimilation_0_79_percent",
                "future_open_loop_20_percent",
            }:
                rows.append((task, row))
    lines = [
        "# 两套无硬固定组织牵拉评估总表",
        "",
        "两套任务均使用同一材质、相机、纹理、固定初始参数和因果在线算法。前 80% 允许 RGB 状态/刚度更新，后 20% 冻结后开环预测；不使用深度、材料真值、人工选参或分支选择。",
        "",
        "| 任务 | 方法 | 区间 | 3D Mean mm ↓ | 2D Mean px ↓ | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for task, row in rows:
        lines.append(
            f"| {task} | {row['method']} | {row['segment_zh']} | "
            f"{row['track_3d_mean_mm']:.3f} | "
            f"{row['track_2d_stereo_mean_px']:.3f} | "
            f"{row['psnr_tissue_layer_db']:.3f} | "
            f"{row['ssim_tissue_layer']:.4f} | "
            f"{row['lpips_tissue_layer']:.4f} |"
        )
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
