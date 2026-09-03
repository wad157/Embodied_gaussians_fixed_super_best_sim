#!/usr/bin/env python3
"""汇总平面内拉扯和夹起拉升两套H1安全长期刚度测评。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CAPABILITIES = ("reconstruction_7to1", "future_80to20")
METHODS = (
    "pbd",
    "pbd_cotracker_foundation_depth",
    "pbd_cotracker_foundation_depth_hierarchical_stiffness_h1_safe",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inplane", type=Path, required=True)
    parser.add_argument("--lift", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    datasets = (
        ("平面内拉扯", load(args.inplane)),
        ("夹起拉升", load(args.lift)),
    )
    lines = [
        "# 两套任务：H1短期约束下的H3/H5长期刚度更新",
        "",
        "两套任务均不做首帧对齐；状态观测只使用CoTracker RGB轨迹与FoundationStereo RGB深度。",
        "",
        "| 数据集 | 能力 | 方法 | 3D Mean mm↓ | 2D Mean px↓ | PSNR↑ | SSIM↑ | LPIPS↓ |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset_label, payload in datasets:
        for capability in CAPABILITIES:
            for method in METHODS:
                row = payload["results"][capability][method]
                capability_label = (
                    "7:1重建"
                    if capability == "reconstruction_7to1"
                    else "80/20预测"
                )
                lines.append(
                    f"| {dataset_label} | {capability_label} | {row['label']} | "
                    f"{row['3d_mean_mm']:.4f} | {row['2d_mean_px']:.3f} | "
                    f"{row['psnr_db']:.3f} | {row['ssim']:.4f} | "
                    f"{row['lpips']:.4f} |"
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
