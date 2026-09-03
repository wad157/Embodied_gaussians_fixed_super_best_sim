#!/usr/bin/env python3
"""汇总三套编号任务的独立H1安全长期刚度测评。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CAPABILITIES = ("reconstruction_7to1", "future_80to20")
BASE_METHODS = ("pbd", "pbd_cotracker_foundation_depth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim01", type=Path, required=True)
    parser.add_argument("--sim02", type=Path, required=True)
    parser.add_argument("--sim03", type=Path, required=True)
    parser.add_argument(
        "--method-c",
        default="pbd_cotracker_foundation_depth_hierarchical_stiffness_h1_safe",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    datasets = (
        ("SIM-01 Planar-X-Pull", load(args.sim01)),
        ("SIM-02 Planar-Y-Pull", load(args.sim02)),
        ("SIM-03 Edge-Z-Lift", load(args.sim03)),
    )
    lines = [
        "# SIM-01/02/03 独立测评总表",
        "",
        "三套数据分别生成RGB深度、CoTracker、夹持边界和评估清单；只在本表阶段汇总指标。",
        "",
        "| 数据集 | 能力 | 方法 | 3D Mean mm↓ | 2D Mean px↓ | PSNR↑ | SSIM↑ | LPIPS↓ |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset_label, payload in datasets:
        for capability in CAPABILITIES:
            capability_label = (
                "7:1重建"
                if capability == "reconstruction_7to1"
                else "80/20预测"
            )
            for method in (*BASE_METHODS, args.method_c):
                row = payload["results"][capability][method]
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
