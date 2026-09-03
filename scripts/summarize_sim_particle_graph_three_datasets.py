#!/usr/bin/env python3
"""汇总三套独立数据上的固定 Particle Graph-LM 测评。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CAPABILITIES = ("reconstruction_7to1", "future_80to20")
METHODS = (
    "pbd",
    "pbd_cotracker_foundation_depth",
    "pbd_cotracker_foundation_depth_particle_graph_lm",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim01", type=Path, required=True)
    parser.add_argument("--sim02", type=Path, required=True)
    parser.add_argument("--sim03", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    datasets = (
        ("SIM-01 Planar-X-Pull", args.sim01),
        ("SIM-02 Planar-Y-Pull", args.sim02),
        ("SIM-03 Edge-Z-Lift", args.sim03),
    )
    lines = [
        "# Particle Graph-LM 三数据集独立测评",
        "",
        "三套数据使用各自的 FoundationStereo/CoTracker 资产、夹持边界和30个非夹持评估点；方法与固定初值完全一致。",
        "",
        "| 数据集 | 能力 | 方法 | 3D Mean mm↓ | 2D Mean px↓ | PSNR↑ | SSIM↑ | LPIPS↓ |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset_label, path in datasets:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for capability in CAPABILITIES:
            capability_label = "7:1重建" if capability == "reconstruction_7to1" else "80/20预测"
            for method in METHODS:
                row = payload["results"][capability][method]
                lines.append(
                    f"| {dataset_label} | {capability_label} | {row['label']} | "
                    f"{row['3d_mean_mm']:.4f} | {row['2d_mean_px']:.3f} | "
                    f"{row['psnr_db']:.3f} | {row['ssim']:.4f} | "
                    f"{row['lpips']:.4f} |"
                )
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()

