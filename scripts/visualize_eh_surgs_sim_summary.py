#!/usr/bin/env python3
"""Plot EH-SurGS SIM three-repeat means with sample-standard-deviation bars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASETS = ("sim01", "sim02", "sim03")
LABELS = ("SIM-01", "SIM-02", "SIM-03")
CAPABILITIES = (
    ("reconstruction_7to1", "7:1 reconstruction"),
    ("future_80to20", "final-20% extrapolation"),
)
METRICS = (
    ("3d_mean_mm", "3D mean error (mm)", "lower is better"),
    ("2d_mean_px", "2D mean error (px)", "lower is better"),
    ("psnr_db", "Tissue-layer PSNR (dB)", "higher is better"),
    ("ssim", "Tissue-layer SSIM", "higher is better"),
    ("lpips", "Tissue-layer LPIPS", "lower is better"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("拒绝覆盖已有图：{}".format(output))
    payload = json.loads(args.summary.expanduser().resolve().read_text(encoding="utf-8"))
    aggregate = payload["aggregate"]
    colors = ("#4472C4", "#ED7D31", "#70AD47")
    fig, axes = plt.subplots(2, 5, figsize=(19, 7.5), constrained_layout=True)
    for row, (capability, capability_label) in enumerate(CAPABILITIES):
        for column, (metric, title, direction) in enumerate(METRICS):
            ax = axes[row, column]
            means = [aggregate[key][capability]["metrics"][metric]["mean"] for key in DATASETS]
            stds = [aggregate[key][capability]["metrics"][metric]["sample_std"] for key in DATASETS]
            x = np.arange(len(DATASETS))
            ax.bar(x, means, yerr=stds, color=colors, capsize=5, edgecolor="black", linewidth=0.6)
            ax.set_xticks(x, LABELS)
            ax.grid(axis="y", alpha=0.25)
            ax.set_title("{}\n{}".format(title, direction), fontsize=10)
            if column == 0:
                ax.set_ylabel(capability_label)
            for index, value in enumerate(means):
                digits = 3 if metric in {"ssim", "lpips"} else 2
                ax.text(index, value, ("{:.%df}" % digits).format(value), ha="center", va="bottom", fontsize=8)
    fig.suptitle("EH-SurGS on fixed SIM protocol — mean ± sample std, seeds 0/1/2", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
