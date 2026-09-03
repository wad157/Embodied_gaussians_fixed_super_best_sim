#!/usr/bin/env python3
"""Run first-pair bilateral paper-LND SE(3) calibration with raw q5 frozen."""

from __future__ import annotations

import sys
from pathlib import Path

from optimize_super_paper_lnd_first_stereo_static import main


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/"
    "raw_paper_lnd_first_stereo_se3_fixed_q5_v1"
)


if __name__ == "__main__":
    if "--freeze-q5" not in sys.argv:
        sys.argv.append("--freeze-q5")
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", str(OUTPUT_ROOT)])
    if "--seed" not in sys.argv:
        sys.argv.extend(["--seed", "420009"])
    main()
