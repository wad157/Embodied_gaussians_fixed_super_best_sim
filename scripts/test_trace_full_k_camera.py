#!/usr/bin/env python3
"""CPU regression test for TRACE's adapter-side asymmetric full-K projection."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path = [
    str(REPO_ROOT / "baselines" / "trace_sim"),
    str(REPO_ROOT / "baselines" / "TRACE"),
] + [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPT_ROOT]
from common import projection_matrix_from_k  # noqa: E402


def main() -> None:
    width, height = 1280, 720
    intrinsic = np.asarray(
        [[901.25, 0.0, 517.75], [0.0, 887.5, 403.25], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    points = np.asarray(
        [[0.0, 0.0, 2.0], [0.25, -0.15, 1.7], [-0.42, 0.31, 3.2]],
        dtype=np.float32,
    )
    direct = (intrinsic @ points.T).T
    direct = direct[:, :2] / direct[:, 2:3]
    projection = projection_matrix_from_k(0.01, 100.0, intrinsic, height, width)
    homogeneous = torch.from_numpy(
        np.concatenate((points, np.ones((len(points), 1), dtype=np.float32)), axis=1)
    )
    clip = homogeneous @ projection.transpose(0, 1)
    ndc = clip[:, :2] / clip[:, 3:4]
    raster_pixels = torch.stack(
        (
            (ndc[:, 0] + 1.0) * float(width) / 2.0,
            (ndc[:, 1] + 1.0) * float(height) / 2.0,
        ),
        dim=1,
    ).numpy()
    error = np.linalg.norm(raster_pixels - direct, axis=1)
    centered_error = math.hypot(intrinsic[0, 2] - width / 2.0, intrinsic[1, 2] - height / 2.0)
    if float(error.max()) > 1.0e-4:
        raise AssertionError("full-K projection error {} px".format(float(error.max())))
    if centered_error < 100.0:
        raise AssertionError("test camera is not sufficiently off-center")
    print("PASS full-K max_error_px={:.8f} principal_offset_px={:.3f}".format(float(error.max()), centered_error))


if __name__ == "__main__":
    main()
