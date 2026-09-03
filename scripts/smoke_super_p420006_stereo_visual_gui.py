#!/usr/bin/env python3
"""Assemble the real GUI environment and replay a selected raw visual driver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import warp as wp


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from embodied_environments.super_embodied.super_embodied import (  # noqa: E402
    PSM_POSE_DRIVER_PATHS,
    apply_psm_lnd_pose,
    build_environment,
)


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_p420006_stereo_v1/"
    "gui_v1/runtime_smoke_report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the current SUPER GUI scene with the stereo-visual "
            "P420006 driver and replay representative raw timestamps."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--version",
        choices=tuple(PSM_POSE_DRIVER_PATHS),
        default="raw_p420006_stereo_visual",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version = args.version
    pose_driver = PSM_POSE_DRIVER_PATHS[version]
    with np.load(pose_driver, allow_pickle=False) as driver:
        expected_poses = driver[
            "poses_gui_world_xyz_xyzw"
        ].astype(np.float32)
        state_count = len(driver["timestamps"])
        link_count = len(driver["link_names"])
    selected = np.unique(
        np.linspace(0, state_count - 1, 5).round().astype(np.int64)
    )
    expected_urdf = (
        "psm_paper_lnd.urdf"
        if version
        in {
            "raw_paper_lnd_sam2_online",
            "raw_paper_lnd_first_stereo_static_q5",
            "raw_paper_lnd_first_stereo_se3_fixed_q5",
            "raw_paper_lnd_sam2_multianchor_closedjaw",
            "raw_paper_lnd_sam2_dense_contact_closedjaw",
            "raw_paper_lnd_sam2_dense_contact_se3_only",
            "raw_paper_lnd_sam2_dense_contact_unbounded_xyz",
        }
        else "psm_p420006.urdf"
    )

    wp.init()
    environment = build_environment(
        num_envs=1,
        add_gaussians=True,
        device=args.device,
        psm_pose_driver_path=pose_driver,
        psm_visual_tip_only=True,
        tissue_mode="adaptive_soft",
    )
    body_ids = environment.super_psm_lnd_body_ids[0]
    errors: list[float] = []
    for state_index in selected:
        apply_psm_lnd_pose(
            environment,
            int(state_index),
            update_gaussians=True,
        )
        actual = wp.to_torch(environment.sim.state_0.body_q)[
            body_ids
        ].detach()
        expected = torch.as_tensor(
            expected_poses[state_index],
            dtype=actual.dtype,
            device=actual.device,
        )
        errors.append(float(torch.max(torch.abs(actual - expected)).item()))
    wp.synchronize()

    gates = {
        "pose_source_selected": (
            environment.super_psm_pose_source == version
        ),
        "instrument_urdf_selected": (
            Path(environment.super_psm_urdf_path).name
            == expected_urdf
        ),
        "driver_state_count": (
            len(environment.super_psm_lnd_timestamps) == 5458
        ),
        "driver_link_count": (
            len(environment.super_psm_lnd_link_names) == link_count == 7
        ),
        "surface_gaussians_loaded": (
            environment.super_psm_gaussian_count > 0
        ),
        "representative_poses_replay_exactly": max(errors) <= 1.0e-7,
    }
    report = {
        "schema": "super_visual_instrument_gui_runtime_smoke_v1",
        "passed": all(gates.values()),
        "device": str(wp.get_device(args.device)),
        "pose_driver": str(pose_driver.resolve()),
        "selected_state_indices": selected.tolist(),
        "maximum_pose_component_errors": errors,
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise RuntimeError("Stereo-visual instrument GUI runtime smoke failed")


if __name__ == "__main__":
    main()
