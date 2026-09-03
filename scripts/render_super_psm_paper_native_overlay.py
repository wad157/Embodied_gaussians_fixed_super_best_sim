#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
PAPER_REPO = Path("/Media_HDD/jwshan/wad/online_dvrk_tracking")
TRACK_ROOT = REPO / "data/super/psm_tracking"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the paper repository's own four CAD meshes from saved states."
    )
    parser.add_argument(
        "--frames", default="0,160,320,480,640,800,960,1120,1280,1440"
    )
    parser.add_argument(
        "--output-dir", default=TRACK_ROOT / "paper_native_projection", type=Path
    )
    parser.add_argument(
        "--states", default=TRACK_ROOT / "tracking_states.npz", type=Path
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path[:0] = [str(PAPER_REPO), str(PAPER_REPO / "SurgicalSAM2")]
    sys.path.insert(0, str(REPO / "scripts"))
    from diffcali.models.CtRNet import CtRNet
    from super_psm_tracking_common import TrackingInputs

    inputs = TrackingInputs.load()
    K = inputs.K_half
    model_args = Namespace(
        use_gpu=True,
        trained_on_multi_gpus=False,
        height=540,
        width=960,
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        px=float(K[0, 2]),
        py=float(K[1, 2]),
        scale=1.0,
        use_nvdiffrast=True,
    )
    model = CtRNet(model_args)
    mesh_dir = PAPER_REPO / "urdfs/dVRK/meshes"
    mesh_files = [
        mesh_dir / "low_res_shaft_multi_cylinder.ply",
        mesh_dir / "low_res_logo_low_res_1.ply",
        mesh_dir / "low_res_jawright_lowres.ply",
        mesh_dir / "low_res_jawleft_lowres.ply",
    ]
    renderer = model.setup_robot_renderer(
        [str(path) for path in mesh_files], downscale_factor=2
    )
    states = np.load(args.states)
    mask_shape = tuple(states["mask_shape"].tolist())
    frame_indices = [int(value) for value in args.frames.split(",") if value]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    colors = [(0, 180, 255), (255, 180, 0), (0, 0, 255), (255, 0, 255)]
    names = ["shaft", "logo", "jaw-right", "jaw-left"]

    for frame_index in frame_indices:
        frame = cv2.imread(
            str(REPO / f"data/super/grasp5_native/rgb/{frame_index:06d}-left.png")
        )
        if frame is None:
            raise FileNotFoundError(frame_index)
        ctr = torch.as_tensor(
            states["pure_ctr"][frame_index], device="cuda", dtype=torch.float32
        )
        joints = torch.as_tensor(
            states["pure_joints"][frame_index], device="cuda", dtype=torch.float32
        )
        mask_half = np.unpackbits(states["masks_packbits"][frame_index])[
            : np.prod(mask_shape)
        ].reshape(mask_shape).astype(bool)
        output = frame.copy()
        sam = cv2.resize(
            mask_half.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        contours, _ = cv2.findContours(
            sam, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
        for part_index, (name, color) in enumerate(zip(names, colors, strict=True)):
            renderer.set_mesh_visibility(
                [index == part_index for index in range(4)]
            )
            robot_mesh = renderer.get_robot_mesh(joints)
            with torch.no_grad():
                rendered = model.render_single_robot_mask(
                    ctr, robot_mesh, renderer, resolution=(270, 480)
                ).squeeze()
            rendered_np = rendered.detach().float().cpu().numpy()
            rendered_full = cv2.resize(
                rendered_np,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            part_mask = rendered_full > 0.25
            tint = np.zeros_like(output)
            tint[part_mask] = color
            output = cv2.addWeighted(output, 1.0, tint, 0.35, 0.0)
            part_contours, _ = cv2.findContours(
                part_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(output, part_contours, -1, color, 2)
            cv2.putText(
                output,
                name,
                (16, 80 + 30 * part_index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
        cv2.rectangle(output, (0, 0), (850, 55), (0, 0, 0), -1)
        cv2.putText(
            output,
            f"frame {frame_index}: native paper CAD; SAM boundary=green",
            (16, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        path = args.output_dir / f"frame{frame_index:06d}_paper_native.png"
        cv2.imwrite(str(path), output)
        print(path)


if __name__ == "__main__":
    main()
