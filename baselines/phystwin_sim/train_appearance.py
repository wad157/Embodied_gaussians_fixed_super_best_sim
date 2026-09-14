#!/usr/bin/env python3
"""Fit PhysTwin's query-frame tissue Gaussians on the two SIM cameras.

The upstream method trains a static 3DGS representation at the query frame,
then deforms it with particle LBS.  This adapter keeps that division while
using the already installed gsplat CUDA rasterizer for the two-camera SIM
interface.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat.rendering import rasterization

from common import PackedTissueMasks, load_calibration, load_rgb
from protocol import CAMERAS, DATASETS, UPSTREAM_COMMIT, dataset_spec, resolve_dataset, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--preprocess-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--depth-weight", type=float, default=0.001)
    return parser.parse_args()


def logit(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(value.clamp(1.0e-4, 1.0 - 1.0e-4))


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    c1, c2 = 0.01**2, 0.03**2
    mu_x = F.avg_pool2d(x, 11, stride=1, padding=5)
    mu_y = F.avg_pool2d(y, 11, stride=1, padding=5)
    sigma_x = F.avg_pool2d(x * x, 11, stride=1, padding=5) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 11, stride=1, padding=5) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 11, stride=1, padding=5) - mu_x * mu_y
    value = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )
    return value.mean()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"拒绝覆盖 {args.output_dir}")
    if args.iterations < 1 or args.downsample < 1:
        raise ValueError("iterations/downsample must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = resolve_dataset(REPO_ROOT, args.dataset_key, args.dataset)
    appearance_path = args.preprocess_dir / "appearance.npz"
    preprocess_meta_path = args.preprocess_dir / "metadata.json"
    if not appearance_path.is_file() or not preprocess_meta_path.is_file():
        raise FileNotFoundError("preprocess appearance is incomplete")
    with np.load(appearance_path, allow_pickle=False) as archive:
        means_np = np.asarray(archive["means_world"], dtype=np.float32)
        colors_np = np.asarray(archive["colors_rgb"], dtype=np.float32)
    device = torch.device(args.device)
    means = torch.nn.Parameter(torch.from_numpy(means_np).to(device))
    color_logits = torch.nn.Parameter(logit(torch.from_numpy(colors_np).to(device)))
    opacity_logits = torch.nn.Parameter(
        torch.full((len(means_np),), 0.0, dtype=torch.float32, device=device)
    )
    log_scales = torch.nn.Parameter(
        torch.full((len(means_np), 3), math.log(0.00022), device=device)
    )
    quats = torch.zeros((len(means_np), 4), dtype=torch.float32, device=device)
    quats[:, 0] = 1.0
    optimizer = torch.optim.Adam(
        [
            {"params": [means], "lr": 2.0e-5},
            {"params": [color_logits], "lr": 2.5e-3},
            {"params": [opacity_logits], "lr": 2.5e-3},
            {"params": [log_scales], "lr": 1.0e-3},
        ]
    )
    masks = PackedTissueMasks(dataset)
    calibration = load_calibration(dataset)
    spec = dataset_spec(args.dataset_key)
    depth_root = dataset / "estimated_depth" / str(spec["depth"])
    targets = {}
    for camera in CAMERAS:
        image = load_rgb(dataset, camera, 0).astype(np.float32) / 255.0
        tissue = masks.get(camera, 0)
        depth = np.load(depth_root / camera / "000000-depth.npy").astype(np.float32)
        h, w = image.shape[:2]
        out_size = (w // args.downsample, h // args.downsample)
        image = cv2.resize(image, out_size, interpolation=cv2.INTER_AREA)
        tissue = cv2.resize(
            tissue.astype(np.uint8), out_size, interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        depth = cv2.resize(depth, out_size, interpolation=cv2.INTER_NEAREST)
        image[~tissue] = 0.0
        intrinsic = calibration[camera]["K"].copy()
        intrinsic[:2] /= float(args.downsample)
        targets[camera] = {
            "image": torch.from_numpy(image).to(device),
            "mask": torch.from_numpy(tissue).to(device),
            "depth": torch.from_numpy(depth).to(device),
            "K": torch.from_numpy(intrinsic.astype(np.float32)).to(device),
            "view": torch.from_numpy(
                np.linalg.inv(calibration[camera]["X_WC_ros_optical"]).astype(np.float32)
            ).to(device),
            "width": out_size[0],
            "height": out_size[1],
        }

    history = []
    for iteration in range(args.iterations):
        camera = CAMERAS[iteration % len(CAMERAS)]
        target = targets[camera]
        rendered, alpha, _ = rasterization(
            means=means,
            quats=quats,
            scales=log_scales.exp().clamp(5.0e-5, 0.003),
            opacities=opacity_logits.sigmoid(),
            colors=color_logits.sigmoid(),
            viewmats=target["view"][None],
            Ks=target["K"][None],
            width=target["width"],
            height=target["height"],
            # gsplat 1.5's RGB+D background channel padding is only supported
            # by the unpacked path; this changes storage, not rasterization.
            packed=False,
            render_mode="RGB+D",
            backgrounds=torch.zeros((1, 3), device=device),
            near_plane=0.01,
            far_plane=2.0,
        )
        rgb = rendered[0, ..., :3]
        depth_premult = rendered[0, ..., 3]
        alpha_image = alpha[0, ..., 0]
        target_rgb = target["image"]
        l1 = torch.abs(rgb - target_rgb).mean()
        dssim = 1.0 - ssim(
            rgb.permute(2, 0, 1)[None], target_rgb.permute(2, 0, 1)[None]
        )
        valid_depth = target["mask"] & torch.isfinite(target["depth"]) & (target["depth"] > 0)
        predicted_depth = depth_premult / alpha_image.clamp_min(1.0e-6)
        depth_loss = torch.abs(predicted_depth[valid_depth] - target["depth"][valid_depth]).mean()
        loss = 0.8 * l1 + 0.2 * dssim + args.depth_weight * depth_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            means[:, 2].clamp_(0.001, 0.2)
            log_scales.clamp_(math.log(5.0e-5), math.log(0.003))
        entry = {
            "iteration": iteration,
            "camera": camera,
            "loss": float(loss.detach()),
            "l1": float(l1.detach()),
            "dssim": float(dssim.detach()),
            "depth_l1_m": float(depth_loss.detach()),
        }
        if iteration % 25 == 0 or iteration == args.iterations - 1:
            history.append(entry)
            print(
                f"[appearance] {iteration}/{args.iterations - 1} {camera} "
                f"loss={entry['loss']:.6g} l1={entry['l1']:.6g}",
                flush=True,
            )

    args.output_dir.mkdir(parents=True)
    np.savez_compressed(
        args.output_dir / "gaussians.npz",
        means_world=means.detach().cpu().numpy().astype(np.float32),
        colors_rgb=color_logits.sigmoid().detach().cpu().numpy().astype(np.float32),
        opacities=opacity_logits.sigmoid().detach().cpu().numpy().astype(np.float32),
        scales=log_scales.exp().detach().cpu().numpy().astype(np.float32),
        quats=quats.cpu().numpy().astype(np.float32),
    )
    metadata = {
        "schema": "fixedsuperbest.phystwin_sim_appearance.v1",
        "dataset_key": args.dataset_key,
        "seed": args.seed,
        "phystwin_commit": UPSTREAM_COMMIT,
        "representation": "query-frame 3D Gaussians deformed by native particle LBS",
        "rasterizer": "gsplat compatibility backend",
        "iterations": args.iterations,
        "downsample": args.downsample,
        "gaussians": int(len(means_np)),
        "preprocess_appearance_sha256": sha256_file(appearance_path),
        "training_frames": [0],
        "training_cameras": list(CAMERAS),
        "future_observations_used": False,
        "evaluation_truth_opened": False,
        "history": history,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
