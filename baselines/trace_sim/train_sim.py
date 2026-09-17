#!/usr/bin/env python3
"""Train the pinned, unmodified TRACE core on legal SIM stereo RGB frames."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

import train_gui as upstream_train
from arguments import ModelParams, OptimizationParams, PipelineParams
from utils.general_utils import safe_state

from common import SimScene
from protocol import (
    DATASETS,
    DEFAULT_INIT_POINTS,
    DEFAULT_ITERATIONS,
    DEFAULT_TRAIN_DOWNSAMPLE,
    UPSTREAM_COMMIT,
    dataset_spec,
    max_observed_time,
    resolve_dataset,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sim-downsample", type=int, default=DEFAULT_TRAIN_DOWNSAMPLE)
    parser.add_argument("--sim-init-points", type=int, default=DEFAULT_INIT_POINTS)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--W", type=int, default=800)
    parser.add_argument("--H", type=int, default=800)
    parser.add_argument("--elevation", type=float, default=0.0)
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--fovy", type=float, default=50.0)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    dataset = resolve_dataset(REPO_ROOT, args.dataset_key, Path(args.source_path))
    frames = int(dataset_spec(args.dataset_key)["frames"])
    model_path = Path(args.model_path).expanduser().resolve()
    if model_path.exists() and any(model_path.iterdir()):
        parser.error("Refusing to reuse non-empty model directory: {}".format(model_path))
    if args.sim_downsample < 1 or args.sim_init_points < 1000:
        parser.error("sim-downsample must be >=1 and sim-init-points >=1000")
    if int(args.iterations) != DEFAULT_ITERATIONS:
        print("WARNING: non-formal TRACE iteration count {}".format(args.iterations), flush=True)
    args.source_path = str(dataset)
    args.model_path = str(model_path)
    args.sim_dataset_key = args.dataset_key
    args.sim_seed = int(args.seed)
    args.max_time = max_observed_time(frames)
    args.white_background = False
    args.freegave = False
    args.fps = 30
    if int(args.iterations) not in args.save_iterations:
        args.save_iterations.append(int(args.iterations))
    return args, model, optimization, pipeline


def reset_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    args, model, optimization, pipeline = parse_args()
    safe_state(args.quiet)
    reset_seed(args.seed)
    upstream_train.Scene = SimScene
    upstream_train.args = args
    model_args = model.extract(args)
    model_args.sim_dataset_key = args.sim_dataset_key
    model_args.sim_downsample = args.sim_downsample
    model_args.sim_init_points = args.sim_init_points
    model_args.sim_seed = args.sim_seed
    model_args.max_time = args.max_time
    model_args.white_background = False
    model_args.freegave = False
    model_args.fps = 30
    output = Path(args.model_path)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "trace_commit": UPSTREAM_COMMIT,
        "dataset_key": args.dataset_key,
        "dataset": args.source_path,
        "seed": args.seed,
        "iterations": int(args.iterations),
        "warm_up": int(args.warm_up),
        "train_downsample": args.sim_downsample,
        "init_points": args.sim_init_points,
        "max_time": args.max_time,
        "fps": 30,
        "freegave": False,
        "training_inputs": "legal prefix stereo RGB and calibrated cameras only",
        "forbidden_inputs_used": [],
    }
    (output / "trace_adapter_config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    trainer = upstream_train.GUI(
        args=args,
        dataset=model_args,
        opt=optimization.extract(args),
        pipe=pipeline.extract(args),
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
        vel_start_time=0.0,
    )
    trainer.train(int(args.iterations))
    checkpoint = output / "point_cloud" / "iteration_{}".format(args.iterations) / "point_cloud.ply"
    deformation = output / "deform" / "iteration_{}".format(args.iterations) / "deform.pth"
    velocity = output / "deform" / "iteration_{}".format(args.iterations) / "vel.pth"
    if not checkpoint.is_file() or not deformation.is_file() or not velocity.is_file():
        raise RuntimeError("TRACE training completed without the formal checkpoint")
    print("TRACE training complete: {}".format(output), flush=True)


if __name__ == "__main__":
    main()
