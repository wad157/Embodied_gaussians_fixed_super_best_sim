#!/usr/bin/env python3
"""Train the pinned EndoGaussian baseline on allowed SIM prefix observations."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

import train as upstream_train
from arguments import ModelHiddenParams, ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui
from utils.general_utils import safe_state
from utils.params_utils import merge_hparams

from common import SimScene
from protocol import DATASETS, UPSTREAM_COMMIT, dataset_spec, resolve_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "sim_unified.py"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    hidden = ModelHiddenParams(parser)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sim-downsample", type=int, default=1)
    parser.add_argument("--sim-init-points", type=int, default=30000)
    parser.add_argument("--configs", default=str(DEFAULT_CONFIG))
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[2000, 3000, 4000, 5000, 6000, 9000, 10000, 14000, 20000, 30000, 45000, 60000],
    )
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", default=None)
    parser.add_argument("--expname", default="")
    args = parser.parse_args()
    import mmcv

    args = merge_hparams(args, mmcv.Config.fromfile(args.configs))
    if not args.model_path:
        parser.error("--model-path 必须显式指定")
    if args.sim_downsample < 1:
        parser.error("--sim-downsample 必须 >= 1")
    dataset = resolve_dataset(REPO_ROOT, args.dataset_key, Path(args.source_path))
    spec = dataset_spec(args.dataset_key)
    if dataset.name != spec["name"]:
        parser.error("数据集目录名与固定协议不符")
    model_path = Path(args.model_path).expanduser().resolve()
    if model_path.exists() and any(model_path.iterdir()):
        parser.error("拒绝复用非空 checkpoint 目录：{}".format(model_path))
    args.source_path = str(dataset)
    args.model_path = str(model_path)
    args.sim_dataset_key = args.dataset_key
    args.sim_seed = int(args.seed)
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    return args, model, optimization, pipeline, hidden


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args, model, optimization, pipeline, hidden = parse_args()
    safe_state(args.quiet)
    reset_seed(args.seed)
    upstream_train.Scene = SimScene
    upstream_train.args = args
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    model_args = model.extract(args)
    model_args.sim_dataset_key = args.sim_dataset_key
    model_args.sim_downsample = args.sim_downsample
    model_args.sim_init_points = args.sim_init_points
    model_args.sim_seed = args.sim_seed
    print("EndoGaussian upstream commit: {}".format(UPSTREAM_COMMIT))
    print("Dataset: {}".format(args.source_path))
    print("Seed: {}; output: {}".format(args.seed, args.model_path))
    upstream_train.training(
        model_args,
        hidden.extract(args),
        optimization.extract(args),
        pipeline.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.expname,
        args.extra_mark,
    )
    print("Training complete.")


if __name__ == "__main__":
    main()
