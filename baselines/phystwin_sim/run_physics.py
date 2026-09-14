#!/usr/bin/env python3
"""Run PhysTwin's CMA + differentiable spring-mass optimization headlessly."""

from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import random
import time
import warnings
from pathlib import Path

import cma
import numpy as np
import open3d as o3d
import torch
import warp as wp

from protocol import DATASETS, UPSTREAM_COMMIT, dataset_spec, future_start, sha256_file

from qqtt.model.diff_simulator import SpringMassSystemWarp
from qqtt.utils import cfg


# Warp repeats these diagnostics once per captured non-differentiable helper
# kernel. The helpers only update targets/neighbour indices and match upstream;
# suppress duplicate log spam without changing graph construction or gradients.
warnings.filterwarnings(
    "ignore",
    message="Running the tape backwards may produce incorrect gradients.*",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--preprocess-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cma-iterations", type=int, default=20)
    parser.add_argument("--adam-iterations", type=int, default=200)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--frame-limit", type=int, default=-1, help="smoke test only")
    parser.add_argument("--substeps", type=int, default=667)
    parser.add_argument(
        "--resume-after-cma",
        action="store_true",
        help=(
            "Resume a run whose complete CMA history and optimal_params.pkl were "
            "already written, but whose Adam stage did not complete."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure(args: argparse.Namespace, frame_count: int) -> int:
    config_path = Path(__file__).resolve().parents[1] / "PhysTwin" / "configs" / "real.yaml"
    cfg.load_from_yaml(str(config_path))
    cfg.device = args.device
    cfg.train_frame = future_start(frame_count)
    if args.frame_limit > 0:
        cfg.train_frame = min(cfg.train_frame, args.frame_limit)
    cfg.num_substeps = int(args.substeps)
    cfg.iterations = int(args.adam_iterations)
    cfg.vis_interval = int(args.checkpoint_interval)
    cfg.reverse_z = False  # SIM world uses +Z above the z=0 support plane.
    cfg.use_graph = True
    cfg.self_collision = False
    cfg.run_name = f"phystwin_{args.dataset_key}_seed{args.seed}"
    return int(cfg.train_frame)


def load_data(path: Path, device: str) -> dict:
    with path.open("rb") as stream:
        data = pickle.load(stream)
    result = {}
    for key in (
        "object_points",
        "object_visibilities",
        "object_motions_valid",
        "controller_points",
    ):
        dtype = torch.bool if "valid" in key or "visibilities" in key else torch.float32
        result[key] = torch.as_tensor(data[key], dtype=dtype, device=device)
    result["structure_points"] = torch.as_tensor(
        np.concatenate(
            (data["object_points"][0], data["surface_points"], data["interior_points"]),
            axis=0,
        ),
        dtype=torch.float32,
        device=device,
    )
    result["num_original_points"] = int(data["object_points"].shape[1])
    result["num_surface_points"] = int(
        data["object_points"].shape[1] + data["surface_points"].shape[0]
    )
    result["num_all_points"] = int(result["structure_points"].shape[0])
    return result


def build_springs(
    object_points: torch.Tensor,
    controller_points: torch.Tensor,
    object_radius: float,
    object_max_neighbours: int,
    controller_radius: float,
    controller_max_neighbours: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """The upstream `_init_start` topology construction without trainer imports."""
    object_np = object_points.detach().cpu().numpy()
    controller_np = controller_points.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(object_np)
    tree = o3d.geometry.KDTreeFlann(pcd)
    spring_flags: set[tuple[int, int]] = set()
    springs: list[list[int]] = []
    rest: list[float] = []
    for i, point in enumerate(object_np):
        _, indices, _ = tree.search_hybrid_vector_3d(
            point, float(object_radius), int(object_max_neighbours)
        )
        for j in indices[1:]:
            edge = (min(i, int(j)), max(i, int(j)))
            distance = float(np.linalg.norm(point - object_np[j]))
            if edge not in spring_flags and distance > 1.0e-4:
                spring_flags.add(edge)
                springs.append([i, int(j)])
                rest.append(distance)
    num_object_springs = len(springs)
    for i, point in enumerate(controller_np):
        _, indices, _ = tree.search_hybrid_vector_3d(
            point, float(controller_radius), int(controller_max_neighbours)
        )
        for j in indices:
            springs.append([len(object_np) + i, int(j)])
            rest.append(float(np.linalg.norm(point - object_np[j])))
    if not springs or not num_object_springs:
        raise ValueError("PhysTwin spring topology is empty")
    vertices = np.concatenate((object_np, controller_np), axis=0)
    return (
        torch.as_tensor(vertices, dtype=torch.float32, device=device),
        torch.as_tensor(np.asarray(springs), dtype=torch.int32, device=device),
        torch.as_tensor(np.asarray(rest), dtype=torch.float32, device=device),
        torch.ones(len(vertices), dtype=torch.float32, device=device),
        num_object_springs,
    )


def denormalize_parameters(x: np.ndarray) -> dict[str, float | int]:
    def linear(value, low, high):
        return float(value) * (high - low) + low

    return {
        "global_spring_Y": linear(x[0], cfg.spring_Y_min, cfg.spring_Y_max),
        "object_radius": linear(x[1], 0.01, 0.05),
        "object_max_neighbours": int(linear(x[2], 10, 50)),
        "controller_radius": linear(x[3], 0.01, 0.08),
        "controller_max_neighbours": int(linear(x[4], 10, 80)),
        "collide_elas": float(x[5]),
        "collide_fric": linear(x[6], 0.0, 2.0),
        "collide_object_elas": float(x[7]),
        "collide_object_fric": linear(x[8], 0.0, 2.0),
        "collision_dist": linear(x[9], 0.01, 0.05),
        "drag_damping": linear(x[10], 0.0, 20.0),
        "dashpot_damping": linear(x[11], 0.0, 200.0),
    }


def initial_normalized_parameters() -> np.ndarray:
    def normal(value, low, high):
        return (value - low) / (high - low)

    return np.asarray(
        [
            normal(cfg.init_spring_Y, cfg.spring_Y_min, cfg.spring_Y_max),
            normal(cfg.object_radius, 0.01, 0.05),
            normal(cfg.object_max_neighbours, 10, 50),
            normal(cfg.controller_radius, 0.01, 0.08),
            normal(cfg.controller_max_neighbours, 10, 80),
            cfg.collide_elas,
            normal(cfg.collide_fric, 0, 2),
            cfg.collide_object_elas,
            normal(cfg.collide_object_fric, 0, 2),
            normal(cfg.collision_dist, 0.01, 0.05),
            normal(cfg.drag_damping, 0, 20),
            normal(cfg.dashpot_damping, 0, 200),
        ],
        dtype=np.float64,
    )


def make_simulator(data: dict, params: dict, *, backward: bool) -> tuple[SpringMassSystemWarp, int]:
    vertices, springs, rest, masses, num_object_springs = build_springs(
        data["structure_points"],
        data["controller_points"][0],
        params["object_radius"],
        params["object_max_neighbours"],
        params["controller_radius"],
        params["controller_max_neighbours"],
        cfg.device,
    )
    simulator = SpringMassSystemWarp(
        vertices,
        springs,
        rest,
        masses,
        dt=cfg.dt,
        num_substeps=cfg.num_substeps,
        spring_Y=params["global_spring_Y"],
        collide_elas=params["collide_elas"],
        collide_fric=params["collide_fric"],
        dashpot_damping=params["dashpot_damping"],
        drag_damping=params["drag_damping"],
        collide_object_elas=params["collide_object_elas"],
        collide_object_fric=params["collide_object_fric"],
        collision_dist=params["collision_dist"],
        num_object_points=data["num_all_points"],
        num_surface_points=data["num_surface_points"],
        num_original_points=data["num_original_points"],
        controller_points=data["controller_points"],
        reverse_z=cfg.reverse_z,
        spring_Y_min=cfg.spring_Y_min,
        spring_Y_max=cfg.spring_Y_max,
        gt_object_points=data["object_points"],
        gt_object_visibilities=data["object_visibilities"],
        gt_object_motions_valid=data["object_motions_valid"],
        self_collision=False,
        disable_backward=not backward,
    )
    return simulator, num_object_springs


def rollout_loss(simulator: SpringMassSystemWarp, train_frame: int) -> float:
    simulator.set_init_state(simulator.wp_init_vertices, simulator.wp_init_velocities)
    simulator.set_acc_count(False)
    total = 0.0
    for frame in range(1, train_frame):
        simulator.set_controller_target(frame)
        wp.capture_launch(simulator.graph)
        if int(wp.to_torch(simulator.acc_count, requires_grad=False)[0]) == 0:
            simulator.set_acc_count(True)
        simulator.update_acc()
        value = float(wp.to_torch(simulator.loss, requires_grad=False).item())
        total += value
        simulator.clear_loss()
        simulator.set_init_state(
            simulator.wp_states[-1].wp_x, simulator.wp_states[-1].wp_v
        )
    result = total / max(train_frame - 1, 1)
    return result if math.isfinite(result) else 1.0e6


def cma_stage(data: dict, args: argparse.Namespace, train_frame: int) -> tuple[dict, list[dict]]:
    history: list[dict] = []
    calls = 0

    def objective(x) -> float:
        nonlocal calls
        calls += 1
        params = denormalize_parameters(np.asarray(x))
        start = time.time()
        try:
            simulator, springs = make_simulator(data, params, backward=False)
            loss = rollout_loss(simulator, train_frame)
            del simulator
        except Exception as error:
            print(f"[CMA] call={calls} failed: {error}", flush=True)
            loss, springs = 1.0e6, -1
        gc.collect()
        torch.cuda.empty_cache()
        entry = {
            "call": calls,
            "loss": loss,
            "springs": springs,
            "seconds": time.time() - start,
            "parameters": params,
        }
        history.append(entry)
        print(
            f"[CMA] call={calls} loss={loss:.8g} springs={springs} "
            f"seconds={entry['seconds']:.2f}",
            flush=True,
        )
        return loss

    x0 = initial_normalized_parameters()
    if args.cma_iterations <= 0:
        objective(x0)
        return denormalize_parameters(x0), history
    strategy = cma.CMAEvolutionStrategy(
        x0,
        1.0 / 6.0,
        {"bounds": [0.0, 1.0], "seed": int(args.seed), "verbose": -9},
    )
    strategy.optimize(objective, iterations=int(args.cma_iterations))
    return denormalize_parameters(np.asarray(strategy.result.xbest)), history


def adam_stage(
    data: dict, params: dict, args: argparse.Namespace, train_frame: int
) -> tuple[dict, list[dict], int]:
    simulator, num_object_springs = make_simulator(data, params, backward=True)
    optimizer = torch.optim.Adam(
        [
            wp.to_torch(simulator.wp_spring_Y),
            wp.to_torch(simulator.wp_collide_elas),
            wp.to_torch(simulator.wp_collide_fric),
            wp.to_torch(simulator.wp_collide_object_elas),
            wp.to_torch(simulator.wp_collide_object_fric),
        ],
        lr=cfg.base_lr,
        betas=(0.9, 0.99),
    )
    history: list[dict] = []
    best_loss = float("inf")
    best = None
    for epoch in range(args.adam_iterations):
        start = time.time()
        simulator.set_init_state(simulator.wp_init_vertices, simulator.wp_init_velocities)
        simulator.set_acc_count(False)
        total = chamfer = track = 0.0
        for frame in range(1, train_frame):
            simulator.set_controller_target(frame)
            wp.capture_launch(simulator.graph)
            optimizer.step()
            total += float(wp.to_torch(simulator.loss, requires_grad=False).item())
            chamfer += float(wp.to_torch(simulator.chamfer_loss, requires_grad=False).item())
            track += float(wp.to_torch(simulator.track_loss, requires_grad=False).item())
            simulator.tape.zero()
            simulator.clear_loss()
            simulator.set_init_state(
                simulator.wp_states[-1].wp_x, simulator.wp_states[-1].wp_v
            )
        denom = max(train_frame - 1, 1)
        total, chamfer, track = total / denom, chamfer / denom, track / denom
        checkpoint_epoch = epoch % args.checkpoint_interval == 0 or epoch == args.adam_iterations - 1
        entry = {
            "epoch": epoch,
            "loss": total,
            "chamfer_loss": chamfer,
            "track_loss": track,
            "seconds": time.time() - start,
            "checkpoint": checkpoint_epoch,
        }
        history.append(entry)
        print(
            f"[Adam] epoch={epoch}/{args.adam_iterations - 1} loss={total:.8g} "
            f"chamfer={chamfer:.8g} track={track:.8g} seconds={entry['seconds']:.2f}",
            flush=True,
        )
        if checkpoint_epoch and math.isfinite(total) and total < best_loss:
            best_loss = total
            best = {
                "epoch": epoch,
                "loss": total,
                "spring_Y": torch.exp(
                    wp.to_torch(simulator.wp_spring_Y, requires_grad=False)
                ).detach().cpu().numpy(),
                "collide_elas": wp.to_torch(
                    simulator.wp_collide_elas, requires_grad=False
                ).detach().cpu().numpy(),
                "collide_fric": wp.to_torch(
                    simulator.wp_collide_fric, requires_grad=False
                ).detach().cpu().numpy(),
                "collide_object_elas": wp.to_torch(
                    simulator.wp_collide_object_elas, requires_grad=False
                ).detach().cpu().numpy(),
                "collide_object_fric": wp.to_torch(
                    simulator.wp_collide_object_fric, requires_grad=False
                ).detach().cpu().numpy(),
            }
    if best is None:
        raise RuntimeError("PhysTwin Adam stage produced no finite checkpoint")
    del optimizer, simulator
    gc.collect()
    torch.cuda.empty_cache()
    return best, history, num_object_springs


def inference_rollout(data: dict, params: dict, checkpoint: dict) -> np.ndarray:
    simulator, num_object_springs = make_simulator(data, params, backward=False)
    if len(checkpoint["spring_Y"]) != simulator.n_springs:
        raise ValueError("checkpoint spring topology does not match rollout")
    simulator.set_spring_Y(
        torch.log(torch.as_tensor(checkpoint["spring_Y"], device=cfg.device))
    )
    simulator.set_collide(
        torch.as_tensor(checkpoint["collide_elas"], device=cfg.device),
        torch.as_tensor(checkpoint["collide_fric"], device=cfg.device),
    )
    simulator.set_collide_object(
        torch.as_tensor(checkpoint["collide_object_elas"], device=cfg.device),
        torch.as_tensor(checkpoint["collide_object_fric"], device=cfg.device),
    )
    simulator.set_init_state(simulator.wp_init_vertices, simulator.wp_init_velocities)
    vertices = [
        wp.to_torch(simulator.wp_states[0].wp_x, requires_grad=False).detach().cpu().numpy()
    ]
    frame_count = int(data["object_points"].shape[0])
    for frame in range(1, frame_count):
        simulator.set_controller_target(frame, pure_inference=True)
        wp.capture_launch(simulator.forward_graph)
        vertices.append(
            wp.to_torch(simulator.wp_states[-1].wp_x, requires_grad=False)
            .detach().cpu().numpy()
        )
        simulator.set_init_state(
            simulator.wp_states[-1].wp_x, simulator.wp_states[-1].wp_v
        )
        if frame % 30 == 0 or frame == frame_count - 1:
            print(f"[rollout] {frame}/{frame_count - 1}", flush=True)
    if num_object_springs <= 0:
        raise AssertionError("no object springs")
    return np.stack(vertices).astype(np.float32)


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and not args.resume_after_cma:
        raise FileExistsError(f"拒绝覆盖 {args.output_dir}")
    if args.cma_iterations < 0 or args.adam_iterations < 1 or args.substeps < 1:
        raise ValueError("invalid optimization counts")
    set_seed(args.seed)
    spec = dataset_spec(args.dataset_key)
    frame_count = int(spec["frames"])
    train_frame = configure(args, frame_count)
    preprocess_file = args.preprocess_dir / "final_data.pkl"
    preprocess_meta = args.preprocess_dir / "metadata.json"
    if not preprocess_file.is_file() or not preprocess_meta.is_file():
        raise FileNotFoundError("preprocess directory is incomplete")
    data = load_data(preprocess_file, args.device)
    if args.resume_after_cma:
        cma_path = args.output_dir / "cma_history.json"
        params_path = args.output_dir / "optimal_params.pkl"
        forbidden = (
            args.output_dir / "adam_history.json",
            args.output_dir / "best_checkpoint.pkl",
            args.output_dir / "raw_particle_rollout.npz",
            args.output_dir / "metadata.json",
        )
        if not cma_path.is_file() or not params_path.is_file():
            raise FileNotFoundError("resume requires complete CMA artifacts")
        if any(path.exists() for path in forbidden):
            raise FileExistsError("resume refuses to overwrite Adam/rollout artifacts")
        cma_history = json.loads(cma_path.read_text(encoding="utf-8"))
        expected_calls = (
            1
            if args.cma_iterations <= 0
            else int(args.cma_iterations) * (4 + int(3 * math.log(12)))
        )
        if len(cma_history) != expected_calls:
            raise ValueError(
                f"CMA history is incomplete: {len(cma_history)} != {expected_calls}"
            )
        if any(not math.isfinite(float(entry["loss"])) for entry in cma_history):
            raise ValueError("CMA history contains non-finite loss")
        with params_path.open("rb") as stream:
            params = pickle.load(stream)
        print(
            f"[resume] loaded {len(cma_history)} complete CMA evaluations; "
            "continuing with Adam",
            flush=True,
        )
    else:
        args.output_dir.mkdir(parents=True)

    start = time.time()
    if not args.resume_after_cma:
        params, cma_history = cma_stage(data, args, train_frame)
        (args.output_dir / "cma_history.json").write_text(
            json.dumps(jsonable(cma_history), indent=2) + "\n", encoding="utf-8"
        )
        with (args.output_dir / "optimal_params.pkl").open("wb") as stream:
            pickle.dump(params, stream, protocol=pickle.HIGHEST_PROTOCOL)
    checkpoint, adam_history, num_object_springs = adam_stage(
        data, params, args, train_frame
    )
    with (args.output_dir / "best_checkpoint.pkl").open("wb") as stream:
        pickle.dump(checkpoint, stream, protocol=pickle.HIGHEST_PROTOCOL)
    (args.output_dir / "adam_history.json").write_text(
        json.dumps(jsonable(adam_history), indent=2) + "\n", encoding="utf-8"
    )
    trajectory = inference_rollout(data, params, checkpoint)
    np.savez_compressed(
        args.output_dir / "raw_particle_rollout.npz",
        particle_positions_world=trajectory,
        frame_indices=np.arange(frame_count, dtype=np.int32),
    )
    metadata = {
        "schema": "fixedsuperbest.phystwin_sim_physics.v1",
        "dataset_key": args.dataset_key,
        "seed": args.seed,
        "phystwin_commit": UPSTREAM_COMMIT,
        "preprocess_sha256": sha256_file(preprocess_file),
        "train_frame_exclusive": train_frame,
        "full_protocol": (
            args.frame_limit < 0
            and args.cma_iterations == 20
            and args.adam_iterations == 200
            and args.substeps == 667
        ),
        "cma_iterations": args.cma_iterations,
        "cma_function_evaluations": len(cma_history),
        "cma_resumed_from_complete_stage": bool(args.resume_after_cma),
        "adam_iterations": args.adam_iterations,
        "substeps_per_frame": args.substeps,
        "num_object_springs": num_object_springs,
        "best_epoch": int(checkpoint["epoch"]),
        "best_loss": float(checkpoint["loss"]),
        "optimal_global_parameters": params,
        "rollout_shape": list(trajectory.shape),
        "future_observations_used": False,
        "evaluation_truth_opened": False,
        "shape_of_motion_used": False,
        "trajectory_source": "PhysTwin native spring-mass particles",
        "wall_seconds": time.time() - start,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(jsonable(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(jsonable(metadata), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
