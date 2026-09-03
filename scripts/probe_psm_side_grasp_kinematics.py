#!/usr/bin/env python3
"""扫描官方 PSM 腕部姿态，寻找侧向进入且夹爪上下分布的候选解。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from omni.isaac.lab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument(
        "--stage", choices=("wrist", "placement"), default="wrist"
    )
    parser.add_argument("--target-frame", type=int, default=130)
    parser.add_argument(
        "--trajectory-path",
        type=Path,
        default=None,
        help="placement 阶段使用的组织轨迹；默认兼容旧 v1 轨迹。",
    )
    parser.add_argument("--target-offset-mm", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--base-center", type=float, nargs=3, default=None)
    parser.add_argument("--base-half-range", type=float, nargs=3, default=None)
    parser.add_argument("--base-grid-steps", type=int, default=17)
    parser.add_argument("--gripper", type=float, default=0.96)
    return parser.parse_args()


ARGS = parse_args()
app_launcher = AppLauncher(
    {"headless": True, "enable_cameras": False, "multi_gpu": False}
)
simulation_app = app_launcher.app

import numpy as np
import torch
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext
from orbit.surgical.assets.psm import PSM_HIGH_PD_CFG


REPO_ROOT = Path(__file__).resolve().parents[1]
PSM_USD = REPO_ROOT / "data/sim_assets/PSM/psm_col.usd"
TIP_MESH_ASSET = (
    REPO_ROOT
    / "data/sim/tissue_retraction_closeup_inplane_full_v1/gui_assets"
    / "official_psm_tip_meshes_v2.npz"
)
Q7_NAMES = (
    "psm_yaw_joint",
    "psm_pitch_end_joint",
    "psm_main_insertion_joint",
    "psm_tool_roll_joint",
    "psm_tool_pitch_joint",
    "psm_tool_yaw_joint",
)


def set_q7(robot: Articulation, q7: np.ndarray) -> None:
    joint_index = {name: index for index, name in enumerate(robot.joint_names)}
    state = robot.data.default_joint_pos.clone()
    for name, value in zip(Q7_NAMES, q7[:6]):
        state[:, joint_index[name]] = float(value)
    state[:, joint_index["psm_tool_gripper1_joint"]] = -0.5 * float(q7[6])
    state[:, joint_index["psm_tool_gripper2_joint"]] = 0.5 * float(q7[6])
    robot.write_joint_state_to_sim(state, torch.zeros_like(state))
    robot.set_joint_position_target(state)
    robot.write_data_to_sim()


def normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1.0e-9)


def quaternion_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(value) for value in quaternion]
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def distal_contact_point(vertices: np.ndarray) -> np.ndarray:
    """夹爪局部 +y 方向最前端 8% 顶点的中心，近似真实接触指尖。"""

    threshold = float(np.quantile(vertices[:, 1], 0.92))
    return vertices[vertices[:, 1] >= threshold].mean(axis=0).astype(np.float32)


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0"))
    robot_cfg = PSM_HIGH_PD_CFG.replace(prim_path="/World/PSM")
    robot_cfg.spawn.usd_path = str(PSM_USD.resolve())
    robot_cfg.init_state.pos = (0.0, 0.0, 0.210)
    robot = Articulation(robot_cfg)
    sim.reset()

    body_index = {name: index for index, name in enumerate(robot.body_names)}
    required = (
        "psm_tool_pitch_link",
        "psm_tool_yaw_link",
        "psm_tool_gripper1_link",
        "psm_tool_gripper2_link",
        "psm_tool_tip_link",
    )
    missing = [name for name in required if name not in body_index]
    if missing:
        raise RuntimeError(f"官方 PSM 缺少刚体：{missing}")
    if not TIP_MESH_ASSET.is_file():
        raise FileNotFoundError(f"缺少官方 PSM 夹爪网格：{TIP_MESH_ASSET}")
    with np.load(TIP_MESH_ASSET, allow_pickle=False) as mesh_asset:
        jaw1_local_contact = distal_contact_point(
            mesh_asset["psm_tool_gripper1_link__vertices"]
        )
        jaw2_local_contact = distal_contact_point(
            mesh_asset["psm_tool_gripper2_link__vertices"]
        )

    target = None
    if ARGS.stage == "wrist":
        bases = [np.asarray([0.011, 0.170, 0.175], dtype=np.float32)]
        rolls = np.linspace(-math.pi, math.pi, 17, endpoint=False)
        pitches = np.linspace(-1.45, 1.45, 17)
        yaws = np.linspace(-1.45, 1.45, 17)
    else:
        trajectory_path = (
            ARGS.trajectory_path.resolve()
            if ARGS.trajectory_path is not None
            else REPO_ROOT
            / "data/sim_precompute/tissue_long_edge_lift_return_sufia_v1.npz"
        )
        with np.load(trajectory_path, allow_pickle=False) as trajectory:
            grasp_mask = trajectory["grasp_mask"].astype(bool)
            target = trajectory["visual_positions"][ARGS.target_frame, grasp_mask].mean(axis=0)
            target = target + np.asarray(ARGS.target_offset_mm, dtype=np.float32) * 1.0e-3
        # 腕部取上一阶段的最优解；扫描 RCM yaw/pitch 与插入量来对准组织中面。
        if ARGS.base_center is None:
            center = np.asarray([0.0, 0.15, 0.175], dtype=np.float32)
            half_range = np.asarray([0.08, 0.20, 0.055], dtype=np.float32)
        else:
            if ARGS.base_half_range is None:
                raise ValueError("--base-center 与 --base-half-range 必须同时提供")
            center = np.asarray(ARGS.base_center, dtype=np.float32)
            half_range = np.asarray(ARGS.base_half_range, dtype=np.float32)
        steps = int(ARGS.base_grid_steps)
        bases = [
            np.asarray([q0, q1, q2], dtype=np.float32)
            for q0 in np.linspace(center[0] - half_range[0], center[0] + half_range[0], steps)
            for q1 in np.linspace(center[1] - half_range[1], center[1] + half_range[1], steps)
            for q2 in np.linspace(center[2] - half_range[2], center[2] + half_range[2], steps)
        ]
        rolls = np.asarray([1.663196086883545])
        pitches = np.asarray([0.0])
        yaws = np.asarray([-1.45])
    records = []
    for base in bases:
        for roll in rolls:
            for pitch in pitches:
                for yaw in yaws:
                    q7 = np.asarray(
                        [base[0], base[1], base[2], roll, pitch, yaw, ARGS.gripper],
                        dtype=np.float32,
                    )
                    set_q7(robot, q7)
                    sim.step(render=False)
                    robot.update(sim.get_physics_dt())
                    positions = robot.data.body_pos_w[0].detach().cpu().numpy()
                    quaternions = robot.data.body_quat_w[0].detach().cpu().numpy()
                    jaw1_index = body_index["psm_tool_gripper1_link"]
                    jaw2_index = body_index["psm_tool_gripper2_link"]
                    jaw1 = (
                        positions[jaw1_index]
                        + quaternion_matrix_wxyz(quaternions[jaw1_index]) @ jaw1_local_contact
                    )
                    jaw2 = (
                        positions[jaw2_index]
                        + quaternion_matrix_wxyz(quaternions[jaw2_index]) @ jaw2_local_contact
                    )
                    jaw_midpoint = 0.5 * (jaw1 + jaw2)
                    jaw_separation = jaw2 - jaw1
                    yaw_link = positions[body_index["psm_tool_yaw_link"]]
                    tip = positions[body_index["psm_tool_tip_link"]]
                    distal_direction = normalized(tip - yaw_link)
                    separation_direction = normalized(jaw_separation)

                    vertical_opening = abs(float(separation_direction[2]))
                    horizontal_distal = math.sqrt(
                        float(distal_direction[0] ** 2 + distal_direction[1] ** 2)
                    )
                    # 目标长边位于 y<0，一次抓取从 y<0 朝 +y 插入。
                    inward_direction = float(distal_direction[1])
                    orthogonality = 1.0 - abs(
                        float(np.dot(distal_direction, separation_direction))
                    )
                    if target is None:
                        target_distance_mm = None
                        score = (
                            4.0 * vertical_opening
                            + 2.0 * horizontal_distal
                            + 2.0 * max(inward_direction, 0.0)
                            + orthogonality
                        )
                    else:
                        target_distance_mm = float(
                            np.linalg.norm(jaw_midpoint - target) * 1.0e3
                        )
                        score = (
                            -target_distance_mm
                            + 2.0 * vertical_opening
                            + horizontal_distal
                            + max(inward_direction, 0.0)
                        )
                    records.append(
                        {
                        "score": score,
                        "q7": q7.tolist(),
                        "tool_tip_world_m": tip.tolist(),
                        "jaw_midpoint_world_m": jaw_midpoint.tolist(),
                        "jaw_midpoint_target_distance_mm": target_distance_mm,
                        "jaw_separation_vector_m": jaw_separation.tolist(),
                        "jaw_separation_direction": separation_direction.tolist(),
                        "jaw_vertical_fraction": vertical_opening,
                        "distal_direction": distal_direction.tolist(),
                        "distal_horizontal_fraction": horizontal_distal,
                        "distal_inward_y": inward_direction,
                        "distal_jaw_orthogonality": orthogonality,
                        }
                    )

    records.sort(key=lambda item: item["score"], reverse=True)
    payload = {
        "schema": "fixedsuperbest.psm_side_grasp_kinematic_probe.v1",
        "criterion": (
            "末端水平并由 y<0 指向 +y；张开夹爪连线沿世界 z；"
            "夹爪连线与末端方向正交"
        ),
        "body_names": robot.body_names,
        "stage": ARGS.stage,
        "target_grasp_centroid_world_m": target.tolist() if target is not None else None,
        "target_frame": ARGS.target_frame if target is not None else None,
        "gripper": ARGS.gripper,
        "jaw1_local_contact_m": jaw1_local_contact.tolist(),
        "jaw2_local_contact_m": jaw2_local_contact.tolist(),
        "candidates": records[: ARGS.top_k],
    }
    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    ARGS.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["candidates"][:5], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
