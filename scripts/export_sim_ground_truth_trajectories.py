#!/usr/bin/env python3
"""从仿真数据导出组织和 PSM 的显式 3D/双目 2D 真值轨迹。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


CAMERAS = ("stereo_left", "stereo_right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def project_world_points(points_world: np.ndarray, intrinsic: np.ndarray, X_WC: np.ndarray):
    """使用 ROS 光学相机轴（x右、y下、z前）投影世界点。"""

    shape = points_world.shape
    flat = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    X_CW = np.linalg.inv(np.asarray(X_WC, dtype=np.float64))
    camera = flat @ X_CW[:3, :3].T + X_CW[:3, 3]
    z = camera[:, 2]
    uv = np.full((len(flat), 2), np.nan, dtype=np.float64)
    positive = z > 1.0e-8
    normalized = camera[positive, :2] / z[positive, None]
    uv[positive, 0] = intrinsic[0, 0] * normalized[:, 0] + intrinsic[0, 2]
    uv[positive, 1] = intrinsic[1, 1] * normalized[:, 1] + intrinsic[1, 2]
    return (
        uv.reshape(*shape[:-1], 2).astype(np.float32),
        camera.reshape(*shape[:-1], 3).astype(np.float32),
    )


def visibility_from_depth_and_mask(
    uv: np.ndarray,
    camera_xyz: np.ndarray,
    depth: np.ndarray,
    semantic_mask: np.ndarray,
    *,
    pixel_radius: int = 1,
    tolerance_base_m: float = 0.0025,
    tolerance_relative: float = 0.005,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """以 3×3 像素邻域的语义和 z-depth 一致性判定顶点可见性。"""

    height, width = depth.shape
    z = camera_xyz[:, 2]
    in_front = z > 1.0e-8
    in_frame = (
        in_front
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] <= width - 1.0)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] <= height - 1.0)
    )
    center_x = np.rint(np.nan_to_num(uv[:, 0], nan=-10000.0)).astype(np.int64)
    center_y = np.rint(np.nan_to_num(uv[:, 1], nan=-10000.0)).astype(np.int64)
    minimum_residual = np.full(len(uv), np.inf, dtype=np.float32)
    semantic_hit = np.zeros(len(uv), dtype=bool)
    visible = np.zeros(len(uv), dtype=bool)
    tolerance = tolerance_base_m + tolerance_relative * np.maximum(z, 0.0)
    for offset_y in range(-pixel_radius, pixel_radius + 1):
        for offset_x in range(-pixel_radius, pixel_radius + 1):
            x = center_x + offset_x
            y = center_y + offset_y
            valid = in_frame & (x >= 0) & (x < width) & (y >= 0) & (y < height)
            ids = np.flatnonzero(valid)
            if not len(ids):
                continue
            sampled_depth = depth[y[ids], x[ids]]
            sampled_semantic = semantic_mask[y[ids], x[ids]] > 0
            finite = np.isfinite(sampled_depth) & (sampled_depth > 0.0)
            residual = np.abs(sampled_depth - z[ids])
            accepted_residual = sampled_semantic & finite
            if np.any(accepted_residual):
                accepted_ids = ids[accepted_residual]
                minimum_residual[accepted_ids] = np.minimum(
                    minimum_residual[accepted_ids], residual[accepted_residual]
                )
            semantic_hit[ids] |= sampled_semantic
            visible[ids] |= sampled_semantic & finite & (residual <= tolerance[ids])
    return in_front, in_frame, visible, minimum_residual


def select_top_surface_nodes(tissue) -> tuple[np.ndarray, np.ndarray]:
    faces = tissue["visual_faces"].astype(np.int64)
    if "top_face_indices" in tissue.files and len(tissue["top_face_indices"]):
        top_faces = tissue["top_face_indices"].astype(np.int64)
        top_node_ids = np.unique(faces[top_faces].reshape(-1))
    else:
        # 旧预览兼容：生成器按 top、bottom 的顺序创建等量节点。
        node_count = tissue["visual_rest_points"].shape[0]
        if node_count % 2:
            raise ValueError("无法从旧数据确定组织顶面节点")
        top_node_ids = np.arange(node_count // 2, dtype=np.int64)

    evaluation_mask = np.zeros(len(top_node_ids), dtype=bool)
    if "material_uv" in tissue.files and len(tissue["material_uv"]):
        uv = tissue["material_uv"][top_node_ids]
        u_values = np.unique(np.round(uv[:, 0], 7))
        v_values = np.unique(np.round(uv[:, 1], 7))
        u_index = np.argmin(np.abs(uv[:, 0, None] - u_values[None, :]), axis=1)
        v_index = np.argmin(np.abs(uv[:, 1, None] - v_values[None, :]), axis=1)
        evaluation_mask = (u_index % 2 == 0) & (v_index % 2 == 0)
    if not evaluation_mask.any():
        evaluation_mask[::4] = True
    return top_node_ids.astype(np.int32), evaluation_mask


def main() -> None:
    root = parse_args().dataset.expanduser().resolve()
    output_3d = root / "ground_truth/trajectories_3d.npz"
    output_manifest = root / "ground_truth/trajectories.json"
    output_2d_root = root / "ground_truth/trajectories_2d"
    existing = [path for path in (output_3d, output_manifest) if path.exists()]
    existing.extend(path for path in output_2d_root.glob("*.npz") if path.exists())
    if existing:
        raise FileExistsError(f"拒绝覆盖已有轨迹真值：{existing}")
    output_2d_root.mkdir(parents=True, exist_ok=True)

    episode = read_json(root / "episode.json")
    camera_manifest = read_json(root / "cameras.json")
    tissue = np.load(root / "ground_truth/tissue_state.npz")
    psm = np.load(root / "ground_truth/psm_link_poses.npz")
    timestamps = tissue["timestamps"].astype(np.float64)
    frames = int(episode["frames"])
    if len(timestamps) != frames:
        raise ValueError("组织真值帧数与 episode.json 不一致")

    top_node_ids, evaluation_mask = select_top_surface_nodes(tissue)
    tissue_positions = tissue["simulation_positions"][:, top_node_ids].astype(np.float32)
    tissue_velocities = tissue["simulation_velocities"][:, top_node_ids].astype(np.float32)
    tissue_region_ids = tissue["simulation_region_ids"][top_node_ids].astype(np.int16)
    psm_X_WL = psm["X_WL"].astype(np.float32)
    psm_link_positions = psm_X_WL[:, :, :3, 3]
    psm_tip_positions = psm["tool_tip_positions"].astype(np.float32)

    np.savez_compressed(
        output_3d,
        timestamps=timestamps,
        tissue_node_ids=top_node_ids,
        tissue_positions_world=tissue_positions,
        tissue_velocities_world=tissue_velocities,
        tissue_region_ids=tissue_region_ids,
        tissue_evaluation_mask=evaluation_mask,
        tissue_evaluation_node_ids=top_node_ids[evaluation_mask],
        psm_link_names=psm["link_names"],
        psm_X_WL=psm_X_WL,
        psm_link_positions_world=psm_link_positions,
        psm_tool_tip_positions_world=psm_tip_positions,
        psm_q7=psm["q7"].astype(np.float32),
    )

    camera_statistics = {}
    for camera_name in CAMERAS:
        camera = camera_manifest[camera_name]
        metadata = read_json(root / camera["metadata_path"])
        intrinsic = np.asarray(metadata["K"], dtype=np.float64)
        X_WC = np.asarray(camera["X_WC_ros_optical"], dtype=np.float64)
        tissue_uv, tissue_camera = project_world_points(tissue_positions, intrinsic, X_WC)
        link_uv, link_camera = project_world_points(psm_link_positions, intrinsic, X_WC)
        tip_uv, tip_camera = project_world_points(psm_tip_positions, intrinsic, X_WC)

        tissue_in_front = np.zeros((frames, len(top_node_ids)), dtype=bool)
        tissue_in_frame = np.zeros_like(tissue_in_front)
        tissue_visible = np.zeros_like(tissue_in_front)
        tissue_depth_residual = np.full(tissue_in_front.shape, np.inf, dtype=np.float32)
        link_in_front = np.zeros(psm_link_positions.shape[:2], dtype=bool)
        link_in_frame = np.zeros_like(link_in_front)
        link_visible = np.zeros_like(link_in_front)
        link_depth_residual = np.full(link_in_front.shape, np.inf, dtype=np.float32)
        tip_in_front = np.zeros(frames, dtype=bool)
        tip_in_frame = np.zeros(frames, dtype=bool)
        tip_visible = np.zeros(frames, dtype=bool)
        tip_depth_residual = np.full(frames, np.inf, dtype=np.float32)

        for frame_index in range(frames):
            depth = np.squeeze(
                np.load(
                    root
                    / "ground_truth/depth"
                    / camera_name
                    / f"{frame_index:06d}.npy"
                )
            ).astype(np.float32)
            tissue_mask = np.asarray(
                Image.open(
                    root
                    / "ground_truth/masks/tissue"
                    / camera_name
                    / f"{frame_index:06d}.png"
                ).convert("L")
            )
            psm_mask = np.asarray(
                Image.open(
                    root
                    / "ground_truth/masks/psm"
                    / camera_name
                    / f"{frame_index:06d}.png"
                ).convert("L")
            )
            (
                tissue_in_front[frame_index],
                tissue_in_frame[frame_index],
                tissue_visible[frame_index],
                tissue_depth_residual[frame_index],
            ) = visibility_from_depth_and_mask(
                tissue_uv[frame_index], tissue_camera[frame_index], depth, tissue_mask
            )
            (
                link_in_front[frame_index],
                link_in_frame[frame_index],
                link_visible[frame_index],
                link_depth_residual[frame_index],
            ) = visibility_from_depth_and_mask(
                link_uv[frame_index], link_camera[frame_index], depth, psm_mask
            )
            tip_result = visibility_from_depth_and_mask(
                tip_uv[frame_index : frame_index + 1],
                tip_camera[frame_index : frame_index + 1],
                depth,
                psm_mask,
                pixel_radius=7,
                tolerance_base_m=0.015,
                tolerance_relative=0.01,
            )
            tip_in_front[frame_index] = tip_result[0][0]
            tip_in_frame[frame_index] = tip_result[1][0]
            tip_visible[frame_index] = tip_result[2][0]
            tip_depth_residual[frame_index] = tip_result[3][0]

        np.savez_compressed(
            output_2d_root / f"{camera_name}.npz",
            timestamps=timestamps,
            image_size_wh=np.asarray(metadata["resolution"], dtype=np.int32),
            K=intrinsic.astype(np.float64),
            X_WC_ros_optical=X_WC.astype(np.float64),
            tissue_node_ids=top_node_ids,
            tissue_evaluation_mask=evaluation_mask,
            tissue_uv_pixels=tissue_uv,
            tissue_camera_xyz_m=tissue_camera,
            tissue_in_front=tissue_in_front,
            tissue_in_frame=tissue_in_frame,
            tissue_visible=tissue_visible,
            tissue_depth_residual_m=tissue_depth_residual,
            psm_link_names=psm["link_names"],
            psm_link_uv_pixels=link_uv,
            psm_link_camera_xyz_m=link_camera,
            psm_link_in_front=link_in_front,
            psm_link_in_frame=link_in_frame,
            psm_link_visible=link_visible,
            psm_link_depth_residual_m=link_depth_residual,
            psm_tool_tip_uv_pixels=tip_uv,
            psm_tool_tip_camera_xyz_m=tip_camera,
            psm_tool_tip_in_front=tip_in_front,
            psm_tool_tip_in_frame=tip_in_frame,
            psm_tool_tip_visible=tip_visible,
            psm_tool_tip_depth_residual_m=tip_depth_residual,
        )
        camera_statistics[camera_name] = {
            "tissue_visible_fraction": float(tissue_visible.mean()),
            "evaluation_tissue_visible_fraction": float(
                tissue_visible[:, evaluation_mask].mean()
            ),
            "psm_link_origin_visible_fraction": float(link_visible.mean()),
            "psm_tool_tip_visible_fraction": float(tip_visible.mean()),
        }

    manifest = {
        "schema": "fixedsuperbest.sim_ground_truth_trajectories.v1",
        "coordinate_system_3d": "世界坐标，单位米，z 轴向上",
        "coordinate_system_2d": "图像连续像素坐标 (u=列/x, v=行/y)，原点位于左上",
        "camera_axes": "ROS optical：x 向右、y 向下、z 向前",
        "tissue_dense_surface_points": int(len(top_node_ids)),
        "tissue_evaluation_points": int(evaluation_mask.sum()),
        "visibility": (
            "点须位于视锥内，并在 3×3 像素邻域命中对应语义 mask，且渲染 z-depth "
            "与点的相机 z 相差不超过 2.5 mm + 0.5% depth；PSM 虚拟尖端位于两夹爪"
            "之间，使用 15×15 邻域与 15 mm + 1% depth 容差"
        ),
        "occluded_definition": "in_frame=True 且 visible=False",
        "files": {
            "3d": "ground_truth/trajectories_3d.npz",
            "2d_left": "ground_truth/trajectories_2d/stereo_left.npz",
            "2d_right": "ground_truth/trajectories_2d/stereo_right.npz",
        },
        "statistics": camera_statistics,
    }
    output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
