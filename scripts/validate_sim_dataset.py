#!/usr/bin/env python3
"""完整审计官方资产组织牵拉仿真数据集。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


CAMERAS = ("stereo_left", "stereo_right")
LABELS = ("tissue", "psm", "liver")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    root = parse_args().dataset.expanduser().resolve()
    output = root / "validation_report.json"
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有验证报告：{output}")
    episode = read_json(root / "episode.json")
    frames = int(episode["frames"])
    fps = int(episode["fps"])
    robots = read_json(root / "robots.json")["PSM1"]
    cameras = read_json(root / "cameras.json")

    gates: dict[str, bool] = {}
    statistics: dict[str, object] = {}
    timestamps = np.asarray(robots["states_timestamps"], dtype=np.float64)
    q7_json = np.asarray([state["q"] for state in robots["states"]], dtype=np.float32)
    gates["robot_state_count"] = len(q7_json) == frames
    gates["timestamps_strictly_increasing"] = bool(np.all(np.diff(timestamps) > 0.0))
    gates["timestamps_match_fps"] = bool(
        np.allclose(timestamps, np.arange(frames) / fps, atol=1.0e-9)
    )
    gates["camera_manifest_exact"] = set(cameras) == set(CAMERAS)

    canonical_scan_info = episode.get("canonical_scan")
    gates["canonical_scan_declared"] = isinstance(canonical_scan_info, dict)
    if isinstance(canonical_scan_info, dict):
        scan_root = root / canonical_scan_info["path"]
        scan_manifest = read_json(scan_root / "cameras.json")
        scan_views = scan_manifest.get("views", [])
        scan_count = int(scan_manifest.get("view_count", -1))
        scan_rgb = sorted((scan_root / "rgb").glob("*.png"))
        scan_depth = sorted((scan_root / "depth").glob("*.npy"))
        scan_masks = sorted((scan_root / "mask").glob("*.png"))
        gates["canonical_scan_count"] = bool(
            scan_count >= 18
            and len(scan_views) == scan_count
            and len(scan_rgb) == scan_count
            and len(scan_depth) == scan_count
            and len(scan_masks) == scan_count
        )
        elevations = np.asarray(
            [float(view["elevation_deg"]) for view in scan_views], dtype=np.float64
        )
        gates["canonical_scan_top_bottom_sides"] = bool(
            np.any(elevations == 90.0)
            and np.any(elevations == -90.0)
            and np.any((elevations > 0.0) & (elevations < 89.0))
            and np.any((elevations < 0.0) & (elevations > -89.0))
        )
        scan_valid = True
        scan_mask_pixels = []
        for view, depth_path, mask_path in zip(scan_views, scan_depth, scan_masks):
            depth = np.load(depth_path)
            mask = np.asarray(Image.open(mask_path).convert("L")) > 0
            valid_depth = depth[mask]
            scan_valid &= bool(
                mask.any()
                and valid_depth.size > 0
                and np.isfinite(valid_depth).all()
                and np.all(valid_depth > 0.0)
                and np.isfinite(np.asarray(view["K"], dtype=np.float64)).all()
                and np.isfinite(
                    np.asarray(view["X_WC_ros_optical"], dtype=np.float64)
                ).all()
            )
            scan_mask_pixels.append(int(mask.sum()))
        gates["canonical_scan_rgbd_masks_valid"] = scan_valid
        gates["canonical_scan_no_stiffness_labels"] = bool(
            scan_manifest.get("contains_material_region_or_stiffness_labels") is False
        )
        statistics["canonical_scan"] = {
            "view_count": scan_count,
            "mask_pixels_min": min(scan_mask_pixels) if scan_mask_pixels else 0,
            "mask_pixels_max": max(scan_mask_pixels) if scan_mask_pixels else 0,
            "elevations_deg": sorted(set(elevations.tolist())),
        }
    else:
        gates["canonical_scan_count"] = False
        gates["canonical_scan_top_bottom_sides"] = False
        gates["canonical_scan_rgbd_masks_valid"] = False
        gates["canonical_scan_no_stiffness_labels"] = False

    psm = np.load(root / "ground_truth/psm_link_poses.npz")
    task_psm_path = root / "task_inputs/psm_link_poses.npz"
    task_phases_path = root / "task_inputs/phases.json"
    gates["task_psm_input_present"] = task_psm_path.is_file()
    gates["task_grasp_schedule_present"] = task_phases_path.is_file()
    if task_psm_path.is_file():
        task_psm = np.load(task_psm_path)
        gates["task_psm_matches_recorded_control"] = bool(
            set(task_psm.files) == set(psm.files)
            and all(np.array_equal(task_psm[key], psm[key]) for key in psm.files)
        )
    else:
        gates["task_psm_matches_recorded_control"] = False
    if task_phases_path.is_file():
        gates["task_grasp_schedule_matches_recorded_control"] = bool(
            read_json(task_phases_path)
            == read_json(root / "ground_truth/phases.json")
        )
    else:
        gates["task_grasp_schedule_matches_recorded_control"] = False
    tissue = np.load(root / "ground_truth/tissue_state.npz")
    q7_gt = psm["q7"]
    positions = tissue["simulation_positions"]
    rest = tissue["simulation_rest_local"]
    anchors = tissue["anchor_mask"]
    support_mode = str(
        np.asarray(tissue["support_mode"]).item()
        if "support_mode" in tissue.files
        else "edge_hard"
    )
    coupling_weights = tissue["grasp_coupling_weights"]
    grasp_nodes = tissue["grasp_mask"]
    gates["q7_json_matches_ground_truth"] = bool(np.array_equal(q7_json, q7_gt))
    gates["psm_arrays_finite"] = bool(
        np.isfinite(psm["X_WL"]).all() and np.isfinite(psm["tool_tip_positions"]).all()
    )
    gates["tissue_arrays_finite"] = bool(
        np.isfinite(positions).all() and np.isfinite(tissue["simulation_velocities"]).all()
    )
    initial_error = np.linalg.norm(positions[0] - rest, axis=1)
    anchor_error = (
        np.linalg.norm(positions[:, anchors] - positions[0:1, anchors], axis=2)
        if np.any(anchors)
        else np.zeros((frames, 0), dtype=np.float32)
    )
    displacement = np.linalg.norm(positions - positions[0:1], axis=2)
    frame_motion = np.linalg.norm(np.diff(positions, axis=0), axis=2)
    gates["initial_tissue_unpolluted_0p05mm"] = float(initial_error.max()) <= 5.0e-5
    gates["support_policy_matches_dataset"] = bool(
        (support_mode == "free" and not np.any(anchors))
        or (
            support_mode == "edge_hard"
            and np.any(anchors)
            and float(anchor_error.max()) <= 5.0e-5
        )
    )
    gates["nontrivial_tissue_motion_10mm"] = float(displacement.max()) >= 0.010
    gates["continuous_tissue_motion_10mm_per_frame"] = float(frame_motion.max()) <= 0.010
    gates["smooth_grasp_coupling_field"] = bool(
        np.isfinite(coupling_weights).all()
        and float(coupling_weights.min()) == 0.0
        and float(coupling_weights.max()) == 1.0
        and np.count_nonzero(
            (coupling_weights > 0.0) & (coupling_weights < 1.0)
        ) >= 50
        and len(np.unique(coupling_weights)) >= 20
    )
    grasp_center = rest[grasp_nodes].mean(axis=0)
    contact_tip = np.asarray(tissue["contact_tip_world"], dtype=np.float64)
    contact_offset = contact_tip - grasp_center
    gates["psm_tip_aligned_with_grasp_marker"] = bool(
        np.linalg.norm(contact_offset[:2]) <= 0.005
        and abs(float(contact_offset[2])) <= 0.012
    )
    region_ids = tissue["simulation_tet_region_ids"]
    tet_youngs = tissue["youngs_modulus_pa_per_simulation_tet"]
    gates["ten_nonempty_material_regions"] = bool(
        np.array_equal(np.unique(region_ids), np.arange(10))
    )
    gates["ten_distinct_regional_youngs_moduli"] = len(np.unique(tet_youngs)) == 10
    statistics.update(
        {
            "initial_tissue_max_error_mm": float(initial_error.max() * 1000.0),
            "support_mode": support_mode,
            "hard_anchor_nodes": int(np.count_nonzero(anchors)),
            "anchor_max_error_mm": (
                float(anchor_error.max() * 1000.0) if anchor_error.size else 0.0
            ),
            "tissue_max_displacement_mm": float(displacement.max() * 1000.0),
            "tissue_max_interframe_motion_mm": float(frame_motion.max() * 1000.0),
            "grasp_coupling_core_nodes": int(np.count_nonzero(coupling_weights == 1.0)),
            "grasp_coupling_transition_nodes": int(
                np.count_nonzero((coupling_weights > 0.0) & (coupling_weights < 1.0))
            ),
            "grasp_marker_center_world_m": grasp_center.tolist(),
            "closed_psm_tip_minus_marker_mm": (
                contact_offset * 1000.0
            ).tolist(),
            "psm_tip_axis_range_mm": (
                (psm["tool_tip_positions"].max(axis=0) - psm["tool_tip_positions"].min(axis=0))
                * 1000.0
            ).tolist(),
            "material_region_tet_counts": np.bincount(
                region_ids.astype(np.int64), minlength=10
            ).tolist(),
            "regional_youngs_modulus_pa": np.unique(tet_youngs).tolist(),
        }
    )

    trajectories_3d_path = root / "ground_truth/trajectories_3d.npz"
    trajectory_manifest_path = root / "ground_truth/trajectories.json"
    gates["trajectory_manifest_present"] = trajectory_manifest_path.is_file()
    gates["trajectory_3d_present"] = trajectories_3d_path.is_file()
    if trajectories_3d_path.is_file():
        trajectories_3d = np.load(trajectories_3d_path)
        tissue_3d = trajectories_3d["tissue_positions_world"]
        psm_tip_3d = trajectories_3d["psm_tool_tip_positions_world"]
        gates["trajectory_3d_shapes"] = bool(
            tissue_3d.shape[0] == frames
            and tissue_3d.shape[1] >= 200
            and psm_tip_3d.shape == (frames, 3)
        )
        gates["trajectory_3d_finite"] = bool(
            np.isfinite(tissue_3d).all() and np.isfinite(psm_tip_3d).all()
        )
        gates["trajectory_evaluation_points_present"] = bool(
            int(trajectories_3d["tissue_evaluation_mask"].sum()) >= 100
        )
    else:
        gates["trajectory_3d_shapes"] = False
        gates["trajectory_3d_finite"] = False
        gates["trajectory_evaluation_points_present"] = False

    camera_stats = {}
    decoded_first: dict[str, np.ndarray] = {}
    for camera in CAMERAS:
        rgb_paths = sorted((root / "rgb" / camera).glob("*.png"))
        gates[f"{camera}_rgb_count"] = len(rgb_paths) == frames
        metadata = read_json(root / cameras[camera]["metadata_path"])
        gates[f"{camera}_metadata_count"] = len(metadata["timestamps"]) == frames
        gates[f"{camera}_intrinsics_finite"] = bool(np.isfinite(metadata["K"]).all())

        label_minimums = {}
        for label in LABELS:
            paths = sorted((root / "ground_truth/masks" / label / camera).glob("*.png"))
            counts = [int(np.count_nonzero(np.asarray(Image.open(path)))) for path in paths]
            gates[f"{camera}_{label}_mask_count"] = len(paths) == frames
            if label == "liver":
                # Close-up tissue may fully occlude the liver.  Its semantic
                # sequence must still be complete, but visibility is not an
                # input or quality requirement for tissue reconstruction.
                gates[f"{camera}_{label}_occlusion_allowed"] = len(paths) == frames
            else:
                gates[f"{camera}_{label}_visible_every_frame"] = bool(
                    counts and min(counts) > 0
                )
            label_minimums[label] = {"min": min(counts), "max": max(counts)}

        depth_paths = sorted((root / "ground_truth/depth" / camera).glob("*.npy"))
        depth_minimum, depth_maximum = float("inf"), -float("inf")
        depth_valid = len(depth_paths) == frames
        for path in depth_paths:
            depth = np.load(path)
            finite = np.isfinite(depth)
            depth_valid &= bool(finite.any() and np.all(depth[finite] > 0.0))
            if finite.any():
                depth_minimum = min(depth_minimum, float(depth[finite].min()))
                depth_maximum = max(depth_maximum, float(depth[finite].max()))
        gates[f"{camera}_depth_count_and_valid"] = depth_valid

        colorfulness = []
        tissue_colorfulness = []
        tissue_achromatic_highlight_fraction = []
        tissue_luminance_std = []
        tissue_mask_paths = sorted(
            (root / "ground_truth/masks/tissue" / camera).glob("*.png")
        )
        for path, tissue_mask_path in zip(rgb_paths, tissue_mask_paths):
            rgb = np.asarray(Image.open(path).convert("RGB"))
            colorfulness.append(float(np.mean(rgb.max(axis=2) - rgb.min(axis=2))))
            tissue_mask = np.asarray(Image.open(tissue_mask_path).convert("L")) > 0
            tissue_rgb = rgb[tissue_mask].astype(np.float32)
            if tissue_rgb.size:
                channel_range = tissue_rgb.max(axis=1) - tissue_rgb.min(axis=1)
                luminance = (
                    0.2126 * tissue_rgb[:, 0]
                    + 0.7152 * tissue_rgb[:, 1]
                    + 0.0722 * tissue_rgb[:, 2]
                )
                tissue_colorfulness.append(float(channel_range.mean()))
                tissue_luminance_std.append(float(luminance.std()))
                achromatic_highlight = (luminance >= 242.0) & (channel_range <= 18.0)
                tissue_achromatic_highlight_fraction.append(
                    float(achromatic_highlight.mean())
                )
        gates[f"{camera}_is_color"] = bool(colorfulness and min(colorfulness) >= 5.0)
        gates[f"{camera}_tissue_has_visible_color"] = bool(
            tissue_colorfulness and float(np.median(tissue_colorfulness)) >= 18.0
        )
        gates[f"{camera}_tissue_low_achromatic_highlight"] = bool(
            tissue_achromatic_highlight_fraction
            and float(np.max(tissue_achromatic_highlight_fraction)) <= 0.08
        )
        gates[f"{camera}_tissue_has_intensity_texture"] = bool(
            tissue_luminance_std and float(np.median(tissue_luminance_std)) >= 4.0
        )

        capture = cv2.VideoCapture(str(root / cameras[camera]["video_path"]))
        decoded = []
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            decoded.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        capture.release()
        gates[f"{camera}_video_decodes_all_frames"] = len(decoded) == frames
        first_png = np.asarray(Image.open(rgb_paths[0]).convert("RGB"))
        decoded_first[camera] = decoded[0] if decoded else np.zeros_like(first_png)
        gui_error = np.abs(decoded_first[camera].astype(np.int16) - first_png.astype(np.int16))
        gates[f"{camera}_gui_video_quality"] = bool(
            # yuv420p 会在极少数高对比红/白边缘产生较大的单通道峰值误差；
            # 用严格平均误差守住整体清晰度，同时允许这种局部色度重采样峰值。
            float(gui_error.mean()) <= 2.0 and int(gui_error.max()) <= 96
        )
        lossless_path = root / "videos" / f"{camera}_lossless_rgb.mp4"
        lossless_capture = cv2.VideoCapture(str(lossless_path))
        lossless_ok, lossless_bgr = lossless_capture.read()
        lossless_capture.release()
        lossless_rgb = (
            cv2.cvtColor(lossless_bgr, cv2.COLOR_BGR2RGB) if lossless_ok else None
        )
        gates[f"{camera}_lossless_archive_exact"] = bool(
            lossless_rgb is not None and np.array_equal(lossless_rgb, first_png)
        )
        camera_stats[camera] = {
            "mask_pixels": label_minimums,
            "depth_range_m": [depth_minimum, depth_maximum],
            "colorfulness_min_mean": min(colorfulness),
            "tissue_colorfulness_median": float(np.median(tissue_colorfulness)),
            "tissue_luminance_std_median": float(np.median(tissue_luminance_std)),
            "tissue_achromatic_highlight_fraction_max": float(
                np.max(tissue_achromatic_highlight_fraction)
            ),
            "decoded_video_frames": len(decoded),
            "gui_video_first_frame_mean_error": float(gui_error.mean()),
            "gui_video_first_frame_max_error": int(gui_error.max()),
        }

        trajectory_2d_path = root / "ground_truth/trajectories_2d" / f"{camera}.npz"
        gates[f"{camera}_trajectory_2d_present"] = trajectory_2d_path.is_file()
        if trajectory_2d_path.is_file():
            trajectory_2d = np.load(trajectory_2d_path)
            tissue_uv = trajectory_2d["tissue_uv_pixels"]
            tissue_in_frame = trajectory_2d["tissue_in_frame"]
            tissue_visible = trajectory_2d["tissue_visible"]
            gates[f"{camera}_trajectory_2d_shapes"] = bool(
                tissue_uv.shape[0] == frames
                and tissue_uv.shape[1] >= 200
                and tissue_visible.shape == tissue_uv.shape[:2]
            )
            gates[f"{camera}_trajectory_2d_finite_in_frame"] = bool(
                np.isfinite(tissue_uv[tissue_in_frame]).all()
            )
            gates[f"{camera}_trajectory_visible_points"] = bool(
                np.any(tissue_visible)
            )
        else:
            gates[f"{camera}_trajectory_2d_shapes"] = False
            gates[f"{camera}_trajectory_2d_finite_in_frame"] = False
            gates[f"{camera}_trajectory_visible_points"] = False

    stereo_difference = np.abs(
        decoded_first["stereo_left"].astype(np.int16)
        - decoded_first["stereo_right"].astype(np.int16)
    )
    gates["stereo_views_are_distinct"] = float(stereo_difference.mean()) > 0.1
    statistics["first_stereo_mean_absolute_rgb_difference"] = float(stereo_difference.mean())
    statistics["cameras"] = camera_stats
    statistics["gui_tissue_asset_present_optional"] = (
        root / "gui_assets/tissue_fixedsuperbest.npz"
    ).is_file()
    gates["render_metric_evaluator_present"] = (
        Path(__file__).resolve().parent / "evaluate_sim_rendering_metrics.py"
    ).is_file()
    gates["chinese_readme_present"] = (root / "README.md").is_file()

    report = {
        "dataset": str(root),
        "frames": frames,
        "fps": fps,
        "gates": gates,
        "statistics": statistics,
        "passed": bool(all(gates.values())),
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"数据集验证失败：{failed}")


if __name__ == "__main__":
    main()
