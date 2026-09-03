#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


REPO = Path(__file__).resolve().parents[1]
TRACK_ROOT = REPO / "data/super/psm_tracking"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project strict, paper, and hybrid PSM poses onto grasp5 frames."
    )
    parser.add_argument(
        "--poses",
        type=Path,
        default=TRACK_ROOT / "visual_poses_video.npz",
    )
    parser.add_argument(
        "--states", type=Path, default=TRACK_ROOT / "tracking_states.npz"
    )
    parser.add_argument(
        "--frames",
        type=str,
        default="0,160,320,480,640,800,960,1120,1280,1440",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=TRACK_ROOT / "projection_comparison"
    )
    parser.add_argument(
        "--include-main-shaft",
        action="store_true",
        help="Also project the long tool_main shaft. Default comparison is tip-only.",
    )
    return parser.parse_args()


def project_gaussians(
    local_means: np.ndarray,
    link_ids: np.ndarray,
    poses: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_camera = np.empty_like(local_means, dtype=np.float64)
    for link_index, pose in enumerate(poses):
        selected = link_ids == link_index
        rotation = Rotation.from_quat(pose[3:]).as_matrix()
        points_camera[selected] = (
            local_means[selected] @ rotation.T + pose[:3]
        )
    depth = points_camera[:, 2]
    uv = np.empty((len(points_camera), 2), dtype=np.float64)
    uv[:, 0] = K[0, 0] * points_camera[:, 0] / depth + K[0, 2]
    uv[:, 1] = K[1, 1] * points_camera[:, 1] / depth + K[1, 2]
    valid = (
        (depth > 1e-5)
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < 1920)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < 1080)
    )
    return uv, depth, valid


def draw_projection(
    image: np.ndarray,
    mask: np.ndarray,
    local_means: np.ndarray,
    link_ids: np.ndarray,
    poses: np.ndarray,
    K: np.ndarray,
    color: tuple[int, int, int],
    title: str,
) -> tuple[np.ndarray, dict[str, float]]:
    output = image.copy()
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    uv, depth, valid = project_gaussians(
        local_means, link_ids, poses, K
    )
    valid_ids = np.flatnonzero(valid)
    valid_ids = valid_ids[np.argsort(depth[valid_ids])[::-1]]
    palette_scale = np.linspace(0.72, 1.0, 7)
    for point_id in valid_ids:
        link_id = int(link_ids[point_id])
        point_color = tuple(
            int(channel * palette_scale[link_id]) for channel in color
        )
        point = tuple(np.rint(uv[point_id]).astype(int))
        cv2.circle(output, point, 2, point_color, -1, lineType=cv2.LINE_AA)

    mask_distance = cv2.distanceTransform(
        (~mask).astype(np.uint8), cv2.DIST_L2, 3
    )
    valid_uv = np.rint(uv[valid]).astype(np.int64)
    outside_distance = mask_distance[valid_uv[:, 1], valid_uv[:, 0]]
    inside_fraction = float(np.mean(mask[valid_uv[:, 1], valid_uv[:, 0]]))
    stats = {
        "valid_projected_gaussians": int(len(valid_uv)),
        "inside_mask_fraction": inside_fraction,
        "outside_distance_px_p50": float(np.percentile(outside_distance, 50)),
        "outside_distance_px_p95": float(np.percentile(outside_distance, 95)),
    }
    cv2.rectangle(output, (0, 0), (880, 80), (0, 0, 0), -1)
    cv2.putText(
        output,
        title,
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        (
            f"points inside SAM={inside_fraction:.3f}; outside distance "
            f"P50/P95={stats['outside_distance_px_p50']:.1f}/"
            f"{stats['outside_distance_px_p95']:.1f}px"
        ),
        (14, 63),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output, stats


def label_reference(
    image: np.ndarray,
    mask: np.ndarray,
    frame_index: int,
    tip_keypoints: np.ndarray | None = None,
) -> np.ndarray:
    output = image.copy()
    tint = np.zeros_like(output)
    tint[mask] = (0, 180, 0)
    output = cv2.addWeighted(output, 1.0, tint, 0.25, 0.0)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    if tip_keypoints is not None:
        keypoints = np.asarray(tip_keypoints) * 2.0
    elif frame_index == 0:
        keypoint_path = (
            TRACK_ROOT / "online_videos/grasp5/PSM1_keypoints.txt"
        )
        keypoints = np.loadtxt(keypoint_path, ndmin=2) * 2.0
    else:
        keypoints = np.empty((0, 2), dtype=np.float32)
    if len(keypoints):
        for point_index, point in enumerate(keypoints):
            center = tuple(np.rint(point).astype(int))
            cv2.circle(output, center, 8, (255, 0, 0), -1)
            cv2.putText(
                output,
                f"tracked tip {point_index + 1}",
                (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
    cv2.rectangle(output, (0, 0), (760, 80), (0, 0, 0), -1)
    cv2.putText(
        output,
        f"REFERENCE frame {frame_index}: SurgicalSAM2 mask (green)",
        (14, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def main() -> None:
    args = parse_args()
    if not args.poses.exists() or not args.states.exists():
        raise FileNotFoundError(
            f"Missing tracking output: {args.poses} or {args.states}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    poses_asset = np.load(args.poses)
    states_asset = np.load(args.states)
    surface = np.load(REPO / "data/super/psm_robot/psm_surface_gaussians.npz")
    strict = np.load(REPO / "data/super/psm_robot/psm_lnd_pose_driver.npz")
    metadata = json.loads(
        (
            REPO
            / "data/super/grasp5_offline_demo/videos/stereo_left.json"
        ).read_text(encoding="utf-8")
    )
    K = np.asarray(metadata["K"], dtype=np.float64)
    video_timestamps = poses_asset["video_timestamps"]
    strict_timestamps = strict["timestamps"]
    strict_indices_right = np.searchsorted(
        strict_timestamps, video_timestamps, side="left"
    )
    strict_indices_right = np.clip(
        strict_indices_right, 0, len(strict_timestamps) - 1
    )
    strict_indices_left = np.maximum(strict_indices_right - 1, 0)
    choose_left = (
        np.abs(video_timestamps - strict_timestamps[strict_indices_left])
        <= np.abs(strict_timestamps[strict_indices_right] - video_timestamps)
    )
    strict_indices = np.where(
        choose_left, strict_indices_left, strict_indices_right
    )
    surface_keep = np.ones(len(surface["means"]), dtype=bool)
    if not args.include_main_shaft:
        surface_link_names = surface["link_names"].tolist()
        main_link_id = surface_link_names.index("PSM1_tool_main_link")
        surface_keep = surface["link_ids"] != main_link_id
    local_means = surface["means"][surface_keep]
    link_ids = surface["link_ids"][surface_keep]
    frame_indices = [int(value) for value in args.frames.split(",") if value]
    frame_indices = [index for index in frame_indices if index < len(video_timestamps)]
    if not frame_indices:
        frame_indices = list(range(min(2, len(video_timestamps))))
    mask_shape = tuple(states_asset["mask_shape"].tolist())
    report: dict[str, dict[str, dict[str, float]]] = {}
    sheets = []
    for frame_index in frame_indices:
        image_path = (
            REPO
            / f"data/super/grasp5_native/rgb/{frame_index:06d}-left.png"
        )
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        mask_half = np.unpackbits(
            states_asset["masks_packbits"][frame_index]
        )[: np.prod(mask_shape)].reshape(mask_shape).astype(bool)
        mask = cv2.resize(
            mask_half.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        tracked_tips = (
            states_asset["tip_keypoints"][frame_index]
            if "tip_keypoints" in states_asset.files
            else None
        )
        panels = [
            label_reference(image, mask, frame_index, tracked_tips)
        ]
        frame_report = {}
        variants = [
            (
                "strict",
                strict["poses_rect_camera_xyz_xyzw"][strict_indices[frame_index]],
                (0, 165, 255),
                "CURRENT strict LND (orange points)",
            ),
            (
                "paper",
                poses_asset["pure_poses_rect_camera_xyz_xyzw"][frame_index],
                (255, 0, 255),
                "PURE PAPER image tracking (magenta points)",
            ),
        ]
        if "hybrid_poses_rect_camera_xyz_xyzw" in poses_asset.files:
            variants.append(
                (
                    "hybrid",
                    poses_asset["hybrid_poses_rect_camera_xyz_xyzw"][frame_index],
                    (255, 255, 0),
                    "HYBRID LND + image residual (cyan points)",
                )
            )
        for name, poses, color, title in variants:
            panel, stats = draw_projection(
                image,
                mask,
                local_means,
                link_ids,
                poses,
                K,
                color,
                title,
            )
            panels.append(panel)
            frame_report[name] = stats
        if len(panels) == 3:
            note = np.zeros_like(image)
            cv2.putText(
                note,
                "PAPER-EXACT: no hybrid estimator in this result",
                (120, 500),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                note,
                "Native four-part paper CAD overlays are saved separately.",
                (120, 555),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            panels.append(note)
        panels = [cv2.resize(panel, (960, 540)) for panel in panels]
        grid = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
        output_path = args.output_dir / f"frame{frame_index:06d}_comparison.png"
        cv2.imwrite(str(output_path), grid)
        sheets.append(cv2.resize(grid, (960, 540)))
        report[str(frame_index)] = frame_report
        print(output_path)
    contact_sheet = np.vstack(sheets)
    cv2.imwrite(str(args.output_dir / "contact_sheet.png"), contact_sheet)
    (args.output_dir / "projection_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
