#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from build_super_psm_lnd_intermediates import (
    lnd_forward_kinematics,
    parse_handeye_yaml,
    parse_lnd_json,
)


REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from examples.embodied_environments.super_embodied.psm_lnd_kinematics import (  # noqa: E402
    PSMLNDKinematics,
)
from examples.embodied_environments.super_embodied.super_embodied import (  # noqa: E402
    PSM_LND_MODEL_PATH,
    PSM_LND_POSE_REPORT_PATH,
    TABLE_FRAME_PATH,
    apply_psm_pose_driver_joint_offsets,
)

GROUP_COLORS = {
    "grip": (80, 80, 255),
    "roll": (0, 170, 255),
    "pitch": (255, 110, 0),
    "ee": (70, 255, 70),
}
SKELETON_COLORS = [
    (0, 255, 255),
    (0, 190, 255),
    (0, 190, 255),
    (70, 255, 70),
    (255, 80, 255),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project raw SuPer LND features and strict LND-driven dVRK surface "
            "Gaussians onto rectified left PNG frames."
        )
    )
    parser.add_argument("--frames", type=int, nargs="+", default=None)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--lnd", type=Path, default=REPO / "data/LND.json")
    parser.add_argument(
        "--handeye", type=Path, default=REPO / "data/handeye.yaml"
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO / "data/super/grasp5_native/calib_rectified.json",
    )
    parser.add_argument(
        "--joints",
        type=Path,
        default=REPO / "data/super/grasp5_native/joints.json",
    )
    parser.add_argument(
        "--camera-metadata",
        type=Path,
        default=REPO
        / "data/super/grasp5_offline_demo/videos/stereo_left.json",
    )
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=REPO / "data/super/grasp5_native/rgb",
    )
    parser.add_argument(
        "--gaussians",
        type=Path,
        default=REPO / "data/super/psm_robot/psm_surface_gaussians.npz",
    )
    parser.add_argument(
        "--pose-driver",
        type=Path,
        default=REPO / "data/super/psm_robot/psm_lnd_pose_driver.npz",
    )
    parser.add_argument(
        "--surface-label",
        default="STRICT LND pose + dVRK surface (cyan)",
        help="Title used for the projected Gaussian surface overlay.",
    )
    parser.add_argument(
        "--tip-only",
        action="store_true",
        help="Project the six GUI tip links and omit the long tool_main shaft.",
    )
    parser.add_argument(
        "--roll-offset-deg",
        type=float,
        default=0.0,
        help="Apply the same tracked-driver self-spin offset used by the GUI.",
    )
    parser.add_argument(
        "--jaw-offset-deg",
        type=float,
        default=0.0,
        help="Apply the same symmetric jaw-opening offset used by the GUI.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "data/super/psm_robot/lnd_png_overlay_10",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_rectified_camera_from_psm(
    handeye: dict[str, np.ndarray], calibration: dict
) -> np.ndarray:
    rotation_raw_psm, _ = cv2.Rodrigues(handeye["PSM1_rvec"].reshape(3, 1))
    transform_raw_psm = np.eye(4, dtype=np.float64)
    transform_raw_psm[:3, :3] = rotation_raw_psm
    transform_raw_psm[:3, 3] = handeye["PSM1_tvec"] * 0.001

    transform_rect_raw = np.eye(4, dtype=np.float64)
    transform_rect_raw[:3, :3] = np.asarray(
        calibration["R1"], dtype=np.float64
    ).reshape(3, 3)
    return transform_rect_raw @ transform_raw_psm


def transform_lnd_point(
    transform_rect_psm: np.ndarray,
    link_transforms: dict[int, np.ndarray],
    link: int,
    position: list[float],
) -> np.ndarray:
    point_psm = (
        link_transforms[int(link)]
        @ np.r_[np.asarray(position, dtype=np.float64), 1.0]
    )[:3]
    return (transform_rect_psm @ np.r_[point_psm, 1.0])[:3]


def project_point(point: np.ndarray, intrinsics: np.ndarray) -> tuple[int, int] | None:
    if point[2] <= 1.0e-6:
        return None
    pixel = intrinsics @ (point / point[2])
    return int(round(float(pixel[0]))), int(round(float(pixel[1])))


def project_points(
    points: np.ndarray, intrinsics: np.ndarray, width: int, height: int
) -> np.ndarray:
    positive = points[:, 2] > 1.0e-6
    normalized = points[positive] / points[positive, 2:3]
    pixels = (intrinsics @ normalized.T).T[:, :2]
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    return pixels[inside]


def draw_lnd_primitives(
    image: np.ndarray,
    lnd: dict,
    link_transforms: dict[int, np.ndarray],
    transform_rect_psm: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, dict]:
    result = image.copy()
    height, width = result.shape[:2]
    visible_features = 0
    visible_segments = 0

    for segment_index, segment in enumerate(lnd.get("skeleton_structure", [])):
        point_a = transform_lnd_point(
            transform_rect_psm,
            link_transforms,
            int(segment["link1"]),
            segment["position1"],
        )
        point_b = transform_lnd_point(
            transform_rect_psm,
            link_transforms,
            int(segment["link2"]),
            segment["position2"],
        )
        pixel_a = project_point(point_a, intrinsics)
        pixel_b = project_point(point_b, intrinsics)
        if pixel_a is None or pixel_b is None:
            continue
        color = SKELETON_COLORS[min(segment_index, len(SKELETON_COLORS) - 1)]
        cv2.line(result, pixel_a, pixel_b, color, 4, cv2.LINE_AA)
        if cv2.clipLine((0, 0, width, height), pixel_a, pixel_b)[0]:
            visible_segments += 1

    for feature in lnd.get("point_features", []):
        point = transform_lnd_point(
            transform_rect_psm,
            link_transforms,
            int(feature["link"]),
            feature["position"],
        )
        pixel = project_point(point, intrinsics)
        if pixel is None:
            continue
        u, v = pixel
        if not (0 <= u < width and 0 <= v < height):
            continue
        group = feature["name"].split("_", 1)[0]
        color = GROUP_COLORS.get(group, (255, 255, 255))
        cv2.circle(result, pixel, 7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(result, pixel, 5, color, -1, cv2.LINE_AA)
        visible_features += 1

    return result, {
        "visible_point_features": visible_features,
        "visible_skeleton_segments": visible_segments,
    }


def pose_to_matrix(pose_xyz_xyzw: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(pose_xyz_xyzw[3:]).as_matrix()
    transform[:3, 3] = pose_xyz_xyzw[:3]
    return transform


def lnd_driven_surface_points(
    local_means: np.ndarray,
    gaussian_link_ids: np.ndarray,
    asset_link_names: list[str],
    driver_link_index: dict[str, int],
    poses: np.ndarray,
) -> np.ndarray:
    points = np.empty_like(local_means, dtype=np.float64)
    for asset_link_index, link_name in enumerate(asset_link_names):
        transform = pose_to_matrix(poses[driver_link_index[link_name]])
        mask = gaussian_link_ids == asset_link_index
        points[mask] = (
            local_means[mask] @ transform[:3, :3].T + transform[:3, 3]
        )
    return points


def draw_surface_projection(image: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    layer = image.copy()
    for u, v in np.rint(pixels).astype(np.int32):
        cv2.circle(layer, (int(u), int(v)), 2, (255, 255, 0), -1, cv2.LINE_AA)
    return cv2.addWeighted(layer, 0.72, image, 0.28, 0.0)


def projected_feature_pixels(
    lnd: dict,
    link_transforms: dict[int, np.ndarray],
    transform_rect_psm: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    pixels = []
    for feature in lnd.get("point_features", []):
        point = transform_lnd_point(
            transform_rect_psm,
            link_transforms,
            int(feature["link"]),
            feature["position"],
        )
        if point[2] > 1.0e-6:
            pixel = intrinsics @ (point / point[2])
            pixels.append(pixel[:2])
    return np.asarray(pixels, dtype=np.float64)


def crop_around_instrument(
    image: np.ndarray,
    feature_pixels: np.ndarray,
    crop_width: int = 960,
    crop_height: int = 640,
) -> np.ndarray:
    height, width = image.shape[:2]
    center = np.median(feature_pixels, axis=0)
    x0 = int(np.clip(round(center[0] - crop_width / 2), 0, width - crop_width))
    y0 = int(np.clip(round(center[1] - crop_height / 2), 0, height - crop_height))
    return image[y0 : y0 + crop_height, x0 : x0 + crop_width]


def interpolate_joint_state(
    timestamp: float,
    joint_timestamps: np.ndarray,
    joint_positions: np.ndarray,
) -> np.ndarray:
    upper = int(np.searchsorted(joint_timestamps, timestamp, side="left"))
    upper = int(np.clip(upper, 0, len(joint_timestamps) - 1))
    lower = max(0, upper - 1)
    interval = joint_timestamps[upper] - joint_timestamps[lower]
    if interval <= 0.0:
        return joint_positions[lower].copy()
    weight = float((timestamp - joint_timestamps[lower]) / interval)
    return joint_positions[lower] * (1.0 - weight) + joint_positions[upper] * weight


def annotate(
    image: np.ndarray,
    label: str,
    video_frame: int,
    state_index: int,
    signed_time_error_ms: float,
) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (920, 70), (0, 0, 0), -1)
    cv2.putText(
        result,
        f"{label} | png={video_frame} joint={state_index}",
        (16, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        f"joint_time - image_time = {signed_time_error_ms:+.3f} ms",
        (16, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (210, 210, 210),
        2,
        cv2.LINE_AA,
    )
    return result


def write_contact_sheet(
    images: list[np.ndarray],
    path: Path,
    tile_width: int = 480,
    tile_height: int = 270,
) -> None:
    tiles = [
        cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        for image in images
    ]
    columns = min(5, len(tiles))
    rows = int(np.ceil(len(tiles) / columns))
    blank = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
    while len(tiles) < rows * columns:
        tiles.append(blank)
    sheet_rows = [
        np.concatenate(tiles[row * columns : (row + 1) * columns], axis=1)
        for row in range(rows)
    ]
    cv2.imwrite(str(path), np.concatenate(sheet_rows, axis=0))


def main() -> None:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    lnd = parse_lnd_json(args.lnd)
    handeye = parse_handeye_yaml(args.handeye)
    calibration = read_json(args.calibration)
    joints = read_json(args.joints)
    camera_metadata = read_json(args.camera_metadata)
    transform_rect_psm = build_rectified_camera_from_psm(handeye, calibration)
    intrinsics = np.asarray(calibration["K_left_rect"], dtype=np.float64)
    video_timestamps = np.asarray(camera_metadata["timestamps"], dtype=np.float64)
    joint_timestamps = np.asarray(joints["states_timestamps"], dtype=np.float64)
    joint_positions = np.asarray(
        [state["q"] for state in joints["states"]], dtype=np.float64
    )

    if args.frames is None:
        frames = np.rint(
            np.linspace(0, len(video_timestamps) - 1, args.num_frames)
        ).astype(np.int64)
    else:
        frames = np.asarray(args.frames, dtype=np.int64)
    frames = np.unique(frames)
    if np.any(frames < 0) or np.any(frames >= len(video_timestamps)):
        raise IndexError(
            f"Frame indices must be in [0, {len(video_timestamps) - 1}]"
        )

    with np.load(args.gaussians, allow_pickle=False) as asset:
        local_means = asset["means"].astype(np.float64)
        gaussian_link_ids = asset["link_ids"].astype(np.int64)
        asset_link_names = asset["link_names"].tolist()
    if args.tip_only:
        main_link_id = asset_link_names.index("PSM1_tool_main_link")
        keep = gaussian_link_ids != main_link_id
        local_means = local_means[keep]
        gaussian_link_ids = gaussian_link_ids[keep]
    with np.load(args.pose_driver, allow_pickle=False) as driver:
        driver_link_names = driver["link_names"].tolist()
        driver_poses = driver["poses_rect_camera_xyz_xyzw"].astype(np.float64)
    driver_link_index = {name: index for index, name in enumerate(driver_link_names)}
    kinematics = PSMLNDKinematics.from_files(
        PSM_LND_MODEL_PATH,
        PSM_LND_POSE_REPORT_PATH,
        TABLE_FRAME_PATH,
        driver_link_names,
    )
    manual_offsets = np.zeros(7, dtype=np.float64)
    manual_offsets[3] = np.deg2rad(args.roll_offset_deg)
    manual_offsets[6] = np.deg2rad(args.jaw_offset_deg)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lnd_images: list[np.ndarray] = []
    surface_images: list[np.ndarray] = []
    lnd_zoom_images: list[np.ndarray] = []
    surface_zoom_images: list[np.ndarray] = []
    summary = []
    for video_frame in frames.tolist():
        image_path = args.rgb_dir / f"{video_frame:06d}-left.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        timestamp = float(video_timestamps[video_frame])
        state_index = int(np.argmin(np.abs(joint_timestamps - timestamp)))
        signed_time_error_ms = float(
            (joint_timestamps[state_index] - timestamp) * 1000.0
        )
        q7 = np.asarray(joints["states"][state_index]["q"], dtype=np.float64)
        link_transforms = lnd_forward_kinematics(lnd, q7)
        feature_pixels = projected_feature_pixels(
            lnd,
            link_transforms,
            transform_rect_psm,
            intrinsics,
        )
        interpolated_q7 = interpolate_joint_state(
            timestamp, joint_timestamps, joint_positions
        )
        interpolated_feature_pixels = projected_feature_pixels(
            lnd,
            lnd_forward_kinematics(lnd, interpolated_q7),
            transform_rect_psm,
            intrinsics,
        )
        interpolation_pixel_shift = np.linalg.norm(
            feature_pixels - interpolated_feature_pixels, axis=1
        )

        lnd_overlay, visibility = draw_lnd_primitives(
            image,
            lnd,
            link_transforms,
            transform_rect_psm,
            intrinsics,
        )
        lnd_overlay = annotate(
            lnd_overlay,
            "RAW LND: skeleton + point features",
            video_frame,
            state_index,
            signed_time_error_ms,
        )

        selected_driver_poses = apply_psm_pose_driver_joint_offsets(
            driver_poses[state_index],
            q7,
            kinematics,
            manual_offsets,
        )
        surface_points = lnd_driven_surface_points(
            local_means,
            gaussian_link_ids,
            asset_link_names,
            driver_link_index,
            selected_driver_poses,
        )
        surface_pixels = project_points(
            surface_points, intrinsics, image.shape[1], image.shape[0]
        )
        surface_overlay = draw_surface_projection(image, surface_pixels)
        surface_overlay, _ = draw_lnd_primitives(
            surface_overlay,
            lnd,
            link_transforms,
            transform_rect_psm,
            intrinsics,
        )
        surface_overlay = annotate(
            surface_overlay,
            args.surface_label,
            video_frame,
            state_index,
            signed_time_error_ms,
        )

        lnd_path = args.out_dir / f"frame{video_frame:06d}_lnd_only.png"
        surface_path = args.out_dir / f"frame{video_frame:06d}_lnd_dvrk.png"
        cv2.imwrite(str(lnd_path), lnd_overlay)
        cv2.imwrite(str(surface_path), surface_overlay)
        lnd_images.append(lnd_overlay)
        surface_images.append(surface_overlay)
        lnd_zoom_images.append(crop_around_instrument(lnd_overlay, feature_pixels))
        surface_zoom_images.append(
            crop_around_instrument(surface_overlay, feature_pixels)
        )
        summary.append(
            {
                "video_frame": video_frame,
                "image": str(image_path.relative_to(REPO)),
                "video_timestamp": timestamp,
                "joint_state_index": state_index,
                "joint_timestamp": float(joint_timestamps[state_index]),
                "joint_minus_image_time_ms": signed_time_error_ms,
                "nearest_vs_interpolated_feature_shift_px_p50": float(
                    np.median(interpolation_pixel_shift)
                ),
                "nearest_vs_interpolated_feature_shift_px_max": float(
                    np.max(interpolation_pixel_shift)
                ),
                **visibility,
                "visible_surface_gaussians": int(len(surface_pixels)),
                "lnd_only_overlay": str(lnd_path.relative_to(REPO)),
                "lnd_dvrk_overlay": str(surface_path.relative_to(REPO)),
            }
        )
        print(lnd_path)
        print(surface_path)

    write_contact_sheet(lnd_images, args.out_dir / "contact_sheet_lnd_only.png")
    write_contact_sheet(surface_images, args.out_dir / "contact_sheet_lnd_dvrk.png")
    write_contact_sheet(
        lnd_zoom_images,
        args.out_dir / "contact_sheet_lnd_only_zoom.png",
        tile_width=600,
        tile_height=400,
    )
    write_contact_sheet(
        surface_zoom_images,
        args.out_dir / "contact_sheet_lnd_dvrk_zoom.png",
        tile_width=600,
        tile_height=400,
    )
    report = {
        "purpose": (
            "Calibration inspection only. The selected pose driver and explicit "
            "GUI-equivalent manual roll/jaw offsets are projected; no frame-0 "
            "fitting or visual force is applied."
        ),
        "pose_driver": str(args.pose_driver),
        "surface_label": args.surface_label,
        "tip_only": args.tip_only,
        "manual_roll_offset_deg": args.roll_offset_deg,
        "manual_jaw_offset_deg": args.jaw_offset_deg,
        "timestamp_matching": "nearest original joint sample to each PNG timestamp",
        "frames": summary,
    }
    (args.out_dir / "overlay_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
