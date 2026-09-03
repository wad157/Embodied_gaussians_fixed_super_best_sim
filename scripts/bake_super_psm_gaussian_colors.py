#!/usr/bin/env python3

"""Bake first-stereo-pair RGB onto link-local PSM surface Gaussians."""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
VISUAL_ROOT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_paper_lnd_stereo_dense_contact_v4"
)
MASK_ROOT = VISUAL_ROOT / "surgicalsam2_multianchor_parts_dense_contact_v6"
GUI_ROOT = (
    MASK_ROOT
    / "online_stereo_cma_se3_unbounded_dense_contact_v4/gui_unbounded_xyz_v1"
)
DEFAULT_ASSET = GUI_ROOT / "psm_paper_lnd_surface_gaussians_dense_v2.npz"
DEFAULT_OUTPUT = (
    GUI_ROOT / "psm_paper_lnd_surface_gaussians_dense_v2_first_stereo_rgb.npz"
)
DEFAULT_DRIVER = GUI_ROOT / "psm_paper_lnd_gui_pose_driver.npz"
DEFAULT_CALIB = REPO_ROOT / "data/super/grasp5_native/calib_rectified.json"
DEFAULT_MASKS = MASK_ROOT / "stereo_multianchor_part_masks.npz"
DEFAULT_PROMPT_MANIFEST = VISUAL_ROOT / "prompt_manifest.json"
DEFAULT_REPORT = (
    GUI_ROOT
    / "psm_paper_lnd_surface_gaussians_dense_v2_first_stereo_rgb_report.json"
)
DEFAULT_PREVIEW = (
    GUI_ROOT
    / "psm_paper_lnd_surface_gaussians_dense_v2_first_stereo_rgb_preview.png"
)
LONG_SHAFT_LINK = "PSM1_tool_main_link"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bake real RGB from one confirmed stereo anchor onto dense PSM "
            "surface Gaussians with masks, front-face filtering, and a z-buffer."
        )
    )
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--driver", type=Path, default=DEFAULT_DRIVER)
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--masks", type=Path, default=DEFAULT_MASKS)
    parser.add_argument(
        "--prompt-manifest", type=Path, default=DEFAULT_PROMPT_MANIFEST
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument(
        "--anchor-index",
        type=int,
        default=0,
        help="Index in prompt_manifest.json. The default is the first stereo pair.",
    )
    parser.add_argument(
        "--mask-erosion-px",
        type=int,
        default=2,
        help="Erode each half-resolution part mask to reject boundary contamination.",
    )
    parser.add_argument(
        "--depth-tolerance-m",
        type=float,
        default=0.0006,
        help="Maximum distance behind the nearest point at the same pixel.",
    )
    parser.add_argument(
        "--black-unobserved-link",
        action="append",
        default=[LONG_SHAFT_LINK],
        help=(
            "Set Gaussians unseen by both eyes to black for this link. May be "
            "repeated; defaults to the long shaft."
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_index(sorted_values: np.ndarray, query: int) -> int:
    right = int(np.searchsorted(sorted_values, query, side="left"))
    right = min(max(right, 0), len(sorted_values) - 1)
    left = max(right - 1, 0)
    if abs(int(query) - int(sorted_values[left])) <= abs(
        int(sorted_values[right]) - int(query)
    ):
        return left
    return right


def pose_matrix(pose_xyz_xyzw: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(pose_xyz_xyzw[3:]).as_matrix()
    transform[:3, 3] = pose_xyz_xyzw[:3]
    return transform


def unpack_mask(
    packed: np.ndarray,
    shape: tuple[int, int],
    bitorder: str,
) -> np.ndarray:
    count = int(shape[0] * shape[1])
    return np.unpackbits(packed, bitorder=bitorder)[:count].reshape(shape).astype(bool)


def scaled_intrinsics(
    calibration: dict,
    side: str,
    image_size_wh: tuple[int, int],
    mask_size_wh: tuple[int, int],
) -> np.ndarray:
    key = "K_left_rect" if side == "left" else "K_right_rect"
    matrix = np.asarray(calibration[key], dtype=np.float64).copy()
    matrix[0] *= mask_size_wh[0] / image_size_wh[0]
    matrix[1] *= mask_size_wh[1] / image_size_wh[1]
    return matrix


def project(points_camera: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points_camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.rint(
            points_camera[:, 0] * intrinsics[0, 0] / z + intrinsics[0, 2]
        ).astype(np.int32)
        v = np.rint(
            points_camera[:, 1] * intrinsics[1, 1] / z + intrinsics[1, 2]
        ).astype(np.int32)
    return u, v


def transform_gaussians_to_camera(
    means: np.ndarray,
    normals: np.ndarray,
    link_ids: np.ndarray,
    asset_link_names: list[str],
    driver_link_names: list[str],
    poses_world: np.ndarray,
    state_index: int,
    X_camera_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points_camera = np.empty_like(means, dtype=np.float64)
    normals_camera = np.empty_like(normals, dtype=np.float64)
    driver_ids = {name: index for index, name in enumerate(driver_link_names)}
    missing = [name for name in asset_link_names if name not in driver_ids]
    if missing:
        raise ValueError(f"Asset links absent from pose driver: {missing}")
    for link_id, link_name in enumerate(asset_link_names):
        selected = link_ids == link_id
        X_world_link = pose_matrix(poses_world[state_index, driver_ids[link_name]])
        X_camera_link = X_camera_world @ X_world_link
        points_camera[selected] = (
            means[selected] @ X_camera_link[:3, :3].T
            + X_camera_link[:3, 3]
        )
        normals_camera[selected] = normals[selected] @ X_camera_link[:3, :3].T
    return points_camera, normals_camera


def sample_one_eye(
    *,
    side: str,
    image_path: Path,
    image_size_wh: tuple[int, int],
    shaft_mask: np.ndarray,
    distal_mask: np.ndarray,
    intrinsics: np.ndarray,
    points_camera: np.ndarray,
    normals_camera: np.ndarray,
    link_ids: np.ndarray,
    asset_link_names: list[str],
    depth_tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray]:
    image_bgr_full = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr_full is None:
        raise FileNotFoundError(image_path)
    if tuple(image_bgr_full.shape[1::-1]) != image_size_wh:
        raise ValueError(
            f"Unexpected {side} image size {image_bgr_full.shape[1::-1]} at {image_path}"
        )
    height, width = shaft_mask.shape
    image_bgr = cv2.resize(
        image_bgr_full,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    u, v = project(points_camera, intrinsics)
    in_frame = (
        np.all(np.isfinite(points_camera), axis=1)
        & (points_camera[:, 2] > 0.0)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )
    front_facing = np.einsum("ij,ij->i", normals_camera, -points_camera) > 0.0

    # Point z-buffer at the sampling resolution. The normal test rejects the
    # complementary back surface even when sparse samples land in adjacent pixels.
    in_frame_ids = np.flatnonzero(in_frame)
    depth = np.full(height * width, np.inf, dtype=np.float32)
    pixel_ids = v[in_frame_ids] * width + u[in_frame_ids]
    np.minimum.at(depth, pixel_ids, points_camera[in_frame_ids, 2])
    nearest_depth = depth[pixel_ids]
    z_visible = np.zeros(len(points_camera), dtype=bool)
    z_visible[in_frame_ids] = (
        points_camera[in_frame_ids, 2] <= nearest_depth + depth_tolerance_m
    )

    in_part_mask = np.zeros(len(points_camera), dtype=bool)
    link_rows = []
    for link_id, link_name in enumerate(asset_link_names):
        selected_ids = np.flatnonzero((link_ids == link_id) & in_frame)
        part_mask = shaft_mask if link_name == LONG_SHAFT_LINK else distal_mask
        in_part_mask[selected_ids] = part_mask[v[selected_ids], u[selected_ids]]

    visible = in_frame & front_facing & z_visible & in_part_mask
    colors = np.full((len(points_camera), 3), np.nan, dtype=np.float32)
    visible_ids = np.flatnonzero(visible)
    colors[visible_ids] = image_rgb[v[visible_ids], u[visible_ids]]

    for link_id, link_name in enumerate(asset_link_names):
        link_mask = link_ids == link_id
        link_rows.append(
            {
                "link_name": link_name,
                "gaussians": int(link_mask.sum()),
                "projected_in_frame": int((link_mask & in_frame).sum()),
                "inside_eroded_part_mask": int((link_mask & in_part_mask).sum()),
                "front_facing": int((link_mask & in_part_mask & front_facing).sum()),
                "visible_rgb_samples": int((link_mask & visible).sum()),
            }
        )
    diagnostics = {
        "side": side,
        "image": str(image_path.relative_to(REPO_ROOT)),
        "visible_rgb_samples": int(visible.sum()),
        "links": link_rows,
    }
    return colors, visible, diagnostics, image_bgr, np.column_stack([u, v])


def main() -> None:
    args = parse_args()
    for name in (
        "asset",
        "output",
        "driver",
        "calib",
        "masks",
        "prompt_manifest",
        "report",
        "preview",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.mask_erosion_px < 0:
        raise ValueError("--mask-erosion-px must be non-negative")
    if args.depth_tolerance_m < 0.0:
        raise ValueError("--depth-tolerance-m must be non-negative")

    required = (
        args.asset,
        args.driver,
        args.calib,
        args.masks,
        args.prompt_manifest,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(args.asset, allow_pickle=False) as source:
        asset = {name: source[name].copy() for name in source.files}
    with np.load(args.driver, allow_pickle=False) as source:
        driver = {name: source[name].copy() for name in source.files}
    with np.load(args.masks, allow_pickle=False) as source:
        masks = {name: source[name].copy() for name in source.files}

    means = asset["means"].astype(np.float64)
    quats_wxyz = asset["quats_wxyz"].astype(np.float64)
    link_ids = asset["link_ids"].astype(np.int64)
    asset_link_names = asset["link_names"].tolist()
    driver_link_names = driver["link_names"].tolist()
    if LONG_SHAFT_LINK not in asset_link_names:
        raise ValueError(f"Long-shaft link is absent from asset: {LONG_SHAFT_LINK}")
    if "poses_gui_world_xyz_xyzw" not in driver:
        raise ValueError("The selected driver must contain GUI-world poses")

    prompt_manifest = read_json(args.prompt_manifest)
    anchors = prompt_manifest["anchors"]
    if args.anchor_index < 0 or args.anchor_index >= len(anchors):
        raise IndexError(f"--anchor-index must lie in [0, {len(anchors) - 1}]")
    anchor = anchors[args.anchor_index]
    strict_pair_slot = int(anchor["strict_pair_slot"])
    if strict_pair_slot != int(masks["anchor_slots"][args.anchor_index]):
        raise RuntimeError("Prompt manifest and mask archive anchor orders differ")
    if not bool(masks["quality_valid"][strict_pair_slot]):
        raise RuntimeError(f"Stereo mask quality gate failed at slot {strict_pair_slot}")

    mask_shape = tuple(int(value) for value in masks["mask_shape"])
    mask_height, mask_width = mask_shape
    bitorder = str(masks["bitorder"].item())
    kernel_size = 2 * args.mask_erosion_px + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    part_masks: dict[str, dict[str, np.ndarray]] = {}
    for side in ("left", "right"):
        part_masks[side] = {}
        for part in ("shaft", "distal"):
            mask = unpack_mask(
                masks[f"{side}_{part}_masks_packbits"][strict_pair_slot],
                mask_shape,
                bitorder,
            )
            if args.mask_erosion_px:
                mask = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
            part_masks[side][part] = mask

    normals = Rotation.from_quat(quats_wxyz, scalar_first=True).as_matrix()[:, :, 2]
    X_left_world = np.linalg.inv(
        driver["X_gui_world_rectified_left_camera"].astype(np.float64)
    )
    T_right_left = driver[
        "T_rectified_right_camera_rectified_left_camera"
    ].astype(np.float64)
    X_camera_world = {
        "left": X_left_world,
        "right": T_right_left @ X_left_world,
    }
    calibration = read_json(args.calib)
    image_size_wh = tuple(int(value) for value in prompt_manifest["image_size_wh"])
    mask_size_wh = (mask_width, mask_height)
    timestamps_ros_ns = driver["timestamps_ros_ns"].astype(np.int64)
    poses_world = driver["poses_gui_world_xyz_xyzw"].astype(np.float64)

    eye_samples = []
    eye_visible = []
    eye_diagnostics = []
    preview_data = []
    for side in ("left", "right"):
        eye_timestamp = int(masks[f"{side}_timestamp_ros_ns"][strict_pair_slot])
        state_index = nearest_index(timestamps_ros_ns, eye_timestamp)
        points_camera, normals_camera = transform_gaussians_to_camera(
            means,
            normals,
            link_ids,
            asset_link_names,
            driver_link_names,
            poses_world,
            state_index,
            X_camera_world[side],
        )
        image_path = args.prompt_manifest.parent / anchor[f"{side}_image"]
        samples, visible, diagnostics, preview_image, preview_uv = sample_one_eye(
            side=side,
            image_path=image_path,
            image_size_wh=image_size_wh,
            shaft_mask=part_masks[side]["shaft"],
            distal_mask=part_masks[side]["distal"],
            intrinsics=scaled_intrinsics(
                calibration, side, image_size_wh, mask_size_wh
            ),
            points_camera=points_camera,
            normals_camera=normals_camera,
            link_ids=link_ids,
            asset_link_names=asset_link_names,
            depth_tolerance_m=args.depth_tolerance_m,
        )
        diagnostics["pose_state_index"] = state_index
        diagnostics["pose_timestamp_error_ms"] = float(
            (int(timestamps_ros_ns[state_index]) - eye_timestamp) / 1.0e6
        )
        eye_samples.append(samples)
        eye_visible.append(visible)
        eye_diagnostics.append(diagnostics)
        preview_data.append((preview_image, preview_uv, visible))

    samples_stereo = np.stack(eye_samples, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        baked = np.nanmedian(samples_stereo, axis=0).astype(np.float32)
    observation_count = np.sum(np.stack(eye_visible, axis=0), axis=0).astype(np.uint8)
    observed = observation_count > 0
    if not observed.any():
        raise RuntimeError("No first-pair RGB samples survived visibility filtering")

    colors = baked.copy()
    black_unobserved = np.zeros(len(means), dtype=bool)
    propagation_rows = []
    black_link_names = set(args.black_unobserved_link)
    unknown_black_links = sorted(black_link_names.difference(asset_link_names))
    if unknown_black_links:
        raise ValueError(f"Unknown --black-unobserved-link values: {unknown_black_links}")
    for link_id, link_name in enumerate(asset_link_names):
        link_mask = link_ids == link_id
        observed_link = link_mask & observed
        missing_link = link_mask & ~observed
        mode = "direct_first_stereo_rgb"
        if link_name in black_link_names:
            colors[missing_link] = 0.0
            black_unobserved[missing_link] = True
            mode = "unobserved_black"
        elif observed_link.any() and missing_link.any():
            tree = cKDTree(means[observed_link])
            _distance, nearest = tree.query(means[missing_link], k=1)
            colors[missing_link] = baked[observed_link][nearest]
            mode = "same_link_nearest_first_stereo_rgb"
        elif missing_link.any():
            colors[missing_link] = asset["colors"][missing_link]
            mode = "source_asset_fallback_no_observations"
        propagation_rows.append(
            {
                "link_name": link_name,
                "directly_observed": int(observed_link.sum()),
                "unobserved": int(missing_link.sum()),
                "unobserved_policy": mode,
            }
        )

    colors = np.clip(colors, 0.0, 1.0).astype(np.float32)
    asset["colors"] = colors
    asset["color_observation_count"] = observation_count
    asset["color_directly_observed"] = observed
    asset["color_unobserved_black"] = black_unobserved
    asset["color_source"] = np.asarray("first_confirmed_stereo_pair_real_rgb")
    asset["color_source_anchor_index"] = np.asarray(args.anchor_index, dtype=np.int64)
    asset["color_source_strict_pair_slot"] = np.asarray(
        strict_pair_slot, dtype=np.int64
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **asset)

    panels = []
    for (image_bgr, uv, visible), side in zip(preview_data, ("left", "right")):
        overlay = image_bgr.copy()
        for gaussian_id in np.flatnonzero(visible):
            color_bgr = tuple(
                int(value)
                for value in np.rint(colors[gaussian_id, ::-1] * 255.0)
            )
            cv2.circle(
                overlay,
                tuple(int(value) for value in uv[gaussian_id]),
                1,
                color_bgr,
                -1,
                lineType=cv2.LINE_AA,
            )
        cv2.putText(
            overlay,
            f"first stereo pair: {side}",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(overlay)
    preview = np.concatenate(panels, axis=1)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.preview), preview):
        raise RuntimeError(f"Failed to write preview: {args.preview}")

    report = {
        "schema": "super_psm_first_confirmed_stereo_rgb_bake_v1",
        "passed": True,
        "method": {
            "source": "first confirmed synchronized left/right rectified images",
            "color_rule": (
                "direct RGB; per-Gaussian median when visible in both first-frame eyes"
            ),
            "visibility": "eroded part mask + outward front face + point z-buffer",
            "unobserved_long_shaft": "black",
            "unobserved_other_links": "nearest visible point within the same rigid link",
            "uses_later_frames": False,
            "uses_urdf_material_color_for_observed_points": False,
            "mask_erosion_px_at_960x540": args.mask_erosion_px,
            "depth_tolerance_m": args.depth_tolerance_m,
        },
        "anchor_index": args.anchor_index,
        "strict_pair_slot": strict_pair_slot,
        "inputs": {
            "asset": str(args.asset.relative_to(REPO_ROOT)),
            "asset_sha256": sha256(args.asset),
            "driver": str(args.driver.relative_to(REPO_ROOT)),
            "driver_sha256": sha256(args.driver),
            "masks": str(args.masks.relative_to(REPO_ROOT)),
            "masks_sha256": sha256(args.masks),
            "prompt_manifest": str(args.prompt_manifest.relative_to(REPO_ROOT)),
        },
        "eyes": eye_diagnostics,
        "fusion": {
            "gaussians": int(len(means)),
            "directly_observed_in_either_eye": int(observed.sum()),
            "observed_in_left_eye": int(eye_visible[0].sum()),
            "observed_in_right_eye": int(eye_visible[1].sum()),
            "observed_in_both_eyes": int(
                np.logical_and(eye_visible[0], eye_visible[1]).sum()
            ),
            "unobserved_set_black": int(black_unobserved.sum()),
            "rgb_nonblack_mean": colors[~black_unobserved].mean(axis=0).tolist(),
            "rgb_nonblack_std": colors[~black_unobserved].std(axis=0).tolist(),
            "links": propagation_rows,
        },
        "output": str(args.output.relative_to(REPO_ROOT)),
        "preview": str(args.preview.relative_to(REPO_ROOT)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["fusion"], indent=2))


if __name__ == "__main__":
    main()
