#!/usr/bin/env python3
"""Render the raw paper-LND first-pair projection in OpenCV coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import build_super_raw_psm_gui_driver as raw_builder
from annotate_super_p420006_stereo_keyframes import rasterize_view
from paper_lnd_mesh_io import load_binary_ply_triangles
from super_psm_tracking_common import _paper_component_transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = (
    REPO_ROOT / "data/super/psm_raw_kinematics_v1（纯机器人学版本）"
)
VISUAL_ROOT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_paper_lnd_stereo_v1"
)
PAPER_MESH_DIR = Path(
    "/Media_HDD/jwshan/wad/online_dvrk_tracking/urdfs/dVRK/meshes"
)
MESH_NAMES = (
    "low_res_shaft_multi_cylinder.ply",
    "low_res_logo_low_res_1.ply",
    "low_res_jawright_lowres.ply",
    "low_res_jawleft_lowres.ply",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-root", type=Path, default=VISUAL_ROOT)
    parser.add_argument(
        "--kinematics",
        type=Path,
        default=RAW_ROOT / "kinematics.npz",
    )
    parser.add_argument(
        "--raw-model",
        type=Path,
        default=RAW_ROOT / "model.json",
    )
    parser.add_argument("--mesh-dir", type=Path, default=PAPER_MESH_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=VISUAL_ROOT / "previews/raw_paper_lnd_first_projection.png",
    )
    return parser.parse_args()


def project(
    vertices: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positive = vertices[:, 2] > 1.0e-5
    pixels = np.empty((len(vertices), 2), dtype=np.float64)
    pixels[:, 0] = (
        K[0, 0] * vertices[:, 0] / vertices[:, 2] + K[0, 2]
    )
    pixels[:, 1] = (
        K[1, 1] * vertices[:, 1] / vertices[:, 2] + K[1, 2]
    )
    return pixels, positive


def render_silhouette(
    transforms: tuple[np.ndarray, ...],
    meshes: list[tuple[np.ndarray, np.ndarray]],
    K: np.ndarray,
    size_wh: tuple[int, int],
) -> np.ndarray:
    width, height = size_wh
    mask = np.zeros((height, width), dtype=np.uint8)
    for transform, (local_vertices, faces) in zip(
        transforms,
        meshes,
        strict=True,
    ):
        camera_vertices = (
            local_vertices @ transform[:3, :3].T + transform[:3, 3]
        )
        pixels, positive = project(camera_vertices, K)
        for face in faces:
            if not np.all(positive[face]):
                continue
            polygon = np.rint(pixels[face]).astype(np.int32)
            cv2.fillConvexPoly(mask, polygon, 255)
    return mask


def projected_tips(
    transforms: tuple[np.ndarray, ...],
    K: np.ndarray,
) -> np.ndarray:
    local = np.asarray(
        [
            [0.0, 0.0004, 0.0096],
            [0.0, -0.0004, 0.0096],
        ],
        dtype=np.float64,
    )
    points = np.stack(
        [
            transforms[2 + index][:3, :3] @ local[index]
            + transforms[2 + index][:3, 3]
            for index in range(2)
        ]
    )
    pixels, positive = project(points, K)
    pixels[~positive] = np.nan
    return pixels


def main() -> None:
    args = parse_args()
    manifest = json.loads(
        (args.visual_root / "pair_manifest.json").read_text(encoding="utf-8")
    )
    annotation = json.loads(
        (args.visual_root / "annotations/keyframe_00.json").read_text(
            encoding="utf-8"
        )
    )
    raw_model = json.loads(args.raw_model.read_text(encoding="utf-8"))
    row = manifest["keyframes"][0]
    with np.load(args.kinematics, allow_pickle=False) as raw:
        q7 = raw["q7"].astype(np.float64)
        T_left_lnd = raw[
            "T_rectified_left_camera_lnd_link"
        ].astype(np.float64)
    with np.load(
        args.visual_root / "keyframes.npz",
        allow_pickle=False,
    ) as keyframes:
        K = {
            "left": keyframes["K_left_rect"].astype(np.float64),
            "right": keyframes["K_right_rect"].astype(np.float64),
        }
        T_right_left = keyframes[
            "T_rectified_right_camera_rectified_left_camera"
        ].astype(np.float64)
    meshes = [
        load_binary_ply_triangles(args.mesh_dir / name)
        for name in MESH_NAMES
    ]
    width, height = manifest["image_size_wh"]
    panels: list[np.ndarray] = []
    metrics: dict[str, dict] = {}
    for side in ("left", "right"):
        q_index = int(row[f"{side}_q_index"])
        joints = q7[q_index]
        fk = raw_builder.lnd_fk(raw_model["lnd"], joints)
        T_left_frame4 = T_left_lnd[q_index, 0] @ fk[4]
        components = _paper_component_transforms(
            np.asarray(
                [
                    joints[4],
                    joints[5],
                    0.5 * joints[6],
                    0.5 * joints[6],
                ]
            )
        )
        camera_from_left = (
            np.eye(4) if side == "left" else T_right_left
        )
        visible = tuple(
            camera_from_left @ T_left_frame4 @ components[index]
            for index in (0, 1, 3, 4)
        )
        predicted = render_silhouette(
            visible,
            meshes,
            K[side],
            (width, height),
        )
        tips = projected_tips(visible, K[side])
        manual_labels = rasterize_view(
            annotation["views"][side],
            width,
            height,
        )
        manual = manual_labels > 0
        prediction = predicted > 0
        union = np.count_nonzero(manual | prediction)
        intersection = np.count_nonzero(manual & prediction)
        metrics[side] = {
            "raw_silhouette_iou": (
                float(intersection / union) if union else 0.0
            ),
            "raw_projected_tip_xy": tips.tolist(),
            "manual_tip_xy": [
                annotation["views"][side]["tips"][name]["point_xy"]
                for name in ("jaw_1", "jaw_2")
            ],
        }
        image = cv2.imread(
            str(args.visual_root / row[f"{side}_image"]),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise RuntimeError(f"Failed to load {side} first frame")
        manual_contours, _ = cv2.findContours(
            manual.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        prediction_contours, _ = cv2.findContours(
            prediction.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(image, manual_contours, -1, (40, 230, 40), 3)
        cv2.drawContours(image, prediction_contours, -1, (40, 40, 255), 3)
        for index, point in enumerate(tips):
            if np.all(np.isfinite(point)):
                xy = tuple(np.rint(point).astype(int))
                cv2.drawMarker(
                    image,
                    xy,
                    (0, 255, 255),
                    cv2.MARKER_CROSS,
                    24,
                    3,
                )
                cv2.putText(
                    image,
                    f"P{index + 1}",
                    (xy[0] + 8, xy[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
        cv2.putText(
            image,
            f"{side.upper()} raw paper LND IoU={metrics[side]['raw_silhouette_iou']:.3f}",
            (25, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
        )
        cv2.putText(
            image,
            "green=manual  red=raw CAD  yellow=raw CAD tips",
            (25, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
        )
        panels.append(image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), np.concatenate(panels, axis=1)):
        raise RuntimeError(f"Failed to write {args.output}")
    report_path = args.output.with_suffix(".json")
    report_path.write_text(
        json.dumps(
            {
                "schema": "super_paper_lnd_raw_first_projection_v1",
                "coordinate_frame": (
                    "rectified OpenCV camera pixels, top-left origin"
                ),
                "metrics": metrics,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
