#!/usr/bin/env python3
"""Render the final lifted GUI driver back into all ten stereo keyframes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import optimize_super_p420006_stereo_keyframes as calibration


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = (
    REPO_ROOT / "data/super/psm_raw_kinematics_v1（纯机器人学版本）"
)
P420_ROOT = RAW_ROOT / "gui_p420006_v1"
VISUAL_ROOT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_p420006_stereo_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project the final 5458-state GUI driver into both rectified "
            "cameras and compare it with the manual annotations."
        )
    )
    parser.add_argument("--root", type=Path, default=VISUAL_ROOT)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--p420-root", type=Path, default=P420_ROOT)
    parser.add_argument("--render-scale", type=float, default=0.25)
    return parser.parse_args()


def poses_to_matrices(poses: np.ndarray) -> np.ndarray:
    matrices = np.broadcast_to(
        np.eye(4, dtype=np.float64),
        (*poses.shape[:-1], 4, 4),
    ).copy()
    matrices[..., :3, 3] = poses[..., :3]
    matrices[..., :3, :3] = Rotation.from_quat(
        poses[..., 3:].reshape(-1, 4)
    ).as_matrix().reshape(*poses.shape[:-1], 3, 3)
    return matrices


@torch.no_grad()
def pair_visual_loss(
    *,
    renderer: calibration.P420StereoRenderer,
    links: dict[str, torch.Tensor],
    targets: dict[str, dict[str, torch.Tensor]],
    distances: dict[str, dict[str, torch.Tensor]],
    target_tips: dict[str, dict[str, torch.Tensor | None]],
    confidence: dict[str, float],
) -> tuple[float, dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    predicted_masks: dict[str, dict[str, torch.Tensor]] = {}
    predicted_tips: dict[str, torch.Tensor] = {}
    side_losses: dict[str, dict[str, torch.Tensor]] = {}
    for side in ("left", "right"):
        predicted_masks[side] = {
            name: renderer.render_group(links[side], side, name)
            for name in ("body", "jaw_a", "jaw_b")
        }
        predicted_tips[side] = renderer.project_tips(links[side], side)
        side_losses[side] = {
            "body": calibration.mask_loss(
                predicted_masks[side]["body"],
                targets[side]["body"],
                distances[side]["body"],
            ),
            "a_to_1": calibration.mask_loss(
                predicted_masks[side]["jaw_a"],
                targets[side]["jaw_1"],
                distances[side]["jaw_1"],
            ),
            "a_to_2": calibration.mask_loss(
                predicted_masks[side]["jaw_a"],
                targets[side]["jaw_2"],
                distances[side]["jaw_2"],
            ),
            "b_to_1": calibration.mask_loss(
                predicted_masks[side]["jaw_b"],
                targets[side]["jaw_1"],
                distances[side]["jaw_1"],
            ),
            "b_to_2": calibration.mask_loss(
                predicted_masks[side]["jaw_b"],
                targets[side]["jaw_2"],
                distances[side]["jaw_2"],
            ),
        }
    body = sum(side_losses[side]["body"] for side in ("left", "right"))
    direct_masks = sum(
        side_losses[side]["a_to_1"] + side_losses[side]["b_to_2"]
        for side in ("left", "right")
    )
    swapped_masks = sum(
        side_losses[side]["a_to_2"] + side_losses[side]["b_to_1"]
        for side in ("left", "right")
    )
    direct_tips = torch.zeros_like(body)
    swapped_tips = torch.zeros_like(body)
    for side in ("left", "right"):
        for predicted_index, target_name in ((0, "jaw_1"), (1, "jaw_2")):
            tip = target_tips[side][target_name]
            if tip is not None:
                direct_tips += torch.linalg.vector_norm(
                    predicted_tips[side][:, predicted_index] - tip,
                    dim=1,
                ) / 20.0
        for predicted_index, target_name in ((0, "jaw_2"), (1, "jaw_1")):
            tip = target_tips[side][target_name]
            if tip is not None:
                swapped_tips += torch.linalg.vector_norm(
                    predicted_tips[side][:, predicted_index] - tip,
                    dim=1,
                ) / 20.0
    direct = (
        0.75 * confidence["jaw_masks"] * direct_masks
        + 0.30 * confidence["jaw_tips"] * direct_tips
    )
    swapped = (
        0.75 * confidence["jaw_masks"] * swapped_masks
        + 0.30 * confidence["jaw_tips"] * swapped_tips
    )
    loss = confidence["body"] * body + torch.minimum(direct, swapped)
    masks_np = {
        side: {
            name: mask[0].cpu().numpy()
            for name, mask in predicted_masks[side].items()
        }
        for side in ("left", "right")
    }
    tips_np = {
        side: predicted_tips[side][0].cpu().numpy()
        for side in ("left", "right")
    }
    return float(loss.item()), masks_np, tips_np


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Final stereo projection validation requires CUDA")
    manifest_path = args.root / "pair_manifest.json"
    keyframes_path = args.root / "keyframes.npz"
    raw_model_path = args.raw_root / "model.json"
    source_driver_path = (
        args.p420_root / "psm_p420006_gui_pose_driver.npz"
    )
    final_driver_path = (
        args.root
        / "gui_v1/psm_p420006_stereo_visual_gui_pose_driver.npz"
    )
    mesh_dir = args.p420_root / "meshes"
    for path in (
        manifest_path,
        keyframes_path,
        raw_model_path,
        source_driver_path,
        final_driver_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["keyframes"]
    raw_model = json.loads(raw_model_path.read_text(encoding="utf-8"))
    with np.load(keyframes_path, allow_pickle=False) as keyframes:
        K_left = keyframes["K_left_rect"].astype(np.float32)
        K_right = keyframes["K_right_rect"].astype(np.float32)
        T_right_left = keyframes[
            "T_rectified_right_camera_rectified_left_camera"
        ].astype(np.float32)
    with np.load(source_driver_path, allow_pickle=False) as source:
        link_offsets = source["T_lndlink_urdf_link"].astype(np.float32)
        lnd_link_ids = source["lnd_link_ids"].astype(np.int64)
        X_gui_left = source[
            "X_gui_world_rectified_left_camera"
        ].astype(np.float64)
        raw_left = (
            np.linalg.inv(X_gui_left)[None, None]
            @ poses_to_matrices(
                source["poses_gui_world_xyz_xyzw"].astype(np.float64)
            )
        )
    with np.load(final_driver_path, allow_pickle=False) as final:
        if not np.array_equal(
            final["X_gui_world_rectified_left_camera"],
            X_gui_left,
        ):
            raise RuntimeError("Final and raw P420006 GUI coordinates differ")
        final_left = (
            np.linalg.inv(X_gui_left)[None, None]
            @ poses_to_matrices(
                final["poses_gui_world_xyz_xyzw"].astype(np.float64)
            )
        )

    device = torch.device("cuda")
    renderer = calibration.P420StereoRenderer(
        mesh_dir=mesh_dir,
        link_offsets=link_offsets,
        lnd_link_ids=lnd_link_ids,
        dh_parameters=raw_model["lnd"]["DH_params"],
        K_left=K_left,
        K_right=K_right,
        T_right_left=T_right_left,
        image_size=tuple(manifest["image_size_wh"]),
        render_scale=args.render_scale,
        bounds=np.ones(10, dtype=np.float32),
        device=device,
    )
    T_right_left_torch = torch.as_tensor(
        T_right_left,
        device=device,
        dtype=torch.float32,
    )

    raw_losses: list[float] = []
    final_losses: list[float] = []
    panels: list[np.ndarray] = []
    panel_width = 800
    for row in rows:
        keyframe = int(row["keyframe_index"])
        targets, distances, target_tips = calibration.load_targets(
            args.root,
            keyframe,
            (renderer.height, renderer.width),
            args.render_scale,
            device,
        )
        confidence = calibration.load_annotation_confidence(
            args.root, keyframe
        )
        raw_links: dict[str, torch.Tensor] = {}
        final_links: dict[str, torch.Tensor] = {}
        for side in ("left", "right"):
            q_index = int(row[f"{side}_q_index"])
            raw_side = torch.as_tensor(
                raw_left[q_index : q_index + 1],
                device=device,
                dtype=torch.float32,
            )
            final_side = torch.as_tensor(
                final_left[q_index : q_index + 1],
                device=device,
                dtype=torch.float32,
            )
            if side == "right":
                raw_side = T_right_left_torch[None, None] @ raw_side
                final_side = T_right_left_torch[None, None] @ final_side
            raw_links[side] = raw_side
            final_links[side] = final_side
        raw_loss, _raw_masks, _raw_tips = pair_visual_loss(
            renderer=renderer,
            links=raw_links,
            targets=targets,
            distances=distances,
            target_tips=target_tips,
            confidence=confidence,
        )
        final_loss, final_masks, final_tips = pair_visual_loss(
            renderer=renderer,
            links=final_links,
            targets=targets,
            distances=distances,
            target_tips=target_tips,
            confidence=confidence,
        )
        raw_losses.append(raw_loss)
        final_losses.append(final_loss)

        side_panels: list[np.ndarray] = []
        for side in ("left", "right"):
            image = cv2.imread(str(args.root / row[f"{side}_image"]))
            overlay = calibration.overlay_masks(
                image,
                final_masks[side],
                final_tips[side],
            )
            scale = panel_width / (2.0 * overlay.shape[1])
            side_panels.append(
                cv2.resize(
                    overlay,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
            )
        panel = np.concatenate(side_panels, axis=1)
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(
            panel,
            f"KF {keyframe:02d}  final-GUI-lifted",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)

    raw_loss_array = np.asarray(raw_losses, dtype=np.float64)
    final_loss_array = np.asarray(final_losses, dtype=np.float64)
    preview = args.root / "previews/stereo_visual_gui_lifted_overlay.png"
    if not cv2.imwrite(str(preview), np.concatenate(panels, axis=0)):
        raise RuntimeError(f"Failed to write {preview}")
    high_confidence = np.asarray(
        [
            calibration.load_annotation_confidence(
                args.root, int(row["keyframe_index"])
            )["jaw_masks"]
            >= 0.999
            for row in rows
        ],
        dtype=bool,
    )
    gates = {
        "mean_loss_improves_raw": bool(
            final_loss_array.mean() < raw_loss_array.mean()
        ),
        "all_high_confidence_pairs_improve_raw": bool(
            np.all(
                final_loss_array[high_confidence]
                < raw_loss_array[high_confidence]
            )
        ),
        "all_pairs_do_not_regress_raw": bool(
            np.all(final_loss_array <= raw_loss_array)
        ),
    }
    report = {
        "schema": "super_p420006_stereo_visual_projection_validation_v1",
        "passed": all(gates.values()),
        "gates": gates,
        "raw_loss": raw_loss_array.tolist(),
        "final_gui_lifted_loss": final_loss_array.tolist(),
        "improvement": (raw_loss_array - final_loss_array).tolist(),
        "raw_mean": float(raw_loss_array.mean()),
        "final_gui_lifted_mean": float(final_loss_array.mean()),
        "preview": str(preview.resolve()),
    }
    report_path = args.root / "gui_v1/projection_validation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise RuntimeError("Final lifted GUI projection validation failed")


if __name__ == "__main__":
    main()
