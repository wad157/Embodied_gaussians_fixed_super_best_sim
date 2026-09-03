#!/usr/bin/env python3

"""Run the online_dvrk_tracking video calibration path without robust additions.

The estimator in this file intentionally keeps the paper repository's choices:
SurgicalSAM2 propagation from frame-0 prompts, BO initialization, MixAngle,
per-frame CMA-ES, symmetric jaws, ContourTipNet, previous joint initialization,
and its Kalman filtering.  Paper-to-current-PSM conversion happens only after
tracking and does not feed any robot/base information back into the estimator.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

from super_psm_tracking_common import (  # noqa: E402
    DEFAULT_TRACK_ROOT,
    TrackingInputs,
    save_runtime_driver,
)
from track_super_psm_paper_and_hybrid import (  # noqa: E402
    configure_paper_imports,
    ctrnet_args,
    initialize_paper_state,
    paper_stdev,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact paper video calibration on SUPER grasp5."
    )
    parser.add_argument(
        "--paper-repo",
        type=Path,
        default=Path("/Media_HDD/jwshan/wad/online_dvrk_tracking"),
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=REPO
        / "data/super/grasp5_offline_demo/videos/stereo_left.mp4",
    )
    parser.add_argument("--track-root", type=Path, default=DEFAULT_TRACK_ROOT)
    parser.add_argument("--sample-number", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--batch-iters", type=int, default=100)
    parser.add_argument("--final-iters", type=int, default=100)
    parser.add_argument("--online-iters", type=int, default=3)
    parser.add_argument("--popsize", type=int, default=70)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-init-cache", action="store_true")
    return parser.parse_args()


def exact_tracker_args(args: argparse.Namespace, paper_repo: Path) -> Namespace:
    return Namespace(
        symmetric_jaw=True,
        searcher="CMA-ES",
        use_filter=True,
        filter_option="Kalman",
        use_mix_angle=True,
        cos_reparams=True,
        use_prev_joint_angles=True,
        use_contour_tip_net=True,
        contour_tip_net_path=str(
            paper_repo / "ContourTipNet/models/cnn_model.pth"
        ),
        downscale_factor=2,
        popsize=args.popsize,
        final_iters=args.final_iters,
        use_render_loss=True,
        use_pts_loss=True,
        mse_weight=6.0,
        dist_weight=0.0,
        app_weight=6e-6,
        pts_weight=3e-3,
    )


def save_checkpoint(
    path: Path,
    timestamps: np.ndarray,
    ctr: list[np.ndarray],
    joints: list[np.ndarray],
    losses: list[float],
    masks_packbits: list[np.ndarray],
    mask_shape: tuple[int, int],
) -> None:
    count = len(ctr)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        video_timestamps=timestamps[:count],
        pure_ctr=np.asarray(ctr, dtype=np.float32),
        pure_joints=np.asarray(joints, dtype=np.float32),
        pure_losses=np.asarray(losses, dtype=np.float32),
        masks_packbits=np.stack(masks_packbits),
        mask_shape=np.asarray(mask_shape, dtype=np.int32),
    )


def main() -> None:
    args = parse_args()
    paper_repo = args.paper_repo.resolve()
    configure_paper_imports(paper_repo)

    from diffcali.eval_dvrk.trackers import Tracker
    from diffcali.models.CtRNet import CtRNet
    from sam2.build_sam import build_sam2_camera_predictor

    inputs = TrackingInputs.load()
    annotation_dir = args.track_root / "online_videos/grasp5"
    prompts_path = annotation_dir / "PSM1_prompts.txt"
    keypoints_path = annotation_dir / "PSM1_keypoints.txt"
    ref_mask_path = annotation_dir / "PSM1_ref_mask.png"
    for path in (args.video, prompts_path, keypoints_path, ref_mask_path):
        if not path.exists():
            raise FileNotFoundError(path)
    prompts = np.loadtxt(prompts_path, ndmin=2)
    clicked_keypoints = np.loadtxt(keypoints_path, ndmin=2)
    if prompts.shape[1] != 3 or not ({0, 1} <= set(prompts[:, 2].astype(int))):
        raise ValueError("Prompts must contain foreground and background labels")
    if clicked_keypoints.shape != (2, 2):
        raise ValueError(
            f"Exactly two jaw-tip clicks are required, got {clicked_keypoints.shape}"
        )
    # The user's click order is [paper jaw-left, paper jaw-right], while the
    # paper project_keypoints() contract is [jaw-right, jaw-left].  This is an
    # input-format correction, not an estimator modification.
    paper_keypoints = clicked_keypoints[[1, 0]].astype(np.float32)

    intrinsics = ctrnet_args(inputs)
    model = CtRNet(intrinsics)
    mesh_dir = paper_repo / "urdfs/dVRK/meshes"
    mesh_files = [
        mesh_dir / "low_res_shaft_multi_cylinder.ply",
        mesh_dir / "low_res_logo_low_res_1.ply",
        mesh_dir / "low_res_jawright_lowres.ply",
        mesh_dir / "low_res_jawleft_lowres.ply",
    ]
    robot_renderer = model.setup_robot_renderer(
        [str(path) for path in mesh_files], downscale_factor=2
    )
    robot_renderer.set_mesh_visibility([True, True, True, True])
    intr = torch.as_tensor(inputs.K_half, device="cuda", dtype=torch.float32)
    p_local1 = torch.tensor(
        [0.0, 0.0004, 0.0096], device="cuda", dtype=torch.float32
    )
    p_local2 = torch.tensor(
        [0.0, -0.0004, 0.0096], device="cuda", dtype=torch.float32
    )

    predictor = build_sam2_camera_predictor(
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        str(
            paper_repo
            / "SurgicalSAM2/checkpoints/sam2.1_hiera_s_endo18.pth"
        ),
        vos_optimized=True,
    )
    init_ctr, init_joints, init_loss = initialize_paper_state(
        args=args,
        model=model,
        robot_renderer=robot_renderer,
        ref_mask_path=ref_mask_path,
        keypoints=paper_keypoints,
        intrinsics=intrinsics,
        cache_path=args.track_root / "paper_init_cache_tip_ordered.pth",
    )
    tracker = Tracker(
        model=model,
        robot_renderer=robot_renderer,
        init_cTr=init_ctr.clone(),
        init_joint_angles=init_joints.clone(),
        num_iters=args.online_iters,
        stdev_init=paper_stdev(hybrid=False),
        intr=intr,
        p_local1=p_local1,
        p_local2=p_local2,
        searcher="CMA-ES",
        args=exact_tracker_args(args, paper_repo),
    )

    cap = cv2.VideoCapture(str(args.video))
    video_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = min(video_count, len(inputs.video_timestamps))
    if args.max_frames is not None:
        frame_count = min(frame_count, args.max_frames)
    if frame_count < 2:
        raise ValueError("At least two video frames are required")

    ctr_sequence: list[np.ndarray] = []
    joint_sequence: list[np.ndarray] = []
    losses: list[float] = []
    masks_packbits: list[np.ndarray] = []
    partial_path = args.track_root / "tracking_states_paper_exact_partial.npz"
    started = time.perf_counter()

    for frame_index in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read video frame {frame_index}")
        frame = cv2.resize(frame, (intrinsics.width, intrinsics.height))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if frame_index == 0:
                predictor.load_first_frame(frame)
                _, _, mask_logits = predictor.add_new_points(
                    frame_idx=0,
                    obj_id=0,
                    points=prompts[:, :2].astype(np.float32),
                    labels=prompts[:, 2].astype(np.int64),
                )
            else:
                _, mask_logits = predictor.track(frame)
        mask = (mask_logits.squeeze() > 0).float()
        mask_numpy = mask.detach().cpu().numpy().astype(bool)
        masks_packbits.append(np.packbits(mask_numpy.reshape(-1)))
        frame_keypoints = (
            torch.as_tensor(
                paper_keypoints, device="cuda", dtype=torch.float32
            )
            if frame_index == 0
            else None
        )
        state_ctr, state_joints, loss = tracker.track_frame(
            ref_mask=mask,
            joint_angles=None,
            is_init=frame_index == 0,
            keypoints=frame_keypoints,
        )
        if not (
            torch.isfinite(state_ctr).all()
            and torch.isfinite(state_joints).all()
            and torch.isfinite(torch.as_tensor(loss)).all()
        ):
            raise FloatingPointError(
                f"Paper tracker returned a non-finite result at frame {frame_index}"
            )
        ctr_sequence.append(state_ctr.detach().cpu().numpy().copy())
        joint_sequence.append(state_joints.detach().cpu().numpy().copy())
        losses.append(float(torch.as_tensor(loss).item()))
        print(
            f"paper_exact frame={frame_index:04d}/{frame_count - 1:04d} "
            f"loss={losses[-1]:.6g} elapsed={time.perf_counter() - started:.1f}s"
        )
        if (
            (frame_index + 1) % args.checkpoint_every == 0
            or frame_index + 1 == frame_count
        ):
            save_checkpoint(
                partial_path,
                inputs.video_timestamps,
                ctr_sequence,
                joint_sequence,
                losses,
                masks_packbits,
                mask_numpy.shape,
            )
    cap.release()

    full_run = frame_count == video_count == len(inputs.video_timestamps)
    states_path = args.track_root / (
        "tracking_states_paper_exact.npz"
        if full_run
        else f"tracking_states_paper_exact_preview_{frame_count}.npz"
    )
    partial_path.replace(states_path)
    video_poses = inputs.paper_states_to_visual_poses(
        np.asarray(ctr_sequence), np.asarray(joint_sequence)
    )
    poses_path = args.track_root / (
        "visual_poses_video_paper_exact.npz"
        if full_run
        else f"visual_poses_video_paper_exact_preview_{frame_count}.npz"
    )
    np.savez_compressed(
        poses_path,
        video_timestamps=inputs.video_timestamps[:frame_count],
        link_names=np.asarray(inputs.link_names),
        pure_poses_rect_camera_xyz_xyzw=video_poses,
    )
    if full_run:
        save_runtime_driver(
            args.track_root / "psm_paper_exact_pose_driver.npz",
            inputs,
            video_poses,
        )
    report = {
        "paper_repo": str(paper_repo),
        "paper_commit": "cb2a264167aaf05b5a9c20da885d48568f78311f",
        "frame_count": frame_count,
        "full_run": full_run,
        "pipeline": (
            "SurgicalSAM2 + BO(1500) + MixAngle + NvDiffRast + "
            "CMA-ES(3) + symmetric jaw + ContourTipNet + paper Kalman"
        ),
        "non_paper_estimator_additions": [],
        "input_keypoint_order_correction": "clicked [left,right] -> paper [right,left]",
        "initialization_loss": init_loss,
        "mean_loss": float(np.mean(losses)),
        "elapsed_seconds": time.perf_counter() - started,
        "states": str(states_path),
        "video_poses": str(poses_path),
    }
    (args.track_root / "tracking_report_paper_exact.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
