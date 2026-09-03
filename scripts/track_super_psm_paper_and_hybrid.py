#!/usr/bin/env python3

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
from scipy.spatial.transform import Rotation


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from super_psm_tracking_common import (  # noqa: E402
    DEFAULT_TRACK_ROOT,
    TrackingInputs,
    ctr_to_matrix,
    matrix_to_ctr,
    paper_jaw_pivot_camera,
    save_runtime_driver,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the paper's joint-free tracker and a strict-LND-initialized "
            "hybrid tracker on the SUPER grasp5 left video."
        )
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
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Development/smoke-test limit. Runtime drivers are only emitted for a full run.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--no-init-cache", action="store_true")
    parser.add_argument("--no-contour-tip-net", action="store_true")
    parser.add_argument(
        "--hybrid-max-translation-mm", type=float, default=8.0
    )
    parser.add_argument("--hybrid-max-rotation-deg", type=float, default=10.0)
    parser.add_argument("--hybrid-max-joint-deg", type=float, default=15.0)
    parser.add_argument("--paper-max-step-mm", type=float, default=0.75)
    parser.add_argument("--paper-max-step-deg", type=float, default=2.5)
    parser.add_argument("--paper-max-total-mm", type=float, default=30.0)
    parser.add_argument("--paper-max-total-deg", type=float, default=25.0)
    parser.add_argument("--paper-max-wrist-step-deg", type=float, default=3.0)
    parser.add_argument("--paper-max-wrist-total-deg", type=float, default=25.0)
    parser.add_argument("--paper-max-jaw-step-deg", type=float, default=15.0)
    parser.add_argument(
        "--sam-reprompt-every",
        type=int,
        default=10,
        help=(
            "Re-prompt SurgicalSAM2 with two image-only LK jaw-tip tracks every "
            "N frames; 0 disables feedback."
        ),
    )
    return parser.parse_args()


def configure_paper_imports(paper_repo: Path) -> None:
    for path in (paper_repo, paper_repo / "SurgicalSAM2"):
        if not path.exists():
            raise FileNotFoundError(path)
        sys.path.insert(0, str(path))


def tracker_args(
    args: argparse.Namespace,
    paper_repo: Path,
    *,
    hybrid: bool,
) -> Namespace:
    return Namespace(
        # The two visible jaws must be calibrated independently.  Forcing the
        # paper's symmetric-jaw shortcut made the second jaw collapse onto the
        # first whenever SAM temporarily omitted it.
        symmetric_jaw=False,
        searcher="CMA-ES",
        # Upstream Tracker updates its Kalman filter three times per video frame
        # (inside the three-axis unwrap loop).  We disable that buggy path and
        # apply one SE(3) temporal gate after each paper observation below.
        use_filter=False,
        filter_option="None",
        use_mix_angle=True,
        cos_reparams=True,
        use_prev_joint_angles=not hybrid,
        # The contour net cannot recover a jaw which SAM omitted.  Both tip
        # points are instead propagated directly from the user's frame-0 clicks.
        use_contour_tip_net=False,
        contour_tip_net_path=str(
            paper_repo / "ContourTipNet/models/cnn_model.pth"
        ),
        downscale_factor=2,
        popsize=args.popsize,
        final_iters=args.final_iters,
        use_render_loss=True,
        use_pts_loss=True,
        mse_weight=6.0,
        # This was the paper code's original distance setting before it was
        # disabled.  It penalizes the visibly wrong shaft-outside-mask solution.
        dist_weight=12e-7,
        app_weight=6e-6,
        pts_weight=3e-3,
    )


def ctrnet_args(inputs: TrackingInputs) -> Namespace:
    K = inputs.K_half
    return Namespace(
        use_gpu=True,
        trained_on_multi_gpus=False,
        height=540,
        width=960,
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        px=float(K[0, 2]),
        py=float(K[1, 2]),
        scale=1.0,
        use_nvdiffrast=True,
    )


def paper_stdev(*, hybrid: bool) -> torch.Tensor:
    stdev = torch.ones(10, dtype=torch.float32, device="cuda")
    stdev[:3] *= torch.tensor([1e-2, 1e-1, 1e-2], device="cuda")
    stdev[3:6] *= 1e-3
    stdev[6:] *= 5e-2
    stdev[6] *= 2.0
    stdev[7] *= 2.0
    stdev[8:] *= 2.0
    if hybrid:
        # The hybrid starts from the timestamp-matched strict state. It searches
        # only a local residual rather than rediscovering the instrument globally.
        stdev[:3] *= 0.5
        stdev[3:6] *= 0.5
        stdev[6:] *= 0.5
    return stdev


def initialize_paper_state(
    *,
    args: argparse.Namespace,
    model: object,
    robot_renderer: object,
    ref_mask_path: Path,
    keypoints: np.ndarray,
    intrinsics: Namespace,
    cache_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    from TuRBO.turbo.turbo_1 import Turbo1
    from diffcali.eval_dvrk.black_box_optimize import BayesOptBatchProblem

    if cache_path.exists() and not args.no_init_cache:
        cache = torch.load(cache_path, map_location="cuda", weights_only=False)
        return (
            cache["cTr"].to("cuda"),
            cache["joint_angles"].to("cuda"),
            float(cache["loss"]),
        )
    if args.sample_number % args.batch_size != 0:
        raise ValueError("--sample-number must be divisible by --batch-size")
    if args.sample_number <= args.batch_size:
        raise ValueError("--sample-number must be larger than --batch-size")

    problem = BayesOptBatchProblem(
        model=model,
        robot_renderer=robot_renderer,
        ref_mask_file=str(ref_mask_path),
        ref_keypoints=torch.as_tensor(keypoints, device="cuda", dtype=torch.float32),
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        px=intrinsics.px,
        py=intrinsics.py,
        batch_size=args.batch_size,
        ld1=3,
        ld2=3,
        ld3=3,
        batch_iters=args.batch_iters,
        lr=3e-3,
    )
    optimizer = Turbo1(
        f=problem,
        lb=np.asarray([0.10, 30.0, 0.0, 0.0, -1.5707, -1.3963, 0.0]),
        ub=np.asarray([0.17, 60.0, 360.0, 360.0, 0.0, 1.3963, 1.5707]),
        n_init=args.batch_size,
        max_evals=args.sample_number,
        batch_size=args.batch_size,
        max_cholesky_size=1000,
        n_training_steps=50,
        verbose=True,
        min_cuda=1000,
        device="cuda",
        batch_eval=True,
    )
    started = time.perf_counter()
    optimizer.optimize()
    elapsed = time.perf_counter() - started

    valid = torch.isfinite(problem.final_loss_batch)
    if not torch.any(valid):
        raise RuntimeError("Paper initializer did not produce any finite candidate")
    valid_losses = problem.final_loss_batch[valid]
    best_index = torch.argmin(valid_losses)
    cTr = problem.final_cTr_batch[valid][best_index].detach()
    joint_angles = problem.joint_angles_batch[valid][best_index].detach()
    loss = float(valid_losses[best_index].item())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cTr": cTr.cpu(),
            "joint_angles": joint_angles.cpu(),
            "loss": loss,
            "elapsed_seconds": elapsed,
            "sample_number": args.sample_number,
            "batch_iters": args.batch_iters,
        },
        cache_path,
    )
    print(
        f"Paper initialization completed in {elapsed:.1f}s: "
        f"loss={loss:.6g}, cTr={cTr.tolist()}, joints={joint_angles.tolist()}"
    )
    return cTr, joint_angles, loss


def clamp_hybrid_state(
    ctr: np.ndarray,
    joints: np.ndarray,
    strict_ctr: np.ndarray,
    strict_joints: np.ndarray,
    *,
    max_translation_m: float,
    max_rotation_rad: float,
    max_joint_rad: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    T_est = ctr_to_matrix(ctr)
    T_ref = ctr_to_matrix(strict_ctr)
    translation_delta = T_est[:3, 3] - T_ref[:3, 3]
    translation_norm = float(np.linalg.norm(translation_delta))
    if translation_norm > max_translation_m:
        translation_delta *= max_translation_m / translation_norm
    relative_rotation = Rotation.from_matrix(
        T_est[:3, :3] @ T_ref[:3, :3].T
    )
    rotation_vector = relative_rotation.as_rotvec()
    rotation_norm = float(np.linalg.norm(rotation_vector))
    if rotation_norm > max_rotation_rad:
        rotation_vector *= max_rotation_rad / rotation_norm
    T_clamped = T_ref.copy()
    T_clamped[:3, :3] = (
        Rotation.from_rotvec(rotation_vector).as_matrix() @ T_ref[:3, :3]
    )
    T_clamped[:3, 3] = T_ref[:3, 3] + translation_delta
    joint_delta = np.clip(
        np.asarray(joints) - np.asarray(strict_joints),
        -max_joint_rad,
        max_joint_rad,
    )
    return (
        matrix_to_ctr(T_clamped),
        (np.asarray(strict_joints) + joint_delta).astype(np.float32),
        {
            "raw_translation_delta_mm": translation_norm * 1000.0,
            "raw_rotation_delta_deg": np.degrees(rotation_norm),
            "clamped_translation_delta_mm": float(
                np.linalg.norm(translation_delta) * 1000.0
            ),
            "clamped_rotation_delta_deg": float(
                np.degrees(np.linalg.norm(rotation_vector))
            ),
        },
    )


def clamp_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm > maximum > 0.0:
        return vector * (maximum / norm)
    return vector


def constrain_paper_state(
    ctr: np.ndarray,
    joints: np.ndarray,
    previous_ctr: np.ndarray,
    previous_joints: np.ndarray,
    anchor_ctr: np.ndarray,
    anchor_joints: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Reject silhouette-equivalent 3-D flips while keeping image-only motion."""
    T_raw = ctr_to_matrix(ctr)
    T_prev = ctr_to_matrix(previous_ctr)
    T_anchor = ctr_to_matrix(anchor_ctr)

    step_translation = clamp_norm(
        T_raw[:3, 3] - T_prev[:3, 3], args.paper_max_step_mm / 1000.0
    )
    T_step = T_prev.copy()
    T_step[:3, 3] = T_prev[:3, 3] + step_translation
    step_rotation = Rotation.from_matrix(
        T_raw[:3, :3] @ T_prev[:3, :3].T
    ).as_rotvec()
    step_rotation = clamp_norm(
        step_rotation, np.deg2rad(args.paper_max_step_deg)
    )
    T_step[:3, :3] = (
        Rotation.from_rotvec(step_rotation).as_matrix() @ T_prev[:3, :3]
    )

    total_translation = clamp_norm(
        T_step[:3, 3] - T_anchor[:3, 3], args.paper_max_total_mm / 1000.0
    )
    total_rotation = Rotation.from_matrix(
        T_step[:3, :3] @ T_anchor[:3, :3].T
    ).as_rotvec()
    total_rotation = clamp_norm(
        total_rotation, np.deg2rad(args.paper_max_total_deg)
    )
    T_accepted = T_anchor.copy()
    T_accepted[:3, 3] = T_anchor[:3, 3] + total_translation
    T_accepted[:3, :3] = (
        Rotation.from_rotvec(total_rotation).as_matrix()
        @ T_anchor[:3, :3]
    )

    joint_step_limit = np.deg2rad(
        np.asarray(
            [
                args.paper_max_wrist_step_deg,
                args.paper_max_wrist_step_deg,
                args.paper_max_jaw_step_deg,
                args.paper_max_jaw_step_deg,
            ]
        )
    )
    accepted_joints = np.asarray(previous_joints) + np.clip(
        np.asarray(joints) - np.asarray(previous_joints),
        -joint_step_limit,
        joint_step_limit,
    )
    wrist_total_limit = np.deg2rad(args.paper_max_wrist_total_deg)
    accepted_joints[:2] = np.asarray(anchor_joints[:2]) + np.clip(
        accepted_joints[:2] - np.asarray(anchor_joints[:2]),
        -wrist_total_limit,
        wrist_total_limit,
    )
    accepted_joints[2:] = np.clip(accepted_joints[2:], 0.0, np.pi / 2.0)
    return (
        matrix_to_ctr(T_accepted),
        accepted_joints.astype(np.float32),
        {
            "raw_step_translation_mm": float(
                np.linalg.norm(T_raw[:3, 3] - T_prev[:3, 3]) * 1000.0
            ),
            "accepted_step_translation_mm": float(
                np.linalg.norm(step_translation) * 1000.0
            ),
            "raw_step_rotation_deg": float(
                np.degrees(
                    Rotation.from_matrix(
                        T_raw[:3, :3] @ T_prev[:3, :3].T
                    ).magnitude()
                )
            ),
            "accepted_step_rotation_deg": float(
                np.degrees(np.linalg.norm(step_rotation))
            ),
        },
    )


def update_lk_keypoints(
    previous_gray: np.ndarray,
    gray: np.ndarray,
    previous_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(previous_points, dtype=np.float32).reshape(-1, 1, 2)
    params = {
        "winSize": (31, 31),
        "maxLevel": 4,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            40,
            1e-3,
        ),
        "minEigThreshold": 1e-5,
    }
    forward, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, gray, points, None, **params
    )
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
        gray, previous_gray, forward, None, **params
    )
    fb_error = np.linalg.norm(backward - points, axis=2).reshape(-1)
    valid = (
        status.reshape(-1).astype(bool)
        & back_status.reshape(-1).astype(bool)
        & (fb_error < 2.0)
    )
    output = points.reshape(-1, 2).copy()
    output[valid] = forward.reshape(-1, 2)[valid]
    return output.astype(np.float32), valid


def snap_keypoints_to_jaw_contour(
    mask: np.ndarray,
    keypoints: np.ndarray,
    pivot: np.ndarray,
    *,
    search_radius_px: float = 65.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Move each LK seed to the far end of its nearby segmented jaw contour."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return np.asarray(keypoints, dtype=np.float32), np.zeros(2, dtype=bool)
    contour_points = np.concatenate(contours, axis=0).reshape(-1, 2).astype(np.float64)
    output = np.asarray(keypoints, dtype=np.float64).copy()
    snapped = np.zeros(len(output), dtype=bool)
    for index, seed in enumerate(output.copy()):
        seed_distance = np.linalg.norm(contour_points - seed, axis=1)
        pivot_distance = np.linalg.norm(contour_points - pivot, axis=1)
        candidates = (
            (seed_distance <= search_radius_px)
            & (pivot_distance >= 12.0)
            & (pivot_distance <= 120.0)
        )
        if not np.any(candidates):
            continue
        candidate_ids = np.flatnonzero(candidates)
        # Prefer the furthest point from the shared jaw pivot, with a small
        # penalty against jumping to another disconnected/spurious contour.
        score = pivot_distance[candidate_ids] - 0.15 * seed_distance[candidate_ids]
        output[index] = contour_points[candidate_ids[np.argmax(score)]]
        snapped[index] = True
    if snapped.all() and np.linalg.norm(output[0] - output[1]) < 12.0:
        # Do not let both identities collapse onto the same visible jaw.
        worse = int(np.argmin(np.linalg.norm(output - keypoints, axis=1)))
        output[1 - worse] = keypoints[1 - worse]
        snapped[1 - worse] = False
    return output.astype(np.float32), snapped


def save_checkpoint(
    path: Path,
    *,
    timestamps: np.ndarray,
    pure_ctr: list[np.ndarray],
    pure_joints: list[np.ndarray],
    pure_losses: list[float],
    hybrid_ctr: list[np.ndarray],
    hybrid_joints: list[np.ndarray],
    hybrid_losses: list[float],
    hybrid_residuals: list[dict[str, float]],
    masks_packbits: list[np.ndarray],
    mask_shape: tuple[int, int],
    pure_failed_frames: list[int],
    hybrid_failed_frames: list[int],
    tip_keypoints: list[np.ndarray],
) -> None:
    count = len(pure_ctr)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        video_timestamps=timestamps[:count],
        pure_ctr=np.asarray(pure_ctr, dtype=np.float32),
        pure_joints=np.asarray(pure_joints, dtype=np.float32),
        pure_losses=np.asarray(pure_losses, dtype=np.float32),
        hybrid_ctr=np.asarray(hybrid_ctr, dtype=np.float32),
        hybrid_joints=np.asarray(hybrid_joints, dtype=np.float32),
        hybrid_losses=np.asarray(hybrid_losses, dtype=np.float32),
        hybrid_raw_translation_delta_mm=np.asarray(
            [item["raw_translation_delta_mm"] for item in hybrid_residuals],
            dtype=np.float32,
        ),
        hybrid_raw_rotation_delta_deg=np.asarray(
            [item["raw_rotation_delta_deg"] for item in hybrid_residuals],
            dtype=np.float32,
        ),
        masks_packbits=np.stack(masks_packbits),
        mask_shape=np.asarray(mask_shape, dtype=np.int32),
        pure_failed_frames=np.asarray(pure_failed_frames, dtype=np.int32),
        hybrid_failed_frames=np.asarray(hybrid_failed_frames, dtype=np.int32),
        tip_keypoints=np.asarray(tip_keypoints, dtype=np.float32),
    )


def set_tracker_fallback_state(
    tracker: object,
    ctr_axis_angle: np.ndarray,
    joints: np.ndarray,
    axis_angle_to_mix_angle: object,
) -> None:
    """Restore a valid previous state after a frame has no CMA-ES solution.

    EvoTorch's CMA-ES logger occasionally has no finite best solution for a
    difficult mask, in which case the upstream Tracker raises while multiplying
    ``None``.  Keeping the previous pure-paper state (or the current strict
    prior for hybrid) lets the next frame run normally without changing any
    successful paper optimization.
    """
    ctr_tensor = torch.as_tensor(
        ctr_axis_angle, device="cuda", dtype=torch.float32
    )
    mix_rotation = axis_angle_to_mix_angle(
        ctr_tensor[:3].unsqueeze(0)
    ).squeeze(0)
    ctr_mix = torch.cat([mix_rotation, ctr_tensor[3:]])
    joints_tensor = torch.as_tensor(
        joints, device="cuda", dtype=torch.float32
    )
    tracker._prev_cTr = ctr_mix.detach().clone()
    tracker._prev_joint_angles = joints_tensor.detach().clone()
    if getattr(tracker, "filter", None) is not None:
        tracker.filter.reset(
            torch.cat([ctr_mix, joints_tensor]).detach().cpu().numpy()
        )


def safe_track_frame(
    *,
    tracker: object,
    fallback_ctr: np.ndarray,
    fallback_joints: np.ndarray,
    axis_angle_to_mix_angle: object,
    label: str,
    frame_index: int,
    **track_kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor, float, bool]:
    try:
        ctr, joints, loss = tracker.track_frame(**track_kwargs)
        if not (
            torch.isfinite(ctr).all()
            and torch.isfinite(joints).all()
            and torch.isfinite(torch.as_tensor(loss)).all()
        ):
            raise FloatingPointError("tracker returned a non-finite state")
        return ctr, joints, float(torch.as_tensor(loss).item()), False
    except Exception as exc:
        set_tracker_fallback_state(
            tracker,
            fallback_ctr,
            fallback_joints,
            axis_angle_to_mix_angle,
        )
        print(
            f"WARNING frame={frame_index:04d} {label} optimization failed: "
            f"{type(exc).__name__}: {exc}; using fallback state"
        )
        return (
            torch.as_tensor(fallback_ctr, device="cuda", dtype=torch.float32),
            torch.as_tensor(fallback_joints, device="cuda", dtype=torch.float32),
            float("nan"),
            True,
        )


def main() -> None:
    args = parse_args()
    paper_repo = args.paper_repo.resolve()
    configure_paper_imports(paper_repo)

    from diffcali.eval_dvrk.trackers import Tracker
    from diffcali.models.CtRNet import CtRNet
    from diffcali.utils.angle_transform_utils import axis_angle_to_mix_angle
    from sam2.build_sam import build_sam2_camera_predictor

    inputs = TrackingInputs.load()
    strict_ctr, strict_joints = inputs.strict_paper_states()
    annotation_dir = args.track_root / "online_videos/grasp5"
    prompts_path = annotation_dir / "PSM1_prompts.txt"
    keypoints_path = annotation_dir / "PSM1_keypoints.txt"
    ref_mask_path = annotation_dir / "PSM1_ref_mask.png"
    for path in (args.video, prompts_path, keypoints_path, ref_mask_path):
        if not path.exists():
            raise FileNotFoundError(path)
    prompts = np.loadtxt(prompts_path, ndmin=2)
    keypoints = np.loadtxt(keypoints_path, ndmin=2)
    if prompts.shape[1] != 3 or not ({0, 1} <= set(prompts[:, 2].astype(int))):
        raise ValueError("Prompts must contain both foreground and background labels")
    if keypoints.shape != (2, 2):
        raise ValueError(
            f"Exactly two jaw-tip keypoints are required, got {keypoints.shape}"
        )
    # User click order is [left-pointing jaw, vertical/right jaw].  The paper's
    # project_keypoints() order is [jaw-right, jaw-left].
    paper_keypoints = keypoints[[1, 0]].copy()

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
    tip_length = 0.0096
    p_local1 = torch.tensor(
        [0.0, 0.0004, tip_length], device="cuda", dtype=torch.float32
    )
    p_local2 = torch.tensor(
        [0.0, -0.0004, tip_length], device="cuda", dtype=torch.float32
    )

    checkpoint = (
        paper_repo / "SurgicalSAM2/checkpoints/sam2.1_hiera_s_endo18.pth"
    )
    predictor = build_sam2_camera_predictor(
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        str(checkpoint),
        vos_optimized=True,
    )

    pure_init_ctr, pure_init_joints, init_loss = initialize_paper_state(
        args=args,
        model=model,
        robot_renderer=robot_renderer,
        ref_mask_path=ref_mask_path,
        keypoints=paper_keypoints,
        intrinsics=intrinsics,
        cache_path=args.track_root / "paper_init_cache_tip_ordered.pth",
    )
    pure_args = tracker_args(args, paper_repo, hybrid=False)
    hybrid_args = tracker_args(args, paper_repo, hybrid=True)
    pure_tracker = Tracker(
        model=model,
        robot_renderer=robot_renderer,
        init_cTr=pure_init_ctr.clone(),
        init_joint_angles=pure_init_joints.clone(),
        num_iters=args.online_iters,
        stdev_init=paper_stdev(hybrid=False),
        intr=intr,
        p_local1=p_local1,
        p_local2=p_local2,
        searcher="CMA-ES",
        args=pure_args,
    )
    hybrid_tracker = Tracker(
        model=model,
        robot_renderer=robot_renderer,
        init_cTr=pure_init_ctr.clone(),
        init_joint_angles=pure_init_joints.clone(),
        num_iters=args.online_iters,
        stdev_init=paper_stdev(hybrid=True),
        intr=intr,
        p_local1=p_local1,
        p_local2=p_local2,
        searcher="CMA-ES",
        args=hybrid_args,
    )

    cap = cv2.VideoCapture(str(args.video))
    video_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = min(video_count, len(inputs.video_timestamps))
    if args.max_frames is not None:
        frame_count = min(frame_count, args.max_frames)
    if frame_count < 2:
        raise ValueError("At least two frames are required")

    pure_ctr: list[np.ndarray] = []
    pure_joints: list[np.ndarray] = []
    pure_losses: list[float] = []
    hybrid_ctr: list[np.ndarray] = []
    hybrid_joints: list[np.ndarray] = []
    hybrid_losses: list[float] = []
    hybrid_residuals: list[dict[str, float]] = []
    masks_packbits: list[np.ndarray] = []
    pure_failed_frames: list[int] = []
    hybrid_failed_frames: list[int] = []
    pure_constraint_stats: list[dict[str, float]] = []
    tip_keypoint_history: list[np.ndarray] = []
    partial_path = args.track_root / "tracking_states_partial.npz"
    started = time.perf_counter()
    previous_gray: np.ndarray | None = None
    optical_keypoints = keypoints.astype(np.float32).copy()
    paper_anchor_ctr: np.ndarray | None = None
    paper_anchor_joints: np.ndarray | None = None
    paper_to_current_frame4: np.ndarray | None = None

    for frame_index in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read video frame {frame_index}")
        frame = cv2.resize(frame, (intrinsics.width, intrinsics.height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous_gray is not None:
            optical_keypoints, lk_valid = update_lk_keypoints(
                previous_gray, gray, optical_keypoints
            )
            if not np.all(lk_valid):
                print(
                    f"WARNING frame={frame_index:04d} LK tip validity="
                    f"{lk_valid.tolist()}"
                )
        previous_gray = gray
        did_sam_reprompt = False
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
                # The camera predictor's interaction API expects every current
                # frame in this list, while its track() method does not append it.
                prepared, _, _ = predictor.perpare_data(
                    frame, image_size=predictor.image_size
                )
                predictor.condition_state["images"].append(prepared)
                if (
                    args.sam_reprompt_every > 0
                    and frame_index % args.sam_reprompt_every == 0
                ):
                    did_sam_reprompt = True
                    raw_mask = (
                        mask_logits.squeeze().detach().cpu().numpy() > 0
                    )
                    ys, xs = np.nonzero(raw_mask)
                    bbox = None
                    if len(xs):
                        margin = 35
                        bbox = np.asarray(
                            [
                                max(0, int(xs.min()) - margin),
                                max(0, int(ys.min()) - margin),
                                min(intrinsics.width - 1, int(xs.max()) + margin),
                                min(intrinsics.height - 1, int(ys.max()) + margin),
                            ],
                            dtype=np.float32,
                        )
                    _, _, mask_logits = predictor.add_new_prompt_during_track(
                        point=optical_keypoints,
                        bbox=bbox,
                        if_new_target=False,
                        obj_id=0,
                        labels=np.ones(2, dtype=np.int32),
                        clear_old_points=True,
                    )
        mask = (mask_logits.squeeze() > 0).float()
        mask_numpy = mask.detach().cpu().numpy().astype(bool)
        if did_sam_reprompt and pure_ctr:
            pivot_camera = paper_jaw_pivot_camera(
                pure_ctr[-1], pure_joints[-1]
            )
            if pivot_camera[2] > 1e-6:
                pivot_image = np.asarray(
                    [
                        intrinsics.fx * pivot_camera[0] / pivot_camera[2]
                        + intrinsics.px,
                        intrinsics.fy * pivot_camera[1] / pivot_camera[2]
                        + intrinsics.py,
                    ],
                    dtype=np.float32,
                )
                optical_keypoints, snapped = snap_keypoints_to_jaw_contour(
                    mask_numpy,
                    optical_keypoints,
                    pivot_image,
                )
                print(
                    f"frame={frame_index:04d} jaw_tip_snap="
                    f"{snapped.tolist()} points={optical_keypoints.tolist()}"
                )
        masks_packbits.append(np.packbits(mask_numpy.reshape(-1)))
        manual_keypoints = torch.as_tensor(
            optical_keypoints[[1, 0]], device="cuda", dtype=torch.float32
        )
        tip_keypoint_history.append(optical_keypoints.copy())

        pure_fallback_ctr = (
            pure_ctr[-1] if pure_ctr else pure_init_ctr.detach().cpu().numpy()
        )
        pure_fallback_joints = (
            pure_joints[-1]
            if pure_joints
            else pure_init_joints.detach().cpu().numpy()
        )
        (
            pure_state_ctr,
            pure_state_joints,
            pure_loss,
            pure_failed,
        ) = safe_track_frame(
            tracker=pure_tracker,
            fallback_ctr=pure_fallback_ctr,
            fallback_joints=pure_fallback_joints,
            axis_angle_to_mix_angle=axis_angle_to_mix_angle,
            label="pure-paper",
            frame_index=frame_index,
            ref_mask=mask,
            joint_angles=None,
            is_init=frame_index == 0,
            keypoints=manual_keypoints,
        )
        if pure_failed:
            pure_failed_frames.append(frame_index)
        raw_pure_ctr = pure_state_ctr.detach().cpu().numpy().copy()
        raw_pure_joints = pure_state_joints.detach().cpu().numpy().copy()
        if frame_index == 0:
            accepted_pure_ctr = raw_pure_ctr
            accepted_pure_joints = raw_pure_joints
            paper_anchor_ctr = accepted_pure_ctr.copy()
            paper_anchor_joints = accepted_pure_joints.copy()
            paper_to_current_frame4 = (
                np.linalg.inv(ctr_to_matrix(paper_anchor_ctr))
                @ ctr_to_matrix(strict_ctr[0])
            )
            pure_constraint_stats.append(
                {
                    "raw_step_translation_mm": 0.0,
                    "accepted_step_translation_mm": 0.0,
                    "raw_step_rotation_deg": 0.0,
                    "accepted_step_rotation_deg": 0.0,
                }
            )
            wrist_limit = np.deg2rad(args.paper_max_wrist_total_deg)
            for tracker in (pure_tracker, hybrid_tracker):
                tracker.problem.joint_angles_lb[:2] = torch.maximum(
                    tracker.problem.joint_angles_lb[:2],
                    torch.as_tensor(
                        paper_anchor_joints[:2] - wrist_limit,
                        device="cuda",
                        dtype=torch.float32,
                    ),
                )
                tracker.problem.joint_angles_ub[:2] = torch.minimum(
                    tracker.problem.joint_angles_ub[:2],
                    torch.as_tensor(
                        paper_anchor_joints[:2] + wrist_limit,
                        device="cuda",
                        dtype=torch.float32,
                    ),
                )
        else:
            assert paper_anchor_ctr is not None
            assert paper_anchor_joints is not None
            (
                accepted_pure_ctr,
                accepted_pure_joints,
                constraint_stats,
            ) = constrain_paper_state(
                raw_pure_ctr,
                raw_pure_joints,
                pure_ctr[-1],
                pure_joints[-1],
                paper_anchor_ctr,
                paper_anchor_joints,
                args,
            )
            pure_constraint_stats.append(constraint_stats)
        set_tracker_fallback_state(
            pure_tracker,
            accepted_pure_ctr,
            accepted_pure_joints,
            axis_angle_to_mix_angle,
        )
        pure_ctr.append(accepted_pure_ctr)
        pure_joints.append(accepted_pure_joints)
        pure_losses.append(pure_loss)

        assert paper_anchor_joints is not None
        assert paper_to_current_frame4 is not None
        paper_prior_ctr = matrix_to_ctr(
            ctr_to_matrix(strict_ctr[frame_index])
            @ np.linalg.inv(paper_to_current_frame4)
        )
        paper_prior_joints = (
            paper_anchor_joints
            + strict_joints[frame_index]
            - strict_joints[0]
        )
        paper_prior_joints[2:] = np.clip(
            paper_prior_joints[2:], 0.0, np.pi / 2.0
        )
        prior_ctr_tensor = torch.as_tensor(
            paper_prior_ctr, device="cuda", dtype=torch.float32
        )
        strict_mix = axis_angle_to_mix_angle(
            prior_ctr_tensor[:3].unsqueeze(0)
        ).squeeze(0)
        strict_mix_ctr = torch.cat([strict_mix, prior_ctr_tensor[3:]])
        set_tracker_fallback_state(
            hybrid_tracker,
            paper_prior_ctr,
            paper_prior_joints,
            axis_angle_to_mix_angle,
        )
        (
            hybrid_state_ctr,
            hybrid_state_joints,
            hybrid_loss,
            hybrid_failed,
        ) = safe_track_frame(
            tracker=hybrid_tracker,
            fallback_ctr=paper_prior_ctr,
            fallback_joints=paper_prior_joints,
            axis_angle_to_mix_angle=axis_angle_to_mix_angle,
            label="hybrid",
            frame_index=frame_index,
            ref_mask=mask,
            joint_angles=torch.as_tensor(
                paper_prior_joints,
                device="cuda",
                dtype=torch.float32,
            ),
            is_init=False,
            keypoints=manual_keypoints,
            cTr_init=strict_mix_ctr,
        )
        if hybrid_failed:
            hybrid_failed_frames.append(frame_index)
        clamped_ctr, clamped_joints, residual = clamp_hybrid_state(
            hybrid_state_ctr.detach().cpu().numpy(),
            hybrid_state_joints.detach().cpu().numpy(),
            paper_prior_ctr,
            paper_prior_joints,
            max_translation_m=args.hybrid_max_translation_mm / 1000.0,
            max_rotation_rad=np.deg2rad(args.hybrid_max_rotation_deg),
            max_joint_rad=np.deg2rad(args.hybrid_max_joint_deg),
        )
        hybrid_ctr.append(clamped_ctr)
        hybrid_joints.append(clamped_joints)
        hybrid_losses.append(hybrid_loss)
        hybrid_residuals.append(residual)

        elapsed = time.perf_counter() - started
        print(
            f"frame={frame_index:04d}/{frame_count - 1:04d} "
            f"pure_loss={pure_losses[-1]:.6g} "
            f"hybrid_loss={hybrid_losses[-1]:.6g} "
            f"hybrid_d={residual['clamped_translation_delta_mm']:.2f}mm/"
            f"{residual['clamped_rotation_delta_deg']:.2f}deg "
            f"paper_step={pure_constraint_stats[-1]['accepted_step_translation_mm']:.2f}mm/"
            f"{pure_constraint_stats[-1]['accepted_step_rotation_deg']:.2f}deg "
            f"elapsed={elapsed:.1f}s"
        )
        if (
            (frame_index + 1) % args.checkpoint_every == 0
            or frame_index + 1 == frame_count
        ):
            save_checkpoint(
                partial_path,
                timestamps=inputs.video_timestamps,
                pure_ctr=pure_ctr,
                pure_joints=pure_joints,
                pure_losses=pure_losses,
                hybrid_ctr=hybrid_ctr,
                hybrid_joints=hybrid_joints,
                hybrid_losses=hybrid_losses,
                hybrid_residuals=hybrid_residuals,
                masks_packbits=masks_packbits,
                mask_shape=mask_numpy.shape,
                pure_failed_frames=pure_failed_frames,
                hybrid_failed_frames=hybrid_failed_frames,
                tip_keypoints=tip_keypoint_history,
            )
    cap.release()

    full_run = frame_count == len(inputs.video_timestamps) == video_count
    output_path = args.track_root / (
        "tracking_states.npz"
        if full_run
        else f"tracking_states_preview_{frame_count}.npz"
    )
    partial_path.replace(output_path)
    pure_video_poses = inputs.paper_states_to_visual_poses(
        np.asarray(pure_ctr), np.asarray(pure_joints)
    )
    hybrid_video_poses = inputs.paper_states_to_visual_poses(
        np.asarray(hybrid_ctr), np.asarray(hybrid_joints)
    )
    np.savez_compressed(
        args.track_root / (
            "visual_poses_video.npz"
            if full_run
            else f"visual_poses_video_preview_{frame_count}.npz"
        ),
        video_timestamps=inputs.video_timestamps[:frame_count],
        link_names=np.asarray(inputs.link_names),
        pure_poses_rect_camera_xyz_xyzw=pure_video_poses,
        hybrid_poses_rect_camera_xyz_xyzw=hybrid_video_poses,
    )
    if full_run:
        save_runtime_driver(
            args.track_root / "psm_paper_pose_driver.npz",
            inputs,
            pure_video_poses,
        )
        save_runtime_driver(
            args.track_root / "psm_hybrid_pose_driver.npz",
            inputs,
            hybrid_video_poses,
        )
    report = {
        "paper_repo": str(paper_repo),
        "video": str(args.video),
        "frame_count": frame_count,
        "full_run": full_run,
        "paper_method": (
            "SurgicalSAM2 mask + paper CAD renderer + BO initialization + "
            "per-frame CMA-ES 6DoF frame-4 pose and visible joints"
        ),
        "hybrid_method": (
            "same image loss/CMA-ES, initialized every frame by strict LND q7; "
            "bounded camera-relative SE3 and distal-joint residual"
        ),
        "initialization_loss": init_loss,
        "mean_pure_loss": float(np.nanmean(pure_losses)),
        "mean_hybrid_loss": float(np.nanmean(hybrid_losses)),
        "pure_failed_frames": pure_failed_frames,
        "hybrid_failed_frames": hybrid_failed_frames,
        "paper_frame_registration": (
            "per-link paper CAD component to current PSM visual body, calibrated once "
            "on the manually initialized first frame"
        ),
        "paper_image_tip_tracking": (
            "two user-clicked frame-0 tips propagated by forward/backward LK optical flow; "
            f"SurgicalSAM2 re-prompted every {args.sam_reprompt_every} frames"
        ),
        "paper_temporal_constraints": {
            "max_step_translation_mm": args.paper_max_step_mm,
            "max_step_rotation_deg": args.paper_max_step_deg,
            "max_total_translation_mm": args.paper_max_total_mm,
            "max_total_rotation_deg": args.paper_max_total_deg,
            "max_wrist_step_deg": args.paper_max_wrist_step_deg,
            "max_wrist_total_deg": args.paper_max_wrist_total_deg,
            "max_jaw_step_deg": args.paper_max_jaw_step_deg,
        },
        "hybrid_translation_delta_mm_p50_p95_max": np.percentile(
            [item["clamped_translation_delta_mm"] for item in hybrid_residuals],
            [50, 95, 100],
        ).tolist(),
        "hybrid_rotation_delta_deg_p50_p95_max": np.percentile(
            [item["clamped_rotation_delta_deg"] for item in hybrid_residuals],
            [50, 95, 100],
        ).tolist(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.track_root / "tracking_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
