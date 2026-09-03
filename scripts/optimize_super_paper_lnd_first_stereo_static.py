#!/usr/bin/env python3
"""Calibrate one static stereo SE(3) + paper-LND q5 zero offset.

Only strict stereo pair 0 is observed.  The optimized state is shared by the
two rectified cameras and contains exactly seven residual parameters:

    [left-camera rotvec(3), left-camera translation(3), wrist-pitch q5(1)]

There is deliberately no per-frame visual state, propagation, filter, q4/q6
or jaw correction.  The loss and CMA-ES initialization settings follow
online_dvrk_tracking's first-frame path; the only method extension is that one
physical state is rendered in both calibrated eyes.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from build_super_raw_psm_kinematics import nearest_indices
from optimize_super_p420006_stereo_keyframes import (
    PAPER_REPO,
    RAW_ROOT,
    axis_angle_matrix,
    lnd_fk_batch,
    transform_matrix,
)
from optimize_super_p420006_stereo_sam2_online import (
    combine_renderer_groups,
    load_manual_first_tips,
    render_whole_and_tips,
    resize_target,
    sha256,
    tip_keypoint_loss,
    unpack_mask,
)
from super_paper_lnd_stereo_renderer import (
    PAPER_LND_MESHES,
    PaperLNDStereoRenderer,
    paper_component_transforms,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
VISUAL_ROOT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_paper_lnd_stereo_v1"
)
SAM2_ROOT = VISUAL_ROOT / "surgicalsam2_stereo_sequence_v1"
OUTPUT_ROOT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/"
    "raw_paper_lnd_first_stereo_static_v1"
)
UPSTREAM_COMMIT = "cb2a264167aaf05b5a9c20da885d48568f78311f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize strict stereo pair 0 for one fixed camera SE(3) and "
            "one mechanically valid paper-LND q5 zero offset."
        )
    )
    parser.add_argument("--visual-root", type=Path, default=VISUAL_ROOT)
    parser.add_argument(
        "--masks",
        type=Path,
        default=SAM2_ROOT / "stereo_surgicalsam2_masks.npz",
    )
    parser.add_argument(
        "--sam2-report", type=Path, default=SAM2_ROOT / "report.json"
    )
    parser.add_argument(
        "--first-annotation",
        type=Path,
        default=VISUAL_ROOT / "annotations/keyframe_00.json",
    )
    parser.add_argument(
        "--kinematics", type=Path, default=RAW_ROOT / "kinematics.npz"
    )
    parser.add_argument(
        "--raw-model", type=Path, default=RAW_ROOT / "model.json"
    )
    parser.add_argument("--paper-repo", type=Path, default=PAPER_REPO)
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=PAPER_REPO / "urdfs/dVRK/meshes",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cuda-device", type=int, default=1)
    parser.add_argument("--render-scale", type=float, default=0.25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--seed", type=int, default=420007)
    parser.add_argument(
        "--freeze-q5",
        action="store_true",
        help=(
            "Optimize only the shared camera-frame SE(3). The seventh CMA "
            "coordinate is retained for upstream numerical compatibility, "
            "but q5 is read exactly from raw q7 and its offset is forced to 0."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_bounds() -> np.ndarray:
    return np.asarray(
        [
            *([math.radians(15.0)] * 3),
            12.0e-3,
            12.0e-3,
            16.0e-3,
            math.radians(30.0),
        ],
        dtype=np.float32,
    )


class StaticStereoPaperRenderer(PaperLNDStereoRenderer):
    """Paper renderer whose only state is shared static SE(3) + q5."""

    def link_transforms(
        self,
        raw_parameters: torch.Tensor,
        raw_q7: torch.Tensor,
        T_left_base: torch.Tensor,
        side: str,
        base_rotation_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del base_rotation_override
        correction = self.decode(raw_parameters)
        q7 = raw_q7.unsqueeze(0).expand(len(correction), -1).clone()
        # The fixed-q5 mode deliberately leaves every raw robot joint bitwise
        # untouched.  Keeping a seventh, inactive CMA coordinate avoids the
        # numerical failure seen in the upstream custom optimizer at lower
        # state dimensions while not exposing that coordinate to rendering.
        if not getattr(self, "freeze_q5", False):
            q7[:, 4] += correction[:, 6]
        fk = lnd_fk_batch(q7, self.dh_parameters)
        paper_joints = torch.stack(
            [q7[:, 4], q7[:, 5], 0.5 * q7[:, 6], 0.5 * q7[:, 6]],
            dim=1,
        )
        components, _jaw_frame = paper_component_transforms(paper_joints)
        T_base = T_left_base.unsqueeze(0).expand(len(correction), -1, -1)
        T_left_frame4_raw = T_base @ fk[:, 4]
        delta_T_left = transform_matrix(
            axis_angle_matrix(correction[:, :3]), correction[:, 3:6]
        )
        # A camera/hand-eye registration residual is left-multiplied once on
        # the common frame4.  Every component therefore remains on paper FK.
        T_left_frame4 = delta_T_left @ T_left_frame4_raw
        links = T_left_frame4[:, None] @ components
        if side == "right":
            links = self.T_right_left[None, None] @ links
        return links


def euclidean_outside_distance(target: torch.Tensor) -> torch.Tensor:
    """Match the paper init's lambda=0 Euclidean geodesic distance term."""

    binary = target.detach().cpu().numpy() > 0.5
    distance = cv2.distanceTransform(
        (~binary).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    ).astype(np.float32)
    return torch.as_tensor(distance, device=target.device)


class FirstStereoObjective:
    def __init__(
        self,
        *,
        renderer: StaticStereoPaperRenderer,
        raw_q7: dict[str, torch.Tensor],
        T_left_base: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        tips: dict[str, torch.Tensor],
    ) -> None:
        self.renderer = renderer
        self.raw_q7 = raw_q7
        self.T_left_base = T_left_base
        self.targets = targets
        self.tips = tips
        self.distance = {
            side: euclidean_outside_distance(target)
            for side, target in targets.items()
        }

    @torch.no_grad()
    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        total = torch.zeros(len(values), device=values.device)
        for side in ("left", "right"):
            predicted, predicted_tips = render_whole_and_tips(
                self.renderer,
                values,
                self.raw_q7[side],
                self.T_left_base[side],
                side,
            )
            target = self.targets[side].unsqueeze(0).expand_as(predicted)
            mse = F.mse_loss(predicted, target, reduction="none").mean(
                dim=(1, 2)
            )
            distance = (
                predicted * self.distance[side].unsqueeze(0)
            ).sum(dim=(1, 2))
            appearance = torch.abs(
                predicted.sum(dim=(1, 2)) - target.sum(dim=(1, 2))
            )
            points = tip_keypoint_loss(
                predicted_tips, self.tips[side], threshold=5.0
            )
            # Exact upstream first-frame weights.  pts_weight is raised from
            # the online 3e-3 to 5e-3 during initialization in trackers.py.
            total += (
                6.0 * mse
                + 12.0e-7 * distance
                + 6.0e-6 * appearance
                + 5.0e-3 * points
            )
        return total / 2.0


def render_overlay(
    *,
    renderer: StaticStereoPaperRenderer,
    raw_parameters: np.ndarray,
    raw_q7: dict[str, torch.Tensor],
    T_left_base: dict[str, torch.Tensor],
    targets_packed: dict[str, np.ndarray],
    mask_shape: tuple[int, int],
    images: dict[str, Path],
    target_tips: dict[str, torch.Tensor],
    label: str,
) -> np.ndarray:
    values = torch.as_tensor(
        raw_parameters[None], device=renderer.device, dtype=torch.float32
    )
    panels: list[np.ndarray] = []
    for side in ("left", "right"):
        with torch.no_grad():
            predicted, predicted_tips = render_whole_and_tips(
                renderer,
                values,
                raw_q7[side],
                T_left_base[side],
                side,
            )
        prediction = predicted[0].cpu().numpy() > 0.5
        target = unpack_mask(targets_packed[side], mask_shape)
        target = cv2.resize(
            target.astype(np.uint8),
            (renderer.width, renderer.height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        image = cv2.imread(str(images[side]), cv2.IMREAD_COLOR)
        image = cv2.resize(
            image,
            (renderer.width, renderer.height),
            interpolation=cv2.INTER_AREA,
        )
        tint = np.zeros_like(image)
        tint[target] = (255, 80, 20)
        tint[prediction] = (30, 40, 255)
        tint[target & prediction] = (60, 220, 60)
        active = target | prediction
        image[active] = cv2.addWeighted(
            image[active], 0.45, tint[active], 0.55, 0.0
        )
        for point in target_tips[side].cpu().numpy():
            cv2.circle(image, tuple(np.rint(point).astype(int)), 5, (0, 255, 255), 2)
        for point in predicted_tips[0].cpu().numpy():
            cv2.circle(image, tuple(np.rint(point).astype(int)), 5, (255, 255, 255), 2)
        cv2.putText(
            image,
            f"{side} {label}: blue=SAM2 red=CAD green=overlap",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(image)
    return np.concatenate(panels, axis=1)


def main() -> None:
    args = parse_args()
    manifest_path = args.visual_root / "pair_manifest.json"
    required = [
        manifest_path,
        args.masks,
        args.sam2_report,
        args.first_annotation,
        args.kinematics,
        args.raw_model,
    ]
    required.extend(args.mesh_dir / name for name in PAPER_LND_MESHES)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    commit = subprocess.run(
        ["git", "-C", str(args.paper_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"Unexpected upstream commit {commit}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.cuda_device)
    device = torch.device(f"cuda:{args.cuda_device}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sam2_report = json.loads(args.sam2_report.read_text(encoding="utf-8"))
    if (
        manifest.get("passed") is not True
        or manifest.get("schema")
        != "super_paper_lnd_stereo_first_frame_manifest_v1"
        or int(manifest.get("count", -1)) != 1
        or sam2_report.get("passed") is not True
        or sam2_report.get("sequence", {}).get(
            "complete_strict_pair_sequence"
        )
        is not True
    ):
        raise RuntimeError("Fresh first-pair/SurgicalSAM2 inputs are not accepted")
    row = manifest["keyframes"][0]
    if int(row["strict_pair_slot"]) != 0:
        raise RuntimeError("The calibration observation must be strict pair 0")
    raw_model = json.loads(args.raw_model.read_text(encoding="utf-8"))

    with np.load(args.masks, allow_pickle=False) as data:
        if data["schema"].item() != "super_paper_lnd_stereo_surgicalsam2_sequence_v1":
            raise RuntimeError("Unexpected SurgicalSAM2 mask schema")
        mask_shape = tuple(data["mask_shape"].astype(int).tolist())
        packed = {
            "left": data["left_masks_packbits"][0].copy(),
            "right": data["right_masks_packbits"][0].copy(),
        }
        stereo_left_index = data["stereo_left_index"].astype(np.int64)
        stereo_right_index = data["stereo_right_index"].astype(np.int64)
    with np.load(args.kinematics, allow_pickle=False) as raw:
        q7 = raw["q7"].astype(np.float32)
        T_left_lnd = raw["T_rectified_left_camera_lnd_link"].astype(np.float32)
        joint_ns = raw["joint_timestamps_ros_ns"].astype(np.int64)
        left_ns = raw["left_timestamps_ros_ns"].astype(np.int64)
        right_ns = raw["right_timestamps_ros_ns"].astype(np.int64)
        if not (
            np.array_equal(stereo_left_index, raw["stereo_left_index"])
            and np.array_equal(stereo_right_index, raw["stereo_right_index"])
        ):
            raise RuntimeError("SAM2 pair map differs from raw kinematics")
    left_q_index = int(
        nearest_indices(left_ns[stereo_left_index[:1]], joint_ns)[0]
    )
    right_q_index = int(
        nearest_indices(right_ns[stereo_right_index[:1]], joint_ns)[0]
    )
    if (left_q_index, right_q_index) != (
        int(row["left_q_index"]),
        int(row["right_q_index"]),
    ):
        raise RuntimeError("First-pair q7 mapping differs from fresh manifest")

    keyframes_path = args.visual_root / "keyframes.npz"
    with np.load(keyframes_path, allow_pickle=False) as keyframes:
        K_left = keyframes["K_left_rect"].astype(np.float32)
        K_right = keyframes["K_right_rect"].astype(np.float32)
        T_right_left = keyframes[
            "T_rectified_right_camera_rectified_left_camera"
        ].astype(np.float32)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} is not empty; pass --overwrite"
            )
        import shutil

        shutil.rmtree(args.output_dir)
    (args.output_dir / "previews").mkdir(parents=True, exist_ok=True)

    bounds = make_bounds()
    renderer = StaticStereoPaperRenderer(
        mesh_dir=args.mesh_dir,
        link_offsets=np.repeat(np.eye(4, dtype=np.float32)[None], 7, axis=0),
        lnd_link_ids=np.arange(7, dtype=np.int64),
        dh_parameters=raw_model["lnd"]["DH_params"],
        K_left=K_left,
        K_right=K_right,
        T_right_left=T_right_left,
        image_size=tuple(manifest["image_size_wh"]),
        render_scale=args.render_scale,
        bounds=bounds,
        device=device,
    )
    renderer.freeze_q5 = bool(args.freeze_q5)
    combine_renderer_groups(renderer)
    raw_q7 = {
        "left": torch.as_tensor(q7[left_q_index], device=device),
        "right": torch.as_tensor(q7[right_q_index], device=device),
    }
    T_left_base = {
        "left": torch.as_tensor(T_left_lnd[left_q_index, 0], device=device),
        "right": torch.as_tensor(T_left_lnd[right_q_index, 0], device=device),
    }
    targets = {
        side: resize_target(
            packed[side],
            mask_shape,
            (renderer.height, renderer.width),
            device,
        )
        for side in ("left", "right")
    }
    tips = load_manual_first_tips(
        annotation_path=args.first_annotation,
        render_size_hw=(renderer.height, renderer.width),
        device=device,
    )
    objective = FirstStereoObjective(
        renderer=renderer,
        raw_q7=raw_q7,
        T_left_base=T_left_base,
        targets=targets,
        tips=tips,
    )

    sys.path.insert(0, str(args.paper_repo))
    from diffcali.utils.cma_es import CMAES_cus
    from evotorch import Problem
    from torch.quasirandom import SobolEngine

    class ProblemAdapter(Problem):
        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped
            super().__init__(
                "min",
                solution_length=7,
                device=device,
                initial_bounds=(
                    [-float("inf")] * 7,
                    [float("inf")] * 7,
                ),
            )

        def _evaluate_batch(self, batch: Any) -> None:
            batch.set_evals(self.wrapped(batch.values))

    zero = torch.zeros(7, device=device)
    initial_loss = float(objective(zero[None])[0].item())
    torch.manual_seed(args.seed)
    searcher = CMAES_cus(
        ProblemAdapter(objective),
        center_init=zero,
        stdev_init=0.35,
        popsize=args.population,
        mu_size=min(15, args.population // 2),
        sobol=SobolEngine(7, scramble=True, seed=args.seed),
    )
    best_raw = zero.clone()
    best_loss = initial_loss
    history = np.empty(args.iterations, dtype=np.float32)
    for iteration in range(args.iterations):
        searcher.step()
        candidate_loss = float(searcher.status["pop_best_eval"])
        history[iteration] = candidate_loss
        if candidate_loss < best_loss:
            best_loss = candidate_loss
            best_raw = searcher.status["pop_best"].values.detach().clone()
        print(
            f"STATIC-STEREO {iteration + 1:03d}/{args.iterations}: "
            f"{candidate_loss:.7f} best={best_loss:.7f}",
            flush=True,
        )
    correction = (
        torch.tanh(best_raw) * torch.as_tensor(bounds, device=device)
    ).cpu().numpy()
    if args.freeze_q5:
        correction[6] = 0.0
    delta_T = np.eye(4, dtype=np.float64)
    from scipy.spatial.transform import Rotation

    delta_T[:3, :3] = Rotation.from_rotvec(correction[:3]).as_matrix()
    delta_T[:3, 3] = correction[3:6]
    active_parameter_count = 6 if args.freeze_q5 else 7
    boundary_fraction = np.abs(
        correction[:active_parameter_count] / bounds[:active_parameter_count]
    )

    correction_path = args.output_dir / (
        "first_stereo_se3_fixed_q5.npz"
        if args.freeze_q5
        else "first_stereo_static_correction.npz"
    )
    correction_schema = (
        "super_paper_lnd_first_stereo_se3_fixed_q5_v1"
        if args.freeze_q5
        else "super_paper_lnd_first_stereo_static_se3_q5_v1"
    )
    np.savez_compressed(
        correction_path,
        schema=np.asarray(correction_schema),
        strict_pair_slot=np.asarray(0, dtype=np.int64),
        left_q_index=np.asarray(left_q_index, dtype=np.int64),
        right_q_index=np.asarray(right_q_index, dtype=np.int64),
        raw_cma_parameters=best_raw.cpu().numpy(),
        parameter_bounds=bounds,
        correction=correction,
        delta_T_rectified_left_camera=delta_T,
        q5_zero_offset_rad=np.asarray(correction[6], dtype=np.float64),
        initial_loss=np.asarray(initial_loss, dtype=np.float64),
        best_loss=np.asarray(best_loss, dtype=np.float64),
        generation_best_loss=history,
        per_frame_visual_corrections=np.zeros((0, 7), dtype=np.float32),
    )

    images = {
        side: args.visual_root / row[f"{side}_image"]
        for side in ("left", "right")
    }
    before = render_overlay(
        renderer=renderer,
        raw_parameters=np.zeros(7, dtype=np.float32),
        raw_q7=raw_q7,
        T_left_base=T_left_base,
        targets_packed=packed,
        mask_shape=mask_shape,
        images=images,
        target_tips=tips,
        label="raw",
    )
    after = render_overlay(
        renderer=renderer,
        raw_parameters=best_raw.cpu().numpy(),
        raw_q7=raw_q7,
        T_left_base=T_left_base,
        targets_packed=packed,
        mask_shape=mask_shape,
        images=images,
        target_tips=tips,
        label=("static SE3; raw q5" if args.freeze_q5 else "static SE3+q5"),
    )
    preview_path = args.output_dir / "previews/first_pair_before_after.png"
    if not cv2.imwrite(str(preview_path), np.concatenate([before, after], axis=0)):
        raise RuntimeError(f"Failed to write {preview_path}")

    report_schema = (
        "super_paper_lnd_first_stereo_se3_fixed_q5_optimization_v1"
        if args.freeze_q5
        else "super_paper_lnd_first_stereo_static_optimization_v1"
    )
    report = {
        "schema": report_schema,
        "passed": bool(
            np.isfinite(correction).all()
            and best_loss < initial_loss
            and float(boundary_fraction.max()) < 0.95
        ),
        "method": (
            "strict raw stereo pair 0; exact paper LND CAD/FK; one shared "
            "left-camera SE(3); q5 and all other q7 joints exactly raw; no "
            "later visual observation, temporal optimization or filter"
            if args.freeze_q5
            else
            "strict raw stereo pair 0; exact paper LND CAD/FK; one shared "
            "left-camera SE(3) and one paper q5 zero offset; no later visual "
            "observation, temporal optimization or filter"
        ),
        "upstream_fidelity": {
            "repository": "https://github.com/hanyang-hu/online_dvrk_tracking",
            "commit": commit,
            "preserved": [
                "SurgicalSAM2 first-frame whole-instrument mask",
                "NvDiffRast hard silhouette without antialiasing",
                "mse_weight=6",
                "initialization distance_weight=12e-7",
                "appearance_area_weight=6e-6",
                "manual two-tip initialization pts_weight=5e-3",
                "permutation-invariant two-tip and centerline L1 loss",
                "CMA-ES first-frame population=min(70,30)=30",
                "final_iters=100",
            ],
            "stereo_extension": (
                "one six-DoF physical SE(3) is evaluated in both rectified "
                "eyes; right pose is frozen T_right_left times left"
                if args.freeze_q5
                else
                "one seven-parameter physical state is evaluated in both "
                "rectified eyes; right pose is frozen T_right_left times left"
            ),
            "static_backbone_extension": (
                "camera correction is left-multiplied on common frame4 and "
                + (
                    "paper component FK consumes unmodified raw q5; "
                    if args.freeze_q5
                    else "q5 is a constant paper-FK zero offset; "
                )
                + "no online stage exists"
            ),
            "distance_implementation": (
                "OpenCV exact Euclidean distance transform; equivalent to "
                "the upstream initialization's FastGeodis lambda=0 setting"
            ),
        },
        "inputs": {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256(manifest_path),
            "first_annotation": str(args.first_annotation.resolve()),
            "first_annotation_sha256": sha256(args.first_annotation),
            "surgicalsam2_masks": str(args.masks.resolve()),
            "surgicalsam2_masks_sha256": sha256(args.masks),
            "raw_kinematics": str(args.kinematics.resolve()),
            "raw_kinematics_sha256": sha256(args.kinematics),
        },
        "optimization": {
            "parameters": (
                ["rx", "ry", "rz", "tx", "ty", "tz"]
                if args.freeze_q5
                else ["rx", "ry", "rz", "tx", "ty", "tz", "q5"]
            ),
            "active_parameter_count": active_parameter_count,
            "cma_state_length": 7,
            "iterations": args.iterations,
            "population": args.population,
            "render_size_wh": [renderer.width, renderer.height],
            "initial_loss": initial_loss,
            "best_loss": best_loss,
            "improvement_fraction": (initial_loss - best_loss) / initial_loss,
            "rotation_vector_deg": np.degrees(correction[:3]).tolist(),
            "translation_mm": (correction[3:6] * 1.0e3).tolist(),
            "q5_zero_offset_deg": float(np.degrees(correction[6])),
            "maximum_boundary_fraction": float(boundary_fraction.max()),
        },
        "hard_constraints": {
            "q4_offset_rad": 0.0,
            "q6_offset_rad": 0.0,
            "q7_jaw_offset_rad": 0.0,
            "q5_offset_rad": 0.0 if args.freeze_q5 else float(correction[6]),
            "all_q7_offsets_rad": (
                [0.0] * 7 if args.freeze_q5 else None
            ),
            "per_frame_visual_state_count": 0,
            "right_eye_independent_pose_parameters": 0,
        },
        "outputs": {
            "correction": str(correction_path.resolve()),
            "preview": str(preview_path.resolve()),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["optimization"], indent=2))
    print(f"Report: {report_path}")
    print(f"Correction: {correction_path}")
    print(f"Preview: {preview_path}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
