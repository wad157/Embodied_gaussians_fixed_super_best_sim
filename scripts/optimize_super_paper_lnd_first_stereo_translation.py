#!/usr/bin/env python3
"""Paper-style first-pair stereo calibration with XYZ translation only."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from build_super_raw_psm_kinematics import nearest_indices
from optimize_super_p420006_stereo_keyframes import (
    PAPER_REPO,
    RAW_ROOT,
    lnd_fk_batch,
    transform_matrix,
)
from optimize_super_p420006_stereo_sam2_online import (
    combine_renderer_groups,
    load_manual_first_tips,
    resize_target,
    sha256,
)
from optimize_super_paper_lnd_first_stereo_static import (
    FirstStereoObjective,
    render_overlay,
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
    "raw_paper_lnd_first_stereo_translation_v1"
)
UPSTREAM_COMMIT = "cb2a264167aaf05b5a9c20da885d48568f78311f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize strict stereo pair 0 for one shared XYZ translation; "
            "all rotations and every robot joint remain exactly raw."
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
    parser.add_argument("--translation-bound-mm", type=float, default=20.0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--seed", type=int, default=420008)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class TranslationOnlyPaperRenderer(PaperLNDStereoRenderer):
    """Exact paper FK with one left-camera translation and no other state."""

    def link_transforms(
        self,
        raw_parameters: torch.Tensor,
        raw_q7: torch.Tensor,
        T_left_base: torch.Tensor,
        side: str,
        base_rotation_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del base_rotation_override
        # Keep the upstream seven-dimensional CMA state layout
        # [rotvec(3), translation(3), q5(1)] for numerical compatibility, but
        # make the four forbidden dimensions identically zero in the decoder.
        translation = self.decode(raw_parameters)[:, 3:6]
        q7 = raw_q7.unsqueeze(0).expand(len(translation), -1)
        fk = lnd_fk_batch(q7, self.dh_parameters)
        paper_joints = torch.stack(
            [q7[:, 4], q7[:, 5], 0.5 * q7[:, 6], 0.5 * q7[:, 6]],
            dim=1,
        )
        components, _jaw_frame = paper_component_transforms(paper_joints)
        T_base = T_left_base.unsqueeze(0).expand(len(translation), -1, -1)
        T_left_frame4_raw = T_base @ fk[:, 4]
        identity_rotation = torch.eye(
            3, device=translation.device, dtype=translation.dtype
        ).unsqueeze(0).expand(len(translation), -1, -1)
        delta_T_left = transform_matrix(identity_rotation, translation)
        links = (delta_T_left @ T_left_frame4_raw)[:, None] @ components
        if side == "right":
            links = self.T_right_left[None, None] @ links
        return links


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
        args.visual_root / "keyframes.npz",
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

    with np.load(args.visual_root / "keyframes.npz", allow_pickle=False) as data:
        K_left = data["K_left_rect"].astype(np.float32)
        K_right = data["K_right_rect"].astype(np.float32)
        T_right_left = data[
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

    translation_bounds = np.full(
        3, args.translation_bound_mm * 1.0e-3, dtype=np.float32
    )
    renderer_bounds = np.asarray(
        [0.0, 0.0, 0.0, *translation_bounds.tolist(), 0.0],
        dtype=np.float32,
    )
    renderer = TranslationOnlyPaperRenderer(
        mesh_dir=args.mesh_dir,
        link_offsets=np.repeat(np.eye(4, dtype=np.float32)[None], 7, axis=0),
        lnd_link_ids=np.arange(7, dtype=np.int64),
        dh_parameters=raw_model["lnd"]["DH_params"],
        K_left=K_left,
        K_right=K_right,
        T_right_left=T_right_left,
        image_size=tuple(manifest["image_size_wh"]),
        render_scale=args.render_scale,
        bounds=renderer_bounds,
        device=device,
    )
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
    valid_generation_count = 0
    for iteration in range(args.iterations):
        searcher.step()
        candidate_eval = searcher.status.get("pop_best_eval")
        candidate = searcher.status.get("pop_best")
        if candidate_eval is None or candidate is None:
            history[iteration] = np.inf
            print(
                f"TRANSLATION-STEREO {iteration + 1:03d}/"
                f"{args.iterations}: empty CMA generation; skipped",
                flush=True,
            )
            continue
        candidate_loss = float(candidate_eval)
        valid_generation_count += 1
        history[iteration] = candidate_loss
        if candidate_loss < best_loss:
            best_loss = candidate_loss
            best_raw = candidate.values.detach().clone()
        print(
            f"TRANSLATION-STEREO {iteration + 1:03d}/{args.iterations}: "
            f"{candidate_loss:.7f} best={best_loss:.7f}",
            flush=True,
        )
    translation = (
        torch.tanh(best_raw[3:6])
        * torch.as_tensor(translation_bounds, device=device)
    ).cpu().numpy()
    boundary_fraction = np.abs(translation / translation_bounds)
    delta_T = np.eye(4, dtype=np.float64)
    delta_T[:3, 3] = translation

    correction_path = args.output_dir / "first_stereo_translation.npz"
    np.savez_compressed(
        correction_path,
        schema=np.asarray("super_paper_lnd_first_stereo_translation_xyz_v1"),
        strict_pair_slot=np.asarray(0, dtype=np.int64),
        left_q_index=np.asarray(left_q_index, dtype=np.int64),
        right_q_index=np.asarray(right_q_index, dtype=np.int64),
        raw_cma_parameters=best_raw.cpu().numpy(),
        parameter_bounds=translation_bounds,
        translation_xyz_m=translation,
        delta_T_rectified_left_camera=delta_T,
        rotation_vector_rad=np.zeros(3, dtype=np.float64),
        q5_zero_offset_rad=np.asarray(0.0, dtype=np.float64),
        initial_loss=np.asarray(initial_loss, dtype=np.float64),
        best_loss=np.asarray(best_loss, dtype=np.float64),
        generation_best_loss=history,
        per_frame_visual_corrections=np.zeros((0, 3), dtype=np.float32),
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
        label="translation XYZ only",
    )
    preview_path = args.output_dir / "previews/first_pair_before_after.png"
    if not cv2.imwrite(
        str(preview_path), np.concatenate([before, after], axis=0)
    ):
        raise RuntimeError(f"Failed to write {preview_path}")

    report = {
        "schema": "super_paper_lnd_first_stereo_translation_optimization_v1",
        "passed": bool(
            np.isfinite(translation).all()
            and best_loss < initial_loss
            and float(boundary_fraction.max()) < 0.95
            and valid_generation_count >= int(0.9 * args.iterations)
        ),
        "method": (
            "strict raw stereo pair 0; exact paper LND CAD/FK; one shared "
            "left-camera XYZ translation; rotation, q5 and all other joints "
            "frozen; no later visual observation or temporal state"
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
                "CMA-ES first-frame population=30 and iterations=100",
            ],
            "stereo_extension": (
                "one XYZ translation is evaluated in both rectified eyes; "
                "right pose is frozen T_right_left times left"
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
            "parameters": ["tx", "ty", "tz"],
            "iterations": args.iterations,
            "population": args.population,
            "active_parameter_count": 3,
            "cma_state_length": 7,
            "valid_generation_count": valid_generation_count,
            "render_size_wh": [renderer.width, renderer.height],
            "initial_loss": initial_loss,
            "best_loss": best_loss,
            "improvement_fraction": (initial_loss - best_loss) / initial_loss,
            "translation_mm": (translation * 1.0e3).tolist(),
            "rotation_vector_deg": [0.0, 0.0, 0.0],
            "q5_zero_offset_deg": 0.0,
            "maximum_boundary_fraction": float(boundary_fraction.max()),
        },
        "hard_constraints": {
            "rotation_vector_rad": [0.0, 0.0, 0.0],
            "all_q7_offsets_rad": [0.0] * 7,
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
