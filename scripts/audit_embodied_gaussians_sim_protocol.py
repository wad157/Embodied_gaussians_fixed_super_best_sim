#!/usr/bin/env python3
"""Fail-closed audit for the reconstructed Embodied Gaussians SIM baseline."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "baselines" / "embodied_gaussians_sim"
sys.path.insert(0, str(BASELINE_ROOT))
from protocol import (  # noqa: E402
    CAMERAS,
    DATASETS,
    HOLDOUT_OFFSET,
    HOLDOUT_STRIDE,
    INITIALIZATION_FRAME,
    PAPER_PARAMETERS,
    UPSTREAM_COMMIT,
    UPSTREAM_FILE_SHA256,
    future_frames,
    reconstruction_holdouts,
    resolve_dataset,
    sha256_file,
)
from embodied_gaussians.scene_builders.simple_body_builder import SimpleBodyBuilder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--body", type=Path)
    parser.add_argument(
        "--variant",
        choices=("paper-soft", "public-rigid"),
        default="paper-soft",
    )
    parser.add_argument("--actuation", choices=("shared-prescribed-boundary", "psm-fk-collision-only"), required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def audit(args: argparse.Namespace) -> dict:
    if args.variant == "public-rigid" and args.actuation != "psm-fk-collision-only":
        raise ValueError("public-rigid 只允许上游语义的 PSM/FK collision-only 驱动")
    dataset = resolve_dataset(REPO_ROOT, args.dataset_key, args.dataset)
    episode = json.loads((dataset / "episode.json").read_text(encoding="utf-8"))
    frames = int(DATASETS[args.dataset_key]["frames"])
    if int(episode["frames"]) != frames or int(episode["fps"]) != 30:
        raise ValueError("数据集帧数/帧率与固定协议不一致")
    if episode.get("privileged_evaluation_only") != ["ground_truth"]:
        raise ValueError("数据集没有把 ground_truth 声明为仅评测可用")

    depth_label = str(DATASETS[args.dataset_key]["depth"])
    depth_root = dataset / "estimated_depth" / depth_label
    required = [
        dataset / "cameras.json",
        depth_root / "depth_generation_summary.json",
        dataset / "task_inputs" / "psm_link_poses.npz",
        dataset / "task_inputs" / "phases.json",
        dataset / "evaluation" / "evaluation_points_30_non_grasp.json",
    ]
    for camera in CAMERAS:
        required.extend(
            [
                dataset / "videos" / f"{camera}.json",
                dataset / "rgb" / camera / f"{INITIALIZATION_FRAME:06d}.png",
                depth_root / camera / f"{INITIALIZATION_FRAME:06d}-depth.npy",
                dataset / "gui_assets" / "visual_force_masks" / camera / "report.json",
                dataset / "gui_assets" / "visual_force_masks" / camera / "tissue_masks_packbits.npy",
            ]
        )
    if args.actuation == "shared-prescribed-boundary":
        required.extend(
            [
                dataset / "task_inputs" / "known_grasp_region_boundary.npz",
                dataset / "task_inputs" / "known_grasp_region_boundary.json",
            ]
        )
    else:
        required.extend(
            [
                dataset / "gui_assets" / "official_psm_tip_meshes_v2.npz",
                dataset / "gui_assets" / "official_psm_tip_meshes_v2.json",
            ]
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"EG baseline 缺少输入：{missing}")

    simple_path = Path(inspect.getfile(SimpleBodyBuilder)).resolve()
    upstream_root = simple_path.parents[3]
    source_hashes = {}
    for relative, expected in UPSTREAM_FILE_SHA256.items():
        path = upstream_root / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"上游源码污染：{relative} -> {actual} != {expected}")
        source_hashes[relative] = actual

    forbidden_imports = (
        "online_tissue_stiffness",
        "sim_benchmark",
        "triangle_skin_contact",
        "flow_depth_particle_observer",
        "tissue_fixedsuperbest.npz",
        "MaterialTetrahedronXPBDProjector",
    )
    method_sources = [
        BASELINE_ROOT / "initialize.py",
        BASELINE_ROOT / "data_adapter.py",
    ]
    if args.variant == "paper-soft":
        method_sources.extend(
            [BASELINE_ROOT / "run_soft.py", BASELINE_ROOT / "shape_matching.py"]
        )
    else:
        method_sources.append(BASELINE_ROOT / "run_public_rigid.py")
    violations = {}
    for path in method_sources:
        text = path.read_text(encoding="utf-8")
        found = [token for token in forbidden_imports if token in text]
        if found:
            violations[path.name] = found
    if violations:
        raise ValueError(f"baseline 源码引用了禁止的方法组件：{violations}")

    body_report = None
    if args.body is not None:
        body = args.body.expanduser().resolve()
        metadata = body.with_suffix(".metadata.json")
        if not body.is_file() or not metadata.is_file():
            raise FileNotFoundError("body 或初始化 metadata 不存在")
        body_report = json.loads(metadata.read_text(encoding="utf-8"))
        if not bool(body_report.get("formal")):
            raise ValueError("smoke 初始化 body 不得进入正式测评")
        if body_report.get("dataset_key") != args.dataset_key:
            raise ValueError("body 与数据集不匹配")
        if body_report.get("input") != (
            "frame-0 calibrated stereo RGB + FoundationStereo depth + "
            "common tissue masks only"
        ):
            raise ValueError("body 初始化输入声明非法")
        if body_report.get("initialization_cameras") != list(CAMERAS):
            raise ValueError("body 初始化相机不是固定左右双目")
        if body_report.get("initialization_frame") != INITIALIZATION_FRAME:
            raise ValueError("body 初始化不是固定 frame 0")
        if body_report.get("initialization_depth") != depth_label:
            raise ValueError("body 初始化深度版本不是固定协议")
        observations = body_report.get("initialization_observations")
        if not isinstance(observations, list) or [
            value.get("camera") for value in observations
        ] != list(CAMERAS):
            raise ValueError("body 双目初始化 provenance 缺失或非法")
        if any(value.get("frame") != INITIALIZATION_FRAME for value in observations):
            raise ValueError("body provenance 包含 frame 0 以外的初始化观测")
        if body_report.get("particle_radius_m") != PAPER_PARAMETERS["particle_radius_m"]:
            raise ValueError("body 粒子半径不是固定公开默认值")
        extent = body_report.get("initial_stereo_geometry", {}).get(
            "oriented_bounding_box_extent_m"
        )
        if not isinstance(extent, list) or len(extent) != 3 or min(extent) <= 0.0:
            raise ValueError("body 双目初始化 OBB provenance 缺失或非法")

    holdout = reconstruction_holdouts(frames)
    future = future_frames(frames)
    report = {
        "schema": "fixedsuperbest.embodied_gaussians_protocol_audit.v1",
        "passed": True,
        "dataset_key": args.dataset_key,
        "dataset": str(dataset),
        "variant": args.variant,
        "frames": frames,
        "split": {
            "training": f"t < floor(4T/5) and t % {HOLDOUT_STRIDE} != {HOLDOUT_OFFSET}",
            "reconstruction_holdout_count": len(holdout),
            "future_start": frames * 4 // 5,
            "future_count": len(future),
        },
        "upstream": {
            "commit": UPSTREAM_COMMIT,
            "checkout": str(upstream_root),
            "verified_source_sha256": source_hashes,
            "public_capability": "rigid-only; paper shape matching absent",
        },
        "representation": (
            {
                "name": "paper-soft reconstruction",
                "source": "paper equations (4)-(5), Algorithms 2-3",
                "constraints": ["sphere collision", "overlapping local-neighbour shape matching"],
                "gaussian_binding": "one closest parent particle with rigid local transform",
                "forbidden_components_absent": list(forbidden_imports),
                "unpublished_detail_policy": {
                    "name": "shape stiffness k_S",
                    "value": PAPER_PARAMETERS["shape_constraint_projection"],
                    "selection": (
                        "no fitted adapter: full Eq. (5) projection; parameter-free "
                        "Delaunay geometric neighbourhood"
                    ),
                },
            }
            if args.variant == "paper-soft"
            else {
                "name": "official public rigid-only capability audit",
                "deformable_degrees_of_freedom": 0,
                "visual_force_parameters": "unmodified public defaults",
                "not_valid_as_paper_soft_result": True,
            }
        ),
        "actuation": {
            "mode": args.actuation,
            "classification": (
                "optional protocol-level prescribed boundary, not an EG observation or contribution"
                if args.actuation == "shared-prescribed-boundary"
                else "primary faithful mode: known PSM/FK particles and paper collision"
            ),
        },
        "online_inputs": [
            "frame-0 calibrated left/right RGB and RGB-only FoundationStereo depth for initialization",
            "prefix stereo RGB only on non-holdout first-80% frames",
            "fixed camera calibration",
            "GT tissue mask as the declared common external-baseline adapter",
            "known task_inputs actuation",
        ],
        "forbidden_until_evaluation": [
            "ground_truth tissue state/3D trajectories/material",
            "evaluation 30-point manifest",
            "future RGB",
        ],
        "paper_parameters": PAPER_PARAMETERS,
        "body": body_report,
    }
    return report


def main() -> None:
    args = parse_args()
    report = audit(args)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"拒绝覆盖协议审计：{output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
