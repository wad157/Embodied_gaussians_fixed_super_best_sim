#!/usr/bin/env python3
"""Build an EG Gaussian-particle body from the permitted frame-0 stereo RGB-D.

The optimization calls the pinned upstream ``SimpleBodyBuilder`` primitives.
The only adapter logic is deterministic file loading, camera resizing, a
non-contacting ground plane required by the public builder API, and persistence
of provenance.  No task trajectory, evaluation manifest, or ground truth is
opened here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
import warp as wp
from PIL import Image
from scipy.spatial import cKDTree


BASELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BASELINE_ROOT.parents[1]
sys.path.insert(0, str(BASELINE_ROOT))
from protocol import (  # noqa: E402
    CAMERAS,
    INITIALIZATION_FRAME,
    PAPER_PARAMETERS,
    UPSTREAM_COMMIT,
    UPSTREAM_FILE_SHA256,
    dataset_spec,
    read_json,
    resolve_dataset,
    sha256_file,
)

from embodied_gaussians.scene_builders.domain import (  # noqa: E402
    Body,
    Ground,
    MaskedPosedImageAndDepth,
)
from embodied_gaussians.scene_builders.simple_body_builder import (  # noqa: E402
    SimpleBodyBuilder,
)
import embodied_gaussians.scene_builders.simple_body_builder as upstream_builder_module  # noqa: E402


def install_gsplat_compatibility_adapter() -> None:
    """Bridge the pinned builder's batched background API to gsplat 1.5.

    The upstream code supplies one background per camera but newer gsplat
    defaults to packed projection, whose low-level background shape is not
    batched.  Selecting unpacked output restores the upstream tensor contract;
    it changes neither RGB loss nor any optimized parameter.
    """
    original = upstream_builder_module.rasterization

    def compatible_rasterization(*args, **kwargs):
        kwargs.setdefault("packed", False)
        return original(*args, **kwargs)

    upstream_builder_module.rasterization = compatible_rasterization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=("sim01", "sim02", "sim03"), required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=int(PAPER_PARAMETERS["online_width"]))
    parser.add_argument("--particle-iterations", type=int, default=int(PAPER_PARAMETERS["initial_particle_iterations"]))
    parser.add_argument("--gaussian-iterations", type=int, default=int(PAPER_PARAMETERS["initial_gaussian_iterations"]))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="工程自检：仍使用固定双目，只把两阶段优化各减至1次；产物标记为非正式。",
    )
    return parser.parse_args()


def assert_upstream_sources() -> dict[str, str]:
    package_root = Path(SimpleBodyBuilder.__module__.replace(".", "/"))
    del package_root
    import inspect

    simple_path = Path(inspect.getfile(SimpleBodyBuilder)).resolve()
    checkout = simple_path.parents[3]
    actual = {}
    for relative, expected_hash in UPSTREAM_FILE_SHA256.items():
        path = checkout / relative
        digest = sha256_file(path)
        if digest != expected_hash:
            raise ValueError(
                f"上游源码哈希不一致：{path} -> {digest}，期望 {expected_hash}"
            )
        actual[relative] = digest
    return actual


def load_initial_stereo_datapoints(
    dataset: Path,
    *,
    dataset_key: str,
    width: int,
) -> tuple[list[MaskedPosedImageAndDepth], list[dict[str, object]]]:
    """Load exactly the two medical stereo observations at frame zero.

    FoundationStereo depths are the same RGB-only caches used by the other
    external baselines.  Dataset ``cameras.json`` stores both Blender/OpenGL
    and ROS optical poses; the upstream dataclass explicitly requires Blender
    ``X_WC`` and performs the optical-axis conversion internally.
    """
    cameras = read_json(dataset / "cameras.json")
    depth_label = str(dataset_spec(dataset_key)["depth"])
    depth_root = dataset / "estimated_depth" / depth_label
    result = []
    provenance = []
    for camera in CAMERAS:
        video_path = dataset / "videos" / f"{camera}.json"
        video = read_json(video_path)
        rgb_path = dataset / "rgb" / camera / f"{INITIALIZATION_FRAME:06d}.png"
        depth_path = depth_root / camera / f"{INITIALIZATION_FRAME:06d}-depth.npy"
        mask_root = dataset / "gui_assets" / "visual_force_masks" / camera
        mask_report_path = mask_root / "report.json"
        mask_report = read_json(mask_report_path)
        packed_path = mask_root / "tissue_masks_packbits.npy"

        image = np.asarray(Image.open(rgb_path).convert("RGB"))
        depth = np.load(depth_path).astype(np.float32)
        packed = np.load(packed_path, mmap_mode="r")
        if INITIALIZATION_FRAME >= packed.shape[0]:
            raise ValueError(f"{camera} mask 缺少 frame {INITIALIZATION_FRAME}")
        mask_width = int(mask_report["resolution_wh"][0])
        mask = (
            np.unpackbits(
                np.asarray(packed[INITIALIZATION_FRAME]), axis=1, bitorder="big"
            )[:, :mask_width]
            > 0
        )
        if image.shape[:2] != depth.shape or depth.shape != mask.shape:
            raise ValueError(f"{camera} frame-0 RGB/depth/mask 尺寸不一致")
        declared_size = tuple(int(value) for value in video["resolution"])
        if declared_size != (image.shape[1], image.shape[0]):
            raise ValueError(f"{camera} 图像尺寸与视频标定不一致")
        height = int(round(image.shape[0] * float(width) / image.shape[1]))
        size = (int(width), height)
        scale_x = float(width) / image.shape[1]
        scale_y = float(height) / image.shape[0]
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
        depth = np.where(np.isfinite(depth) & (depth > 0.0), depth, 0.0).astype(np.float32)
        intrinsic = np.asarray(video["K"], dtype=np.float32).copy()
        intrinsic[0, :] *= scale_x
        intrinsic[1, :] *= scale_y
        intrinsic[2, :] = (0.0, 0.0, 1.0)
        result.append(
            MaskedPosedImageAndDepth(
                X_WC=np.asarray(cameras[camera]["X_WC"], dtype=np.float32),
                K=intrinsic,
                image=np.ascontiguousarray(image),
                format="rgb",
                depth=np.ascontiguousarray(depth),
                depth_scale=1.0,
                mask=np.ascontiguousarray(mask),
            )
        )
        provenance.append(
            {
                "camera": camera,
                "frame": INITIALIZATION_FRAME,
                "rgb_sha256": sha256_file(rgb_path),
                "depth_sha256": sha256_file(depth_path),
                "mask_frame_sha256": hashlib.sha256(
                    np.ascontiguousarray(mask).tobytes()
                ).hexdigest(),
                "video_metadata_sha256": sha256_file(video_path),
                "mask_report_sha256": sha256_file(mask_report_path),
                "source_resolution_wh": list(declared_size),
                "depth_valid_fraction_in_mask": float(
                    np.count_nonzero((depth > 0.0) & np.isfinite(depth) & mask)
                    / max(np.count_nonzero(mask), 1)
                ),
            }
        )
    return result, provenance


def geometry_pointcloud(datapoints: list[MaskedPosedImageAndDepth]) -> o3d.geometry.PointCloud:
    # Public SimpleBodyBuilder routes RGB-D through Open3D's millimetre default.
    # Passing geometry-only copies makes its already-present depth_scale branch
    # use metres correctly; photometric optimization still uses the RGB copies.
    geometry = [
        MaskedPosedImageAndDepth(
            X_WC=item.X_WC.copy(),
            K=item.K.copy(),
            image=None,  # type: ignore[arg-type]
            format="rgb",
            depth=item.depth.copy(),
            depth_scale=1.0,
            mask=item.mask.copy(),
        )
        for item in datapoints
    ]
    pointcloud = SimpleBodyBuilder._merge_into_pointcloud(geometry, max_depth=2.0)
    if pointcloud is None:
        raise RuntimeError("frame-0 双目 RGB-D 没有生成点云")
    return pointcloud


def build_body(
    datapoints: list[MaskedPosedImageAndDepth],
    *,
    name: str,
    radius: float,
    particle_iterations: int,
    gaussian_iterations: int,
) -> tuple[Body, dict[str, object]]:
    pointcloud = geometry_pointcloud(datapoints)
    bounding_box = SimpleBodyBuilder._filter_and_get_bounding_box(
        pointcloud, outlier_radius=0.01, outlier_nb_points=20
    )
    if bounding_box is None:
        raise RuntimeError("frame-0 双目点云无法计算包围盒")
    initial = SimpleBodyBuilder._fill_bounding_box_with_spheres(bounding_box, radius)
    bounding_box_candidate_count = len(initial)
    initial = initial[SimpleBodyBuilder._prune_points_not_in_masks(initial, datapoints)]
    if len(initial) < 4:
        raise RuntimeError(f"mask 剪枝后粒子过少：{len(initial)}")

    # Use the public builder's unmodified z=0 ground.  The observed tissue is
    # above it, so no data-specific ground offset is introduced.
    ground = Ground()
    from embodied_gaussians.scene_builders.domain import GaussianLearningRates

    rates = GaussianLearningRates(means=1.0e-4)
    particles = SimpleBodyBuilder._optimize_particles(
        initial_points=initial,
        radius=radius,
        num_iterations=particle_iterations,
        learning_rates=rates,
        ground=ground,
        opacity_threshold=float(PAPER_PARAMETERS["initial_opacity_threshold"]),
        datapoints=datapoints,
        max_depth=2.0,
        cohesion_distance=0.002,
        visualize=False,
    )
    if len(particles.means) < 4:
        raise RuntimeError("上游粒子初始化剪枝后不足四个粒子")
    gaussians = SimpleBodyBuilder._grow_gaussians(
        initial_points=initial,
        radius=radius,
        num_iterations=gaussian_iterations,
        learning_rates=rates,
        datapoints=datapoints,
        min_scale=0.5 * radius,
        max_scale=2.0 * radius,
        max_depth=2.0,
        visualize=False,
    )
    particle_points = np.asarray(particles.means, dtype=np.float32)
    gaussian_points = np.asarray(gaussians.means, dtype=np.float32)
    distances, _ = cKDTree(particle_points).query(gaussian_points, k=1, workers=-1)
    gaussians = gaussians.mask(distances <= 2.3 * radius)
    if len(gaussians.means) == 0:
        raise RuntimeError("上游 Gaussian 初始化全部远离粒子")
    transform = SimpleBodyBuilder._convert_to_body_frame(gaussians, particles)
    body = Body(
        name=name,
        X_WB=transform.tolist(),
        particles=particles,
        gaussians=gaussians,
    )
    geometry_report = {
        "merged_point_count": len(pointcloud.points),
        "oriented_bounding_box_center_m": np.asarray(
            bounding_box.center, dtype=np.float64
        ).tolist(),
        "oriented_bounding_box_extent_m": np.asarray(
            bounding_box.extent, dtype=np.float64
        ).tolist(),
        "bounding_box_candidate_particle_count": bounding_box_candidate_count,
        "mask_pruned_candidate_particle_count": len(initial),
    }
    return body, geometry_report


def main() -> None:
    args = parse_args()
    if args.seed < 0 or args.width < 64:
        raise ValueError("seed/width 参数非法")
    if args.particle_iterations < 1 or args.gaussian_iterations < 1:
        raise ValueError("初始化迭代次数必须为正")
    if not args.smoke and (
        args.width != int(PAPER_PARAMETERS["online_width"])
        or args.particle_iterations
        != int(PAPER_PARAMETERS["initial_particle_iterations"])
        or args.gaussian_iterations
        != int(PAPER_PARAMETERS["initial_gaussian_iterations"])
    ):
        raise ValueError("正式初始化不允许覆盖论文 width/n/m 参数")
    dataset = resolve_dataset(REPO_ROOT, args.dataset_key, args.dataset)
    output = args.output.expanduser().resolve()
    metadata_path = output.with_suffix(".metadata.json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"拒绝覆盖已有 EG 初始化产物：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("EG 上游 SimpleBodyBuilder 需要 CUDA")
    torch.cuda.manual_seed_all(args.seed)
    wp.init()
    install_gsplat_compatibility_adapter()
    source_hashes = assert_upstream_sources()
    particle_iterations = 1 if args.smoke else args.particle_iterations
    gaussian_iterations = 1 if args.smoke else args.gaussian_iterations
    datapoints, input_provenance = load_initial_stereo_datapoints(
        dataset, dataset_key=args.dataset_key, width=args.width
    )
    body, geometry_report = build_body(
        datapoints,
        name=f"{dataset_spec(args.dataset_key)['name']}_eg_paper_soft",
        radius=float(PAPER_PARAMETERS["particle_radius_m"]),
        particle_iterations=particle_iterations,
        gaussian_iterations=gaussian_iterations,
    )
    output.write_text(body.model_dump_json(indent=2) + "\n", encoding="utf-8")
    metadata = {
        "schema": "fixedsuperbest.embodied_gaussians_initialization.v1",
        "formal": not args.smoke,
        "dataset_key": args.dataset_key,
        "dataset": str(dataset),
        "seed": args.seed,
        "input": "frame-0 calibrated stereo RGB + FoundationStereo depth + common tissue masks only",
        "forbidden_inputs_opened": [],
        "camera_manifest_sha256": sha256_file(dataset / "cameras.json"),
        "depth_generation_summary_sha256": sha256_file(
            dataset
            / "estimated_depth"
            / str(dataset_spec(args.dataset_key)["depth"])
            / "depth_generation_summary.json"
        ),
        "initialization_cameras": list(CAMERAS),
        "initialization_frame": INITIALIZATION_FRAME,
        "initialization_depth": str(dataset_spec(args.dataset_key)["depth"]),
        "initialization_observations": input_provenance,
        "initialization_width": args.width,
        "particle_radius_m": float(PAPER_PARAMETERS["particle_radius_m"]),
        "particle_iterations": particle_iterations,
        "gaussian_iterations": gaussian_iterations,
        "opacity_threshold": float(PAPER_PARAMETERS["initial_opacity_threshold"]),
        "particle_count": len(body.particles.means),
        "gaussian_count": len(body.gaussians.means),
        "initial_stereo_geometry": geometry_report,
        "upstream_commit": UPSTREAM_COMMIT,
        "verified_upstream_source_sha256": source_hashes,
        "known_upstream_difference_from_paper": (
            "public SimpleBodyBuilder grow stage has no operative densification despite "
            "the paper describing densification"
        ),
        "compatibility_adapter": (
            "gsplat 1.5 packed projection rejects upstream per-camera backgrounds; "
            "request unpacked output; RGB+D loss and optimizer unchanged"
        ),
        "ground": "public Ground default z=0 in initialization and runtime",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
