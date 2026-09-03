#!/usr/bin/env python3
"""Build one ellipsoidal Gaussian at every fine visual-triangle centroid.

The tetrahedral PBD mesh remains the mechanics and collision representation.
The dense stereo-union surface is retained as a separate visual mesh.  Every
visual vertex is embedded in the physical boundary with barycentric weights and
a rest offset transported by the physical face frame.  A Gaussian center is
always the exact arithmetic centroid of its deformed visual triangle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch
from tqdm import tqdm
from gsplat.rendering import rasterization


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_gaussians.scene_builders.domain import GaussianActivations  # noqa: E402
from embodied_gaussians.scene_builders.simple_body_builder import (  # noqa: E402
    SimpleBodyBuilder,
)
from build_super_multiview_tissue_gaussians import (  # noqa: E402
    MULTIVIEW_ROOT,
    bind_to_surface_faces,
    five_number,
    make_observations,
    project_visibility,
    read_json,
    sha256,
    surface_scene,
    write_binding_preview,
)


DEFAULT_PHYSICS_ASSET = (
    MULTIVIEW_ROOT
    / "paper_pbd_tissue_v15_denser_mild_paper_constraints_physics/"
    "tissue_soft_adaptive.npz"
)
DEFAULT_SOURCE_GAUSSIAN_ASSET = (
    MULTIVIEW_ROOT
    / "paper_pbd_tissue_v4_dense_multiview_uncropped/"
    "tissue_paper_pbd_dense_multiview.npz"
)
DEFAULT_FINE_SURFACE = (
    MULTIVIEW_ROOT / "rest_surface_adaptive_grasp_v1/rest_surface.npz"
)
DEFAULT_FINE_SURFACE_REPORT = (
    MULTIVIEW_ROOT / "rest_surface_adaptive_grasp_v1/report.json"
)
DEFAULT_OUTPUT_DIR = (
    MULTIVIEW_ROOT
    / "paper_pbd_tissue_v15_denser_mild_paper_constraints_centroid_ellipsoids"
)
DEFAULT_REUSE_APPEARANCE_ASSET = (
    MULTIVIEW_ROOT
    / "paper_pbd_tissue_v14_center_mild_paper_constraints_centroid_ellipsoids/"
    "tissue_paper_pbd_centroid_gaussians.npz"
)
NATIVE_ROOT = REPO_ROOT / "data/super/grasp5_native"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics-asset", type=Path, default=DEFAULT_PHYSICS_ASSET)
    parser.add_argument(
        "--source-gaussian-asset",
        type=Path,
        default=DEFAULT_SOURCE_GAUSSIAN_ASSET,
        help="Existing dense ellipsoids used only to initialize visual attributes.",
    )
    parser.add_argument("--fine-surface", type=Path, default=DEFAULT_FINE_SURFACE)
    parser.add_argument(
        "--fine-surface-report", type=Path, default=DEFAULT_FINE_SURFACE_REPORT
    )
    parser.add_argument(
        "--stage-b-report", type=Path, default=MULTIVIEW_ROOT / "stage_b_report.json"
    )
    parser.add_argument(
        "--calibration", type=Path, default=NATIVE_ROOT / "calib_rectified.json"
    )
    parser.add_argument(
        "--cameras",
        type=Path,
        default=REPO_ROOT / "data/super/grasp5_offline_demo/cameras.json",
    )
    parser.add_argument("--rgb-dir", type=Path, default=NATIVE_ROOT / "rgb")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--reuse-appearance-asset",
        type=Path,
        default=DEFAULT_REUSE_APPEARANCE_ASSET,
        help=(
            "Copy the already optimized ellipsoid RGB/opacity/quaternion/scale "
            "when its visual centroids exactly match this fine surface."
        ),
    )
    parser.add_argument("--asset-version", type=int, default=15)
    parser.add_argument("--max-gaussians", type=int, default=60000)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--training-resolution-scale", type=float, default=1.0)
    parser.add_argument("--minimum-scale-mm", type=float, default=0.175)
    parser.add_argument("--maximum-scale-mm", type=float, default=0.700)
    parser.add_argument("--maximum-depth-m", type=float, default=0.25)
    parser.add_argument("--visibility-depth-tolerance-mm", type=float, default=2.0)
    parser.add_argument(
        "--observation-confidence-policy",
        choices=("acceptable", "dense_union"),
        default="dense_union",
    )
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def face_frames(face_points: np.ndarray) -> np.ndarray:
    """Return right-handed [tangent-x, tangent-y, normal] column frames."""
    tangent_x = face_points[:, 1] - face_points[:, 0]
    normal = np.cross(tangent_x, face_points[:, 2] - face_points[:, 0])
    edge_norm = np.linalg.norm(tangent_x, axis=1, keepdims=True)
    normal_norm = np.linalg.norm(normal, axis=1, keepdims=True)
    if np.any(edge_norm <= 1.0e-12) or np.any(normal_norm <= 1.0e-12):
        raise RuntimeError("Fine visual surface contains a degenerate triangle")
    tangent_x = tangent_x / edge_norm
    normal = normal / normal_norm
    tangent_y = np.cross(normal, tangent_x)
    tangent_y /= np.maximum(
        np.linalg.norm(tangent_y, axis=1, keepdims=True), 1.0e-12
    )
    return np.stack((tangent_x, tangent_y, normal), axis=-1).astype(np.float32)


def inverse_sigmoid(values: np.ndarray) -> torch.Tensor:
    clipped = np.clip(values, 1.0e-5, 1.0 - 1.0e-5)
    return torch.from_numpy(np.log(clipped / (1.0 - clipped))).float().cuda()


def optimize_fixed_centroid_appearance(
    *,
    means: np.ndarray,
    initial_quats: np.ndarray,
    initial_scales: np.ndarray,
    initial_opacities: np.ndarray,
    initial_colors: np.ndarray,
    datapoints: list,
    iterations: int,
    minimum_scale_m: float,
    maximum_scale_m: float,
    maximum_depth_m: float,
) -> dict[str, np.ndarray]:
    """Optimize appearance and ellipsoid covariance without moving centroids."""
    fixed_means = torch.from_numpy(means).float().cuda()
    quats = torch.from_numpy(initial_quats).float().cuda().requires_grad_(True)
    scales = torch.from_numpy(np.log(initial_scales)).float().cuda().requires_grad_(True)
    opacities = inverse_sigmoid(initial_opacities).requires_grad_(True)
    colors = inverse_sigmoid(initial_colors).requires_grad_(True)
    params = {
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
    }
    optimizers = {
        "quats": torch.optim.Adam([quats], lr=0.01),
        "scales": torch.optim.Adam([scales], lr=0.01),
        "opacities": torch.optim.Adam([opacities], lr=0.001),
        "colors": torch.optim.Adam([colors], lr=0.01),
    }
    groundtruth = SimpleBodyBuilder._get_rasterization_groundtruth(
        datapoints, max_depth=maximum_depth_m
    )
    backgrounds = torch.rand((iterations, 3), dtype=torch.float32).cuda()
    image_count = groundtruth.images.shape[0]
    min_log_scale = float(np.log(minimum_scale_m))
    max_log_scale = float(np.log(maximum_scale_m))
    losses: list[float] = []

    for iteration in tqdm(range(iterations), desc="centroid Gaussian optimization"):
        background = backgrounds[iteration]
        groundtruth.images[groundtruth.masks == 0, :] = background
        render_colors, _, _ = rasterization(
            means=fixed_means,
            quats=GaussianActivations.quat(quats),
            scales=GaussianActivations.scale(scales),
            colors=GaussianActivations.color(colors),
            opacities=GaussianActivations.opacity(opacities),
            viewmats=groundtruth.X_CWs,
            Ks=groundtruth.Ks,
            width=groundtruth.width,
            height=groundtruth.height,
            camera_model="pinhole",
            render_mode="RGB+D",
            packed=False,
            backgrounds=background.reshape(1, 3).repeat(image_count, 1),
        )
        pixel_loss = torch.nn.functional.mse_loss(
            render_colors[..., :3], groundtruth.images, reduction="none"
        )
        loss = pixel_loss[groundtruth.masks != 2].mean()
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
        scales.detach().clamp_(min_log_scale, max_log_scale)
        losses.append(float(loss.detach().cpu()))

    return {
        "quats": GaussianActivations.quat(quats).detach().cpu().numpy(),
        "scales": GaussianActivations.scale(scales).detach().cpu().numpy(),
        "opacities": GaussianActivations.opacity(opacities).detach().cpu().numpy(),
        "colors": GaussianActivations.color(colors).detach().cpu().numpy(),
        "losses": np.asarray(losses, dtype=np.float32),
    }


def face_source_classes(vertex_classes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    has_left = np.any(np.isin(vertex_classes, (1, 3)), axis=1)
    has_right = np.any(np.isin(vertex_classes, (2, 3)), axis=1)
    all_inferred = np.all(vertex_classes == 4, axis=1)
    result = np.full(len(vertex_classes), 3, dtype=np.uint8)
    result[has_left & ~has_right] = 1
    result[has_right & ~has_left] = 2
    result[all_inferred] = 4
    return result, has_left, has_right


def main() -> None:
    args = parse_args()
    for name in (
        "physics_asset",
        "source_gaussian_asset",
        "fine_surface",
        "fine_surface_report",
        "stage_b_report",
        "calibration",
        "cameras",
        "rgb_dir",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.reuse_appearance_asset is not None:
        args.reuse_appearance_asset = args.reuse_appearance_asset.resolve()
    if args.reuse_appearance_asset is None and not torch.cuda.is_available():
        raise RuntimeError("Centroid Gaussian optimization requires CUDA")
    if args.iterations <= 0 or args.max_gaussians <= 0 or args.asset_version < 1:
        raise ValueError("Iterations, asset version and Gaussian budget must be positive")
    if not 0.0 < args.training_resolution_scale <= 1.0:
        raise ValueError("--training-resolution-scale must lie in (0, 1]")
    if not 0.0 < args.minimum_scale_mm <= args.maximum_scale_mm:
        raise ValueError("Gaussian scale bounds are invalid")

    output_asset = args.output_dir / "tissue_paper_pbd_centroid_gaussians.npz"
    output_report = args.output_dir / "metadata.json"
    output_preview = args.output_dir / "tissue_centroid_gaussian_coverage.ply"
    existing = [path for path in (output_asset, output_report, output_preview) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Outputs exist; pass --overwrite: " + ", ".join(map(str, existing)))

    with np.load(args.physics_asset, allow_pickle=False) as loaded:
        base = {name: loaded[name].copy() for name in loaded.files}
    with np.load(args.source_gaussian_asset, allow_pickle=False) as loaded:
        source_gaussians = {name: loaded[name].copy() for name in loaded.files}
    with np.load(args.fine_surface, allow_pickle=False) as loaded:
        fine = {name: loaded[name].copy() for name in loaded.files}
    fine_report = read_json(args.fine_surface_report)
    relevant_surface_gates = {
        name: passed
        for name, passed in fine_report["gates"].items()
        if name not in {"surface_triangle_budget", "surface_vertex_budget"}
    }
    if not all(relevant_surface_gates.values()):
        raise RuntimeError(f"Fine visual surface failed gates: {relevant_surface_gates}")

    visual_vertices = fine["surface_vertices_table"].astype(np.float32)
    visual_faces = fine["surface_faces"].astype(np.int32)
    visual_vertex_source_class = fine["surface_vertex_source_class"].astype(np.uint8)
    if len(visual_faces) > args.max_gaussians:
        raise RuntimeError(
            f"Fine surface has {len(visual_faces)} faces, exceeding --max-gaussians={args.max_gaussians}"
        )
    visual_face_points = visual_vertices[visual_faces]
    means = visual_face_points.mean(axis=1).astype(np.float32)
    centroid_weights = np.full((len(means), 3), 1.0 / 3.0, dtype=np.float32)
    rest_visual_frames = face_frames(visual_face_points)
    face_vertex_classes = visual_vertex_source_class[visual_faces]
    gaussian_source_class, has_left_source, has_right_source = face_source_classes(
        face_vertex_classes
    )

    source_means = source_gaussians["gaussian_rest_means_table"].astype(np.float64)
    nearest_source_distance, nearest_source_ids = cKDTree(source_means).query(
        means.astype(np.float64), k=1, workers=-1
    )
    initial_quats = source_gaussians["gaussian_rest_quats_table_wxyz"][
        nearest_source_ids
    ].astype(np.float32)
    initial_scales = source_gaussians["gaussian_scales"][nearest_source_ids].astype(
        np.float32
    )
    initial_opacities = source_gaussians["gaussian_opacities"][nearest_source_ids].astype(
        np.float32
    )
    initial_colors = source_gaussians["gaussian_colors_rgb"][nearest_source_ids].astype(
        np.float32
    )

    stage_b = read_json(args.stage_b_report)
    calibration = read_json(args.calibration)
    cameras = read_json(args.cameras)
    datapoints, observations = make_observations(
        stage_b=stage_b,
        calibration=calibration,
        cameras=cameras,
        rgb_dir=args.rgb_dir,
        resolution_scale=args.training_resolution_scale,
        confidence_policy=args.observation_confidence_policy,
    )
    appearance_reused = args.reuse_appearance_asset is not None
    if appearance_reused:
        with np.load(args.reuse_appearance_asset, allow_pickle=False) as loaded:
            appearance = {name: loaded[name].copy() for name in loaded.files}
        required_appearance = {
            "gaussian_rest_means_table",
            "gaussian_rest_quats_table_wxyz",
            "gaussian_scales",
            "gaussian_opacities",
            "gaussian_colors_rgb",
            "visual_surface_rest_vertices_table",
            "visual_surface_faces",
        }
        missing_appearance = sorted(required_appearance - set(appearance))
        if missing_appearance:
            raise RuntimeError(
                f"Appearance asset is missing arrays: {missing_appearance}"
            )
        if (
            not np.array_equal(
                appearance["visual_surface_rest_vertices_table"], visual_vertices
            )
            or not np.array_equal(appearance["visual_surface_faces"], visual_faces)
            or np.max(
                np.linalg.norm(
                    appearance["gaussian_rest_means_table"].astype(np.float32)
                    - means,
                    axis=1,
                )
            )
            > 1.0e-8
        ):
            raise RuntimeError(
                "Reuse appearance asset does not exactly match the visual surface"
            )
        quats = appearance["gaussian_rest_quats_table_wxyz"].astype(np.float32)
        scales = appearance["gaussian_scales"].astype(np.float32)
        opacities = appearance["gaussian_opacities"].astype(np.float32)
        colors = appearance["gaussian_colors_rgb"].astype(np.float32)
        optimization_losses = np.empty(0, dtype=np.float32)
    else:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        optimized = optimize_fixed_centroid_appearance(
            means=means,
            initial_quats=initial_quats,
            initial_scales=initial_scales,
            initial_opacities=initial_opacities,
            initial_colors=initial_colors,
            datapoints=datapoints,
            iterations=args.iterations,
            minimum_scale_m=args.minimum_scale_mm / 1000.0,
            maximum_scale_m=args.maximum_scale_mm / 1000.0,
            maximum_depth_m=args.maximum_depth_m,
        )
        quats = optimized["quats"].astype(np.float32)
        scales = optimized["scales"].astype(np.float32)
        opacities = optimized["opacities"].astype(np.float32)
        colors = optimized["colors"].astype(np.float32)
        optimization_losses = optimized["losses"]
    if not all(
        np.isfinite(value).all() for value in (means, quats, scales, opacities, colors)
    ):
        raise RuntimeError("Centroid Gaussian optimization produced non-finite values")

    positions = base["rest_positions_table"].astype(np.float64)
    tets = base["tet_indices"].astype(np.int32)
    physical_faces = base["collision_skin_faces"].astype(np.int32)
    vertex_binding = bind_to_surface_faces(
        visual_vertices.astype(np.float64), positions, tets, physical_faces
    )
    visual_vertex_offsets = (
        visual_vertices - vertex_binding["means"]
    ).astype(np.float32)
    reconstructed_visual_vertices = (
        np.einsum(
            "vi,vij->vj",
            vertex_binding["face_weights"],
            positions[vertex_binding["face_particles"]],
        )
        + visual_vertex_offsets
    )
    if np.max(np.linalg.norm(reconstructed_visual_vertices - visual_vertices, axis=1)) > 2.0e-8:
        raise RuntimeError("Visual vertex physical embedding failed to reconstruct")

    force_binding = bind_to_surface_faces(means, positions, tets, physical_faces)
    visibility = np.stack(
        [
            project_visibility(
                means,
                observation,
                args.visibility_depth_tolerance_mm / 1000.0,
            )
            for observation in observations
        ],
        axis=0,
    )
    observation_count = visibility.sum(axis=0).astype(np.uint8)
    first_pair_observed = visibility[:2].any(axis=0)
    later_only_observed = ~first_pair_observed & visibility[2:].any(axis=0)
    observed_any = observation_count > 0

    global_rotations = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
    local_rotations = np.transpose(rest_visual_frames, (0, 2, 1)) @ global_rotations
    local_quats_xyzw = Rotation.from_matrix(local_rotations).as_quat().astype(np.float32)
    local_quats_wxyz = local_quats_xyzw[:, [3, 0, 1, 2]]

    output = {
        name: value
        for name, value in base.items()
        if not name.startswith("gaussian_")
        and not name.startswith("visual_surface_")
        and not name.startswith("visual_vertex_")
    }
    output.update(
        {
            "gaussian_rest_means_table": means,
            "gaussian_rest_quats_table_wxyz": quats,
            "gaussian_rest_local_quats_wxyz": local_quats_wxyz,
            "gaussian_scales": scales,
            "gaussian_opacities": opacities,
            "gaussian_colors_rgb": colors,
            "gaussian_tet_ids": force_binding["tet_ids"],
            "gaussian_particle_indices": force_binding["tet_particles"],
            "gaussian_barycentric_weights": force_binding["tet_weights"],
            "gaussian_binding_distance": force_binding["projection_distance"],
            "gaussian_rest_offset_table": np.zeros_like(means),
            "gaussian_binding_mode": np.asarray("visual_surface_face_centroid"),
            "gaussian_face_ids": force_binding["face_ids"],
            "gaussian_face_particle_indices": force_binding["face_particles"],
            "gaussian_face_barycentric_weights": force_binding["face_weights"],
            "gaussian_visual_face_ids": np.arange(len(visual_faces), dtype=np.int32),
            "gaussian_visual_face_barycentric_weights": centroid_weights,
            "gaussian_visual_face_vertex_source_class": face_vertex_classes,
            "gaussian_surface_source_class": gaussian_source_class,
            "gaussian_face_has_left_source": has_left_source,
            "gaussian_face_has_right_source": has_right_source,
            "gaussian_observation_count": observation_count,
            "gaussian_first_pair_observed": first_pair_observed,
            "gaussian_later_only_observed": later_only_observed,
            "gaussian_nearest_source_ids": nearest_source_ids.astype(np.int32),
            "gaussian_nearest_source_distance": nearest_source_distance.astype(np.float32),
            "visual_surface_rest_vertices_table": visual_vertices,
            "visual_surface_faces": visual_faces,
            "visual_surface_vertex_source_class": visual_vertex_source_class,
            "visual_surface_face_density_zone": fine[
                "surface_face_density_zone"
            ].astype(np.uint8),
            "visual_surface_vertex_dense_source_ids": fine[
                "surface_vertex_dense_source_ids"
            ].astype(np.int32),
            "visual_vertex_particle_indices": vertex_binding["face_particles"],
            "visual_vertex_barycentric_weights": vertex_binding["face_weights"],
            "visual_vertex_rest_offset_table": visual_vertex_offsets,
            "visual_vertex_physical_face_ids": vertex_binding["face_ids"],
            "visual_vertex_physical_projection_distance": vertex_binding[
                "projection_distance"
            ],
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_asset, **output)
    write_binding_preview(
        output_preview,
        means,
        first_pair_observed,
        later_only_observed,
        observed_any,
    )

    centroid_reconstruction = np.einsum(
        "ni,nij->nj", centroid_weights, visual_vertices[visual_faces]
    )
    initial_sorted = np.sort(initial_scales, axis=1)
    optimized_sorted = np.sort(scales, axis=1)
    physics_array_names = [
        name
        for name in base
        if not name.startswith("gaussian_")
        and not name.startswith("visual_surface_")
        and not name.startswith("visual_vertex_")
    ]
    gates = {
        "fine_surface_visual_gates_passed": bool(all(relevant_surface_gates.values())),
        "physics_arrays_preserved": bool(
            all(
                np.array_equal(output[name], base[name], equal_nan=True)
                for name in physics_array_names
            )
        ),
        "one_gaussian_per_visual_face": len(means) == len(visual_faces),
        "all_visual_vertices_and_source_classes_retained": bool(
            np.array_equal(visual_vertices, fine["surface_vertices_table"])
            and np.array_equal(visual_faces, fine["surface_faces"])
            and np.array_equal(
                visual_vertex_source_class, fine["surface_vertex_source_class"]
            )
        ),
        "gaussian_centers_are_exact_face_centroids": bool(
            np.max(np.linalg.norm(means - centroid_reconstruction, axis=1)) <= 1.0e-8
            and np.array_equal(centroid_weights, np.full_like(centroid_weights, 1.0 / 3.0))
        ),
        "visual_vertices_reconstruct_from_physical_embedding": bool(
            np.max(
                np.linalg.norm(reconstructed_visual_vertices - visual_vertices, axis=1)
            )
            <= 2.0e-8
        ),
        "ellipsoid_axes_not_sorted_or_forced_to_face_normal": bool(
            np.mean((scales[:, 0] >= scales[:, 1]) & (scales[:, 1] >= scales[:, 2]))
            < 0.95
        ),
        "ellipsoid_anisotropy_retained": bool(
            np.percentile(optimized_sorted[:, 2] / optimized_sorted[:, 0], 95) > 1.5
        ),
        "scale_bounds_respected": bool(
            scales.min() >= args.minimum_scale_mm / 1000.0 - 1.0e-7
            and scales.max() <= args.maximum_scale_mm / 1000.0 + 1.0e-7
        ),
        "left_and_right_union_support_retained": bool(
            np.any(has_left_source & ~has_right_source)
            and np.any(has_right_source & ~has_left_source)
            and np.any(has_left_source & has_right_source)
        ),
        "later_frames_add_first_pair_occlusion_coverage": bool(later_only_observed.any()),
        "all_outputs_finite": bool(
            all(np.isfinite(value).all() for value in (means, quats, scales, opacities, colors))
        ),
    }
    report = {
        "asset_version": args.asset_version,
        "stage": "fine_visual_face_centroid_ellipsoids_on_frozen_tetrahedral_pbd",
        "method": {
            "gaussian_center_formula": "mu_f = (v1 + v2 + v3) / 3; barycentric weights fixed to (1/3, 1/3, 1/3)",
            "ellipsoid_transport_formula": "R(t) = B_visual(t) B_visual(0)^T R_rest; scales remain optimized ellipsoid axes",
            "visual_vertex_embedding_formula": "v(t) = sum_i b_i x_i(t) + R_physical_face(t) R_physical_face(0)^T d_rest",
            "appearance_optimization": (
                "reused bit-identical optimized opacity, RGB, quaternion, and scales; only physics bindings rebuilt"
                if appearance_reused
                else "fixed centroids; jointly optimize opacity, RGB, quaternion, and three scales over six stereo pairs"
            ),
            "visual_force_mapping": "existing nearest physical boundary-face three-node translational support; exact visual-offset Jacobian deferred",
            "physics_geometry_changed": False,
            "physics_topology_fixed_but_vertices_deformable": True,
            "stiffness_changed": False,
            "depth_residual_changed": False,
        },
        "parameters": {
            "iterations": args.iterations,
            "training_resolution_scale": args.training_resolution_scale,
            "minimum_scale_mm": args.minimum_scale_mm,
            "maximum_scale_mm": args.maximum_scale_mm,
            "observation_confidence_policy": args.observation_confidence_policy,
            "visibility_depth_tolerance_mm": args.visibility_depth_tolerance_mm,
            "seed": args.seed,
            "appearance_reused": appearance_reused,
        },
        "counts": {
            "particles": int(len(base["rest_positions_table"])),
            "tetrahedra": int(len(base["tet_indices"])),
            "collision_triangles": int(len(physical_faces)),
            "visual_vertices": int(len(visual_vertices)),
            "visual_triangles": int(len(visual_faces)),
            "visual_fine_triangles": int(
                np.count_nonzero(fine["surface_face_density_zone"] == 0)
            ),
            "visual_transition_triangles": int(
                np.count_nonzero(fine["surface_face_density_zone"] == 1)
            ),
            "visual_outer_triangles": int(
                np.count_nonzero(fine["surface_face_density_zone"] == 2)
            ),
            "gaussians": int(len(means)),
            "left_only_visual_vertices": int(np.count_nonzero(visual_vertex_source_class == 1)),
            "right_only_visual_vertices": int(np.count_nonzero(visual_vertex_source_class == 2)),
            "dual_visual_vertices": int(np.count_nonzero(visual_vertex_source_class == 3)),
            "inferred_visual_vertices": int(np.count_nonzero(visual_vertex_source_class == 4)),
            "left_only_source_faces": int(np.count_nonzero(has_left_source & ~has_right_source)),
            "right_only_source_faces": int(np.count_nonzero(has_right_source & ~has_left_source)),
            "bilateral_source_faces": int(np.count_nonzero(has_left_source & has_right_source)),
            "first_stereo_pair_observed": int(first_pair_observed.sum()),
            "later_only_observed": int(later_only_observed.sum()),
            "observed_in_any_selected_view": int(observed_any.sum()),
        },
        "statistics": {
            "initial_scale_mm_percentiles_xyz": np.percentile(
                initial_scales * 1000.0, [0, 5, 50, 95, 100], axis=0
            ).tolist(),
            "optimized_scale_mm_percentiles_xyz": np.percentile(
                scales * 1000.0, [0, 5, 50, 95, 100], axis=0
            ).tolist(),
            "initial_anisotropy_max_over_min_percentiles": five_number(
                initial_sorted[:, 2] / initial_sorted[:, 0]
            ),
            "optimized_anisotropy_max_over_min_percentiles": five_number(
                optimized_sorted[:, 2] / optimized_sorted[:, 0]
            ),
            "visual_vertex_physical_offset_mm_percentiles": five_number(
                np.linalg.norm(visual_vertex_offsets, axis=1), 1000.0
            ),
            "gaussian_centroid_to_physical_surface_mm_percentiles": five_number(
                force_binding["projection_distance"], 1000.0
            ),
            "nearest_source_gaussian_mm_percentiles": five_number(
                nearest_source_distance, 1000.0
            ),
            "centroid_float32_reconstruction_error_nm_max": float(
                np.max(np.linalg.norm(means - centroid_reconstruction, axis=1))
                * 1.0e9
            ),
            "training_loss_first_last": (
                None
                if not len(optimization_losses)
                else [
                    float(optimization_losses[0]),
                    float(optimization_losses[-1]),
                ]
            ),
            "rgb_mean": colors.mean(axis=0).tolist(),
            "opacity_percentiles": five_number(opacities),
        },
        "observations": [
            {
                key: value
                for key, value in observation.items()
                if key not in {"K", "X_WC", "depth", "training_mask"}
            }
            for observation in observations
        ],
        "inputs": {
            "physics_asset": str(args.physics_asset.relative_to(REPO_ROOT)),
            "physics_asset_sha256": sha256(args.physics_asset),
            "source_gaussian_asset": str(
                args.source_gaussian_asset.relative_to(REPO_ROOT)
            ),
            "source_gaussian_asset_sha256": sha256(args.source_gaussian_asset),
            "reuse_appearance_asset": (
                None
                if args.reuse_appearance_asset is None
                else str(args.reuse_appearance_asset.relative_to(REPO_ROOT))
            ),
            "reuse_appearance_asset_sha256": (
                None
                if args.reuse_appearance_asset is None
                else sha256(args.reuse_appearance_asset)
            ),
            "fine_surface": str(args.fine_surface.relative_to(REPO_ROOT)),
            "fine_surface_sha256": sha256(args.fine_surface),
            "fine_surface_report": str(args.fine_surface_report.relative_to(REPO_ROOT)),
        },
        "outputs": {
            "asset": str(output_asset.relative_to(REPO_ROOT)),
            "preview": str(output_preview.relative_to(REPO_ROOT)),
        },
        "gates": gates,
        "passed": bool(all(gates.values())),
    }
    output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        failed = [name for name, passed in gates.items() if not passed]
        raise SystemExit(f"Centroid Gaussian asset gates failed: {failed}")


if __name__ == "__main__":
    main()
