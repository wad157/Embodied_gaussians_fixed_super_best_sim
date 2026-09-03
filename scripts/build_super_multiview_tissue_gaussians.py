#!/usr/bin/env python3
"""Build dense, small, multiview-trained tissue Gaussians on frozen PBD geometry.

The optimization intentionally reuses the original SUPER tissue Gaussian path:
surface point initialization followed by ``SimpleBodyBuilder._grow_gaussians``
with the same five optimized parameter groups and learning rates.  The extension
is limited to denser multiview surface initialization, multiple static stereo
observations, smaller scale bounds, and correct mask=2 occlusion handling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_gaussians.scene_builders.domain import (  # noqa: E402
    GaussianLearningRates,
    MaskedPosedImageAndDepth,
)
from embodied_gaussians.scene_builders.simple_body_builder import (  # noqa: E402
    SimpleBodyBuilder,
)
NATIVE_ROOT = REPO_ROOT / "data/super/grasp5_native"
MULTIVIEW_ROOT = NATIVE_ROOT / "tissue_multiview_v1"
DEFAULT_BASE_ASSET = (
    MULTIVIEW_ROOT / "paper_pbd_tissue_v2/tissue_paper_pbd.npz"
)
DEFAULT_FINE_SURFACE = (
    MULTIVIEW_ROOT / "rest_surface_gaussian_view_union_v3/rest_surface.npz"
)
DEFAULT_FINE_SURFACE_REPORT = (
    MULTIVIEW_ROOT / "rest_surface_gaussian_view_union_v3/report.json"
)
DEFAULT_OUTPUT_DIR = (
    MULTIVIEW_ROOT / "paper_pbd_tissue_v5_gaustar_face_union"
)
OPENCV_TO_BLENDER = np.diag([1.0, -1.0, -1.0, 1.0])
GAUSSIAN_KEYS = {
    "gaussian_rest_means_table",
    "gaussian_rest_quats_table_wxyz",
    "gaussian_scales",
    "gaussian_opacities",
    "gaussian_colors_rgb",
    "gaussian_tet_ids",
    "gaussian_particle_indices",
    "gaussian_barycentric_weights",
    "gaussian_binding_distance",
    "gaussian_rest_offset_table",
    "gaussian_observation_count",
    "gaussian_first_pair_observed",
    "gaussian_later_only_observed",
    "gaussian_surface_source_class",
    "gaussian_surface_distance",
    "gaussian_optimization_displacement",
    "gaussian_raw_optimization_displacement",
    "gaussian_restored_to_initial_surface",
    "gaussian_binding_mode",
    "gaussian_face_ids",
    "gaussian_face_particle_indices",
    "gaussian_face_barycentric_weights",
    "gaussian_face_projection_distance",
    "gaussian_preprojection_surface_distance",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse the original Gaussian optimization with a dense unsmoothed "
            "multiview tissue surface and all six selected stereo pairs."
        )
    )
    parser.add_argument("--base-asset", type=Path, default=DEFAULT_BASE_ASSET)
    parser.add_argument("--fine-surface", type=Path, default=DEFAULT_FINE_SURFACE)
    parser.add_argument(
        "--fine-surface-report", type=Path, default=DEFAULT_FINE_SURFACE_REPORT
    )
    parser.add_argument(
        "--stage-b-report",
        type=Path,
        default=MULTIVIEW_ROOT / "stage_b_report.json",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=NATIVE_ROOT / "calib_rectified.json",
    )
    parser.add_argument(
        "--cameras",
        type=Path,
        default=REPO_ROOT / "data/super/grasp5_offline_demo/cameras.json",
    )
    parser.add_argument("--rgb-dir", type=Path, default=NATIVE_ROOT / "rgb")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-gaussians", type=int, default=30000)
    parser.add_argument("--initial-radius-mm", type=float, default=0.35)
    parser.add_argument("--minimum-scale-factor", type=float, default=0.5)
    parser.add_argument("--maximum-scale-factor", type=float, default=2.0)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--training-resolution-scale", type=float, default=1.0)
    parser.add_argument("--maximum-depth-m", type=float, default=0.25)
    parser.add_argument("--visibility-depth-tolerance-mm", type=float, default=2.0)
    parser.add_argument(
        "--observation-confidence-policy",
        choices=("acceptable", "dense_union"),
        default="dense_union",
        help=(
            "dense_union retains per-view finite FoundationStereo depths even "
            "when stereo/RAFT confidence disagrees; confidence is audit-only."
        ),
    )
    parser.add_argument(
        "--maximum-surface-distance-mm",
        type=float,
        default=None,
        help=(
            "Optional emergency outlier gate. By default no Gaussian is removed "
            "according to its distance from the mechanics skin because stereo "
            "depth is uncertain; the distance is stored as audit metadata instead."
        ),
    )
    parser.add_argument("--binding-candidates", type=int, default=128)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def five_number(values: np.ndarray, scale: float = 1.0) -> list[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)] * scale
    if not len(finite):
        return []
    return np.percentile(finite, [0, 5, 50, 95, 100]).tolist()


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def selected_pairs(stage_b: dict) -> list[dict[str, int]]:
    return [
        {
            "left": int(row["left_frame"]),
            "right": int(row["right_frame"]),
        }
        for row in stage_b["coverage"]["selected_temporal_order"]
    ]


def resize_inputs(
    image_rgb: np.ndarray,
    depth: np.ndarray,
    tissue: np.ndarray,
    visible: np.ndarray,
    tool: np.ndarray,
    acceptable: np.ndarray,
    intrinsics: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if scale == 1.0:
        return image_rgb, depth, tissue, visible, tool, acceptable, intrinsics
    width = int(round(image_rgb.shape[1] * scale))
    height = int(round(image_rgb.shape[0] * scale))
    if width <= 0 or height <= 0:
        raise ValueError("--training-resolution-scale produced an empty image")
    image_rgb = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)
    depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    tissue = cv2.resize(
        tissue.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    visible = cv2.resize(
        visible.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    tool = cv2.resize(
        tool.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    acceptable = cv2.resize(
        acceptable.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    intrinsics = intrinsics.copy()
    intrinsics[0] *= scale
    intrinsics[1] *= scale
    return image_rgb, depth, tissue, visible, tool, acceptable, intrinsics


def make_observations(
    *,
    stage_b: dict,
    calibration: dict,
    cameras: dict,
    rgb_dir: Path,
    resolution_scale: float,
    confidence_policy: str,
) -> tuple[list[MaskedPosedImageAndDepth], list[dict]]:
    datapoints: list[MaskedPosedImageAndDepth] = []
    observations: list[dict] = []
    for pair_index, pair in enumerate(selected_pairs(stage_b)):
        for side in ("left", "right"):
            frame = pair[side]
            image_path = rgb_dir / f"{frame:06d}-{side}.png"
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(image_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            depth_path = MULTIVIEW_ROOT / f"depth_{side}/{frame:06d}-depth.npy"
            confidence_path = (
                MULTIVIEW_ROOT / f"depth_{side}/{frame:06d}-confidence.npz"
            )
            tissue_path = (
                MULTIVIEW_ROOT / f"masks_{side}/tissue/{frame:06d}-tissue.png"
            )
            visibility_path = (
                MULTIVIEW_ROOT
                / f"masks_{side}/stage_b_visibility_proxy/"
                f"{frame:06d}-visibility-proxy.png"
            )
            tool_path = (
                MULTIVIEW_ROOT
                / f"masks_{side}/tool_dilated/{frame:06d}-tool-dilated.png"
            )
            depth = np.load(depth_path).astype(np.float32)
            with np.load(confidence_path, allow_pickle=False) as confidence:
                acceptable = confidence["stage_c_acceptable"].astype(bool)
                dense_valid = confidence["foundation_dense_valid"].astype(bool)
            selected_confidence = (
                dense_valid if confidence_policy == "dense_union" else acceptable
            )
            tissue = load_gray(tissue_path) > 0
            visible = load_gray(visibility_path) > 0
            tool = load_gray(tool_path) > 0
            dense_union_only_pixels = int(
                np.count_nonzero(dense_valid & ~acceptable & visible)
            )
            intrinsics = np.asarray(
                calibration[f"K_{side}_rect"], dtype=np.float32
            )
            (
                image_rgb,
                depth,
                tissue,
                visible,
                tool,
                selected_confidence,
                intrinsics,
            ) = resize_inputs(
                image_rgb,
                depth,
                tissue,
                visible,
                tool,
                selected_confidence,
                intrinsics,
                resolution_scale,
            )
            visible = (
                visible
                & selected_confidence
                & np.isfinite(depth)
                & (depth > 0.0)
            )
            # 0: true background; 1: visible tissue supervision;
            # 2: tissue/tool/highlight/depth occlusion ignored by RGB loss.
            training_mask = np.zeros(tissue.shape, dtype=np.uint8)
            training_mask[tissue | tool] = 2
            training_mask[visible] = 1
            X_WC = np.asarray(cameras[f"stereo_{side}"]["X_WC"], dtype=np.float32)
            datapoints.append(
                MaskedPosedImageAndDepth(
                    mask=training_mask,
                    X_WC=X_WC,
                    K=intrinsics,
                    image=image_rgb,
                    format="rgb",
                    depth=depth,
                    depth_scale=1.0,
                )
            )
            observations.append(
                {
                    "pair_index": pair_index,
                    "side": side,
                    "frame": frame,
                    "image": str(image_path),
                    "K": intrinsics,
                    "X_WC": X_WC,
                    "depth": depth,
                    "training_mask": training_mask,
                    "visible_pixels": int(np.count_nonzero(training_mask == 1)),
                    "ignored_pixels": int(np.count_nonzero(training_mask == 2)),
                    "background_pixels": int(np.count_nonzero(training_mask == 0)),
                    "confidence_policy": confidence_policy,
                    "dense_union_only_pixels": int(
                        dense_union_only_pixels
                    ),
                }
            )
    return datapoints, observations


def surface_scene(vertices: np.ndarray, faces: np.ndarray) -> o3d.t.geometry.RaycastingScene:
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices.astype(np.float64)),
        o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def bind_to_surface_faces(
    means: np.ndarray,
    positions: np.ndarray,
    tets: np.ndarray,
    surface_faces: np.ndarray,
) -> dict[str, np.ndarray]:
    """Project Gaussians onto boundary faces and embed 3-node weights in tets.

    GauSTAR places Gaussian centers at predefined barycentric coordinates on a
    triangular face.  Each boundary face here belongs to exactly one physical
    tetrahedron, so a zero weight on the fourth corner preserves compatibility
    with the existing tetrahedral skinning and visual-force kernels.
    """
    scene = surface_scene(positions, surface_faces)
    closest = scene.compute_closest_points(
        o3d.core.Tensor(means.astype(np.float32))
    )
    projected = closest["points"].numpy().astype(np.float32)
    face_ids = closest["primitive_ids"].numpy().astype(np.int32)
    primitive_uvs = closest["primitive_uvs"].numpy().astype(np.float64)
    face_weights = np.column_stack(
        (
            1.0 - primitive_uvs[:, 0] - primitive_uvs[:, 1],
            primitive_uvs[:, 0],
            primitive_uvs[:, 1],
        )
    )
    face_weights = np.clip(face_weights, 0.0, 1.0)
    face_weights /= np.maximum(face_weights.sum(axis=1, keepdims=True), 1.0e-12)
    face_particles = surface_faces[face_ids]

    face_to_tet: dict[tuple[int, int, int], int] = {}
    for tet_id, tet in enumerate(tets.tolist()):
        a, b, c, d = tet
        for face in ((b, c, d), (a, d, c), (a, b, d), (a, c, b)):
            key = tuple(sorted(face))
            if key in face_to_tet:
                face_to_tet[key] = -1
            else:
                face_to_tet[key] = tet_id
    surface_tet_ids = np.asarray(
        [face_to_tet[tuple(sorted(face))] for face in surface_faces.tolist()],
        dtype=np.int32,
    )
    if np.any(surface_tet_ids < 0):
        raise RuntimeError("A collision-skin face is not a unique tet boundary")
    tet_ids = surface_tet_ids[face_ids]
    tet_particles = tets[tet_ids]
    tet_weights = np.zeros((len(means), 4), dtype=np.float64)
    rows = np.arange(len(means))
    for face_corner in range(3):
        matches = tet_particles == face_particles[:, face_corner, None]
        if not np.all(matches.sum(axis=1) == 1):
            raise RuntimeError("Surface face vertices do not match adjacent tetrahedra")
        tet_corner = np.argmax(matches, axis=1)
        tet_weights[rows, tet_corner] = face_weights[:, face_corner]

    reconstructed = np.einsum(
        "ni,nij->nj", tet_weights, positions[tet_particles]
    )
    if np.max(np.linalg.norm(reconstructed - projected, axis=1)) > 2.0e-7:
        raise RuntimeError("Surface-face barycentric embedding failed")
    projection_distance = np.linalg.norm(means - projected, axis=1).astype(
        np.float32
    )

    face_xyz = positions[face_particles].astype(np.float64)
    tangent_x = face_xyz[:, 1] - face_xyz[:, 0]
    tangent_x /= np.maximum(
        np.linalg.norm(tangent_x, axis=1, keepdims=True), 1.0e-12
    )
    normal = np.cross(
        face_xyz[:, 1] - face_xyz[:, 0],
        face_xyz[:, 2] - face_xyz[:, 0],
    )
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1.0e-12)
    tangent_y = np.cross(normal, tangent_x)
    tangent_y /= np.maximum(
        np.linalg.norm(tangent_y, axis=1, keepdims=True), 1.0e-12
    )
    frames = np.stack((tangent_x, tangent_y, normal), axis=-1)
    quats_xyzw = Rotation.from_matrix(frames).as_quat().astype(np.float32)
    quats_wxyz = quats_xyzw[:, [3, 0, 1, 2]]

    return {
        "means": projected,
        "quats_wxyz": quats_wxyz,
        "face_ids": face_ids,
        "face_particles": face_particles.astype(np.int32),
        "face_weights": face_weights.astype(np.float32),
        "tet_ids": tet_ids,
        "tet_particles": tet_particles.astype(np.int32),
        "tet_weights": tet_weights.astype(np.float32),
        "projection_distance": projection_distance,
    }


def project_visibility(
    points_table: np.ndarray,
    observation: dict,
    depth_tolerance_m: float,
) -> np.ndarray:
    X_WC_opencv = observation["X_WC"].astype(np.float64) @ OPENCV_TO_BLENDER
    X_CW = np.linalg.inv(X_WC_opencv)
    points_camera = points_table @ X_CW[:3, :3].T + X_CW[:3, 3]
    z = points_camera[:, 2]
    K = observation["K"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.rint(points_camera[:, 0] * K[0, 0] / z + K[0, 2]).astype(
            np.int32
        )
        v = np.rint(points_camera[:, 1] * K[1, 1] / z + K[1, 2]).astype(
            np.int32
        )
    height, width = observation["training_mask"].shape
    inside = (
        np.isfinite(points_camera).all(axis=1)
        & (z > 0.0)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )
    ids = np.flatnonzero(inside)
    visible = np.zeros(len(points_table), dtype=bool)
    if not len(ids):
        return visible
    sampled_depth = observation["depth"][v[ids], u[ids]]
    depth_consistent = (
        np.isfinite(sampled_depth)
        & (sampled_depth > 0.0)
        & (np.abs(z[ids] - sampled_depth) <= depth_tolerance_m)
    )
    supervised = observation["training_mask"][v[ids], u[ids]] == 1
    visible[ids] = supervised & depth_consistent
    return visible


def write_binding_preview(
    path: Path,
    means: np.ndarray,
    first_pair: np.ndarray,
    later_only: np.ndarray,
    observed_any: np.ndarray,
) -> None:
    colors = np.tile(np.asarray([[0.45, 0.45, 0.45]]), (len(means), 1))
    colors[first_pair] = np.asarray([1.0, 0.45, 0.05])
    colors[later_only] = np.asarray([0.05, 0.85, 1.0])
    colors[~observed_any] = np.asarray([0.8, 0.1, 0.15])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(means.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False):
        raise RuntimeError(f"Failed to write {path}")


def main() -> None:
    args = parse_args()
    for name in (
        "base_asset",
        "fine_surface",
        "fine_surface_report",
        "stage_b_report",
        "calibration",
        "cameras",
        "rgb_dir",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if not torch.cuda.is_available():
        raise RuntimeError("Multiview Gaussian optimization requires CUDA")
    if args.max_gaussians <= 0 or args.iterations <= 0:
        raise ValueError("--max-gaussians and --iterations must be positive")
    if not 0.0 < args.training_resolution_scale <= 1.0:
        raise ValueError("--training-resolution-scale must lie in (0, 1]")
    if args.initial_radius_mm <= 0.0:
        raise ValueError("--initial-radius-mm must be positive")
    if (
        args.maximum_surface_distance_mm is not None
        and args.maximum_surface_distance_mm <= 0.0
    ):
        raise ValueError("--maximum-surface-distance-mm must be positive when set")

    output_asset = args.output_dir / "tissue_paper_pbd_dense_multiview.npz"
    output_report = args.output_dir / "metadata.json"
    output_preview = args.output_dir / "tissue_gaussian_multiview_coverage.ply"
    collisions = [p for p in (output_asset, output_report, output_preview) if p.exists()]
    if collisions and not args.overwrite:
        raise FileExistsError(
            "Outputs already exist; pass --overwrite:\n- "
            + "\n- ".join(str(path) for path in collisions)
        )
    for path in (
        args.base_asset,
        args.fine_surface,
        args.fine_surface_report,
        args.stage_b_report,
        args.calibration,
        args.cameras,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(args.base_asset, allow_pickle=False) as source:
        base = {name: source[name].copy() for name in source.files}
    with np.load(args.fine_surface, allow_pickle=False) as source:
        fine = {name: source[name].copy() for name in source.files}
    fine_report = read_json(args.fine_surface_report)
    relevant_surface_gates = {
        name: value
        for name, value in fine_report["gates"].items()
        if name not in {"surface_triangle_budget", "surface_vertex_budget"}
    }
    if not all(relevant_surface_gates.values()):
        raise RuntimeError(
            f"Fine multiview surface failed a visual-relevant gate: {relevant_surface_gates}"
        )

    surface_vertices = fine["surface_vertices_table"].astype(np.float32)
    surface_faces = fine["surface_faces"].astype(np.int32)
    surface_source_class = fine["surface_vertex_source_class"].astype(np.uint8)
    observed_surface = surface_source_class != 4
    # Keep the small topology-completion set as well.  It prevents uncovered
    # holes in the render surface and is explicitly marked source class 4.
    initial_points = surface_vertices.copy()
    initial_source_class = surface_source_class.copy()
    rng = np.random.default_rng(args.seed)
    if len(initial_points) > args.max_gaussians:
        keep = np.sort(
            rng.choice(len(initial_points), size=args.max_gaussians, replace=False)
        )
        initial_points = initial_points[keep]
        initial_source_class = initial_source_class[keep]

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

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    radius_m = args.initial_radius_mm / 1000.0
    learning_rates = GaussianLearningRates(
        means=0.0001,
        opacities=0.001,
        colors=0.01,
        quats=0.01,
        scales=0.01,
    )
    gaussians = SimpleBodyBuilder._grow_gaussians(
        initial_points=initial_points,
        radius=radius_m,
        num_iterations=args.iterations,
        learning_rates=learning_rates,
        datapoints=datapoints,
        min_scale=radius_m * args.minimum_scale_factor,
        max_scale=radius_m * args.maximum_scale_factor,
        max_depth=args.maximum_depth_m,
        visualize=False,
    )
    means = np.asarray(gaussians.means, dtype=np.float32)
    quats = np.asarray(gaussians.quats, dtype=np.float32)
    scales = np.asarray(gaussians.scales, dtype=np.float32)
    opacities = np.asarray(gaussians.opacities, dtype=np.float32)
    colors = np.asarray(gaussians.colors, dtype=np.float32)
    raw_optimization_displacement = np.linalg.norm(
        means - initial_points, axis=1
    ).astype(np.float32)

    finite = (
        np.isfinite(means).all(axis=1)
        & np.isfinite(quats).all(axis=1)
        & np.isfinite(scales).all(axis=1)
        & np.isfinite(opacities)
        & np.isfinite(colors).all(axis=1)
    )
    means = means[finite]
    quats = quats[finite]
    scales = scales[finite]
    opacities = opacities[finite]
    colors = colors[finite]
    initial_points = initial_points[finite]
    initial_source_class = initial_source_class[finite]
    raw_optimization_displacement = raw_optimization_displacement[finite]
    if not len(means):
        raise RuntimeError("All optimized Gaussians were non-finite")

    raw_visibility = np.stack(
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
    raw_observation_count = raw_visibility.sum(axis=0).astype(np.uint8)
    restored_to_initial = raw_observation_count == 0
    means[restored_to_initial] = initial_points[restored_to_initial]

    # Inferred topology-completion points have no direct color supervision.
    # Interpolate them from the closest directly reconstructed surface point.
    inferred = initial_source_class == 4
    if inferred.any():
        directly_reconstructed = ~inferred
        if not directly_reconstructed.any():
            raise RuntimeError("Fine surface has no directly reconstructed vertices")
        nearest_direct = cKDTree(initial_points[directly_reconstructed]).query(
            initial_points[inferred], k=1, workers=-1
        )[1]
        direct_ids = np.flatnonzero(directly_reconstructed)
        colors[inferred] = colors[direct_ids[nearest_direct]]

    mechanics_scene = surface_scene(
        base["rest_positions_table"], base["collision_skin_faces"]
    )
    surface_distance = mechanics_scene.compute_distance(
        o3d.core.Tensor(means)
    ).numpy().astype(np.float32)
    optimization_displacement = np.linalg.norm(
        means - initial_points, axis=1
    ).astype(np.float32)
    if args.maximum_surface_distance_mm is None:
        keep = np.ones(len(means), dtype=bool)
    else:
        keep = (
            surface_distance <= args.maximum_surface_distance_mm / 1000.0
        )
    means = means[keep]
    quats = quats[keep]
    scales = scales[keep]
    opacities = opacities[keep]
    colors = colors[keep]
    surface_distance = surface_distance[keep]
    optimization_displacement = optimization_displacement[keep]
    raw_optimization_displacement = raw_optimization_displacement[keep]
    restored_to_initial = restored_to_initial[keep]
    initial_points = initial_points[keep]
    initial_source_class = initial_source_class[keep]
    if not len(means):
        raise RuntimeError("All optimized Gaussians were non-finite or explicitly gated")

    preprojection_surface_distance = surface_distance.copy()
    face_binding = bind_to_surface_faces(
        means,
        base["rest_positions_table"].astype(np.float64),
        base["tet_indices"].astype(np.int32),
        base["collision_skin_faces"].astype(np.int32),
    )
    means = face_binding["means"]
    quats = face_binding["quats_wxyz"]
    # GauSTAR constrains the Gaussian z axis to the face normal and keeps it
    # thin.  Reorder the learned scales so z receives the smallest axis while
    # preserving all optimized scale magnitudes.
    sorted_scales = np.sort(scales, axis=1)
    scales = sorted_scales[:, [2, 1, 0]].astype(np.float32)
    surface_distance = mechanics_scene.compute_distance(
        o3d.core.Tensor(means)
    ).numpy().astype(np.float32)
    optimization_displacement = np.linalg.norm(
        means - initial_points, axis=1
    ).astype(np.float32)

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
    later_observed = visibility[2:].any(axis=0)
    later_only_observed = ~first_pair_observed & later_observed
    observed_any = observation_count > 0

    tet_ids = face_binding["tet_ids"]
    barycentric = face_binding["tet_weights"]
    binding_distance = face_binding["projection_distance"]
    binding_particles = face_binding["tet_particles"]
    contained = np.ones(len(means), dtype=bool)
    rest_offsets = np.zeros_like(means, dtype=np.float32)

    output = {
        name: value
        for name, value in base.items()
        if name not in GAUSSIAN_KEYS
    }
    output.update(
        {
            "gaussian_rest_means_table": means,
            "gaussian_rest_quats_table_wxyz": quats,
            "gaussian_scales": scales,
            "gaussian_opacities": opacities,
            "gaussian_colors_rgb": colors,
            "gaussian_tet_ids": tet_ids,
            "gaussian_particle_indices": binding_particles.astype(np.int32),
            "gaussian_barycentric_weights": barycentric.astype(np.float32),
            "gaussian_binding_distance": binding_distance.astype(np.float32),
            "gaussian_rest_offset_table": rest_offsets.astype(np.float32),
            "gaussian_observation_count": observation_count,
            "gaussian_first_pair_observed": first_pair_observed,
            "gaussian_later_only_observed": later_only_observed,
            "gaussian_surface_source_class": initial_source_class,
            "gaussian_surface_distance": surface_distance,
            "gaussian_optimization_displacement": optimization_displacement,
            "gaussian_raw_optimization_displacement": (
                raw_optimization_displacement
            ),
            "gaussian_restored_to_initial_surface": restored_to_initial,
            "gaussian_binding_mode": np.asarray("surface_face_barycentric"),
            "gaussian_face_ids": face_binding["face_ids"],
            "gaussian_face_particle_indices": face_binding["face_particles"],
            "gaussian_face_barycentric_weights": face_binding["face_weights"],
            "gaussian_face_projection_distance": face_binding[
                "projection_distance"
            ],
            "gaussian_preprojection_surface_distance": (
                preprojection_surface_distance
            ),
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

    old_count = len(base["gaussian_rest_means_table"])
    old_scales = base["gaussian_scales"].astype(np.float64)
    nearest_old = cKDTree(
        base["gaussian_rest_means_table"].astype(np.float64)
    ).query(means, k=1, workers=-1)[0]
    scale_min = radius_m * args.minimum_scale_factor
    scale_max = radius_m * args.maximum_scale_factor
    gates = {
        "fine_surface_visual_gates_passed": bool(
            all(relevant_surface_gates.values())
        ),
        "physics_arrays_preserved": all(
            np.array_equal(output[name], base[name])
            for name in base
            if name not in GAUSSIAN_KEYS
        ),
        "gaussian_count_increased_at_least_4x": len(means) >= 4 * old_count,
        "gaussian_scales_are_smaller": bool(
            np.median(scales) < 0.75 * np.median(old_scales)
        ),
        "gaussian_scales_inside_configured_bounds": bool(
            scales.min() >= scale_min - 1.0e-7
            and scales.max() <= scale_max + 1.0e-7
        ),
        "later_frames_add_initially_occluded_gaussians": bool(
            later_only_observed.any()
        ),
        "all_gaussians_finite": bool(
            np.isfinite(means).all()
            and np.isfinite(quats).all()
            and np.isfinite(scales).all()
            and np.isfinite(opacities).all()
            and np.isfinite(colors).all()
        ),
        "gaussian_weights_are_convex": bool(
            np.all(barycentric >= -1.0e-7)
            and np.all(barycentric <= 1.0 + 1.0e-7)
            and np.max(np.abs(barycentric.sum(axis=1) - 1.0)) <= 1.0e-6
        ),
        "all_view_source_classes_retained": bool(
            np.any(initial_source_class == 1)
            and np.any(initial_source_class == 2)
            and np.any(initial_source_class == 3)
        ),
        "gaussians_are_surface_face_bound": bool(
            np.max(surface_distance) <= 2.0e-7
            and np.max(np.abs(rest_offsets)) == 0.0
            and np.allclose(
                face_binding["face_weights"].sum(axis=1), 1.0, atol=1.0e-6
            )
            and np.all(np.sum(barycentric == 0.0, axis=1) >= 1)
        ),
        "visual_force_uses_only_surface_face_supports": bool(
            np.all(np.count_nonzero(barycentric > 0.0, axis=1) <= 3)
            and np.all(
                barycentric[
                    ~np.any(
                        binding_particles[:, :, None]
                        == face_binding["face_particles"][:, None, :],
                        axis=2,
                    )
                ]
                == 0.0
            )
        ),
        "depth_uncertainty_does_not_prune_finite_gaussians": bool(
            args.maximum_surface_distance_mm is not None
            or np.all(keep)
        ),
        "unsupported_optimized_positions_restored_not_deleted": bool(
            restored_to_initial.any()
        ),
    }
    report = {
        "asset_version": 5,
        "stage": (
            "view_union_gaustar_face_bound_gaussians_on_frozen_paper_pbd"
        ),
        "method": {
            "faithful_original_components": [
                "surface-point initialization",
                "SimpleBodyBuilder._grow_gaussians",
                "600-step RGB Gaussian splatting optimization",
                "joint means/opacity/color/quaternion/scale optimization",
                "original learning rates 1e-4/1e-3/1e-2/1e-2/1e-2",
            ],
            "extensions": [
                "0.41 mm unsmoothed left-only union right-only union dual-view surface initialization",
                "six selected synchronized stereo pairs instead of first left frame only",
                "mask value 2 excludes instrument/highlight/invalid-depth occlusions from RGB loss",
                "dense per-view depth is retained while stereo/RAFT consistency becomes audit-only",
                "smaller Gaussian initialization and scale bounds",
                "surface distance retained as audit metadata instead of a depth-based hard prune",
                "zero-view optimized positions restored to their own multiview initialization instead of deleted",
                "topology-completion Gaussians retained with nearest observed-surface color",
                "GauSTAR-style Gaussian centers represented by surface-face barycentric coordinates",
                "face-normal Gaussian orientation with the smallest scale on the normal axis",
                "three surface-node visual-force support embedded in the compatible four-slot tetra array",
            ],
            "physics_geometry_changed": False,
            "stiffness_changed": False,
            "depth_residual_changed": False,
        },
        "parameters": {
            "material_density_kg_m3": float(
                base["particle_mass"].sum() / base["rest_tet_volume"].sum()
            ),
            "max_gaussians": args.max_gaussians,
            "initial_radius_mm": args.initial_radius_mm,
            "minimum_scale_mm": scale_min * 1000.0,
            "maximum_scale_mm": scale_max * 1000.0,
            "iterations": args.iterations,
            "training_resolution_scale": args.training_resolution_scale,
            "observation_confidence_policy": args.observation_confidence_policy,
            "maximum_surface_distance_mm": args.maximum_surface_distance_mm,
            "visibility_depth_tolerance_mm": args.visibility_depth_tolerance_mm,
            "seed": args.seed,
        },
        "counts": {
            "particles": int(len(base["rest_positions_table"])),
            "tetrahedra": int(len(base["tet_indices"])),
            "skin_triangles": int(len(base["surface_faces"])),
            "old_gaussians": old_count,
            "fine_surface_vertices": int(len(surface_vertices)),
            "fine_surface_observed_vertices": int(observed_surface.sum()),
            "fine_surface_inferred_vertices": int((~observed_surface).sum()),
            "optimized_before_filtering": int(len(keep)),
            "nonfinite_pruned": int((~finite).sum()),
            "surface_distance_pruned": int((~keep).sum()),
            "unsupported_positions_restored_to_initial_surface": int(
                restored_to_initial.sum()
            ),
            "gaussians": int(len(means)),
            "first_stereo_pair_observed": int(first_pair_observed.sum()),
            "later_only_observed": int(later_only_observed.sum()),
            "observed_in_any_selected_view": int(observed_any.sum()),
            "unobserved_in_selected_views": int((~observed_any).sum()),
            "face_bound_gaussians": int(contained.sum()),
            "left_only_source_gaussians": int(
                np.count_nonzero(initial_source_class == 1)
            ),
            "right_only_source_gaussians": int(
                np.count_nonzero(initial_source_class == 2)
            ),
            "dual_source_gaussians": int(
                np.count_nonzero(initial_source_class == 3)
            ),
            "inferred_source_gaussians": int(
                np.count_nonzero(initial_source_class == 4)
            ),
        },
        "statistics": {
            "scale_mm_min_p05_p50_p95_max_xyz": np.percentile(
                scales * 1000.0, [0, 5, 50, 95, 100], axis=0
            ).tolist(),
            "opacity_min_p05_p50_p95_max": five_number(opacities),
            "surface_distance_mm_min_p05_p50_p95_max": five_number(
                surface_distance, 1000.0
            ),
            "preprojection_surface_distance_mm_min_p05_p50_p95_max": five_number(
                preprojection_surface_distance, 1000.0
            ),
            "face_projection_distance_mm_min_p05_p50_p95_max": five_number(
                binding_distance, 1000.0
            ),
            "optimization_displacement_mm_min_p05_p50_p95_max": five_number(
                optimization_displacement, 1000.0
            ),
            "raw_optimization_displacement_mm_min_p05_p50_p95_max": five_number(
                raw_optimization_displacement, 1000.0
            ),
            "binding_distance_mm_min_p05_p50_p95_max": five_number(
                binding_distance, 1000.0
            ),
            "distance_to_old_gaussian_mm_min_p05_p50_p95_max": five_number(
                nearest_old, 1000.0
            ),
            "rgb_mean": colors.mean(axis=0).tolist(),
            "rgb_std": colors.std(axis=0).tolist(),
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
            "base_asset": str(args.base_asset.relative_to(REPO_ROOT)),
            "base_asset_sha256": sha256(args.base_asset),
            "fine_surface": str(args.fine_surface.relative_to(REPO_ROOT)),
            "fine_surface_sha256": sha256(args.fine_surface),
            "stage_b_report": str(args.stage_b_report.relative_to(REPO_ROOT)),
            "stage_b_report_sha256": sha256(args.stage_b_report),
        },
        "outputs": {
            "asset": str(output_asset.relative_to(REPO_ROOT)),
            "preview": str(output_preview.relative_to(REPO_ROOT)),
        },
        "gates": gates,
        "passed": bool(all(gates.values())),
    }
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Dense multiview tissue Gaussian gates failed")


if __name__ == "__main__":
    main()
