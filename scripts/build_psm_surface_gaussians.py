#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = REPO_ROOT / "data/super/psm_robot/psm.urdf"
DEFAULT_OUTPUT = REPO_ROOT / "data/super/psm_robot/psm_surface_gaussians.npz"
DEFAULT_REPORT = REPO_ROOT / "data/super/psm_robot/psm_surface_gaussians_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample link-local, surface-aligned Gaussians from URDF visual meshes."
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--root-link",
        default="PSM1_tool_main_link",
        help="Only sample this link and its descendants in the URDF joint tree.",
    )
    parser.add_argument("--density", type=float, default=30000.0)
    parser.add_argument("--min-samples-per-mesh", type=int, default=64)
    parser.add_argument("--max-samples-per-mesh", type=int, default=12000)
    parser.add_argument("--tangent-scale-factor", type=float, default=0.65)
    parser.add_argument("--min-tangent-scale", type=float, default=0.0007)
    parser.add_argument("--max-tangent-scale", type=float, default=0.0040)
    parser.add_argument("--normal-scale-factor", type=float, default=0.20)
    parser.add_argument("--min-normal-scale", type=float, default=0.0002)
    parser.add_argument("--max-normal-scale", type=float, default=0.0008)
    parser.add_argument("--opacity", type=float, default=0.90)
    parser.add_argument("--color", type=float, nargs=3, default=[0.65, 0.67, 0.70])
    parser.add_argument(
        "--use-urdf-material-colors",
        action="store_true",
        help=(
            "Use each visual element's URDF material color when present; "
            "fall back to --color otherwise."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def parse_vector(value: str | None, default: tuple[float, ...]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64)
    parsed = np.asarray([float(item) for item in value.split()], dtype=np.float64)
    if parsed.shape != (len(default),):
        raise ValueError(f"Expected {len(default)} values, got {value!r}")
    return parsed


def visual_transform(visual: ET.Element) -> np.ndarray:
    origin = visual.find("origin")
    xyz = parse_vector(None if origin is None else origin.get("xyz"), (0.0, 0.0, 0.0))
    rpy = parse_vector(None if origin is None else origin.get("rpy"), (0.0, 0.0, 0.0))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


def visual_color(visual: ET.Element, fallback: np.ndarray) -> np.ndarray:
    color = visual.find("material/color")
    if color is None or color.get("rgba") is None:
        return fallback.copy()
    rgba = parse_vector(color.get("rgba"), (0.0, 0.0, 0.0, 1.0))
    rgb = rgba[:3]
    if not np.all(np.isfinite(rgb)) or np.any(rgb <= 0.0) or np.any(rgb >= 1.0):
        return fallback.copy()
    return rgb.astype(np.float32)


def resolve_mesh_path(urdf_path: Path, filename: str) -> Path:
    if filename.startswith("package://"):
        raise ValueError(
            f"Package URI is not supported in the preprocessed PSM URDF: {filename}"
        )
    mesh_path = Path(filename)
    if not mesh_path.is_absolute():
        mesh_path = urdf_path.parent / mesh_path
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)
    return mesh_path


def tangent_frames(normals: np.ndarray) -> np.ndarray:
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-12)
    references = np.tile(np.array([0.0, 0.0, 1.0]), (len(normals), 1))
    near_parallel = np.abs(normals[:, 2]) > 0.9
    references[near_parallel] = np.array([0.0, 1.0, 0.0])
    tangent_x = np.cross(references, normals)
    tangent_x /= np.linalg.norm(tangent_x, axis=1, keepdims=True).clip(1e-12)
    tangent_y = np.cross(normals, tangent_x)
    return np.stack([tangent_x, tangent_y, normals], axis=2)


def link_subtree(root: ET.Element, root_link: str) -> set[str]:
    children: dict[str, list[str]] = {}
    all_links = {link.get("name") for link in root.findall("link")}
    if root_link not in all_links:
        raise KeyError(f"Root link is absent from the URDF: {root_link}")
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        children.setdefault(parent.get("link", ""), []).append(child.get("link", ""))
    selected: set[str] = set()
    pending = [root_link]
    while pending:
        link_name = pending.pop()
        if link_name in selected:
            continue
        selected.add(link_name)
        pending.extend(children.get(link_name, []))
    return selected


def sample_visual(
    mesh_path: Path,
    mesh_scale: np.ndarray,
    X_link_visual: np.ndarray,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise ValueError(f"Visual mesh has no triangles: {mesh_path}")
    vertices = np.asarray(mesh.vertices)
    vertices *= mesh_scale.reshape(1, 3)
    vertices[:] = (
        X_link_visual[:3, :3] @ vertices.T + X_link_visual[:3, 3:4]
    ).T
    mesh.compute_vertex_normals()
    cloud = mesh.sample_points_poisson_disk(num_samples)
    means = np.asarray(cloud.points, dtype=np.float32)
    normals = np.asarray(cloud.normals, dtype=np.float32)
    if normals.shape != means.shape:
        raise RuntimeError(f"Poisson sampling did not return normals for {mesh_path}")

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    distances = scene.compute_distance(o3d.core.Tensor(means)).numpy()
    return means, normals, distances


def main() -> None:
    args = parse_args()
    args.urdf = args.urdf.resolve()
    args.output = args.output.resolve()
    args.report = args.report.resolve()
    if args.density <= 0:
        raise ValueError("--density must be positive")
    if not 0.0 < args.opacity < 1.0:
        raise ValueError("--opacity must lie strictly between 0 and 1")
    fallback_color = np.asarray(args.color, dtype=np.float32)
    if np.any(fallback_color <= 0.0) or np.any(fallback_color >= 1.0):
        raise ValueError("--color values must lie strictly between 0 and 1")

    o3d.utility.random.seed(args.seed)
    root = ET.parse(args.urdf).getroot()
    selected_links = link_subtree(root, args.root_link)
    all_means: list[np.ndarray] = []
    all_quats: list[np.ndarray] = []
    all_scales: list[np.ndarray] = []
    all_opacities: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_link_ids: list[np.ndarray] = []
    link_names: list[str] = []
    link_reports: list[dict] = []

    for link in root.findall("link"):
        link_name = link.get("name")
        if link_name is None or link_name not in selected_links:
            continue
        link_id = len(link_names)
        link_means: list[np.ndarray] = []
        link_quats: list[np.ndarray] = []
        link_scales: list[np.ndarray] = []
        link_colors: list[np.ndarray] = []
        link_distances: list[np.ndarray] = []
        visual_reports: list[dict] = []

        for visual_index, visual in enumerate(link.findall("visual")):
            mesh_element = visual.find("geometry/mesh")
            if mesh_element is None:
                continue
            filename = mesh_element.get("filename")
            if filename is None:
                continue
            mesh_path = resolve_mesh_path(args.urdf, filename)
            mesh_scale = parse_vector(mesh_element.get("scale"), (1.0, 1.0, 1.0))
            X_link_visual = visual_transform(visual)

            probe = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
            probe_vertices = np.asarray(probe.vertices)
            probe_vertices *= mesh_scale.reshape(1, 3)
            probe.compute_triangle_normals()
            area = float(probe.get_surface_area())
            num_samples = int(round(area * args.density))
            num_samples = max(args.min_samples_per_mesh, num_samples)
            num_samples = min(args.max_samples_per_mesh, num_samples)

            means, normals, distances = sample_visual(
                mesh_path, mesh_scale, X_link_visual, num_samples
            )
            spacing = np.sqrt(area / max(num_samples, 1))
            tangent_scale = float(
                np.clip(
                    spacing * args.tangent_scale_factor,
                    args.min_tangent_scale,
                    args.max_tangent_scale,
                )
            )
            normal_scale = float(
                np.clip(
                    tangent_scale * args.normal_scale_factor,
                    args.min_normal_scale,
                    args.max_normal_scale,
                )
            )
            quats = Rotation.from_matrix(tangent_frames(normals)).as_quat(
                scalar_first=True
            ).astype(np.float32)
            scales = np.tile(
                np.array([tangent_scale, tangent_scale, normal_scale], np.float32),
                (num_samples, 1),
            )
            sampled_color = (
                visual_color(visual, fallback_color)
                if args.use_urdf_material_colors
                else fallback_color
            )

            link_means.append(means)
            link_quats.append(quats)
            link_scales.append(scales)
            link_colors.append(np.tile(sampled_color, (num_samples, 1)))
            link_distances.append(distances)
            visual_reports.append(
                {
                    "visual_index": visual_index,
                    "mesh": str(mesh_path.relative_to(REPO_ROOT)),
                    "surface_area_m2": area,
                    "num_gaussians": num_samples,
                    "estimated_spacing_m": float(spacing),
                    "tangent_scale_m": tangent_scale,
                    "normal_scale_m": normal_scale,
                    "color_rgb": sampled_color.tolist(),
                    "surface_distance_max_m": float(np.max(distances)),
                }
            )

        if not link_means:
            continue
        means = np.concatenate(link_means)
        quats = np.concatenate(link_quats)
        scales = np.concatenate(link_scales)
        colors = np.concatenate(link_colors)
        distances = np.concatenate(link_distances)
        link_names.append(link_name)
        all_means.append(means)
        all_quats.append(quats)
        all_scales.append(scales)
        all_opacities.append(np.full(len(means), args.opacity, dtype=np.float32))
        all_colors.append(colors)
        all_link_ids.append(np.full(len(means), link_id, dtype=np.int16))
        link_reports.append(
            {
                "link_name": link_name,
                "num_gaussians": int(len(means)),
                "surface_distance_p95_m": float(np.percentile(distances, 95)),
                "surface_distance_max_m": float(np.max(distances)),
                "visuals": visual_reports,
            }
        )

    if not all_means:
        raise RuntimeError(f"No visual mesh Gaussians were generated from {args.urdf}")

    output = {
        "means": np.concatenate(all_means).astype(np.float32),
        "quats_wxyz": np.concatenate(all_quats).astype(np.float32),
        "scales": np.concatenate(all_scales).astype(np.float32),
        "opacities": np.concatenate(all_opacities).astype(np.float32),
        "colors": np.concatenate(all_colors).astype(np.float32),
        "link_ids": np.concatenate(all_link_ids),
        "link_names": np.asarray(link_names),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)

    all_distances = np.concatenate(
        [
            np.asarray(
                [visual["surface_distance_max_m"] for visual in link["visuals"]],
                dtype=np.float64,
            )
            for link in link_reports
        ]
    )
    report = {
        "urdf": str(args.urdf.relative_to(REPO_ROOT)),
        "urdf_sha256": hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
        "output": str(args.output.relative_to(REPO_ROOT)),
        "num_gaussians": int(len(output["means"])),
        "num_links": len(link_names),
        "root_link": args.root_link,
        "selected_link_subtree": sorted(selected_links),
        "density_per_m2": args.density,
        "min_samples_per_mesh": args.min_samples_per_mesh,
        "max_samples_per_mesh": args.max_samples_per_mesh,
        "tangent_scale_factor": args.tangent_scale_factor,
        "min_tangent_scale_m": args.min_tangent_scale,
        "max_tangent_scale_m": args.max_tangent_scale,
        "normal_scale_factor": args.normal_scale_factor,
        "min_normal_scale_m": args.min_normal_scale,
        "max_normal_scale_m": args.max_normal_scale,
        "use_urdf_material_colors": args.use_urdf_material_colors,
        "surface_distance_max_m": float(np.max(all_distances)),
        "scale_m_percentiles": np.percentile(
            output["scales"], [0, 5, 50, 95, 100], axis=0
        ).tolist(),
        "links": link_reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "links"}, indent=2))


if __name__ == "__main__":
    main()
