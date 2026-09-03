#!/usr/bin/env python3
"""仅由 canonical_scan RGB-D 重建 FixedSuperBest 组织资产。

该脚本刻意不读取 ``ground_truth/``。它先融合任务前的隔离组织多视角 RGB-D，
再建立两套相互独立的几何：

* 闭合、较稀疏的双层四面体物理网格；
* 较密的闭合视觉三角面，每个面中心放置一个各向异性 Gaussian。

视觉顶点通过物理边界三角面的重心坐标与静止偏移嵌入四面体组织，输出格式与
项目现有 ``paper_pbd`` 资产一致。十区域材料真值不会进入资产；GUI 初始化继续
使用统一的 distance/volume/shape 刚度，局部差异只允许后续在线更新得到。
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import Delaunay, cKDTree
from scipy.spatial.transform import Rotation


TOP_MARKER = 1
SIDE_MARKER = 2
BOTTOM_MARKER = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fusion-voxel-mm", type=float, default=0.35)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--minimum-view-observations", type=int, default=2)
    parser.add_argument(
        "--scan-pose-lag-frames",
        type=int,
        default=1,
        help=(
            "当前已录制扫描的 Replicator 数据相对相机 pose 滞后一帧；"
            "使用上一条 pose 反投影，首条缓存帧自动丢弃。"
        ),
    )
    parser.add_argument("--physical-spacing-mm", type=float, default=2.40)
    parser.add_argument("--visual-spacing-mm", type=float, default=0.90)
    parser.add_argument("--maximum-grid-edge-factor", type=float, default=2.25)
    parser.add_argument("--density", type=float, default=1000.0)
    parser.add_argument("--visual-radius-mm", type=float, default=0.40)
    parser.add_argument("--fixed-support-width-mm", type=float, default=6.0)
    parser.add_argument(
        "--support-mode",
        choices=("free", "edge_hard"),
        default="edge_hard",
        help="free 不在首帧重建资产中设置任何固定点；edge_hard 保留旧边缘支撑。",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def voxel_average_per_view(
    points: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray,
    voxel: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = np.floor(points / voxel + 0.5).astype(np.int64)
    unique, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    averaged = []
    for values in (points, colors, normals):
        sums = np.stack(
            [np.bincount(inverse, weights=values[:, axis]) for axis in range(3)],
            axis=1,
        )
        averaged.append(sums / counts[:, None])
    return unique, averaged[0], averaged[1], averaged[2]


def fuse_canonical_scan(
    scan_root: Path,
    manifest: dict,
    voxel: float,
    stride: int,
    minimum_views: int,
    pose_lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    all_keys: list[np.ndarray] = []
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_normals: list[np.ndarray] = []
    used_views: list[int] = []
    skipped_views: list[int] = []

    views = manifest["views"]
    for image_index, record in enumerate(views):
        pose_index = image_index - pose_lag
        if pose_index < 0:
            skipped_views.append(int(record["view"]))
            continue
        pose_record = views[pose_index]
        depth = np.load(scan_root / record["depth_path"]).astype(np.float64)
        mask = np.asarray(Image.open(scan_root / record["mask_path"]).convert("L")) > 0
        rgb = np.asarray(Image.open(scan_root / record["rgb_path"]).convert("RGB"))
        valid = mask & np.isfinite(depth) & (depth > 0.0)
        rows, columns = np.nonzero(valid)
        keep = (rows % stride == 0) & (columns % stride == 0)
        rows, columns = rows[keep], columns[keep]
        z = depth[rows, columns]
        K = np.asarray(pose_record["K"], dtype=np.float64)
        X_WC = np.asarray(pose_record["X_WC_ros_optical"], dtype=np.float64)
        camera_points = np.column_stack(
            (
                (columns - K[0, 2]) * z / K[0, 0],
                (rows - K[1, 2]) * z / K[1, 1],
                z,
            )
        )
        world_points = camera_points @ X_WC[:3, :3].T + X_WC[:3, 3]
        world_normals = X_WC[:3, 3] - world_points
        world_normals /= np.maximum(
            np.linalg.norm(world_normals, axis=1, keepdims=True), 1.0e-12
        )
        keys, view_points, view_colors, view_normals = voxel_average_per_view(
            world_points,
            rgb[rows, columns].astype(np.float64) / 255.0,
            world_normals,
            voxel,
        )
        all_keys.append(keys)
        all_points.append(view_points)
        all_colors.append(view_colors)
        all_normals.append(view_normals)
        used_views.append(int(record["view"]))

    keys = np.concatenate(all_keys, axis=0)
    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    normals = np.concatenate(all_normals, axis=0)
    _, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    averaged = []
    for values in (points, colors, normals):
        sums = np.stack(
            [np.bincount(inverse, weights=values[:, axis]) for axis in range(3)],
            axis=1,
        )
        averaged.append(sums / counts[:, None])
    keep = counts >= minimum_views
    fused_points = averaged[0][keep]
    fused_colors = np.clip(averaged[1][keep], 0.0, 1.0)
    fused_normals = averaged[2][keep]
    fused_normals /= np.maximum(
        np.linalg.norm(fused_normals, axis=1, keepdims=True), 1.0e-12
    )
    report = {
        "used_image_views": used_views,
        "skipped_cached_views": skipped_views,
        "pose_lag_frames": pose_lag,
        "points_after_fusion": int(len(fused_points)),
        "bounds_m": [fused_points.min(axis=0).tolist(), fused_points.max(axis=0).tolist()],
    }
    return fused_points, fused_colors, fused_normals, report


def two_layer_split(values: np.ndarray) -> float:
    centers = np.percentile(values, (15.0, 85.0)).astype(np.float64)
    for _ in range(24):
        distances = np.abs(values[:, None] - centers[None, :])
        labels = np.argmin(distances, axis=1)
        updated = np.asarray(
            [np.median(values[labels == index]) for index in range(2)],
            dtype=np.float64,
        )
        if np.max(np.abs(updated - centers)) < 1.0e-10:
            break
        centers = updated
    centers.sort()
    if centers[1] - centers[0] < 0.00035:
        raise RuntimeError(f"扫描无法分出组织上下表面：centers={centers.tolist()}")
    return float(centers.mean())


def largest_face_component(faces: np.ndarray) -> np.ndarray:
    vertex_faces: dict[int, list[int]] = {}
    for face_id, face in enumerate(faces):
        for vertex in face:
            vertex_faces.setdefault(int(vertex), []).append(face_id)
    unseen = set(range(len(faces)))
    components: list[list[int]] = []
    while unseen:
        start = unseen.pop()
        component = [start]
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for vertex in faces[current]:
                for neighbor in vertex_faces[int(vertex)]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.append(neighbor)
                        queue.append(neighbor)
        components.append(component)
    return faces[max(components, key=len)]


def compact_mesh(
    top: np.ndarray,
    bottom: np.ndarray,
    top_colors: np.ndarray,
    bottom_colors: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    used = np.unique(faces)
    remap = np.full(len(top), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return top[used], bottom[used], top_colors[used], bottom_colors[used], remap[faces]


def sampled_layers(
    points: np.ndarray,
    colors: np.ndarray,
    spacing: float,
    edge_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    split = two_layer_split(points[:, 2])
    top_mask = points[:, 2] > split
    bottom_mask = points[:, 2] < split
    top_points, top_colors_source = points[top_mask], colors[top_mask]
    bottom_points, bottom_colors_source = points[bottom_mask], colors[bottom_mask]
    origin = top_points[:, :2].min(axis=0)
    cells = np.floor((top_points[:, :2] - origin) / spacing + 0.5).astype(np.int64)
    _, inverse, counts = np.unique(
        cells, axis=0, return_inverse=True, return_counts=True
    )
    top = np.stack(
        [
            np.bincount(inverse, weights=top_points[:, axis]) / counts
            for axis in range(3)
        ],
        axis=1,
    )
    top_color = np.stack(
        [
            np.bincount(inverse, weights=top_colors_source[:, axis]) / counts
            for axis in range(3)
        ],
        axis=1,
    )
    bottom_tree = cKDTree(bottom_points[:, :2])
    distance, bottom_ids = bottom_tree.query(top[:, :2], workers=-1)
    keep = distance <= edge_factor * spacing
    top, top_color = top[keep], top_color[keep]
    bottom = bottom_points[bottom_ids[keep]].copy()
    bottom[:, :2] = top[:, :2]
    bottom_color = bottom_colors_source[bottom_ids[keep]]
    thickness = top[:, 2] - bottom[:, 2]
    keep = (thickness >= 0.00035) & (thickness <= 0.0030)
    top, bottom = top[keep], bottom[keep]
    top_color, bottom_color = top_color[keep], bottom_color[keep]

    faces = Delaunay(top[:, :2]).simplices.astype(np.int32)
    triangles = top[faces, :2]
    edge_01 = triangles[:, 1] - triangles[:, 0]
    edge_02 = triangles[:, 2] - triangles[:, 0]
    signed_twice_area = (
        edge_01[:, 0] * edge_02[:, 1]
        - edge_01[:, 1] * edge_02[:, 0]
    )
    negative = signed_twice_area < 0.0
    faces[negative, 1], faces[negative, 2] = (
        faces[negative, 2].copy(),
        faces[negative, 1].copy(),
    )
    edges = np.stack(
        (
            np.linalg.norm(top[faces[:, 1], :2] - top[faces[:, 0], :2], axis=1),
            np.linalg.norm(top[faces[:, 2], :2] - top[faces[:, 1], :2], axis=1),
            np.linalg.norm(top[faces[:, 0], :2] - top[faces[:, 2], :2], axis=1),
        ),
        axis=1,
    )
    faces = faces[np.max(edges, axis=1) <= edge_factor * spacing]
    faces = largest_face_component(faces)
    top, bottom, top_color, bottom_color, faces = compact_mesh(
        top, bottom, top_color, bottom_color, faces
    )
    report = {
        "spacing_mm": spacing * 1000.0,
        "vertices_per_layer": int(len(top)),
        "top_faces": int(len(faces)),
        "thickness_mm": np.percentile(
            top[:, 2] - bottom[:, 2], (0, 5, 50, 95, 100)
        ).tolist(),
        "z_split_m": split,
    }
    return top, bottom, top_color, bottom_color, faces, report


def directed_boundary_edges(faces: np.ndarray) -> np.ndarray:
    counts: Counter[tuple[int, int]] = Counter()
    direction: dict[tuple[int, int], tuple[int, int]] = {}
    for a, b, c in faces.tolist():
        for edge in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(edge))
            counts[key] += 1
            direction[key] = edge
    boundary = [direction[key] for key, count in counts.items() if count == 1]
    if not boundary:
        raise RuntimeError("表面三角网格没有外边界")
    degree: Counter[int] = Counter()
    for a, b in boundary:
        degree[a] += 1
        degree[b] += 1
    if any(value != 2 for value in degree.values()):
        raise RuntimeError("扫描轮廓不是简单闭合边界")
    return np.asarray(boundary, dtype=np.int32)


def close_layers(
    top: np.ndarray,
    bottom: np.ndarray,
    top_colors: np.ndarray,
    bottom_colors: np.ndarray,
    top_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(top)
    bottom_faces = top_faces[:, ::-1] + count
    side_faces: list[tuple[int, int, int]] = []
    for a, b in directed_boundary_edges(top_faces):
        side_faces.extend(((a, b + count, b), (a, a + count, b + count)))
    vertices = np.concatenate((top, bottom), axis=0)
    colors = np.concatenate((top_colors, bottom_colors), axis=0)
    faces = np.concatenate(
        (top_faces, bottom_faces, np.asarray(side_faces, dtype=np.int32)), axis=0
    )
    return vertices, faces, np.clip(colors, 0.01, 0.99)


def structured_tetrahedra(top_faces: np.ndarray, vertices_per_layer: int) -> np.ndarray:
    tets: list[tuple[int, int, int, int]] = []
    for face in top_faces:
        i, j, k = sorted(int(value) for value in face)
        ib, jb, kb = i + vertices_per_layer, j + vertices_per_layer, k + vertices_per_layer
        tets.extend(((i, j, k, kb), (i, j, jb, kb), (i, ib, jb, kb)))
    return np.asarray(tets, dtype=np.int32)


def orient_tetrahedra(
    points: np.ndarray, tetrahedra: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    output = tetrahedra.copy()
    vectors = points[output[:, 1:]] - points[output[:, :1]]
    signed = np.linalg.det(np.transpose(vectors, (0, 2, 1))) / 6.0
    negative = signed < 0.0
    output[negative, 1], output[negative, 2] = (
        output[negative, 2].copy(), output[negative, 1].copy()
    )
    volumes = np.abs(signed)
    if np.any(volumes <= 1.0e-14):
        raise RuntimeError(
            f"扫描体网格含退化四面体：min={float(volumes.min()):.3e} m3"
        )
    return output, volumes.astype(np.float32)


def boundary_faces(tetrahedra: np.ndarray) -> np.ndarray:
    incidence: dict[tuple[int, int, int], tuple[int, tuple[int, int, int]]] = {}
    for a, b, c, d in tetrahedra.tolist():
        for face in ((b, c, d), (a, d, c), (a, b, d), (a, c, b)):
            key = tuple(sorted(face))
            count, saved = incidence.get(key, (0, face))
            incidence[key] = (count + 1, saved)
    return np.asarray(
        [face for count, face in incidence.values() if count == 1], dtype=np.int32
    )


def face_markers(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = points[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
    markers = np.full(len(faces), SIDE_MARKER, dtype=np.uint8)
    markers[normals[:, 2] > 0.45] = TOP_MARKER
    markers[normals[:, 2] < -0.45] = BOTTOM_MARKER
    return markers


def unique_tet_edges(tetrahedra: np.ndarray) -> np.ndarray:
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    return np.unique(
        np.sort(np.concatenate([tetrahedra[:, pair] for pair in pairs]), axis=1),
        axis=0,
    ).astype(np.int32)


def triangle_bindings(
    queries: np.ndarray, points: np.ndarray, faces: np.ndarray, candidates: int = 32
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = points[faces]
    tree = cKDTree(triangles.mean(axis=1))
    ids = tree.query(queries, k=min(candidates, len(faces)), workers=-1)[1]
    if ids.ndim == 1:
        ids = ids[:, None]
    selected = np.empty((len(queries), 3), dtype=np.int32)
    weights = np.empty((len(queries), 3), dtype=np.float64)
    offsets = np.empty((len(queries), 3), dtype=np.float64)
    for query_id, (query, candidates_for_query) in enumerate(zip(queries, ids)):
        best = None
        for face_id in np.atleast_1d(candidates_for_query):
            tri = triangles[int(face_id)]
            e0, e1 = tri[1] - tri[0], tri[2] - tri[0]
            gram = np.asarray(((e0 @ e0, e0 @ e1), (e0 @ e1, e1 @ e1)))
            if np.linalg.det(gram) <= 1.0e-18:
                continue
            vw = np.linalg.solve(gram, (e0 @ (query - tri[0]), e1 @ (query - tri[0])))
            barycentric = np.asarray((1.0 - vw.sum(), vw[0], vw[1]))
            projected = barycentric @ tri
            score = float(
                np.linalg.norm(query - projected)
                + 0.01 * np.maximum(-barycentric, 0.0).sum()
            )
            if best is None or score < best[0]:
                best = (score, int(face_id), barycentric, query - projected)
        if best is None:
            raise RuntimeError("视觉顶点无法绑定到物理边界")
        selected[query_id] = faces[best[1]]
        weights[query_id] = best[2]
        offsets[query_id] = best[3]
    return selected, weights.astype(np.float32), offsets.astype(np.float32)


def tetrahedral_bindings(
    queries: np.ndarray, points: np.ndarray, tetrahedra: np.ndarray, candidates: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    tet_points = points[tetrahedra]
    tree = cKDTree(tet_points.mean(axis=1))
    ids = tree.query(queries, k=min(candidates, len(tetrahedra)), workers=-1)[1]
    if ids.ndim == 1:
        ids = ids[:, None]
    selected = np.empty(len(queries), dtype=np.int32)
    weights = np.empty((len(queries), 4), dtype=np.float64)
    for query_id, (query, candidates_for_query) in enumerate(zip(queries, ids)):
        best = None
        for tet_id in np.atleast_1d(candidates_for_query):
            vertices = tet_points[int(tet_id)]
            matrix = np.vstack((vertices.T, np.ones(4)))
            barycentric = np.linalg.solve(matrix, np.append(query, 1.0))
            projected = np.clip(barycentric, 0.0, 1.0)
            projected /= max(float(projected.sum()), 1.0e-12)
            score = float(np.linalg.norm(projected @ vertices - query))
            if best is None or score < best[0]:
                best = (score, int(tet_id), projected)
        assert best is not None
        selected[query_id], weights[query_id] = best[1], best[2]
    return selected, weights.astype(np.float32)


def face_gaussians(
    vertices: np.ndarray, faces: np.ndarray, vertex_colors: np.ndarray
) -> dict[str, np.ndarray]:
    triangles = vertices[faces]
    edge_x = triangles[:, 1] - triangles[:, 0]
    edge_y_raw = triangles[:, 2] - triangles[:, 0]
    tangent_x = edge_x / np.maximum(np.linalg.norm(edge_x, axis=1, keepdims=True), 1.0e-12)
    normal = np.cross(edge_x, edge_y_raw)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1.0e-12)
    tangent_y = np.cross(normal, tangent_x)
    tangent_y /= np.maximum(np.linalg.norm(tangent_y, axis=1, keepdims=True), 1.0e-12)
    frames = np.stack((tangent_x, tangent_y, normal), axis=-1)
    quaternions = Rotation.from_matrix(frames).as_quat().astype(np.float32)
    length = np.linalg.norm(edge_x, axis=1)
    height = np.abs(np.einsum("ni,ni->n", edge_y_raw, tangent_y))
    scales = np.column_stack(
        (
            np.maximum(0.58 * length, 0.00025),
            np.maximum(0.58 * height, 0.00025),
            np.full(len(faces), 0.00020),
        )
    )
    return {
        "means": triangles.mean(axis=1).astype(np.float32),
        "quats": quaternions[:, [3, 0, 1, 2]],
        "scales": scales.astype(np.float32),
        "colors": np.clip(vertex_colors[faces].mean(axis=1), 0.01, 0.99).astype(np.float32),
    }


def mesh_is_closed(faces: np.ndarray) -> bool:
    counts: Counter[tuple[int, int]] = Counter()
    for a, b, c in faces.tolist():
        for edge in ((a, b), (b, c), (c, a)):
            counts[tuple(sorted(edge))] += 1
    return bool(counts) and all(value == 2 for value in counts.values())


def write_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces)
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False):
        raise RuntimeError(f"无法写出预览网格：{path}")


def main() -> None:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    scan_root = dataset / "canonical_scan"
    manifest_path = scan_root / "cameras.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少 canonical_scan 标定：{manifest_path}")
    if any(
        value <= 0
        for value in (
            args.fusion_voxel_mm,
            args.physical_spacing_mm,
            args.visual_spacing_mm,
            args.density,
            args.visual_radius_mm,
        )
    ):
        raise ValueError("所有距离、密度和半径参数必须为正数")
    if args.fixed_support_width_mm <= 0.0:
        raise ValueError("固定边宽度必须为正数")
    output_dir = dataset / "gui_assets"
    output = output_dir / "tissue_fixedsuperbest.npz"
    report_path = output_dir / "tissue_fixedsuperbest_report.json"
    if output_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有 GUI 资产目录：{output_dir}")

    manifest = read_json(manifest_path)
    points, colors, normals, fusion_report = fuse_canonical_scan(
        scan_root,
        manifest,
        args.fusion_voxel_mm / 1000.0,
        args.pixel_stride,
        args.minimum_view_observations,
        args.scan_pose_lag_frames,
    )
    physical = sampled_layers(
        points,
        colors,
        args.physical_spacing_mm / 1000.0,
        args.maximum_grid_edge_factor,
    )
    p_top, p_bottom, p_top_color, p_bottom_color, p_top_faces, physical_report = physical
    physical_vertices, _, physical_colors = close_layers(
        p_top, p_bottom, p_top_color, p_bottom_color, p_top_faces
    )
    tetrahedra = structured_tetrahedra(p_top_faces, len(p_top))
    tetrahedra, volumes = orient_tetrahedra(physical_vertices, tetrahedra)
    surfaces = boundary_faces(tetrahedra)
    markers = face_markers(physical_vertices, surfaces)

    visual = sampled_layers(
        points,
        colors,
        args.visual_spacing_mm / 1000.0,
        args.maximum_grid_edge_factor,
    )
    v_top, v_bottom, v_top_color, v_bottom_color, v_top_faces, visual_report = visual
    visual_vertices, visual_faces, visual_colors = close_layers(
        v_top, v_bottom, v_top_color, v_bottom_color, v_top_faces
    )

    visual_support, visual_weights, visual_offsets = triangle_bindings(
        visual_vertices, physical_vertices, surfaces
    )
    reconstructed = (
        np.einsum("ni,nij->nj", visual_weights, physical_vertices[visual_support])
        + visual_offsets
    )
    embedding_error = np.linalg.norm(reconstructed - visual_vertices, axis=1)
    gaussian = face_gaussians(visual_vertices, visual_faces, visual_colors)
    tet_ids, tet_weights = tetrahedral_bindings(
        gaussian["means"], physical_vertices, tetrahedra
    )

    edges = unique_tet_edges(tetrahedra)
    edge_lengths = np.linalg.norm(
        physical_vertices[edges[:, 0]] - physical_vertices[edges[:, 1]], axis=1
    )
    masses = np.zeros(len(physical_vertices), dtype=np.float64)
    for corner in range(4):
        np.add.at(masses, tetrahedra[:, corner], args.density * volumes / 4.0)
    fixed = (
        np.zeros(len(physical_vertices), dtype=bool)
        if args.support_mode == "free"
        else physical_vertices[:, 0]
        <= physical_vertices[:, 0].min() + args.fixed_support_width_mm / 1000.0
    )
    surface_nodes = np.zeros(len(physical_vertices), dtype=bool)
    surface_nodes[np.unique(surfaces)] = True
    top_nodes = np.zeros(len(physical_vertices), dtype=bool)
    top_nodes[np.unique(surfaces[markers == TOP_MARKER])] = True

    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output,
        rest_positions_table=physical_vertices.astype(np.float32),
        tet_indices=tetrahedra,
        surface_faces=surfaces,
        surface_face_markers=markers,
        collision_skin_faces=surfaces,
        collision_skin_face_markers=markers,
        collision_skin_enabled_faces=np.ones(len(surfaces), dtype=bool),
        particle_mass=masses.astype(np.float32),
        particle_radius=np.zeros(len(physical_vertices), dtype=np.float32),
        particle_visual_radius=np.full(
            len(physical_vertices), args.visual_radius_mm / 1000.0, dtype=np.float32
        ),
        particle_target_spacing=np.full(
            len(physical_vertices), np.median(edge_lengths), dtype=np.float32
        ),
        particle_inward_depth=np.zeros(len(physical_vertices), dtype=np.float32),
        surface_node_mask=surface_nodes,
        top_node_mask=top_nodes,
        fixed_mask=fixed,
        support_candidate_mask=fixed,
        rest_tet_volume=volumes,
        pbd_edge_indices=edges,
        pbd_rest_edge_length=edge_lengths.astype(np.float32),
        pbd_shape_cluster_indices=tetrahedra,
        gaussian_rest_means_table=gaussian["means"],
        gaussian_rest_quats_table_wxyz=gaussian["quats"],
        gaussian_scales=gaussian["scales"],
        gaussian_opacities=np.full(len(visual_faces), 0.97, dtype=np.float32),
        gaussian_colors_rgb=gaussian["colors"],
        gaussian_tet_ids=tet_ids,
        gaussian_particle_indices=tetrahedra[tet_ids],
        gaussian_barycentric_weights=tet_weights,
        gaussian_rest_offset_table=np.zeros_like(gaussian["means"]),
        gaussian_binding_mode=np.asarray("visual_surface_face_centroid"),
        gaussian_visual_face_ids=np.arange(len(visual_faces), dtype=np.int32),
        visual_surface_rest_vertices_table=visual_vertices.astype(np.float32),
        visual_surface_faces=visual_faces,
        visual_vertex_particle_indices=visual_support,
        visual_vertex_barycentric_weights=visual_weights,
        visual_vertex_rest_offset_table=visual_offsets,
    )
    write_mesh(
        output_dir / "physical_surface_scan_reconstruction.ply",
        physical_vertices,
        surfaces,
        physical_colors,
    )
    write_mesh(
        output_dir / "visual_surface_scan_reconstruction.ply",
        visual_vertices,
        visual_faces,
        visual_colors,
    )
    gates = {
        "canonical_scan_only": True,
        "no_ground_truth_input": True,
        "physical_surface_closed": mesh_is_closed(surfaces),
        "visual_surface_closed": mesh_is_closed(visual_faces),
        "all_tetrahedra_positive": bool(np.all(volumes > 0.0)),
        "finite_asset_arrays": bool(
            np.isfinite(physical_vertices).all()
            and np.isfinite(visual_vertices).all()
            and np.isfinite(gaussian["colors"]).all()
        ),
        "visual_embedding_exact": bool(float(embedding_error.max()) <= 2.0e-8),
        "uniform_initial_stiffness_only": True,
        "support_policy_valid": bool(
            (args.support_mode == "free" and not np.any(fixed))
            or (
                args.support_mode == "edge_hard"
                and np.any(fixed)
                and np.any(~fixed)
            )
        ),
    }
    report = {
        "schema": "fixedsuperbest.canonical_scan_tissue_asset.v1",
        "说明": (
            "资产仅由 canonical_scan RGB-D/mask/标定重建；未读取 ground_truth，"
            "未读取十区域 ID 或刚度。"
        ),
        "input": "canonical_scan/",
        "support_mode": args.support_mode,
        "output": str(output.relative_to(dataset)),
        "fusion": fusion_report,
        "physical_mesh": physical_report
        | {
            "particles": int(len(physical_vertices)),
            "tetrahedra": int(len(tetrahedra)),
            "boundary_faces": int(len(surfaces)),
            "fixed_particles": int(fixed.sum()),
            "mass_g": float(masses.sum() * 1000.0),
            "tet_volume_mm3": (
                np.percentile(volumes, (0, 5, 50, 95, 100)) * 1.0e9
            ).tolist(),
        },
        "visual_mesh": visual_report
        | {
            "vertices": int(len(visual_vertices)),
            "faces": int(len(visual_faces)),
            "gaussians": int(len(gaussian["means"])),
            "embedding_error_max_m": float(embedding_error.max()),
        },
        "initial_material": {
            "regional_labels_loaded": False,
            "distance_stiffness_uniform": 0.20,
            "volume_stiffness_uniform": 1.0e10,
            "shape_stiffness_uniform": 0.004,
        },
        "gates": gates,
        "passed": bool(all(gates.values())),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not report["passed"]:
        raise RuntimeError(f"扫描组织资产门禁失败：{gates}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
