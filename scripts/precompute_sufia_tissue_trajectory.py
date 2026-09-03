#!/usr/bin/env python3
"""预计算连续薄组织的十区域异质材料形变真值。

这个脚本只负责组织物理，不启动 Isaac Sim。输出随后由
``generate_sim_tissue_retraction_dataset.py`` 作为逐帧可变形网格读取；
因此官方 PSM/腹部资产和光线追踪仍由 Isaac Sim 渲染，而逐四面体杨氏模量
由 FixedSuperBest 已验证的材料感知 XPBD 求解器真正参与计算。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import warp as wp
import warp.sim

from embodied_gaussians.physics_simulator.integrator import (
    MaterialTetrahedronXPBDProjector,
)


TISSUE_CENTER = np.array([0.0022, 0.0, 0.0515], dtype=np.float64)
TISSUE_HALF_X = 0.0430
TISSUE_HALF_Y = 0.0420
TISSUE_THICKNESS = 0.0009
TISSUE_COLUMNS = 31
TISSUE_ROWS = 29
TISSUE_GRASP_OFFSET_X = 0.0400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--physics-hz", type=int, default=120)
    parser.add_argument("--regional-seed", type=int, default=240610788)
    parser.add_argument("--regional-youngs-min-pa", type=float, default=450.0)
    parser.add_argument("--regional-youngs-max-pa", type=float, default=1450.0)
    parser.add_argument("--poissons-ratio", type=float, default=0.45)
    parser.add_argument(
        "--geometry-profile",
        choices=(
            "fascia_sheet",
            "rectangular_tissue_strip",
            "irregular_liver_lobe",
        ),
        default="fascia_sheet",
        help=(
            "fascia_sheet 保留旧近方形连续薄片；rectangular_tissue_strip 使用长宽比"
            "约 1.7:1 的长方形薄组织；irregular_liver_lobe 使用非对称叶状组织瓣。"
        ),
    )
    parser.add_argument(
        "--motion-profile",
        choices=("retract_hold", "edge_lift_return"),
        default="retract_hold",
        help=(
            "retract_hold 保留旧抬升后牵拉；edge_lift_return 从长边夹持、"
            "抬升后平滑放回原位再释放。"
        ),
    )
    parser.add_argument("--material-iterations", type=int, default=24)
    parser.add_argument(
        "--grasp-coupling-frequency-hz",
        type=float,
        default=12.0,
        help="夹持核心周围平滑力传递区的响应频率；不是组织刚度参数。",
    )
    parser.add_argument(
        "--lift-displacement-mm",
        type=float,
        nargs=3,
        default=(0.20, 0.0, 25.0),
        metavar=("DX", "DY", "DZ"),
        help="夹持后第一阶段位移，单位 mm。",
    )
    parser.add_argument(
        "--pull-displacement-mm",
        type=float,
        nargs=3,
        default=(6.0, 0.0, 28.0),
        metavar=("DX", "DY", "DZ"),
        help="最终牵拉位移，单位 mm。",
    )
    parser.add_argument(
        "--grasp-offset-mm",
        type=float,
        nargs=2,
        default=None,
        metavar=("DX", "DY"),
        help="抓取核心相对组织中心的平面偏移，单位 mm；优先于旧的 x-only 参数。",
    )
    parser.add_argument(
        "--grasp-offset-x-mm",
        type=float,
        default=40.0,
        help="兼容旧数据生成命令的 x-only 抓取偏移，单位 mm。",
    )
    parser.add_argument(
        "--anchor-side",
        choices=("left", "right"),
        default="left",
        help="仅 support-mode=edge_hard 时使用的旧硬固定边。",
    )
    parser.add_argument(
        "--support-mode",
        choices=("free", "edge_hard"),
        default="edge_hard",
        help=(
            "free 不设置任何硬固定点，适合零重力的自由组织牵拉；"
            "edge_hard 仅用于复现旧数据。"
        ),
    )
    parser.add_argument("--device", default="cuda")
    args, _ = parser.parse_known_args()
    if args.frames < 2:
        parser.error("--frames 必须至少为 2")
    if args.fps <= 0 or args.physics_hz <= 0 or args.physics_hz % args.fps != 0:
        parser.error("--physics-hz 必须是 --fps 的正整数倍")
    if not 0.0 < args.regional_youngs_min_pa < args.regional_youngs_max_pa:
        parser.error("十区域杨氏模量范围必须满足 0 < min < max")
    if not 0.0 <= args.poissons_ratio < 0.5:
        parser.error("--poissons-ratio 必须位于 [0, 0.5)")
    if args.grasp_coupling_frequency_hz <= 0.0:
        parser.error("--grasp-coupling-frequency-hz 必须为正数")
    return args


def rounded_coordinate(sx: float, sy: float) -> tuple[float, float]:
    corner_fraction = 0.13
    ax, ay = abs(sx), abs(sy)
    corner_start = 1.0 - corner_fraction
    if ax > corner_start and ay > corner_start:
        dx = (ax - corner_start) / corner_fraction
        dy = (ay - corner_start) / corner_fraction
        radius = math.hypot(dx, dy)
        if radius > 1.0:
            dx /= radius
            dy /= radius
            ax = corner_start + corner_fraction * dx
            ay = corner_start + corner_fraction * dy
    return math.copysign(ax, sx), math.copysign(ay, sy)


def build_geometry(profile: str = "fascia_sheet") -> dict[str, np.ndarray]:
    """构建连续封闭组织体；渲染节点和物理节点一一对应。"""

    points: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    material_uv: list[tuple[float, float]] = []

    def add_surface(top: bool) -> int:
        offset = len(points)
        for row in range(TISSUE_ROWS):
            v = row / (TISSUE_ROWS - 1)
            sy = 2.0 * v - 1.0
            for column in range(TISSUE_COLUMNS):
                u = column / (TISSUE_COLUMNS - 1)
                sx = 2.0 * u - 1.0
                if profile == "irregular_liver_lobe":
                    # 将规则参数域平滑映射到近似圆盘，再叠加低频叶状起伏和一个
                    # 自然凹口。拓扑保持规则且闭合，但外轮廓不再是矩形或圆角矩形。
                    disc_x = sx * math.sqrt(max(1.0 - 0.48 * sy * sy, 0.50))
                    disc_y = sy * math.sqrt(max(1.0 - 0.48 * sx * sx, 0.50))
                    radius = math.hypot(disc_x, disc_y)
                    theta = math.atan2(disc_y, disc_x) if radius > 1.0e-9 else 0.0
                    angular_delta = math.atan2(
                        math.sin(theta - 2.30), math.cos(theta - 2.30)
                    )
                    boundary_scale = (
                        1.0
                        + 0.090 * math.sin(3.0 * theta + 0.35)
                        + 0.048 * math.sin(5.0 * theta - 1.10)
                        + 0.026 * math.cos(7.0 * theta + 0.20)
                        - 0.105
                        * math.exp(-0.5 * (angular_delta / 0.24) ** 2)
                    )
                    center_blend = min(1.0, radius / 0.30)
                    scale = 1.0 + center_blend * (boundary_scale - 1.0)
                    rx, ry = disc_x * scale, disc_y * scale
                    # 肝组织瓣略呈斜置的非对称椭圆，右侧为主要牵拉叶。
                    local_x = 0.0470 * (rx + 0.095 * ry)
                    local_y = 0.0355 * (ry - 0.035 * rx)
                    angle = math.radians(-5.0)
                    shaped_x = math.cos(angle) * local_x - math.sin(angle) * local_y
                    shaped_y = math.sin(angle) * local_x + math.cos(angle) * local_y
                    radial_clamped = min(radius, 1.0)
                    dome = 0.0042 * max(0.0, 1.0 - radial_clamped**2) ** 1.25
                    asymmetric_curve = 0.00115 * rx + 0.00055 * ry
                    broad_fold = (
                        0.00055
                        * math.sin(math.pi * (0.85 * rx + 0.15))
                        * math.cos(0.75 * math.pi * ry)
                    )
                    middle_z = TISSUE_CENTER[2] + dome + asymmetric_curve + broad_fold
                    thickness = (
                        0.00315
                        + 0.00075 * max(0.0, 1.0 - radial_clamped**2)
                        + 0.00040
                        * (0.5 + 0.5 * math.sin(2.0 * math.pi * u + 0.7))
                        * (0.35 + 0.65 * max(0.0, 1.0 - abs(ry)))
                    )
                    z = middle_z + (0.5 if top else -0.5) * thickness
                elif profile == "rectangular_tissue_strip":
                    # 明确的长方形组织薄层：长轴沿 x，夹取发生在 y 方向的长边。
                    # 仅对轮廓和中层加入低频自然起伏，不将其做成正方形或刚性平板。
                    rx, ry = rounded_coordinate(sx, sy)
                    local_x = 0.0500 * (
                        rx + 0.018 * math.sin(math.pi * v) * (1.0 - sx * sx)
                    )
                    local_y = 0.0290 * (
                        ry + 0.020 * math.sin(2.0 * math.pi * u) * (1.0 - sy * sy)
                    )
                    angle = math.radians(-3.0)
                    shaped_x = math.cos(angle) * local_x - math.sin(angle) * local_y
                    shaped_y = math.sin(angle) * local_x + math.cos(angle) * local_y
                    dome = (
                        0.00048
                        * max(0.0, 1.0 - sx * sx)
                        * max(0.0, 1.0 - sy * sy)
                    )
                    broad_fold = (
                        0.00018
                        * math.sin(2.0 * math.pi * u + 0.35)
                        * math.cos(math.pi * v)
                    )
                    thickness = 0.00110 * (
                        1.0
                        + 0.06
                        * math.sin(2.0 * math.pi * u)
                        * math.sin(math.pi * v)
                    )
                    middle_z = TISSUE_CENTER[2] + dome + broad_fold
                    z = middle_z + (0.5 if top else -0.5) * thickness
                else:
                    rx, ry = rounded_coordinate(sx, sy)
                    rx += 0.070 * ry + 0.022 * math.sin(math.pi * v)
                    ry += 0.040 * math.sin(math.pi * u) - 0.018 * u
                    angle = math.radians(-6.0)
                    local_x = TISSUE_HALF_X * rx
                    local_y = TISSUE_HALF_Y * ry
                    shaped_x = math.cos(angle) * local_x - math.sin(angle) * local_y
                    shaped_y = math.sin(angle) * local_x + math.cos(angle) * local_y
                    dome = 0.00042 * max(0.0, 1.0 - sx * sx) * max(0.0, 1.0 - sy * sy)
                    wrinkle = 0.00024 * math.sin(2.0 * math.pi * u) * math.sin(math.pi * v)
                    z = TISSUE_CENTER[2] + (0.5 if top else -0.5) * TISSUE_THICKNESS
                    z += (dome + wrinkle) if top else 0.20 * (dome + wrinkle)
                points.append(
                    (
                        float(TISSUE_CENTER[0] + shaped_x),
                        float(TISSUE_CENTER[1] + shaped_y),
                        float(z),
                    )
                )
                if profile == "irregular_liver_lobe":
                    # 取官方 Liver atlas 中央、无放射拉伸的实质纹理区域。
                    uvs.append((0.22 + 0.56 * u, 0.18 + 0.64 * v))
                else:
                    # 取官方 Body atlas 中央的天然斑驳区域；相对最初版本将窗口
                    # 缩到约 50%，使同一官方纹理的空间尺度放大约 1.9 倍。
                    uvs.append((0.18 + 0.16 * u, 0.325 + 0.29 * v))
                material_uv.append((u, v))
        return offset

    top_offset = add_surface(True)
    bottom_offset = add_surface(False)

    def vertex(offset: int, row: int, column: int) -> int:
        return offset + row * TISSUE_COLUMNS + column

    faces: list[tuple[int, int, int]] = []
    top_face_indices: list[int] = []
    for row in range(TISSUE_ROWS - 1):
        for column in range(TISSUE_COLUMNS - 1):
            a = vertex(top_offset, row, column)
            b = vertex(top_offset, row, column + 1)
            c = vertex(top_offset, row + 1, column + 1)
            d = vertex(top_offset, row + 1, column)
            top_face_indices.extend((len(faces), len(faces) + 1))
            if (row + column) % 2 == 0:
                faces.extend(((a, b, c), (a, c, d)))
            else:
                faces.extend(((a, b, d), (b, c, d)))

    for row in range(TISSUE_ROWS - 1):
        for column in range(TISSUE_COLUMNS - 1):
            a = vertex(bottom_offset, row, column)
            b = vertex(bottom_offset, row, column + 1)
            c = vertex(bottom_offset, row + 1, column + 1)
            d = vertex(bottom_offset, row + 1, column)
            if (row + column) % 2 == 0:
                faces.extend(((a, c, b), (a, d, c)))
            else:
                faces.extend(((a, d, b), (b, d, c)))

    boundary: list[tuple[int, int]] = []
    boundary.extend((0, column) for column in range(TISSUE_COLUMNS))
    boundary.extend((row, TISSUE_COLUMNS - 1) for row in range(1, TISSUE_ROWS))
    boundary.extend(
        (TISSUE_ROWS - 1, column)
        for column in range(TISSUE_COLUMNS - 2, -1, -1)
    )
    boundary.extend((row, 0) for row in range(TISSUE_ROWS - 2, 0, -1))
    for index, (row_a, column_a) in enumerate(boundary):
        row_b, column_b = boundary[(index + 1) % len(boundary)]
        top_a = vertex(top_offset, row_a, column_a)
        top_b = vertex(top_offset, row_b, column_b)
        bottom_a = vertex(bottom_offset, row_a, column_a)
        bottom_b = vertex(bottom_offset, row_b, column_b)
        faces.extend(((top_a, bottom_a, bottom_b), (top_a, bottom_b, top_b)))

    # 每个薄六面体沿同一体对角线分成 6 个四面体，形成一张连续体网格。
    tets: list[list[int]] = []
    for row in range(TISSUE_ROWS - 1):
        for column in range(TISSUE_COLUMNS - 1):
            v000 = vertex(bottom_offset, row, column)
            v100 = vertex(bottom_offset, row, column + 1)
            v110 = vertex(bottom_offset, row + 1, column + 1)
            v010 = vertex(bottom_offset, row + 1, column)
            v001 = vertex(top_offset, row, column)
            v101 = vertex(top_offset, row, column + 1)
            v111 = vertex(top_offset, row + 1, column + 1)
            v011 = vertex(top_offset, row + 1, column)
            tets.extend(
                (
                    [v000, v100, v110, v111],
                    [v000, v110, v010, v111],
                    [v000, v010, v011, v111],
                    [v000, v011, v001, v111],
                    [v000, v001, v101, v111],
                    [v000, v101, v100, v111],
                )
            )

    point_array = np.asarray(points, dtype=np.float64)
    tet_array = np.asarray(tets, dtype=np.int32)
    for tet in tet_array:
        tet_points = point_array[tet]
        determinant = np.linalg.det(
            np.stack(
                (
                    tet_points[1] - tet_points[0],
                    tet_points[2] - tet_points[0],
                    tet_points[3] - tet_points[0],
                ),
                axis=-1,
            )
        )
        if determinant < 0.0:
            tet[1], tet[2] = tet[2], tet[1]
    signed_six_volumes = np.linalg.det(
        np.stack(
            (
                point_array[tet_array[:, 1]] - point_array[tet_array[:, 0]],
                point_array[tet_array[:, 2]] - point_array[tet_array[:, 0]],
                point_array[tet_array[:, 3]] - point_array[tet_array[:, 0]],
            ),
            axis=-1,
        )
    )
    if np.any(signed_six_volumes <= 1.0e-15):
        raise RuntimeError(
            "连续薄片包含退化四面体："
            f"min(6V)={signed_six_volumes.min():.3e}"
        )

    return {
        "rest_positions": point_array.astype(np.float32),
        "visual_uvs": np.asarray(uvs, dtype=np.float32),
        "material_uv": np.asarray(material_uv, dtype=np.float32),
        "visual_faces": np.asarray(faces, dtype=np.int32),
        "top_face_indices": np.asarray(top_face_indices, dtype=np.int32),
        "tet_indices": tet_array,
        "tet_rest_volumes_m3": (signed_six_volumes / 6.0).astype(np.float32),
        "geometry_profile": np.asarray(profile),
        "visual_material_profile": np.asarray(
            "official_liver" if profile == "irregular_liver_lobe" else "official_body_fascia"
        ),
        "visual_uv_center": np.asarray(
            (0.50, 0.50) if profile == "irregular_liver_lobe" else (0.26, 0.47),
            dtype=np.float32,
        ),
    }


def sample_region_centers(rng: np.random.Generator) -> np.ndarray:
    """在 UV 域中采样十个分散的随机 Voronoi 中心。"""

    centers: list[np.ndarray] = []
    minimum_distance = 0.19
    for _ in range(20000):
        candidate = rng.uniform((0.07, 0.07), (0.93, 0.93))
        if not centers or min(np.linalg.norm(candidate - center) for center in centers) >= minimum_distance:
            centers.append(candidate)
            if len(centers) == 10:
                return np.asarray(centers, dtype=np.float32)
        if len(centers) < 10 and _ == 9999:
            minimum_distance = 0.15
    raise RuntimeError("无法生成十个非退化随机材料区域")


def assign_regions(
    geometry: dict[str, np.ndarray], seed: int, youngs_min: float, youngs_max: float
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = sample_region_centers(rng)
    tet_uv = geometry["material_uv"][geometry["tet_indices"]].mean(axis=1)
    node_uv = geometry["material_uv"]
    # 轻微各向异性的度量让区域轮廓不呈规则蜂窝状，同时保持完整覆盖。
    metric_scale = rng.uniform(0.82, 1.18, size=(10, 2)).astype(np.float32)
    tet_distance = np.sum(
        ((tet_uv[:, None, :] - centers[None, :, :]) * metric_scale[None, :, :]) ** 2,
        axis=2,
    )
    node_distance = np.sum(
        ((node_uv[:, None, :] - centers[None, :, :]) * metric_scale[None, :, :]) ** 2,
        axis=2,
    )
    tet_region_ids = np.argmin(tet_distance, axis=1).astype(np.int16)
    node_region_ids = np.argmin(node_distance, axis=1).astype(np.int16)
    counts = np.bincount(tet_region_ids, minlength=10)
    if np.any(counts == 0):
        raise RuntimeError(f"随机区域没有覆盖四面体：counts={counts.tolist()}")

    # 分层对数采样确保 10 个区域彼此可辨，同时随机打乱软硬空间位置。
    log_min, log_max = math.log(youngs_min), math.log(youngs_max)
    bins = np.linspace(log_min, log_max, 11)
    values = np.exp(rng.uniform(bins[:-1], bins[1:])).astype(np.float32)
    regional_youngs = values[rng.permutation(10)]
    return {
        "region_centers_uv": centers,
        "region_metric_scale": metric_scale,
        "tet_region_ids": tet_region_ids,
        "node_region_ids": node_region_ids,
        "region_tet_counts": counts.astype(np.int32),
        "regional_youngs_modulus_pa": regional_youngs,
    }


def lame_parameters(youngs: float, poisson: float) -> tuple[float, float]:
    mu = youngs / (2.0 * (1.0 + poisson))
    lame_lambda = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return float(mu), float(lame_lambda)


def grasp_displacement(
    progress: float,
    lift: np.ndarray,
    pull: np.ndarray,
    motion_profile: str,
) -> tuple[bool, np.ndarray]:
    """与 PSM 轨迹同相位的已知抓取边界位移。"""

    def smoothstep(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

    if progress < 0.34:
        return False, np.zeros(3, dtype=np.float32)
    if progress < 0.40:
        return True, np.zeros(3, dtype=np.float32)
    if progress < 0.68:
        alpha = smoothstep((progress - 0.40) / 0.28)
        return True, alpha * lift
    if progress < 0.80:
        alpha = smoothstep((progress - 0.68) / 0.12)
        return True, (1.0 - alpha) * lift + alpha * pull
    if progress < 0.92:
        return True, pull
    return False, pull


def build_model(
    geometry: dict[str, np.ndarray], regions: dict[str, np.ndarray], poisson: float, device: str
):
    positions = geometry["rest_positions"].astype(np.float64)
    tet_indices = geometry["tet_indices"]
    tet_volumes = geometry["tet_rest_volumes_m3"].astype(np.float64)
    density = 1000.0
    masses = np.zeros(len(positions), dtype=np.float64)
    np.add.at(masses, tet_indices.reshape(-1), np.repeat(density * tet_volumes / 4.0, 4))
    if np.any(masses <= 0.0):
        raise RuntimeError("连续组织包含零质量节点")

    builder = warp.sim.ModelBuilder(gravity=0.0)
    for position, mass in zip(positions, masses):
        builder.add_particle(position, (0.0, 0.0, 0.0), float(mass), radius=0.0)
    regional_youngs = regions["regional_youngs_modulus_pa"]
    for tet, region_id in zip(tet_indices, regions["tet_region_ids"]):
        mu, lame_lambda = lame_parameters(float(regional_youngs[region_id]), poisson)
        volume = builder.add_tetrahedron(
            int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3]), mu, lame_lambda, 0.0
        )
        if volume <= 0.0:
            raise RuntimeError(f"Warp 拒绝四面体，体积={volume}")
    return builder.finalize(device=device), masses.astype(np.float32)


def simulate(args: argparse.Namespace) -> dict[str, np.ndarray]:
    geometry = build_geometry(args.geometry_profile)
    regions = assign_regions(
        geometry,
        args.regional_seed,
        args.regional_youngs_min_pa,
        args.regional_youngs_max_pa,
    )
    model, masses = build_model(geometry, regions, args.poissons_ratio, args.device)
    state_0 = model.state()
    state_1 = model.state()
    projector = MaterialTetrahedronXPBDProjector(
        model,
        iterations=args.material_iterations,
        relaxation=0.16,
        compliance_scale=1.0,
        constraint_model="neo_hookean",
    )

    rest_cpu = geometry["rest_positions"]
    rest = torch.as_tensor(rest_cpu, device=args.device)
    x_min, x_max = float(rest_cpu[:, 0].min()), float(rest_cpu[:, 0].max())
    y_center = float(np.median(rest_cpu[:, 1]))
    if args.support_mode == "free":
        # 新基准不把图像中的某一角或某一整条边设成静止真值。零重力下组织在
        # 夹持前自然保持静止，闭合后只由小片区位移边界和材料内力决定运动。
        anchor_mask = np.zeros(len(rest_cpu), dtype=bool)
    else:
        anchor_mask = (
            rest_cpu[:, 0] < x_min + 0.006
            if args.anchor_side == "left"
            else rest_cpu[:, 0] > x_max - 0.006
        )
    # 只有红色抓取标记附近约 8×8 mm 的核心严格跟随夹尖；核心外围使用连续
    # 余弦权重施加柔顺力，而不是把一块矩形组织直接设成刚体。这样材料仍决定
    # 最终形变，耦合也不会在某条网格边上从 1 突然跳到 0。
    grasp_offset_mm = np.asarray(
        args.grasp_offset_mm
        if args.grasp_offset_mm is not None
        else (args.grasp_offset_x_mm, 0.0),
        dtype=np.float32,
    )
    grasp_center = TISSUE_CENTER[:2].astype(np.float32) + grasp_offset_mm * 1.0e-3
    grasp_dx = rest_cpu[:, 0] - grasp_center[0]
    grasp_dy = rest_cpu[:, 1] - grasp_center[1]
    grasp_mask = (np.abs(grasp_dx) <= 0.004) & (np.abs(grasp_dy) <= 0.004)
    if args.geometry_profile == "irregular_liver_lobe":
        coupling_radius_x, coupling_radius_y = 0.030, 0.024
    elif args.geometry_profile == "rectangular_tissue_strip":
        coupling_radius_x, coupling_radius_y = 0.040, 0.022
    else:
        coupling_radius_x, coupling_radius_y = 0.034, 0.030
    normalized_radius = np.sqrt(
        (grasp_dx / coupling_radius_x) ** 2
        + (grasp_dy / coupling_radius_y) ** 2
    )
    grasp_coupling_weights = np.zeros(len(rest_cpu), dtype=np.float32)
    coupling_support = normalized_radius < 1.0
    grasp_coupling_weights[coupling_support] = 0.5 * (
        1.0 + np.cos(np.pi * normalized_radius[coupling_support])
    )
    grasp_coupling_weights[grasp_mask] = 1.0
    if not grasp_mask.any():
        raise RuntimeError("连续组织抓取点集合为空")
    anchor_ids = torch.as_tensor(np.flatnonzero(anchor_mask), device=args.device, dtype=torch.long)
    grasp_ids = torch.as_tensor(np.flatnonzero(grasp_mask), device=args.device, dtype=torch.long)
    coupling_weights = torch.as_tensor(
        grasp_coupling_weights, device=args.device, dtype=rest.dtype
    )
    particle_masses = torch.as_tensor(masses, device=args.device, dtype=rest.dtype)
    coupling_omega = 2.0 * math.pi * args.grasp_coupling_frequency_hz
    coupling_damping = 2.0 * 0.90 * coupling_omega
    inverse_mass = wp.to_torch(model.particle_inv_mass)
    base_inverse_mass = inverse_mass.detach().clone()
    if len(anchor_ids):
        inverse_mass[anchor_ids] = 0.0

    timestamps = np.arange(args.frames, dtype=np.float64) / args.fps
    duration = float(timestamps[-1])
    substeps = args.physics_hz // args.fps
    dt = 1.0 / args.physics_hz
    velocity_damping = math.exp(-12.0 * dt)
    positions_history: list[np.ndarray] = []
    velocities_history: list[np.ndarray] = []
    lift_displacement = np.asarray(
        args.lift_displacement_mm, dtype=np.float32
    ) * 1.0e-3
    pull_displacement = np.asarray(
        args.pull_displacement_mm, dtype=np.float32
    ) * 1.0e-3
    previous_grasp_displacement = np.zeros(3, dtype=np.float32)

    for frame_index, timestamp in enumerate(timestamps):
        if frame_index > 0:
            previous_timestamp = float(timestamps[frame_index - 1])
            for substep_index in range(substeps):
                alpha = (substep_index + 1) / substeps
                physics_time = previous_timestamp + alpha * (float(timestamp) - previous_timestamp)
                progress = physics_time / duration
                grasped, displacement = grasp_displacement(
                    progress,
                    lift_displacement,
                    pull_displacement,
                    args.motion_profile,
                )
                positions = wp.to_torch(state_0.particle_q)
                velocities = wp.to_torch(state_0.particle_qd)
                forces = wp.to_torch(state_0.particle_f)
                forces.zero_()
                positions[anchor_ids] = rest[anchor_ids]
                velocities[anchor_ids] = 0.0
                if grasped:
                    displacement_tensor = torch.as_tensor(
                        displacement, device=args.device, dtype=rest.dtype
                    )
                    displacement_velocity = torch.as_tensor(
                        (displacement - previous_grasp_displacement) / dt,
                        device=args.device,
                        dtype=rest.dtype,
                    )
                    target_positions = (
                        rest + coupling_weights[:, None] * displacement_tensor
                    )
                    target_velocities = (
                        coupling_weights[:, None] * displacement_velocity
                    )
                    coupling_acceleration = coupling_weights[:, None] * (
                        coupling_omega * coupling_omega
                        * (target_positions - positions)
                        + coupling_damping * (target_velocities - velocities)
                    )
                    forces += particle_masses[:, None] * coupling_acceleration
                    inverse_mass[grasp_ids] = 0.0
                    positions[grasp_ids] = rest[grasp_ids] + displacement_tensor
                    velocities[grasp_ids] = displacement_velocity
                else:
                    inverse_mass[grasp_ids] = base_inverse_mass[grasp_ids]
                projector.simulate_unconstrained_particles(
                    model,
                    state_0,
                    state_1,
                    dt,
                    velocity_damping=velocity_damping,
                    material_min_volume_ratio=0.02,
                )
                next_positions = wp.to_torch(state_1.particle_q)
                next_velocities = wp.to_torch(state_1.particle_qd)
                next_positions[anchor_ids] = rest[anchor_ids]
                next_velocities[anchor_ids] = 0.0
                if grasped:
                    next_positions[grasp_ids] = rest[grasp_ids] + torch.as_tensor(
                        displacement, device=args.device
                    )
                    next_velocities[grasp_ids] = torch.as_tensor(
                        (displacement - previous_grasp_displacement) / dt,
                        device=args.device,
                    )
                state_0, state_1 = state_1, state_0
                previous_grasp_displacement = displacement.copy()

        positions_history.append(
            wp.to_torch(state_0.particle_q).detach().cpu().numpy().astype(np.float32).copy()
        )
        velocities_history.append(
            wp.to_torch(state_0.particle_qd).detach().cpu().numpy().astype(np.float32).copy()
        )
        if frame_index == 0 or (frame_index + 1) % args.fps == 0 or frame_index + 1 == args.frames:
            print(
                f"[十区域组织预计算] {frame_index + 1:04d}/{args.frames:04d} "
                f"t={timestamp:6.3f}s",
                flush=True,
            )

    result = {
        **geometry,
        **regions,
        "timestamps": timestamps,
        "visual_positions": np.asarray(positions_history, dtype=np.float32),
        "visual_velocities": np.asarray(velocities_history, dtype=np.float32),
        "particle_mass_kg": masses,
        "anchor_mask": anchor_mask.astype(bool),
        "grasp_mask": grasp_mask.astype(bool),
        "grasp_coupling_weights": grasp_coupling_weights,
        "grasp_coupling_frequency_hz": np.asarray(
            args.grasp_coupling_frequency_hz, dtype=np.float32
        ),
        "lift_displacement_m": lift_displacement,
        "pull_displacement_m": pull_displacement,
        "grasp_offset_m": (grasp_offset_mm * 1.0e-3).astype(np.float32),
        "grasp_offset_x_m": np.asarray(grasp_offset_mm[0] * 1.0e-3, dtype=np.float32),
        "support_mode": np.asarray(args.support_mode),
        "motion_profile": np.asarray(args.motion_profile),
        "anchor_side": np.asarray(args.anchor_side),
        "poissons_ratio": np.asarray(args.poissons_ratio, dtype=np.float32),
        "regional_seed": np.asarray(args.regional_seed, dtype=np.int64),
        "physics_hz": np.asarray(args.physics_hz, dtype=np.int32),
        "fps": np.asarray(args.fps, dtype=np.int32),
        "solver_material_iterations": np.asarray(args.material_iterations, dtype=np.int32),
    }
    if not np.isfinite(result["visual_positions"]).all():
        raise RuntimeError("十区域组织预计算产生非有限坐标")
    maximum_displacement = np.linalg.norm(
        result["visual_positions"] - result["rest_positions"][None, ...], axis=2
    ).max()
    print(
        "[十区域组织预计算] "
        f"nodes={len(rest_cpu)}, tets={len(geometry['tet_indices'])}, "
        f"max_displacement={maximum_displacement * 1e3:.2f} mm",
        flush=True,
    )
    return result


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = simulate(args)
    np.savez_compressed(output, **result)
    manifest = {
        "schema": "fixedsuperbest.continuous_tissue_trajectory.v1",
        "regions": 10,
        "regional_seed": args.regional_seed,
        "regional_youngs_modulus_pa": result["regional_youngs_modulus_pa"].tolist(),
        "region_tet_counts": result["region_tet_counts"].tolist(),
        "nodes": int(len(result["rest_positions"])),
        "tetrahedra": int(len(result["tet_indices"])),
        "frames": args.frames,
        "fps": args.fps,
        "physics_hz": args.physics_hz,
        "constitutive_model": "compressible Neo-Hookean XPBD",
        "geometry_profile": args.geometry_profile,
        "motion_profile": args.motion_profile,
        "visual_material_profile": str(
            np.asarray(result["visual_material_profile"]).item()
        ),
        "support_mode": args.support_mode,
        "hard_anchor_nodes": int(result["anchor_mask"].sum()),
        "grasp_offset_mm": (
            np.asarray(result["grasp_offset_m"], dtype=np.float64) * 1.0e3
        ).tolist(),
        "note_zh": "十区域参数是可复现的合成真值，不是人体材料实测值。",
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[完成] 连续十区域组织轨迹：{output}", flush=True)


if __name__ == "__main__":
    main()
