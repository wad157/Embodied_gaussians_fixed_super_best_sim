#!/usr/bin/env python3
"""从数据集使用的官方 ORBIT-Surgical PSM USD 导出末端 GUI 网格。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--usd",
        type=Path,
        default=Path("data/sim_assets/PSM/psm_col.usd"),
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


ARGS = parse_args()

from omni.isaac.lab.app import AppLauncher


app_launcher = AppLauncher(
    {
        "headless": ARGS.headless,
        "enable_cameras": False,
        "multi_gpu": False,
        "sync_loads": True,
    }
)
simulation_app = app_launcher.app

import omni.usd
from omni.isaac.core import World
from omni.isaac.lab.assets import Articulation
from pxr import Gf, Usd, UsdGeom

from orbit.surgical.assets.psm import PSM_HIGH_PD_CFG


TIP_LINKS = (
    "psm_tool_roll_link",
    "psm_tool_pitch_link",
    "psm_tool_yaw_link",
    "psm_tool_gripper1_link",
    "psm_tool_gripper2_link",
)


def triangulate(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    cursor = 0
    for count in counts.tolist():
        face = indices[cursor : cursor + count]
        cursor += count
        if count < 3:
            continue
        triangles.extend(
            (int(face[0]), int(face[i]), int(face[i + 1]))
            for i in range(1, count - 1)
        )
    return np.asarray(triangles, dtype=np.int32).reshape(-1, 3)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float64)
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(norms, 1.0e-12)
    return normals.astype(np.float32)


def nearest_link_prim(
    prim: Usd.Prim, valid_names: set[str]
) -> Usd.Prim | None:
    current = prim
    while current and not current.IsPseudoRoot():
        if current.GetName() in valid_names:
            return current
        current = current.GetParent()
    return None


def main() -> None:
    output = ARGS.output.resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有官方 PSM GUI 网格：{output}")
    usd = ARGS.usd.resolve()
    if not usd.is_file():
        raise FileNotFoundError(usd)

    world = World(
        physics_dt=1.0 / 120.0,
        rendering_dt=1.0 / 30.0,
        stage_units_in_meters=1.0,
        backend="torch",
        device="cuda",
    )
    config = PSM_HIGH_PD_CFG.replace(prim_path="/World/PSM")
    config.spawn.usd_path = str(usd)
    config.init_state.pos = (0.0, 0.0, 0.21)
    robot = Articulation(config)
    world.reset(soft=False)
    robot.update(world.get_physics_dt())

    body_names = list(robot.body_names)
    missing = sorted(set(TIP_LINKS).difference(body_names))
    if missing:
        raise KeyError(f"官方 PSM 缺少末端 link：{missing}")
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    vertices_by_link: dict[str, list[np.ndarray]] = {name: [] for name in TIP_LINKS}
    faces_by_link: dict[str, list[np.ndarray]] = {name: [] for name in TIP_LINKS}
    vertex_counts = {name: 0 for name in TIP_LINKS}
    predicate = Usd.TraverseInstanceProxies()
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), predicate):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = str(prim.GetPath())
        if not path.startswith("/World/PSM/"):
            continue
        lower_path = path.lower()
        if "collision" in lower_path or "/collisions/" in lower_path:
            continue
        link_prim = nearest_link_prim(prim, set(TIP_LINKS))
        if link_prim is None:
            continue
        link_name = link_prim.GetName()
        mesh = UsdGeom.Mesh(prim)
        local_points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        if len(local_points) == 0 or len(indices) == 0:
            continue
        mesh_X_world = cache.GetLocalToWorldTransform(prim)
        # Use the USD hierarchy on both sides. Mixing physics body poses with
        # unsynchronised stage Xforms flips the distal-link convention and
        # places the exported jaws centimetres away from the recorded tip.
        world_X_link = cache.GetLocalToWorldTransform(link_prim)
        link_X_world = world_X_link.GetInverse()
        link_points = np.asarray(
            [
                link_X_world.Transform(
                    mesh_X_world.Transform(Gf.Vec3d(*point))
                )
                for point in local_points
            ],
            dtype=np.float64,
        )
        faces = triangulate(counts, indices) + vertex_counts[link_name]
        vertices_by_link[link_name].append(link_points.astype(np.float32))
        faces_by_link[link_name].append(faces)
        vertex_counts[link_name] += len(link_points)

    payload: dict[str, np.ndarray] = {
        "schema": np.asarray("fixedsuperbest.official_psm_tip_meshes.v1"),
        "link_names": np.asarray(TIP_LINKS),
        "source_usd": np.asarray(str(usd)),
    }
    report_links: dict[str, dict[str, int | list[list[float]]]] = {}
    for name in TIP_LINKS:
        if not vertices_by_link[name]:
            raise RuntimeError(f"官方 PSM link 没有可见网格：{name}")
        vertices = np.concatenate(vertices_by_link[name], axis=0)
        faces = np.concatenate(faces_by_link[name], axis=0)
        normals = vertex_normals(vertices, faces)
        payload[f"{name}__vertices"] = vertices
        payload[f"{name}__faces"] = faces
        payload[f"{name}__normals"] = normals
        report_links[name] = {
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
            "bounds_m": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    report = {
        "schema": "fixedsuperbest.official_psm_tip_meshes.v1",
        "source_usd": str(usd),
        "links": report_links,
        "passed": True,
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


try:
    main()
finally:
    simulation_app.close()
