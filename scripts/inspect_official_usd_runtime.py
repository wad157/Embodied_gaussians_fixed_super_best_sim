#!/usr/bin/env python3
"""在 Isaac Sim 运行时只读检查官方 USD 的层级、相机、包围盒和材质绑定。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd", type=Path, required=True)
parser.add_argument("--max-depth", type=int, default=3)
parser.add_argument("--pattern", default="", help="仅输出路径匹配该正则表达式的 prim。")
args = parser.parse_args()

from omni.isaac.lab.app import AppLauncher


app_launcher = AppLauncher({"headless": True, "enable_cameras": False})
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdShade


def main() -> None:
    usd_path = args.usd.expanduser().resolve()
    stage = Usd.Stage.Open(str(usd_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"无法打开 USD：{usd_path}")
    default_prim = stage.GetDefaultPrim()
    print(f"stage={usd_path}", flush=True)
    print(f"default_prim={default_prim.GetPath() if default_prim else '<none>'}", flush=True)
    print(f"meters_per_unit={UsdGeom.GetStageMetersPerUnit(stage)}", flush=True)
    print(f"up_axis={UsdGeom.GetStageUpAxis(stage)}", flush=True)

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    matcher = re.compile(args.pattern, re.IGNORECASE) if args.pattern else None
    for prim in stage.Traverse():
        path = prim.GetPath()
        depth = path.pathString.count("/") - 1
        if depth > args.max_depth and not prim.IsA(UsdGeom.Camera):
            continue
        if matcher is not None and not matcher.search(path.pathString):
            continue
        type_name = prim.GetTypeName() or "<untyped>"
        details = []
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            details.append(f"visibility={imageable.ComputeVisibility()}")
        if prim.IsA(UsdGeom.Boundable) or depth <= 2:
            try:
                extent = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
                details.append(f"bbox_min={tuple(round(float(v), 5) for v in extent.GetMin())}")
                details.append(f"bbox_max={tuple(round(float(v), 5) for v in extent.GetMax())}")
            except Exception as error:
                details.append(f"bbox_error={error!r}")
        if prim.IsA(UsdGeom.Mesh):
            material, relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            details.append(f"material={material.GetPath() if material else '<none>'}")
            details.append(f"binding={relationship.GetPath() if relationship else '<none>'}")
            primvars = []
            for value in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
                payload = value.Get()
                sample = None
                if payload is not None and hasattr(payload, "__len__") and len(payload):
                    sample = payload[0]
                primvars.append(
                    f"{value.GetPrimvarName()}:{value.GetTypeName()}:{value.GetInterpolation()}:sample={sample}"
                )
            details.append(f"primvars={primvars}")
        if prim.IsA(UsdGeom.Camera):
            camera = UsdGeom.Camera(prim)
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            details.append(f"focal_length={camera.GetFocalLengthAttr().Get()}")
            details.append(f"horizontal_aperture={camera.GetHorizontalApertureAttr().Get()}")
            details.append(f"X_WC={[[round(float(matrix[row][column]), 5) for column in range(4)] for row in range(4)]}")
        print(f"{type_name:12s} {path} {' '.join(details)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
