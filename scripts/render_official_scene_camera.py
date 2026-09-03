#!/usr/bin/env python3
"""直接从官方 main_scene.usd 的已有相机渲染原生材质基准图。"""

from __future__ import annotations

import argparse
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd", type=Path, required=True)
parser.add_argument("--camera", default="/CameraTop")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--renderer", choices=("RayTracedLighting", "PathTracing"), default="PathTracing")
parser.add_argument("--samples-per-pixel", type=int, default=32)
args = parser.parse_args()

from omni.isaac.lab.app import AppLauncher


app_launcher = AppLauncher(
    {
        "headless": True,
        "enable_cameras": True,
        "renderer": args.renderer,
        "samples_per_pixel_per_frame": args.samples_per_pixel,
        "width": args.width,
        "height": args.height,
        "multi_gpu": False,
        "sync_loads": True,
    }
)
simulation_app = app_launcher.app

import carb
import numpy as np
import omni.usd
from omni.isaac.core.utils.extensions import enable_extension
from PIL import Image


enable_extension("omni.isaac.sensor")
simulation_app.update()
from omni.isaac.sensor import Camera


def main() -> None:
    usd_path = args.usd.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    settings = carb.settings.get_settings()
    settings.set_bool("/rtx/post/motionblur/enabled", False)
    settings.set_bool("/rtx/post/tvNoise/enabled", False)
    if not omni.usd.get_context().open_stage(str(usd_path)):
        raise RuntimeError(f"无法打开官方场景：{usd_path}")
    for _ in range(24):
        simulation_app.update()
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(args.camera).IsValid():
        raise RuntimeError(f"官方相机不存在：{args.camera}")
    camera = Camera(
        prim_path=args.camera,
        name="official_reference_camera",
        resolution=(args.width, args.height),
    )
    camera.initialize()
    for _ in range(32):
        simulation_app.update()
    rgb = np.asarray(camera.get_rgb())
    if np.issubdtype(rgb.dtype, np.floating) and rgb.size and float(np.nanmax(rgb)) <= 1.0:
        rgb = rgb * 255.0
    rgb = np.clip(rgb[..., :3], 0, 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(output_path)
    print(f"[官方基准图] {output_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
