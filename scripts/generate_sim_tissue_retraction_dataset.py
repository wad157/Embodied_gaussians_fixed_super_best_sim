#!/usr/bin/env python3
"""使用官方 PSM 与写实肝脏资产生成组织牵拉数据集。

重建输入与仿真真值严格分开：GUI/重建只读取根目录下的视频、相机和
机器人状态；组织节点、材料参数、深度和语义 mask 位于 ``ground_truth``。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--physics-hz", type=int, default=120)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--youngs-modulus",
        type=float,
        default=900.0,
        help="合成基准的中间区域杨氏模量（Pa）；默认值用于柔软预览，不代表人体实测值。",
    )
    parser.add_argument("--regional-seed", type=int, default=240610788)
    parser.add_argument("--regional-youngs-min-pa", type=float, default=450.0)
    parser.add_argument("--regional-youngs-max-pa", type=float, default=1450.0)
    parser.add_argument(
        "--tissue-trajectory",
        type=Path,
        default=None,
        help="连续十区域组织预计算 NPZ；sufia_viewpoint_train 场景必须提供。",
    )
    parser.add_argument("--poissons-ratio", type=float, default=0.45)
    parser.add_argument("--damping-scale", type=float, default=0.10)
    parser.add_argument("--elasticity-damping", type=float, default=0.05)
    parser.add_argument(
        "--scene-style",
        choices=("legacy", "sufia_viewpoint_train"),
        default="legacy",
        help="legacy 保留旧场景；sufia_viewpoint_train 复现 SuFIA-BC Viewpoint Robustness 第一段的完整腹部构图。",
    )
    parser.add_argument(
        "--task-variant",
        choices=("front_pull", "side_pull", "edge_lift_return"),
        default="front_pull",
        help="front_pull 为正面面内牵拉；side_pull 为从组织侧边夹起并横向牵拉。",
    )
    parser.add_argument(
        "--motion-profile",
        choices=("retract_hold", "edge_lift_return"),
        default="retract_hold",
        help="edge_lift_return 表示从长边夹起、抬升、放回并释放。",
    )
    parser.add_argument(
        "--material-provenance",
        default="preview_only_unvalidated_not_for_stiffness_evaluation",
        help="材料参数来源/标定编号；会原样写入 ground_truth/material.json。",
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--renderer",
        choices=("RayTracedLighting", "PathTracing"),
        default="PathTracing",
    )
    parser.add_argument("--samples-per-pixel", type=int, default=64)
    parser.add_argument(
        "--canonical-scan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="在任务开始前记录隔离组织的全视角 RGB-D、mask 和相机标定。",
    )
    parser.add_argument(
        "--canonical-scan-azimuths",
        type=int,
        default=12,
        help="每个非极点仰角的等间隔方位角数量；默认 12。",
    )
    parser.add_argument(
        "--canonical-scan-elevations-deg",
        type=float,
        nargs="+",
        default=(-60.0, -25.0, 25.0, 60.0),
        help="全视角扫描的仰角列表（度）；默认覆盖上、下表面和侧缘。",
    )
    parser.add_argument(
        "--canonical-scan-radius-m",
        type=float,
        default=0.13,
        help="全视角扫描相机到静止组织中心的距离（m）。",
    )
    parser.add_argument(
        "--task-camera-eye",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="任务双目中心相机的世界坐标；左右目在 x 方向各偏移 4 mm。",
    )
    parser.add_argument(
        "--task-camera-target",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="任务相机共同注视点的世界坐标。",
    )
    parser.add_argument(
        "--tissue-uv-detail-scale",
        type=float,
        default=1.0,
        help="相对当前官方纹理 UV 窗口的比例；小于 1 会放大天然纹理。",
    )
    parser.add_argument(
        "--tissue-texture-contrast",
        type=float,
        default=1.55,
        help="官方 Body albedo 的等通道对比度增益。",
    )
    parser.add_argument(
        "--tissue-texture-bias",
        type=float,
        nargs=3,
        default=(-0.445, -0.305, -0.190),
        metavar=("R", "G", "B"),
        help="官方 Body albedo 的 RGB bias。",
    )
    parser.add_argument(
        "--psm-lift-q7",
        type=float,
        nargs=7,
        default=(0.345, 0.0, 0.134, 0.0, 0.0, 0.0, 0.14),
        help="第一牵拉阶段的 PSM q7。",
    )
    parser.add_argument(
        "--psm-contact-q7",
        type=float,
        nargs=7,
        default=(0.29, 0.0, 0.158, 0.0, 0.0, 0.0, 0.96),
        help="接触组织时、夹爪张开的 PSM q7。",
    )
    parser.add_argument(
        "--psm-ready-q7",
        type=float,
        nargs=7,
        default=(0.37, 0.0, 0.123, 0.0, 0.0, 0.0, 0.96),
        help="接近前/释放后的 PSM q7。",
    )
    parser.add_argument(
        "--psm-pull-q7",
        type=float,
        nargs=7,
        default=(0.400, 0.0, 0.134, 0.0, 0.0, 0.0, 0.14),
        help="最终牵拉阶段的 PSM q7。",
    )
    parser.add_argument(
        "--preview-frame-index",
        type=int,
        default=None,
        help="只渲染指定代表帧并退出，不生成完整数据集。",
    )
    parser.add_argument(
        "--grasp-offset-x-mm",
        type=float,
        default=40.0,
        help="抓取标记相对组织中心的 x 偏移，单位 mm。",
    )
    parser.add_argument(
        "--grasp-offset-mm",
        type=float,
        nargs=2,
        default=None,
        metavar=("DX", "DY"),
        help="抓取标记相对组织中心的二维偏移；优先使用预计算轨迹中的记录。",
    )
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("--frames 必须至少为 2")
    if args.fps <= 0 or args.physics_hz <= 0 or args.physics_hz % args.fps != 0:
        parser.error("--physics-hz 必须是 --fps 的正整数倍")
    if args.width < 128 or args.height < 128:
        parser.error("图像分辨率过小")
    if not 0.0 <= args.poissons_ratio < 0.5:
        parser.error("--poissons-ratio 必须位于 [0, 0.5)")
    if args.samples_per_pixel < 1:
        parser.error("--samples-per-pixel 必须至少为 1")
    if args.youngs_modulus <= 0.0:
        parser.error("杨氏模量必须为正数")
    if not 0.0 < args.regional_youngs_min_pa < args.regional_youngs_max_pa:
        parser.error("随机区域杨氏模量范围必须满足 0 < min < max")
    if args.canonical_scan_azimuths < 4:
        parser.error("--canonical-scan-azimuths 必须至少为 4")
    if args.canonical_scan_radius_m <= 0.08:
        parser.error("--canonical-scan-radius-m 必须大于 0.08 m，避免相机进入组织")
    if not args.canonical_scan_elevations_deg:
        parser.error("--canonical-scan-elevations-deg 不能为空")
    if any(abs(value) >= 89.0 for value in args.canonical_scan_elevations_deg):
        parser.error("非极点扫描仰角绝对值必须小于 89 度")
    if not 0.1 <= args.tissue_uv_detail_scale <= 2.0:
        parser.error("--tissue-uv-detail-scale 必须位于 [0.1, 2.0]")
    if args.tissue_texture_contrast <= 0.0:
        parser.error("--tissue-texture-contrast 必须为正数")
    if args.preview_frame_index is not None and not 0 <= args.preview_frame_index < args.frames:
        parser.error("--preview-frame-index 必须位于数据帧范围内")
    return args


ARGS = parse_args()

from omni.isaac.lab.app import AppLauncher


app_launcher = AppLauncher(
    {
        "headless": ARGS.headless,
        "enable_cameras": True,
        "renderer": ARGS.renderer,
        "samples_per_pixel_per_frame": ARGS.samples_per_pixel,
        "anti_aliasing": 2,
        "width": ARGS.width,
        "height": ARGS.height,
        # Isaac Sim 4.0 的 Replicator 语义分割回读在本机双 A800 PathTracing
        # 下会触发 CUDA illegal address；固定单卡以保证 RGB-D/mask 一致完整。
        "multi_gpu": False,
        "sync_loads": True,
    }
)
simulation_app = app_launcher.app

import carb
import omni.usd
import torch
from omni.isaac.core import World
from omni.isaac.core.materials.deformable_material import DeformableMaterial
from omni.isaac.core.prims.soft.deformable_prim import DeformablePrim
from omni.isaac.core.prims.soft.deformable_prim_view import DeformablePrimView
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.lab.assets import Articulation
from PIL import Image
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

from orbit.surgical.assets.psm import PSM_HIGH_PD_CFG


enable_extension("omni.isaac.sensor")
simulation_app.update()
from omni.isaac.sensor import Camera


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = REPO_ROOT / "data/sim_assets"
ORGAN_ROOT = ASSET_ROOT / "OR_scene_CTLiver-Prostate-Bladder"
ORGAN_MODEL = ORGAN_ROOT / "models/organs/models_topo_blender.usdc"
ORGAN_MAIN_SCENE = ORGAN_ROOT / "main_scene.usd"
PSM_USD = ASSET_ROOT / "PSM/psm_col.usd"
PSM_TIP_MESH_ASSET = (
    REPO_ROOT
    / "data/sim/tissue_retraction_closeup_inplane_full_v1/gui_assets"
    / "official_psm_tip_meshes_v2.npz"
)

CAMERA_NAMES = ("stereo_left", "stereo_right")
CAMERA_BASELINE_M = 0.008
CAMERA_FOCAL_LENGTH_MM = 24.0
CAMERA_HORIZONTAL_APERTURE_MM = 20.955

TISSUE_CENTER = np.array([0.0022, 0.0, 0.0415], dtype=np.float32)
TISSUE_HALF_X = 0.0430
TISSUE_HALF_Y = 0.0300
TISSUE_THICKNESS = 0.0080
TISSUE_DOME_HEIGHT = 0.0030
TISSUE_GRASP_OFFSET_X = 0.0400
PSM_ROOT_POSITION = (0.0, 0.0, 0.20)

PSM_Q7_NAMES = (
    "outer_yaw",
    "outer_pitch",
    "outer_insertion",
    "outer_roll",
    "outer_wrist_pitch",
    "outer_wrist_yaw",
    "jaw",
)


def ensure_assets() -> None:
    missing = [path for path in (ORGAN_MODEL, ORGAN_MAIN_SCENE, PSM_USD) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"缺少仿真资产：{missing}")


def ensure_new_output_tree(root: Path, include_canonical_scan: bool) -> None:
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"拒绝覆盖非空目录：{root}")
    for camera_name in CAMERA_NAMES:
        (root / "rgb" / camera_name).mkdir(parents=True, exist_ok=True)
        (root / "ground_truth" / "depth" / camera_name).mkdir(parents=True, exist_ok=True)
        for label in ("tissue", "psm", "liver"):
            (root / "ground_truth" / "masks" / label / camera_name).mkdir(parents=True, exist_ok=True)
    (root / "videos").mkdir(parents=True, exist_ok=True)
    (root / "ground_truth").mkdir(parents=True, exist_ok=True)
    (root / "task_inputs").mkdir(parents=True, exist_ok=True)
    if include_canonical_scan:
        for directory in ("rgb", "depth", "mask"):
            (root / "canonical_scan" / directory).mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        diagonal = np.diagonal(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    return quaternion.astype(np.float32)


def camera_pose_ros(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, world_up))) > 0.98:
        # 极点扫描相机的视线几乎平行 z 轴，改用 y 轴定义图像上方向。
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    rotation = np.column_stack((right, down, forward))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = eye
    return rotation_matrix_to_quaternion_wxyz(rotation), rotation.astype(np.float32), transform


def configure_rendering() -> None:
    settings = carb.settings.get_settings()
    for setting in (
        "/rtx/post/tvNoise/enabled",
        "/rtx/post/tvNoise/enableFilmGrain",
        "/rtx/post/tvNoise/enableScanlines",
        "/rtx/post/tvNoise/enableRandomSplotches",
        "/rtx/post/motionblur/enabled",
        "/rtx/post/chromaticAberration/enabled",
        "/rtx/directLighting/sampledLighting/enabled",
        "/rtx/directLighting/sampledLighting/autoEnable",
        "/rtx/reflections/sampledLighting/enabled",
        "/rtx/indirectDiffuse/enabled",
        "/rtx/ambientOcclusion/enabled",
    ):
        settings.set_bool(setting, False)
    settings.set_int("/rtx/post/aa/op", 2)
    settings.set_bool("/app/window/hideUi", True)


def create_preview_material(stage, path: str, color: tuple[float, float, float], roughness: float):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def create_textured_material(
    stage,
    path: str,
    uv_primvar: str,
    diffuse_path: Path,
    roughness_path: Path | None,
    normal_path: Path | None,
    fallback_color: tuple[float, float, float],
    *,
    fixed_roughness: float = 0.46,
    specular_color: tuple[float, float, float] | None = None,
    diffuse_scale: tuple[float, float, float] | None = None,
    diffuse_bias: tuple[float, float, float] | None = None,
):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(fixed_roughness))
    shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(0.0)
    if specular_color is not None:
        shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*specular_color)
        )

    reader = UsdShade.Shader.Define(stage, f"{path}/UVReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(uv_primvar)
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    diffuse = UsdShade.Shader.Define(stage, f"{path}/DiffuseTexture")
    diffuse.CreateIdAttr("UsdUVTexture")
    diffuse.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(diffuse_path.resolve()))
    diffuse.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    diffuse.CreateInput("fallback", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*fallback_color, 1.0))
    if diffuse_scale is not None:
        diffuse.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
            Gf.Vec4f(*diffuse_scale, 1.0)
        )
    if diffuse_bias is not None:
        diffuse.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
            Gf.Vec4f(*diffuse_bias, 0.0)
        )
    diffuse.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    diffuse.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        diffuse.ConnectableAPI(), "rgb"
    )

    if roughness_path is not None:
        roughness = UsdShade.Shader.Define(stage, f"{path}/RoughnessTexture")
        roughness.CreateIdAttr("UsdUVTexture")
        roughness.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(roughness_path.resolve()))
        roughness.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        roughness.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        roughness.CreateOutput("r", Sdf.ValueTypeNames.Float)
        shader.GetInput("roughness").ConnectToSource(roughness.ConnectableAPI(), "r")

    if normal_path is not None:
        normal = UsdShade.Shader.Define(stage, f"{path}/NormalTexture")
        normal.CreateIdAttr("UsdUVTexture")
        normal.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(normal_path.resolve()))
        normal.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        normal.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2.0, 2.0, 2.0, 1.0))
        normal.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1.0, -1.0, -1.0, 0.0))
        normal.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        normal.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
            normal.ConnectableAPI(), "rgb"
        )

    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def create_low_glare_official_fascia_material(
    stage,
    path: str,
    *,
    contrast: float,
    bias: tuple[float, float, float],
):
    """用官方 Body albedo/normal 构建可控、低高光的筋膜材质。

    官方材质图的浅色筋膜区域本身含有细密斑驳纹理，但完整 PBR 网络在当前
    手术灯角度会产生大面积高光。这里不生成新的纹理图，只对官方 albedo 做
    温和的通道级对比度/色彩增强，并固定为高粗糙度、低镜面反射。
    """

    body_dir = ORGAN_ROOT / "materials/organs/Body"
    return create_textured_material(
        stage,
        path,
        "DiffuseUV",
        body_dir / "Body_topo_diffuseReflectionColor.1001.png",
        None,
        body_dir / "Body_topo_geometryNormal.1001.png",
        (0.76, 0.43, 0.34),
        fixed_roughness=0.88,
        specular_color=(0.012, 0.012, 0.012),
        # 放大官方 albedo 原有的浅棕/粉红差异，使纹理可用于视觉残差与跟踪；
        # 该变换不引入区域 ID、刚度或程序生成的标记图案。
        # 三个通道使用相同的高对比度增益以保留明显纹理；不同 bias 将平均色调
        # 调整为中等亮度暖肉色，避开发橙或粉白失去层次。
        diffuse_scale=(contrast, contrast, contrast),
        diffuse_bias=bias,
    )


def create_low_glare_official_liver_tissue_material(
    stage,
    path: str,
    *,
    contrast: float,
    bias: tuple[float, float, float],
):
    """使用 SuFIA-BC 官方 Liver albedo/normal 的低高光肝组织材质。"""

    liver_dir = ORGAN_ROOT / "materials/organs/Liver"
    return create_textured_material(
        stage,
        path,
        "DiffuseUV",
        liver_dir / "Liver_topo_blender_diffuseReflectionColor.1001.png",
        None,
        # 官方 geometryNormal 是为原生 MDL 网络作者化的 object-space 图，
        # 直接接入 UsdPreviewSurface.normal 会把大部分曲面法线翻到背光侧。
        # 这里保留官方 albedo；细节法线由曲面几何本身承担。
        None,
        (0.42, 0.11, 0.09),
        fixed_roughness=0.86,
        specular_color=(0.014, 0.014, 0.014),
        diffuse_scale=(contrast, contrast, contrast),
        diffuse_bias=bias,
    )


def bind_material(geometry_or_prim, material) -> None:
    prim = geometry_or_prim.GetPrim() if hasattr(geometry_or_prim, "GetPrim") else geometry_or_prim
    UsdShade.MaterialBindingAPI(prim).Bind(material)


def create_native_official_material(stage, path: str, graph_name: str):
    """直接引用官方 main_scene.usd 中已作者化的完整材质网络。"""

    material = UsdShade.Material.Define(stage, path)
    source_path = f"/organ_shaders/Looks/c_organ_graph_{graph_name}"
    material.GetPrim().GetReferences().AddReference(str(ORGAN_MAIN_SCENE.resolve()), source_path)
    return material


def create_capsule_tissue_mesh(
    stage,
    path: str,
    material,
    marker_material,
    scene_style: str = "legacy",
    radial_rings: int = 9,
    angular_segments: int = 64,
):
    is_sufia_view = scene_style == "sufia_viewpoint_train"
    tissue_center = TISSUE_CENTER.copy()
    if is_sufia_view:
        # 完整躯干的腹部表面比旧“仅肝脏”场景高约 8--10 mm。
        tissue_center[2] += 0.010
    tissue_half_y = 0.042 if is_sufia_view else TISSUE_HALF_Y
    tissue_thickness = 0.0045 if is_sufia_view else TISSUE_THICKNESS
    tissue_dome_height = 0.0015 if is_sufia_view else TISSUE_DOME_HEIGHT
    points: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []

    def boundary_xy(angle: float) -> tuple[float, float]:
        exponent = 8.0 if is_sufia_view else 3.2
        cosine = math.cos(angle)
        sine = math.sin(angle)
        x = TISSUE_HALF_X * math.copysign(abs(cosine) ** (2.0 / exponent), cosine)
        y = tissue_half_y * math.copysign(abs(sine) ** (2.0 / exponent), sine)
        return x, y

    def add_surface(top: bool) -> int:
        offset = len(points)
        center_height = (
            tissue_center[2]
            + (0.5 if top else -0.5) * tissue_thickness
            + (tissue_dome_height if top else 0.35 * tissue_dome_height)
        )
        points.append((float(tissue_center[0]), float(tissue_center[1]), float(center_height)))
        uvs.append((0.38, 0.48))
        for ring in range(1, radial_rings + 1):
            radius = ring / radial_rings
            dome = (1.0 - radius * radius) * tissue_dome_height
            z = (
                tissue_center[2]
                + (0.5 if top else -0.5) * tissue_thickness
                + (dome if top else 0.35 * dome)
            )
            for segment in range(angular_segments):
                angle = 2.0 * math.pi * segment / angular_segments
                bx, by = boundary_xy(angle)
                x = float(tissue_center[0] + radius * bx)
                y = float(tissue_center[1] + radius * by)
                points.append((x, y, float(z)))
                # SuFIA 构图模式使用官方 Body 贴图中的浅色筋膜/皮下组织区域。
                normalized_x = (x - tissue_center[0]) / (2.0 * TISSUE_HALF_X) + 0.5
                normalized_y = (y - tissue_center[1]) / (2.0 * tissue_half_y) + 0.5
                if is_sufia_view:
                    uvs.append((0.18 + 0.16 * normalized_x, 0.325 + 0.29 * normalized_y))
                else:
                    uvs.append((0.20 + 0.60 * normalized_x, 0.18 + 0.64 * normalized_y))
        return offset

    top_offset = add_surface(True)
    bottom_offset = add_surface(False)

    def vertex(offset: int, ring: int, segment: int) -> int:
        if ring == 0:
            return offset
        return offset + 1 + (ring - 1) * angular_segments + segment % angular_segments

    faces: list[tuple[int, int, int]] = []
    top_face_indices: list[int] = []
    for segment in range(angular_segments):
        top_face_indices.append(len(faces))
        faces.append(
            (
                vertex(top_offset, 0, 0),
                vertex(top_offset, 1, segment),
                vertex(top_offset, 1, segment + 1),
            )
        )
    for ring in range(1, radial_rings):
        for segment in range(angular_segments):
            inner = vertex(top_offset, ring, segment)
            inner_next = vertex(top_offset, ring, segment + 1)
            outer = vertex(top_offset, ring + 1, segment)
            outer_next = vertex(top_offset, ring + 1, segment + 1)
            top_face_indices.extend((len(faces), len(faces) + 1))
            faces.extend(((inner, outer, outer_next), (inner, outer_next, inner_next)))

    for segment in range(angular_segments):
        faces.append(
            (
                vertex(bottom_offset, 0, 0),
                vertex(bottom_offset, 1, segment + 1),
                vertex(bottom_offset, 1, segment),
            )
        )
    for ring in range(1, radial_rings):
        for segment in range(angular_segments):
            inner = vertex(bottom_offset, ring, segment)
            inner_next = vertex(bottom_offset, ring, segment + 1)
            outer = vertex(bottom_offset, ring + 1, segment)
            outer_next = vertex(bottom_offset, ring + 1, segment + 1)
            faces.extend(((inner, outer_next, outer), (inner, inner_next, outer_next)))

    for segment in range(angular_segments):
        top = vertex(top_offset, radial_rings, segment)
        top_next = vertex(top_offset, radial_rings, segment + 1)
        bottom = vertex(bottom_offset, radial_rings, segment)
        bottom_next = vertex(bottom_offset, radial_rings, segment + 1)
        faces.extend(((top, bottom, bottom_next), (top, bottom_next, top_next)))

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set(
        [Gf.Vec3f(float(point[0]), float(point[1]), float(point[2])) for point in points]
    )
    mesh.GetFaceVertexCountsAttr().Set([3] * len(faces))
    mesh.GetFaceVertexIndicesAttr().Set([index for face in faces for index in face])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set([Gf.Vec2f(*uv) for uv in uvs])
    # 官方器官网格和 c_organ_graph_* 材质网络使用 DiffuseUV；同时保留 st 便于
    # 通用工具读取。两个 primvar 指向完全相同的逐顶点 UV。
    diffuse_uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "DiffuseUV", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    diffuse_uv.Set([Gf.Vec2f(*uv) for uv in uvs])
    bind_material(mesh, material)

    point_array = np.asarray(points, dtype=np.float32)
    marker_center = tissue_center[:2] + np.array(
        [TISSUE_GRASP_OFFSET_X, 0.0], dtype=np.float32
    )
    marker_faces = []
    for face_index in top_face_indices:
        center = point_array[np.asarray(faces[face_index], dtype=np.int64)].mean(axis=0)
        if np.linalg.norm(center[:2] - marker_center) <= 0.009:
            marker_faces.append(face_index)
    if not marker_faces:
        raise RuntimeError("红色抓取标记没有覆盖任何组织表面三角形")
    marker_subset = UsdShade.MaterialBindingAPI(mesh.GetPrim()).CreateMaterialBindSubset(
        "GraspMarker", marker_faces
    )
    UsdShade.MaterialBindingAPI(marker_subset.GetPrim()).Bind(marker_material)
    add_update_semantics(mesh.GetPrim(), "tissue")
    return mesh, np.asarray(points, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def create_sufia_thin_tissue_mesh(
    stage,
    path: str,
    material,
    marker_material,
    columns: int = 7,
    rows: int = 15,
    u_min: float = 0.0,
    u_max: float = 1.0,
    v_min: float = 0.0,
    v_max: float = 1.0,
    include_marker: bool = True,
):
    """生成一个封闭组织分区；相邻分区共享全局几何/UV 边界。"""

    if not 0.0 <= u_min < u_max <= 1.0:
        raise ValueError(f"组织分区 UV 范围非法：{u_min}, {u_max}")
    if not 0.0 <= v_min < v_max <= 1.0:
        raise ValueError(f"组织分区 UV 范围非法：{v_min}, {v_max}")

    center = TISSUE_CENTER.copy()
    center[2] += 0.010
    half_x = TISSUE_HALF_X
    half_y = 0.042
    # 网页参考中的牵拉层接近筋膜薄片，不应呈现为厚板。0.9 mm 仍保留
    # 封闭体积，便于 PhysX 构建单层六面体仿真网格。
    thickness = 0.0009
    corner_fraction = 0.13
    points: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []

    def rounded_coordinate(sx: float, sy: float) -> tuple[float, float]:
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

    def add_surface(top: bool) -> int:
        offset = len(points)
        for row in range(rows):
            local_v = row / (rows - 1)
            v = v_min + (v_max - v_min) * local_v
            sy = 2.0 * v - 1.0
            for column in range(columns):
                local_u = column / (columns - 1)
                u = u_min + (u_max - u_min) * local_u
                sx = 2.0 * u - 1.0
                rx, ry = rounded_coordinate(sx, sy)
                # 保持规则拓扑和 UV，但让轮廓成为轻微倾斜、非完全对称的组织片，
                # 避免视觉上像工业矩形板。
                rx += 0.070 * ry + 0.022 * math.sin(math.pi * v)
                ry += 0.040 * math.sin(math.pi * u) - 0.018 * u
                angle = math.radians(-6.0)
                local_x = half_x * rx
                local_y = half_y * ry
                shaped_x = math.cos(angle) * local_x - math.sin(angle) * local_y
                shaped_y = math.sin(angle) * local_x + math.cos(angle) * local_y
                # 亚毫米尺度起伏只用于呈现官方法线/粗糙度细节，不改变材料参数真值。
                dome = 0.00042 * max(0.0, 1.0 - sx * sx) * max(0.0, 1.0 - sy * sy)
                wrinkle = 0.00024 * math.sin(2.0 * math.pi * u) * math.sin(math.pi * v)
                z = center[2] + (0.5 if top else -0.5) * thickness
                z += (dome + wrinkle) if top else 0.20 * (dome + wrinkle)
                points.append(
                    (
                        float(center[0] + shaped_x),
                        float(center[1] + shaped_y),
                        float(z),
                    )
                )
                # 只取官方 Body atlas 的浅色、斑驳软组织区域；不生成或修改贴图像素。
                # 缩小官方 atlas 的取样窗口，使天然斑驳纹理在表面放大约 1.9 倍；
                # 所有分区仍使用同一全局 u/v，因此缝合处纹理连续。
                uvs.append((0.18 + 0.16 * u, 0.325 + 0.29 * v))
        return offset

    top_offset = add_surface(True)
    bottom_offset = add_surface(False)

    def vertex(offset: int, row: int, column: int) -> int:
        return offset + row * columns + column

    faces: list[tuple[int, int, int]] = []
    top_face_indices: list[int] = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            a = vertex(top_offset, row, column)
            b = vertex(top_offset, row, column + 1)
            c = vertex(top_offset, row + 1, column + 1)
            d = vertex(top_offset, row + 1, column)
            top_face_indices.extend((len(faces), len(faces) + 1))
            if (row + column) % 2 == 0:
                faces.extend(((a, b, c), (a, c, d)))
            else:
                faces.extend(((a, b, d), (b, c, d)))

    for row in range(rows - 1):
        for column in range(columns - 1):
            a = vertex(bottom_offset, row, column)
            b = vertex(bottom_offset, row, column + 1)
            c = vertex(bottom_offset, row + 1, column + 1)
            d = vertex(bottom_offset, row + 1, column)
            if (row + column) % 2 == 0:
                faces.extend(((a, c, b), (a, d, c)))
            else:
                faces.extend(((a, d, b), (b, d, c)))

    boundary: list[tuple[int, int]] = []
    boundary.extend((0, column) for column in range(columns))
    boundary.extend((row, columns - 1) for row in range(1, rows))
    boundary.extend((rows - 1, column) for column in range(columns - 2, -1, -1))
    boundary.extend((row, 0) for row in range(rows - 2, 0, -1))
    for index, (row_a, column_a) in enumerate(boundary):
        row_b, column_b = boundary[(index + 1) % len(boundary)]
        top_a = vertex(top_offset, row_a, column_a)
        top_b = vertex(top_offset, row_b, column_b)
        bottom_a = vertex(bottom_offset, row_a, column_a)
        bottom_b = vertex(bottom_offset, row_b, column_b)
        faces.extend(((top_a, bottom_a, bottom_b), (top_a, bottom_b, top_b)))

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set([Gf.Vec3f(*point) for point in points])
    mesh.GetFaceVertexCountsAttr().Set([3] * len(faces))
    face_vertex_indices = [index for face in faces for index in face]
    mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    primvars = UsdGeom.PrimvarsAPI(mesh)
    for uv_name in ("DiffuseUV", "st"):
        uv_primvar = primvars.CreatePrimvar(
            uv_name, Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        )
        uv_primvar.Set([Gf.Vec2f(float(uv[0]), float(uv[1])) for uv in uvs])
    attribute = primvars.CreatePrimvar(
        "Attribute", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.faceVarying
    )
    attribute.Set([Gf.Vec3f(1.0, 1.0, 1.0)] * len(face_vertex_indices))
    bind_material(mesh, material)

    point_array = np.asarray(points, dtype=np.float32)
    marker_center = center[:2] + np.array(
        [TISSUE_GRASP_OFFSET_X, 0.0], dtype=np.float32
    )
    if include_marker:
        marker_faces = []
        for face_index in top_face_indices:
            face_center = point_array[np.asarray(faces[face_index], dtype=np.int64)].mean(axis=0)
            if np.linalg.norm(face_center[:2] - marker_center) <= 0.0022:
                marker_faces.append(face_index)
        if not marker_faces:
            nearest = min(
                top_face_indices,
                key=lambda face_index: np.linalg.norm(
                    point_array[np.asarray(faces[face_index], dtype=np.int64)].mean(axis=0)[:2]
                    - marker_center
                ),
            )
            marker_faces = [nearest]
        marker_subset = UsdShade.MaterialBindingAPI(mesh.GetPrim()).CreateMaterialBindSubset(
            "GraspMarker", marker_faces
        )
        UsdShade.MaterialBindingAPI(marker_subset.GetPrim()).Bind(marker_material)
    add_update_semantics(mesh.GetPrim(), "tissue")
    return mesh, point_array, np.asarray(faces, dtype=np.int32)


def load_precomputed_tissue_trajectory(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.tissue_trajectory is None:
        raise ValueError(
            "sufia_viewpoint_train 需要 --tissue-trajectory；请通过 "
            "scripts/run_generate_sufia_viewpoint_dataset.sh 启动"
        )
    trajectory_path = args.tissue_trajectory.expanduser().resolve()
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"十区域组织轨迹不存在：{trajectory_path}")
    with np.load(trajectory_path) as loaded:
        trajectory = {key: loaded[key] for key in loaded.files}
    required = {
        "timestamps",
        "visual_positions",
        "visual_velocities",
        "rest_positions",
        "visual_faces",
        "visual_uvs",
        "top_face_indices",
        "tet_indices",
        "tet_region_ids",
        "node_region_ids",
        "region_centers_uv",
        "region_metric_scale",
        "region_tet_counts",
        "regional_youngs_modulus_pa",
        "anchor_mask",
        "grasp_mask",
        "grasp_coupling_weights",
    }
    missing = sorted(required.difference(trajectory))
    if missing:
        raise ValueError(f"十区域组织轨迹缺少字段：{missing}")
    if trajectory["visual_positions"].shape != (
        args.frames,
        len(trajectory["rest_positions"]),
        3,
    ):
        raise ValueError(
            "十区域组织轨迹帧数/节点数与生成参数不一致："
            f"{trajectory['visual_positions'].shape}"
        )
    expected_timestamps = np.arange(args.frames, dtype=np.float64) / args.fps
    if not np.allclose(trajectory["timestamps"], expected_timestamps, atol=1.0e-9):
        raise ValueError("十区域组织轨迹时间戳与 --frames/--fps 不一致")
    if len(trajectory["regional_youngs_modulus_pa"]) != 10:
        raise ValueError("组织轨迹必须恰好包含 10 个材料区域")
    if int(np.asarray(trajectory.get("regional_seed", -1)).item()) != args.regional_seed:
        raise ValueError("组织轨迹随机种子与 --regional-seed 不一致")
    trajectory_motion_profile = str(
        np.asarray(trajectory.get("motion_profile", "retract_hold")).item()
    )
    if trajectory_motion_profile != args.motion_profile:
        raise ValueError(
            "组织轨迹运动协议与渲染参数不一致："
            f"{trajectory_motion_profile} != {args.motion_profile}"
        )
    return trajectory


def create_precomputed_tissue_mesh(
    stage,
    path: str,
    material,
    marker_material,
    trajectory,
    *,
    uv_detail_scale: float,
    grasp_offset_xy_m: np.ndarray,
):
    """创建单一连续渲染网格；逐帧点坐标来自材料感知 XPBD 真值。"""

    points = np.asarray(trajectory["rest_positions"], dtype=np.float32)
    faces = np.asarray(trajectory["visual_faces"], dtype=np.int32)
    uvs = np.asarray(trajectory["visual_uvs"], dtype=np.float32)
    uv_center = np.asarray(
        trajectory.get("visual_uv_center", np.array([0.26, 0.47], dtype=np.float32)),
        dtype=np.float32,
    )
    uvs = uv_center + (uvs - uv_center) * float(uv_detail_scale)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set(
        [Gf.Vec3f(float(point[0]), float(point[1]), float(point[2])) for point in points]
    )
    mesh.GetFaceVertexCountsAttr().Set([3] * len(faces))
    face_vertex_indices = faces.reshape(-1).tolist()
    mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    primvars = UsdGeom.PrimvarsAPI(mesh)
    for uv_name in ("DiffuseUV", "st"):
        uv_primvar = primvars.CreatePrimvar(
            uv_name, Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        )
        uv_primvar.Set([Gf.Vec2f(float(uv[0]), float(uv[1])) for uv in uvs])
    attribute = primvars.CreatePrimvar(
        "Attribute", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.faceVarying
    )
    attribute.Set([Gf.Vec3f(1.0, 1.0, 1.0)] * len(face_vertex_indices))
    bind_material(mesh, material)

    marker_center = TISSUE_CENTER[:2] + np.asarray(
        grasp_offset_xy_m, dtype=np.float32
    )
    marker_faces = []
    for face_index in np.asarray(trajectory["top_face_indices"], dtype=np.int32):
        center = points[faces[face_index]].mean(axis=0)
        if np.linalg.norm(center[:2] - marker_center) <= 0.0022:
            marker_faces.append(int(face_index))
    if not marker_faces:
        raise RuntimeError("连续组织的红色抓取标记没有覆盖三角形")
    marker_subset = UsdShade.MaterialBindingAPI(mesh.GetPrim()).CreateMaterialBindSubset(
        "GraspMarker", marker_faces
    )
    UsdShade.MaterialBindingAPI(marker_subset.GetPrim()).Bind(marker_material)
    add_update_semantics(mesh.GetPrim(), "tissue")
    return mesh


def create_official_anatomy(stage, liver_material, scene_style: str):
    anatomy = UsdGeom.Xform.Define(stage, "/World/Scene/OfficialAnatomy")
    anatomy.GetPrim().GetReferences().AddReference(str(ORGAN_MODEL.resolve()), "/root")
    xformable = UsdGeom.Xformable(anatomy)
    xformable.AddTranslateOp().Set(Gf.Vec3d(-0.1840, -0.1484, -0.3301))
    xformable.AddScaleOp().Set(Gf.Vec3f(0.55, 0.55, 0.55))
    if scene_style == "legacy":
        for child in anatomy.GetPrim().GetChildren():
            if child.GetName() not in {"Liver_topo_blender", "_materials"}:
                imageable = UsdGeom.Imageable(child)
                if imageable:
                    imageable.MakeInvisible()
    liver_prim = stage.GetPrimAtPath(
        "/World/Scene/OfficialAnatomy/Liver_topo_blender/Liver_topo_blender"
    )
    if not liver_prim.IsValid():
        raise RuntimeError("官方肝脏 mesh prim 未找到")
    if scene_style == "legacy":
        bind_material(liver_prim, liver_material)
    add_update_semantics(liver_prim, "liver")
    return anatomy, liver_prim


def bind_official_anatomy_pbr(stage, anatomy_prim) -> dict[str, int]:
    """绑定官方 main_scene.usd 的原生材质 prim，不重建任何纹理/着色节点。"""

    specs = {
        "Merged_Body3": "body",
        "Merged_Body4": "body",
        "Back_muscles_topo_blender": "back_muscles",
        "Ribs_topo_blender": "ribs",
        "Hips_topo_blender": "hips",
        "Liver_topo_blender": "liver",
        "Stomach_topo_blender": "stomach",
        "Spleen_topo_blender": "spleen",
        "Veins_topo_blender": "veins",
        "Gallbladder_topo_blender": "gallbladder",
        "Colon_topo_blender": "colon",
        "Small_bowel_topo_blender": "small_bowel",
        "Kidney_topo_blender": "kidney",
        "Pancreas_topo_blender": "pancreas",
        "Spine_topo_blender": "spine",
    }
    materials = {
        prim_name: create_native_official_material(
            stage, f"/World/Looks/Native_{prim_name}", graph_name
        )
        for prim_name, graph_name in specs.items()
    }

    bound_counts = {name: 0 for name in materials}
    anatomy_path = str(anatomy_prim.GetPath())
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not prim_path.startswith(anatomy_path + "/") or not prim.IsA(UsdGeom.Mesh):
            continue
        for prim_name, material in materials.items():
            if f"/{prim_name}/" in prim_path or prim_path.endswith(f"/{prim_name}"):
                bind_material(prim, material)
                bound_counts[prim_name] += 1
                break
    if bound_counts.get("Liver_topo_blender", 0) == 0:
        raise RuntimeError("完整解剖场景未绑定到官方肝脏 PBR 网格")
    return bound_counts


def create_scene(world: World, args: argparse.Namespace):
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World/Scene")

    tissue_trajectory = (
        load_precomputed_tissue_trajectory(args)
        if args.scene_style == "sufia_viewpoint_train"
        else None
    )
    tissue_visual_material_profile = str(
        np.asarray(
            tissue_trajectory.get("visual_material_profile", "official_body_fascia")
            if tissue_trajectory is not None
            else "official_liver"
        ).item()
    )

    liver_dir = ORGAN_ROOT / "materials/organs/Liver"
    liver_material = create_textured_material(
        stage,
        "/World/Looks/OfficialLiver",
        "UVMap",
        liver_dir / "Liver_topo_blender_diffuseReflectionColor.1001.png",
        None,
        None,
        (0.38, 0.045, 0.035),
    )
    if (
        args.scene_style == "sufia_viewpoint_train"
        and tissue_visual_material_profile == "official_liver"
    ):
        tissue_material = create_low_glare_official_liver_tissue_material(
            stage,
            "/World/Looks/LowGlareOfficialLiverTissue",
            contrast=float(args.tissue_texture_contrast),
            bias=tuple(float(value) for value in args.tissue_texture_bias),
        )
    elif args.scene_style == "sufia_viewpoint_train":
        tissue_material = create_low_glare_official_fascia_material(
            stage,
            "/World/Looks/LowGlareOfficialFasciaTissue",
            contrast=float(args.tissue_texture_contrast),
            bias=tuple(float(value) for value in args.tissue_texture_bias),
        )
    else:
        tissue_material = create_textured_material(
            stage,
            "/World/Looks/OfficialLiverTissue",
            "st",
            liver_dir / "Liver_topo_blender_diffuseReflectionColor.1001.png",
            None,
            None,
            (0.48, 0.09, 0.075),
        )
    marker_material = create_preview_material(stage, "/World/Looks/GraspMarker", (0.75, 0.005, 0.01), 0.32)

    if args.scene_style == "legacy":
        background_material = create_preview_material(
            stage, "/World/Looks/SurgicalDrape", (0.018, 0.115, 0.095), 0.83
        )
        background_name = "SurgicalDrape"
        background_height = -0.135
    else:
        background_material = create_preview_material(
            stage, "/World/Looks/StudioBackground", (0.55, 0.57, 0.60), 0.88
        )
        background_name = "StudioBackground"
        background_height = -0.185
    background = UsdGeom.Cube.Define(stage, f"/World/Scene/{background_name}")
    background.CreateSizeAttr(1.0)
    background_xform = UsdGeom.Xformable(background)
    background_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, background_height))
    background_xform.AddScaleOp().Set(Gf.Vec3f(0.42, 0.36, 0.012))
    bind_material(background, background_material)

    anatomy, liver_prim = create_official_anatomy(stage, liver_material, args.scene_style)
    anatomy_pbr_bindings = {}
    if args.scene_style == "sufia_viewpoint_train":
        anatomy_pbr_bindings = bind_official_anatomy_pbr(stage, anatomy.GetPrim())
    tissue_meshes = []
    deformable_views = []
    visual_point_parts = []
    visual_face_parts = []
    region_youngs_modulus_pa = []
    region_uv_bounds = []
    marker_region = 0
    if args.scene_style == "sufia_viewpoint_train":
        assert tissue_trajectory is not None
        tissue_path = "/World/Scene/Tissue"
        tissue_mesh = create_precomputed_tissue_mesh(
            stage,
            tissue_path,
            tissue_material,
            marker_material,
            tissue_trajectory,
            uv_detail_scale=float(args.tissue_uv_detail_scale),
            grasp_offset_xy_m=np.asarray(
                tissue_trajectory.get(
                    "grasp_offset_m",
                    np.asarray(
                        args.grasp_offset_mm
                        if args.grasp_offset_mm is not None
                        else (args.grasp_offset_x_mm, 0.0),
                        dtype=np.float32,
                    )
                    * 1.0e-3,
                ),
                dtype=np.float32,
            ),
        )
        tissue_meshes.append(tissue_mesh)
        visual_point_parts.append(tissue_trajectory["rest_positions"])
        visual_face_parts.append(tissue_trajectory["visual_faces"])
        region_youngs_modulus_pa = tissue_trajectory[
            "regional_youngs_modulus_pa"
        ].astype(np.float32).tolist()
        # 随机 Voronoi 区域不是矩形，中心和各向异性度量单独保存在 scene/GT。
        region_uv_bounds = [(0.0, 1.0, 0.0, 1.0)] * 10
    else:
        tissue_path = "/World/Scene/Tissue"
        tissue_mesh, visual_rest_points, visual_faces = create_capsule_tissue_mesh(
            stage, tissue_path, tissue_material, marker_material, args.scene_style
        )
        tissue_meshes.append(tissue_mesh)
        visual_point_parts.append(visual_rest_points)
        visual_face_parts.append(visual_faces)
        region_youngs_modulus_pa = [float(args.youngs_modulus)]
        region_uv_bounds = [(0.0, 1.0, 0.0, 1.0)]
        deformable_material = DeformableMaterial(
            prim_path="/World/PhysicsMaterials/TissueMaterial",
            dynamic_friction=0.55,
            youngs_modulus=args.youngs_modulus,
            poissons_ratio=args.poissons_ratio,
            damping_scale=args.damping_scale,
            elasticity_damping=args.elasticity_damping,
        )
        DeformablePrim(
            prim_path=tissue_path,
            deformable_material=deformable_material,
            vertex_velocity_damping=0.02,
            self_collision=False,
            solver_position_iteration_count=20,
            simulation_hexahedral_resolution=12,
            collision_simplification=False,
        )
        deformable_view = DeformablePrimView(prim_paths_expr=tissue_path, name="tissue_view")
        world.scene.add(deformable_view)
        deformable_views.append(deformable_view)

    visual_rest_points = np.concatenate(visual_point_parts, axis=0)
    visual_faces = np.concatenate(visual_face_parts, axis=0)
    visual_region_ids = (
        tissue_trajectory["node_region_ids"].astype(np.int16)
        if tissue_trajectory is not None
        else np.concatenate(
            [
                np.full(len(points), index, dtype=np.int16)
                for index, points in enumerate(visual_point_parts)
            ]
        )
    )

    robot_cfg = PSM_HIGH_PD_CFG.replace(prim_path="/World/PSM")
    robot_cfg.spawn.usd_path = str(PSM_USD.resolve())
    robot_cfg.init_state.pos = (
        (PSM_ROOT_POSITION[0], PSM_ROOT_POSITION[1], PSM_ROOT_POSITION[2] + 0.010)
        if args.scene_style == "sufia_viewpoint_train"
        else PSM_ROOT_POSITION
    )
    robot = Articulation(robot_cfg)
    psm_root = stage.GetPrimAtPath("/World/PSM")
    if not psm_root.IsValid():
        raise RuntimeError("官方 PSM 根 prim 未生成")
    add_update_semantics(psm_root, "psm")
    disabled_collision_prims = 0
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith("/World/PSM/"):
            continue
        if prim.IsA(UsdGeom.Gprim):
            add_update_semantics(prim, "psm")
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            disabled_collision_prims += 1
    if disabled_collision_prims == 0:
        raise RuntimeError("官方 PSM 没有找到可关闭的碰撞 prim")

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(720.0 if args.scene_style == "sufia_viewpoint_train" else 850.0)
    dome.CreateColorAttr(Gf.Vec3f(0.96, 0.98, 1.0) if args.scene_style == "sufia_viewpoint_train" else Gf.Vec3f(0.86, 0.91, 1.0))
    key = UsdLux.RectLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(2300.0 if args.scene_style == "sufia_viewpoint_train" else 3200.0)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.86, 0.78))
    key.CreateWidthAttr(0.28 if args.scene_style == "sufia_viewpoint_train" else 0.18)
    key.CreateHeightAttr(0.24 if args.scene_style == "sufia_viewpoint_train" else 0.18)
    key_xform = UsdGeom.Xformable(key)
    key_xform.AddTranslateOp().Set(Gf.Vec3d(-0.08, -0.12, 0.30))
    key_xform.AddRotateXYZOp().Set(Gf.Vec3f(28.0, -18.0, -22.0))
    fill = UsdLux.SphereLight.Define(stage, "/World/Lights/Fill")
    fill.CreateIntensityAttr(760.0 if args.scene_style == "sufia_viewpoint_train" else 1200.0)
    fill.CreateRadiusAttr(0.045)
    fill.CreateColorAttr(Gf.Vec3f(0.72, 0.84, 1.0))
    fill_xform = UsdGeom.Xformable(fill)
    fill_xform.AddTranslateOp().Set(Gf.Vec3d(0.15, 0.07, 0.18))

    camera_target = np.array(
        args.task_camera_target
        if args.task_camera_target is not None
        else [0.008, 0.0, 0.032],
        dtype=np.float64,
    )
    central_eye = np.array(
        args.task_camera_eye
        if args.task_camera_eye is not None
        else [0.085, -0.205, 0.172],
        dtype=np.float64,
    )
    cameras: dict[str, Camera] = {}
    camera_data: dict[str, dict] = {}
    for camera_name, camera_x_offset in zip(CAMERA_NAMES, (-0.004, 0.004)):
        eye = central_eye + np.array([camera_x_offset, 0.0, 0.0], dtype=np.float64)
        quaternion, rotation, transform_ros = camera_pose_ros(eye, camera_target)
        transform_blender = transform_ros @ np.diag([1.0, -1.0, -1.0, 1.0])
        camera = Camera(
            prim_path=f"/World/Cameras/{camera_name}",
            name=camera_name,
            resolution=(args.width, args.height),
        )
        world.scene.add(camera)
        cameras[camera_name] = camera
        camera_data[camera_name] = {
            "eye": eye.astype(np.float32),
            "target": camera_target.astype(np.float32),
            "quaternion_wxyz_ros": quaternion,
            "rotation_ros": rotation,
            "X_WC_ros": transform_ros.astype(np.float64),
            "X_WC_blender": transform_blender.astype(np.float64),
        }

    return {
        "stage": stage,
        "robot": robot,
        "deformable_views": deformable_views,
        "cameras": cameras,
        "camera_data": camera_data,
        "visual_rest_points": visual_rest_points,
        "visual_faces": visual_faces,
        "visual_region_ids": visual_region_ids,
        "liver_prim": liver_prim,
        "tissue_meshes": tissue_meshes,
        "tissue_trajectory": tissue_trajectory,
        "region_youngs_modulus_pa": np.asarray(region_youngs_modulus_pa, dtype=np.float32),
        "region_uv_bounds": np.asarray(region_uv_bounds, dtype=np.float32),
        "region_centers_uv": (
            tissue_trajectory["region_centers_uv"].astype(np.float32)
            if tissue_trajectory is not None
            else np.empty((0, 2), dtype=np.float32)
        ),
        "region_metric_scale": (
            tissue_trajectory["region_metric_scale"].astype(np.float32)
            if tissue_trajectory is not None
            else np.empty((0, 2), dtype=np.float32)
        ),
        "region_tet_counts": (
            tissue_trajectory["region_tet_counts"].astype(np.int32)
            if tissue_trajectory is not None
            else np.empty((0,), dtype=np.int32)
        ),
        "regional_seed": int(args.regional_seed),
        "marker_region": int(marker_region),
        "psm_disabled_collision_prims": disabled_collision_prims,
        "anatomy_pbr_bindings": anatomy_pbr_bindings,
        "tissue_visual_material_profile": tissue_visual_material_profile,
        "canonical_scan_occluders": [anatomy.GetPrim(), psm_root, background.GetPrim()],
    }


def configure_cameras(cameras: dict[str, Camera], camera_data: dict[str, dict], args: argparse.Namespace) -> None:
    vertical_aperture = CAMERA_HORIZONTAL_APERTURE_MM * args.height / args.width
    for camera_name, camera in cameras.items():
        data = camera_data[camera_name]
        camera.set_world_pose(
            position=data["eye"],
            orientation=data["quaternion_wxyz_ros"],
            camera_axes="ros",
        )
        camera.set_focal_length(CAMERA_FOCAL_LENGTH_MM)
        camera.set_horizontal_aperture(CAMERA_HORIZONTAL_APERTURE_MM)
        camera.set_vertical_aperture(vertical_aperture)
        camera.set_clipping_range(near_distance=0.01, far_distance=2.0)
        camera.set_focus_distance(
            float(np.linalg.norm(data["eye"] - data["target"]))
        )
        camera.add_semantic_segmentation_to_frame()
        camera.add_distance_to_image_plane_to_frame()


def to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(rgb)
    if np.issubdtype(array.dtype, np.floating) and array.size and float(np.nanmax(array)) <= 1.0:
        array = array * 255.0
    return np.clip(array[..., :3], 0, 255).astype(np.uint8)


def get_camera_annotator_data(camera: Camera, name: str):
    annotator = camera._custom_annotators.get(name)
    return None if annotator is None else annotator.get_data()


def semantic_payload_to_masks(payload, labels: tuple[str, ...]) -> tuple[dict[str, np.ndarray], dict]:
    if not isinstance(payload, dict):
        raise RuntimeError("语义分割 annotator 没有返回 ID 映射")
    data = np.squeeze(np.asarray(payload.get("data")))
    info = payload.get("info", {}) or {}
    id_to_labels = info.get("idToLabels", {}) if isinstance(info, dict) else {}
    masks: dict[str, np.ndarray] = {}
    for label in labels:
        ids = []
        for raw_id, semantic_labels in id_to_labels.items():
            text = json.dumps(semantic_labels, ensure_ascii=False).lower()
            if label.lower() in text:
                ids.append(int(raw_id))
        masks[label] = (np.isin(data, ids).astype(np.uint8) * 255) if ids else np.zeros_like(data, dtype=np.uint8)
    if not np.any(masks["tissue"]):
        raise RuntimeError(f"语义分割中没有组织像素：{id_to_labels}")
    return masks, json.loads(json.dumps(info, default=str))


def canonical_scan_view_specs(center: np.ndarray, args: argparse.Namespace) -> list[dict]:
    """生成覆盖顶面、底面和四周侧缘的确定性相机球面采样。"""

    views: list[dict] = []
    radius = float(args.canonical_scan_radius_m)
    for elevation_deg in args.canonical_scan_elevations_deg:
        elevation = math.radians(float(elevation_deg))
        for azimuth_index in range(args.canonical_scan_azimuths):
            azimuth_deg = 360.0 * azimuth_index / args.canonical_scan_azimuths
            azimuth = math.radians(azimuth_deg)
            offset = radius * np.array(
                [
                    math.cos(elevation) * math.cos(azimuth),
                    math.cos(elevation) * math.sin(azimuth),
                    math.sin(elevation),
                ],
                dtype=np.float64,
            )
            views.append(
                {
                    "azimuth_deg": float(azimuth_deg),
                    "elevation_deg": float(elevation_deg),
                    "eye": center + offset,
                }
            )
    for elevation_deg in (-90.0, 90.0):
        views.append(
            {
                "azimuth_deg": None,
                "elevation_deg": elevation_deg,
                "eye": center
                + np.array([0.0, 0.0, math.copysign(radius, elevation_deg)], dtype=np.float64),
            }
        )
    return views


def capture_canonical_tissue_scan(
    world: World,
    scene: dict,
    output: Path,
    args: argparse.Namespace,
) -> dict:
    """在任务开始前记录不含器械/腹部遮挡的静止组织全视角 RGB-D。"""

    camera_name = CAMERA_NAMES[0]
    camera: Camera = scene["cameras"][camera_name]
    center = np.asarray(scene["visual_rest_points"], dtype=np.float64).mean(axis=0)
    views = canonical_scan_view_specs(center, args)

    for prim in scene["canonical_scan_occluders"]:
        UsdGeom.Imageable(prim).MakeInvisible()
    world.render()

    intrinsic = to_numpy(camera.get_intrinsics_matrix()).astype(np.float64)
    records = []
    try:
        for view_index, view in enumerate(views):
            quaternion, rotation, transform_ros = camera_pose_ros(view["eye"], center)
            transform_blender = transform_ros @ np.diag([1.0, -1.0, -1.0, 1.0])
            camera.set_world_pose(
                position=np.asarray(view["eye"], dtype=np.float32),
                orientation=quaternion,
                camera_axes="ros",
            )
            camera.set_focus_distance(float(args.canonical_scan_radius_m))
            world.render()

            rgb = normalize_rgb(camera.get_rgb())
            depth = np.asarray(
                get_camera_annotator_data(camera, "distance_to_image_plane"),
                dtype=np.float32,
            )
            masks, _ = semantic_payload_to_masks(
                get_camera_annotator_data(camera, "semantic_segmentation"),
                ("tissue",),
            )
            mask = masks["tissue"]
            if rgb.shape[:2] != (args.height, args.width):
                raise RuntimeError(f"全视角扫描 RGB 尺寸错误：{rgb.shape}")
            if depth.shape != (args.height, args.width):
                raise RuntimeError(f"全视角扫描深度尺寸错误：{depth.shape}")
            if int(np.count_nonzero(mask)) == 0:
                raise RuntimeError(f"全视角扫描视角 {view_index} 没有组织像素")

            filename = f"{view_index:06d}"
            Image.fromarray(rgb, mode="RGB").save(
                output / "canonical_scan" / "rgb" / f"{filename}.png",
                compress_level=1,
            )
            np.save(output / "canonical_scan" / "depth" / f"{filename}.npy", depth)
            Image.fromarray(mask, mode="L").save(
                output / "canonical_scan" / "mask" / f"{filename}.png",
                compress_level=1,
            )
            records.append(
                {
                    "view": view_index,
                    "azimuth_deg": view["azimuth_deg"],
                    "elevation_deg": view["elevation_deg"],
                    "rgb_path": f"rgb/{filename}.png",
                    "depth_path": f"depth/{filename}.npy",
                    "mask_path": f"mask/{filename}.png",
                    "K": intrinsic.tolist(),
                    "X_WC_ros_optical": transform_ros.tolist(),
                    "X_WC_blender": transform_blender.tolist(),
                    "tissue_mask_pixels": int(np.count_nonzero(mask)),
                }
            )
            print(
                f"[全视角扫描] {view_index + 1:03d}/{len(views):03d} "
                f"elev={view['elevation_deg']:6.1f}",
                flush=True,
            )
    finally:
        for prim in scene["canonical_scan_occluders"]:
            UsdGeom.Imageable(prim).MakeVisible()
        configure_cameras(scene["cameras"], scene["camera_data"], args)
        world.render()

    manifest = {
        "schema": "fixedsuperbest.canonical_rgbd_scan.v1",
        "purpose": "静止组织完整几何与外观初始化；不作为任务帧或未来预测观测",
        "resolution": [args.width, args.height],
        "camera_axes": (
            "X_WC_ros_optical 为相机到世界的 ROS 光学轴变换；"
            "X_WC_blender 为相机到世界的 Blender/OpenGL 轴变换"
        ),
        "isolated_tissue": True,
        "contains_material_region_or_stiffness_labels": False,
        "radius_m": float(args.canonical_scan_radius_m),
        "view_count": len(records),
        "coverage": "12 方位角×4 仰角（默认）+正上/正下极点，覆盖顶面、底面和侧缘",
        "views": records,
    }
    dump_json(output / "canonical_scan" / "cameras.json", manifest)
    return manifest


def make_joint_state(robot: Articulation, q7: np.ndarray) -> torch.Tensor:
    state = robot.data.default_joint_pos.clone()
    index = {name: i for i, name in enumerate(robot.joint_names)}
    state[:, index["psm_yaw_joint"]] = float(q7[0])
    state[:, index["psm_pitch_end_joint"]] = float(q7[1])
    state[:, index["psm_main_insertion_joint"]] = float(q7[2])
    state[:, index["psm_tool_roll_joint"]] = float(q7[3])
    state[:, index["psm_tool_pitch_joint"]] = float(q7[4])
    state[:, index["psm_tool_yaw_joint"]] = float(q7[5])
    state[:, index["psm_tool_gripper1_joint"]] = -0.5 * float(q7[6])
    state[:, index["psm_tool_gripper2_joint"]] = 0.5 * float(q7[6])
    return state


def psm_trajectory(
    progress: float,
    args: argparse.Namespace,
) -> tuple[str, np.ndarray, bool]:
    contact = np.asarray(args.psm_contact_q7, dtype=np.float32)
    ready = np.asarray(args.psm_ready_q7, dtype=np.float32)
    closed = contact.copy()
    closed[6] = 0.14
    lifted = np.asarray(args.psm_lift_q7, dtype=np.float32)
    pulled = np.asarray(args.psm_pull_q7, dtype=np.float32)

    if progress < 0.12:
        return "ready", ready, False
    if progress < 0.28:
        alpha = smoothstep((progress - 0.12) / 0.16)
        return "approach", ready * (1.0 - alpha) + contact * alpha, False
    if progress < 0.40:
        alpha = smoothstep((progress - 0.28) / 0.12)
        q7 = contact.copy()
        q7[6] = contact[6] * (1.0 - alpha) + closed[6] * alpha
        return "close", q7, progress >= 0.34
    if progress < 0.68:
        alpha = smoothstep((progress - 0.40) / 0.28)
        return "lift", closed * (1.0 - alpha) + lifted * alpha, True
    if progress < 0.80:
        alpha = smoothstep((progress - 0.68) / 0.12)
        phase = "lower" if args.motion_profile == "edge_lift_return" else "lateral_pull"
        return phase, lifted * (1.0 - alpha) + pulled * alpha, True
    if progress < 0.90:
        phase = "placed_hold" if args.motion_profile == "edge_lift_return" else "hold"
        return phase, pulled, True
    if progress < 0.96:
        alpha = smoothstep((progress - 0.90) / 0.06)
        q7 = pulled.copy()
        q7[6] = pulled[6] * (1.0 - alpha) + ready[6] * alpha
        return "release", q7, progress < 0.92
    return "retreat", ready, False


def build_region_seams(rest_worlds: list[torch.Tensor]) -> list[tuple[int, int, torch.Tensor, torch.Tensor]]:
    """用互为最近邻的仿真节点构建相邻区域的无缝运动约束。"""

    seams = []
    seam_degree = np.zeros(len(rest_worlds), dtype=np.int32)
    for left_region in range(len(rest_worlds)):
        for right_region in range(left_region + 1, len(rest_worlds)):
            left = to_numpy(rest_worlds[left_region])[0]
            right = to_numpy(rest_worlds[right_region])[0]
            right_tree = cKDTree(right)
            left_tree = cKDTree(left)
            left_distance, left_to_right = right_tree.query(left, k=1)
            _, right_to_left = left_tree.query(right, k=1)
            left_ids = np.arange(len(left), dtype=np.int64)
            mutual = right_to_left[left_to_right] == left_ids
            # 48 级六面体网格的边长约 1 mm。0.8 mm 只连接真正共享边界的节点，
            # 不会把相邻区域内部大面积钉死。
            accepted = mutual & (left_distance <= 0.0008)
            left_ids = left_ids[accepted]
            right_ids = left_to_right[accepted].astype(np.int64)
            # 斜对角区域最多只会在角点附近偶遇几个节点；至少 8 对才认定为共享边。
            if len(left_ids) < 8:
                continue
            device = rest_worlds[left_region].device
            seams.append(
                (
                    left_region,
                    right_region,
                    torch.as_tensor(left_ids, device=device, dtype=torch.long),
                    torch.as_tensor(right_ids, device=device, dtype=torch.long),
                )
            )
            seam_degree[left_region] += 1
            seam_degree[right_region] += 1
    isolated = np.flatnonzero(seam_degree == 0)
    if len(rest_worlds) > 1 and len(isolated):
        raise RuntimeError(f"以下随机组织区域没有物理缝合邻居：{isolated.tolist()}")
    return seams


def update_tissue_targets(
    deformable_views: list[DeformablePrimView],
    rest_worlds: list[torch.Tensor],
    anchor_masks: list[torch.Tensor],
    grasp_masks: list[torch.Tensor],
    seams: list[tuple[int, int, torch.Tensor, torch.Tensor]],
    displacement: np.ndarray,
    grasped: bool,
) -> None:
    targets = []
    current_positions = [view.get_simulation_mesh_nodal_positions() for view in deformable_views]
    for rest_world, anchor_mask, grasp_mask in zip(rest_worlds, anchor_masks, grasp_masks):
        target = torch.zeros((*rest_world.shape[:2], 4), device=rest_world.device, dtype=rest_world.dtype)
        target[..., :3] = rest_world
        target[..., 3] = 1.0
        target[:, anchor_mask, 3] = 0.0
        if grasped:
            delta = torch.as_tensor(displacement, device=rest_world.device, dtype=rest_world.dtype)
            target[:, grasp_mask, :3] = rest_world[:, grasp_mask, :] + delta
            target[:, grasp_mask, 3] = 0.0
        targets.append(target)

    for left_region, right_region, left_ids, right_ids in seams:
        stitched = 0.5 * (
            current_positions[left_region][:, left_ids, :]
            + current_positions[right_region][:, right_ids, :]
        )
        targets[left_region][:, left_ids, :3] = stitched
        targets[left_region][:, left_ids, 3] = 0.0
        targets[right_region][:, right_ids, :3] = stitched
        targets[right_region][:, right_ids, 3] = 0.0

    for view, target in zip(deformable_views, targets):
        view.set_simulation_mesh_kinematic_targets(target)


def body_pose_matrices(robot: Articulation) -> np.ndarray:
    positions = to_numpy(robot.data.body_pos_w)[0]
    quaternions = to_numpy(robot.data.body_quat_w)[0]
    matrices = np.repeat(np.eye(4, dtype=np.float32)[None, ...], len(positions), axis=0)
    for index, (position, quaternion) in enumerate(zip(positions, quaternions)):
        w, x, y, z = [float(value) for value in quaternion]
        matrices[index, :3, :3] = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )
        matrices[index, :3, 3] = position
    return matrices


def distal_gripper_contact_point(vertices: np.ndarray) -> np.ndarray:
    """返回官方夹爪局部 +y 最前端网格的几何中心。"""

    threshold = float(np.quantile(vertices[:, 1], 0.92))
    return vertices[vertices[:, 1] >= threshold].mean(axis=0).astype(np.float32)


def psm_jaw_contact_geometry(robot: Articulation) -> dict[str, np.ndarray | float]:
    """用真实夹爪网格而非重合的关节原点计算上下夹持间隙。"""

    if not PSM_TIP_MESH_ASSET.is_file():
        raise FileNotFoundError(f"缺少官方 PSM 夹爪网格：{PSM_TIP_MESH_ASSET}")
    with np.load(PSM_TIP_MESH_ASSET, allow_pickle=False) as mesh_asset:
        jaw1_local = distal_gripper_contact_point(
            mesh_asset["psm_tool_gripper1_link__vertices"]
        )
        jaw2_local = distal_gripper_contact_point(
            mesh_asset["psm_tool_gripper2_link__vertices"]
        )
    poses = body_pose_matrices(robot)
    jaw1_pose = poses[robot.body_names.index("psm_tool_gripper1_link")]
    jaw2_pose = poses[robot.body_names.index("psm_tool_gripper2_link")]
    jaw1_world = jaw1_pose[:3, :3] @ jaw1_local + jaw1_pose[:3, 3]
    jaw2_world = jaw2_pose[:3, :3] @ jaw2_local + jaw2_pose[:3, 3]
    separation = jaw2_world - jaw1_world
    separation_norm = float(np.linalg.norm(separation))
    return {
        "jaw1_world": jaw1_world.astype(np.float32),
        "jaw2_world": jaw2_world.astype(np.float32),
        "midpoint_world": (0.5 * (jaw1_world + jaw2_world)).astype(np.float32),
        "separation_world": separation.astype(np.float32),
        "separation_mm": separation_norm * 1.0e3,
        "vertical_fraction": abs(float(separation[2])) / max(separation_norm, 1.0e-9),
    }


def write_chinese_readme(output: Path, args: argparse.Namespace) -> None:
    if args.scene_style == "sufia_viewpoint_train":
        visual_profile = "official_body_fascia"
        geometry_profile = "legacy_fascia_sheet"
        if args.tissue_trajectory is not None:
            with np.load(args.tissue_trajectory, allow_pickle=False) as trajectory:
                visual_profile = str(
                    np.asarray(
                        trajectory.get(
                            "visual_material_profile", "official_body_fascia"
                        )
                    ).item()
                )
                geometry_profile = str(
                    np.asarray(
                        trajectory.get("geometry_profile", "legacy_fascia_sheet")
                    ).item()
                )
        if visual_profile == "official_liver":
            tissue_description = """- 可变形组织：本地构建的不规则曲面肝组织瓣，具有非对称叶状轮廓、自然凹口和
  约 3--4.5 mm 的平滑渐变厚度。表面直接使用官方 4096×4096 Liver
  diffuse/normal 纹理，并设置高粗糙度、低镜面反射以保留血管状和斑驳细节；
  没有生成随机纹理，也没有把材料区域编码进颜色。它不是 SuFIA-BC 未公开的
  原始 tissue-retraction 任务网格。"""
        elif geometry_profile == "rectangular_tissue_strip":
            tissue_description = """- 可变形组织：本地构建的长宽比约 1.7:1 长方形薄组织体，长轴沿画面
  左右方向，夹持点位于靠近相机的一条长边而非中心。厚度约 1.1 mm，边缘带轻微
  圆角，表面具有低频自然弧度。它直接使用官方 4096×4096 Body diffuse/normal
  纹理，并采用高粗糙度、低镜面反射；没有生成随机纹理或将刚度编码进颜色。"""
        else:
            tissue_description = """- 可变形组织：本地构建的约 0.9 mm 浅色不规则薄层体，直接使用官方 4096×4096
  Body diffuse/normal 纹理。为避免当前手术灯下的高光淹没纹理，组织使用固定
  roughness=0.88、低镜面反射，并只对官方 albedo 做通道级颜色/对比度增强；没有
  生成随机纹理，也没有把材料区域编码进颜色。它不是 SuFIA-BC 未公开的原始
  tissue-retraction 任务网格。"""
        scene_description = f"""- 场景：以 SuFIA-BC 主页 `Viewpoint Robustness` 第一组
  `Tissue Retraction (train)` 为视觉参考，保留官方完整腹部躯干、骨性结构和多器官背景。
{tissue_description}
- 区域柔软度：组织是一张连续四面体网格，按固定种子的随机 Voronoi 划分成恰好
  10 个不规则区域；不存在分块网格接缝。每个四面体使用所属区域的独立杨氏模量，
  并由 FixedSuperBest 材料感知 Neo-Hookean XPBD 真正参与每帧形变计算。
  随机种子为 {args.regional_seed}，每区杨氏模量在
  [{args.regional_youngs_min_pa:.1f}, {args.regional_youngs_max_pa:.1f}] Pa 内分层对数随机采样，
  再随机分配到空间区域，以保证软硬差异覆盖整个给定范围。
  它们是人为明确规定、可完全复现的合成 benchmark 真值，不冒充人体实测材料参数。
"""
    else:
        scene_description = """- 肝脏：SuFIA-BC/ORBIT-Surgical 发布的 CT 解剖肝脏网格与 PBR 纹理。
- 可变形组织：本地重新构建的圆角薄层体，使用官方肝脏 PBR 纹理；它不是
  SuFIA-BC 未公开的原始 tissue-retraction 任务网格。
"""
    text = f"""# 官方资产组织牵拉仿真数据集

本数据集由本仓库的 `scripts/generate_sim_tissue_retraction_dataset.py` 生成，分辨率为
{args.width}×{args.height}，共 {args.frames} 帧，采样频率 {args.fps} Hz，物理频率
{args.physics_hz} Hz。

正式 RGB 使用 `{args.renderer}` 渲染，每像素每帧 {args.samples_per_pixel} 个样本。

## 资产来源

- 器械：ORBIT-Surgical 发布的 dVRK PSM OpenUSD 资产。
{scene_description.rstrip()}
- 任务轨迹、组织约束、相机和数据导出代码均为本地实现。

因此，本数据集是“使用官方资产复现的组织牵拉任务”，不能表述成官方 SuFIA-BC
任务代码的原样运行结果。

## 目录说明

- `canonical_scan/`：任务开始前、静止组织与腹部/器械隔离后的全视角 RGB-D、组织
  mask 和逐视角相机标定。默认 12 个方位角×4 个仰角，再加正上/正下两个极点，
  共 50 个视角，覆盖组织顶面、底面和侧缘；用于完整初始几何/颜色建模。
- `videos/stereo_*.mp4`、`cameras.json`、`robots.json`：FixedSuperBest GUI 输入。
- `rgb/`：无损主 RGB 帧；`videos/*_lossless_rgb.mp4` 是无损归档，普通 MP4
  采用 ThinLinc/GUI 通用的 H.264 High/yuv420p 高质量编码并启用 fast-start，不能
  反向替代 PNG 主数据。
- `ground_truth/depth/`：仿真深度真值。
- `ground_truth/masks/`：组织、PSM 和肝脏的逐帧语义 mask。
- `ground_truth/tissue_state.npz`：组织仿真/碰撞节点、速度、拓扑和约束集合。
- `ground_truth/psm_link_poses.npz`：官方 PSM 每个刚体的逐帧世界位姿。
- `ground_truth/trajectories_3d.npz`：组织顶面密集点、评估子集、PSM 连杆与尖端的
  世界坐标 3D 真值轨迹。
- `ground_truth/trajectories_2d/stereo_*.npz`：上述点在双目图像中的连续像素坐标、
  相机坐标、视锥内标记、深度/语义联合可见性与遮挡标记。
- `ground_truth/material.json`：十区域逐四面体材料真值，仅允许评估程序读取。
- `task_inputs/psm_link_poses.npz`、`task_inputs/phases.json`：重建可用的已知
  PSM link 位姿和夹持开合时序；不包含组织形变、材料或未来组织状态。
- `evaluation/rendering_metrics_protocol.json`：重建完成后计算 PSNR、SSIM、LPIPS 的
  固定协议；指标同时报告全图和组织 mask 区域。

## 信息隔离

重建初始化允许读取 `canonical_scan/`，任务阶段允许读取视频、相机参数、
`robots.json` 和 `task_inputs/` 中的已知 PSM 控制。全视角扫描只含静止组织
RGB-D/mask/标定，不含十区域 ID、刚度或未来组织运动。`ground_truth/` 是特权信息，
只用于计算 2D/3D 轨迹误差和渲染参考指标，不能反馈给优化过程。

实验序列固定以前 80%（第 0--{int(args.frames * 0.8) - 1} 帧）在线估计状态/参数；
后 20%（第 {int(args.frames * 0.8)}--{args.frames - 1} 帧）
冻结状态、材料与外观参数，关闭视觉残差和在线刚度修正，只根据已知 PSM 控制开环预测。
最终只报告 3D/2D 点跟踪误差以及 PSNR、SSIM、LPIPS，不报告刚度误差。

## 柔软度状态

本次材料来源标记为 `{args.material_provenance}`。只有该字段指向已完成的实验数据拟合、
并同时保存拟合脚本和误差报告时，材料参数才能用于刚度恢复评估；
`preview_only_unvalidated_not_for_stiffness_evaluation` 只允许做画面和数据管线验收。

## 渲染指标

重建程序需要把左右相机预测图保存为 `<预测目录>/rgb/stereo_left/%06d.png` 和
`<预测目录>/rgb/stereo_right/%06d.png`。随后运行：

```bash
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/evaluate_sim_rendering_metrics.py \
  --reference-dataset <本数据集目录> \
  --prediction-dir <预测目录> \
  --output <本数据集目录>/evaluation/reconstruction_render_metrics.json
```

数据生成阶段没有“重建预测图”，因此不会伪造 PSNR/SSIM/LPIPS 数值；正式数值在重建
完成后由同一脚本计算。恒等输入的实现自检应得到 PSNR=∞、SSIM=1、LPIPS=0。

组织轨迹评估使用固定的 240 个顶面节点，不做 SE(3)、尺度或时间对齐：

```bash
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/evaluate_sim_trajectory_metrics.py \
  --reference-dataset <本数据集目录> \
  --prediction <重建轨迹.npz> \
  --output <本数据集目录>/evaluation/reconstruction_trajectory_metrics.json
```

脚本报告 3D mean/RMSE/median/p95（mm）及 1/2/5/10 mm 成功率，也报告双目
2D EPE 分布和 PCK@1/3/5/10 pixel；2D 只在真值可见的点上统计。
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = ARGS
    ensure_assets()
    output = args.output.expanduser().resolve()
    ensure_new_output_tree(output, args.canonical_scan)
    configure_rendering()

    world = World(
        physics_dt=1.0 / args.physics_hz,
        rendering_dt=1.0 / args.fps,
        stage_units_in_meters=1.0,
        backend="torch",
        device="cuda",
    )
    world.get_physics_context().set_gravity(0.0)
    scene = create_scene(world, args)
    robot: Articulation = scene["robot"]
    deformable_views: list[DeformablePrimView] = scene["deformable_views"]
    cameras: dict[str, Camera] = scene["cameras"]
    camera_data: dict[str, dict] = scene["camera_data"]

    world.reset(soft=False)
    configure_cameras(cameras, camera_data, args)
    robot.update(world.get_physics_dt())
    tip_index = robot.body_names.index("psm_tool_tip_link")

    contact_q7 = np.asarray(args.psm_contact_q7, dtype=np.float32).copy()
    contact_q7[6] = 0.14
    contact_joint_state = make_joint_state(robot, contact_q7)
    robot.write_joint_state_to_sim(contact_joint_state, torch.zeros_like(contact_joint_state))
    robot.update(0.0)
    world.step(render=False)
    robot.update(world.get_physics_dt())
    contact_tip = to_numpy(robot.data.body_pos_w)[0, tip_index].astype(np.float32)

    world.reset(soft=True)
    configure_cameras(cameras, camera_data, args)
    _, ready_q7, _ = psm_trajectory(0.0, args)
    ready_joint_state = make_joint_state(robot, ready_q7)
    robot.write_joint_state_to_sim(ready_joint_state, torch.zeros_like(ready_joint_state))
    robot.set_joint_position_target(ready_joint_state)
    robot.write_data_to_sim()
    world.step(render=False)
    robot.update(world.get_physics_dt())
    for _ in range(12):
        world.render()

    if args.preview_frame_index is not None:
        if scene["tissue_trajectory"] is None:
            raise ValueError("代表帧预览目前要求预计算的连续组织轨迹")
        frame_index = int(args.preview_frame_index)
        progress = frame_index / float(args.frames - 1)
        phase, q7, grasped = psm_trajectory(progress, args)
        joint_state = make_joint_state(robot, q7)
        robot.write_joint_state_to_sim(
            joint_state, torch.zeros_like(joint_state)
        )
        robot.set_joint_position_target(joint_state)
        robot.write_data_to_sim()
        world.step(render=False)
        robot.update(world.get_physics_dt())
        frame_points = scene["tissue_trajectory"]["visual_positions"][frame_index]
        scene["tissue_meshes"][0].GetPointsAttr().Set(
            [
                Gf.Vec3f(float(point[0]), float(point[1]), float(point[2]))
                for point in frame_points
            ]
        )
        for _ in range(8):
            world.render()

        preview_dir = output / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_images = []
        camera_statistics = {}
        for camera_name, camera in cameras.items():
            rgb = normalize_rgb(camera.get_rgb())
            rgb_path = preview_dir / f"{camera_name}_frame_{frame_index:06d}.png"
            Image.fromarray(rgb, mode="RGB").save(rgb_path, compress_level=1)
            preview_images.append(rgb)
            payload = get_camera_annotator_data(camera, "semantic_segmentation")
            masks, _ = semantic_payload_to_masks(payload, ("tissue", "psm", "liver"))
            tissue_mask = masks["tissue"] > 0
            psm_mask = masks["psm"] > 0
            Image.fromarray(tissue_mask.astype(np.uint8) * 255, mode="L").save(
                preview_dir / f"{camera_name}_tissue_mask.png", compress_level=1
            )
            y, x = np.nonzero(tissue_mask)
            tissue_bbox = (
                [int(x.min()), int(y.min()), int(x.max()), int(y.max())]
                if len(x)
                else None
            )
            camera_statistics[camera_name] = {
                "tissue_pixels": int(tissue_mask.sum()),
                "tissue_fraction": float(tissue_mask.mean()),
                "tissue_bbox_xyxy": tissue_bbox,
                "psm_pixels": int(psm_mask.sum()),
                "psm_touches_image_border": bool(
                    psm_mask[0].any()
                    or psm_mask[-1].any()
                    or psm_mask[:, 0].any()
                    or psm_mask[:, -1].any()
                ),
            }
        tool_tip_world = to_numpy(robot.data.body_pos_w)[0, tip_index].astype(np.float32)
        jaw_geometry = psm_jaw_contact_geometry(robot)
        jaw1_world = jaw_geometry["jaw1_world"]
        jaw2_world = jaw_geometry["jaw2_world"]
        jaw_midpoint_world = jaw_geometry["midpoint_world"]
        jaw_separation_world = jaw_geometry["separation_world"]
        grasp_mask = scene["tissue_trajectory"]["grasp_mask"].astype(bool)
        grasp_centroid_world = frame_points[grasp_mask].mean(axis=0).astype(np.float32)
        stereo = np.concatenate(preview_images, axis=1)
        Image.fromarray(stereo, mode="RGB").save(
            preview_dir / f"stereo_frame_{frame_index:06d}.png", compress_level=1
        )
        dump_json(
            preview_dir / "preview.json",
            {
                "frame_index": frame_index,
                "phase": phase,
                "grasped": bool(grasped),
                "psm_q7": q7.tolist(),
                "tool_tip_world_m": tool_tip_world.tolist(),
                "jaw1_contact_world_m": jaw1_world.tolist(),
                "jaw2_contact_world_m": jaw2_world.tolist(),
                "jaw_midpoint_world_m": jaw_midpoint_world.tolist(),
                "jaw_separation_vector_m": jaw_separation_world.tolist(),
                "jaw_separation_mm": jaw_geometry["separation_mm"],
                "jaw_vertical_fraction": jaw_geometry["vertical_fraction"],
                "tissue_grasp_centroid_world_m": grasp_centroid_world.tolist(),
                "tip_to_grasp_centroid_mm": float(
                    np.linalg.norm(tool_tip_world - grasp_centroid_world) * 1.0e3
                ),
                "jaw_midpoint_to_grasp_centroid_mm": float(
                    np.linalg.norm(jaw_midpoint_world - grasp_centroid_world) * 1.0e3
                ),
                "task_camera_eye": (
                    list(args.task_camera_eye)
                    if args.task_camera_eye is not None
                    else [0.085, -0.205, 0.172]
                ),
                "task_camera_target": (
                    list(args.task_camera_target)
                    if args.task_camera_target is not None
                    else [0.008, 0.0, 0.032]
                ),
                "tissue_uv_detail_scale": float(args.tissue_uv_detail_scale),
                "tissue_texture_contrast": float(args.tissue_texture_contrast),
                "tissue_texture_bias": list(args.tissue_texture_bias),
                "camera_statistics": camera_statistics,
            },
        )
        print(
            f"[预览完成] phase={phase}, frame={frame_index}, "
            f"output={preview_dir}",
            flush=True,
        )
        return

    canonical_scan_manifest = None
    if args.canonical_scan:
        canonical_scan_manifest = capture_canonical_tissue_scan(world, scene, output, args)

    tissue_trajectory = scene["tissue_trajectory"]
    if tissue_trajectory is not None:
        # 这是一个连续网格，区域只标记逐节点/逐四面体材料，不存在拼接缝。
        rest_worlds = []
        anchor_masks = [tissue_trajectory["anchor_mask"].astype(bool)]
        grasp_masks = [tissue_trajectory["grasp_mask"].astype(bool)]
        seams = []
        simulation_indices = tissue_trajectory["tet_indices"].astype(np.int32)
        collision_indices = tissue_trajectory["visual_faces"].astype(np.int32)
        simulation_rest_local = tissue_trajectory["rest_positions"].astype(np.float32)
        simulation_region_ids = tissue_trajectory["node_region_ids"].astype(np.int16)
        simulation_tet_region_ids = tissue_trajectory["tet_region_ids"].astype(np.int16)
        collision_region_ids = tissue_trajectory["node_region_ids"].astype(np.int16)
        seam_pairs = np.empty((0, 2), dtype=np.int64)
    else:
        rest_worlds = [
            view.get_simulation_mesh_nodal_positions().clone()
            for view in deformable_views
        ]
        simulation_rest_local_tensors = [
            view.get_simulation_mesh_rest_points() for view in deformable_views
        ]
        initial_rest_error = max(
            float(torch.linalg.vector_norm(rest_world - rest_local, dim=-1).max().item())
            for rest_world, rest_local in zip(rest_worlds, simulation_rest_local_tensors)
        )
        if initial_rest_error > 5.0e-5:
            raise RuntimeError(
                "第 0 帧组织已偏离仿真静止形状："
                f"max={initial_rest_error * 1e3:.4f} mm"
            )
        anchor_masks = []
        grasp_masks = []
        for rest_world in rest_worlds:
            rest_single = rest_world[0]
            x_min, x_max = rest_single[:, 0].min(), rest_single[:, 0].max()
            y_center = torch.median(rest_single[:, 1])
            anchor_masks.append(rest_single[:, 0] < x_min + 0.006)
            grasp_masks.append(
                (rest_single[:, 0] > x_max - 0.015)
                & (torch.abs(rest_single[:, 1] - y_center) < 0.018)
            )
        if sum(int(mask.sum()) for mask in anchor_masks) == 0 or sum(
            int(mask.sum()) for mask in grasp_masks
        ) == 0:
            raise RuntimeError("组织锚点或抓取点集合为空")
        seams = build_region_seams(rest_worlds)

        simulation_node_counts = [rest.shape[1] for rest in rest_worlds]
        collision_node_counts = [
            view.get_collision_mesh_nodal_positions().shape[1]
            for view in deformable_views
        ]
        simulation_node_offsets = np.cumsum(
            [0, *simulation_node_counts[:-1]], dtype=np.int64
        )
        collision_node_offsets = np.cumsum(
            [0, *collision_node_counts[:-1]], dtype=np.int64
        )
        simulation_index_parts = []
        collision_index_parts = []
        simulation_region_parts = []
        simulation_tet_region_parts = []
        collision_region_parts = []
        simulation_rest_parts = []
        for region_index, (view, rest_local, sim_offset, collision_offset) in enumerate(
            zip(
                deformable_views,
                simulation_rest_local_tensors,
                simulation_node_offsets,
                collision_node_offsets,
            )
        ):
            region_sim_indices = to_numpy(view.get_simulation_mesh_indices())[0].astype(np.int64)
            region_collision_indices = to_numpy(view.get_collision_mesh_indices())[0].astype(np.int64)
            simulation_index_parts.append(region_sim_indices + sim_offset)
            collision_index_parts.append(region_collision_indices + collision_offset)
            simulation_region_parts.append(
                np.full(simulation_node_counts[region_index], region_index, dtype=np.int16)
            )
            simulation_tet_region_parts.append(
                np.full(len(region_sim_indices), region_index, dtype=np.int16)
            )
            collision_region_parts.append(
                np.full(collision_node_counts[region_index], region_index, dtype=np.int16)
            )
            simulation_rest_parts.append(to_numpy(rest_local)[0])
        simulation_indices = np.concatenate(simulation_index_parts, axis=0)
        collision_indices = np.concatenate(collision_index_parts, axis=0)
        simulation_rest_local = np.concatenate(simulation_rest_parts, axis=0)
        simulation_region_ids = np.concatenate(simulation_region_parts)
        simulation_tet_region_ids = np.concatenate(simulation_tet_region_parts)
        collision_region_ids = np.concatenate(collision_region_parts)
        seam_pair_parts = []
        for left_region, right_region, left_ids, right_ids in seams:
            left_global = to_numpy(left_ids) + simulation_node_offsets[left_region]
            right_global = to_numpy(right_ids) + simulation_node_offsets[right_region]
            seam_pair_parts.append(np.stack((left_global, right_global), axis=1))
        seam_pairs = (
            np.concatenate(seam_pair_parts, axis=0).astype(np.int64)
            if seam_pair_parts
            else np.empty((0, 2), dtype=np.int64)
        )

    timestamps = np.arange(args.frames, dtype=np.float64) / args.fps
    duration_s = float(timestamps[-1])
    substeps = args.physics_hz // args.fps
    previous_time = 0.0

    sim_positions = []
    sim_velocities = []
    collision_positions = []
    psm_q7_history = []
    psm_joint_history = []
    psm_link_pose_history = []
    psm_tip_history = []
    phase_records = []
    semantic_info = {}

    for frame_index, timestamp in enumerate(timestamps):
        if frame_index > 0:
            for substep_index in range(substeps):
                alpha = (substep_index + 1) / substeps
                physics_time = previous_time + alpha * (timestamp - previous_time)
                progress = physics_time / duration_s
                _, q7, grasped = psm_trajectory(progress, args)
                joint_state = make_joint_state(robot, q7)
                robot.write_joint_state_to_sim(joint_state, torch.zeros_like(joint_state))
                robot.set_joint_position_target(joint_state)
                robot.write_data_to_sim()
                robot.update(0.0)
                tip = to_numpy(robot.data.body_pos_w)[0, tip_index]
                displacement = tip - contact_tip if grasped else np.zeros(3, dtype=np.float32)
                if tissue_trajectory is None:
                    update_tissue_targets(
                        deformable_views,
                        rest_worlds,
                        anchor_masks,
                        grasp_masks,
                        seams,
                        displacement,
                        grasped,
                    )
                world.step(render=False)
                robot.update(world.get_physics_dt())

        progress = float(timestamp / duration_s)
        phase, q7, grasped = psm_trajectory(progress, args)
        joint_state = make_joint_state(robot, q7)
        robot.write_joint_state_to_sim(joint_state, torch.zeros_like(joint_state))
        robot.set_joint_position_target(joint_state)
        robot.write_data_to_sim()
        robot.update(0.0)
        tip = to_numpy(robot.data.body_pos_w)[0, tip_index].astype(np.float32)
        displacement = tip - contact_tip if grasped else np.zeros(3, dtype=np.float32)
        if tissue_trajectory is None:
            update_tissue_targets(
                deformable_views,
                rest_worlds,
                anchor_masks,
                grasp_masks,
                seams,
                displacement,
                grasped,
            )
        else:
            frame_points = tissue_trajectory["visual_positions"][frame_index]
            scene["tissue_meshes"][0].GetPointsAttr().Set(
                [
                    Gf.Vec3f(float(point[0]), float(point[1]), float(point[2]))
                    for point in frame_points
                ]
            )
        world.render()

        for camera_name, camera in cameras.items():
            rgb = normalize_rgb(camera.get_rgb())
            if rgb.shape[:2] != (args.height, args.width):
                raise RuntimeError(f"{camera_name} RGB 尺寸错误：{rgb.shape}")
            Image.fromarray(rgb, mode="RGB").save(
                output / "rgb" / camera_name / f"{frame_index:06d}.png", compress_level=1
            )

            depth = np.asarray(get_camera_annotator_data(camera, "distance_to_image_plane"), dtype=np.float32)
            if depth.size == 0:
                raise RuntimeError(f"{camera_name} 深度帧为空")
            np.save(
                output / "ground_truth" / "depth" / camera_name / f"{frame_index:06d}.npy",
                depth,
            )

            payload = get_camera_annotator_data(camera, "semantic_segmentation")
            masks, info = semantic_payload_to_masks(payload, ("tissue", "psm", "liver"))
            for label, mask in masks.items():
                Image.fromarray(mask, mode="L").save(
                    output
                    / "ground_truth"
                    / "masks"
                    / label
                    / camera_name
                    / f"{frame_index:06d}.png",
                    compress_level=1,
                )
            semantic_info[camera_name] = info

        psm_q7_history.append(q7.copy())
        psm_joint_history.append(to_numpy(joint_state)[0].copy())
        psm_link_pose_history.append(body_pose_matrices(robot))
        psm_tip_history.append(tip.copy())
        phase_records.append(
            {
                "frame": frame_index,
                "timestamp": float(timestamp),
                "phase": phase,
                "grasped": bool(grasped),
            }
        )
        if tissue_trajectory is not None:
            frame_positions = tissue_trajectory["visual_positions"][frame_index]
            sim_positions.append(frame_positions.copy())
            sim_velocities.append(tissue_trajectory["visual_velocities"][frame_index].copy())
            collision_positions.append(frame_positions.copy())
        else:
            sim_positions.append(
                np.concatenate(
                    [
                        to_numpy(view.get_simulation_mesh_nodal_positions())[0]
                        for view in deformable_views
                    ],
                    axis=0,
                ).copy()
            )
            sim_velocities.append(
                np.concatenate(
                    [
                        to_numpy(view.get_simulation_mesh_nodal_velocities())[0]
                        for view in deformable_views
                    ],
                    axis=0,
                ).copy()
            )
            collision_positions.append(
                np.concatenate(
                    [
                        to_numpy(view.get_collision_mesh_nodal_positions())[0]
                        for view in deformable_views
                    ],
                    axis=0,
                ).copy()
            )
        previous_time = float(timestamp)
        if frame_index == 0 or (frame_index + 1) % args.fps == 0 or frame_index + 1 == args.frames:
            print(
                f"[仿真数据] {frame_index + 1:04d}/{args.frames:04d} "
                f"t={timestamp:6.3f}s phase={phase}",
                flush=True,
            )

    camera_manifest = {}
    for camera_name, camera in cameras.items():
        intrinsic = to_numpy(camera.get_intrinsics_matrix()).astype(np.float64)
        dump_json(
            output / "videos" / f"{camera_name}.json",
            {
                "serial": camera_name,
                "K": intrinsic.tolist(),
                "resolution": [args.width, args.height],
                "timestamps": timestamps.tolist(),
                "source_images": f"../rgb/{camera_name}/%06d.png",
                "color_space": "sRGB",
                "compression": "lossless_png",
            },
        )
        camera_manifest[camera_name] = {
            "X_WC": camera_data[camera_name]["X_WC_blender"].tolist(),
            "X_WC_ros_optical": camera_data[camera_name]["X_WC_ros"].tolist(),
            "video_path": f"videos/{camera_name}.mp4",
            "lossless_video_path": f"videos/{camera_name}_lossless_rgb.mp4",
            "metadata_path": f"videos/{camera_name}.json",
            "image_directory": f"rgb/{camera_name}",
            "camera_axes": "X_WC 使用 Blender/OpenGL 相机轴；X_WC_ros_optical 使用 ROS 光学轴",
        }
    dump_json(output / "cameras.json", camera_manifest)

    q7_array = np.asarray(psm_q7_history, dtype=np.float32)
    states = []
    for q7, tip, phase in zip(q7_array, psm_tip_history, phase_records):
        states.append(
            {
                "q": q7.tolist(),
                "q_d": [0.0] * 7,
                "effort": [0.0] * 7,
                "names": list(PSM_Q7_NAMES),
                "tool_tip_position_world": np.asarray(tip).tolist(),
                "phase": phase["phase"],
                "grasped": phase["grasped"],
            }
        )
    dump_json(
        output / "robots.json",
        {
            "PSM1": {
                "control": q7_array.tolist(),
                "control_timestamps": timestamps.tolist(),
                "states": states,
                "states_timestamps": timestamps.tolist(),
            }
        },
    )

    np.savez_compressed(
        output / "ground_truth" / "tissue_state.npz",
        timestamps=timestamps,
        simulation_positions=np.asarray(sim_positions, dtype=np.float32),
        simulation_velocities=np.asarray(sim_velocities, dtype=np.float32),
        collision_positions=np.asarray(collision_positions, dtype=np.float32),
        simulation_indices=simulation_indices.astype(np.int32),
        collision_indices=collision_indices.astype(np.int32),
        simulation_rest_local=simulation_rest_local.astype(np.float32),
        visual_rest_points=scene["visual_rest_points"].astype(np.float32),
        visual_faces=scene["visual_faces"].astype(np.int32),
        visual_uvs=(
            tissue_trajectory["visual_uvs"]
            if tissue_trajectory is not None
            else np.empty((0, 2), dtype=np.float32)
        ),
        material_uv=(
            tissue_trajectory["material_uv"]
            if tissue_trajectory is not None
            else np.empty((0, 2), dtype=np.float32)
        ),
        top_face_indices=(
            tissue_trajectory["top_face_indices"]
            if tissue_trajectory is not None
            else np.empty((0,), dtype=np.int32)
        ),
        anchor_mask=np.concatenate([to_numpy(mask) for mask in anchor_masks]).astype(bool),
        grasp_mask=np.concatenate([to_numpy(mask) for mask in grasp_masks]).astype(bool),
        grasp_coupling_weights=(
            tissue_trajectory["grasp_coupling_weights"].astype(np.float32)
            if tissue_trajectory is not None
            else np.concatenate(
                [to_numpy(mask).astype(np.float32) for mask in grasp_masks]
            )
        ),
        support_mode=np.asarray(
            tissue_trajectory.get("support_mode", "edge_hard")
            if tissue_trajectory is not None
            else "edge_hard"
        ),
        motion_profile=np.asarray(args.motion_profile),
        geometry_profile=np.asarray(
            tissue_trajectory.get("geometry_profile", "legacy_fascia_sheet")
            if tissue_trajectory is not None
            else "legacy_physx_sheet"
        ),
        visual_material_profile=np.asarray(scene["tissue_visual_material_profile"]),
        visual_uv_center=(
            tissue_trajectory["visual_uv_center"].astype(np.float32)
            if tissue_trajectory is not None
            and "visual_uv_center" in tissue_trajectory
            else np.empty((0, 2), dtype=np.float32)
        ),
        simulation_region_ids=simulation_region_ids,
        simulation_tet_region_ids=simulation_tet_region_ids,
        collision_region_ids=collision_region_ids,
        visual_region_ids=scene["visual_region_ids"],
        seam_pairs=seam_pairs,
        youngs_modulus_pa_per_simulation_node=scene["region_youngs_modulus_pa"][
            simulation_region_ids
        ],
        youngs_modulus_pa_per_simulation_tet=scene["region_youngs_modulus_pa"][
            simulation_tet_region_ids
        ],
        region_centers_uv=scene["region_centers_uv"],
        region_metric_scale=scene["region_metric_scale"],
        region_tet_counts=scene["region_tet_counts"],
        tet_rest_volumes_m3=(
            tissue_trajectory["tet_rest_volumes_m3"]
            if tissue_trajectory is not None
            else np.empty((0,), dtype=np.float32)
        ),
        particle_mass_kg=(
            tissue_trajectory["particle_mass_kg"]
            if tissue_trajectory is not None
            else np.empty((0,), dtype=np.float32)
        ),
        contact_tip_world=contact_tip,
    )
    psm_pose_payload = dict(
        timestamps=timestamps,
        q7=q7_array,
        articulation_joint_positions=np.asarray(psm_joint_history, dtype=np.float32),
        link_names=np.asarray(robot.body_names),
        X_WL=np.asarray(psm_link_pose_history, dtype=np.float32),
        tool_tip_positions=np.asarray(psm_tip_history, dtype=np.float32),
    )
    np.savez_compressed(
        output / "ground_truth" / "psm_link_poses.npz", **psm_pose_payload
    )
    # Robot poses/grasp state are known task controls. Keep an explicit
    # reconstruction-side copy so algorithms never need to read ground_truth.
    np.savez_compressed(
        output / "task_inputs" / "psm_link_poses.npz", **psm_pose_payload
    )
    dump_json(
        output / "ground_truth" / "material.json",
        {
            "simulator": (
                "FixedSuperBest Warp material-aware XPBD + Isaac Sim 4.0 renderer"
                if tissue_trajectory is not None
                else "Isaac Sim 4.0 / PhysX GPU deformable body"
            ),
            "youngs_modulus_pa_legacy_or_reference": args.youngs_modulus,
            "regional_model": (
                "single continuous tetrahedral body with ten seeded anisotropic-Voronoi per-tet material regions"
                if tissue_trajectory is not None
                else "uniform PhysX deformable body"
            ),
            "geometry_profile": (
                str(np.asarray(tissue_trajectory.get("geometry_profile", "legacy_fascia_sheet")).item())
                if tissue_trajectory is not None
                else "legacy_physx_sheet"
            ),
            "motion_profile": args.motion_profile,
            "visual_material_profile": scene["tissue_visual_material_profile"],
            "regional_seed": int(args.regional_seed),
            "regional_distribution": "stratified_log_uniform_random_permutation",
            "regional_youngs_range_pa": [
                args.regional_youngs_min_pa,
                args.regional_youngs_max_pa,
            ],
            "region_uv_bounds": scene["region_uv_bounds"].tolist(),
            "region_centers_uv": scene["region_centers_uv"].tolist(),
            "region_metric_scale": scene["region_metric_scale"].tolist(),
            "region_tet_counts": scene["region_tet_counts"].tolist(),
            "regional_youngs_modulus_pa": scene["region_youngs_modulus_pa"].tolist(),
            "seam_pair_count": int(len(seam_pairs)),
            "poissons_ratio": args.poissons_ratio,
            "dynamic_friction": 0.55,
            "damping_scale": args.damping_scale,
            "elasticity_damping": args.elasticity_damping,
            "gravity_m_s2": [0.0, 0.0, 0.0],
            "support_mode": (
                str(np.asarray(tissue_trajectory.get("support_mode", "edge_hard")).item())
                if tissue_trajectory is not None
                else "edge_hard"
            ),
            "hard_anchor_nodes": int(
                np.count_nonzero(tissue_trajectory["anchor_mask"])
                if tissue_trajectory is not None
                else sum(int(mask.sum()) for mask in anchor_masks)
            ),
            "psm_collision_prims_disabled": int(scene["psm_disabled_collision_prims"]),
            "tissue_tool_coupling": (
                "8x8 mm kinematic grasp core plus cosine-tapered compliant force transfer"
                if tissue_trajectory is not None
                else "scripted kinematic grasp-region targets"
            ),
            "material_provenance": args.material_provenance,
            "calibration_status": (
                "preview_only_not_for_stiffness_evaluation"
                if args.material_provenance == "preview_only_unvalidated_not_for_stiffness_evaluation"
                else "externally_declared_check_calibration_report"
            ),
            "说明": (
                "十个区域参数是为合成识别基准明确规定的逐四面体 XPBD 真值，不是人体材料实测值；"
                "也不假设与 EG distance/shape stiffness 数值相同。"
            ),
        },
    )
    dump_json(output / "ground_truth" / "phases.json", phase_records)
    dump_json(output / "task_inputs" / "phases.json", phase_records)
    dump_json(output / "ground_truth" / "semantic_labels.json", semantic_info)
    dump_json(
        output / "episode.json",
        {
            "name": output.name,
            "frames": args.frames,
            "fps": args.fps,
            "physics_hz": args.physics_hz,
            "duration_s": duration_s,
            "resolution": [args.width, args.height],
            "renderer": args.renderer,
            "samples_per_pixel_per_frame": args.samples_per_pixel,
            "task": "official-asset PSM tissue retraction",
            "task_variant": args.task_variant,
            "motion_profile": args.motion_profile,
            "scene_style": args.scene_style,
            "geometry_profile": (
                str(np.asarray(tissue_trajectory.get("geometry_profile", "legacy_fascia_sheet")).item())
                if tissue_trajectory is not None
                else "legacy_physx_sheet"
            ),
            "visual_reference": (
                "https://orbit-surgical.github.io/sufia-bc/static/videos/Tissue_view_train.mp4"
                if args.scene_style == "sufia_viewpoint_train"
                else None
            ),
            "official_assets": {
                "psm": str(PSM_USD.relative_to(REPO_ROOT)),
                "organ_model": str(ORGAN_MODEL.relative_to(REPO_ROOT)),
                "native_material_stage": str(ORGAN_MAIN_SCENE.relative_to(REPO_ROOT)),
            },
            "official_anatomy_pbr_mesh_bindings": scene["anatomy_pbr_bindings"],
            "tissue_visual_material": {
                "profile": scene["tissue_visual_material_profile"],
                "source": (
                    "official Liver diffuse/normal maps"
                    if scene["tissue_visual_material_profile"] == "official_liver"
                    else "official Body diffuse/normal maps"
                ),
                "roughness": (
                    0.86
                    if scene["tissue_visual_material_profile"] == "official_liver"
                    else (0.88 if args.scene_style == "sufia_viewpoint_train" else None)
                ),
                "specular_color": (
                    [0.014, 0.014, 0.014]
                    if scene["tissue_visual_material_profile"] == "official_liver"
                    else [0.012, 0.012, 0.012]
                    if args.scene_style == "sufia_viewpoint_train"
                    else None
                ),
                "procedurally_generated_texture": False,
            },
            "canonical_scan": (
                {
                    "path": "canonical_scan",
                    "view_count": canonical_scan_manifest["view_count"],
                    "isolated_tissue": True,
                    "contains_stiffness_labels": False,
                }
                if canonical_scan_manifest is not None
                else None
            ),
            "reconstruction_inputs": (
                [
                    "canonical_scan",
                    "videos",
                    "cameras.json",
                    "robots.json",
                    "task_inputs",
                ]
                if canonical_scan_manifest is not None
                else ["videos", "cameras.json", "robots.json", "task_inputs"]
            ),
            "privileged_evaluation_only": ["ground_truth"],
        },
    )
    write_chinese_readme(output, args)
    print(f"[完成] 数据集已写入：{output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # SimulationApp.close() 在部分 headless 配置中会吞掉随后才打印的回溯；
        # 先明确输出异常，避免生成空目录却没有诊断信息。
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
