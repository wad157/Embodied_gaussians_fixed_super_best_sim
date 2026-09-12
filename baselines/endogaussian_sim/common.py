#!/usr/bin/env python3
"""SIM data and camera adapter for the unmodified EndoGaussian implementation."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from protocol import CAMERAS, INTERNAL_UNITS_PER_METER, dataset_spec, training_frames

# These imports resolve to the pinned upstream checkout through PYTHONPATH set by
# scripts/run_endogaussian_baseline_python.sh.
from scene.cameras import Camera
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2
from utils.system_utils import searchForMaxIteration


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_calibration(dataset_root: Path) -> Dict[str, dict]:
    cameras = read_json(dataset_root / "cameras.json")
    result = {}
    for name in CAMERAS:
        metadata = read_json(dataset_root / "videos" / "{}.json".format(name))
        result[name] = {
            "K": np.asarray(metadata["K"], dtype=np.float64),
            "timestamps": np.asarray(metadata["timestamps"], dtype=np.float64),
            "X_WC_ros_optical": np.asarray(
                cameras[name]["X_WC_ros_optical"], dtype=np.float64
            ),
            "resolution_wh": tuple(int(value) for value in metadata["resolution"]),
        }
    return result


class PackedTissueMasks:
    def __init__(self, dataset_root: Path):
        self.arrays = {}
        self.widths = {}
        for name in CAMERAS:
            root = dataset_root / "gui_assets" / "visual_force_masks" / name
            report = read_json(root / "report.json")
            width, _ = report["resolution_wh"]
            self.widths[name] = int(width)
            self.arrays[name] = np.load(
                root / "tissue_masks_packbits.npy", mmap_mode="r"
            )

    def get(self, camera: str, frame: int) -> np.ndarray:
        packed = np.asarray(self.arrays[camera][frame])
        return np.unpackbits(packed, axis=1, bitorder="big")[
            :, : self.widths[camera]
        ].astype(bool)


def scaled_intrinsic(intrinsic: np.ndarray, downsample: int) -> np.ndarray:
    result = np.asarray(intrinsic, dtype=np.float64).copy()
    result[0, :] /= float(downsample)
    result[1, :] /= float(downsample)
    result[2, :] = np.asarray([0.0, 0.0, 1.0])
    return result


def camera_rt_from_world_from_camera(world_from_camera: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    camera_from_world = np.linalg.inv(np.asarray(world_from_camera, dtype=np.float64))
    # EndoGaussian/3DGS stores R transposed and getWorld2View2 transposes it back.
    return camera_from_world[:3, :3].T, camera_from_world[:3, 3]


def internal_world_from_camera(world_from_camera_m: np.ndarray) -> np.ndarray:
    result = np.asarray(world_from_camera_m, dtype=np.float64).copy()
    result[:3, 3] *= INTERNAL_UNITS_PER_METER
    return result


def load_training_arrays(
    dataset_root: Path,
    depth_root: Path,
    masks: PackedTissueMasks,
    camera: str,
    frame: int,
    downsample: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_path = dataset_root / "rgb" / camera / "{:06d}.png".format(frame)
    image = np.asarray(Image.open(str(image_path)).convert("RGB"), dtype=np.float32) / 255.0
    depth_path = depth_root / camera / "{:06d}-depth.npy".format(frame)
    depth = (
        np.asarray(np.load(str(depth_path)), dtype=np.float32)
        * INTERNAL_UNITS_PER_METER
    )
    tissue = masks.get(camera, frame)
    if downsample > 1:
        height, width = image.shape[:2]
        size = (width // downsample, height // downsample)
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
        tissue = cv2.resize(
            tissue.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    valid_depth = np.isfinite(depth) & (depth > 0.0)
    valid = tissue & valid_depth
    depth = np.where(valid_depth, depth, 0.0).astype(np.float32)
    image_tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    depth_tensor = torch.from_numpy(np.ascontiguousarray(depth))
    mask_tensor = torch.from_numpy(np.ascontiguousarray(valid[None]))
    return image_tensor, depth_tensor, mask_tensor


class SimTrainingData:
    """Loads only allowed prefix observations; it never opens evaluation truth."""

    def __init__(
        self,
        dataset_root: Path,
        dataset_key: str,
        downsample: int = 1,
        init_points: int = 30000,
        seed: int = 0,
    ):
        self.dataset_root = dataset_root.resolve()
        self.spec = dataset_spec(dataset_key)
        self.frames = int(self.spec["frames"])
        self.downsample = int(downsample)
        self.init_points = int(init_points)
        self.seed = int(seed)
        if self.downsample < 1:
            raise ValueError("downsample 必须为正整数")
        if self.init_points < 100:
            raise ValueError("init_points 过小")
        self.depth_root = (
            self.dataset_root / "estimated_depth" / str(self.spec["depth"])
        )
        self.calibration = load_calibration(self.dataset_root)
        self.masks = PackedTissueMasks(self.dataset_root)
        self.allowed_frames = training_frames(self.frames)
        self.train_cameras = self._load_training_cameras()

    def _load_training_cameras(self) -> List[Camera]:
        result = []
        uid = 0
        for frame in self.allowed_frames:
            normalized_time = float(frame) / float(self.frames - 1)
            for camera_name in CAMERAS:
                calibration = self.calibration[camera_name]
                image, depth, mask = load_training_arrays(
                    self.dataset_root,
                    self.depth_root,
                    self.masks,
                    camera_name,
                    frame,
                    self.downsample,
                )
                intrinsic = scaled_intrinsic(calibration["K"], self.downsample)
                width = int(image.shape[2])
                height = int(image.shape[1])
                rotation, translation = camera_rt_from_world_from_camera(
                    internal_world_from_camera(calibration["X_WC_ros_optical"])
                )
                camera = Camera(
                    colmap_id=uid,
                    R=rotation,
                    T=translation,
                    FoVx=focal2fov(float(intrinsic[0, 0]), width),
                    FoVy=focal2fov(float(intrinsic[1, 1]), height),
                    image=image,
                    depth=depth,
                    mask=mask,
                    gt_alpha_mask=None,
                    image_name="{}_{:06d}".format(camera_name, frame),
                    uid=uid,
                    data_device=torch.device("cuda"),
                    time=normalized_time,
                    Znear=0.01,
                    Zfar=1000.0,
                )
                camera.sim_camera_name = camera_name
                camera.sim_frame_index = frame
                camera.sim_intrinsic = intrinsic
                result.append(camera)
                uid += 1
        expected = len(self.allowed_frames) * len(CAMERAS)
        if len(result) != expected:
            raise AssertionError("训练视图数量错误：{} != {}".format(len(result), expected))
        return result

    def initial_point_cloud(self) -> BasicPointCloud:
        rng = np.random.RandomState(self.seed)
        per_view = int(math.ceil(float(self.init_points) / len(self.train_cameras)))
        all_points = []
        all_colors = []
        for camera in self.train_cameras:
            depth = camera.original_depth.numpy()
            valid = camera.mask.squeeze(0).numpy().astype(bool)
            valid_indices = np.flatnonzero(valid.reshape(-1))
            if len(valid_indices) == 0:
                continue
            count = min(per_view, len(valid_indices))
            selected = rng.choice(valid_indices, size=count, replace=False)
            y, x = np.unravel_index(selected, depth.shape)
            z = depth[y, x].astype(np.float64)
            intrinsic = camera.sim_intrinsic
            points_camera = np.stack(
                (
                    (x.astype(np.float64) - intrinsic[0, 2]) / intrinsic[0, 0] * z,
                    (y.astype(np.float64) - intrinsic[1, 2]) / intrinsic[1, 1] * z,
                    z,
                ),
                axis=1,
            )
            calibration = self.calibration[camera.sim_camera_name]
            homogeneous = np.concatenate(
                (points_camera, np.ones((len(points_camera), 1), dtype=np.float64)),
                axis=1,
            )
            points_world = (
                internal_world_from_camera(calibration["X_WC_ros_optical"])
                @ homogeneous.T
            ).T[:, :3]
            colors = (
                camera.original_image[:, y, x].permute(1, 0).numpy().astype(np.float64)
            )
            all_points.append(points_world)
            all_colors.append(colors)
        if not all_points:
            raise ValueError("没有可用于初始化的 FoundationStereo 组织点")
        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        if len(points) < self.init_points:
            selected = rng.choice(len(points), size=self.init_points, replace=True)
        else:
            selected = rng.choice(len(points), size=self.init_points, replace=False)
        points = points[selected].astype(np.float32)
        colors = colors[selected].astype(np.float32)
        normals = np.zeros_like(points, dtype=np.float32)
        return BasicPointCloud(points=points, colors=colors, normals=normals)


class SimScene:
    """Drop-in Scene used only by the external training wrapper."""

    def __init__(
        self,
        args,
        gaussians,
        load_iteration=None,
        shuffle=True,
        resolution_scales=(1.0,),
        load_coarse=False,
    ):
        del shuffle, resolution_scales
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.mode = args.mode
        if load_iteration is not None:
            self.loaded_iter = (
                searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
                if int(load_iteration) == -1
                else int(load_iteration)
            )
        data = SimTrainingData(
            Path(args.source_path),
            args.sim_dataset_key,
            downsample=args.sim_downsample,
            init_points=args.sim_init_points,
            seed=args.sim_seed,
        )
        self.train_camera = data.train_cameras
        self.test_camera = []
        self.video_camera = []
        point_cloud = data.initial_point_cloud()
        xyz_max = point_cloud.points.max(axis=0)
        xyz_min = point_cloud.points.min(axis=0)
        self.maxtime = int(data.frames - 1)
        self.cameras_extent = float(args.camera_extent)
        self.gaussians._deformation.deformation_net.grid.set_aabb(xyz_max, xyz_min)
        if self.loaded_iter is not None:
            prefix = "coarse_iteration_" if load_coarse else "iteration_"
            model_root = os.path.join(
                self.model_path, "point_cloud", prefix + str(self.loaded_iter)
            )
            self.gaussians.load_ply(os.path.join(model_root, "point_cloud.ply"))
            self.gaussians.load_model(model_root)
        else:
            self.gaussians.create_from_pcd(
                point_cloud, self.cameras_extent, self.maxtime
            )

    def save(self, iteration, stage):
        path = os.path.join(
            self.model_path,
            "point_cloud",
            ("coarse_iteration_" if stage == "coarse" else "iteration_")
            + str(iteration),
        )
        self.gaussians.save_ply(os.path.join(path, "point_cloud.ply"))
        self.gaussians.save_deformation(path)

    def getTrainCameras(self, scale=1.0):
        del scale
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        del scale
        return self.test_camera

    def getVideoCameras(self, scale=1.0):
        del scale
        return self.video_camera


class RenderCamera:
    """Minimal full-resolution camera accepted by the upstream renderer."""

    def __init__(
        self,
        intrinsic: np.ndarray,
        world_from_camera: np.ndarray,
        resolution_wh: Tuple[int, int],
        normalized_time: float,
    ):
        width, height = resolution_wh
        rotation, translation = camera_rt_from_world_from_camera(
            internal_world_from_camera(world_from_camera)
        )
        self.image_width = int(width)
        self.image_height = int(height)
        self.FoVx = focal2fov(float(intrinsic[0, 0]), self.image_width)
        self.FoVy = focal2fov(float(intrinsic[1, 1]), self.image_height)
        self.znear = 0.01
        self.zfar = 1000.0
        self.time = float(normalized_time)
        world_view = getWorld2View2(rotation, translation)
        projection = getProjectionMatrix(
            self.znear, self.zfar, self.FoVx, self.FoVy
        ).numpy()
        self.world_view_transform = torch.from_numpy(world_view).transpose(0, 1)
        self.projection_matrix = torch.from_numpy(projection).transpose(0, 1)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def load_trained_gaussians(model_path: Path, iteration: int, sh_degree: int, hyper):
    from scene.gaussian_model import GaussianModel

    model_root = model_path / "point_cloud"
    resolved_iteration = (
        searchForMaxIteration(str(model_root)) if iteration < 0 else int(iteration)
    )
    checkpoint = model_root / "iteration_{}".format(resolved_iteration)
    if not checkpoint.is_dir():
        raise FileNotFoundError("找不到 EndoGaussian fine checkpoint：{}".format(checkpoint))
    gaussians = GaussianModel(sh_degree, hyper)
    gaussians.load_ply(str(checkpoint / "point_cloud.ply"))
    gaussians.load_model(str(checkpoint))
    return gaussians, resolved_iteration


def deformed_centers(gaussians, normalized_time: float) -> torch.Tensor:
    means = gaussians.get_xyz
    selected = gaussians._deformation_table
    result = means.clone()
    if bool(selected.any()):
        times = torch.full(
            (int(selected.sum().item()), 1),
            float(normalized_time),
            dtype=means.dtype,
            device=means.device,
        )
        deformed, _, _, _ = gaussians._deformation(
            means[selected],
            gaussians._scaling[selected],
            gaussians._rotation[selected],
            gaussians._opacity[selected],
            times,
        )
        result[selected] = deformed
    return result
