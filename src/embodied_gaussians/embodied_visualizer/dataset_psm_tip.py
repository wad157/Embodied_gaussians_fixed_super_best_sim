"""在 3-D GUI 中显示仿真数据集记录的官方 PSM 末端。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from pyglet import gl
from scipy.spatial.transform import Rotation

from marsoom.cuda import InstancedMeshRenderer


# roll link 含约 48 cm 的整根插入杆。这里有意只画手腕和夹爪，让 GUI
# 显示用户要的“简短尖端”，而不是再出现一整套旧 SUPER 机械臂。
DISPLAY_LINKS = (
    "psm_tool_pitch_link",
    "psm_tool_yaw_link",
    "psm_tool_gripper1_link",
    "psm_tool_gripper2_link",
)

LINK_COLORS = {
    "psm_tool_pitch_link": (0.38, 0.42, 0.46),
    "psm_tool_yaw_link": (0.58, 0.62, 0.66),
    "psm_tool_gripper1_link": (0.72, 0.74, 0.76),
    "psm_tool_gripper2_link": (0.72, 0.74, 0.76),
}


class DatasetPSMTipRenderer:
    """按数据集 frame index 绘制官方 USD 的逐 link 网格和位姿。"""

    def __init__(
        self,
        mesh_asset: Path,
        pose_asset: Path,
        frame_index: Callable[[], int],
    ) -> None:
        if not mesh_asset.is_file():
            raise FileNotFoundError(mesh_asset)
        if not pose_asset.is_file():
            raise FileNotFoundError(pose_asset)
        self.frame_index = frame_index
        self.renderers: dict[str, InstancedMeshRenderer] = {}

        with np.load(mesh_asset, allow_pickle=False) as meshes, np.load(
            pose_asset, allow_pickle=False
        ) as poses:
            available_meshes = set(meshes["link_names"].tolist())
            pose_names = poses["link_names"].tolist()
            transforms = np.asarray(poses["X_WL"], dtype=np.float32)
            if transforms.ndim != 4 or transforms.shape[2:] != (4, 4):
                raise ValueError("PSM X_WL 必须是 (frames, links, 4, 4)")
            self.frame_count = int(transforms.shape[0])
            self.positions: dict[str, torch.Tensor] = {}
            self.rotations: dict[str, torch.Tensor] = {}
            self.scales: dict[str, torch.Tensor] = {}
            self.colors: dict[str, torch.Tensor] = {}

            for name in DISPLAY_LINKS:
                if name not in available_meshes or name not in pose_names:
                    raise KeyError(f"官方 PSM GUI 资产缺少 link：{name}")
                renderer = InstancedMeshRenderer(
                    np.asarray(meshes[f"{name}__vertices"], dtype=np.float32),
                    np.asarray(meshes[f"{name}__faces"], dtype=np.int32),
                    np.asarray(meshes[f"{name}__normals"], dtype=np.float32),
                    default_color=np.asarray(LINK_COLORS[name], dtype=np.float32),
                )
                link_transforms = transforms[:, pose_names.index(name)]
                quaternions_xyzw = Rotation.from_matrix(
                    link_transforms[:, :3, :3]
                ).as_quat().astype(np.float32)
                quaternions_wxyz = quaternions_xyzw[:, [3, 0, 1, 2]]
                device = torch.device("cuda")
                self.renderers[name] = renderer
                self.positions[name] = torch.as_tensor(
                    link_transforms[:, :3, 3], device=device
                ).contiguous()
                self.rotations[name] = torch.as_tensor(
                    quaternions_wxyz, device=device
                ).contiguous()
                self.scales[name] = torch.ones(
                    (1, 3), device=device, dtype=torch.float32
                )
                self.colors[name] = torch.as_tensor(
                    LINK_COLORS[name], device=device, dtype=torch.float32
                ).reshape(1, 3)

    def draw(self) -> None:
        index = max(0, min(int(self.frame_index()), self.frame_count - 1))
        gl.glEnable(gl.GL_DEPTH_TEST)
        with torch.no_grad():
            for name, renderer in self.renderers.items():
                renderer.update(
                    positions=self.positions[name][index : index + 1],
                    rotations=self.rotations[name][index : index + 1],
                    scaling=self.scales[name],
                    colors=self.colors[name],
                )
                renderer.draw()
