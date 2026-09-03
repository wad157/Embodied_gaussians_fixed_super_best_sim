from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class TissueResistedJawActuator:
    """Effort-limited q7 follower driven by bilateral tissue resistance.

    ``commanded_angle_rad`` is the unmodified recorded q7 command.  Opening is
    released immediately, while closure advances at a finite motor speed and
    stops only when both jaws report contacts that oppose their hinge motion.
    """

    actual_angle_rad: float
    maximum_closing_speed_rad_s: float = 2.4
    maximum_compression_after_contact_rad: float = 0.12
    minimum_contacts_per_jaw: int = 8
    angle_epsilon_rad: float = 1.0e-4
    commanded_angle_rad: float | None = None
    closure_blocked_by_tissue: bool = False
    contact_onset_angle_rad: float | None = None
    last_contact_counts: tuple[int, int] = (0, 0)
    last_resistance_m2: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        self.actual_angle_rad = float(self.actual_angle_rad)
        if not np.isfinite(self.actual_angle_rad):
            raise ValueError("Initial jaw angle must be finite")
        if self.maximum_closing_speed_rad_s <= 0.0:
            raise ValueError("Jaw closing speed must be positive")
        if self.maximum_compression_after_contact_rad <= 0.0:
            raise ValueError("Jaw contact compression travel must be positive")
        if self.minimum_contacts_per_jaw < 1:
            raise ValueError("Jaw contact threshold must be positive")
        if self.angle_epsilon_rad <= 0.0:
            raise ValueError("Jaw angle epsilon must be positive")
        if self.commanded_angle_rad is None:
            self.commanded_angle_rad = self.actual_angle_rad

    @property
    def closing_requested(self) -> bool:
        return bool(
            self.commanded_angle_rad
            < self.actual_angle_rad - self.angle_epsilon_rad
        )

    def reset(self, angle_rad: float) -> None:
        angle = float(angle_rad)
        if not np.isfinite(angle):
            raise ValueError("Jaw reset angle must be finite")
        self.actual_angle_rad = angle
        self.commanded_angle_rad = angle
        self.closure_blocked_by_tissue = False
        self.contact_onset_angle_rad = None
        self.last_contact_counts = (0, 0)
        self.last_resistance_m2 = (0.0, 0.0)

    def set_command(self, angle_rad: float) -> None:
        """Accept raw q7; an opening command releases without artificial lag."""
        command = float(angle_rad)
        if not np.isfinite(command):
            raise ValueError("Jaw command angle must be finite")
        self.commanded_angle_rad = command
        if command > self.actual_angle_rad + self.angle_epsilon_rad:
            self.actual_angle_rad = command
            self.closure_blocked_by_tissue = False
            self.contact_onset_angle_rad = None
            self.last_contact_counts = (0, 0)
            self.last_resistance_m2 = (0.0, 0.0)

    def step(
        self,
        dt: float,
        contact_counts: tuple[int, int] = (0, 0),
        resistance_m2: tuple[float, float] = (0.0, 0.0),
    ) -> float:
        """Advance closure until actual bilateral resistance blocks the motor."""
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("Jaw actuator dt must be finite and positive")
        if len(contact_counts) != 2 or len(resistance_m2) != 2:
            raise ValueError("Jaw actuator requires exactly two jaw metrics")
        self.last_contact_counts = tuple(int(v) for v in contact_counts)
        self.last_resistance_m2 = tuple(float(v) for v in resistance_m2)
        if not self.closing_requested:
            self.actual_angle_rad = float(self.commanded_angle_rad)
            self.closure_blocked_by_tissue = False
            self.contact_onset_angle_rad = None
            return self.actual_angle_rad

        bilateral_resistance = all(
            count >= self.minimum_contacts_per_jaw and resistance > 0.0
            for count, resistance in zip(
                self.last_contact_counts,
                self.last_resistance_m2,
                strict=True,
            )
        )
        maximum_step = self.maximum_closing_speed_rad_s * dt
        next_angle = max(
            float(self.commanded_angle_rad),
            self.actual_angle_rad - maximum_step,
        )
        if bilateral_resistance:
            if self.contact_onset_angle_rad is None:
                self.contact_onset_angle_rad = self.actual_angle_rad
            resisted_limit = (
                self.contact_onset_angle_rad
                - self.maximum_compression_after_contact_rad
            )
            next_angle = max(next_angle, resisted_limit)
            self.closure_blocked_by_tissue = bool(
                next_angle <= resisted_limit + self.angle_epsilon_rad
                and self.commanded_angle_rad
                < resisted_limit - self.angle_epsilon_rad
            )
        else:
            self.contact_onset_angle_rad = None
            self.closure_blocked_by_tissue = False
        self.actual_angle_rad = next_angle
        return self.actual_angle_rad


def _rotx(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray(
        [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )


def _rotz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray(
        [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )


def _transx(distance: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = distance
    return transform


def _transz(distance: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[2, 3] = distance
    return transform


def _modified_dh(alpha: float, a: float, theta: float, d: float) -> np.ndarray:
    return _rotx(alpha) @ _transx(a) @ _rotz(theta) @ _transz(d)


@dataclass(frozen=True)
class PSMLNDKinematics:
    dh_params: tuple[dict, ...]
    T_rectified_camera_psm_base: np.ndarray
    X_table_camera: np.ndarray
    link_names: tuple[str, ...]
    lnd_link_ids: np.ndarray
    T_lndlink_urdf_link: np.ndarray
    manual_offset_conventions: dict

    @classmethod
    def from_files(
        cls,
        lnd_model_path: Path,
        pose_report_path: Path,
        table_frame_path: Path,
        link_names: list[str],
    ) -> "PSMLNDKinematics":
        lnd_model = json.loads(lnd_model_path.read_text(encoding="utf-8"))
        pose_report = json.loads(pose_report_path.read_text(encoding="utf-8"))
        table_frame = json.loads(table_frame_path.read_text(encoding="utf-8"))
        registration = pose_report.get("canonical_registration", pose_report)
        lnd_link_ids = np.asarray(
            [registration["link_mapping"][name] for name in link_names],
            dtype=np.int64,
        )
        link_offsets = np.stack(
            [
                np.asarray(
                    registration["T_lndlink_urdf_link"][name],
                    dtype=np.float64,
                )
                for name in link_names
            ]
        )
        if "lnd" in lnd_model:
            dh_params = lnd_model["lnd"]["DH_params"]
            T_rectified_camera_psm_base = lnd_model[
                "calibration_and_static_transforms"
            ]["T_rectified_left_camera_psm_base"]
        else:
            dh_params = lnd_model["dh_params"]
            T_rectified_camera_psm_base = lnd_model[
                "T_rectified_camera_psm_base"
            ]
        return cls(
            dh_params=tuple(dh_params),
            T_rectified_camera_psm_base=np.asarray(
                T_rectified_camera_psm_base, dtype=np.float64
            ),
            X_table_camera=np.asarray(table_frame["X_table_camera"], dtype=np.float64),
            link_names=tuple(link_names),
            lnd_link_ids=lnd_link_ids,
            T_lndlink_urdf_link=link_offsets,
            manual_offset_conventions=pose_report.get(
                "manual_offset_conventions", {}
            ),
        )

    def lnd_forward_kinematics(self, q7: np.ndarray) -> dict[int, np.ndarray]:
        q7 = np.asarray(q7, dtype=np.float64)
        if q7.shape != (7,):
            raise ValueError(f"Expected q7 shape (7,), got {q7.shape}")
        transforms: dict[int, np.ndarray] = {0: np.eye(4, dtype=np.float64)}
        transform = np.eye(4, dtype=np.float64)
        for index, dh in enumerate(self.dh_params, start=1):
            theta0 = float(dh.get("theta", 0.0))
            d0 = float(dh.get("D", 0.0))
            offset = float(dh.get("offset", 0.0))
            if dh["type"] == "revolute":
                theta = theta0 + float(q7[index - 1]) + offset
                d = d0
            elif dh["type"] == "prismatic":
                theta = theta0
                d = d0 + float(q7[index - 1]) + offset
            else:
                raise ValueError(f"Unknown LND joint type: {dh['type']}")
            transform = transform @ _modified_dh(
                float(dh.get("alpha", 0.0)),
                float(dh.get("A", 0.0)),
                theta,
                d,
            )
            transforms[index] = transform.copy()
        jaw = float(np.clip(q7[6], -1.2, 1.6))
        transforms[7] = transforms[6] @ _rotz(0.5 * jaw)
        transforms[8] = transforms[6] @ _rotz(-0.5 * jaw)
        return transforms

    def visual_matrices_table(self, q7: np.ndarray) -> np.ndarray:
        lnd_transforms = self.lnd_forward_kinematics(q7)
        matrices = []
        for link_index, lnd_link_id in enumerate(self.lnd_link_ids):
            T_rectified_camera_visual = (
                self.T_rectified_camera_psm_base
                @ lnd_transforms[int(lnd_link_id)]
                @ self.T_lndlink_urdf_link[link_index]
            )
            matrices.append(self.X_table_camera @ T_rectified_camera_visual)
        return np.stack(matrices)

    def visual_poses_table(self, q7: np.ndarray) -> np.ndarray:
        matrices = self.visual_matrices_table(q7)
        poses = np.empty((len(matrices), 7), dtype=np.float32)
        poses[:, :3] = matrices[:, :3, 3]
        poses[:, 3:] = Rotation.from_matrix(matrices[:, :3, :3]).as_quat().astype(
            np.float32
        )
        return poses

    def gaussian_world_means(
        self,
        q7: np.ndarray,
        local_means: np.ndarray,
        gaussian_link_ids: np.ndarray,
    ) -> np.ndarray:
        matrices = self.visual_matrices_table(q7)
        result = np.empty_like(local_means, dtype=np.float64)
        for link_index, matrix in enumerate(matrices):
            mask = gaussian_link_ids == link_index
            result[mask] = (
                local_means[mask] @ matrix[:3, :3].T + matrix[:3, 3]
            )
        return result
