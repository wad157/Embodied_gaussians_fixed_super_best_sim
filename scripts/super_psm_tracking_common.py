from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


REPO = Path(__file__).resolve().parents[1]
DEFAULT_TRACK_ROOT = REPO / "data/super/psm_tracking"
DEFAULT_JOINTS = REPO / "data/super/grasp5_native/joints.json"
DEFAULT_VIDEO_METADATA = (
    REPO / "data/super/grasp5_offline_demo/videos/stereo_left.json"
)
DEFAULT_LND_MODEL = (
    REPO / "data/super/grasp5_offline_demo/instruments/psm1_lnd_model.json"
)
DEFAULT_POSE_REPORT = REPO / "data/super/psm_robot/psm_lnd_pose_driver_report.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def matrix_to_pose(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(matrix[:3, 3], dtype=np.float64),
            Rotation.from_matrix(matrix[:3, :3]).as_quat(),
        ]
    )


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def matrix_to_ctr(matrix: np.ndarray) -> np.ndarray:
    """Return the paper state [axis-angle, translation] for T_camera_frame4."""
    return np.concatenate(
        [Rotation.from_matrix(matrix[:3, :3]).as_rotvec(), matrix[:3, 3]]
    ).astype(np.float32)


def ctr_to_matrix(ctr: np.ndarray) -> np.ndarray:
    ctr = np.asarray(ctr, dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_rotvec(ctr[:3]).as_matrix()
    matrix[:3, 3] = ctr[3:]
    return matrix


def _paper_component_transforms(joints: np.ndarray) -> tuple[np.ndarray, ...]:
    """Paper LND transforms for shaft, logo, frame-6 and the two jaws.

    These reproduce ``diffcali/eval_dvrk/LND_fk.py`` in NumPy.  In particular,
    the paper applies static mesh-frame rotations which are not the same as the
    current URDF/LND frame convention.  Treating paper cTr as current frame-4
    was therefore incorrect even though both are called "frame 4".
    """
    theta0, theta1, theta2, theta3 = np.asarray(joints, dtype=np.float64)

    def matrix(rows: list[list[float]]) -> np.ndarray:
        return np.asarray(rows, dtype=np.float64)

    T45 = matrix(
        [
            [np.sin(theta0), np.cos(theta0), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [np.cos(theta0), -np.sin(theta0), 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    T56 = matrix(
        [
            [np.sin(theta1), np.cos(theta1), 0.0, 0.0091],
            [0.0, 0.0, 1.0, 0.0],
            [np.cos(theta1), -np.sin(theta1), 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    T4mesh = matrix(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    T5mesh = matrix(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    T7mesh = matrix(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    T6 = T45 @ T56
    jaw_right = np.eye(4, dtype=np.float64)
    jaw_right[:3, :3] = Rotation.from_euler("z", -theta2).as_matrix()
    jaw_left = np.eye(4, dtype=np.float64)
    jaw_left[:3, :3] = Rotation.from_euler("z", theta3).as_matrix()
    return (
        T4mesh,
        T45 @ T5mesh,
        T6,
        T6 @ jaw_right @ T7mesh,
        T6 @ jaw_left @ T7mesh,
    )


def paper_jaw_pivot_camera(ctr: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """Return the shared paper jaw pivot in OpenCV camera coordinates."""
    T_camera_paper = ctr_to_matrix(ctr)
    T_paper_frame6 = _paper_component_transforms(joints)[2]
    return (T_camera_paper @ T_paper_frame6)[:3, 3]


@dataclass(frozen=True)
class TrackingInputs:
    joint_timestamps: np.ndarray
    q7_states: np.ndarray
    video_timestamps: np.ndarray
    K_half: np.ndarray
    T_rectified_camera_psm_base: np.ndarray
    link_names: tuple[str, ...]
    lnd_link_ids: np.ndarray
    T_lndlink_urdf_link: np.ndarray
    dh_params: tuple[dict, ...]

    @classmethod
    def load(
        cls,
        joints_path: Path = DEFAULT_JOINTS,
        video_metadata_path: Path = DEFAULT_VIDEO_METADATA,
        lnd_model_path: Path = DEFAULT_LND_MODEL,
        pose_report_path: Path = DEFAULT_POSE_REPORT,
    ) -> "TrackingInputs":
        joints = read_json(joints_path)
        video_metadata = read_json(video_metadata_path)
        lnd_model = read_json(lnd_model_path)
        pose_report = read_json(pose_report_path)
        joint_timestamps = np.asarray(joints["states_timestamps"], dtype=np.float64)
        q7_states = np.asarray([state["q"] for state in joints["states"]])
        video_timestamps = np.asarray(video_metadata["timestamps"], dtype=np.float64)
        K = np.asarray(video_metadata["K"], dtype=np.float64)
        link_names = tuple(pose_report["link_mapping"].keys())
        lnd_link_ids = np.asarray(
            [pose_report["link_mapping"][name] for name in link_names],
            dtype=np.int64,
        )
        link_offsets = np.stack(
            [
                np.asarray(
                    pose_report["T_lndlink_urdf_link"][name], dtype=np.float64
                )
                for name in link_names
            ]
        )
        K_half = K.copy()
        K_half[:2] *= 0.5
        return cls(
            joint_timestamps=joint_timestamps,
            q7_states=q7_states,
            video_timestamps=video_timestamps,
            K_half=K_half,
            T_rectified_camera_psm_base=np.asarray(
                lnd_model["T_rectified_camera_psm_base"], dtype=np.float64
            ),
            link_names=link_names,
            lnd_link_ids=lnd_link_ids,
            T_lndlink_urdf_link=link_offsets,
            dh_params=tuple(lnd_model["dh_params"]),
        )

    def video_joint_indices(self) -> np.ndarray:
        right = np.searchsorted(
            self.joint_timestamps, self.video_timestamps, side="left"
        )
        right = np.clip(right, 0, len(self.joint_timestamps) - 1)
        left = np.clip(right - 1, 0, len(self.joint_timestamps) - 1)
        use_right = (
            np.abs(self.joint_timestamps[right] - self.video_timestamps)
            < np.abs(self.joint_timestamps[left] - self.video_timestamps)
        )
        return np.where(use_right, right, left)

    def strict_paper_states(self) -> tuple[np.ndarray, np.ndarray]:
        indices = self.video_joint_indices()
        ctr = np.empty((len(indices), 6), dtype=np.float32)
        joints = np.empty((len(indices), 4), dtype=np.float32)
        for frame_index, state_index in enumerate(indices):
            q7 = self.q7_states[state_index]
            transforms = self.lnd_forward_kinematics(q7)
            ctr[frame_index] = matrix_to_ctr(
                self.T_rectified_camera_psm_base @ transforms[4]
            )
            # The paper starts at LND frame 4 and models wrist pitch, wrist yaw,
            # and the two independently rotating jaw halves.
            joints[frame_index] = [q7[4], q7[5], 0.5 * q7[6], 0.5 * q7[6]]
        return ctr, joints

    def lnd_forward_kinematics(self, q7: np.ndarray) -> dict[int, np.ndarray]:
        from examples.embodied_environments.super_embodied.psm_lnd_kinematics import (
            PSMLNDKinematics,
        )

        helper = PSMLNDKinematics(
            dh_params=self.dh_params,
            T_rectified_camera_psm_base=self.T_rectified_camera_psm_base,
            X_table_camera=np.eye(4),
            link_names=self.link_names,
            lnd_link_ids=self.lnd_link_ids,
            T_lndlink_urdf_link=self.T_lndlink_urdf_link,
        )
        return helper.lnd_forward_kinematics(q7)

    def paper_states_to_visual_poses(
        self,
        ctr: np.ndarray,
        paper_joints: np.ndarray,
        *,
        registration_ctr: np.ndarray | None = None,
        registration_joints: np.ndarray | None = None,
        camera_alignment: np.ndarray | None = None,
    ) -> np.ndarray:
        ctr = np.asarray(ctr, dtype=np.float64)
        paper_joints = np.asarray(paper_joints, dtype=np.float64)
        if ctr.shape != (len(paper_joints), 6) or paper_joints.shape[1:] != (4,):
            raise ValueError(
                f"Expected ctr (N,6) and paper_joints (N,4), got "
                f"{ctr.shape} and {paper_joints.shape}"
            )
        if (registration_ctr is None) != (registration_joints is None):
            raise ValueError(
                "registration_ctr and registration_joints must be provided together"
            )
        if registration_ctr is None:
            # Legacy paper-exact callers use their own first state as the link
            # registration reference.  Registered/corrected callers must pass
            # the registered-LND reference explicitly so a corrected first
            # state cannot cancel its own residual.
            registration_ctr = ctr[0]
            registration_joints = paper_joints[0]
        registration_ctr = np.asarray(registration_ctr, dtype=np.float64)
        registration_joints = np.asarray(
            registration_joints, dtype=np.float64
        )
        if registration_ctr.shape != (6,) or registration_joints.shape != (4,):
            raise ValueError(
                "Expected registration_ctr (6,) and registration_joints (4,), "
                f"got {registration_ctr.shape} and {registration_joints.shape}"
            )
        if camera_alignment is None:
            camera_alignment = np.eye(4, dtype=np.float64)
        camera_alignment = np.asarray(camera_alignment, dtype=np.float64)
        if camera_alignment.shape != (4, 4):
            raise ValueError(
                f"Expected camera_alignment (4,4), got {camera_alignment.shape}"
            )

        # Register each paper CAD/kinematic component to the corresponding
        # current PSM visual body at the explicit registered-LND reference.
        calibration_q7 = self.q7_states[self.video_joint_indices()[0]]
        calibration_lnd = self.lnd_forward_kinematics(calibration_q7)
        calibration_current_poses: list[np.ndarray] = []
        for link_index, lnd_link_id in enumerate(self.lnd_link_ids):
            calibration_current_poses.append(
                self.T_rectified_camera_psm_base
                @ calibration_lnd[int(lnd_link_id)]
                @ self.T_lndlink_urdf_link[link_index]
            )
        calibration_paper_camera = ctr_to_matrix(registration_ctr)
        calibration_components = _paper_component_transforms(
            registration_joints
        )
        # current jaw-1 rotates in the same direction as the paper left jaw;
        # jaw-2 matches the paper right jaw, hence the deliberate 4/3 swap.
        component_by_link = (0, 0, 0, 1, 2, 4, 3)
        component_to_current = []
        for link_index, component_index in enumerate(component_by_link):
            T_camera_component = (
                calibration_paper_camera
                @ calibration_components[component_index]
            )
            component_to_current.append(
                np.linalg.inv(T_camera_component)
                @ calibration_current_poses[link_index]
            )

        output = np.empty((len(ctr), len(self.link_names), 7), dtype=np.float32)
        for state_index, (state_ctr, visible_joints) in enumerate(
            zip(ctr, paper_joints, strict=True)
        ):
            T_camera_paper = ctr_to_matrix(state_ctr)
            components = _paper_component_transforms(visible_joints)
            for link_index, component_index in enumerate(component_by_link):
                output[state_index, link_index] = matrix_to_pose(
                    camera_alignment
                    @ T_camera_paper
                    @ components[component_index]
                    @ component_to_current[link_index]
                )
        return output

    def gui_wrist_angles_from_paper_residual(
        self,
        prior_paper_joints: np.ndarray,
        corrected_paper_joints: np.ndarray,
    ) -> np.ndarray:
        """Apply only image wrist residuals to the real dVRK wrist signal.

        Paper/CtRNet wrist coordinates have a different static zero from the
        current URDF (about 30 degrees for wrist-pitch in this sequence).  The
        first-frame component registration already absorbs that zero offset.
        Feeding the absolute paper angle into URDF FK would apply it a second
        time and visibly twist the jaw plane away from the shaft.  Moreover,
        the paper q4 residual is expressed in the paper/CAD frame and directly
        adding it to URDF wrist-pitch breaks the shaft-in-jaw-plane constraint.
        Keep URDF q4 exactly on the encoder and transfer only the paper q5
        residual to URDF wrist-yaw.
        """
        prior = np.asarray(prior_paper_joints, dtype=np.float64)
        corrected = np.asarray(corrected_paper_joints, dtype=np.float64)
        expected_shape = (len(prior), 4)
        if prior.shape != expected_shape or corrected.shape != expected_shape:
            raise ValueError(
                "Expected prior/corrected paper joints (N,4), got "
                f"{prior.shape} and {corrected.shape}"
            )
        video_indices = self.video_joint_indices()[: len(prior)]
        encoder_wrist = self.q7_states[video_indices, 4:6].astype(np.float64)
        result = encoder_wrist.copy()
        result[:, 1] += corrected[:, 1] - prior[:, 1]
        result[:, 0] = np.clip(result[:, 0], -1.5707, 1.5707)
        result[:, 1] = np.clip(result[:, 1], -1.3963, 1.3963)
        return result.astype(np.float32)

    def gui_jaw_angles_from_paper_residual(
        self,
        prior_paper_joints: np.ndarray,
        corrected_paper_joints: np.ndarray,
    ) -> np.ndarray:
        """Apply the image-estimated jaw residual to the real dVRK jaw signal.

        The paper state stores one angle per jaw half and clamps both halves at
        zero when the encoder moves slightly past mechanical closure.  The GUI
        geometry, however, was calibrated against the original full dVRK jaw
        coordinate.  Keeping that coordinate as the baseline preserves its
        physical closed offset; doubling the common paper half-angle residual
        transfers only the image-derived opening correction.
        """
        prior = np.asarray(prior_paper_joints, dtype=np.float64)
        corrected = np.asarray(corrected_paper_joints, dtype=np.float64)
        expected_shape = (len(prior), 4)
        if prior.shape != expected_shape or corrected.shape != expected_shape:
            raise ValueError(
                "Expected prior/corrected paper joints (N,4), got "
                f"{prior.shape} and {corrected.shape}"
            )
        video_indices = self.video_joint_indices()[: len(prior)]
        encoder_jaw = self.q7_states[video_indices, 6].astype(np.float64)
        half_angle_residual = np.mean(
            corrected[:, 2:] - prior[:, 2:], axis=1
        )
        return np.clip(
            encoder_jaw + 2.0 * half_angle_residual,
            -1.2,
            1.6,
        ).astype(np.float32)

    def enforce_shared_gui_jaw_kinematics(
        self,
        visual_poses: np.ndarray,
        gui_jaw_angles: np.ndarray,
    ) -> np.ndarray:
        """Rebuild both GUI jaws from their common wrist parent.

        A per-link paper-to-GUI registration is valid only at its calibration
        pose.  Reusing separate static registrations for the two jaw links
        makes their apparent pivots diverge as the jaws close.  Conjugating
        the jaw rotation by the paper-to-URDF CAD registration also tilts the
        hinge axis and makes the circular jaw hubs look as if they are being
        bent apart.  This method keeps links 0..4 unchanged and replaces links
        5/6 with the actual URDF relative kinematics:

            parent^-1 * child = Rz(+/- jaw/2)

        Both jaw joints have origin=(0,0,0) and axis=(0,0,1) in
        ``psm.urdf``.  Their link origins therefore remain on one common hinge
        and their local z/hinge axes remain parallel at every opening angle.
        """
        poses = np.asarray(visual_poses, dtype=np.float64)
        jaw_angles = np.asarray(gui_jaw_angles, dtype=np.float64)
        expected_shape = (len(jaw_angles), len(self.link_names), 7)
        if poses.shape != expected_shape:
            raise ValueError(
                f"Expected visual poses {expected_shape}, got {poses.shape}"
            )
        parent_name = "PSM1_tool_wrist_sca_shaft_link"
        jaw_names = (
            "PSM1_tool_wrist_sca_ee_link_1",
            "PSM1_tool_wrist_sca_ee_link_2",
        )
        try:
            parent_index = self.link_names.index(parent_name)
            jaw_indices = tuple(self.link_names.index(name) for name in jaw_names)
        except ValueError as error:
            raise KeyError(
                "The current PSM visual-link set is missing the shared jaw chain"
            ) from error

        output = poses.copy()
        for frame_index, jaw_angle in enumerate(jaw_angles):
            parent_pose = pose_to_matrix(output[frame_index, parent_index])
            for jaw_index, sign in zip(
                jaw_indices, (1.0, -1.0), strict=True
            ):
                rotation = np.eye(4, dtype=np.float64)
                rotation[:3, :3] = Rotation.from_euler(
                    "z", sign * 0.5 * jaw_angle
                ).as_matrix()
                output[frame_index, jaw_index] = matrix_to_pose(
                    parent_pose @ rotation
                )
        return output.astype(np.float32)

    def enforce_urdf_distal_kinematics(
        self,
        visual_poses: np.ndarray,
        wrist_joints: np.ndarray,
        gui_jaw_angles: np.ndarray,
    ) -> np.ndarray:
        """Rebuild the complete distal chain from one tracked wrist anchor.

        The paper adapter estimates absolute poses for several CAD components.
        Those poses are useful image observations but are not guaranteed to
        form one exact URDF parent/child chain after independent component
        registration.  Mixing those wrist poses with only a URDF-correct jaw
        can therefore leave the jaw plane inconsistent with the shaft.

        Link 1 (``tool_wrist_link`` / paper frame 4) remains the tracked spatial
        anchor.  Links 2..6 are then rebuilt with the literal joints from
        ``psm.urdf``:

        - roll_shaft: fixed identity
        - wrist_pitch: origin rpy=(-1.5708,-1.5708,0), xyz=(0,0,0), axis=z
        - wrist_yaw: origin rpy=(-1.5708,-1.5708,0), xyz=(0.0091,0,0), axis=z
        - jaws: shared origin=(0,0,0), axis=z, angles +/-jaw/2

        This preserves the corrected camera-space anchor while making every
        downstream motion identical to standalone URDF forward kinematics.
        """
        poses = np.asarray(visual_poses, dtype=np.float64)
        wrist = np.asarray(wrist_joints, dtype=np.float64)
        jaw_angles = np.asarray(gui_jaw_angles, dtype=np.float64)
        expected_shape = (len(jaw_angles), len(self.link_names), 7)
        if poses.shape != expected_shape:
            raise ValueError(
                f"Expected visual poses {expected_shape}, got {poses.shape}"
            )
        if wrist.shape != (len(poses), 2):
            raise ValueError(
                f"Expected wrist joints {(len(poses), 2)}, got {wrist.shape}"
            )

        required_names = (
            "PSM1_tool_wrist_link",
            "PSM1_tool_wrist_shaft_link",
            "PSM1_tool_wrist_sca_link",
            "PSM1_tool_wrist_sca_shaft_link",
            "PSM1_tool_wrist_sca_ee_link_1",
            "PSM1_tool_wrist_sca_ee_link_2",
        )
        missing = [name for name in required_names if name not in self.link_names]
        if missing:
            raise KeyError(
                f"The current PSM visual-link set is missing the URDF distal chain: {missing}"
            )
        indices = {name: self.link_names.index(name) for name in required_names}

        def origin(
            rpy: tuple[float, float, float],
            xyz: tuple[float, float, float],
        ) -> np.ndarray:
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
            transform[:3, 3] = xyz
            return transform

        def rotz(angle: float) -> np.ndarray:
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = Rotation.from_euler("z", angle).as_matrix()
            return transform

        wrist_pitch_origin = origin((-1.5708, -1.5708, 0.0), (0.0, 0.0, 0.0))
        wrist_yaw_origin = origin(
            (-1.5708, -1.5708, 0.0), (0.0091, 0.0, 0.0)
        )
        output = poses.copy()
        for frame_index in range(len(output)):
            wrist_link = pose_to_matrix(
                output[frame_index, indices["PSM1_tool_wrist_link"]]
            )
            wrist_shaft = wrist_link.copy()
            wrist_sca = (
                wrist_shaft
                @ wrist_pitch_origin
                @ rotz(float(wrist[frame_index, 0]))
            )
            wrist_sca_shaft = (
                wrist_sca
                @ wrist_yaw_origin
                @ rotz(float(wrist[frame_index, 1]))
            )
            jaw_left = wrist_sca_shaft @ rotz(0.5 * float(jaw_angles[frame_index]))
            jaw_right = wrist_sca_shaft @ rotz(-0.5 * float(jaw_angles[frame_index]))
            for name, matrix in (
                ("PSM1_tool_wrist_shaft_link", wrist_shaft),
                ("PSM1_tool_wrist_sca_link", wrist_sca),
                ("PSM1_tool_wrist_sca_shaft_link", wrist_sca_shaft),
                ("PSM1_tool_wrist_sca_ee_link_1", jaw_left),
                ("PSM1_tool_wrist_sca_ee_link_2", jaw_right),
            ):
                output[frame_index, indices[name]] = matrix_to_pose(matrix)
        return output.astype(np.float32)


def resample_pose_sequence(
    source_timestamps: np.ndarray,
    source_poses: np.ndarray,
    target_timestamps: np.ndarray,
) -> np.ndarray:
    source_timestamps = np.asarray(source_timestamps, dtype=np.float64)
    source_poses = np.asarray(source_poses, dtype=np.float64)
    target_timestamps = np.asarray(target_timestamps, dtype=np.float64)
    if len(source_timestamps) < 2:
        raise ValueError("At least two source poses are required for resampling")
    query = np.clip(target_timestamps, source_timestamps[0], source_timestamps[-1])
    output = np.empty((len(query), source_poses.shape[1], 7), dtype=np.float32)
    for link_index in range(source_poses.shape[1]):
        for axis in range(3):
            output[:, link_index, axis] = np.interp(
                query,
                source_timestamps,
                source_poses[:, link_index, axis],
            )
        rotations = Rotation.from_quat(source_poses[:, link_index, 3:])
        output[:, link_index, 3:] = Slerp(source_timestamps, rotations)(
            query
        ).as_quat()
    return output


def save_runtime_driver(
    path: Path,
    inputs: TrackingInputs,
    video_poses: np.ndarray,
) -> None:
    runtime_poses = resample_pose_sequence(
        inputs.video_timestamps[: len(video_poses)],
        video_poses,
        inputs.joint_timestamps,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        timestamps=inputs.joint_timestamps,
        link_names=np.asarray(inputs.link_names),
        poses_rect_camera_xyz_xyzw=runtime_poses,
    )
