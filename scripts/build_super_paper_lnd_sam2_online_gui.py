#!/usr/bin/env python3
"""Build the paper-LND GUI asset and lift stereo online corrections.

The output is rebuilt from the frozen raw q7/LND/calibration/hand-eye
backbone, the upstream paper repository's four LND meshes/FK convention, and
the newly generated stereo SAM2 corrections.  No historical P420006 pose,
mask, annotation, CAD adapter, or GUI driver is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

import build_super_raw_psm_gui_driver as raw_builder
from build_super_p420006_sam2_online_gui_driver import (
    interpolate_clamped_linear,
)
from build_super_raw_p420006_gui_driver import add_mesh_visual
from paper_lnd_mesh_io import load_binary_ply_triangles
from super_psm_tracking_common import _paper_component_transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = (
    REPO_ROOT / "data/super/psm_raw_kinematics_v1（纯机器人学版本）"
)
VISUAL_ROOT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_paper_lnd_stereo_v1"
)
SAM2_ROOT = VISUAL_ROOT / "surgicalsam2_stereo_sequence_v1"
OPTIMIZATION_ROOT = SAM2_ROOT / "online_stereo_cma_v1"
PAPER_REPO = Path("/Media_HDD/jwshan/wad/online_dvrk_tracking")
UPSTREAM_COMMIT = "cb2a264167aaf05b5a9c20da885d48568f78311f"

PAPER_VISIBLE_LINKS = {
    "PSM1_tool_main_link": "shaft_multi_cylinder.ply",
    "PSM1_tool_wrist_sca_shaft_link": "logo_low_res_1.ply",
    "PSM1_tool_wrist_sca_ee_link_1": "jawright_lowres.ply",
    "PSM1_tool_wrist_sca_ee_link_2": "jawleft_lowres.ply",
}
PAPER_LINK_NAMES = (
    "PSM1_tool_main_link",
    "PSM1_tool_wrist_link",
    "PSM1_tool_wrist_shaft_link",
    "PSM1_tool_wrist_sca_link",
    "PSM1_tool_wrist_sca_shaft_link",
    "PSM1_tool_wrist_sca_ee_link_1",
    "PSM1_tool_wrist_sca_ee_link_2",
)
PAPER_LND_LINK_IDS = np.asarray([4, 4, 5, 6, 5, 7, 8], dtype=np.int64)
PAPER_T5_MESH = np.asarray(
    [
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build exact paper-LND CAD/poses for the current GUI from the "
            "complete stereo SurgicalSAM2 correction."
        )
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--visual-root", type=Path, default=VISUAL_ROOT)
    parser.add_argument(
        "--optimization-root",
        type=Path,
        default=OPTIMIZATION_ROOT,
    )
    parser.add_argument(
        "--dvrk-root",
        type=Path,
        default=REPO_ROOT / "data/dvrk_model",
    )
    parser.add_argument("--paper-repo", type=Path, default=PAPER_REPO)
    parser.add_argument(
        "--table-frame",
        type=Path,
        default=REPO_ROOT / "data/super/table_frame.json",
    )
    parser.add_argument(
        "--camera-manifest",
        type=Path,
        default=(
            REPO_ROOT / "data/super/grasp5_offline_demo/cameras.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OPTIMIZATION_ROOT / "gui_v1",
    )
    parser.add_argument(
        "--pose-version",
        default="raw_paper_lnd_sam2_online",
    )
    parser.add_argument(
        "--require-frozen-jaw-correction",
        action="store_true",
    )
    parser.add_argument(
        "--require-frozen-all-joint-corrections",
        action="store_true",
        help=(
            "Require global and per-pair q4..q7 visual residuals to be "
            "exactly zero and preserve all seven raw q columns bitwise."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_mesh(path: Path) -> trimesh.Trimesh:
    vertices, faces = load_binary_ply_triangles(path)
    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )


def build_fresh_urdf_and_mimic(
    *,
    dvrk_root: Path,
    paper_mesh_dir: Path,
    output_dir: Path,
) -> tuple[Path, Path, list[dict[str, Any]], list[Path]]:
    classic_sources = [
        dvrk_root / "urdf/Classic/PSM1.urdf.xacro",
        dvrk_root / "urdf/Classic/psm_base.urdf.xacro",
        dvrk_root / "urdf/Classic/psm_tool.urdf.xacro",
        dvrk_root / "urdf/Classic/psm_tool_sca.urdf.xacro",
        dvrk_root / "urdf/common.urdf.xacro",
    ]
    robot = raw_builder.expand_root_psm_xacro(dvrk_root)
    mimic = raw_builder.extract_mimic_map(robot)
    raw_builder.widen_raw_limits(robot)
    mesh_reports = raw_builder.rewrite_root_meshes(
        robot,
        dvrk_root,
        output_dir / "meshes",
    )

    for link_name in PAPER_LINK_NAMES:
        link = robot.find(f"./link[@name='{link_name}']")
        if link is None:
            raise KeyError(f"Classic topology is missing {link_name}")
        for tag in ("visual", "collision"):
            for element in list(link.findall(tag)):
                link.remove(element)

    paper_sources: list[Path] = []
    for link_name, source_name in PAPER_VISIBLE_LINKS.items():
        source = paper_mesh_dir / source_name
        paper_sources.append(source)
        output_name = f"paper_{source.stem}.stl"
        output_path = output_dir / "meshes" / output_name
        mesh = load_mesh(source)
        mesh.export(output_path)
        link = robot.find(f"./link[@name='{link_name}']")
        assert link is not None
        color = (
            "0.77647 0.75686 0.73725 1"
            if "ee_link" in link_name
            else "0.79216 0.81961 0.93333 1"
        )
        add_mesh_visual(link, f"meshes/{output_name}", color)
        mesh_reports.append(
            {
                "source": str(source.resolve()),
                "source_sha256": sha256(source),
                "output": f"meshes/{output_name}",
                "output_sha256": sha256(output_path),
                "vertices": int(len(mesh.vertices)),
                "triangles": int(len(mesh.faces)),
                "bounds_m": np.asarray(mesh.bounds).tolist(),
                "surface_area_m2": float(mesh.area),
                "paper_destination_link": link_name,
            }
        )

    raw_builder.ensure_collision_and_inertial(robot)
    urdf_path = output_dir / "psm_paper_lnd.urdf"
    raw_builder.write_fresh_urdf(
        robot,
        urdf_path,
        [*classic_sources, *paper_sources],
    )
    mimic_path = output_dir / "psm_paper_lnd_mimic_map.json"
    raw_builder.write_json(
        mimic_path,
        {
            "source": (
                "fresh Classic arm topology with exact paper-LND "
                "four-component visual/contact geometry"
            ),
            "input_joint_names": list(raw_builder.INPUT_JOINT_NAMES),
            "input_to_urdf_joint": raw_builder.INPUT_TO_URDF_JOINT,
            "mimic": mimic,
        },
    )
    return urdf_path, mimic_path, mesh_reports, classic_sources


def current_gui_base_transform(
    *,
    robot: Any,
    mimic: dict[str, dict[str, Any]],
    raw_model: dict[str, Any],
    X_table_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    joints, children = raw_builder.parse_urdf_tree(robot)
    canonical_world = raw_builder.urdf_fk(
        joints,
        children,
        raw_builder.expand_q7(np.zeros(7), mimic),
    )
    canonical_base = canonical_world["PSM1_psm_base_link"]
    canonical_urdf = {
        name: np.linalg.inv(canonical_base) @ transform
        for name, transform in canonical_world.items()
    }
    canonical_lnd = raw_builder.lnd_fk(
        raw_model["lnd"],
        np.zeros(7, dtype=np.float64),
    )
    source = np.asarray(
        [
            canonical_urdf[urdf_link][:3, 3]
            for urdf_link, _lnd_link, _weight
            in raw_builder.MODEL_ALIGNMENT_CORRESPONDENCES
        ]
    )
    target = np.asarray(
        [
            canonical_lnd[lnd_link, :3, 3]
            for _urdf_link, lnd_link, _weight
            in raw_builder.MODEL_ALIGNMENT_CORRESPONDENCES
        ]
    )
    weights = np.asarray(
        [
            weight
            for _urdf_link, _lnd_link, weight
            in raw_builder.MODEL_ALIGNMENT_CORRESPONDENCES
        ]
    )
    X_lnd_base_urdf_base = raw_builder.weighted_rigid_fit(
        source,
        target,
        weights,
    )
    T_left_psm_base = np.asarray(
        raw_model["calibration_and_static_transforms"][
            "T_rectified_left_camera_psm_base"
        ],
        dtype=np.float64,
    )
    return (
        X_table_camera @ T_left_psm_base @ X_lnd_base_urdf_base,
        X_lnd_base_urdf_base,
    )


def static_helper_registration(
    lnd_model: dict[str, Any],
) -> tuple[dict[str, int], dict[str, list]]:
    """Provide a q=0 helper mapping for optional manual GUI offsets."""

    q0_components = _paper_component_transforms(np.zeros(4))
    q0_lnd = raw_builder.lnd_fk(lnd_model, np.zeros(7))
    T_base_frame4 = q0_lnd[4]
    q0_T45 = q0_components[1] @ np.linalg.inv(PAPER_T5_MESH)
    desired = (
        T_base_frame4 @ q0_components[0],
        T_base_frame4,
        T_base_frame4 @ q0_T45,
        T_base_frame4 @ q0_components[2],
        T_base_frame4 @ q0_components[1],
        T_base_frame4 @ q0_components[3],
        T_base_frame4 @ q0_components[4],
    )
    offsets: dict[str, list] = {}
    for name, link_id, target in zip(
        PAPER_LINK_NAMES,
        PAPER_LND_LINK_IDS,
        desired,
        strict=True,
    ):
        offsets[name] = (
            np.linalg.inv(q0_lnd[int(link_id)]) @ target
        ).tolist()
    return (
        {
            name: int(link_id)
            for name, link_id in zip(
                PAPER_LINK_NAMES,
                PAPER_LND_LINK_IDS,
                strict=True,
            )
        },
        offsets,
    )


def main() -> None:
    args = parse_args()
    if args.require_frozen_all_joint_corrections:
        args.require_frozen_jaw_correction = True
    raw_model_path = args.raw_root / "model.json"
    raw_kinematics_path = args.raw_root / "kinematics.npz"
    raw_report_path = args.raw_root / "report.json"
    manifest_path = args.visual_root / "pair_manifest.json"
    if not manifest_path.is_file():
        manifest_path = args.visual_root / "prompt_manifest.json"
    keyframes_path = args.visual_root / "keyframes.npz"
    correction_path = (
        args.optimization_root / "online_stereo_corrections.npz"
    )
    optimization_report_path = args.optimization_root / "report.json"
    paper_mesh_dir = args.paper_repo / "urdfs/dVRK/meshes"
    paper_mesh_paths = [
        paper_mesh_dir / name for name in PAPER_VISIBLE_LINKS.values()
    ]
    for path in (
        raw_model_path,
        raw_kinematics_path,
        raw_report_path,
        manifest_path,
        correction_path,
        optimization_report_path,
        args.table_frame,
        args.camera_manifest,
        *paper_mesh_paths,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    commit = subprocess.run(
        ["git", "-C", str(args.paper_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"Unexpected upstream commit {commit}")
    raw_report = json.loads(raw_report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    optimization_report = json.loads(
        optimization_report_path.read_text(encoding="utf-8")
    )
    if raw_report.get("passed") is not True:
        raise RuntimeError("Frozen raw-kinematics report has not passed")
    if (
        optimization_report.get("schema")
        != "super_paper_lnd_stereo_sam2_online_optimization_v1"
        or optimization_report.get("passed") is not True
        or optimization_report.get("sequence", {}).get(
            "complete_strict_pair_sequence"
        )
        is not True
    ):
        raise RuntimeError("Complete paper-LND stereo optimization is not accepted")
    if (
        args.require_frozen_all_joint_corrections
        and optimization_report.get("optimization", {}).get(
            "freeze_all_joint_corrections"
        )
        is not True
    ):
        raise RuntimeError("Optimization did not declare all joint residuals frozen")

    raw_model = json.loads(raw_model_path.read_text(encoding="utf-8"))
    table_frame = json.loads(args.table_frame.read_text(encoding="utf-8"))
    camera_manifest = json.loads(
        args.camera_manifest.read_text(encoding="utf-8")
    )
    X_table_camera = np.asarray(
        table_frame["X_table_camera"],
        dtype=np.float64,
    )
    X_blender_from_opencv = np.diag([1.0, -1.0, -1.0, 1.0])
    manifest_camera = np.asarray(
        camera_manifest["stereo_left"]["X_WC"],
        dtype=np.float64,
    )
    camera_error = float(
        np.max(
            np.abs(
                X_table_camera @ X_blender_from_opencv - manifest_camera
            )
        )
    )
    if camera_error > 1.0e-12:
        raise RuntimeError(
            "Current GUI camera/table frame gate failed: "
            f"{camera_error:.3e}"
        )

    with np.load(raw_kinematics_path, allow_pickle=False) as raw:
        timestamps_s = raw["joint_timestamps_s"].astype(np.float64)
        timestamps_ros_ns = raw["joint_timestamps_ros_ns"].astype(np.int64)
        q7_raw = raw["q7"].astype(np.float64)
        T_left_lnd_raw = raw[
            "T_rectified_left_camera_lnd_link"
        ].astype(np.float64)
    if keyframes_path.is_file():
        with np.load(keyframes_path, allow_pickle=False) as keyframes:
            T_right_left = keyframes[
                "T_rectified_right_camera_rectified_left_camera"
            ].astype(np.float64)
    else:
        T_right_left = np.asarray(
            raw_model["calibration_and_static_transforms"][
                "T_rectified_right_camera_rectified_left_camera"
            ],
            dtype=np.float64,
        )
    with np.load(correction_path, allow_pickle=False) as correction:
        schema = str(correction["schema"].item())
        pair_timestamps_ns = correction[
            "pair_timestamp_ros_ns"
        ].astype(np.int64)
        pair_slots = correction["strict_pair_slot"].astype(np.int64)
        global_correction = correction[
            "global_correction"
        ].astype(np.float64)
        local_pair_correction = correction[
            "local_filtered_corrections"
        ].astype(np.float64)
        global_bounds = correction[
            "global_parameter_bounds"
        ].astype(np.float64)
        local_bounds = correction[
            "local_parameter_bounds"
        ].astype(np.float64)
        translation_hard_bounded = bool(
            correction["translation_hard_bounded"].item()
        ) if "translation_hard_bounded" in correction.files else True
    expected_pairs = int(manifest["source_pair_count"])
    if (
        schema != "super_paper_lnd_stereo_sam2_online_corrections_v1"
        or len(pair_timestamps_ns) != expected_pairs
        or not np.array_equal(
            pair_slots,
            np.arange(expected_pairs, dtype=np.int64),
        )
        or local_pair_correction.shape != (expected_pairs, 10)
        or global_correction.shape != (10,)
    ):
        raise RuntimeError("Paper-LND corrections do not cover all strict pairs")
    bounded_indices = (
        np.arange(10)
        if translation_hard_bounded
        else np.asarray([0, 1, 2, 6, 7, 8, 9])
    )
    if (
        float(
            np.max(
                np.abs(
                    global_correction[bounded_indices]
                    / global_bounds[bounded_indices]
                )
            )
        ) >= 0.995
        or float(
            np.max(
                np.abs(
                    local_pair_correction[:, bounded_indices]
                    / local_bounds[None, bounded_indices]
                )
            )
        ) >= 0.995
    ):
        raise RuntimeError("Refusing a correction that saturates its bounds")
    if args.require_frozen_jaw_correction and not (
        float(global_correction[9]) == 0.0
        and np.count_nonzero(local_pair_correction[:, 9]) == 0
    ):
        raise RuntimeError("Visual correction changed the frozen jaw state")
    if args.require_frozen_all_joint_corrections and not (
        np.count_nonzero(global_correction[6:10]) == 0
        and np.count_nonzero(local_pair_correction[:, 6:10]) == 0
    ):
        raise RuntimeError("Visual correction changed a frozen raw robot joint")

    local_correction = interpolate_clamped_linear(
        pair_timestamps_ns,
        local_pair_correction,
        timestamps_ros_ns,
    )
    q7_corrected = q7_raw.copy()
    q7_corrected[:, 3:7] += (
        global_correction[None, 6:10] + local_correction[:, 6:10]
    )
    if not np.array_equal(q7_corrected[:, :3], q7_raw[:, :3]):
        raise RuntimeError("Visual correction modified raw q0..q2")
    if (
        args.require_frozen_jaw_correction
        and not np.array_equal(q7_corrected[:, 6], q7_raw[:, 6])
    ):
        raise RuntimeError("GUI lift changed raw q7 jaw closure")
    if (
        args.require_frozen_all_joint_corrections
        and not np.array_equal(q7_corrected, q7_raw)
    ):
        raise RuntimeError("GUI lift did not preserve all seven raw joints")

    state_count = len(q7_raw)
    T_left_links = np.empty(
        (state_count, len(PAPER_LINK_NAMES), 4, 4),
        dtype=np.float64,
    )
    global_rotation = Rotation.from_rotvec(
        global_correction[:3]
    ).as_matrix()
    local_rotations = Rotation.from_rotvec(
        local_correction[:, :3]
    ).as_matrix()
    delta_rotations = local_rotations @ global_rotation[None]
    for state_index in range(state_count):
        fk = raw_builder.lnd_fk(
            raw_model["lnd"],
            q7_corrected[state_index],
        )
        T_left_frame4 = T_left_lnd_raw[state_index, 0] @ fk[4]
        paper_joints = np.asarray(
            [
                q7_corrected[state_index, 4],
                q7_corrected[state_index, 5],
                0.5 * q7_corrected[state_index, 6],
                0.5 * q7_corrected[state_index, 6],
            ],
            dtype=np.float64,
        )
        components = _paper_component_transforms(paper_joints)
        T45 = components[1] @ np.linalg.inv(PAPER_T5_MESH)
        links = np.empty((len(PAPER_LINK_NAMES), 4, 4), dtype=np.float64)
        links[0] = T_left_frame4 @ components[0]
        links[1] = T_left_frame4
        links[2] = T_left_frame4 @ T45
        links[3] = T_left_frame4 @ components[2]
        links[4] = T_left_frame4 @ components[1]
        links[5] = T_left_frame4 @ components[3]
        links[6] = T_left_frame4 @ components[4]
        anchor = (T_left_frame4 @ components[2])[:3, 3]
        delta_rotation = delta_rotations[state_index]
        links[:, :3, :3] = delta_rotation[None] @ links[:, :3, :3]
        links[:, :3, 3] = (
            (links[:, :3, 3] - anchor[None]) @ delta_rotation.T
            + anchor[None]
            + global_correction[None, 3:6]
            + local_correction[state_index][None, 3:6]
        )
        T_left_links[state_index] = links

    T_right_links = T_right_left[None, None] @ T_left_links
    T_gui_links = X_table_camera[None, None] @ T_left_links
    poses_gui_world = raw_builder.matrix_series_to_poses(T_gui_links)
    if not (
        np.isfinite(T_left_links).all()
        and np.isfinite(T_right_links).all()
        and np.isfinite(poses_gui_world).all()
    ):
        raise RuntimeError("Paper-LND lifted poses are not finite")

    raw_builder.prepare_output_dir(args.output_dir, args.overwrite)
    urdf_path, mimic_path, mesh_reports, _classic_sources = (
        build_fresh_urdf_and_mimic(
            dvrk_root=args.dvrk_root,
            paper_mesh_dir=paper_mesh_dir,
            output_dir=args.output_dir,
        )
    )
    # Re-expand only for the canonical arm-base placement.  This source model
    # is independent of every historical GUI derivative.
    registration_robot = raw_builder.expand_root_psm_xacro(args.dvrk_root)
    registration_mimic = raw_builder.extract_mimic_map(registration_robot)
    X_gui_urdf_base, X_lnd_base_urdf_base = current_gui_base_transform(
        robot=registration_robot,
        mimic=registration_mimic,
        raw_model=raw_model,
        X_table_camera=X_table_camera,
    )

    link_mapping, helper_offsets = static_helper_registration(
        raw_model["lnd"]
    )
    driver_path = args.output_dir / "psm_paper_lnd_gui_pose_driver.npz"
    helper_offset_array = np.stack(
        [
            np.asarray(helper_offsets[name], dtype=np.float64)
            for name in PAPER_LINK_NAMES
        ]
    )
    np.savez_compressed(
        driver_path,
        schema=np.asarray(
            "super_psm_raw_paper_lnd_sam2_online_gui_pose_driver_v1"
        ),
        coordinate_frame=np.asarray("current_gui_table_world"),
        transform_convention=np.asarray(
            "current GUI X_table_rectLeft @ raw q7/LND frame4 @ exact "
            "paper LND four-component FK + shared stereo online residual"
        ),
        timestamps=timestamps_s,
        timestamps_ros_ns=timestamps_ros_ns,
        q7=q7_corrected,
        q7_raw=q7_raw,
        link_names=np.asarray(PAPER_LINK_NAMES),
        lnd_link_ids=PAPER_LND_LINK_IDS,
        poses_gui_world_xyz_xyzw=poses_gui_world,
        T_lndlink_urdf_link=helper_offset_array,
        X_gui_world_rectified_left_camera=X_table_camera,
        X_gui_world_urdf_base=X_gui_urdf_base,
        T_rectified_right_camera_rectified_left_camera=T_right_left,
        visual_pair_timestamps_ros_ns=pair_timestamps_ns,
        visual_global_correction=global_correction,
        visual_local_pair_correction=local_pair_correction,
        visual_local_q7_time_correction=local_correction,
        visual_jaw_correction_frozen=np.asarray(
            args.require_frozen_jaw_correction
        ),
        visual_all_joint_corrections_frozen=np.asarray(
            args.require_frozen_all_joint_corrections
        ),
        visual_translation_hard_bounded=np.asarray(
            translation_hard_bounded
        ),
    )

    surface_path = args.output_dir / "psm_paper_lnd_surface_gaussians.npz"
    surface_report_path = (
        args.output_dir / "psm_paper_lnd_surface_gaussians_report.json"
    )
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/build_psm_surface_gaussians.py"),
            "--urdf",
            str(urdf_path),
            "--output",
            str(surface_path),
            "--report",
            str(surface_report_path),
            "--density",
            "30000",
            "--min-samples-per-mesh",
            "64",
            "--max-samples-per-mesh",
            "12000",
            "--seed",
            "420",
        ],
        check=True,
    )

    report = {
        "schema": "super_psm_raw_paper_lnd_sam2_online_gui_registration_v1",
        "passed": True,
        "version": args.pose_version,
        "method": (
            "frozen raw q7/LND/calibration/hand-eye backbone + exact upstream "
            "paper LND CAD/FK + "
            f"{optimization_report.get('optimization', {}).get('observation_mode', 'first_pair_union')} "
            "bilateral SurgicalSAM2 + "
            "shared-state stereo CMA-ES/Kalman residual"
            + (
                " with all q4..q7 residuals frozen"
                if args.require_frozen_all_joint_corrections
                else ""
            )
        ),
        "uses_historical_psm_derivatives": False,
        "uses_p420006_assets": False,
        "upstream": {
            "repository": (
                "https://github.com/hanyang-hu/online_dvrk_tracking"
            ),
            "commit": commit,
            "paper_fk": "diffcali/eval_dvrk/LND_fk.py",
            "paper_meshes": {
                path.name: {
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                }
                for path in paper_mesh_paths
            },
        },
        "coordinate_chain": (
            "T_GUIworld_component(t) = X_table_rectifiedLeft(current GUI) @ "
            "T_rectifiedLeft_PSMbase(raw hand-eye) @ LND_FK_to_frame4(raw q7) "
            + (
                "@ paper_component_FK(raw q7 exactly) @ "
                + (
                    "bounded"
                    if translation_hard_bounded
                    else "unbounded-XYZ soft-regularized"
                )
                + " stereo SE(3)-only residual"
                if args.require_frozen_all_joint_corrections
                else "@ paper_component_FK(corrected distal q) @ bounded "
                "stereo residual"
            )
        ),
        "state_rule": (
            f"all {state_count} raw robot states are retained; the old GUI "
            "selects them by its unchanged 1441-frame video timeline"
        ),
        "canonical_registration": {
            "link_mapping": link_mapping,
            "T_lndlink_urdf_link": helper_offsets,
            "X_lnd_base_urdf_base": X_lnd_base_urdf_base.tolist(),
            "note": (
                "This static map is only the optional GUI manual-offset "
                "helper. Saved absolute poses use exact paper component FK."
            ),
        },
        "manual_offset_conventions": {
            "paper_wrist_pitch": {
                "model": "paper_lnd_exact_component_fk",
                "joint_index": 4,
                "anchor_link": "PSM1_tool_wrist_link",
                "moving_links": [
                    "PSM1_tool_wrist_shaft_link",
                    "PSM1_tool_wrist_sca_link",
                    "PSM1_tool_wrist_sca_shaft_link",
                    "PSM1_tool_wrist_sca_ee_link_1",
                    "PSM1_tool_wrist_sca_ee_link_2",
                ],
            }
        },
        "correction_statistics": {
            "global_rotation_vector_deg": np.degrees(
                global_correction[:3]
            ).tolist(),
            "global_translation_mm": (
                global_correction[3:6] * 1.0e3
            ).tolist(),
            "global_q3_q6_deg": np.degrees(
                global_correction[6:10]
            ).tolist(),
            "maximum_global_boundary_fraction": float(
                np.max(
                    np.abs(
                        global_correction[bounded_indices]
                        / global_bounds[bounded_indices]
                    )
                )
            ),
            "maximum_local_boundary_fraction": float(
                np.max(
                    np.abs(
                        local_pair_correction[:, bounded_indices]
                        / local_bounds[None, bounded_indices]
                    )
                )
            ),
            "translation_hard_bounded": translation_hard_bounded,
        },
        "jaw_constraint": {
            "required": bool(args.require_frozen_jaw_correction),
            "global_visual_residual_exact_zero": bool(
                float(global_correction[9]) == 0.0
            ),
            "all_local_visual_residuals_exact_zero": bool(
                np.count_nonzero(local_pair_correction[:, 9]) == 0
            ),
            "gui_q7_jaw_exactly_raw": bool(
                np.array_equal(q7_corrected[:, 6], q7_raw[:, 6])
            ),
        },
        "all_joint_constraint": {
            "required": bool(args.require_frozen_all_joint_corrections),
            "global_q4_q7_visual_residuals_exact_zero": bool(
                np.count_nonzero(global_correction[6:10]) == 0
            ),
            "all_local_q4_q7_visual_residuals_exact_zero": bool(
                np.count_nonzero(local_pair_correction[:, 6:10]) == 0
            ),
            "gui_all_q7_columns_exactly_raw": bool(
                np.array_equal(q7_corrected, q7_raw)
            ),
        },
        "fresh_gui_assets": {
            "urdf": str(urdf_path.resolve()),
            "urdf_sha256": sha256(urdf_path),
            "mimic_map": str(mimic_path.resolve()),
            "mimic_map_sha256": sha256(mimic_path),
            "surface_gaussians": str(surface_path.resolve()),
            "surface_gaussians_sha256": sha256(surface_path),
            "converted_meshes": mesh_reports,
        },
        "inputs": {
            "raw_model": {
                "path": str(raw_model_path.resolve()),
                "sha256": sha256(raw_model_path),
            },
            "raw_kinematics": {
                "path": str(raw_kinematics_path.resolve()),
                "sha256": sha256(raw_kinematics_path),
            },
            "fresh_first_frame_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": sha256(manifest_path),
            },
            "online_corrections": {
                "path": str(correction_path.resolve()),
                "sha256": sha256(correction_path),
            },
            "optimization_report": {
                "path": str(optimization_report_path.resolve()),
                "sha256": sha256(optimization_report_path),
            },
        },
        "driver": {
            "path": str(driver_path.resolve()),
            "sha256": sha256(driver_path),
            "states": state_count,
            "links": len(PAPER_LINK_NAMES),
            "poses_shape": list(poses_gui_world.shape),
            "raw_q0_q2_exact": bool(
                np.array_equal(q7_corrected[:, :3], q7_raw[:, :3])
            ),
            "raw_q7_jaw_exact": bool(
                np.array_equal(q7_corrected[:, 6], q7_raw[:, 6])
            ),
            "all_raw_q7_columns_exact": bool(
                np.array_equal(q7_corrected, q7_raw)
            ),
        },
    }
    report_path = args.output_dir / "registration_report.json"
    raw_builder.write_json(report_path, report)
    print(
        json.dumps(
            {
                "passed": True,
                "version": report["version"],
                "states": state_count,
                "strict_pairs": len(pair_timestamps_ns),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
