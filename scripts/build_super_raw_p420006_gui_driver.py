#!/usr/bin/env python3
"""Build a second raw-kinematics GUI version using the dVRK P420006 CAD.

The motion and coordinate chain remain the frozen raw backbone:

    original bag q7 + LND.json + calibration + hand-eye + current GUI world.

Only the seven tool visual/contact meshes and their fixed link-frame adapters
are replaced.  The Classic arm topology is retained so this version can be
selected by the existing GUI without replacing the robot or its q7 convention.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

import build_super_raw_psm_gui_driver as raw_builder


PREFIX = raw_builder.PREFIX

# Destination names deliberately match the existing GUI/contact contract.
# Each tuple contains the native Si P420006 link, raw LND link id and mesh.
P420006_LINKS = {
    "PSM1_tool_main_link": (
        "PSM1_tool_main_link",
        3,
        "tool_main_link.STL",
    ),
    "PSM1_tool_wrist_link": (
        "PSM1_tool_wrist_link",
        4,
        "tool_wrist_link.STL",
    ),
    "PSM1_tool_wrist_shaft_link": (
        "PSM1_tool_wrist_shaft_link",
        4,
        "tool_wrist_shaft_link.STL",
    ),
    # The Si model calls this wrist link "scal"; the GUI contract uses "sca".
    "PSM1_tool_wrist_sca_link": (
        "PSM1_tool_wrist_scal_link",
        5,
        "tool_wrist_scal_link.STL",
    ),
    "PSM1_tool_wrist_sca_shaft_link": (
        "PSM1_tool_wrist_sca_shaft_link",
        6,
        "tool_wrist_sca_shaft_link.STL",
    ),
    "PSM1_tool_wrist_sca_ee_link_1": (
        "PSM1_tool_wrist_sca_ee_link_1",
        7,
        "tool_wrist_sca_ee_link_1.STL",
    ),
    "PSM1_tool_wrist_sca_ee_link_2": (
        "PSM1_tool_wrist_sca_ee_link_2",
        8,
        "tool_wrist_sca_ee_link_2.STL",
    ),
}

# Anatomical axis registration at the shared distal hinge:
#   P420006 +x (wrist-yaw/jaw hinge) -> raw LND +z
#   P420006 +z (jaw long direction)   -> raw LND +y
#   P420006 +y                        -> raw LND +x
X_LND6_P420006_DISTAL = np.asarray(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    raw_root = (
        repo / "data/super/psm_raw_kinematics_v1（纯机器人学版本）"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Build the independent raw-P420006 GUI version without reading "
            "historical pose/CAD derivatives."
        )
    )
    parser.add_argument("--raw-root", type=Path, default=raw_root)
    parser.add_argument(
        "--dvrk-root", type=Path, default=repo / "data/dvrk_model"
    )
    parser.add_argument(
        "--table-frame",
        type=Path,
        default=repo / "data/super/table_frame.json",
    )
    parser.add_argument(
        "--camera-manifest",
        type=Path,
        default=repo / "data/super/grasp5_offline_demo/cameras.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=raw_root / "gui_p420006_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def add_mesh_visual(
    link: raw_builder.ET.Element,
    mesh_uri: str,
    color: str,
) -> None:
    for tag in ("visual", "collision"):
        for element in list(link.findall(tag)):
            link.remove(element)

    visual = raw_builder.ET.SubElement(link, "visual")
    raw_builder.ET.SubElement(
        visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"}
    )
    geometry = raw_builder.ET.SubElement(visual, "geometry")
    raw_builder.ET.SubElement(geometry, "mesh", {"filename": mesh_uri})
    material = raw_builder.ET.SubElement(
        visual, "material", {"name": "p420006"}
    )
    raw_builder.ET.SubElement(material, "color", {"rgba": color})

    collision = raw_builder.ET.SubElement(link, "collision")
    raw_builder.ET.SubElement(
        collision, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"}
    )
    collision_geometry = raw_builder.ET.SubElement(collision, "geometry")
    raw_builder.ET.SubElement(
        collision_geometry, "mesh", {"filename": mesh_uri}
    )


def replace_tool_geometry_with_p420006(
    robot: raw_builder.ET.Element,
) -> None:
    prefix = "package://dvrk_model/meshes/instruments/420006/"
    for destination, (_native, _lnd_link, mesh_name) in P420006_LINKS.items():
        link = robot.find(f"./link[@name='{destination}']")
        if link is None:
            raise KeyError(f"Classic topology is missing {destination}")
        color = (
            "0.77647 0.75686 0.73725 1"
            if "ee_link" in destination
            else "0.79216 0.81961 0.93333 1"
        )
        add_mesh_visual(link, prefix + mesh_name, color)


def expand_native_p420006_tool(
    xacro_path: Path,
) -> tuple[
    raw_builder.ET.Element,
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, list[str]],
]:
    macro = raw_builder.find_macro(xacro_path, "psm_tool")
    condition = next(
        (
            child
            for child in macro
            if child.tag == raw_builder.XACRO + "if"
            and "P420006" in child.get("value", "")
        ),
        None,
    )
    if condition is None:
        raise KeyError(f"Missing P420006 branch in {xacro_path}")

    robot = raw_builder.ET.Element("robot", {"name": "p420006_native"})
    robot.append(raw_builder.ET.Element("link", {"name": "world"}))
    for child in condition:
        clone = copy.deepcopy(child)
        raw_builder.substitute_tree(
            clone,
            {"prefix": PREFIX, "parent_link": "world"},
        )
        robot.append(clone)
    if any(element.tag.startswith(raw_builder.XACRO) for element in robot.iter()):
        raise RuntimeError("Native P420006 expansion still contains xacro")
    mimic = raw_builder.extract_mimic_map(robot)
    expected = {"jaw_1", "jaw_2"}
    if set(mimic) != expected:
        raise RuntimeError(
            f"Unexpected P420006 mimic joints: {sorted(mimic)}"
        )
    joints, children = raw_builder.parse_urdf_tree(robot)
    return robot, mimic, joints, children


def p420006_fk(
    joints: dict[str, dict[str, Any]],
    children: dict[str, list[str]],
    mimic: dict[str, dict[str, Any]],
    q7: np.ndarray,
) -> dict[str, np.ndarray]:
    # P420006 and LND use opposite signs for wrist pitch; the other distal
    # encoder conventions agree after the anatomical axis registration above.
    values = {name: 0.0 for name in joints}
    values.update(
        {
            "roll": float(q7[3]),
            "wrist_pitch": -float(q7[4]),
            "wrist_yaw": float(q7[5]),
            "jaw": float(q7[6]),
        }
    )
    for joint_name, spec in mimic.items():
        values[joint_name] = (
            values[str(spec["source"])] * float(spec["multiplier"])
            + float(spec["offset"])
        )
    return raw_builder.urdf_fk(joints, children, values)


def compatibility_gate(
    lnd_model: dict[str, Any],
    link_offsets: dict[str, np.ndarray],
    X_lnd_base_p420006: np.ndarray,
    p420006_joints: dict[str, dict[str, Any]],
    p420006_children: dict[str, list[str]],
    p420006_mimic: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rng = np.random.default_rng(420006)
    samples = np.zeros((256, 7), dtype=np.float64)
    samples[:, 3] = rng.uniform(-2.0, 2.0, len(samples))
    samples[:, 4] = rng.uniform(-1.0, 1.0, len(samples))
    samples[:, 5] = rng.uniform(-1.0, 1.0, len(samples))
    samples[:, 6] = rng.uniform(-0.5, 1.0, len(samples))

    max_translation_m = 0.0
    max_rotation_deg = 0.0
    worst: dict[str, Any] | None = None
    for sample_index, q7 in enumerate(samples):
        lnd_fk = raw_builder.lnd_fk(lnd_model, q7)
        native_fk = p420006_fk(
            p420006_joints,
            p420006_children,
            p420006_mimic,
            q7,
        )
        for destination, (native, lnd_link, _mesh) in P420006_LINKS.items():
            from_raw_lnd = lnd_fk[lnd_link] @ link_offsets[destination]
            from_native_p420006 = X_lnd_base_p420006 @ native_fk[native]
            translation_m = float(
                np.linalg.norm(
                    from_raw_lnd[:3, 3] - from_native_p420006[:3, 3]
                )
            )
            rotation_deg = float(
                np.degrees(
                    Rotation.from_matrix(
                        from_raw_lnd[:3, :3].T
                        @ from_native_p420006[:3, :3]
                    ).magnitude()
                )
            )
            if (
                translation_m > max_translation_m
                or rotation_deg > max_rotation_deg
            ):
                worst = {
                    "sample_index": sample_index,
                    "destination_link": destination,
                    "q7": q7.tolist(),
                    "translation_m": translation_m,
                    "rotation_deg": rotation_deg,
                }
            max_translation_m = max(max_translation_m, translation_m)
            max_rotation_deg = max(max_rotation_deg, rotation_deg)

    # The two public descriptions differ by 0.403 mm inside the wrist.  This
    # gate permits that measured model discrepancy, but not a frame/axis error.
    passed = max_translation_m <= 0.0005 and max_rotation_deg <= 0.002
    return {
        "passed": passed,
        "samples": len(samples),
        "joint_mapping": {
            "roll": "+raw_q7[3]",
            "wrist_pitch": "-raw_q7[4]",
            "wrist_yaw": "+raw_q7[5]",
            "jaw": "+raw_q7[6]",
        },
        "maximum_translation_m": max_translation_m,
        "maximum_rotation_deg": max_rotation_deg,
        "limits": {
            "maximum_translation_m": 0.0005,
            "maximum_rotation_deg": 0.002,
        },
        "worst": worst,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    raw_model_path = args.raw_root / "model.json"
    raw_kinematics_path = args.raw_root / "kinematics.npz"
    raw_report_path = args.raw_root / "report.json"
    classic_sources = [
        args.dvrk_root / "urdf/Classic/PSM1.urdf.xacro",
        args.dvrk_root / "urdf/Classic/psm_base.urdf.xacro",
        args.dvrk_root / "urdf/Classic/psm_tool.urdf.xacro",
        args.dvrk_root / "urdf/Classic/psm_tool_sca.urdf.xacro",
        args.dvrk_root / "urdf/common.urdf.xacro",
    ]
    p420006_xacro = args.dvrk_root / "urdf/Si/psm_tool.urdf.xacro"
    p420006_mesh_root = (
        args.dvrk_root / "meshes/instruments/420006"
    )
    p420006_mesh_paths = [
        p420006_mesh_root / spec[2] for spec in P420006_LINKS.values()
    ]
    for path in (
        raw_model_path,
        raw_kinematics_path,
        raw_report_path,
        args.table_frame,
        args.camera_manifest,
        *classic_sources,
        p420006_xacro,
        *p420006_mesh_paths,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    raw_report = json.loads(raw_report_path.read_text(encoding="utf-8"))
    if raw_report.get("passed") is not True:
        raise RuntimeError("Raw kinematic backbone report has not passed")
    raw_model = json.loads(raw_model_path.read_text(encoding="utf-8"))
    table_frame = json.loads(args.table_frame.read_text(encoding="utf-8"))
    camera_manifest = json.loads(
        args.camera_manifest.read_text(encoding="utf-8")
    )
    X_table_camera = np.asarray(
        table_frame["X_table_camera"], dtype=np.float64
    )
    X_blender_from_opencv = np.diag([1.0, -1.0, -1.0, 1.0])
    manifest_camera = np.asarray(
        camera_manifest["stereo_left"]["X_WC"], dtype=np.float64
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
            "Current GUI camera manifest and table frame disagree: "
            f"{camera_error:.3e}"
        )

    raw_builder.prepare_output_dir(args.output_dir, args.overwrite)

    print("[1/5] Expanding fresh Classic arm topology")
    robot = raw_builder.expand_root_psm_xacro(args.dvrk_root)
    classic_mimic = raw_builder.extract_mimic_map(robot)
    expected_mimic = {
        "pitch_1",
        "pitch_2",
        "pitch_3",
        "pitch_4",
        "pitch_5",
        "jaw_mimic_1",
        "jaw_mimic_2",
    }
    if set(classic_mimic) != expected_mimic:
        raise RuntimeError(
            f"Unexpected Classic mimic joints: {sorted(classic_mimic)}"
        )
    raw_builder.widen_raw_limits(robot)

    print("[2/5] Replacing only tool geometry with native P420006 meshes")
    replace_tool_geometry_with_p420006(robot)
    mesh_reports = raw_builder.rewrite_root_meshes(
        robot, args.dvrk_root, args.output_dir / "meshes"
    )
    raw_builder.ensure_collision_and_inertial(robot)
    urdf_path = args.output_dir / "psm_p420006.urdf"
    raw_builder.write_fresh_urdf(
        robot, urdf_path, [*classic_sources, p420006_xacro]
    )
    mimic_path = args.output_dir / "psm_p420006_mimic_map.json"
    raw_builder.write_json(
        mimic_path,
        {
            "source": (
                "fresh Classic arm topology; P420006 tool geometry; "
                "root xacro mimic tags"
            ),
            "input_joint_names": list(raw_builder.INPUT_JOINT_NAMES),
            "input_to_urdf_joint": raw_builder.INPUT_TO_URDF_JOINT,
            "mimic": classic_mimic,
        },
    )

    print("[3/5] Registering Classic arm base and native P420006 distal axes")
    classic_joints, classic_children = raw_builder.parse_urdf_tree(robot)
    canonical_classic_world = raw_builder.urdf_fk(
        classic_joints,
        classic_children,
        raw_builder.expand_q7(np.zeros(7), classic_mimic),
    )
    canonical_classic_base = canonical_classic_world["PSM1_psm_base_link"]
    canonical_classic = {
        name: np.linalg.inv(canonical_classic_base) @ transform
        for name, transform in canonical_classic_world.items()
    }
    canonical_lnd = raw_builder.lnd_fk(
        raw_model["lnd"], np.zeros(7, dtype=np.float64)
    )
    source_points = np.asarray(
        [
            canonical_classic[urdf_link][:3, 3]
            for urdf_link, _lnd_link, _weight
            in raw_builder.MODEL_ALIGNMENT_CORRESPONDENCES
        ]
    )
    target_points = np.asarray(
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
    X_lnd_base_classic_base = raw_builder.weighted_rigid_fit(
        source_points, target_points, weights
    )

    (
        _p420006_robot,
        p420006_mimic,
        p420006_joints,
        p420006_children,
    ) = expand_native_p420006_tool(p420006_xacro)
    canonical_p420006 = p420006_fk(
        p420006_joints,
        p420006_children,
        p420006_mimic,
        np.zeros(7, dtype=np.float64),
    )
    native_anchor = P420006_LINKS[
        "PSM1_tool_wrist_sca_shaft_link"
    ][0]
    X_lnd_base_p420006 = (
        canonical_lnd[6]
        @ X_LND6_P420006_DISTAL
        @ np.linalg.inv(canonical_p420006[native_anchor])
    )
    link_names = list(P420006_LINKS)
    link_offsets = {
        destination: (
            np.linalg.inv(canonical_lnd[lnd_link])
            @ X_lnd_base_p420006
            @ canonical_p420006[native]
        )
        for destination, (native, lnd_link, _mesh)
        in P420006_LINKS.items()
    }
    q0_errors = {
        destination: float(
            np.max(
                np.abs(
                    canonical_lnd[lnd_link] @ link_offsets[destination]
                    - X_lnd_base_p420006 @ canonical_p420006[native]
                )
            )
        )
        for destination, (native, lnd_link, _mesh)
        in P420006_LINKS.items()
    }
    if max(q0_errors.values()) > 1.0e-12:
        raise RuntimeError(
            f"P420006 q=0 assembly mismatch: {max(q0_errors.values()):.3e}"
        )
    compatibility = compatibility_gate(
        raw_model["lnd"],
        link_offsets,
        X_lnd_base_p420006,
        p420006_joints,
        p420006_children,
        p420006_mimic,
    )
    if not compatibility["passed"]:
        raise RuntimeError(
            "P420006/LND distal kinematic compatibility gate failed: "
            f"{compatibility}"
        )

    T_rectified_left_psm_base = np.asarray(
        raw_model["calibration_and_static_transforms"][
            "T_rectified_left_camera_psm_base"
        ],
        dtype=np.float64,
    )
    X_gui_world_urdf_base = (
        X_table_camera
        @ T_rectified_left_psm_base
        @ X_lnd_base_classic_base
    )

    print("[4/5] Building 5458-state P420006 driver in current GUI world")
    with np.load(raw_kinematics_path, allow_pickle=False) as raw:
        timestamps = raw["joint_timestamps_s"].astype(np.float64)
        timestamps_ros_ns = raw["joint_timestamps_ros_ns"].astype(np.int64)
        q7 = raw["q7"].astype(np.float64)
        T_rectified_left_lnd = raw[
            "T_rectified_left_camera_lnd_link"
        ].astype(np.float64)
    lnd_link_ids = np.asarray(
        [spec[1] for spec in P420006_LINKS.values()], dtype=np.int64
    )
    offset_array = np.stack([link_offsets[name] for name in link_names])
    T_rectified_left_p420006 = (
        T_rectified_left_lnd[:, lnd_link_ids] @ offset_array[None]
    )
    T_gui_world_p420006 = np.einsum(
        "ij,nljk->nlik", X_table_camera, T_rectified_left_p420006
    )
    poses_gui_world = raw_builder.matrix_series_to_poses(
        T_gui_world_p420006
    )
    if not np.isfinite(poses_gui_world).all():
        raise RuntimeError("P420006 GUI driver contains non-finite values")
    driver_path = args.output_dir / "psm_p420006_gui_pose_driver.npz"
    np.savez_compressed(
        driver_path,
        schema=np.asarray("super_psm_raw_p420006_gui_pose_driver_v1"),
        coordinate_frame=np.asarray("current_gui_table_world"),
        transform_convention=np.asarray(
            "T_GUIworld_P420006link = X_table_rectLeft @ "
            "T_rectLeft_LNDlink(raw) @ T_LNDlink_P420006link"
        ),
        timestamps=timestamps,
        timestamps_ros_ns=timestamps_ros_ns,
        q7=q7,
        link_names=np.asarray(link_names),
        lnd_link_ids=lnd_link_ids,
        poses_gui_world_xyz_xyzw=poses_gui_world,
        T_lndlink_urdf_link=offset_array,
        X_gui_world_rectified_left_camera=X_table_camera,
        X_gui_world_urdf_base=X_gui_world_urdf_base,
    )

    print("[5/5] Writing provenance and registration report")
    p420_mesh_report = {
        str(path.resolve()): {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in p420006_mesh_paths
    }
    report = {
        "schema": "super_psm_raw_p420006_gui_registration_v1",
        "passed": True,
        "version": "raw_p420006",
        "method": (
            "frozen raw q7/LND/hand-eye/rectification backbone + "
            "anatomically registered real-model P420006 CAD + current GUI world"
        ),
        "uses_historical_psm_derivatives": False,
        "changes_from_raw_kinematics_v1": [
            "seven tool visual/contact meshes",
            "fixed LND-link to P420006-link adapters",
        ],
        "unchanged_from_raw_kinematics_v1": [
            "original q7 and timestamps",
            "LND.json forward kinematics",
            "camera calibration and rectification",
            "hand-eye transform",
            "current GUI table/world transform",
            "Classic arm topology and q7/mimic contract",
        ],
        "coordinate_chain": (
            "T_GUIworld_P420006link(t) = "
            "X_table_rectifiedLeftCamera(current GUI) @ "
            "T_rectifiedLeftCamera_LNDlink(t, raw backbone) @ "
            "T_LNDlink_P420006link(fixed)"
        ),
        "forbidden_inputs": [
            "data/super/psm_robot/*",
            "data/super/psm_tracking/*",
            "historical strict/registered/corrected/depth pose drivers",
            "raw_kinematics gui_v1 pose/CAD/registration outputs",
        ],
        "inputs": {
            "raw_model": {
                "path": str(raw_model_path.resolve()),
                "sha256": sha256(raw_model_path),
            },
            "raw_kinematics": {
                "path": str(raw_kinematics_path.resolve()),
                "sha256": sha256(raw_kinematics_path),
            },
            "raw_report": {
                "path": str(raw_report_path.resolve()),
                "sha256": sha256(raw_report_path),
                "source_bag_sha256": raw_report["inputs"]["bag"]["sha256"],
            },
            "classic_topology_xacros": {
                str(path.resolve()): sha256(path) for path in classic_sources
            },
            "p420006_xacro": {
                "path": str(p420006_xacro.resolve()),
                "sha256": sha256(p420006_xacro),
            },
            "p420006_meshes": p420_mesh_report,
            "current_gui_table_frame": {
                "path": str(args.table_frame.resolve()),
                "sha256": sha256(args.table_frame),
            },
            "current_gui_camera_manifest": {
                "path": str(args.camera_manifest.resolve()),
                "sha256": sha256(args.camera_manifest),
            },
        },
        "gui_coordinate_gate": {
            "X_gui_world_rectified_left_camera": X_table_camera.tolist(),
            "camera_manifest_max_error": camera_error,
        },
        "fresh_urdf": {
            "path": str(urdf_path.resolve()),
            "sha256": sha256(urdf_path),
            "mimic_map": str(mimic_path.resolve()),
            "converted_mesh_count": len(mesh_reports),
            "p420006_destination_links": link_names,
        },
        "canonical_registration": {
            "link_mapping": {
                name: spec[1] for name, spec in P420006_LINKS.items()
            },
            "native_p420006_link_mapping": {
                name: spec[0] for name, spec in P420006_LINKS.items()
            },
            "anatomical_axis_constraints": {
                "P420006_x_to_LND6": "+z",
                "P420006_z_to_LND6": "+y",
                "P420006_y_to_LND6": "+x",
            },
            "X_LND6_P420006_distal": (
                X_LND6_P420006_DISTAL.tolist()
            ),
            "X_lnd_base_p420006": X_lnd_base_p420006.tolist(),
            "T_lndlink_urdf_link": {
                name: link_offsets[name].tolist() for name in link_names
            },
            "q0_max_abs_errors": q0_errors,
            "compatibility_with_native_p420006_urdf": compatibility,
        },
        "manual_offset_conventions": {
            "roll": {
                "parent_link": "PSM1_tool_main_link",
                "origin_xyz": [0.05591, 0.0, -0.53559],
                "axis_xyz": [0.0, 0.0, 1.0],
            },
            "jaw": {
                "parent_link": "PSM1_tool_wrist_sca_shaft_link",
                "origin_xyz": [0.0, 0.0, 0.0],
                "axis_xyz": [1.0, 0.0, 0.0],
                "child_links": [
                    "PSM1_tool_wrist_sca_ee_link_1",
                    "PSM1_tool_wrist_sca_ee_link_2",
                ],
                "half_angle_signs": [1.0, -1.0],
            },
        },
        "driver": {
            "path": str(driver_path.resolve()),
            "sha256": sha256(driver_path),
            "states": len(timestamps),
            "links": len(link_names),
            "poses_shape": list(poses_gui_world.shape),
            "q7_shape": list(q7.shape),
            "coordinate_frame": "current_gui_table_world",
            "X_gui_world_urdf_base": X_gui_world_urdf_base.tolist(),
        },
        "outputs": {
            "urdf": urdf_path.name,
            "mimic_map": mimic_path.name,
            "pose_driver": driver_path.name,
            "surface_gaussians_pending": (
                "psm_p420006_surface_gaussians.npz"
            ),
        },
    }
    report_path = args.output_dir / "registration_report.json"
    raw_builder.write_json(report_path, report)
    print(
        json.dumps(
            {
                "passed": True,
                "version": "raw_p420006",
                "states": len(timestamps),
                "links": len(link_names),
                "native_compatibility_max_mm": (
                    compatibility["maximum_translation_m"] * 1000.0
                ),
                "native_compatibility_max_deg": (
                    compatibility["maximum_rotation_deg"]
                ),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
