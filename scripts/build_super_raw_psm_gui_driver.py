#!/usr/bin/env python3
"""Build fresh GUI assets from the raw LND backbone and root dVRK model.

Allowed pose inputs:

* ``data/super/psm_raw_kinematics_v1（纯机器人学版本）/``
  ``{model.json,kinematics.npz,report.json}``
* root ``data/dvrk_model`` xacro/meshes
* the current GUI table/world coordinate definition

The script never reads a historical PSM URDF, Gaussian asset, registration
report, or pose driver.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


XACRO_NS = "http://www.ros.org/wiki/xacro"
XACRO = f"{{{XACRO_NS}}}"
PREFIX = "PSM1_"

INPUT_JOINT_NAMES = (
    "outer_yaw",
    "outer_pitch",
    "outer_insertion",
    "outer_roll",
    "outer_wrist_pitch",
    "outer_wrist_yaw",
    "jaw",
)
INPUT_TO_URDF_JOINT = {
    "outer_yaw": "yaw",
    "outer_pitch": "pitch",
    "outer_insertion": "insertion",
    "outer_roll": "roll",
    "outer_wrist_pitch": "wrist_pitch",
    "outer_wrist_yaw": "wrist_yaw",
    "jaw": "jaw",
}
URDF_TO_LND_LINK = {
    "PSM1_tool_main_link": 3,
    "PSM1_tool_wrist_link": 4,
    "PSM1_tool_wrist_shaft_link": 4,
    "PSM1_tool_wrist_sca_link": 5,
    "PSM1_tool_wrist_sca_shaft_link": 6,
    "PSM1_tool_wrist_sca_ee_link_1": 7,
    "PSM1_tool_wrist_sca_ee_link_2": 8,
}
MODEL_ALIGNMENT_CORRESPONDENCES = (
    ("PSM1_psm_base_link", 0, 3.0),
    ("PSM1_outer_pitch_link", 2, 3.0),
    ("PSM1_tool_main_link", 3, 3.0),
    ("PSM1_tool_wrist_link", 4, 2.0),
    ("PSM1_tool_wrist_sca_link", 5, 2.0),
    ("PSM1_tool_wrist_sca_shaft_link", 6, 2.0),
)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    raw_root = (
        repo / "data/super/psm_raw_kinematics_v1（纯机器人学版本）"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Build a fresh LND-to-dVRK registration and current-GUI-world pose "
            "driver without historical PSM derivatives."
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
        help="Current GUI world definition: X_table_camera.",
    )
    parser.add_argument(
        "--camera-manifest",
        type=Path,
        default=repo / "data/super/grasp5_offline_demo/cameras.json",
        help="Current GUI camera manifest, used only for a coordinate gate.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=raw_root / "gui_v1"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def find_macro(path: Path, name: str) -> ET.Element:
    root = ET.parse(path).getroot()
    macro = root.find(f"{XACRO}macro[@name='{name}']")
    if macro is None:
        raise KeyError(f"Missing xacro macro {name!r} in {path}")
    return macro


def substitute_xacro_value(value: str, variables: dict[str, str]) -> str:
    replacements = {
        **variables,
        "PI": f"{math.pi:.16g}",
        "PI/2": f"{math.pi / 2.0:.16g}",
    }

    def replace(match: re.Match[str]) -> str:
        expression = match.group(1).strip()
        if expression not in replacements:
            raise ValueError(f"Unsupported xacro expression ${{{expression}}}")
        return replacements[expression]

    return re.sub(r"\$\{([^}]+)\}", replace, value)


def substitute_tree(element: ET.Element, variables: dict[str, str]) -> None:
    for key, value in list(element.attrib.items()):
        element.set(key, substitute_xacro_value(value, variables))
    if element.text:
        element.text = substitute_xacro_value(element.text, variables)
    if element.tail:
        element.tail = substitute_xacro_value(element.tail, variables)
    for child in element:
        substitute_tree(child, variables)


def expand_root_psm_xacro(dvrk_root: Path) -> ET.Element:
    base_path = dvrk_root / "urdf/Classic/psm_base.urdf.xacro"
    tool_path = dvrk_root / "urdf/Classic/psm_tool_sca.urdf.xacro"
    base_macro = find_macro(base_path, "psm_base")
    tool_macro = find_macro(tool_path, "psm_tool_sca")

    robot = ET.Element("robot", {"name": "dvrk_psm_raw_lnd"})
    robot.append(ET.Element("link", {"name": "world"}))
    base_variables = {
        "prefix": PREFIX,
        "parent_link": "world",
        "xyz": "0 0 0",
        "rpy": "0 0 0",
    }
    for child in base_macro:
        clone = copy.deepcopy(child)
        substitute_tree(clone, base_variables)
        robot.append(clone)
    for child in tool_macro:
        clone = copy.deepcopy(child)
        substitute_tree(clone, {"prefix": PREFIX})
        robot.append(clone)

    if any(element.tag.startswith(XACRO) for element in robot.iter()):
        raise RuntimeError("Fresh PSM expansion still contains xacro elements")
    fixed = robot.find("./joint[@name='fixed']/origin")
    if fixed is None:
        raise RuntimeError("Expanded PSM has no fixed world-to-base origin")
    if fixed.get("xyz") != "0 0 0" or fixed.get("rpy") != "0 0 0":
        raise RuntimeError("Fresh PSM base is not identity")
    return robot


def source_mesh_path(dvrk_root: Path, filename: str) -> Path:
    prefix = "package://dvrk_model/"
    if not filename.startswith(prefix):
        raise ValueError(f"Unexpected root dVRK mesh URI: {filename}")
    path = dvrk_root / filename[len(prefix) :]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_source_mesh(path: Path) -> trimesh.Trimesh:
    if path.suffix.lower() == ".dae":
        try:
            scene = trimesh.load_scene(path, process=False)
        except ImportError as error:
            scene = None
        if scene is None:
            mesh = load_simple_collada_visual_mesh(path)
        else:
            mesh = scene.to_geometry()
    else:
        loaded = trimesh.load(path, process=False)
        if isinstance(loaded, trimesh.Scene):
            mesh = loaded.to_geometry()
        elif isinstance(loaded, trimesh.Trimesh):
            mesh = loaded
        else:
            raise TypeError(f"Unsupported mesh object {type(loaded)!r} from {path}")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"Root dVRK mesh has no triangles: {path}")
    return mesh


def load_simple_collada_visual_mesh(path: Path) -> trimesh.Trimesh:
    """Read the simple triangle-only dVRK Classic DAE files without pycollada.

    The source files contain one identity-transformed visual geometry plus a
    separate PhysX collision geometry.  Only geometries instantiated by
    ``library_visual_scenes`` are selected, matching trimesh/pycollada.
    """

    root = ET.parse(path).getroot()
    namespace = root.tag.split("}", 1)[0] + "}" if "}" in root.tag else ""

    def tag(name: str) -> str:
        return namespace + name

    visual_scenes = root.find(tag("library_visual_scenes"))
    if visual_scenes is None:
        raise ValueError(f"COLLADA file has no visual scene: {path}")
    selected_geometry_ids = {
        instance.get("url", "").removeprefix("#")
        for instance in visual_scenes.iter(tag("instance_geometry"))
        if instance.get("url")
    }
    if not selected_geometry_ids:
        raise ValueError(f"COLLADA visual scene has no geometry: {path}")

    pieces: list[trimesh.Trimesh] = []
    library = root.find(tag("library_geometries"))
    if library is None:
        raise ValueError(f"COLLADA file has no geometry library: {path}")
    for geometry in library.findall(tag("geometry")):
        if geometry.get("id") not in selected_geometry_ids:
            continue
        mesh_element = geometry.find(tag("mesh"))
        if mesh_element is None:
            continue
        sources: dict[str, np.ndarray] = {}
        for source in mesh_element.findall(tag("source")):
            source_id = source.get("id")
            float_array = source.find(tag("float_array"))
            accessor = source.find(
                f"{tag('technique_common')}/{tag('accessor')}"
            )
            if source_id is None or float_array is None:
                continue
            values = np.fromstring(float_array.text or "", sep=" ")
            stride = int(
                accessor.get("stride", "1")
                if accessor is not None
                else "1"
            )
            if len(values) % stride != 0:
                raise ValueError(
                    f"Malformed COLLADA source {source_id} in {path}"
                )
            sources[source_id] = values.reshape(-1, stride)

        vertex_positions: dict[str, np.ndarray] = {}
        for vertices in mesh_element.findall(tag("vertices")):
            vertices_id = vertices.get("id")
            position_input = next(
                (
                    item
                    for item in vertices.findall(tag("input"))
                    if item.get("semantic") == "POSITION"
                ),
                None,
            )
            if vertices_id is None or position_input is None:
                continue
            source_id = position_input.get("source", "").removeprefix("#")
            if source_id not in sources:
                raise KeyError(
                    f"Missing COLLADA position source {source_id} in {path}"
                )
            vertex_positions[vertices_id] = sources[source_id][:, :3]

        for triangles in mesh_element.findall(tag("triangles")):
            inputs = triangles.findall(tag("input"))
            if not inputs:
                continue
            vertex_input = next(
                (
                    item
                    for item in inputs
                    if item.get("semantic") == "VERTEX"
                ),
                None,
            )
            indices_element = triangles.find(tag("p"))
            if vertex_input is None or indices_element is None:
                continue
            stride = max(int(item.get("offset", "0")) for item in inputs) + 1
            vertex_offset = int(vertex_input.get("offset", "0"))
            vertices_id = vertex_input.get("source", "").removeprefix("#")
            vertices_array = vertex_positions.get(vertices_id)
            if vertices_array is None:
                raise KeyError(
                    f"Missing COLLADA vertices {vertices_id} in {path}"
                )
            raw_indices = np.fromstring(
                indices_element.text or "", sep=" ", dtype=np.int64
            )
            if len(raw_indices) % stride != 0:
                raise ValueError(f"Malformed COLLADA triangle list in {path}")
            faces = raw_indices.reshape(-1, stride)[:, vertex_offset]
            if len(faces) % 3 != 0:
                raise ValueError(f"Non-triangular COLLADA index list in {path}")
            pieces.append(
                trimesh.Trimesh(
                    vertices=vertices_array.copy(),
                    faces=faces.reshape(-1, 3),
                    process=False,
                )
            )
    if not pieces:
        raise ValueError(f"No visual triangles found in COLLADA file: {path}")
    return trimesh.util.concatenate(pieces)


def rewrite_root_meshes(
    robot: ET.Element, dvrk_root: Path, mesh_output_dir: Path
) -> list[dict[str, Any]]:
    mesh_output_dir.mkdir()
    converted_by_source: dict[Path, str] = {}
    reports: list[dict[str, Any]] = []
    for mesh_element in robot.findall(".//mesh"):
        filename = mesh_element.get("filename")
        if filename is None:
            raise ValueError("Mesh element has no filename")
        source = source_mesh_path(dvrk_root, filename)
        if source not in converted_by_source:
            output_name = f"{source.stem.lower()}.stl"
            output = mesh_output_dir / output_name
            mesh = load_source_mesh(source)
            mesh.export(output)
            converted_by_source[source] = f"meshes/{output_name}"
            reports.append(
                {
                    "source": str(source.resolve()),
                    "source_sha256": sha256_file(source),
                    "output": converted_by_source[source],
                    "output_sha256": sha256_file(output),
                    "vertices": int(len(mesh.vertices)),
                    "triangles": int(len(mesh.faces)),
                    "bounds_m": np.asarray(mesh.bounds).tolist(),
                    "surface_area_m2": float(mesh.area),
                }
            )
        mesh_element.set("filename", converted_by_source[source])
    return reports


def clone_element(element: ET.Element) -> ET.Element:
    return ET.fromstring(ET.tostring(element, encoding="unicode"))


def ensure_collision_and_inertial(robot: ET.Element) -> None:
    for link in robot.findall("link"):
        name = link.get("name")
        if name == "world":
            continue
        if link.find("collision") is None:
            visuals = link.findall("visual")
            if visuals:
                for visual in visuals:
                    collision = ET.Element("collision")
                    origin = visual.find("origin")
                    geometry = visual.find("geometry")
                    if origin is not None:
                        collision.append(clone_element(origin))
                    if geometry is not None:
                        collision.append(clone_element(geometry))
                    link.append(collision)
            else:
                collision = ET.Element("collision")
                ET.SubElement(
                    collision, "origin", {"rpy": "0 0 0", "xyz": "0 0 0"}
                )
                geometry = ET.SubElement(collision, "geometry")
                ET.SubElement(geometry, "box", {"size": "0.001 0.001 0.001"})
                link.append(collision)
        if link.find("inertial") is None:
            inertial = ET.Element("inertial")
            ET.SubElement(inertial, "mass", {"value": "0.001"})
            ET.SubElement(
                inertial,
                "inertia",
                {
                    "ixx": "1e-8",
                    "ixy": "0",
                    "ixz": "0",
                    "iyy": "1e-8",
                    "iyz": "0",
                    "izz": "1e-8",
                },
            )
            link.append(inertial)


def extract_mimic_map(robot: ET.Element) -> dict[str, dict[str, Any]]:
    mimic_map: dict[str, dict[str, Any]] = {}
    for joint in robot.findall("joint"):
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        mimic_map[joint.get("name", "")] = {
            "source": mimic.get("joint", ""),
            "multiplier": float(mimic.get("multiplier", "1")),
            "offset": float(mimic.get("offset", "0")),
        }
        joint.remove(mimic)
    return mimic_map


def widen_raw_limits(robot: ET.Element) -> None:
    for joint_name, lower, upper in (
        ("roll", "-3.5", "3.5"),
        ("jaw", "-1.2", "1.6"),
    ):
        joint = robot.find(f"./joint[@name='{joint_name}']")
        if joint is None:
            raise KeyError(f"Missing root dVRK joint {joint_name}")
        limit = joint.find("limit")
        if limit is None:
            limit = ET.SubElement(joint, "limit")
        limit.set("lower", lower)
        limit.set("upper", upper)
        limit.set("velocity", limit.get("velocity", "50"))
        limit.set("effort", limit.get("effort", "1000"))


def write_fresh_urdf(
    robot: ET.Element, output: Path, source_files: list[Path]
) -> None:
    robot.insert(
        0,
        ET.Comment(
            "Freshly expanded from root data/dvrk_model for raw LND GUI mode; "
            "world-to-base is identity and no historical PSM URDF is an input."
        ),
    )
    robot.insert(
        1,
        ET.Comment(
            "Mimic tags are saved separately and removed for Warp parser compatibility."
        ),
    )
    robot.insert(
        2,
        ET.Comment(
            "Root xacro SHA256: "
            + ", ".join(f"{path.name}={sha256_file(path)}" for path in source_files)
        ),
    )
    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(output, encoding="utf-8", xml_declaration=True)


def parse_vector(value: str | None, length: int) -> np.ndarray:
    if value is None:
        return np.zeros(length, dtype=np.float64)
    result = np.asarray([float(item) for item in value.split()], dtype=np.float64)
    if result.shape != (length,):
        raise ValueError(f"Expected {length} values, got {value!r}")
    return result


def parse_urdf_tree(
    robot: ET.Element,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    joints: dict[str, dict[str, Any]] = {}
    children: dict[str, list[str]] = {}
    for joint in robot.findall("joint"):
        name = joint.get("name")
        parent = joint.find("parent")
        child = joint.find("child")
        if name is None or parent is None or child is None:
            raise ValueError("Malformed joint in fresh URDF")
        origin = joint.find("origin")
        axis = joint.find("axis")
        joints[name] = {
            "type": joint.get("type", "fixed"),
            "parent": parent.get("link"),
            "child": child.get("link"),
            "xyz": parse_vector(None if origin is None else origin.get("xyz"), 3),
            "rpy": parse_vector(None if origin is None else origin.get("rpy"), 3),
            "axis": parse_vector(
                "0 0 1" if axis is None else axis.get("xyz"), 3
            ),
        }
        children.setdefault(parent.get("link", ""), []).append(name)
    return joints, children


def origin_matrix(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


def expand_q7(q7: np.ndarray, mimic_map: dict[str, dict[str, Any]]) -> dict[str, float]:
    values = {
        INPUT_TO_URDF_JOINT[name]: float(value)
        for name, value in zip(INPUT_JOINT_NAMES, q7, strict=True)
    }
    for joint_name, spec in mimic_map.items():
        values[joint_name] = (
            values[str(spec["source"])] * float(spec["multiplier"])
            + float(spec["offset"])
        )
    return values


def urdf_fk(
    joints: dict[str, dict[str, Any]],
    children: dict[str, list[str]],
    q_by_joint: dict[str, float],
) -> dict[str, np.ndarray]:
    transforms = {"world": np.eye(4, dtype=np.float64)}
    queue = ["world"]
    while queue:
        parent = queue.pop(0)
        for joint_name in children.get(parent, []):
            joint = joints[joint_name]
            motion = np.eye(4, dtype=np.float64)
            value = float(q_by_joint.get(joint_name, 0.0))
            axis = np.asarray(joint["axis"], dtype=np.float64)
            axis /= max(np.linalg.norm(axis), 1e-12)
            if joint["type"] in {"revolute", "continuous"}:
                motion[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
            elif joint["type"] == "prismatic":
                motion[:3, 3] = axis * value
            elif joint["type"] != "fixed":
                raise ValueError(f"Unsupported URDF joint type {joint['type']}")
            child = str(joint["child"])
            transforms[child] = (
                transforms[parent]
                @ origin_matrix(joint["xyz"], joint["rpy"])
                @ motion
            )
            queue.append(child)
    return transforms


def rotx(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray(
        [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )


def rotz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray(
        [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )


def transx(distance: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = distance
    return transform


def transz(distance: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[2, 3] = distance
    return transform


def lnd_fk(lnd: dict[str, Any], q7: np.ndarray) -> np.ndarray:
    transforms = np.empty((9, 4, 4), dtype=np.float64)
    transforms[0] = np.eye(4)
    current = transforms[0].copy()
    for index, dh in enumerate(lnd["DH_params"], start=1):
        theta = float(dh.get("theta", 0.0))
        distance = float(dh.get("D", 0.0))
        offset = float(dh.get("offset", 0.0))
        if dh["type"] == "revolute":
            theta += float(q7[index - 1]) + offset
        elif dh["type"] == "prismatic":
            distance += float(q7[index - 1]) + offset
        else:
            raise ValueError(f"Unsupported LND joint type {dh['type']}")
        current = (
            current
            @ rotx(float(dh.get("alpha", 0.0)))
            @ transx(float(dh.get("A", 0.0)))
            @ rotz(theta)
            @ transz(distance)
        )
        transforms[index] = current
    transforms[7] = transforms[6] @ rotz(0.5 * float(q7[6]))
    transforms[8] = transforms[6] @ rotz(-0.5 * float(q7[6]))
    return transforms


def weighted_rigid_fit(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    normalized = weights / np.sum(weights)
    source_mean = np.sum(source * normalized[:, None], axis=0)
    target_mean = np.sum(target * normalized[:, None], axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (source_centered * weights[:, None]).T @ target_centered
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1] *= -1
        rotation = right_t.T @ left.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_mean - rotation @ source_mean
    return transform


def matrix_series_to_poses(matrices: np.ndarray) -> np.ndarray:
    poses = np.empty((*matrices.shape[:-2], 7), dtype=np.float32)
    poses[..., :3] = matrices[..., :3, 3]
    rotations = Rotation.from_matrix(matrices[..., :3, :3].reshape(-1, 3, 3))
    poses[..., 3:] = rotations.as_quat().reshape(*matrices.shape[:-2], 4)
    return poses


def angular_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.degrees(
            Rotation.from_matrix(first[:3, :3].T @ second[:3, :3]).magnitude()
        )
    )


def main() -> None:
    args = parse_args()
    raw_model_path = args.raw_root / "model.json"
    raw_kinematics_path = args.raw_root / "kinematics.npz"
    raw_report_path = args.raw_root / "report.json"
    source_xacros = [
        args.dvrk_root / "urdf/Classic/PSM1.urdf.xacro",
        args.dvrk_root / "urdf/Classic/psm_base.urdf.xacro",
        args.dvrk_root / "urdf/Classic/psm_tool.urdf.xacro",
        args.dvrk_root / "urdf/Classic/psm_tool_sca.urdf.xacro",
        args.dvrk_root / "urdf/common.urdf.xacro",
    ]
    for path in (
        raw_model_path,
        raw_kinematics_path,
        raw_report_path,
        args.table_frame,
        args.camera_manifest,
        *source_xacros,
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
    expected_gui_left_camera = X_table_camera @ X_blender_from_opencv
    manifest_gui_left_camera = np.asarray(
        camera_manifest["stereo_left"]["X_WC"], dtype=np.float64
    )
    gui_camera_error = float(
        np.max(np.abs(expected_gui_left_camera - manifest_gui_left_camera))
    )
    if gui_camera_error > 1e-12:
        raise RuntimeError(
            "Current GUI camera manifest and table frame disagree: "
            f"max error {gui_camera_error:.3e}"
        )

    prepare_output_dir(args.output_dir, args.overwrite)
    print("[1/4] Expanding root dVRK PSM xacro without historical URDF")
    robot = expand_root_psm_xacro(args.dvrk_root)
    mimic_map = extract_mimic_map(robot)
    expected_mimic = {
        "pitch_1",
        "pitch_2",
        "pitch_3",
        "pitch_4",
        "pitch_5",
        "jaw_mimic_1",
        "jaw_mimic_2",
    }
    if set(mimic_map) != expected_mimic:
        raise RuntimeError(f"Unexpected root dVRK mimic joints: {sorted(mimic_map)}")
    widen_raw_limits(robot)
    print("[2/4] Converting root dVRK visual meshes")
    mesh_reports = rewrite_root_meshes(
        robot, args.dvrk_root, args.output_dir / "meshes"
    )
    ensure_collision_and_inertial(robot)
    urdf_path = args.output_dir / "psm_raw.urdf"
    write_fresh_urdf(robot, urdf_path, source_xacros)
    mimic_config = {
        "source": "root data/dvrk_model xacro mimic tags",
        "input_joint_names": list(INPUT_JOINT_NAMES),
        "input_to_urdf_joint": INPUT_TO_URDF_JOINT,
        "mimic": mimic_map,
    }
    mimic_path = args.output_dir / "psm_raw_mimic_map.json"
    write_json(mimic_path, mimic_config)

    print("[3/4] Registering fresh dVRK link frames to raw LND at q=0")
    joints, children = parse_urdf_tree(robot)
    canonical_urdf_world = urdf_fk(
        joints, children, expand_q7(np.zeros(7), mimic_map)
    )
    canonical_urdf_base = canonical_urdf_world["PSM1_psm_base_link"]
    canonical_urdf = {
        name: np.linalg.inv(canonical_urdf_base) @ transform
        for name, transform in canonical_urdf_world.items()
    }
    lnd = raw_model["lnd"]
    canonical_lnd = lnd_fk(lnd, np.zeros(7, dtype=np.float64))
    source_points = np.asarray(
        [
            canonical_urdf[urdf_link][:3, 3]
            for urdf_link, _lnd_link, _weight in MODEL_ALIGNMENT_CORRESPONDENCES
        ]
    )
    target_points = np.asarray(
        [
            canonical_lnd[lnd_link, :3, 3]
            for _urdf_link, lnd_link, _weight in MODEL_ALIGNMENT_CORRESPONDENCES
        ]
    )
    weights = np.asarray(
        [weight for _urdf_link, _lnd_link, weight in MODEL_ALIGNMENT_CORRESPONDENCES]
    )
    T_lnd_base_urdf_base = weighted_rigid_fit(
        source_points, target_points, weights
    )
    alignment_residuals = []
    for correspondence, source, target in zip(
        MODEL_ALIGNMENT_CORRESPONDENCES,
        source_points,
        target_points,
        strict=True,
    ):
        predicted = (
            T_lnd_base_urdf_base @ np.r_[source, 1.0]
        )[:3]
        alignment_residuals.append(
            {
                "urdf_link": correspondence[0],
                "lnd_link": correspondence[1],
                "weight": correspondence[2],
                "residual_m": float(np.linalg.norm(predicted - target)),
            }
        )
    maximum_alignment_residual = max(
        item["residual_m"] for item in alignment_residuals
    )
    if maximum_alignment_residual > 2e-6:
        raise RuntimeError(
            "Fresh LND/dVRK canonical registration exceeds 2 micrometres: "
            f"{maximum_alignment_residual * 1e6:.3f} um"
        )

    link_names = list(URDF_TO_LND_LINK)
    T_lndlink_urdf_link = np.stack(
        [
            np.linalg.inv(canonical_lnd[URDF_TO_LND_LINK[name]])
            @ T_lnd_base_urdf_base
            @ canonical_urdf[name]
            for name in link_names
        ]
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
        @ T_lnd_base_urdf_base
    )

    print("[4/4] Converting raw LND poses into current GUI table/world")
    with np.load(raw_kinematics_path, allow_pickle=False) as raw:
        timestamps = raw["joint_timestamps_s"].astype(np.float64)
        timestamps_ros_ns = raw["joint_timestamps_ros_ns"].astype(np.int64)
        q7 = raw["q7"].astype(np.float64)
        T_rectified_left_lnd = raw[
            "T_rectified_left_camera_lnd_link"
        ].astype(np.float64)
    selected_lnd = T_rectified_left_lnd[
        :, np.asarray(list(URDF_TO_LND_LINK.values()), dtype=np.int64)
    ]
    T_rectified_left_urdf = selected_lnd @ T_lndlink_urdf_link[None]
    T_gui_world_urdf = np.einsum(
        "ij,nljk->nlik", X_table_camera, T_rectified_left_urdf
    )
    poses_gui_world = matrix_series_to_poses(T_gui_world_urdf)
    if not np.isfinite(poses_gui_world).all():
        raise RuntimeError("Fresh raw GUI pose driver contains non-finite values")

    driver_path = args.output_dir / "psm_raw_gui_pose_driver.npz"
    np.savez_compressed(
        driver_path,
        schema=np.asarray("super_psm_raw_gui_pose_driver_v1"),
        coordinate_frame=np.asarray("current_gui_table_world"),
        transform_convention=np.asarray(
            "T_GUIworld_CADlink = X_table_rectLeft @ "
            "T_rectLeft_LNDlink(raw) @ T_LNDlink_CADlink(fresh)"
        ),
        timestamps=timestamps,
        timestamps_ros_ns=timestamps_ros_ns,
        q7=q7,
        link_names=np.asarray(link_names),
        lnd_link_ids=np.asarray(list(URDF_TO_LND_LINK.values()), dtype=np.int64),
        poses_gui_world_xyz_xyzw=poses_gui_world,
        T_lndlink_urdf_link=T_lndlink_urdf_link,
        X_gui_world_rectified_left_camera=X_table_camera,
        X_gui_world_urdf_base=X_gui_world_urdf_base,
    )

    # Reconstruct a sparse set directly from the saved driver to gate serialization.
    with np.load(driver_path, allow_pickle=False) as saved:
        saved_poses = saved["poses_gui_world_xyz_xyzw"]
    maximum_pose_roundtrip = float(
        np.max(np.abs(saved_poses.astype(np.float32) - poses_gui_world))
    )
    if maximum_pose_roundtrip != 0.0:
        raise RuntimeError(
            f"GUI pose driver serialization changed values: {maximum_pose_roundtrip}"
        )

    report = {
        "schema": "super_psm_raw_gui_registration_report_v1",
        "passed": True,
        "method": (
            "raw q7/LND/hand-eye/rectification backbone -> fresh canonical "
            "root-dVRK registration -> current GUI table/world"
        ),
        "coordinate_chain": (
            "T_GUIworld_CADlink(t) = X_table_rectifiedLeftCamera(current GUI) "
            "@ T_rectifiedLeftCamera_LNDlink(t, raw backbone) "
            "@ T_LNDlink_CADlink(fresh q=0 registration)"
        ),
        "uses_historical_psm_derivatives": False,
        "forbidden_inputs": [
            "data/super/psm_robot/*",
            "data/super/psm_tracking/*",
            "historical strict/registered/corrected/depth pose drivers",
            "historical psm.urdf, surface Gaussian and registration report",
        ],
        "inputs": {
            "raw_model": {
                "path": str(raw_model_path.resolve()),
                "sha256": sha256_file(raw_model_path),
            },
            "raw_kinematics": {
                "path": str(raw_kinematics_path.resolve()),
                "sha256": sha256_file(raw_kinematics_path),
            },
            "raw_report": {
                "path": str(raw_report_path.resolve()),
                "sha256": sha256_file(raw_report_path),
                "source_bag_sha256": raw_report["inputs"]["bag"]["sha256"],
            },
            "root_dvrk_xacros": {
                str(path.resolve()): sha256_file(path) for path in source_xacros
            },
            "root_dvrk_meshes": mesh_reports,
            "current_gui_table_frame": {
                "path": str(args.table_frame.resolve()),
                "sha256": sha256_file(args.table_frame),
            },
            "current_gui_camera_manifest": {
                "path": str(args.camera_manifest.resolve()),
                "sha256": sha256_file(args.camera_manifest),
            },
        },
        "gui_coordinate_gate": {
            "convention": "GUI world is the current dense-ground table frame",
            "X_gui_world_rectified_left_camera": X_table_camera.tolist(),
            "opencv_to_blender_camera_axis_transform": (
                X_blender_from_opencv.tolist()
            ),
            "camera_manifest_max_error": gui_camera_error,
        },
        "fresh_urdf": {
            "path": str(urdf_path.resolve()),
            "sha256": sha256_file(urdf_path),
            "fixed_base_is_identity": True,
            "link_count": len(robot.findall("link")),
            "joint_count": len(robot.findall("joint")),
            "mimic_map": str(mimic_path.resolve()),
            "converted_mesh_count": len(mesh_reports),
        },
        "canonical_registration": {
            "link_mapping": URDF_TO_LND_LINK,
            "T_lnd_base_urdf_base": T_lnd_base_urdf_base.tolist(),
            "T_lndlink_urdf_link": {
                name: transform.tolist()
                for name, transform in zip(
                    link_names, T_lndlink_urdf_link, strict=True
                )
            },
            "residuals": alignment_residuals,
            "maximum_residual_m": maximum_alignment_residual,
        },
        "driver": {
            "path": str(driver_path.resolve()),
            "sha256": sha256_file(driver_path),
            "states": len(timestamps),
            "links": len(link_names),
            "poses_shape": list(poses_gui_world.shape),
            "q7_shape": list(q7.shape),
            "coordinate_frame": "current_gui_table_world",
            "serialization_max_abs_error": maximum_pose_roundtrip,
            "X_gui_world_urdf_base": X_gui_world_urdf_base.tolist(),
        },
        "outputs": {
            "urdf": urdf_path.name,
            "mimic_map": mimic_path.name,
            "pose_driver": driver_path.name,
            "surface_gaussians_pending": "psm_raw_surface_gaussians.npz",
        },
    }
    report_path = args.output_dir / "registration_report.json"
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "states": len(timestamps),
                "links": len(link_names),
                "canonical_registration_max_um": (
                    maximum_alignment_residual * 1e6
                ),
                "gui_camera_coordinate_error": gui_camera_error,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
