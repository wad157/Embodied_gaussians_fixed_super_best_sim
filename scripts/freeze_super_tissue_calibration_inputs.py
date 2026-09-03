#!/usr/bin/env python3
"""Freeze the inputs for SUPER tissue material calibration stage A.

This script is intentionally independent of Warp/CUDA.  It records the exact
stereo sequence, PSM trajectory/CAD, tetrahedral tissue asset, masks, runtime
configuration, and manually approved trajectory offsets used by later stages.
All paths stored in the manifest are repository-relative.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data/super/tissue_calibration_v1/stage_a_frozen_manifest.json"
)

POSE_DRIVER_NAME = "raw_paper_lnd_sam2_dense_contact_closedjaw"
POSE_ROOT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/"
    "raw_paper_lnd_stereo_dense_contact_v3/"
    "surgicalsam2_multianchor_parts_dense_contact_v5/"
    "online_stereo_cma_closedjaw_dense_contact_v2/gui_dense_contact_v2"
)
POSE_DRIVER_PATH = POSE_ROOT / "psm_paper_lnd_gui_pose_driver.npz"
POSE_URDF_PATH = POSE_ROOT / "psm_paper_lnd.urdf"
POSE_MIMIC_PATH = POSE_ROOT / "psm_paper_lnd_mimic_map.json"
POSE_GAUSSIANS_PATH = POSE_ROOT / "psm_paper_lnd_surface_gaussians.npz"
POSE_REPORT_PATH = POSE_ROOT / "registration_report.json"

DATASET_ROOT = REPO_ROOT / "data/super/grasp5_offline_demo"
LEFT_VIDEO_PATH = DATASET_ROOT / "videos/stereo_left.mp4"
RIGHT_VIDEO_PATH = DATASET_ROOT / "videos/stereo_right.mp4"
LEFT_METADATA_PATH = DATASET_ROOT / "videos/stereo_left.json"
RIGHT_METADATA_PATH = DATASET_ROOT / "videos/stereo_right.json"
CAMERA_MANIFEST_PATH = DATASET_ROOT / "cameras.json"
RECTIFIED_CALIBRATION_PATH = (
    REPO_ROOT / "data/super/grasp5_native/calib_rectified.json"
)

TISSUE_ROOT = (
    REPO_ROOT
    / "data/super/grasp5_native/tissue_multiview_v1/"
    "soft_tissue_adaptive_v1"
)
TISSUE_ASSET_PATH = TISSUE_ROOT / "tissue_soft_adaptive.npz"
TISSUE_METADATA_PATH = TISSUE_ROOT / "metadata.json"

RAW_IDENTITY_REPORT_PATH = (
    REPO_ROOT
    / "data/super/psm_raw_kinematics_v1（纯机器人学版本）/report.json"
)
RUNTIME_SOURCE_PATH = (
    REPO_ROOT
    / "examples/embodied_environments/super_embodied/super_embodied.py"
)
GUI_SOURCE_PATH = REPO_ROOT / "examples/example_embodied_super_offline.py"
THINLINC_ENTRYPOINT_PATH = REPO_ROOT / "scripts/run_demo_thinlinc.sh"

MANUAL_CORRECTIONS = {
    "psm_self_spin_offset_deg": -9.184,
    "manual_image_x_mm_right_positive": 0.332,
    "manual_image_y_mm_down_positive": 0.0,
    "manual_camera_z_mm_far_positive": -6.735,
    "psm_jaw_opening_offset_deg_open_positive": 0.0,
    "all_other_manual_pose_offsets": 0.0,
}

# Left-video landmarks reviewed on the frozen sequence.  Mechanical landmarks
# are additionally checked against the q7 trajectory below.
LANDMARK_FRAMES = {
    "initial_reference": 0,
    "first_contact": 270,
    "grasp_closed": 548,
    "traction_reference": 700,
    "maximum_retraction": 906,
    "release_complete": 1276,
    "rebound_reference": 1300,
}

CONTACT_LINKS = [
    "PSM1_tool_wrist_sca_ee_link_1",
    "PSM1_tool_wrist_sca_ee_link_2",
]

RUNTIME_CONSTANT_NAMES = [
    "PAPER_SOFT_TISSUE_YOUNG_MODULUS_PA",
    "PAPER_SOFT_TISSUE_POISSON_RATIO",
    "PAPER_SOFT_TISSUE_GRAVITY_M_S2",
    "PAPER_SOFT_TISSUE_VELOCITY_DAMPING_PER_SECOND",
    "PAPER_SOFT_TISSUE_MATERIAL_MIN_VOLUME_RATIO",
    "PAPER_SOFT_TISSUE_MATERIAL_ITERATIONS",
    "PAPER_SOFT_TISSUE_MATERIAL_RELAXATION",
    "PAPER_SOFT_TISSUE_SURFACE_MAX_CORRECTION_M",
    "PAPER_SOFT_TISSUE_SURFACE_ITERATIONS",
    "PAPER_SOFT_TISSUE_JAW_FRICTION_COEFFICIENT",
    "ADAPTIVE_TISSUE_CONTACT_MARGIN_M",
    "ADAPTIVE_TISSUE_CONTACT_QUERY_DISTANCE_M",
    "SUPER_VISUAL_FORCE_LR_MEANS",
    "SUPER_VISUAL_FORCE_KP",
    "SUPER_SOFT_VISUAL_FORCE_MAX_GAUSSIAN_N",
    "SUPER_SOFT_VISUAL_FORCE_MAX_PARTICLE_N",
    "SUPER_SOFT_VISUAL_FORCE_MAX_TOTAL_N",
    "SUPER_SOFT_VISUAL_FORCE_SPREAD_LAYERS",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing manifest without rewriting it.",
    )
    return parser.parse_args()


def repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": repo_path(path),
        "size_bytes": stat.st_size,
        "sha256": sha256(path),
    }


def trusted_large_file_record(
    path: Path, source_report: Path, source_key: str
) -> dict[str, Any]:
    report = json.loads(source_report.read_text())
    trusted = report["inputs"][source_key]
    stat = path.stat()
    if stat.st_size != int(trusted["size_bytes"]):
        raise RuntimeError(
            f"{path} size changed since {source_report}: "
            f"{stat.st_size} != {trusted['size_bytes']}"
        )
    return {
        "path": repo_path(path),
        "size_bytes": stat.st_size,
        "sha256": trusted["sha256"],
        "sha256_source": repo_path(source_report),
        "full_hash_reused_after_size_check": True,
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def literal_constants(path: Path, names: list[str]) -> dict[str, Any]:
    tree = ast.parse(path.read_text(), filename=str(path))
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value_node = node.value
        for target in targets:
            if isinstance(target, ast.Name) and target.id in names:
                try:
                    values[target.id] = ast.literal_eval(value_node)
                except (ValueError, TypeError):
                    pass
    missing = sorted(set(names) - set(values))
    if missing:
        raise RuntimeError(f"Could not read literal runtime constants: {missing}")
    return values


def git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "head": run("rev-parse", "HEAD"),
        "tracked_worktree_dirty": bool(status),
        "note": (
            "Relevant source files are content-hashed because this working "
            "tree contains uncommitted development."
        ),
    }


def nearest_indices(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    upper = np.searchsorted(reference, query, side="left")
    upper = np.clip(upper, 1, len(reference) - 1)
    lower = upper - 1
    choose_upper = np.abs(reference[upper] - query) < np.abs(
        reference[lower] - query
    )
    return np.where(choose_upper, upper, lower)


def zoh_indices(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.clip(
        np.searchsorted(reference, query, side="right") - 1,
        0,
        len(reference) - 1,
    )


def npz_array_summary(path: Path) -> dict[str, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            name: {
                "shape": list(archive[name].shape),
                "dtype": str(archive[name].dtype),
            }
            for name in archive.files
        }


def landmark_record(
    frame: int,
    left_timestamps: np.ndarray,
    right_timestamps: np.ndarray,
    right_for_left: np.ndarray,
    driver_timestamps: np.ndarray,
    state_for_left: np.ndarray,
    q7: np.ndarray,
) -> dict[str, Any]:
    right_frame = int(right_for_left[frame])
    state = int(state_for_left[frame])
    return {
        "left_frame": frame,
        "left_timestamp_s": float(left_timestamps[frame]),
        "right_frame_nearest": right_frame,
        "right_timestamp_s": float(right_timestamps[right_frame]),
        "stereo_abs_delta_ms": float(
            abs(right_timestamps[right_frame] - left_timestamps[frame]) * 1e3
        ),
        "pose_driver_state_zoh": state,
        "pose_driver_timestamp_s": float(driver_timestamps[state]),
        "q7_jaw_deg": float(np.degrees(q7[state, 6])),
    }


def build_manifest() -> dict[str, Any]:
    left_metadata = load_json(LEFT_METADATA_PATH)
    right_metadata = load_json(RIGHT_METADATA_PATH)
    camera_manifest = load_json(CAMERA_MANIFEST_PATH)
    calibration = load_json(RECTIFIED_CALIBRATION_PATH)
    tissue_metadata = load_json(TISSUE_METADATA_PATH)
    pose_report = load_json(POSE_REPORT_PATH)

    left_timestamps = np.asarray(left_metadata["timestamps"], dtype=np.float64)
    right_timestamps = np.asarray(
        right_metadata["timestamps"], dtype=np.float64
    )
    with np.load(POSE_DRIVER_PATH, allow_pickle=False) as driver:
        driver_timestamps = np.asarray(driver["timestamps"], dtype=np.float64)
        q7 = np.asarray(driver["q7"], dtype=np.float64)
        q7_raw = np.asarray(driver["q7_raw"], dtype=np.float64)
        driver_schema = str(driver["schema"].item())
        driver_coordinate_frame = str(driver["coordinate_frame"].item())
        link_names = [str(value) for value in driver["link_names"].tolist()]
        visual_jaw_correction_frozen = bool(
            driver["visual_jaw_correction_frozen"].item()
        )
        poses = np.asarray(
            driver["poses_gui_world_xyz_xyzw"], dtype=np.float64
        )

    right_for_left = nearest_indices(right_timestamps, left_timestamps)
    state_for_left = zoh_indices(driver_timestamps, left_timestamps)
    all_stereo_pair_delta_ms = (
        np.abs(
            right_timestamps[right_for_left]
            - left_timestamps
        )
        * 1e3
    )
    stereo_eligible_mask = all_stereo_pair_delta_ms <= 20.0
    stereo_eligible_left_frames = np.flatnonzero(stereo_eligible_mask)
    stereo_excluded_left_frames = np.flatnonzero(~stereo_eligible_mask)
    stereo_last_left_frame = int(stereo_eligible_left_frames[-1])
    stereo_pair_delta_ms = all_stereo_pair_delta_ms[stereo_eligible_mask]

    mean_distal_height = poses[state_for_left, :, 2].mean(axis=1)
    maximum_retraction_frame = int(
        548 + np.argmax(mean_distal_height[548:1247])
    )
    if maximum_retraction_frame != LANDMARK_FRAMES["maximum_retraction"]:
        raise RuntimeError(
            "Frozen maximum-retraction landmark changed: "
            f"{maximum_retraction_frame}"
        )

    runtime_constants = literal_constants(
        RUNTIME_SOURCE_PATH, RUNTIME_CONSTANT_NAMES
    )
    left_x_wc = np.asarray(
        camera_manifest["stereo_left"]["X_WC"], dtype=np.float64
    )
    right_x_wc = np.asarray(
        camera_manifest["stereo_right"]["X_WC"], dtype=np.float64
    )
    manifest_baseline = float(
        np.linalg.norm(right_x_wc[:3, 3] - left_x_wc[:3, 3])
    )
    calibration_baseline = float(calibration["baseline_m"])

    landmark_sources = {
        "initial_reference": "first frozen stereo playback frame",
        "first_contact": (
            "manual left-video review; cross-checked against the existing "
            "dense-contact diagnostic near pose-driver state 895"
        ),
        "grasp_closed": (
            "manual left-video review plus q7 transition from open to closed"
        ),
        "traction_reference": (
            "manual left-video review during sustained closed-jaw lifting"
        ),
        "maximum_retraction": (
            "maximum mean distal-link world Z between grasp and release"
        ),
        "release_complete": (
            "manual left-video review plus q7 return to the open plateau"
        ),
        "rebound_reference": (
            "manual left-video review after release, with jaws fully open"
        ),
    }
    landmarks = {}
    for name, frame in LANDMARK_FRAMES.items():
        record = landmark_record(
            frame,
            left_timestamps,
            right_timestamps,
            right_for_left,
            driver_timestamps,
            state_for_left,
            q7,
        )
        record["selection_basis"] = landmark_sources[name]
        landmarks[name] = record

    mesh_records = {
        path.name: file_record(path)
        for path in sorted((POSE_ROOT / "meshes").glob("*"))
        if path.is_file()
    }
    observation_files = {
        "left_tissue_masks": file_record(
            REPO_ROOT
            / "data/super/grasp5_native/visual_force_masks_v1/"
            "tissue_masks_packbits.npy"
        ),
        "left_tissue_mask_report": file_record(
            REPO_ROOT
            / "data/super/grasp5_native/visual_force_masks_v1/report.json"
        ),
        "right_tissue_masks": file_record(
            REPO_ROOT
            / "data/super/grasp5_native/visual_force_masks_right_v1/"
            "tissue_masks_packbits.npy"
        ),
        "right_tissue_mask_report": file_record(
            REPO_ROOT
            / "data/super/grasp5_native/visual_force_masks_right_v1/"
            "report.json"
        ),
        "stereo_tool_part_masks": file_record(
            REPO_ROOT
            / "data/super/psm_visual_calibration/"
            "raw_paper_lnd_stereo_dense_contact_v3/"
            "surgicalsam2_multianchor_parts_dense_contact_v5/"
            "stereo_multianchor_part_masks.npz"
        ),
    }

    relevant_code_paths = [
        Path(__file__).resolve(),
        RUNTIME_SOURCE_PATH,
        GUI_SOURCE_PATH,
        THINLINC_ENTRYPOINT_PATH,
        REPO_ROOT
        / "src/embodied_gaussians/physics_simulator/integrator.py",
        REPO_ROOT
        / "src/embodied_gaussians/physics_simulator/triangle_skin_contact.py",
        REPO_ROOT
        / "src/embodied_gaussians/embodied_simulator/simulator.py",
        REPO_ROOT
        / "src/embodied_gaussians/embodied_simulator/gaussians.py",
        REPO_ROOT
        / "scripts/replay_super_tissue_calibration_stage_c.py",
        REPO_ROOT
        / "scripts/validate_super_tissue_calibration_stage_c.py",
    ]

    thinlinc_source = THINLINC_ENTRYPOINT_PATH.read_text()
    gates = {
        "left_timestamps_strictly_increasing": bool(
            np.all(np.diff(left_timestamps) > 0.0)
        ),
        "right_timestamps_strictly_increasing": bool(
            np.all(np.diff(right_timestamps) > 0.0)
        ),
        "driver_timestamps_strictly_increasing": bool(
            np.all(np.diff(driver_timestamps) > 0.0)
        ),
        "left_and_right_have_1441_frames": bool(
            len(left_timestamps) == len(right_timestamps) == 1441
        ),
        "stereo_calibration_range_within_20ms": bool(
            np.max(stereo_pair_delta_ms) <= 20.0
        ),
        "camera_intrinsics_match_rectified_calibration": bool(
            np.allclose(left_metadata["K"], calibration["K_left_rect"])
            and np.allclose(right_metadata["K"], calibration["K_right_rect"])
        ),
        "camera_manifest_baseline_matches_calibration": bool(
            np.isclose(manifest_baseline, calibration_baseline, atol=1e-12)
        ),
        "driver_retains_all_5458_states": bool(
            len(driver_timestamps) == len(q7) == 5458
        ),
        "driver_preserves_raw_jaw_command": bool(
            visual_jaw_correction_frozen
            and np.array_equal(q7[:, 6], q7_raw[:, 6])
        ),
        "driver_contains_both_contact_jaws": bool(
            set(CONTACT_LINKS).issubset(link_names)
        ),
        "tissue_asset_counts_match_metadata": bool(
            tissue_metadata["counts"]["particles"] == 37570
            and tissue_metadata["counts"]["tetrahedra"] == 193845
            and tissue_metadata["counts"]["skin_triangles"] == 24196
            and tissue_metadata["counts"]["gaussians"] == 2863
        ),
        "tissue_gaussian_binding_is_four_particle": bool(
            npz_array_summary(TISSUE_ASSET_PATH)[
                "gaussian_particle_indices"
            ]["shape"]
            == [2863, 4]
        ),
        "thinlinc_entrypoint_uses_frozen_driver": (
            f"--psm-pose-driver {POSE_DRIVER_NAME}" in thinlinc_source
        ),
        "thinlinc_entrypoint_uses_paper_soft": (
            "--tissue-mode paper_soft" in thinlinc_source
        ),
    }
    if not all(gates.values()):
        failed = sorted(name for name, passed in gates.items() if not passed)
        raise RuntimeError(f"Stage-A freeze gates failed: {failed}")

    raw_report = load_json(RAW_IDENTITY_REPORT_PATH)
    manifest: dict[str, Any] = {
        "schema": "super_tissue_calibration_stage_a_manifest_v1",
        "stage": "A_frozen_inputs",
        "status": "complete",
        "passed": True,
        "created_on": "2026-07-30",
        "purpose": (
            "Freeze every input that may otherwise let pose, contact, or "
            "geometry errors be absorbed into calibrated tissue parameters."
        ),
        "paper_method": {
            "title": (
                "Real-to-Sim Deformable Object Manipulation: Optimizing "
                "Physics Models with Residual Mappings for Robotic Surgery"
            ),
            "arxiv_id": "2309.11656",
            "paper_version": "v2",
            "project_adaptation": (
                "Later stages will use visible-surface residual mapping plus "
                "tetrahedral material energy, then calibrate the current XPBD "
                "model's global Young's modulus and damping before any local "
                "stiffness field is considered."
            ),
        },
        "sequence": {
            "raw_bag": trusted_large_file_record(
                REPO_ROOT / "data/grasp5/grasp5.bag",
                RAW_IDENTITY_REPORT_PATH,
                "bag",
            ),
            "raw_stream_summary": raw_report["raw_bag"],
            "offline_stereo": {
                "left_video": file_record(LEFT_VIDEO_PATH),
                "right_video": file_record(RIGHT_VIDEO_PATH),
                "left_metadata": file_record(LEFT_METADATA_PATH),
                "right_metadata": file_record(RIGHT_METADATA_PATH),
                "playback_left_frame_range_inclusive": [0, 1440],
                "stereo_calibration_left_frame_range_inclusive": [
                    0,
                    1440,
                ],
                "stereo_excluded_left_frames": [
                    int(value) for value in stereo_excluded_left_frames
                ],
                "stereo_eligible_left_frame_count": int(
                    len(stereo_eligible_left_frames)
                ),
                "stereo_exclusion_reason": (
                    "nearest right timestamp is more than the frozen 20 ms "
                    "stereo association limit"
                ),
                "left_frame_count": len(left_timestamps),
                "right_frame_count": len(right_timestamps),
                "left_time_range_s": [
                    float(left_timestamps[0]),
                    float(left_timestamps[-1]),
                ],
                "right_time_range_s": [
                    float(right_timestamps[0]),
                    float(right_timestamps[-1]),
                ],
                "association": (
                    "nearest right timestamp for observations; "
                    "searchsorted(pose timestamps, side='right')-1 "
                    "zero-order hold for the pose driver"
                ),
                "nearest_stereo_delta_ms_p50_p95_max": [
                    float(value)
                    for value in np.percentile(
                        stereo_pair_delta_ms, [50, 95, 100]
                    )
                ],
            },
            "landmarks": landmarks,
            "phase_ranges_left_frames_inclusive": {
                "approach_and_contact": [0, 543],
                "jaw_closure": [544, 548],
                "closed_jaw_traction": [548, 906],
                "held_and_return": [907, 1246],
                "release": [1247, 1276],
                "post_release_rebound": [1277, stereo_last_left_frame],
            },
        },
        "camera": {
            "camera_manifest": file_record(CAMERA_MANIFEST_PATH),
            "rectified_calibration": file_record(
                RECTIFIED_CALIBRATION_PATH
            ),
            "source_calibration": file_record(
                REPO_ROOT / "data/camera_calibration.yaml"
            ),
            "table_frame": file_record(
                REPO_ROOT / "data/super/table_frame.json"
            ),
            "image_size_wh": left_metadata["resolution"],
            "K_left_rect": left_metadata["K"],
            "K_right_rect": right_metadata["K"],
            "baseline_m": calibration_baseline,
            "X_world_camera": {
                "stereo_left": camera_manifest["stereo_left"]["X_WC"],
                "stereo_right": camera_manifest["stereo_right"]["X_WC"],
            },
        },
        "psm_trajectory": {
            "driver_name": POSE_DRIVER_NAME,
            "driver": file_record(POSE_DRIVER_PATH),
            "driver_schema": driver_schema,
            "coordinate_frame": driver_coordinate_frame,
            "state_selection": "searchsorted(timestamp, side='right') - 1",
            "manual_corrections": MANUAL_CORRECTIONS,
            "manual_corrections_source": "组织软化.md section 1",
            "manual_corrections_frozen_for_all_calibration_runs": True,
            "urdf": file_record(POSE_URDF_PATH),
            "mimic_map": file_record(POSE_MIMIC_PATH),
            "surface_gaussians": file_record(POSE_GAUSSIANS_PATH),
            "registration_report": file_record(POSE_REPORT_PATH),
            "registration_version": pose_report["version"],
            "upstream_repository": pose_report["upstream"],
            "cad_meshes": mesh_records,
            "jaw_command": {
                "source": "q7[:, 6] embedded in the frozen pose driver",
                "visual_jaw_correction_frozen": (
                    visual_jaw_correction_frozen
                ),
                "manual_jaw_offset_deg": 0.0,
                "range_deg": [
                    float(np.degrees(q7[:, 6].min())),
                    float(np.degrees(q7[:, 6].max())),
                ],
            },
            "contact_links": CONTACT_LINKS,
            "shaft_contact": False,
            "physics_may_modify_psm_pose": False,
        },
        "tissue": {
            "mode": "paper_soft",
            "asset": file_record(TISSUE_ASSET_PATH),
            "metadata": file_record(TISSUE_METADATA_PATH),
            "asset_version": tissue_metadata["asset_version"],
            "representation": tissue_metadata["representation"],
            "counts": tissue_metadata["counts"],
            "mass": tissue_metadata["mass"],
            "npz_arrays": npz_array_summary(TISSUE_ASSET_PATH),
            "fixed_boundary_source": "fixed_mask embedded in the tissue asset",
            "gaussian_binding": (
                "gaussian_tet_ids + four particle indices + barycentric "
                "weights + rest offset; never replaced by per-frame animation"
            ),
            "initial_material_baseline": runtime_constants,
        },
        "observation_inputs_for_stage_b": observation_files,
        "calibration_protocol": {
            "visual_force_iterations": 0,
            "depth_residual_mapping_enabled": False,
            "online_stiffness_optimization_enabled": False,
            "persistent_grip_constraint_enabled": False,
            "psm_absolute_pose_is_kinematic": True,
            "first_parameters_to_calibrate": [
                "global Young's modulus",
                "velocity damping",
            ],
            "parameters_frozen_during_first_material_search": [
                "Poisson ratio",
                "jaw friction",
                "contact margin and query distance",
                "contact solver settings",
                "fixed boundary",
                "PSM trajectory and manual corrections",
            ],
        },
        "code_snapshot": {
            "git": git_snapshot(),
            "relevant_files": {
                repo_path(path): file_record(path)
                for path in relevant_code_paths
            },
        },
        "validation": {
            "gates": gates,
            "all_gates_passed": True,
        },
        "next_stage": (
            "B: generate per-frame visible tissue surface observations from "
            "the frozen stereo range while removing tool occlusion."
        ),
    }
    return manifest


def verify_manifest(manifest: dict[str, Any]) -> None:
    failures: list[str] = []

    def walk(value: Any, key_path: str = "") -> None:
        if isinstance(value, dict):
            if {"path", "sha256"}.issubset(value):
                path = REPO_ROOT / value["path"]
                if not path.is_file():
                    failures.append(f"{key_path}: missing {path}")
                elif value.get("full_hash_reused_after_size_check"):
                    if path.stat().st_size != value["size_bytes"]:
                        failures.append(f"{key_path}: size changed")
                elif sha256(path) != value["sha256"]:
                    failures.append(f"{key_path}: sha256 changed")
            for key, child in value.items():
                walk(child, f"{key_path}.{key}" if key_path else key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{key_path}[{index}]")

    walk(manifest)
    if failures:
        raise RuntimeError("Manifest verification failed:\n" + "\n".join(failures))


def main() -> None:
    args = parse_args()
    output = args.output
    if not output.is_absolute():
        output = (REPO_ROOT / output).resolve()

    if args.verify_only:
        manifest = load_json(output)
        verify_manifest(manifest)
        print(f"verified: {output}")
        return

    manifest = build_manifest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    verify_manifest(manifest)
    print(f"wrote: {output}")
    print(
        "stage A gates: "
        f"{sum(manifest['validation']['gates'].values())}/"
        f"{len(manifest['validation']['gates'])} passed"
    )


if __name__ == "__main__":
    main()
